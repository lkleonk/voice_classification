import argparse
import copy
import hashlib
import math
import pickle
import random
from pathlib import Path
from typing import Any
from collections.abc import Sequence
from dataclasses import dataclass
import multiprocessing as mp


import yaml

import src.train_spectrogram as train_spectrogram
from src.run_optuna_tuning import optimize as optuna_optimize
from src.run_evaluation_protocol import (
    get_effective_k_folds,
    get_effective_random_seed_runs,
    run_evaluation_protocol,
    validate_seed_repeat_settings,
)
from src.schemas.dataclasses import SheetResult
from src.schemas.typed_dicts import CrossValSummary
from src.settings import (
    EVALUATION_PROTOCOL,
    EXCEL,
    FILES,
    MLFLOW,
    PATHS,
    STUDY_IDS,
)
from src.utils.reporting.excel_study import create_and_save_excel_file
from src.utils.reporting.logger_setup import setup_run_logger
from src.utils.reporting.mlflow_helper import (
    log_study_sheet_artifacts_to_mlflow,
)
from src.utils.toolkit.naming import (
    _best_configs_demo_data_key,
    _canonicalize_csv_tokens,
    _excel_uses_tuning_results,
    abbreviate,
    build_best_configs_dict_filename,
    build_sheet_results_pickle_filename,
    build_study_run_name,
    construct_excel_summary_filename,
    construct_exhaustive_artifact_filename_stem,
    construct_sheet_results_pickle_filename_stem,
    create_study_id,
)

ctx = mp.get_context("spawn")



"""
GOAL

create excel data for run data

for every dataset + model (ex.: uk - copd, control) = 1 sheet
In every sheet: different models and different recording categories
important metrics: val + test MBA (mean balanced accuracy) and val + test AUROC
bonus: nice colors. Different color for every sheet.
"""




def _excel_runs_optuna_first() -> bool:
    return EXCEL.TUNING_MODE is EXCEL.TuningMode.RUN_OPTUNA_FIRST


def _excel_uses_precomputed_best_configs() -> bool:
    return EXCEL.TUNING_MODE is EXCEL.TuningMode.USE_PRECOMPUTED_BEST_CONFIGS




def _cross_val_or_rseeds_output_dir_id() -> str:
    return (
        PATHS.RSEEDS_ID
        if EVALUATION_PROTOCOL.ONLY_CHANGE_RANDOM_SEEDS
        else PATHS.CROSS_VAL_ID
    )





def _simulate_cross_val_metrics(
    cmd_args,
    *,
    nan_probability: float = 0.3,   # 0.0 = never NaN
) -> CrossValSummary:
    """
    Simulates cross-validation metrics for test/debug runs.

    Returns the SAME metric-key shape as run_evaluation_protocol().

    Deterministic per (dataset, model, recording_category, classes),
    but can optionally inject NaNs to simulate failed runs.
    """

    identity = (
        f"{cmd_args.dataset_v}|"
        f"{cmd_args.model_type}|"
        f"{cmd_args.recording_category}|"
        f"{cmd_args.selected_classes}"
    )

    seed = int(hashlib.md5(identity.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    effective_k = get_effective_k_folds(cmd_args)
    only_change_random_seeds = getattr(EVALUATION_PROTOCOL, "ONLY_CHANGE_RANDOM_SEEDS", False)
    run_count = (
        get_effective_random_seed_runs(cmd_args)
        if only_change_random_seeds
        else int(effective_k)
    )
    aggregation_mode = "seed_repeat" if only_change_random_seeds else "kfold"

    def clamp_mean(x: float) -> float:
        return max(0.5, min(0.99, x))

    def clamp_std(x: float) -> float:
        return max(0.0, min(0.25, x))

    # --------------------------------------------------
    # Optional NaN injection (simulated failure)
    # --------------------------------------------------
    if nan_probability > 0.0 and rng.random() < nan_probability:
        nan = math.nan
        return {
            "run_count": run_count,
            "aggregation_mode": aggregation_mode,
            # FINAL model
            "final_model_mean_val_mba": nan,
            "final_model_std_val_mba": nan,
            "final_model_mean_test_mba": nan,
            "final_model_std_test_mba": nan,
            "final_model_mean_val_auroc": nan,
            "final_model_std_val_auroc": nan,
            "final_model_mean_test_auroc": nan,
            "final_model_std_test_auroc": nan,

            # BEST model
            "best_model_mean_val_mba": nan,
            "best_model_std_val_mba": nan,
            "best_model_mean_test_mba": nan,
            "best_model_std_test_mba": nan,
            "best_model_mean_val_auroc": nan,
            "best_model_std_val_auroc": nan,
            "best_model_mean_test_auroc": nan,
            "best_model_std_test_auroc": nan,
        }

    # --- generate plausible FINAL means ---
    final_model_val_mba = rng.uniform(0.50, 0.95)
    final_model_test_mba = final_model_val_mba - rng.uniform(0.01, 0.05)

    final_model_val_auroc = min(final_model_val_mba + rng.uniform(0.05, 0.12), 0.95)
    final_model_test_auroc = final_model_val_auroc - rng.uniform(0.01, 0.04)

    # --- generate plausible FINAL stds ---
    final_model_val_mba_std = rng.uniform(0.01, 0.05)
    final_model_test_mba_std = max(0.01, final_model_val_mba_std + rng.uniform(-0.005, 0.010))

    final_model_val_auroc_std = rng.uniform(0.01, 0.04)
    final_model_test_auroc_std = max(0.01, final_model_val_auroc_std + rng.uniform(-0.005, 0.010))

    # --- generate plausible BEST means (slightly better than FINAL, but bounded) ---
    best_model_val_mba = min(final_model_val_mba + rng.uniform(0.00, 0.03), 0.99)
    best_model_test_mba = best_model_val_mba - rng.uniform(0.01, 0.05)

    best_model_val_auroc = min(best_model_val_mba + rng.uniform(0.05, 0.12), 0.99)
    best_model_test_auroc = best_model_val_auroc - rng.uniform(0.01, 0.04)

    # --- generate plausible BEST stds (similar scale) ---
    best_model_val_mba_std = max(0.01, final_model_val_mba_std + rng.uniform(-0.010, 0.010))
    best_model_test_mba_std = max(0.01, best_model_val_mba_std + rng.uniform(-0.005, 0.010))

    best_model_val_auroc_std = max(0.01, final_model_val_auroc_std + rng.uniform(-0.010, 0.010))
    best_model_test_auroc_std = max(0.01, best_model_val_auroc_std + rng.uniform(-0.005, 0.010))

    return {
        "run_count": run_count,
        "aggregation_mode": aggregation_mode,
        # FINAL model
        "final_model_mean_val_mba": round(clamp_mean(final_model_val_mba), 4),
        "final_model_std_val_mba": round(clamp_std(final_model_val_mba_std), 4),

        "final_model_mean_test_mba": round(clamp_mean(final_model_test_mba), 4),
        "final_model_std_test_mba": round(clamp_std(final_model_test_mba_std), 4),

        "final_model_mean_val_auroc": round(clamp_mean(final_model_val_auroc), 4),
        "final_model_std_val_auroc": round(clamp_std(final_model_val_auroc_std), 4),

        "final_model_mean_test_auroc": round(clamp_mean(final_model_test_auroc), 4),
        "final_model_std_test_auroc": round(clamp_std(final_model_test_auroc_std), 4),

        # BEST model
        "best_model_mean_val_mba": round(clamp_mean(best_model_val_mba), 4),
        "best_model_std_val_mba": round(clamp_std(best_model_val_mba_std), 4),

        "best_model_mean_test_mba": round(clamp_mean(best_model_test_mba), 4),
        "best_model_std_test_mba": round(clamp_std(best_model_test_mba_std), 4),

        "best_model_mean_val_auroc": round(clamp_mean(best_model_val_auroc), 4),
        "best_model_std_val_auroc": round(clamp_std(best_model_val_auroc_std), 4),

        "best_model_mean_test_auroc": round(clamp_mean(best_model_test_auroc), 4),
        "best_model_std_test_auroc": round(clamp_std(best_model_test_auroc_std), 4),
    }


def _build_sheet_result_from_summary(
    *,
    cmd_args: argparse.Namespace,
    summary: CrossValSummary,
) -> SheetResult:
    add_demographic_data_var = False
    
    add_demographic_data_var = cmd_args.add_demo_data

    return SheetResult(
        dataset=cmd_args.dataset_v,
        lung_conditions=cmd_args.selected_classes,
        recording_category=cmd_args.recording_category,
        model=cmd_args.model_type,
        norm_mode=cmd_args.norm_mode,
        final_model_val_mba_mean=float(summary["final_model_mean_val_mba"]),
        final_model_test_mba_mean=float(summary["final_model_mean_test_mba"]),
        final_model_val_auroc_mean=float(summary["final_model_mean_val_auroc"]),
        final_model_test_auroc_mean=float(summary["final_model_mean_test_auroc"]),
        final_model_val_mba_std=float(summary["final_model_std_val_mba"]),
        final_model_test_mba_std=float(summary["final_model_std_test_mba"]),
        final_model_val_auroc_std=float(summary["final_model_std_val_auroc"]),
        final_model_test_auroc_std=float(summary["final_model_std_test_auroc"]),
        best_model_val_mba_mean=float(summary["best_model_mean_val_mba"]),
        best_model_test_mba_mean=float(summary["best_model_mean_test_mba"]),
        best_model_val_auroc_mean=float(summary["best_model_mean_val_auroc"]),
        best_model_test_auroc_mean=float(summary["best_model_mean_test_auroc"]),
        best_model_val_mba_std=float(summary["best_model_std_val_mba"]),
        best_model_test_mba_std=float(summary["best_model_std_test_mba"]),
        best_model_val_auroc_std=float(summary["best_model_std_val_auroc"]),
        best_model_test_auroc_std=float(summary["best_model_std_test_auroc"]),
        add_demographic_data_var=add_demographic_data_var,
    )



def simulate_cross_val_summary(
    cmd_args: argparse.Namespace,
    *,
    nan_probability: float = 0.3,
) -> SheetResult:
    """
    Simulates one full SheetResult for test/debug runs.
    """
    summary = _simulate_cross_val_metrics(
        cmd_args,
        nan_probability=nan_probability,
    )
    return _build_sheet_result_from_summary(cmd_args=cmd_args, summary=summary)





def permissible_combination(cli_dataset_v, dataset_v: str, conditions: str, category: str) -> bool:
    dataset = dataset_v.lower()
    cat = category.lower()
    conds = conditions.lower().replace(" ", "")

    # Hard gate: only evaluate combos for the dataset selected on CLI
    if dataset != cli_dataset_v.lower():
        return False

    if dataset == "uk":
        # UK: Only 'poem' and strictly 'copd,control'
        if cat != "poem":
            return False
        if conds not in ("copd,control", "control,copd"):
            return False
        return True

    return False  # Unknown dataset


def perform_cross_val_run(cmd_args: argparse.Namespace) -> SheetResult:    
    """
    Orchestrates the cross-validation flow, optionally performing tuning first.
    """
    if cmd_args.test_run:
        return simulate_cross_val_summary(cmd_args)

    summary = run_evaluation_protocol(cmd_args)
    return _build_sheet_result_from_summary(cmd_args=cmd_args, summary=summary)


# returns updated args
def get_new_crossval_args(original_args, conditions: str, model: str, category: str): # to change any other args, one has to change main snakefile or change this function accordingly
    args_copy = copy.deepcopy(original_args)
    

    args_copy.selected_classes = conditions
    args_copy.recording_category = category
    args_copy.model_type = model

    abbreviated_conditions = abbreviate(conditions) # 'copd, control, fibrosis' --> 'cop_con_fib'
    args_copy.output_dir = f'{args_copy.output_dir}_XL_{args_copy.dataset_v}_{abbreviated_conditions}_{category}_{model}'.replace(',', '_').replace(" ", "")

    # to-do: raise validation error (if for example rec-category is in  (a, o , i) and dataset_v is uk) or (fibrosis and uk)
    return args_copy





## LOAD CONFIG ##
def load_config(path):
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_best_configs_dict(dict_path: Path, logger=None) -> dict[str, Any]:
    """
    Load persisted best-config mapping from YAML.
    """
    if not dict_path.exists():
        raise FileNotFoundError(f"Best-config dict file not found: {dict_path}")

    with dict_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Best-config dict must be a mapping, got: {type(data)}")

    if logger:
        logger.info("Loaded best-config dict from: %s", dict_path)
    return data


def discover_best_configs_dict_paths(load_dir: Path, logger=None) -> list[Path]:
    """
    Discover precomputed best-config dict YAML files from one version directory.
    """
    resolved_dir = Path(load_dir)
    if not resolved_dir.exists():
        raise FileNotFoundError(f"Best-config version directory not found: {resolved_dir}")
    if not resolved_dir.is_dir():
        raise NotADirectoryError(
            f"Best-config version path is not a directory: {resolved_dir}"
        )

    pattern = f"*{PATHS.BEST_CONFIGS_DICT_FILENAME_SUFFIX}"
    dict_paths = sorted(
        path for path in resolved_dir.glob(pattern)
        if path.is_file()
    )
    if not dict_paths:
        raise FileNotFoundError(
            "No best-config dict YAML files found in version directory: "
            f"{resolved_dir} (pattern={pattern})"
        )

    if logger:
        logger.info(
            "Discovered %s best-config dict YAML files in %s",
            len(dict_paths),
            resolved_dir,
        )
    return dict_paths


def load_and_merge_best_configs_dicts(
    dict_paths: Sequence[Path | str],
    logger=None,
) -> dict[str, Any]:
    """
    Load multiple best-config dict YAML files and merge them into one lookup dict.

    Expected payload shape per file (reserved top-level `_meta` is allowed):
      norm_mode -> demo_data_key -> dataset_v -> canonical_lung_conditions -> rec_category -> model -> entry
    """
    if not dict_paths:
        raise ValueError(
            "No precomputed best-config dict YAML paths were provided while "
            "EXCEL.TUNING_MODE=USE_PRECOMPUTED_BEST_CONFIGS is set"
        )

    valid_norm_keys = {"norm_on", "norm_off"}
    valid_demo_data_keys = {"w_demo_data", "no_demo_data"}
    resolved_paths = [Path(p) for p in dict_paths]
    merged: dict[str, Any] = {}
    merged_combo_count = 0

    for dict_path in resolved_paths:
        data = load_best_configs_dict(dict_path, logger=logger)

        meta = data.get("_meta")
        if isinstance(meta, dict) and logger:
            schema_version = meta.get("schema_version")
            logger.info(
                "Best-config dict meta | path=%s | schema_version=%s",
                dict_path,
                schema_version,
            )
            if schema_version != EXCEL.BEST_CONFIG_DICT_SCHEMA_VERSION:
                logger.warning(
                    "Schema version mismatch in %s (got=%s, expected=%s)",
                    dict_path,
                    schema_version,
                    EXCEL.BEST_CONFIG_DICT_SCHEMA_VERSION,
                )

        payload_norm_keys = [key for key in data.keys() if key != "_meta"]
        if not payload_norm_keys:
            raise ValueError(f"Best-config dict has no norm-mode payload keys: {dict_path}")

        invalid_top_keys = [key for key in payload_norm_keys if key not in valid_norm_keys]
        if invalid_top_keys:
            raise ValueError(
                f"Best-config dict top-level keys must be norm modes ({sorted(valid_norm_keys)}). "
                f"Invalid keys in {dict_path}: {invalid_top_keys}"
            )

        for norm_mode in payload_norm_keys:
            norm_block = data[norm_mode]
            if not isinstance(norm_block, dict):
                raise ValueError(
                    f"Expected mapping at [{norm_mode}] in {dict_path}, got {type(norm_block)}"
                )

            payload_demo_keys = list(norm_block.keys())
            if not payload_demo_keys:
                raise ValueError(
                    f"Best-config dict missing demo-data keys under [{norm_mode}] in {dict_path}"
                )

            invalid_demo_keys = [
                key for key in payload_demo_keys if key not in valid_demo_data_keys
            ]
            if invalid_demo_keys:
                raise ValueError(
                    "Best-config dict is missing the demo-data level under "
                    f"[{norm_mode}] in {dict_path}. Expected keys "
                    f"{sorted(valid_demo_data_keys)} before dataset keys, got {invalid_demo_keys}"
                )

            target_norm = merged.setdefault(norm_mode, {})
            for demo_data_key in payload_demo_keys:
                demo_data_block = norm_block[demo_data_key]
                if not isinstance(demo_data_block, dict):
                    raise ValueError(
                        f"Expected mapping at [{norm_mode}][{demo_data_key}] in {dict_path}, "
                        f"got {type(demo_data_block)}"
                    )

                target_demo = target_norm.setdefault(demo_data_key, {})
                for dataset_v, dataset_block in demo_data_block.items():
                    if not isinstance(dataset_block, dict):
                        raise ValueError(
                            f"Expected mapping at [{norm_mode}][{demo_data_key}][{dataset_v}] "
                            f"in {dict_path}, got {type(dataset_block)}"
                        )

                    target_dataset = target_demo.setdefault(dataset_v, {})
                    for lung_conditions, lung_block in dataset_block.items():
                        if not isinstance(lung_block, dict):
                            raise ValueError(
                                "Expected mapping at "
                                f"[{norm_mode}][{demo_data_key}][{dataset_v}]"
                                f"[{lung_conditions}] in {dict_path}, got {type(lung_block)}"
                            )

                        target_lung = target_dataset.setdefault(lung_conditions, {})
                        for rec_category, rec_block in lung_block.items():
                            if not isinstance(rec_block, dict):
                                raise ValueError(
                                    "Expected mapping at "
                                    f"[{norm_mode}][{demo_data_key}][{dataset_v}]"
                                    f"[{lung_conditions}][{rec_category}] in {dict_path}, "
                                    f"got {type(rec_block)}"
                                )

                            target_rec = target_lung.setdefault(rec_category, {})
                            for model, entry in rec_block.items():
                                if model in target_rec:
                                    raise ValueError(
                                        "Duplicate precomputed best-config entry while merging dicts: "
                                        f"norm_mode={norm_mode}, demo_data_key={demo_data_key}, "
                                        f"dataset={dataset_v}, conditions={lung_conditions}, "
                                        f"category={rec_category}, model={model}, source={dict_path}"
                                    )
                                if not isinstance(entry, dict):
                                    raise ValueError(
                                        "Expected mapping at "
                                        f"[{norm_mode}][{demo_data_key}][{dataset_v}]"
                                        f"[{lung_conditions}][{rec_category}][{model}] in {dict_path}, "
                                        f"got {type(entry)}"
                                    )
                                path_list = entry.get("best_trial_config_path_list")
                                if (not isinstance(path_list, list) or not path_list) and logger:
                                    logger.warning(
                                        "Best-config entry has missing/empty best_trial_config_path_list; "
                                        "keeping entry for combo-level failure if selected. "
                                        "norm_mode=%s, demo_data_key=%s, dataset=%s, conditions=%s, "
                                        "category=%s, model=%s, source=%s",
                                        norm_mode,
                                        demo_data_key,
                                        dataset_v,
                                        lung_conditions,
                                        rec_category,
                                        model,
                                        dict_path,
                                    )

                                target_rec[model] = copy.deepcopy(entry)
                                merged_combo_count += 1

    if logger:
        logger.info(
            "Merged %s best-config dict YAML files into %s combo entries",
            len(resolved_paths),
            merged_combo_count,
        )
    return merged


def select_best_config_yaml(
    best_configs_dict: dict[str, Any],
    norm_mode: str,
    add_demo_data: bool,
    dataset_v: str,
    lung_conditions: str,
    rec_category: str,
    model: str,
) -> str:
    """
    Select the best-trial YAML path for one experiment combination.
    """
    demo_data_key = _best_configs_demo_data_key(add_demo_data)
    canonical_lung_conditions = _canonicalize_csv_tokens(lung_conditions)
    try:
        entry = best_configs_dict[norm_mode][demo_data_key][dataset_v][canonical_lung_conditions][rec_category][model]
    except KeyError as exc:
        raise KeyError(
            "Missing combination in best-config dict: "
            f"norm_mode={norm_mode}, demo_data_key={demo_data_key}, dataset={dataset_v}, "
            f"conditions={lung_conditions}, canonical_conditions={canonical_lung_conditions}, "
            f"category={rec_category}, model={model}"
        ) from exc

    path_list = entry.get("best_trial_config_path_list")
    if not isinstance(path_list, list) or not path_list:
        raise ValueError(
            "best_trial_config_path_list missing/empty for combination: "
            f"norm_mode={norm_mode}, demo_data_key={demo_data_key}, dataset={dataset_v}, "
            f"conditions={lung_conditions}, canonical_conditions={canonical_lung_conditions}, "
            f"category={rec_category}, model={model}"
        )

    selected_path = str(path_list[0])
    if not Path(selected_path).exists():
        raise FileNotFoundError(
            f"Selected best-config YAML does not exist: {selected_path}"
        )
    return selected_path


def store_best_configs_dict_entry(
    best_params_dict: dict[str, Any],
    *,
    norm_mode: str,
    add_demo_data: bool,
    dataset_v: str,
    lung_conditions: str,
    rec_category: str,
    model: str,
    best_trial_config_path_list: list[str],
    optuna_id: str | None,
) -> None:
    demo_data_key = _best_configs_demo_data_key(add_demo_data)
    canonical_lung_conditions = _canonicalize_csv_tokens(lung_conditions)
    combo_best_params = (
        best_params_dict
        .setdefault(norm_mode, {})
        .setdefault(demo_data_key, {})
        .setdefault(dataset_v, {})
        .setdefault(canonical_lung_conditions, {})
        .setdefault(rec_category, {})
        .setdefault(model, {})
    )
    combo_best_params["best_trial_config_path_list"] = best_trial_config_path_list
    combo_best_params["optuna_id"] = optuna_id


@dataclass
class RunComboResults:
    sheet_result: SheetResult
    best_trial_config_path_list: list[str]
    optuna_id: str | None = None
    best_trial_number: int | None = None
    best_value: float | None = None


def run_exhaustive_combo(
    *,
    cmd_args,
    logger: Any,
    original_output_dir: str,
    norm_mode: str,
    combo_tag: str,
    dataset_v: str,
    lung_conditions: str,
    rec_category: str,
    model: str,
    loaded_best_configs_dict: dict,
):
    """
    Runs exactly one (dataset, conditions, category, model) combo:
      - optionally runs Optuna tuning (or loads best config from dict)
      - runs perform_cross_val_run()
      - returns a SheetResult (or a NaN SheetResult on failure)

    Returns lightweight, process-safe Optuna metadata for parent aggregation.
    """
    config_file_path = cmd_args.config_file
    best_trial_config_path_list: list[str] = []
    optuna_id: str | None = None
    best_trial_number: int | None = None
    best_value: float | None = None


    logger.info(
        f"--STARTING CROSSVAL RUN W FOLLOWING SETTINGS: \n"
        f"Norm={norm_mode} | Dataset={dataset_v} | Conds={lung_conditions} | "
        f"Cat={rec_category} | Model={model}--\n"
    )

    new_args = get_new_crossval_args(
        cmd_args,
        conditions=lung_conditions,
        category=rec_category,
        model=model,
    )
    new_args.norm_mode = norm_mode



    try:
        if _excel_uses_tuning_results() and not cmd_args.test_run:
            logger.info(
                "\n\n--PREPARING CONFIG FROM TUNING MODE (%s) FOR FIRST TRAINING SETUP--\n\n",
                EXCEL.TUNING_MODE.value,
            )

            if _excel_uses_precomputed_best_configs():
                config_file_path = select_best_config_yaml(
                    loaded_best_configs_dict,
                    norm_mode=norm_mode,
                    add_demo_data=cmd_args.add_demo_data,
                    dataset_v=dataset_v,
                    lung_conditions=lung_conditions,
                    rec_category=rec_category,
                    model=model,
                )
                logger.info(
                    "Loaded precomputed best config for combo from dict: %s",
                    config_file_path,
                )
                best_trial_config_path_list = [config_file_path]
            elif _excel_runs_optuna_first():
                new_args.output_dir = str(
                    Path(original_output_dir) / f"{combo_tag}_{PATHS.OPTUNA_ID}"
                )
                xl_study_id = create_study_id(STUDY_IDS.EXHAUSTIVE_SHEET_PREFIX)

                optuna_results = optuna_optimize(
                    new_args,
                    last_optuna_trial_after_tuning=False,
                    study_id=xl_study_id,
                )
                best_trial_config_path_list = optuna_results.best_trial_config_path_list
                optuna_id = optuna_results.study_id
                best_trial_number = optuna_results.best_trial_number
                best_value = optuna_results.best_value
                config_file_path = optuna_results.best_trial_config_path_list[0]

                logger.info(
                    "===== config_file_path overridden with yaml from tuning. "
                    "NOW STARTING PERFORM_CROSS_VAL_RUN AFTER GETTING BEST PARAMS FROM OPTUNA STUDY ====="
                )
            else:
                raise ValueError(f"Unsupported EXCEL.TUNING_MODE: {EXCEL.TUNING_MODE!r}")

        new_args.config_file = config_file_path

        
        new_args.output_dir = str(
            Path(original_output_dir) / f"{combo_tag}_{_cross_val_or_rseeds_output_dir_id()}"
        )

        sheet_result = perform_cross_val_run(cmd_args=new_args)
        
        return RunComboResults(
            sheet_result=sheet_result,
            best_trial_config_path_list=best_trial_config_path_list,
            optuna_id=optuna_id,
            best_trial_number=best_trial_number,
            best_value=best_value,
        )

    except Exception:
        logger.exception(
            f"X FAILED COMBO {combo_tag}: \n"
            f"Norm={norm_mode} | Dataset={dataset_v} | Conds={lung_conditions} | "
            f"Cat={rec_category} | Model={model}"
        )

        nan = math.nan

        failed_result = SheetResult(
            dataset=dataset_v,
            lung_conditions=lung_conditions,
            recording_category=rec_category,
            model=model,
            norm_mode=norm_mode,

            final_model_val_mba_mean=nan,
            final_model_test_mba_mean=nan,
            final_model_val_auroc_mean=nan,
            final_model_test_auroc_mean=nan,

            final_model_val_mba_std=nan,
            final_model_test_mba_std=nan,
            final_model_val_auroc_std=nan,
            final_model_test_auroc_std=nan,

            best_model_val_mba_mean=nan,
            best_model_test_mba_mean=nan,
            best_model_val_auroc_mean=nan,
            best_model_test_auroc_mean=nan,

            best_model_val_mba_std=nan,
            best_model_test_mba_std=nan,
            best_model_val_auroc_std=nan,
            best_model_test_auroc_std=nan,

            add_demographic_data_var=cmd_args.add_demo_data,
        )

        return RunComboResults(
            sheet_result=failed_result,
            best_trial_config_path_list=best_trial_config_path_list,
            optuna_id=optuna_id,
            best_trial_number=best_trial_number,
            best_value=best_value,
        )



def run_combo_worker(
    *,
    cmd_args: argparse.Namespace,
    logger: Any,
    original_output_dir: str,
    combo: "Combo",
    loaded_best_configs_dict: dict,
) -> RunComboResults:
    """
    Worker entrypoint for one Combo.
    Runs in a separate process when used with ProcessPoolExecutor.
    Creates its own logger (do NOT pass parent logger into workers).
    """

    # Ensure logs directory exists (safe even if already exists)
    logs_dir = Path(original_output_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    combo_tag = combo.tag

    # Per-combo logger (unique name + file)
    if not logger:
        logger = setup_run_logger(__name__, str(Path(cmd_args.output_dir) / f"{combo_tag}.log"))


    logger.info(
        "WORKER START: %s | Norm=%s | Dataset=%s | Conds=%s | Cat=%s | Model=%s",
        combo_tag,
        combo.norm_mode,
        combo.dataset_v,
        combo.lung_conditions,
        combo.rec_category,
        combo.model,
    )

    try:
        # Call your existing function that contains the old loop body
        run_combo_results = run_exhaustive_combo(
            cmd_args=cmd_args,
            logger=logger,
            original_output_dir=original_output_dir,
            combo_tag=combo.tag,
            norm_mode=combo.norm_mode,
            dataset_v=combo.dataset_v,
            lung_conditions=combo.lung_conditions,
            rec_category=combo.rec_category,
            model=combo.model,
            loaded_best_configs_dict=loaded_best_configs_dict,
        )

        logger.info("WORKER DONE: %s", combo_tag)
        return run_combo_results


    except Exception:
        logger.exception("WORKER CRASHED BEFORE RESULT: %s", combo.tag)
        raise




def save_best_configs_dict_artifacts(
    *,
    best_params_dict: dict,
    cmd_args: argparse.Namespace,
    test_mode: bool,
    run_id: str,
    logger: Any,
) -> Path | None:
    if not _excel_uses_tuning_results():
        logger.info("Skipping best configs dict save: hyperparameter tuning disabled")
        return None

    try:
        config_save_dir = Path(PATHS.BEST_CONFIGS_DIR) / PATHS.TARGET_BEST_CONFIG_VERSION_DIR
        config_save_dir.mkdir(parents=True, exist_ok=True)

        best_configs_dict_payload = {
            "_meta": {
                "schema_version": EXCEL.BEST_CONFIG_DICT_SCHEMA_VERSION,
            },
            **best_params_dict,
        }

        yaml_save_path = config_save_dir / build_best_configs_dict_filename(
            cmd_args=cmd_args,
            is_test_mode=test_mode,
            run_id=run_id,
        )

        with yaml_save_path.open("w", encoding="utf-8") as f:
            yaml.dump(best_configs_dict_payload, f, indent=4, sort_keys=False)
        logger.info(f"Saved best trials config paths to: {yaml_save_path}")
        return yaml_save_path
    except Exception:
        logger.exception("Failed to save best configs dict")
        return None


def save_excel_summaries(
    *,
    all_results: list[SheetResult],
    original_output_dir: str,
    cmd_args: argparse.Namespace,
    best_configs_dict_path: Path | None,
    logger: Any,
) -> tuple[str, str]:
    final_model_filename = construct_excel_summary_filename(
        model_variant="final",
        cmd_args=cmd_args,
    )
    best_model_filename = construct_excel_summary_filename(
        model_variant="best",
        cmd_args=cmd_args,
    )

    final_model_out_xlsx = create_and_save_excel_file(
        all_results,
        original_output_dir,
        filename=final_model_filename,
        model_variant="final",
        best_configs_dict_path=best_configs_dict_path,
        add_demo_data_column=False,
    )
    logger.info(f"Saved Excel summary to: {final_model_out_xlsx}")

    best_model_out_xlsx = create_and_save_excel_file(
        all_results,
        original_output_dir,
        filename=best_model_filename,
        model_variant="best",
        best_configs_dict_path=best_configs_dict_path,
        add_demo_data_column=False,
    )
    logger.info(f"Saved Excel summary to  : {best_model_out_xlsx}")
    return str(final_model_out_xlsx), str(best_model_out_xlsx)


def save_pickled_sheet_results(
    *,
    all_results: list[SheetResult],
    study_id: str,
    cmd_args,
    test_mode: bool | None = None,
    logger: Any,
) -> Path | None:
    try:
        target_dir = Path(PATHS.TARGET_SHEET_RESULTS_DIR)
        target_dir.mkdir(parents=True, exist_ok=True)

        pickle_path = target_dir / build_sheet_results_pickle_filename(
            study_id,
            cmd_args=cmd_args,
            test_mode=test_mode,
        )
        with pickle_path.open("wb") as f:
            pickle.dump(all_results, f)
        logger.info("Saved pickled sheet results to: %s", pickle_path)
        return pickle_path
    except Exception:
        logger.exception("Failed to save pickled sheet results")
        return None


@dataclass(frozen=True)
class Combo:
    tag: str
    norm_mode: str
    dataset_v: str
    lung_conditions: str
    rec_category: str
    model: str




# order: dataset_v --> lung_conditions --> recording_category --> model
def main(cmd_args):
    """
    Entry point for creating an exhaustive cross-validation study sheet.

    This function orchestrates a grid of cross-validation runs across
    combinations of:
        - lung condition sets
        - recording categories
        - model architectures

    Dataset selection and normalization mode (norm on/off)
    are intentionally handled **outside** this function via CLI arguments,
    so this function operates on exactly one dataset configuration per call.

    Behavior depends on execution mode:
        - Normal run:
            * Iterates over all permissible
              combinations for the dataset provided via CLI.
            * Optionally performs Optuna hyperparameter tuning BEFORE
              cross-validation and uses the best trial config.
            * Runs k-fold cross-validation and aggregates mean/std metrics.
            * Collects results into a unified Excel summary sheet.
            * Pickles the final `list[SheetResult]` at the end so the Excel
              summaries can be recreated later without rerunning the study.
            * Logs metadata, metrics, and artifacts to MLflow.
        - Lightweight simulation (`--test_run`):
            * Uses the TEST grid (small combo set) for fast pipeline checks.
            * Skips Optuna tuning.
            * Simulates summary metrics (dummy results) instead of running
              full cross-validation/training.
            * Validates high-level orchestration and output wiring only.

    Responsibilities:
        - Validate permissible experiment combinations.
        - Create isolated output directories per experiment.
        - Coordinate optional Optuna tuning.
        - Run or simulate cross-validation.
        - Aggregate results into a single Excel study sheet.
        - Persist the final `SheetResult` dataclass list as a pickle artifact.
        - Persist best hyperparameter configs (if tuning is enabled).
        - Log all relevant artifacts to MLflow.

    This function does NOT:
        - Decide dataset splits or normalization variants as that is handled by the caller
        - Implement model training logic itself (delegated to train_spectrogram).

    Args:
        cmd_args (argparse.Namespace):
            Parsed CLI arguments defining the dataset, model config,
            paths, and execution mode.
    """
    if cmd_args.model_type == "autoencoder":
        raise ValueError(
            "exhaustive_study mode does not support model_type='autoencoder'. "
            "Autoencoder is supported in train and optuna_tuning modes, but the "
            "exhaustive_study workflow delegates to cross-validation, which is "
            "currently implemented only for spectrogram training."
        )

    validate_seed_repeat_settings()


    Path(cmd_args.output_dir).mkdir(parents=True, exist_ok=True)
    test_mode = bool(cmd_args.test_run)


    if test_mode:
        cmd_args.dataset_v = "uk"
        NORM_MODES = [cmd_args.norm_mode]
        DATASET_V = [cmd_args.dataset_v]
        LUNG_CONDITIONS = list(EXCEL.TEST_GRID.LUNG_CONDITIONS)
        RECORDING_CATEGORY = list(EXCEL.TEST_GRID.RECORDING_CATEGORY)
        MODELS = [cmd_args.model_type]
        cmd_args.trials = 3
        cmd_args.epochs = 2
    else:
        NORM_MODES = [cmd_args.norm_mode]
        DATASET_V = [cmd_args.dataset_v]
        LUNG_CONDITIONS = list(EXCEL.FULL_GRID.LUNG_CONDITIONS)
        RECORDING_CATEGORY = list(EXCEL.FULL_GRID.RECORDING_CATEGORY)
        MODELS = [cmd_args.model_type]

    best_params_dict = {}

    original_output_dir = cmd_args.output_dir
    
    # setup logger
    logger = setup_run_logger(__name__, str(Path(cmd_args.output_dir) / FILES.LOGGER_FILENAME))

    loaded_best_configs_dict: dict[str, Any] = {}
    if _excel_uses_precomputed_best_configs():
        load_dir = Path(PATHS.BEST_CONFIGS_DIR) / PATHS.LOAD_BEST_CONFIG_VERSION_DIR
        best_config_dict_paths = discover_best_configs_dict_paths(
            load_dir,
            logger=logger,
        )
        loaded_best_configs_dict = load_and_merge_best_configs_dicts(
            best_config_dict_paths,
            logger=logger,
        )


    # --- Log Experiment Grid ---
    logger.info("="*20)
    logger.info("INITIALIZING EXHAUSTIVE STUDY RUN")
    logger.info(f"Norm modes: {NORM_MODES}")
    logger.info(f"Datasets:   {DATASET_V}")
    logger.info(f"Conditions: {LUNG_CONDITIONS}")
    logger.info(f"Categories: {RECORDING_CATEGORY}")
    logger.info(f"Models:     {MODELS}")
    logger.info(f"EXCEL.TUNING_MODE: {EXCEL.TUNING_MODE.value}")
    logger.info("="*20)

    sheet_study_id = build_study_run_name(cmd_args, test_run=test_mode)
    cmd_args.study_id = sheet_study_id
    all_results = []
    jobs: list[Combo] = []
    combo_idx = 0

    for norm_mode in NORM_MODES:
        for dataset_v in DATASET_V:
            for lung_conditions in LUNG_CONDITIONS:
                for rec_category in RECORDING_CATEGORY:
                    for model in MODELS:
                        combo_idx += 1
                        combo_tag = f"{combo_idx:03d}_combi"

                        if permissible_combination(
                            cmd_args.dataset_v,
                            dataset_v,
                            lung_conditions,
                            rec_category
                        ):
                            jobs.append(
                                Combo(
                                    tag=combo_tag,
                                    norm_mode=norm_mode,
                                    dataset_v=dataset_v,
                                    lung_conditions=lung_conditions,
                                    rec_category=rec_category,
                                    model=model,
                                )
                            )

    total_jobs = len(jobs)
    if total_jobs == 0:
        raise ValueError("No permissible experiment combinations found for exhaustive study sheet")

    yaml_save_path: Path | None = None
    final_model_out_xlsx: str | None = None
    best_model_out_xlsx: str | None = None
    sheet_results_pickle_path: Path | None = None

    # ---- Sequential fallback ----
    for combo_position, combo in enumerate(jobs, start=1):
        combo_result = run_combo_worker(
            cmd_args=cmd_args,
            logger=logger,
            original_output_dir=original_output_dir,
            combo=combo,
            loaded_best_configs_dict=loaded_best_configs_dict,
        )

        all_results.append(combo_result.sheet_result)

        if _excel_uses_tuning_results():
            store_best_configs_dict_entry(
                best_params_dict,
                norm_mode=combo.norm_mode,
                add_demo_data=cmd_args.add_demo_data,
                dataset_v=combo.dataset_v,
                lung_conditions=combo.lung_conditions,
                rec_category=combo.rec_category,
                model=combo.model,
                best_trial_config_path_list=combo_result.best_trial_config_path_list,
                optuna_id=combo_result.optuna_id,
            )

            yaml_save_path = save_best_configs_dict_artifacts(
                best_params_dict=best_params_dict,
                cmd_args=cmd_args,
                test_mode=test_mode,
                run_id=sheet_study_id,
                logger=logger,
            )

        final_model_out_xlsx, best_model_out_xlsx = save_excel_summaries(
            all_results=all_results,
            original_output_dir=original_output_dir,
            cmd_args=cmd_args,
            best_configs_dict_path=yaml_save_path,
            logger=logger,
        )
        logger.info(
            "Progress checkpoint saved after combo %s/%s (%s)",
            combo_position,
            total_jobs,
            combo.tag,
        )

    if final_model_out_xlsx is None or best_model_out_xlsx is None:
        raise ValueError("No Excel summaries were generated during exhaustive study run")

    sheet_results_pickle_path = save_pickled_sheet_results(
        all_results=all_results,
        study_id=sheet_study_id,
        cmd_args=cmd_args,
        test_mode=test_mode,
        logger=logger,
    )

    log_study_sheet_artifacts_to_mlflow(
        logger=logger,
        experiment_name=MLFLOW.EXPERIMENT_NAME,
        study_id=sheet_study_id,
        yaml_save_path=yaml_save_path,
        final_model_out_xlsx=final_model_out_xlsx,
        best_model_out_xlsx=best_model_out_xlsx,
        sheet_results_pickle_path=sheet_results_pickle_path,
    )
    logger.info(
        "EXHAUSTIVE_STUDY_RUN_COMPLETED: combos=%s/%s | final_excel=%s | best_excel=%s",
        len(all_results),
        total_jobs,
        final_model_out_xlsx,
        best_model_out_xlsx,
    )



        









