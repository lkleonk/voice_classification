from __future__ import annotations

import random
from typing import Any, Optional
from collections.abc import Sequence

from src.settings import EVALUATION_PROTOCOL, EXCEL, PATHS, STUDY_IDS


def _new_study_numeric_id() -> int:
    """Return a random numeric ID within the configured study ID range."""
    return random.randint(STUDY_IDS.MIN_NUMBER, STUDY_IDS.MAX_NUMBER)


def _sanitize_token(value: str) -> str:
    """Replace commas with underscores for safe use in filenames."""
    return value.replace(",", "_")


def _compose_name_from_tokens(*tokens: str) -> str:
    """Join non-empty tokens with underscores into a single name string."""
    return "_".join(token for token in tokens if token)


def _canonicalize_csv_tokens(value: str) -> str:
    """Make comma-separated labels order-invariant, e.g. 'copd,control' -> 'cont_copd'."""
    parts = [part.strip().lower() for part in str(value).split(",") if part.strip()]
    if len(parts) <= 1:
        return str(value).strip().lower()[:4]
    canonical_parts = sorted(set(parts))
    shortened_parts = [part[:4] for part in canonical_parts]
    return "_".join(shortened_parts)


def abbreviate(terms: str) -> str:
    """Abbreviate comma-separated terms to their first 3 characters joined by underscores."""
    return '_'.join(term.strip()[:3] for term in terms.split(','))


def _best_configs_demo_data_key(add_demo_data: bool) -> str:
    """Return 'w_demo_data' or 'no_demo_data' based on the demo data flag."""
    return "w_demo_data" if add_demo_data else "no_demo_data"


def get_norm_str(norm_mode: str) -> str:
    """Return 'norm_off', 'norm_on', or 'unknown' depending on the norm_mode string."""
    if "norm_off" == norm_mode:
        return "norm_off"
    if "norm_on" == norm_mode:
        return "norm_on"
    return "unknown"


def _build_stem_components(*, cmd_args: Any, test_mode: bool | None = None) -> tuple[str, str, bool, str]:
    """Compute shared (tuning_label, demo_data_label, effective_test_mode, mode_label) for filename stems."""
    from src.run_evaluation_protocol import (
        get_effective_k_folds,
        get_effective_random_seed_runs,
    )

    tuning_label = (
        "with_tuning"
        if _excel_uses_tuning_results()
        else "no_tuning"
    )

    demo_data_label = _best_configs_demo_data_key(cmd_args.add_demo_data)

    effective_test_mode = (
        bool(getattr(cmd_args, "test_run", False))
        if test_mode is None
        else bool(test_mode)
    )
    if EVALUATION_PROTOCOL.ONLY_CHANGE_RANDOM_SEEDS:
        seed_run_count = (
            get_effective_random_seed_runs(cmd_args)
            if test_mode is None
            else (
                int(getattr(EVALUATION_PROTOCOL, "TEST_RUN_RANDOM_SEEDS_RUNS", 1))
                if effective_test_mode
                else int(getattr(EVALUATION_PROTOCOL, "RANDOM_SEEDS_RUNS", 1))
            )
        )
        mode_label = f"{seed_run_count}seeds"
    else:
        k = (
            get_effective_k_folds(cmd_args)
            if test_mode is None
            else (
                int(EVALUATION_PROTOCOL.TEST_RUN_K_FOLDS)
                if effective_test_mode
                else int(EVALUATION_PROTOCOL.K_FOLDS)
            )
        )
        mode_label = f"{k}fold"

    return tuning_label, demo_data_label, effective_test_mode, mode_label


def _excel_uses_tuning_results() -> bool:
    """Check whether the configured EXCEL.TUNING_MODE requires hyperparameter tuning results."""
    return EXCEL.TUNING_MODE in {
        EXCEL.TuningMode.RUN_OPTUNA_FIRST,
        EXCEL.TuningMode.USE_PRECOMPUTED_BEST_CONFIGS,
    }


def construct_exhaustive_artifact_filename_stem(
    *,
    cmd_args: Any,
    test_mode: bool | None = None,
) -> str:
    """Build a descriptive filename stem encoding norm, dataset, tuning, demo-data, and eval mode."""
    tuning_label, demo_data_label, _, mode_label = _build_stem_components(
        cmd_args=cmd_args, test_mode=test_mode,
    )
    return (
        f"{cmd_args.norm_mode}_"
        f"{cmd_args.dataset_v}_"
        f"{tuning_label}_"
        f"{demo_data_label}_"
        f"{mode_label}"
    )


def construct_sheet_results_pickle_filename_stem(
    *,
    cmd_args: Any,
    test_mode: bool | None = None,
) -> str:
    """Build a filename stem for pickled SheetResult artifacts, including model type and test-run flag."""
    tuning_label, demo_data_label, effective_test_mode, mode_label = _build_stem_components(
        cmd_args=cmd_args, test_mode=test_mode,
    )
    filename_parts = [
        cmd_args.dataset_v,
        cmd_args.norm_mode,
        tuning_label,
        demo_data_label,
    ]
    filename_parts.insert(1, cmd_args.model_type)
    if effective_test_mode:
        filename_parts.append("test_run")
    filename_parts.append(mode_label)
    return "_".join(filename_parts)


def build_best_configs_dict_filename(
    *,
    cmd_args: Any,
    is_test_mode: bool,
    combo: Any = None,
    run_id: str | None = None,
) -> str:
    """Build a YAML filename for the best hyperparameter configs dictionary."""
    filename_stem = construct_exhaustive_artifact_filename_stem(
        cmd_args=cmd_args,
        test_mode=is_test_mode,
    )
    if run_id:
        filename_stem = f"{run_id}_{filename_stem}"
    return f"{filename_stem}{PATHS.BEST_CONFIGS_DICT_FILENAME_SUFFIX}"


def construct_excel_summary_filename(
    *,
    model_variant: str,
    cmd_args: Any,
) -> str:
    """Build an Excel summary filename prefixed with the model variant (FINAL/BEST)."""
    filename_stem = construct_exhaustive_artifact_filename_stem(
        cmd_args=cmd_args,
    )
    return (
        f"{model_variant.upper()}_MODEL_"
        f"{filename_stem}_"
        f"excel_summary.xlsx"
    )


def build_sheet_results_pickle_filename(
    study_id: str,
    *,
    cmd_args: Any,
    test_mode: bool | None = None,
) -> str:
    """Build a pickle filename for SheetResult lists, prefixed with the study ID."""
    filename_stem = construct_sheet_results_pickle_filename_stem(
        cmd_args=cmd_args,
        test_mode=test_mode,
    )
    return f"{study_id}_{filename_stem}{PATHS.SHEET_RESULTS_PICKLE_SUFFIX}"


def build_training_run_name(
    args: Any,
    *,
    pretrained: bool = False,
    run_id: int | None = None,
) -> str:
    """Build a human-readable name for a single training run."""
    if run_id is None:
        run_id = _new_study_numeric_id()

    norm_token = get_norm_str(args.norm_mode)
    selected_conditions_token = _canonicalize_csv_tokens(args.selected_classes)

    return _compose_name_from_tokens(
        str(run_id),
        norm_token,
        selected_conditions_token,
        str(args.dataset_v),
        "TEST_run" if getattr(args, "test_run", False) else "",
        "pretrained" if pretrained else "",
        str(args.model_type),
        _sanitize_token(str(args.recording_category)),
    )


def build_cv_run_name(
    cmd_args: Any,
    effective_k: int,
    *,
    run_prefix: str = STUDY_IDS.CROSS_VALIDATION_PREFIX,
    run_id_range: Sequence[int] = (STUDY_IDS.MIN_NUMBER, STUDY_IDS.MAX_NUMBER),
    run_id: int | None = None,
) -> str:
    """Build a name for a cross-validation run including fold count and test/real indicator."""
    if run_id is None:
        run_id = random.randint(int(run_id_range[0]), int(run_id_range[1]))

    norm_token = get_norm_str(cmd_args.norm_mode)
    return _compose_name_from_tokens(
        str(run_prefix),
        str(run_id),
        str(cmd_args.dataset_v),
        _sanitize_token(str(cmd_args.selected_classes)),
        _sanitize_token(str(cmd_args.recording_category)),
        str(cmd_args.model_type),
        norm_token,
        f"k{effective_k}",
        "TEST" if getattr(cmd_args, "test_run", False) else "REAL",
    )


def build_study_run_name(
    cmd_args: Any,
    *,
    test_run: bool = False,
    study_id: str | None = None,
    run_id: int | None = None,
) -> str:
    """Build a name for an exhaustive study run, optionally using a pre-assigned study ID."""
    if study_id is None and run_id is None:
        run_id = _new_study_numeric_id()

    norm_token = get_norm_str(cmd_args.norm_mode)
    test_token = "TEST" if (getattr(cmd_args, "test_run", False) or test_run) else "REAL"
    id_part = (
        study_id
        if study_id is not None
        else f"{STUDY_IDS.EXHAUSTIVE_SHEET_RUN_NAME_PREFIX}_{run_id}"
    )

    return _compose_name_from_tokens(
        str(id_part),
        str(cmd_args.dataset_v),
        _sanitize_token(str(cmd_args.selected_classes)),
        _sanitize_token(str(cmd_args.recording_category)),
        str(cmd_args.model_type),
        norm_token,
        test_token,
    )


def build_best_config_filename(
    *,
    model_type: str,
    dataset_v: str,
    selected_classes: str,
    recording_category: str,
    timestamp: str,
    rank: int,
    study_id: str | None = None,
) -> str:
    """Build a YAML filename for an individual best-config artifact with rank and timestamp."""
    canonical_selected_classes = _canonicalize_csv_tokens(str(selected_classes))
    canonical_recording_category = _canonicalize_csv_tokens(str(recording_category))
    return (
        _compose_name_from_tokens(
            str(study_id) if study_id else "",
            "best_config",
            str(model_type),
            str(dataset_v),
            _sanitize_token(canonical_selected_classes),
            _sanitize_token(canonical_recording_category),
            str(timestamp),
            f"rank_{rank}",
        )
        + ".yaml"
    )


def create_study_id(prefix: str) -> str:
    """Create a unique study ID by appending a random numeric suffix to the given prefix."""
    study_numeric_id = _new_study_numeric_id()
    return f"{prefix}_{study_numeric_id}"
