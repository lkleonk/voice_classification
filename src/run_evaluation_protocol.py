import argparse
import copy
import os
import random
import time
from datetime import datetime, timedelta
from typing import Any, cast

import yaml

# Add project root to Python path
from numpy import mean, std

import src.train_spectrogram as train_spectrogram
from src.model.pretrained.pretrained_models_params import overwrite_config_values
from src.schemas.enums import RunMode
from src.schemas.typed_dicts import CrossValSummary
from src.settings import EVALUATION_PROTOCOL, MODELS, STUDY_IDS
from src.utils.data_preprocessing.dataset_split_manager import (
    load_dataset_managers_kfold,
)
from src.utils.reporting.logger_setup import setup_run_logger
from src.utils.reporting.mlflow_helper import (
    finalize_cv_mlflow,
    init_cv_mlflow_run,
)
from src.utils.toolkit.naming import build_cv_run_name


def make_cv_run_name(cmd_args, effective_k: int, run_id: int) -> str:
    return build_cv_run_name(
        cmd_args,
        effective_k,
        run_prefix=STUDY_IDS.CROSS_VALIDATION_PREFIX,
        run_id_range=(STUDY_IDS.MIN_NUMBER, STUDY_IDS.MAX_NUMBER),
        run_id=run_id,
    )


def prepare_effective_config(config: dict[str, Any], model_type: str) -> dict[str, Any]:
    model_key = model_type.lower()
    if (
        model_key in MODELS.PRETRAINED_MODELS_LIST
        or model_key in MODELS.DEMO_ONLY_NO_AUDIO_MODELS_LIST
    ):
        config = overwrite_config_values(config, model_type)
    return config




def validate_seed_repeat_settings() -> None:
    only_change_random_seeds = getattr(EVALUATION_PROTOCOL, "ONLY_CHANGE_RANDOM_SEEDS", False)
    if not only_change_random_seeds:
        return

    random_seed_runs = int(getattr(EVALUATION_PROTOCOL, "RANDOM_SEEDS_RUNS", 1))
    if random_seed_runs < 1:
        raise ValueError(f"EVALUATION_PROTOCOL.RANDOM_SEEDS_RUNS must be >= 1, got {random_seed_runs}")

    if random_seed_runs > len(EVALUATION_PROTOCOL.EVAL_SEED_POOL):
        raise ValueError(
            f"random_seed_runs ({random_seed_runs}) exceeds EVAL_SEED_POOL size ({len(EVALUATION_PROTOCOL.EVAL_SEED_POOL)})"
        )


def get_effective_random_seed_runs(cmd_args: argparse.Namespace) -> int:
    if getattr(cmd_args, "test_run", False):
        return int(getattr(EVALUATION_PROTOCOL, "TEST_RUN_RANDOM_SEEDS_RUNS", 1))
    return int(getattr(EVALUATION_PROTOCOL, "RANDOM_SEEDS_RUNS", 1))


def get_effective_k_folds(cmd_args: argparse.Namespace) -> int:
    if getattr(cmd_args, "test_run", False):
        return int(EVALUATION_PROTOCOL.TEST_RUN_K_FOLDS)
    return int(EVALUATION_PROTOCOL.K_FOLDS)


def run_evaluation_protocol(cmd_args: argparse.Namespace) -> CrossValSummary:
    """
    Runs k-fold cross-validation using fixed hyperparameters loaded from a YAML
    configuration file.

    For each fold, a separate dataset manager is created and training is executed.
    Validation and test metrics are collected per fold and averaged at the end.

    If `cmd_args.test_run` is enabled, the number of folds is reduced for faster
    debugging. Each fold writes its outputs to a dedicated, timestamped directory.

    Args:
        config_file: Path to the YAML configuration file.
        cmd_args: Parsed CLI arguments controlling training and output paths.

    Returns:
        CrossValSummary with mean validation/test MBA and AUROC across folds.
    """
    cv_start_dt = datetime.now()
    cv_start_t = time.perf_counter()
    os.makedirs(cmd_args.output_dir, exist_ok=True)
    # Load config
    with open(cmd_args.config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config = prepare_effective_config(config, cmd_args.model_type)
    logger = setup_run_logger(__name__, os.path.join(cmd_args.output_dir, 'logger.log'))



    effective_k = get_effective_k_folds(cmd_args)
    validate_seed_repeat_settings()
    only_change_random_seeds = getattr(EVALUATION_PROTOCOL, "ONLY_CHANGE_RANDOM_SEEDS", False)
    random_seed_runs = get_effective_random_seed_runs(cmd_args)
    mode_label = "seed-repeat" if only_change_random_seeds else "k-fold"
    planned_runs = random_seed_runs if only_change_random_seeds else effective_k

    logger.info(
        "CV SETUP | dataset=%s | classes=%s | category=%s | model=%s | mode=%s | runs=%s",
        cmd_args.dataset_v,
        cmd_args.selected_classes,
        cmd_args.recording_category,
        cmd_args.model_type,
        mode_label,
        planned_runs,
    )

    original_output_dir = cmd_args.output_dir

    final_model_val_mba_list = []
    final_model_test_mba_list = []
    final_model_val_auroc_list = []
    final_model_test_auroc_list = []

    best_model_val_mba_list = []
    best_model_test_mba_list = []
    best_model_val_auroc_list = []
    best_model_test_auroc_list = []


    cv_run_number = random.randint(STUDY_IDS.MIN_NUMBER, STUDY_IDS.MAX_NUMBER)
    cv_study_id = f"{STUDY_IDS.CROSS_VALIDATION_PREFIX}_{cv_run_number}"
    cmd_args.study_id = cv_study_id
    run_name = make_cv_run_name(cmd_args, effective_k, run_id=cv_run_number)
    logger.info("CV IDs | study_id=%s | run_name=%s", cv_study_id, run_name)

    mlflow_enabled, run_context = init_cv_mlflow_run(
        cmd_args=cmd_args,
        effective_k=effective_k,
        cv_study_id=cv_study_id,
        run_name=run_name,
        logger=logger,
    )

    with run_context:
        if only_change_random_seeds:
            logger.info("Seed-repeat mode enabled. Skipping k-fold and running %s seed variants.", random_seed_runs)

            for run_idx in range(random_seed_runs):
                run_args = copy.deepcopy(cmd_args)
                run_args.random_seed = EVALUATION_PROTOCOL.EVAL_SEED_POOL[run_idx]
                run_args.study_id = cv_study_id
                logger.info(
                    "[%s/%s] START seed_run | idx=%s | seed=%s | output_dir=%s",
                    run_idx + 1,
                    random_seed_runs,
                    run_idx,
                    run_args.random_seed,
                    run_args.output_dir,
                )

                _run_mode = RunMode.NORMAL_TRAIN_ALWAYS_WITHOUT_RESULTS
                train_result = train_spectrogram.run_training(
                    run_args,
                    config=config,
                    run_mode=_run_mode,
                    training_run_name=f"{cv_study_id}_seed_run{run_idx}_seed{run_args.random_seed}",
                )

                final_model_val_mba_list.append(train_result.final_model_val_mba)
                final_model_test_mba_list.append(train_result.final_model_test_mba)
                final_model_val_auroc_list.append(train_result.final_model_val_auroc)
                final_model_test_auroc_list.append(train_result.final_model_test_auroc)

                best_model_val_mba_list.append(train_result.best_model_val_mba)
                best_model_test_mba_list.append(train_result.best_model_test_mba)
                best_model_val_auroc_list.append(train_result.best_model_val_auroc)
                best_model_test_auroc_list.append(train_result.best_model_test_auroc)
                logger.info(
                    "[%s/%s] DONE seed_run | idx=%s | seed=%s | "
                    "final(val_mba=%.4f,test_mba=%.4f,val_auroc=%.4f,test_auroc=%.4f) | "
                    "best(val_mba=%.4f,test_mba=%.4f,val_auroc=%.4f,test_auroc=%.4f)",
                    run_idx + 1,
                    random_seed_runs,
                    run_idx,
                    run_args.random_seed,
                    train_result.final_model_val_mba,
                    train_result.final_model_test_mba,
                    train_result.final_model_val_auroc,
                    train_result.final_model_test_auroc,
                    train_result.best_model_val_mba,
                    train_result.best_model_test_mba,
                    train_result.best_model_val_auroc,
                    train_result.best_model_test_auroc,
                )
        else:
            dataset_manager_list = load_dataset_managers_kfold(cmd_args, config, logger, k_folds=effective_k)

            for x, fold in enumerate(dataset_manager_list):
                cmd_args.output_dir = os.path.join(original_output_dir, f"fold{x}")
                cmd_args.study_id = cv_study_id
                # train_spectrogram.run_training(...) opens a nested child run because this CV run is active
                logger.info(
                    "[%s/%s] START fold | fold_idx=%s | output_dir=%s",
                    x + 1,
                    effective_k,
                    x,
                    cmd_args.output_dir,
                )
                _run_mode = RunMode.NORMAL_TRAIN_ALWAYS_WITHOUT_RESULTS
                train_result = train_spectrogram.run_training(
                    cmd_args,
                    config=config,
                    dataset_manager=fold,
                    run_mode=_run_mode,
                    training_run_name=f"{cv_study_id}_fold{x}",
                )
                # FINAL model
                final_model_val_mba_list.append(train_result.final_model_val_mba)
                final_model_test_mba_list.append(train_result.final_model_test_mba)
                final_model_val_auroc_list.append(train_result.final_model_val_auroc)
                final_model_test_auroc_list.append(train_result.final_model_test_auroc)

                # BEST model
                best_model_val_mba_list.append(train_result.best_model_val_mba)
                best_model_test_mba_list.append(train_result.best_model_test_mba)
                best_model_val_auroc_list.append(train_result.best_model_val_auroc)
                best_model_test_auroc_list.append(train_result.best_model_test_auroc)
                logger.info(
                    "[%s/%s] DONE fold | fold_idx=%s | "
                    "final(val_mba=%.4f,test_mba=%.4f,val_auroc=%.4f,test_auroc=%.4f) | "
                    "best(val_mba=%.4f,test_mba=%.4f,val_auroc=%.4f,test_auroc=%.4f)",
                    x + 1,
                    effective_k,
                    x,
                    train_result.final_model_val_mba,
                    train_result.final_model_test_mba,
                    train_result.final_model_val_auroc,
                    train_result.final_model_test_auroc,
                    train_result.best_model_val_mba,
                    train_result.best_model_test_mba,
                    train_result.best_model_val_auroc,
                    train_result.best_model_test_auroc,
                )


        logger.info('Cross validation finished. Now mean and std values are going to be calculated.')
        summary_dict = {
            "run_count": random_seed_runs if only_change_random_seeds else effective_k,
            "aggregation_mode": "rseeds" if only_change_random_seeds else "kfold",

             # FINAL model
            "final_model_mean_val_mba": mean(final_model_val_mba_list),
            "final_model_std_val_mba": std(final_model_val_mba_list),
            "final_model_mean_test_mba": mean(final_model_test_mba_list),
            "final_model_std_test_mba": std(final_model_test_mba_list),
            "final_model_mean_val_auroc": mean(final_model_val_auroc_list),
            "final_model_std_val_auroc": std(final_model_val_auroc_list),
            "final_model_mean_test_auroc": mean(final_model_test_auroc_list),
            "final_model_std_test_auroc": std(final_model_test_auroc_list),

            # BEST model
            "best_model_mean_val_mba": mean(best_model_val_mba_list),
            "best_model_std_val_mba": std(best_model_val_mba_list),
            "best_model_mean_test_mba": mean(best_model_test_mba_list),
            "best_model_std_test_mba": std(best_model_test_mba_list),
            "best_model_mean_val_auroc": mean(best_model_val_auroc_list),
            "best_model_std_val_auroc": std(best_model_val_auroc_list),
            "best_model_mean_test_auroc": mean(best_model_test_auroc_list),
            "best_model_std_test_auroc": std(best_model_test_auroc_list),
        }
        logger.info(
            f"Final aggregated metrics ({'seed-repeats' if only_change_random_seeds else 'k-fold'}): "
            f"final_model_val_mba={summary_dict['final_model_mean_val_mba']:.6f}Â±{summary_dict['final_model_std_val_mba']:.6f}, "
            f"final_model_test_mba={summary_dict['final_model_mean_test_mba']:.6f}Â±{summary_dict['final_model_std_test_mba']:.6f}, "
            f"final_model_val_auroc={summary_dict['final_model_mean_val_auroc']:.6f}Â±{summary_dict['final_model_std_val_auroc']:.6f}, "
            f"final_model_test_auroc={summary_dict['final_model_mean_test_auroc']:.6f}Â±{summary_dict['final_model_std_test_auroc']:.6f}"
        )
        logger.info(
            f"Best aggregated metrics ({'seed-repeats' if only_change_random_seeds else 'k-fold'}): "
            f"best_model_val_mba={summary_dict['best_model_mean_val_mba']:.6f}Â±{summary_dict['best_model_std_val_mba']:.6f}, "
            f"best_model_test_mba={summary_dict['best_model_mean_test_mba']:.6f}Â±{summary_dict['best_model_std_test_mba']:.6f}, "
            f"best_model_val_auroc={summary_dict['best_model_mean_val_auroc']:.6f}Â±{summary_dict['best_model_std_val_auroc']:.6f}, "
            f"best_model_test_auroc={summary_dict['best_model_mean_test_auroc']:.6f}Â±{summary_dict['best_model_std_test_auroc']:.6f}"
        )
        finalize_cv_mlflow(
            summary_dict=summary_dict,
            config_file=cmd_args.config_file,
            mlflow_enabled=mlflow_enabled,
            logger=logger,
        )

    cv_end_dt = datetime.now()
    elapsed_seconds = time.perf_counter() - cv_start_t
    duration_str = str(timedelta(seconds=round(elapsed_seconds)))
    logger.info(
        "CV FINISHED | start=%s | end=%s | duration=%s | mode=%s | runs=%s | study_id=%s",
        cv_start_dt.isoformat(timespec="seconds"),
        cv_end_dt.isoformat(timespec="seconds"),
        duration_str,
        mode_label,
        planned_runs,
        cv_study_id,
    )

    return cast(
        CrossValSummary,
        summary_dict,
    )


run_cross_validation = run_evaluation_protocol
