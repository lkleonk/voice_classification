"""MLflow utility functions for the spectrogram pipeline."""

from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, ContextManager

import mlflow

from src.settings import MLFLOW
from src.utils.toolkit.naming import get_norm_str


def safe_mlflow_call(
    enabled: bool,
    logger: Any,
    action: str,
    fn: Callable[..., Any],
    *fn_args: Any,
    **fn_kwargs: Any,
) -> tuple[bool, Any]:
    """
    Execute one MLflow call safely.

    Returns:
        (still_enabled, result)
        - still_enabled=False after the first failure, so callers can skip
          subsequent MLflow calls in the same run.
    """
    if not enabled:
        return False, None

    try:
        return True, fn(*fn_args, **fn_kwargs)
    except Exception as exc:  # pragma: no cover - depends on external MLflow runtime
        logger.warning("MLflow disabled after failure in %s: %s", action, exc)
        return False, None


def log_optuna_study_to_mlflow(
    *,
    enabled: bool,
    logger: Any,
    best_trial: Any,
    args: Any,
    study_id: str,
    output_dir: Path,
) -> bool:
    """
    Log final Optuna study metadata and artifacts to MLflow in one guarded block.
    """
    if not enabled:
        return False

    try:
        mlflow.log_param("best_trial_number", best_trial.number)
        mlflow.log_metric("best_value", best_trial.value)
        mlflow.log_params(best_trial.params)

        mlflow.set_tag("model_type", args.model_type)
        mlflow.set_tag("recording_category", args.recording_category)
        mlflow.set_tag("dataset_v", args.dataset_v)
        mlflow.set_tag("study_id", study_id)

        if output_dir.exists():
            mlflow.log_artifacts(str(output_dir))
            logger.info("Successfully logged artifacts from %s to MLflow", args.output_dir)

        tuning_file_path = Path(args.tuning_file)
        if tuning_file_path.exists():
            mlflow.log_artifact(str(tuning_file_path), artifact_path="config")
    except Exception as exc:
        logger.warning("MLflow Optuna summary logging failed: %s", exc)
        return False

    return True


def log_study_sheet_artifacts_to_mlflow(
    *,
    logger: Any,
    experiment_name: str,
    study_id: str,
    yaml_save_path: Path | None,
    final_model_out_xlsx: str,
    best_model_out_xlsx: str,
    sheet_results_pickle_path: Path | None,
) -> None:
    """
    Log exhaustive study sheet artifacts to MLflow in one guarded block.
    """
    try:
        mlflow.set_experiment(experiment_name)
        nested_run = mlflow.active_run() is not None
        with mlflow.start_run(run_name=study_id, nested=nested_run):
            mlflow.set_tag("study_id", study_id)
            if yaml_save_path is not None and yaml_save_path.exists():
                mlflow.log_artifact(str(yaml_save_path))
            if Path(final_model_out_xlsx).exists():
                mlflow.log_artifact(str(final_model_out_xlsx))
            if Path(best_model_out_xlsx).exists():
                mlflow.log_artifact(str(best_model_out_xlsx))
            if sheet_results_pickle_path is not None and sheet_results_pickle_path.exists():
                mlflow.log_artifact(str(sheet_results_pickle_path))
    except Exception:  # pragma: no cover - depends on external MLflow runtime
        logger.exception("MLflow logging failed")


def init_cv_mlflow_run(
    cmd_args: Any,
    effective_k: int,
    cv_study_id: str,
    run_name: str,
    logger: Any,
) -> tuple[bool, ContextManager[Any]]:
    """
    Initialize MLflow for a cross-validation parent run.

    Returns:
        (mlflow_enabled, run_context)
    """
    mlflow_enabled = True
    run_context: ContextManager[Any] = nullcontext()

    try:
        mlflow.set_experiment(MLFLOW.EXPERIMENT_NAME)
        if mlflow.active_run() is None:
            run_context = mlflow.start_run(run_name=run_name)
        mlflow.log_params(vars(cmd_args))
        mlflow.log_param("k_folds_effective", effective_k)
        mlflow.set_tag("study_id", cv_study_id)
    except Exception as exc:  # pragma: no cover - depends on external MLflow runtime
        mlflow_enabled = False
        run_context = nullcontext()
        logger.warning("MLflow setup/logging failed. Continuing without MLflow: %s", exc)

    return mlflow_enabled, run_context


def finalize_cv_mlflow(
    summary_dict: dict[str, Any],
    config_file: str,
    mlflow_enabled: bool,
    logger: Any,
) -> None:
    """Log final cross-validation metrics and config artifact."""
    if not mlflow_enabled:
        return

    try:
        mlflow.log_metrics(summary_dict)
        if os.path.exists(config_file):
            mlflow.log_artifact(config_file)
    except Exception as exc:  # pragma: no cover - depends on external MLflow runtime
        logger.warning("MLflow final logging failed. Continuing without MLflow: %s", exc)


def init_training_mlflow_run(
    *,
    run_name: str,
    logger: Any,
) -> tuple[bool, ContextManager[Any]]:
    """
    Initialize MLflow context for one training run.

    Returns:
        (mlflow_enabled, run_context)
    """
    mlflow_enabled = True
    run_context: ContextManager[Any] = nullcontext()

    try:
        mlflow.set_experiment(MLFLOW.EXPERIMENT_NAME)
        nested_run = mlflow.active_run() is not None
        run_context = mlflow.start_run(run_name=run_name, nested=nested_run)
    except Exception as exc:  # pragma: no cover - depends on external MLflow runtime
        mlflow_enabled = False
        run_context = nullcontext()
        logger.warning("MLflow training run init failed. Continuing without MLflow: %s", exc)

    return mlflow_enabled, run_context


def log_training_summary_to_mlflow(
    *,
    mlflow_enabled: bool,
    logger: Any,
    name_info: str,
    cli_args: Any,
    config: dict[str, Any],
    mark_good_run: bool,
    best_model_epoch: Any,
    best_model_val_mba: float,
    best_model_test_mba: float,
    final_model_val_mba: float,
    final_model_test_mba: float,
) -> bool:
    """
    Log training run identity, tags, and summary metrics in one guarded block.
    """
    if not mlflow_enabled:
        return False

    try:
        if mark_good_run:
            ignore_cli_keys = {"config_file", "metadata_file"}
            run_identity = {
                "run.name": name_info,
                "run.norm": get_norm_str(cli_args.norm_mode),
            }
            mlflow.log_params(run_identity)
            mlflow.log_params(
                {f"cli.{k}": v for k, v in vars(cli_args).items() if k not in ignore_cli_keys}
            )
            mlflow.log_params(config)
            mlflow.log_metric("best_model_epoch", best_model_epoch)
            mlflow.log_metric("best_model.val_mba", best_model_val_mba)
            mlflow.log_metric("best_model.test_mba", best_model_test_mba)
            mlflow.log_metric("final_model.val_mba", final_model_val_mba)
            mlflow.log_metric("final_model.test_mba", final_model_test_mba)
            mlflow.set_tag("good_run", "true")
        else:
            mlflow.set_tag("good_run", "false")
    except Exception as exc:  # pragma: no cover - depends on external MLflow runtime
        logger.warning("MLflow training summary logging failed. Continuing without MLflow: %s", exc)
        return False

    return True

