import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.schemas.dataclasses import TrainingArtefactsManager


def save_metrics_json_files(
    output_dir: str,
    best_model_name_info: str,
    final_model_name_info: str,
    art: TrainingArtefactsManager,
    final_model_test_results: Any,
    best_model_testing_results: Any,
    best_epoch: Optional[int] = None,
    final_epoch: Optional[int] = None,
    args_obj: Optional[Any] = None,
    trial=None,
    logger=None,
) -> None:
    metrics_output_dir = Path(output_dir)
    metrics_output_dir.mkdir(parents=True, exist_ok=True)

    trial_index = None
    if trial is not None and hasattr(trial, "number"):
        # Human-friendly 1-based trial index for filenames.
        trial_index = int(trial.number) + 1
    trial_prefix = f"{trial_index}_" if trial_index is not None else ""
    trial_number = int(trial.number) if trial is not None and hasattr(trial, "number") else None

    common_metadata = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_id": getattr(args_obj, "study_id", None) if args_obj is not None else None,
        "model_type": getattr(args_obj, "model_type", None) if args_obj is not None else None,
        "dataset_v": getattr(args_obj, "dataset_v", None) if args_obj is not None else None,
        "recording_category": getattr(args_obj, "recording_category", None) if args_obj is not None else None,
        "selected_classes": getattr(args_obj, "selected_classes", None) if args_obj is not None else None,
        "random_seed": getattr(args_obj, "random_seed", None) if args_obj is not None else None,
        "trial_number": trial_number,
        "trial_index_human": trial_index,
    }

    training_loop_payload = {
        "training_loop": {
            "train_losses": [float(x) for x in art.train_losses],
            "val_losses": [float(x) for x in art.val_losses],
            "val_balanced_accuracies": [float(x) for x in art.val_balanced_accuracies],
            "val_aurocs": [float(x) for x in art.val_aurocs],
            "weighted_train_losses": [float(x) for x in art.weighted_train_losses],
            "weighted_val_losses": [float(x) for x in art.weighted_val_losses],
            "lrs": [float(x) for x in art.lrs],
            "final_metrics": art.get_final_metrics(),
        },
    }

    best_model_payload = {
        **training_loop_payload,
        "metadata": {
            **common_metadata,
            "model_label": "best_model",
            "epoch_index": best_epoch,
            "epoch_human": (best_epoch + 1) if best_epoch is not None else None,
        },
        "best_model_test": {
            "avg_loss": float(best_model_testing_results.avg_loss),
            "accuracy": float(best_model_testing_results.accuracy),
            "balanced_accuracy": float(best_model_testing_results.balanced_accuracy),
            "precision_weighted": float(best_model_testing_results.precision_weighted),
            "recall_weighted": float(best_model_testing_results.recall_weighted),
            "f1_weighted": float(best_model_testing_results.f1_weighted),
            "auroc_macro_ovr": float(best_model_testing_results.auroc_macro_ovr),
            "confusion_matrix": best_model_testing_results.confusion_matrix.tolist(),
            "classification_report": str(best_model_testing_results.classification_report),
        },
    }

    final_model_payload = {
        **training_loop_payload,
        "metadata": {
            **common_metadata,
            "model_label": "final_model",
            "epoch_index": final_epoch,
            "epoch_human": (final_epoch + 1) if final_epoch is not None else None,
        },
        "final_model_test": {
            "avg_loss": float(final_model_test_results.avg_loss),
            "accuracy": float(final_model_test_results.accuracy),
            "balanced_accuracy": float(final_model_test_results.balanced_accuracy),
            "precision_weighted": float(final_model_test_results.precision_weighted),
            "recall_weighted": float(final_model_test_results.recall_weighted),
            "f1_weighted": float(final_model_test_results.f1_weighted),
            "auroc_macro_ovr": float(final_model_test_results.auroc_macro_ovr),
            "confusion_matrix": final_model_test_results.confusion_matrix.tolist(),
            "classification_report": str(final_model_test_results.classification_report),
        },
    }

    best_metrics_path = metrics_output_dir / f"{trial_prefix}{best_model_name_info}_best_model_metrics.json"
    with best_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(best_model_payload, f, indent=2)

    final_metrics_path = metrics_output_dir / f"{trial_prefix}{final_model_name_info}_final_model_metrics.json"
    with final_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(final_model_payload, f, indent=2)

    if logger:
        logger.info("Saved metrics JSON for best model: %s", best_metrics_path)
        logger.info("Saved metrics JSON for final model: %s", final_metrics_path)

