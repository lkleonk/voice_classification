from typing import List

# AUROC (multiclass OVR macro). Can fail if a class is missing in y_true.
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.schemas.dataclasses import EvalResult
from src.utils.toolkit.auroc import calculate_dynamic_auroc


def eval_model(
    model,
    device,
    test_dataloader,
    label_encoder,
    loss_func,
    logger,
    add_demographic_data_bool: bool,
    info: str = "",
) -> EvalResult:
    model.eval()
    model.to(device)

    total_loss = 0.0
    total_samples = 0

    all_targets: List[int] = []
    all_preds: List[int] = []
    all_logits: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for xb, yb, demo_b in test_dataloader:
            xb, yb, demo_b = xb.to(device), yb.to(device), demo_b.to(device)

            logits = model(xb, demo_b) if add_demographic_data_bool else model(xb)  # (B, C)
            loss = loss_func(logits, yb)

            bs = xb.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            probs = torch.softmax(logits, dim=1)  # (B, C)
            preds = probs.argmax(dim=1)          # (B,)

            all_targets.extend(yb.detach().cpu().numpy().tolist())
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_logits.append(logits.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())

    targets = np.asarray(all_targets, dtype=int)
    preds = np.asarray(all_preds, dtype=int)
    logits_np = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0,))
    probs_np = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0,))

    avg_loss = (total_loss / total_samples) if total_samples > 0 else float("nan")

    accuracy = accuracy_score(targets, preds) if targets.size else float("nan")
    bal_accuracy = balanced_accuracy_score(targets, preds) if targets.size else float("nan")

    precision_w = precision_score(targets, preds, average="weighted", zero_division=0) if targets.size else float("nan")
    recall_w = recall_score(targets, preds, average="weighted", zero_division=0) if targets.size else float("nan")
    f1_w = f1_score(targets, preds, average="weighted", zero_division=0) if targets.size else float("nan")

    conf = confusion_matrix(targets, preds) if targets.size else np.zeros((0, 0), dtype=int)


    # --- AUROC (multiclass OVR macro) with detailed diagnostics ---
    auroc_macro_ovr = calculate_dynamic_auroc(targets, probs_np, logger, info)


    class_names = getattr(label_encoder, "classes_", None)
    if class_names is None:
        # fallback: just use numeric labels as strings
        class_names = [str(i) for i in range(probs_np.shape[1] if probs_np.ndim == 2 else int(targets.max() + 1))]

    report = classification_report(
        targets,
        preds,
        target_names=class_names,
        zero_division=0,
    ) if targets.size else "Classification report: No samples."

    metrics_text = (
        f"\n=== Evaluation Metrics {info} ===\n"
        f"Avg Loss:           {avg_loss:.4f}\n"
        f"Precision (w):      {precision_w:.4f}\n"
        f"Recall (w):         {recall_w:.4f}\n"
        f"F1 (w):             {f1_w:.4f}\n"
        f"Accuracy:           {accuracy:.4f}\n"
        f"Balanced Accuracy:  {bal_accuracy:.4f}\n"
        f"AUROC (macro-ovr):  {auroc_macro_ovr:.4f}\n"
        f"\nConfusion Matrix:\n{conf}\n"
        f"\n{'=' * 60}\n"
        f"Classification Report ({info})\n"
        f"{'=' * 60}\n"
        f"{report}\n"
    )

    logger.info(metrics_text)

    return EvalResult(
        info=info,
        avg_loss=avg_loss,
        accuracy=float(accuracy),
        balanced_accuracy=bal_accuracy,
        precision_weighted=float(precision_w),
        recall_weighted=float(recall_w),
        f1_weighted=float(f1_w),
        auroc_macro_ovr=float(auroc_macro_ovr),
        confusion_matrix=conf,
        classification_report=str(report),
        metrics_text=metrics_text,
        targets=targets,
        preds=preds,
        logits=logits_np if logits_np.size else None,
        probs=probs_np if probs_np.size else None,
    )

