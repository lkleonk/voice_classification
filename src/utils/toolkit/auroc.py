import logging
from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score


def calculate_dynamic_auroc(
    targets: np.ndarray, 
    probs: np.ndarray, 
    logger: Optional[logging.Logger] = None, 
    info: str = ""
) -> float:
    """
    Calculates AUROC, automatically handling binary vs multiclass (OVR macro).
    Returns float('nan') if calculation fails or if only one class is present.
    """
    if targets.size == 0 or probs.size == 0:
        return float("nan")

    if probs.ndim != 2:
        if logger:
            logger.warning(f"AUROC {info}: probs must be 2D, got {probs.shape}")
        return float("nan")

    present_classes = np.unique(targets)
    if len(present_classes) < 2:
        # AUROC undefined for single class
        return float("nan")

    try:
        # Case 1: Binary classification (2 columns in probs)
        if probs.shape[1] == 2:
            return float(roc_auc_score(targets, probs[:, 1]))
        
        # Case 2: Multiclass OVR Macro
        return float(
            roc_auc_score(
                targets, 
                probs, 
                multi_class="ovr", 
                average="macro"
            )
        )
    except ValueError as e:
        if logger:
            logger.warning(f"AUROC {info} failed: {e}")
        return float("nan")
    except Exception as e:
        if logger:
            logger.error(f"AUROC {info} unexpected error: {e}")
        return float("nan")
