from __future__ import annotations

from typing import Iterator

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold


def _iter_group_folds(
    y: np.ndarray,
    group_ids: np.ndarray,
    *,
    n_splits: int,
    random_state: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    dummy_x = np.zeros(len(y), dtype=np.float32)
    return splitter.split(dummy_x, y, groups=group_ids)


def _first_group_split(
    y: np.ndarray,
    group_ids: np.ndarray,
    *,
    n_splits: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    return next(
        _iter_group_folds(
            y=y,
            group_ids=group_ids,
            n_splits=n_splits,
            random_state=random_state,
        )
    )


def _slice_list(values: list[str], indices: np.ndarray) -> list[str]:
    return [values[int(i)] for i in indices]


def _ensure_no_group_overlap(
    left_group_ids: np.ndarray,
    right_group_ids: np.ndarray,
    *,
    split_name: str,
) -> None:
    overlap = set(left_group_ids.tolist()) & set(right_group_ids.tolist())
    if overlap:
        raise ValueError(
            f"Grouped split '{split_name}' leaked patient IDs across partitions: "
            f"{sorted(overlap)[:10]}"
        )


def grouped_train_val_test_split(
    X: list[str],
    y: np.ndarray,
    demographics_tensor: torch.Tensor,
    group_ids: np.ndarray,
    *,
    seed: int,
) -> tuple[
    list[str],
    list[str],
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    train_idx, temp_idx = _first_group_split(
        y=y,
        group_ids=group_ids,
        n_splits=5,
        random_state=seed,
    )

    y_temp = y[temp_idx]
    group_ids_temp = group_ids[temp_idx]
    val_relative_idx, test_relative_idx = _first_group_split(
        y=y_temp,
        group_ids=group_ids_temp,
        n_splits=2,
        random_state=seed + 1,
    )

    val_idx = temp_idx[val_relative_idx]
    test_idx = temp_idx[test_relative_idx]

    _ensure_no_group_overlap(group_ids[train_idx], group_ids[val_idx], split_name="train/val")
    _ensure_no_group_overlap(group_ids[train_idx], group_ids[test_idx], split_name="train/test")
    _ensure_no_group_overlap(group_ids[val_idx], group_ids[test_idx], split_name="val/test")

    X_train = _slice_list(X, train_idx)
    X_val = _slice_list(X, val_idx)
    X_test = _slice_list(X, test_idx)

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    demo_tensor_train = demographics_tensor[train_idx]
    demo_tensor_val = demographics_tensor[val_idx]
    demo_tensor_test = demographics_tensor[test_idx]

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        demo_tensor_train,
        demo_tensor_val,
        demo_tensor_test,
    )


def grouped_cv_with_holdout_indices(
    y: np.ndarray,
    group_ids: np.ndarray,
    *,
    k_folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    cv_idx, test_idx = _first_group_split(
        y=y,
        group_ids=group_ids,
        n_splits=5,
        random_state=seed,
    )

    y_cv = y[cv_idx]
    group_ids_cv = group_ids[cv_idx]

    fold_indices: list[tuple[np.ndarray, np.ndarray]] = []
    for train_relative_idx, val_relative_idx in _iter_group_folds(
        y=y_cv,
        group_ids=group_ids_cv,
        n_splits=k_folds,
        random_state=seed,
    ):
        train_idx = cv_idx[train_relative_idx]
        val_idx = cv_idx[val_relative_idx]
        _ensure_no_group_overlap(
            group_ids[train_idx],
            group_ids[val_idx],
            split_name="cv train/val",
        )
        _ensure_no_group_overlap(
            group_ids[test_idx],
            group_ids[val_idx],
            split_name="cv val/test",
        )
        _ensure_no_group_overlap(
            group_ids[test_idx],
            group_ids[train_idx],
            split_name="cv train/test",
        )
        fold_indices.append((train_idx, val_idx))

    return cv_idx, test_idx, fold_indices
