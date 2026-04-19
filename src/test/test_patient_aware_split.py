import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from src.utils.data_preprocessing import dataset_split_manager
from src.utils.data_preprocessing.patient_aware_split import (
    grouped_cv_with_holdout_indices,
    grouped_train_val_test_split,
)
from src.utils.data_preprocessing.demographic_metadata_loader import (
    load_data_from_metadata_csv,
)


@pytest.mark.unit
def test_grouped_train_val_test_split_keeps_patients_isolated():
    X = [f"sample_{idx}.wav" for idx in range(24)]
    y = np.asarray([0] * 12 + [1] * 12)
    group_ids = np.asarray(
        [f"p{group_idx}" for group_idx in range(6) for _ in range(2)]
        + [f"p{group_idx}" for group_idx in range(6, 12) for _ in range(2)]
    )
    demographics_tensor = torch.arange(24, dtype=torch.float32).unsqueeze(1)

    X_train, X_val, X_test, y_train, y_val, y_test, _, _, _ = grouped_train_val_test_split(
        X,
        y,
        demographics_tensor,
        group_ids,
        seed=7,
    )

    train_groups = {group_ids[X.index(path)] for path in X_train}
    val_groups = {group_ids[X.index(path)] for path in X_val}
    test_groups = {group_ids[X.index(path)] for path in X_test}

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)
    assert len(y_train) + len(y_val) + len(y_test) == len(y)


@pytest.mark.unit
def test_grouped_cv_with_holdout_indices_keeps_patients_isolated():
    y = np.asarray([0] * 20 + [1] * 20)
    group_ids = np.asarray([f"p{group_idx}" for group_idx in range(20) for _ in range(2)])

    cv_idx, test_idx, fold_indices = grouped_cv_with_holdout_indices(
        y=y,
        group_ids=group_ids,
        k_folds=3,
        seed=11,
    )

    test_groups = set(group_ids[test_idx].tolist())

    assert set(cv_idx.tolist()).isdisjoint(set(test_idx.tolist()))

    for train_idx, val_idx in fold_indices:
        train_groups = set(group_ids[train_idx].tolist())
        val_groups = set(group_ids[val_idx].tolist())
        assert train_groups.isdisjoint(val_groups)
        assert train_groups.isdisjoint(test_groups)
        assert val_groups.isdisjoint(test_groups)


@pytest.mark.unit
def test_load_data_from_metadata_csv_returns_group_ids_for_uk():
    metadata_path = Path(__file__).resolve().parent / "test_run_dummy_files" / "test_metadata_uk_v2.csv"

    filepaths, labels, label_encoder, demographic_data_dict, features_tensor, group_ids = load_data_from_metadata_csv(
        str(metadata_path),
        recording_category="a",
        selected_conditions="control,copd",
        features_norm_mode="minmax_0_1",
        logger=None,
        demo_data_mode="only_sex_age",
        add_acoustic_features=False,
    )

    expected_group_ids = pd.read_csv(metadata_path)["audio_id"].to_numpy()

    assert len(filepaths) == len(labels) == len(features_tensor) == len(group_ids)
    assert set(demographic_data_dict.keys()).issubset(set(filepaths))
    assert np.array_equal(group_ids, expected_group_ids)
    assert set(label_encoder.classes_) == {"control", "copd"}


@pytest.mark.unit
def test_load_data_from_metadata_csv_rejects_non_uk_metadata_schema():
    metadata_path = Path(__file__).resolve().parent / "test_run_dummy_files" / "test_metadata_non_uk.csv"

    with pytest.raises(ValueError, match="Missing required UK metadata columns"):
        load_data_from_metadata_csv(
            str(metadata_path),
            recording_category="a",
            selected_conditions="control",
            features_norm_mode="minmax_0_1",
            logger=None,
            demo_data_mode="only_sex_age",
            add_acoustic_features=False,
        )


@pytest.mark.unit
def test_load_dataset_manager_single_split_uses_group_ids_without_patient_overlap(monkeypatch):
    class DummyDataset:
        def __init__(self, X_paths, y_values, demo_tensor, **kwargs):
            self.X_paths = list(X_paths)
            self.y_values = np.asarray(y_values)
            self.demo_tensor = demo_tensor

        def __len__(self):
            return len(self.X_paths)

        def __getitem__(self, index):
            return torch.zeros(1, 4, 4), int(self.y_values[index])

    def fake_prepare_dataset_setup(args, config, logger):
        X_paths = [f"sample_{idx}.wav" for idx in range(24)]
        y = np.asarray([0] * 12 + [1] * 12)
        demographic_data_dict = {path: {"age": 50} for path in X_paths}
        demographic_data_tensor = torch.zeros((24, 2), dtype=torch.float32)
        group_ids = np.asarray(
            [f"p{group_idx}" for group_idx in range(6) for _ in range(2)]
            + [f"p{group_idx}" for group_idx in range(6, 12) for _ in range(2)]
        )
        return (
            X_paths,
            y,
            SimpleNamespace(classes_=np.asarray(["control", "copd"])),
            demographic_data_dict,
            demographic_data_tensor,
            group_ids,
            4,
            4,
            torch.nn.Sequential(),
            False,
            False,
        )

    monkeypatch.setattr(dataset_split_manager, "_prepare_dataset_setup", fake_prepare_dataset_setup)
    monkeypatch.setattr(dataset_split_manager, "SpectrogramFileDataset", DummyDataset)
    monkeypatch.setattr(dataset_split_manager, "validate_dataloader", lambda *args, **kwargs: (4, 4))

    args = SimpleNamespace(random_seed=3, model_type="test_cnn")
    config = {"spectrogram.norm": "minmax_0_1"}
    logger = logging.getLogger(__name__)

    dataset_manager = dataset_split_manager.load_dataset_manager_single_split(args, config, logger)

    path_to_group = {
        f"sample_{idx}.wav": (
            [f"p{group_idx}" for group_idx in range(6) for _ in range(2)]
            + [f"p{group_idx}" for group_idx in range(6, 12) for _ in range(2)]
        )[idx]
        for idx in range(24)
    }

    train_groups = {path_to_group[path] for path in dataset_manager.X_train_paths}
    val_groups = {path_to_group[path] for path in dataset_manager.X_val_paths}
    test_groups = {path_to_group[path] for path in dataset_manager.X_test_paths}

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

