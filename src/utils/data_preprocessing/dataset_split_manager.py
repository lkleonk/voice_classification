from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset

import src.settings as settings
import src.utils.data_preprocessing.demographic_metadata_loader as prep
from src.settings import MODELS
from src.utils.data_preprocessing.patient_aware_split import (
    grouped_cv_with_holdout_indices,
    grouped_train_val_test_split,
)
from src.utils.data_preprocessing.spectrogram_file_dataset import (
    SpectrogramFileDataset,
)


@dataclass
class DatasetManager:
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    # Data splits (inputs + targets)
    X_train_paths: List[str] #torch.Tensor
    X_val_paths: List[str] #torch.Tensor
    X_test_paths: List[str] #torch.Tensor
    y_train: torch.Tensor
    y_val: torch.Tensor
    y_test: torch.Tensor
    img_height: Union[int, float]
    img_width: Union[int, float] 
    label_encoder: Any
    demographic_data_tensor_length: Union[int, float]
    demographic_data_dict: Dict[str, Any]
    demographic_data_tensor: torch.Tensor
    paths_distribution_info: str


def deduct_spec_shape_from_config(config, logger=None):
    """
    Deduce the spectrogram shape (height, width) from a config dictionary.
    """
    spec_height = config["spectrogram.n_mels"]

    duration_sec = config["spectrogram.duration"]
    sample_rate = config["spectrogram.sample_rate"]
    hop_length = config["spectrogram.hop_length"]
    n_fft = config["spectrogram.n_fft"]

    n_samples = int(duration_sec * sample_rate)
    spec_width = 1 + int(np.ceil((n_samples - n_fft) / hop_length))

    if config["spectrogram.add_chromagram"]:
        spec_height += 12

    if config["spectrogram.add_mfcc"]:
        spec_height += 13

    if config["spectrogram.add_delta_mfcc"]:
        spec_height += 13

    return spec_height, spec_width




def stratified_split(X, y, demographics_tensor, seed):
    # First get the indices split
    indices = np.arange(len(X))
    indices_train, indices_temp, y_train, y_temp = train_test_split(
        indices, y,
        test_size=0.2,
        stratify=y,
        random_state=seed
    )

    indices_val, indices_test, y_val, y_test = train_test_split(
        indices_temp, y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=seed + 1
    )

    # Now use the indices to split all data
    X_train = [X[i] for i in indices_train]
    X_val = [X[i] for i in indices_val]
    X_test = [X[i] for i in indices_test]
    
    # Split demographic tensor
    demo_tensor_train = demographics_tensor[indices_train]
    demo_tensor_val = demographics_tensor[indices_val]
    demo_tensor_test = demographics_tensor[indices_test]

    return X_train, X_val, X_test, y_train, y_val, y_test, demo_tensor_train, demo_tensor_val, demo_tensor_test


def stratified_split_no_test(X, y, demographics_tensor, seed): # currently not used
    # Get indices split
    indices = np.arange(len(X))
    indices_train, indices_temp, y_train, y_temp = train_test_split(
        indices, y,
        test_size=0.2,
        stratify=y,
        random_state=seed
    )

    # validation and test data use the same indices
    X_train = [X[i] for i in indices_train]
    X_val = [X[i] for i in indices_temp]
    X_test = [X[i] for i in indices_temp]
    
    # Split demographic tensor
    demo_tensor_train = demographics_tensor[indices_train]
    demo_tensor_val = demographics_tensor[indices_temp]
    demo_tensor_test = demographics_tensor[indices_temp]

    return X_train, X_val, X_test, y_train, y_temp, y_temp, demo_tensor_train, demo_tensor_val, demo_tensor_test









def get_stratified_split_info(y_train, y_val, y_test, classes=None): # i do this outside
    """Log the distribution of classes for train, val, test sets."""
    if classes is None:
        classes = np.unique(np.concatenate([y_train, y_val, y_test]))

    def class_distribution(y):
        return ", ".join(f"class {c}: {np.mean(y == c):.1%}" for c in classes)

    msg = (
        f"Stratified split completed:\n"
        f"- Train: {len(y_train)} samples ({class_distribution(y_train)})\n"
        f"- Val: {len(y_val)} samples ({class_distribution(y_val)})\n"
        f"- Test: {len(y_test)} samples ({class_distribution(y_test)})"
    )
    return msg








def validate_dataloader(train_dataset, logger, args, config, expected_height, expected_width):    

    expected_shape = (expected_height, expected_width)

    # Run validations on a batch
    first_sample = train_dataset[0]
    img = first_sample[0]  # Assuming (inputs, labels)
    validate_spectrogram_range(img, config, logger)

    
    actual_height, actual_width = validate_spectrogram_shape(img, args, expected_shape, logger)

    return actual_height, actual_width






def validate_spectrogram_shape(inputs, args, expected_shape, logger): # consider minmax range
    actual_shape = inputs.shape[-2:]  # (freq_bins, time_steps)
    logger.info(f"Spectrogram shape - Expected: {expected_shape}, Actual: {actual_shape}")
    tolerance = settings.DATA_VALIDATION.SPECTROGRAM_WIDTH_TOLERANCE_PX
    expected_height, expected_width = expected_shape
    

    if args.model_type not in settings.MODELS.OWN_SPECTROGRAM_INPUT_MODELS_LIST:
        actual_height, actual_width = actual_shape
        assert actual_height == expected_height, \
            f"Spectrogram height should be {expected_height}, but got {actual_height}"

        assert abs(actual_width - expected_width) <= tolerance, \
            f"Spectrogram width should be around {expected_width} Â± {tolerance}, but got {actual_width}"

        assert actual_height <= actual_width, \
            f"Spectrogram height should be smaller {actual_height} than width {actual_width}"
        
    else:
        actual_height, actual_width = 0, 0
        logger.info(f"Model (actual model: {args.model_type}) input has no height/width - raw audio")
    return actual_height, actual_width
    





def validate_spectrogram_range(inputs, config, logger):
    min_val, max_val = inputs.min().item(), inputs.max().item()
    logger.info(f"Spectrogram value range - Min: {min_val:.4f}, Max: {max_val:.4f}")
    if config["spectrogram.norm"] == 'minmax_0_1':
        assert 0.0 <= min_val and max_val <= 1.0, \
            f"Spectrogram values must be in the [0, 1] range. Min: {min_val:.4f}, Max: {max_val:.4f}"
    elif config["spectrogram.norm"] == 'minmax_-1_1':
        assert -1.0 <= min_val and max_val <= 1.0, \
            f"Spectrogram values must be in the [-1, 1] range. Min: {min_val:.4f}, Max: {max_val:.4f}"




def _prepare_dataset_setup(args, config, logger):
    """
    Shared setup for both single-split and k-fold:
    - loads paths/labels/demographics
    - logs stats
    - computes expected spec shape
    - builds augmentation transform_2D
    - determines return_only_audio_bool
    """

    add_acoustic_features = config["features.add_acoustic_feature"]
    features_norm_mode = config["features.norm"]
    demo_data_mode = config["features.demographic_data_mode"]
    freq_mask_ratio = config["spectrogram.freq_mask"]
    time_mask_ratio = config["spectrogram.time_mask"]

    if logger:
        logger.info(
            "load_data_from_metadata_csv called",
            extra={
                "metadata_file": args.metadata_file,
                "recording_category": args.recording_category,
                "selected_classes": args.selected_classes,
                "spectrogram_norm_mode": config["spectrogram.norm"],
                "features_norm_mode": features_norm_mode,
                "demo_data_mode": demo_data_mode,
                "add_acoustic_features": add_acoustic_features,
            },
        )


    X_paths, y, label_encoder, demographic_data_dict, demographic_data_tensor, group_ids = prep.load_data_from_metadata_csv(
        args.metadata_file,
        args.recording_category,
        args.selected_classes,
        features_norm_mode,
        logger,
        demo_data_mode,
        add_acoustic_features,
    )

    prep.sanity_check_data(X_paths, y, label_encoder, demographic_data_dict, demographic_data_tensor, logger)

    logger.info(f"Number of X_paths: {len(X_paths)}")
    logger.info(f"Number of y: {len(y)}")
    assert len(X_paths) == len(y), f"Mismatch: X_paths={len(X_paths)} vs y={len(y)}"
    assert len(X_paths) == len(demographic_data_tensor), (
        f"Mismatch: X_paths={len(X_paths)} vs demographic_data_tensor={len(demographic_data_tensor)}"
    )

    model_type_key = args.model_type.lower()
    return_only_audio_bool = model_type_key in MODELS.OWN_SPECTROGRAM_INPUT_MODELS_LIST
    return_dummy_audio_bool = model_type_key in MODELS.DEMO_ONLY_NO_AUDIO_MODELS_LIST

    if not return_dummy_audio_bool:
        sample_avg_dur, sample_median_dur, sample_std_dur = prep.get_stats_on_sample_duration(X_paths)
        logger.info(
            f"\nSample duration statistics({len(X_paths)} samples):\n"
            f"  Average Duration: {sample_avg_dur:.2f} seconds\n"
            f"  Median Duration: {sample_median_dur:.2f} seconds\n"
            f"  Standard Deviation: {sample_std_dur:.2f} seconds"
        )
    else:
        logger.info("Skipping audio duration stats because model '%s' uses demographic data only.", args.model_type)

    # expected image shape
    if return_dummy_audio_bool or return_only_audio_bool:
        expected_img_height, expected_img_width = 0, 0
    else:
        expected_img_height, expected_img_width = deduct_spec_shape_from_config(config, logger)

    logger.info(
        f"Projected image shape(Mel-spectrogram + more): height: {expected_img_height}; width: {expected_img_width}"
    )

    # masking lengths (guard against zeros)
    freq_mask_len = int(freq_mask_ratio * expected_img_height) if expected_img_height else 0
    time_mask_len = int(time_mask_ratio * expected_img_width) if expected_img_width else 0

    # If height/width are 0 (raw waveform models), keep transform as identity-ish
    transform_2D = nn.Sequential()
    if freq_mask_len > 0:
        transform_2D.append(T.FrequencyMasking(freq_mask_param=freq_mask_len))
    if time_mask_len > 0:
        transform_2D.append(T.TimeMasking(time_mask_param=time_mask_len))

    return (
        X_paths,
        y,
        label_encoder,
        demographic_data_dict,
        demographic_data_tensor,
        group_ids,
        expected_img_height,
        expected_img_width,
        transform_2D,
        return_only_audio_bool,
        return_dummy_audio_bool,
    )


def load_dataset_manager_single_split(args, config, logger) -> DatasetManager:
    """
    Returns ONE DatasetManager instance using your stratified_split().
    """

    (
        X_paths,
        y,
        label_encoder,
        demographic_data_dict,
        demographic_data_tensor,
        group_ids,
        expected_img_height,
        expected_img_width,
        transform_2D,
        return_only_audio_bool,
        return_dummy_audio_bool,
    ) = _prepare_dataset_setup(args, config, logger)

    if group_ids is None:
        X_train, X_val, X_test, y_train, y_val, y_test, demo_train, demo_val, demo_test = stratified_split(
            X_paths, y, demographic_data_tensor, seed=args.random_seed
        )
    else:
        X_train, X_val, X_test, y_train, y_val, y_test, demo_train, demo_val, demo_test = grouped_train_val_test_split(
            X_paths,
            np.asarray(y),
            demographic_data_tensor,
            np.asarray(group_ids),
            seed=args.random_seed,
        )
        logger.info(
            "Using grouped patient-aware split for UK data | unique_patients=%s",
            len(set(np.asarray(group_ids).tolist())),
        )
    logger.info("80:20:20 train test split")
    logger.info(get_stratified_split_info(y_train, y_val, y_test))

    partial_data_msg = (
        f"\n Training data paths (first 20):\n{X_train[:20]}"
        f"\n Validation data paths (first 20):\n{X_val[:20]}"
        f"\n Test data paths (first 20):\n{X_test[:20]}"
    )
    logger.info(partial_data_msg)

    train_dataset = SpectrogramFileDataset(
        X_train, y_train, demo_train,
        config=config,
        transform_1D=None,
        transform_2D=transform_2D,
        return_only_audio_bool=return_only_audio_bool,
        return_dummy_audio_bool=return_dummy_audio_bool,
    )
    val_dataset = SpectrogramFileDataset(
        X_val, y_val, demo_val,
        config=config,
        return_only_audio_bool=return_only_audio_bool,
        return_dummy_audio_bool=return_dummy_audio_bool,
    )
    test_dataset = SpectrogramFileDataset(
        X_test, y_test, demo_test,
        config=config,
        return_only_audio_bool=return_only_audio_bool,
        return_dummy_audio_bool=return_dummy_audio_bool,
    )

    if return_dummy_audio_bool:
        img_height, img_width = 0, 0
        logger.info("Skipping spectrogram validation because model '%s' consumes no audio input.", args.model_type)
    else:
        img_height, img_width = validate_dataloader(
            train_dataset,
            logger,
            args,
            config,
            expected_height=expected_img_height,
            expected_width=expected_img_width,
        )

    data_msg = f"\n Training data paths:\n{X_train}\n Validation data paths:\n{X_val}\n Test data paths:\n{X_test}"

    return DatasetManager(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        X_train_paths=X_train,
        X_val_paths=X_val,
        X_test_paths=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        img_height=img_height,
        img_width=img_width,
        label_encoder=label_encoder,
        demographic_data_tensor_length=demographic_data_tensor.shape[1],
        demographic_data_dict=demographic_data_dict,
        demographic_data_tensor=demographic_data_tensor,
        paths_distribution_info=data_msg,
    )


def load_dataset_managers_kfold(args, config, logger, k_folds: int) -> List[DatasetManager]:
    """
    Returns MULTIPLE DatasetManager instances (one per fold),
    with a fixed hold-out test set shared across folds.
    """

    if not k_folds or k_folds <= 1:
        raise ValueError(f"k_folds must be > 1 for CV, got: {k_folds}")

    (
        X_paths,
        y,
        label_encoder,
        demographic_data_dict,
        demographic_data_tensor,
        group_ids,
        expected_img_height,
        expected_img_width,
        transform_2D,
        return_only_audio_bool,
        return_dummy_audio_bool,
    ) = _prepare_dataset_setup(args, config, logger)

    logger.info(f"--- STARTING {k_folds}-FOLD CROSS VALIDATION ---")

    X_paths_np = np.array(X_paths)
    y_np = np.array(y)

    if group_ids is None:
        # Hold-out test set (never seen by folds)
        X_cv, X_test, y_cv, y_test, idx_cv, idx_test = train_test_split(
            X_paths_np,
            y_np,
            np.arange(len(X_paths_np)),
            test_size=0.2,
            stratify=y_np,
            random_state=args.random_seed,
        )
        fold_indices = None
    else:
        group_ids_np = np.asarray(group_ids)
        idx_cv, idx_test, fold_indices = grouped_cv_with_holdout_indices(
            y=y_np,
            group_ids=group_ids_np,
            k_folds=k_folds,
            seed=args.random_seed,
        )
        X_cv, X_test = X_paths_np[idx_cv], X_paths_np[idx_test]
        y_cv, y_test = y_np[idx_cv], y_np[idx_test]
        logger.info(
            "Using grouped patient-aware CV split for UK data | unique_patients=%s",
            len(set(group_ids_np.tolist())),
        )

    demo_tensor_test = demographic_data_tensor[idx_test]
    demo_tensor_cv = demographic_data_tensor[idx_cv]

    test_dataset = SpectrogramFileDataset(
        X_test, y_test, demo_tensor_test,
        config=config,
        return_only_audio_bool=return_only_audio_bool,
        return_dummy_audio_bool=return_dummy_audio_bool,
    )

    dataset_managers: List[DatasetManager] = []

    skf = None
    if fold_indices is None:
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=args.random_seed)

    img_height, img_width = 0, 0  # validated once on fold 0

    fold_iterator = fold_indices
    if fold_iterator is None:
        fold_iterator = skf.split(X_cv, y_cv)

    for fold_idx, (train_indices, val_indices) in enumerate(fold_iterator):
        logger.info(f"Preparing Fold {fold_idx + 1}/{k_folds}")

        if group_ids is None:
            X_train_fold, y_train_fold = X_cv[train_indices], y_cv[train_indices]
            X_val_fold, y_val_fold = X_cv[val_indices], y_cv[val_indices]
            demo_train_fold = demo_tensor_cv[train_indices]
            demo_val_fold = demo_tensor_cv[val_indices]
        else:
            X_train_fold, y_train_fold = X_paths_np[train_indices], y_np[train_indices]
            X_val_fold, y_val_fold = X_paths_np[val_indices], y_np[val_indices]
            demo_train_fold = demographic_data_tensor[train_indices]
            demo_val_fold = demographic_data_tensor[val_indices]

            if set(group_ids_np[train_indices].tolist()) & set(group_ids_np[val_indices].tolist()):
                raise ValueError(f"Fold {fold_idx + 1} leaked patient IDs between train and val")
            if set(group_ids_np[train_indices].tolist()) & set(group_ids_np[idx_test].tolist()):
                raise ValueError(f"Fold {fold_idx + 1} leaked patient IDs between train and test")
            if set(group_ids_np[val_indices].tolist()) & set(group_ids_np[idx_test].tolist()):
                raise ValueError(f"Fold {fold_idx + 1} leaked patient IDs between val and test")

        train_dataset = SpectrogramFileDataset(
            X_train_fold, y_train_fold, demo_train_fold,
            config=config,
            transform_2D=transform_2D,
            return_only_audio_bool=return_only_audio_bool, # return only audio bool should be gone
            return_dummy_audio_bool=return_dummy_audio_bool,
        )
        val_dataset = SpectrogramFileDataset(
            X_val_fold, y_val_fold, demo_val_fold,
            config=config,
            return_only_audio_bool=return_only_audio_bool, # return only audio bool should be gone
            return_dummy_audio_bool=return_dummy_audio_bool,
        )

        if fold_idx == 0 and not return_dummy_audio_bool:
            img_height, img_width = validate_dataloader(
                train_dataset,
                logger,
                args,
                config,
                expected_height=expected_img_height,
                expected_width=expected_img_width,
            )
        elif fold_idx == 0:
            logger.info("Skipping spectrogram validation because model '%s' consumes no audio input.", args.model_type)

        dataset_managers.append(
            DatasetManager(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                X_train_paths=X_train_fold,
                X_val_paths=X_val_fold,
                X_test_paths=X_test,
                y_train=y_train_fold,
                y_val=y_val_fold,
                y_test=y_test,
                img_height=img_height,
                img_width=img_width,
                label_encoder=label_encoder,
                demographic_data_tensor_length=demographic_data_tensor.shape[1],
                demographic_data_dict=demographic_data_dict,
                demographic_data_tensor=demographic_data_tensor,
                paths_distribution_info=f"Fold {fold_idx + 1}",
            )
        )

    return dataset_managers

