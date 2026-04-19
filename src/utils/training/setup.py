from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import src.model.model_factory as models
import src.utils.training.model_manager as model_manager
from src.schemas.dataclasses import TrainingSetup
from src.utils.data_preprocessing.dataset_split_manager import (
    load_dataset_manager_single_split,
)


def count_and_log_model_params(model, logger):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total Parameters: {total_params:,}")
    logger.info(f"Trainable Parameters: {trainable_params:,}")


def get_weighted_criterion_loss(
    y_train,
    y_val,
    y_test,
    label_encoder,
    logger=None,
    device: Optional[torch.device] = None,
):
    class_sample_counts = np.bincount(y_train)
    assert len(np.unique(y_train)) == len(np.unique(y_val)) == len(np.unique(y_test)) == len(
        label_encoder.classes_
    ), "Class count mismatch between splits"
    class_weights = 1.0 / (class_sample_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(class_sample_counts)

    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = torch.tensor(class_weights, dtype=torch.float32).to(target_device)
    new_criterion_w_weights = nn.CrossEntropyLoss(weight=weights, reduction="mean")

    if logger:
        logger.info(
            f"Weighted loss activated (as set in hyperparameters)- Class weights (auto): {class_weights}"
        )

    return new_criterion_w_weights


def get_batch_size(config: dict, model_type: str) -> int:
    return int(config[f"{model_type}.batch_size"])


def validate_config(args: Any, config: dict, model_type: str, dataset_manager: Any) -> None:
    mt = (model_type or "").lower()

    if mt == "vit":
        if args.add_demo_data:
            early = config.get("vit.early_fusion_add_extra_data", False)
            late = config.get("vit.late_fusion_add_extra_data", False)
            if not early and not late:
                raise ValueError(
                    "When add-demographic-data is activated, at least one of "
                    "vit.early_fusion_add_extra_data or vit.late_fusion_add_extra_data must be True."
                )

    if mt == "vggish":
        if getattr(dataset_manager, "img_height", None) != 64:
            raise ValueError(f"Image height not correct for {model_type} model. Expected: exactly 64.")


def build_training_setup(
    *,
    args: Any,
    config: dict,
    logger: Any,
    device: torch.device,
    dataset_manager: Any,
    num_workers: int = 2,
) -> TrainingSetup:
    """
    Build the full training setup (data, loaders, model, losses, optimizer) and return as a dataclass.
    """
    batch_size = get_batch_size(config, args.model_type)
    beta1 = float(config[f"{args.model_type}.beta1"])
    beta2 = float(config[f"{args.model_type}.beta2"])
    lr = float(config[f"{args.model_type}.lr"])
    weight_decay = float(config["optimizer.weight_decay"])
    weighted_loss_mode = config["loss.weighted_loss"]
    print(f"{args.metadata_file}")

    # 1) Dataset manager
    if dataset_manager is None:
        dataset_manager = load_dataset_manager_single_split(args, config, logger)

    y_train = dataset_manager.y_train
    y_val = dataset_manager.y_val
    y_test = dataset_manager.y_test
    label_encoder = dataset_manager.label_encoder
    demographic_len = dataset_manager.demographic_data_tensor_length

    # Sanity check: all splits must contain all classes.
    assert (
        len(np.unique(y_train))
        == len(np.unique(y_val))
        == len(np.unique(y_test))
        == len(label_encoder.classes_)
    ), "Class count mismatch between splits"

    # 2) Optional weighted criterion
    criterion_w_weights: Optional[nn.Module] = None
    if weighted_loss_mode:
        criterion_w_weights = get_weighted_criterion_loss(
            y_train, y_val, y_test, label_encoder, logger, device=device
        )

    # 3) Dataloaders
    train_dataloader = DataLoader(
        dataset_manager.train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_dataloader = DataLoader(
        dataset_manager.val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # 4) Config validation
    validate_config(args, config, args.model_type, dataset_manager)

    # 5) Model
    img_height = int(dataset_manager.img_height)
    img_width = int(dataset_manager.img_width)
    num_classes = len(label_encoder.classes_)

    logger.info(f"Before creating model - num_classes: {num_classes}")
    logger.info(f"INPUT dim (height, width): {(img_height, img_width)}")
    logger.info(f"demographic (plus maybe acoustic) data shape: {demographic_len}")

    model = models.build_model(
        model_type=args.model_type,
        num_classes=num_classes,
        img_width=img_width,
        img_height=img_height,
        config=config,
        demographic_data_tensor_length=int(demographic_len),
        add_demographic_data=args.add_demo_data,
    ).to(device)

    logger.info(f"Printing model:\n{model}")
    count_and_log_model_params(model, logger)

    # 6) Model manager + losses + optimizer
    criterion = nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
    )
    model_m = model_manager.ModelManager()

    return TrainingSetup(
        dataset_manager=dataset_manager,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        criterion=criterion,
        criterion_w_weights=criterion_w_weights,
        optimizer=optimizer,
        model_m=model_m,
        num_classes=num_classes,
        img_height=img_height,
        img_width=img_width,
    )


