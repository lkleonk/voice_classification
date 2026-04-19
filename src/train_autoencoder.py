import argparse
import pprint
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import optuna
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

import src.utils.data_preprocessing.demographic_metadata_loader as prep
import src.utils.reporting.pdf_report as pdf_report
import src.utils.toolkit.cuda_handling as cuda_handling
import src.utils.toolkit.time_utils as time_utils
import src.utils.visualization.graph_visualization as vis
from src.model.self_supervised_autoencoder.autoencoder import Autoencoder
from src.schemas.dataclasses import SavedModelInfo
from src.settings import CONFIG, DATALOADER, FILES
from src.utils.data_preprocessing.dataset_split_manager import (
    load_dataset_manager_single_split,
)
from src.utils.data_preprocessing.spectrogram_file_dataset import (
    SpectrogramFileDataset,
)
from src.utils.reporting.logger_setup import setup_run_logger
from src.utils.toolkit.naming import build_training_run_name
from src.utils.training.setup import (
    count_and_log_model_params,
    get_batch_size,
)
from src.utils.training.loop import DecayLR
from src.utils.training.model_manager import save_model_w_yaml


device = cuda_handling.set_cuda_to_gpu_nr()
def filter_config_for_autoencoder_run(config: Dict[str, Any]) -> Dict[str, Any]:
    prefixes = (*CONFIG.SHARED_FLAT_KEY_PREFIXES, "autoencoder.")
    return {key: value for key, value in config.items() if key.startswith(prefixes)}


def _load_control_dataset(args: argparse.Namespace, config: Dict[str, Any], logger) -> SpectrogramFileDataset:
    features_norm_mode = config["features.norm"]
    demo_data_mode = config["features.demographic_data_mode"]
    add_acoustic_features = config["features.add_acoustic_feature"]

    (
        control_paths,
        control_labels,
        _control_label_encoder,
        _control_demo_dict,
        control_demo_tensor,
        _control_group_ids,
    ) = prep.load_data_from_metadata_csv(
        args.metadata_file,
        args.recording_category,
        "control",
        features_norm_mode,
        logger,
        demo_data_mode,
        add_acoustic_features,
    )

    return SpectrogramFileDataset(
        control_paths,
        control_labels,
        control_demo_tensor,
        config=config,
        return_only_audio_bool=False,
        return_dummy_audio_bool=False,
    )


def _evaluate_autoencoder_loss(model, dataloader, loss_func) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for xb, _yb, _demo_b in dataloader:
            xb = xb.to(device)
            preds = model(xb)
            batch_loss = loss_func(preds, xb)
            total_loss += batch_loss.item() * xb.size(0)
            total_samples += xb.size(0)

    return total_loss / total_samples


def run_training(
    args: argparse.Namespace,
    config: Optional[Dict[str, Any]] = None,
    trial=None,
    logger=None,
    create_training_report: bool = False,
    last_training_after_optuna_tuning: bool = False,
):
    del create_training_report
    del last_training_after_optuna_tuning

    args.selected_classes = "copd, control"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if logger is None:
        logger = setup_run_logger(__name__, str(output_dir / FILES.LOGGER_FILENAME))

    logger.info("=== AUTOENCODER UNSUPERVISED TRAINING STARTED ===")
    logger.info(
        f"Environment | cwd={Path.cwd()} | python={sys.executable} | "
        "unsupervised training cohort=COPD | downstream evaluation includes control samples"
    )


    if config is None:
        with open(args.config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info("Loaded config from %s", args.config_file)
    else:
        config = dict(config)


    assert config is not None, "Config must be available before filtering for the autoencoder run."
    config = filter_config_for_autoencoder_run(config)

    if not trial:
        logger.info("Arguments:\n%s", pprint.pformat(vars(args)))
    logger.info("YML configuration for autoencoder run:\n%s", pprint.pformat(config, indent=2))

    batch_size = get_batch_size(config, "autoencoder")
    beta1 = float(config["autoencoder.beta1"])
    beta2 = float(config["autoencoder.beta2"])
    lr = float(config["autoencoder.lr"])
    lr_decay_mode = bool(config["autoencoder.lr_decay"])
    lr_decay_type = config["autoencoder.lr_decay_type"]
    eta_min = float(config["scheduler.eta_min"])
    weight_decay = float(config["optimizer.weight_decay"])

    dataset_manager = load_dataset_manager_single_split(args, config, logger)
    train_dataset = dataset_manager.train_dataset
    val_dataset = dataset_manager.val_dataset
    test_dataset = dataset_manager.test_dataset
    control_dataset = _load_control_dataset(args, config, logger)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=DATALOADER.NUM_WORKERS,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=DATALOADER.NUM_WORKERS,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=DATALOADER.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=DATALOADER.NUM_WORKERS,
    )

    img_height = int(dataset_manager.img_height)
    img_width = int(dataset_manager.img_width)

    logger.info("Before creating model - num_classes: %s", len(dataset_manager.label_encoder.classes_))
    logger.info("INPUT dim (height, width): %s", (img_height, img_width))
    logger.info(
        "demographic (plus maybe acoustic) data shape: %s",
        dataset_manager.demographic_data_tensor_length,
    )

    model = Autoencoder(img_width=img_width, img_height=img_height, activation_function="relu").to(device)
    logger.info("Printing model:\n%s", model)
    count_and_log_model_params(model, logger)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
    )
    mae_loss_func = nn.L1Loss(reduction="mean")

    lr_scheduler = None
    if lr_decay_mode:
        if lr_decay_type == "custom_lr":
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=DecayLR(args.epochs, 0, 0).step,
            )
        elif lr_decay_type == "cosine_annealing_w_lr_decay":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs,
                eta_min=eta_min,
            )
        else:
            raise ValueError(f"Unknown lr_decay_type: {lr_decay_type}")

    ae_train_losses = []
    ae_val_losses = []

    for epoch in range(args.epochs):
        logger.info("epoch nr. %s", epoch + 1)
        model.train()

        total_train_loss = 0.0
        total_train_samples = 0

        for xb, _yb, _demo_b in train_dataloader:
            xb = xb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            train_loss = mae_loss_func(preds, xb)
            train_loss.backward()
            optimizer.step()

            total_train_loss += train_loss.item() * xb.size(0)
            total_train_samples += xb.size(0)

        avg_train_loss = total_train_loss / total_train_samples
        ae_train_losses.append(avg_train_loss)

        avg_val_loss = _evaluate_autoencoder_loss(model, val_dataloader, mae_loss_func)
        ae_val_losses.append(avg_val_loss)

        if lr_scheduler is not None:
            lr_scheduler.step()

        logger.info(
            "Epoch %s: Train Loss: %.4f; Val Loss: %.4f;",
            epoch + 1,
            avg_train_loss,
            avg_val_loss,
        )

        if trial is not None:
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                logger.info("-- TRIAL WILL BE PRUNED --")
                raise optuna.TrialPruned()

    avg_test_loss = _evaluate_autoencoder_loss(model, test_dataloader, mae_loss_func)
    report_info = f"TEST DATA loss for Autoencoder: {avg_test_loss}"
    logger.info("AUTOENCODER\n%s", report_info)

    base_run_name = build_training_run_name(args, pretrained=False)
    study_id = getattr(args, "study_id", None)
    training_run_name = f"{study_id}_{base_run_name}" if study_id else base_run_name

    model_info = SavedModelInfo(
        configuration=config,
        arguments=vars(args),
        test_report_info=report_info,
        final_val_loss=float(ae_val_losses[-1]),
        final_test_loss=float(avg_test_loss),
    )
    save_model_w_yaml(
        model,
        args,
        training_run_name,
        str(output_dir),
        logger=logger,
        model_info=model_info,
    )

    loss_curve_pil_img = vis.visualize_loss_curve(
        f"{training_run_name}_train_val_losses",
        ae_train_losses,
        ae_val_losses,
        str(output_dir),
        return_plot=True,
    )
    logger.info("Loss curve visualization saved in %s", args.output_dir)

    mae_distr_pil_img, mae_stats, input_img, output_img, diff_img = vis.get_rec_error_distr(
        model,
        train_dataset,
        val_dataset,
        test_dataset,
        control_dataset,
        loss=mae_loss_func,
        additional_info=" With MAE loss",
        return_sample_imgs=True,
    )
    mse_distr_pil_img, mse_stats, _mse_input_img, _mse_output_img, _mse_diff_img = vis.get_rec_error_distr(
        model,
        train_dataset,
        val_dataset,
        test_dataset,
        control_dataset,
        loss=nn.MSELoss(),
        additional_info=" With MSE loss",
    )
    bench_stats = f"\n{mae_stats}\n\n\n{mse_stats}"
    logger.info(bench_stats)

    report_doc = pdf_report.TuningReportPDF(title=f"Training run({training_run_name.replace('_', ', ')})")
    report_doc.add_config_dict(config)
    report_doc.add_final_training_pil_imgs(
        [loss_curve_pil_img, mae_distr_pil_img, mse_distr_pil_img, input_img, output_img, diff_img],
        new_title_for_images="Training run info",
    )
    report_doc.add_classification_report_str(f"{report_info}\n\n{bench_stats}")
    report_doc.save_full_pdf(
        f"{args.output_dir}/PDF_TRAINING_{training_run_name}_{time_utils.get_year_month_day_hour_minute()}.pdf"
    )
    logger.info("PDF report complete - saved in %s", args.output_dir)

    return float(ae_val_losses[-1]), float(avg_test_loss)
