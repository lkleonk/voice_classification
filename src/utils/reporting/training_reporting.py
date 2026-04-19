from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import mlflow
import numpy as np
import src.utils.data_preprocessing.demographic_metadata_loader as prep
import src.utils.reporting.pdf_report as pdf_report
import src.utils.visualization.gradcam as grd
import src.utils.visualization.graph_visualization as vis
from PIL.Image import Image
from src.schemas.dataclasses import SpecSample
from src.utils.reporting.demo_data_table import (
    get_distr_of_demo_data,
    get_test_samples_table,
    save_tabular_data,
)
from src.utils.reporting.mlflow_helper import safe_mlflow_call


@dataclass(frozen=True)
class LossCurveLogResult:
    non_weighted_pil: Any
    weighted_pil: Optional[Any] = None
    non_weighted_artifact_path: str = "plots/best_model/non-weighted-loss-curve.png"
    weighted_artifact_path: str = "plots/best_model/weighted-loss-curve.png"
    mlflow_enabled: bool = True


def save_and_log_best_model_loss_artifacts(
    *,
    logger,
    args,
    model_m,
    art,
    mlflow_enabled: bool,
    weighted_loss_mode: bool,
    title_prefix: str = "Best epoch model",
    artifact_dir: str = "plots/best_model",
) -> LossCurveLogResult:
    non_weighted_name = "best_loss_curve_non_weighted"
    non_weighted_title = f"{title_prefix} - Training and Validation Loss - Non-Weighted loss visualization"
    non_weighted_pil = vis.visualize_loss_curve(
        non_weighted_name,
        art.train_losses,
        art.val_losses,
        output_dir=model_m.best_model_output_folder,
        title=non_weighted_title,
        return_plot=True,
    )
    logger.info("For the best model: non-weighted loss just saved successfully")

    weighted_pil = None
    if weighted_loss_mode:
        weighted_name = "best_loss_curve_weighted"
        weighted_title = f"{title_prefix} - Training and Validation Loss - Weighted loss visualization"
        weighted_pil = vis.visualize_loss_curve(
            weighted_name,
            art.weighted_train_losses,
            art.weighted_val_losses,
            output_dir=model_m.best_model_output_folder,
            title=weighted_title,
            return_plot=True,
        )
        logger.info("For the best model: weighted loss just saved successfully")

    non_weighted_artifact_path = f"{artifact_dir}/non-weighted-loss-curve.png"
    mlflow_enabled, _ = safe_mlflow_call(
        mlflow_enabled,
        logger,
        "log best-model non-weighted loss curve",
        mlflow.log_image,
        non_weighted_pil,
        artifact_file=non_weighted_artifact_path,
    )

    weighted_artifact_path = f"{artifact_dir}/weighted-loss-curve.png"
    if weighted_pil is not None:
        mlflow_enabled, _ = safe_mlflow_call(
            mlflow_enabled,
            logger,
            "log best-model weighted loss curve",
            mlflow.log_image,
            weighted_pil,
            artifact_file=weighted_artifact_path,
        )

    logger.info("Best model loss artifacts saved locally.")
    if mlflow_enabled:
        logger.info("Best model loss artifacts logged to MLflow.")

    return LossCurveLogResult(
        non_weighted_pil=non_weighted_pil,
        weighted_pil=weighted_pil,
        non_weighted_artifact_path=non_weighted_artifact_path,
        weighted_artifact_path=weighted_artifact_path,
        mlflow_enabled=mlflow_enabled,
    )


def save_class_spectrogram_samples_as_png(output_dir, dataset) -> List[SpecSample]:
    """
    Saves one spectrogram sample per unique label as PNG images
    and returns them as a list of SpecSample objects.

    Args:
        output_dir (str): Directory to save PNG files.
        dataset: Iterable providing (x, y) samples (not batches).

    Returns:
        List[SpecSample]: List of spectrogram samples with metadata.
    """
    unique_labels = set()
    spec_samples: List[SpecSample] = []

    for idx, (x, y, demo) in enumerate(dataset):
        # Handle scalar tensor labels
        current_label = y.item() if hasattr(y, "item") else y

        # Save only the first occurrence of each label
        if current_label not in unique_labels:
            unique_labels.add(current_label)

            # If dataset outputs raw audio, convert to spectrogram first
            if hasattr(dataset, "return_only_audio") and dataset.return_only_audio:
                from src.utils.data_preprocessing.spectrogram_file_dataset import (
                    create_spectrogram_from_waveform_w_config,
                )
                waveform = x.detach().cpu().numpy()
                spec = create_spectrogram_from_waveform_w_config(waveform, dataset.config)
                spec_np = np.asarray(spec)
            else:
                # Otherwise assume x is already a spectrogram tensor
                spec = x
                spec_np = spec.detach().cpu().numpy()

            # Normalize to 2D for visualization
            if spec_np.ndim == 3 and spec_np.shape[0] == 1:
                spec_np = spec_np[0]
            elif spec_np.ndim != 2:
                continue
            #print(f'before visualization: {spec.shape}')
            pil_img = vis.save_spectrogram_sample_as_png(output_dir, spec_np, current_label)

            spec_samples.append(
                SpecSample(
                    spec=spec_np,
                    label=current_label,
                    index=idx,
                    pil_img=pil_img,
                )
            )

    return spec_samples



def _append_if_not_none(dst: List[Image], img: Optional[Image]) -> None:
    if img is not None:
        dst.append(img)


def _extend_if_nonempty(
    dst: List[Image],
    imgs: Optional[Sequence[Union[Image, None]]],
) -> None:
    if not imgs:
        return
    dst.extend(img for img in imgs if img is not None)


@dataclass(frozen=True)
class TrainingReportArtifacts:
    pil_images: List[Image]
    report_path: str
    gradcam_success: bool
    mlflow_enabled: bool


def finalize_training_reporting(
    *,
    args: Any,
    name_info: str,
    config: Dict[str, Any],
    model: Any,
    art: Any,
    dataset_manager: Any,
    label_encoder: Any,
    report_info: str,
    all_targets,
    all_preds,
    logits_list: Optional[list],
    weighted_loss_mode: bool,
    lr_decay_mode: bool,
    model_type: str,
    model_m: Any,
    logger: Any,
    mlflow_enabled: bool,
) -> TrainingReportArtifacts:
    list_of_pil_imgs: List[Image] = []

    prep.save_loss_as_json(
        args.output_dir,
        f"{name_info}_train_val_losses",
        art.train_losses,
        art.val_losses,
    )
    logger.info("Saved train/val loss JSON to %s", args.output_dir)

    loss_curve = vis.visualize_loss_curve(
        "final_loss_curve_non_weighted",
        art.train_losses,
        art.val_losses,
        output_dir=model_m.final_model_output_folder,
        title="Training and Validation Loss - Non-Weighted",
        return_plot=True,
    )
    _append_if_not_none(list_of_pil_imgs, loss_curve)
    if loss_curve is not None:
        mlflow_enabled, _ = safe_mlflow_call(
            mlflow_enabled,
            logger,
            "log final-model non-weighted loss curve",
            mlflow.log_image,
            loss_curve,
            "plots/final_model/non-weighted-loss-curve.png",
        )

    if weighted_loss_mode:
        weighted_curve = vis.visualize_loss_curve(
            "final_loss_curve_weighted",
            art.weighted_train_losses,
            art.weighted_val_losses,
            output_dir=model_m.final_model_output_folder,
            title="Training and Validation Loss - Weighted",
            return_plot=True,
        )
        _append_if_not_none(list_of_pil_imgs, weighted_curve)
        if weighted_curve is not None:
            mlflow_enabled, _ = safe_mlflow_call(
                mlflow_enabled,
                logger,
                "log final-model weighted loss curve",
                mlflow.log_image,
                weighted_curve,
                "plots/final_model/weighted-loss-curve.png",
            )

    if lr_decay_mode:
        lr_vis = vis.plot_lrs(art.lrs)
        _append_if_not_none(list_of_pil_imgs, lr_vis)

    spec_samples = save_class_spectrogram_samples_as_png(
        args.output_dir, dataset_manager.train_dataset
    )

    demo_distr_imgs = get_distr_of_demo_data(dataset_manager)
    _extend_if_nonempty(list_of_pil_imgs, demo_distr_imgs)

    pil_gradcam_imgs: List[Image] = []
    gradcam_success = False

    try:
        if model_type in grd.GRADCAM_SUITED_MODEL_TYPES:
            for sample in spec_samples:
                gradcam_sample = grd.get_gradcam_img_w_sample(
                    args,
                    config,
                    model,
                    sample,
                    dataset_manager.demographic_data_tensor[sample.index],
                    label_encoder,
                )
                _append_if_not_none(pil_gradcam_imgs, gradcam_sample.pil_gradcam_img_by_label)

        gradcam_success = len(pil_gradcam_imgs) > 0
    except Exception:
        logger.info("GradCAM failed - falling back to raw spectrogram samples")

    all_targets_str = label_encoder.inverse_transform(all_targets)
    all_preds_str = label_encoder.inverse_transform(all_preds)

    table = get_test_samples_table(
        all_targets,
        all_preds,
        all_targets_str,
        all_preds_str,
        dataset_manager.X_test_paths,
        dataset_manager.demographic_data_dict,
        order_by_correctness=True,
    )

    logger.info("\nFor the first 300 datapoints:\n%s", table.head(300))
    save_tabular_data(
        args.output_dir,
        f"{name_info}_TEST_SAMPLE_ANALYSIS",
        table,
        logger,
    )

    boxplot_imgs = vis.get_boxplots_for_demographic_data(table)
    _extend_if_nonempty(list_of_pil_imgs, boxplot_imgs)

    if logits_list:
        conf_vis = vis.plot_confidence_distribution(
            args.output_dir, logits_list, all_targets
        )
        _append_if_not_none(list_of_pil_imgs, conf_vis)

    conf_mat = vis.plot_confusion_matrix_and_return(
        all_targets_str,
        all_preds_str,
        labels=label_encoder.classes_,
        output_path=None,
    )
    _append_if_not_none(list_of_pil_imgs, conf_mat)

    if pil_gradcam_imgs:
        _extend_if_nonempty(list_of_pil_imgs, pil_gradcam_imgs)
        logger.info("GradCAM images added: %d", len(pil_gradcam_imgs))
    else:
        spec_imgs = [s.pil_img for s in spec_samples]
        _extend_if_nonempty(list_of_pil_imgs, spec_imgs)
        logger.info("No GradCAM images; spectrogram samples added: %d", len(spec_imgs))

    report_doc = pdf_report.TuningReportPDF(
        title=f"Training run({name_info.replace('_', ', ')})"
    )
    report_doc.add_final_training_pil_imgs(
        list_of_pil_imgs,
        new_title_for_images="Training run info",
    )
    report_doc.add_config_dict(config)
    report_doc.add_classification_report_str(report_info)

    report_path = f"{args.output_dir}/PDF_TRAIN_{name_info}.pdf"
    report_doc.save_full_pdf(report_path)
    logger.info("PDF report complete - saved in %s", report_path)

    mlflow_enabled, _ = safe_mlflow_call(
        mlflow_enabled,
        logger,
        "log training PDF artifact",
        mlflow.log_artifact,
        report_path,
    )
    if mlflow_enabled:
        logger.info("Training PDF logged to MLflow")
    logger.info("train_spectrogram: Final checkpoint: All operations completed")

    return TrainingReportArtifacts(
        pil_images=list_of_pil_imgs,
        report_path=report_path,
        gradcam_success=gradcam_success,
        mlflow_enabled=mlflow_enabled,
    )

