import copy
from typing import Literal, Optional
import os
from src.schemas.dataclasses import SavedModelInfo
from src.settings import MODEL_SAVE
import torch

PERFORMANCE_STATUS = Literal["good_performance", "bad_performance", ""]
MODEL_LABEL = Literal["best_model", "final_model"]


class ModelManager:
    """
    Utility class for tracking, updating, and saving models during training
    based on performance metrics.
    """

    def __init__(self):
        self.best_model = None
        self.best_epoch = None
        self.best_model_val_accuracy: float = -1.0
        self.best_model_test_accuracy: float = -1.0
        self.best_model_performance_str: PERFORMANCE_STATUS = ""
        self.best_model_output_folder = ""
        self.best_model_auroc: float = -1.0

        self.final_model = None
        self.final_epoch = None
        self.final_val_accuracy: float = -1.0
        self.final_test_accuracy: float = -1.0
        self.final_performance_str: PERFORMANCE_STATUS = ""
        self.final_model_output_folder = ""

    def set_model_weights_from_final_epoch(self, model, final_model_val_mba, final_model_test_mba, epoch):
        self.final_model = model
        self.final_val_accuracy = final_model_val_mba
        self.final_test_accuracy = final_model_test_mba
        self.final_epoch = epoch

    def consider_updating_best_model(
        self,
        new_model,
        new_mba_value,
        new_auroc_value,
        epoch,
        logger=None,
        min_epochs: int = 0,
        test_run: bool = False,
    ):
        if not (test_run or (epoch + 1) >= min_epochs):
            if self.best_model is None:
                self.best_model = copy.deepcopy(new_model).cpu()
                if logger:
                    logger.info(
                        "Stored fallback model before min_epochs was reached, so that model is not None - relevant for test runs with less than min_epochs"
                    )
            return

        if new_mba_value >= self.best_model_val_accuracy:
            self.best_model = copy.deepcopy(new_model).cpu()
            self.best_model_val_accuracy = new_mba_value
            self.best_model_auroc = new_auroc_value
            self.best_epoch = epoch
            if logger:
                logger.info("Best model in ModelManager instance updated successfully")

    def should_stop_early(
        self,
        *,
        current_epoch: int,
        patience: int,
        min_epochs: int = 0,
    ) -> bool:
        if patience <= 0:
            return False
        if (current_epoch + 1) < min_epochs:
            return False
        if self.best_epoch is None:
            return False
        return (current_epoch - self.best_epoch) >= patience

    def consider_saving_model(
        self,
        *,
        label: MODEL_LABEL,
        model,
        val_accuracy: float,
        test_accuracy: Optional[float],
        epoch: Optional[int],
        name_info: str,
        args,
        num_classes: int,
        logger=None,
        model_info: Optional[SavedModelInfo] = None,
    ) -> PERFORMANCE_STATUS:
        """
        Save a model if it passes threshold into the run output directory.

        `label` must be one of: "best_model", "final_model".
        """

        # Threshold selection
        threshold_map = MODEL_SAVE.THRESHOLD_MAP_UK
        threshold = threshold_map.get(num_classes, MODEL_SAVE.DEFAULT_THRESHOLD)

        if model is None:
            if logger:
                logger.info("No model provided for saving (%s).", label)
            return ""

        # Gate: val must be above threshold; if test provided, it must be above too
        passes = (val_accuracy > threshold) and (test_accuracy is None or test_accuracy > threshold)

        output_folder = args.output_dir

        # Store for later inspection / debugging
        if label == "best_model":
            self.best_model_output_folder = output_folder
        else:  # label == "final_model"
            self.final_model_output_folder = output_folder

        if passes:
            save_model_w_yaml(
                model,
                args,
                label,
                output_folder,
                logger,
                model_info=model_info,
            )

            if logger:
                epoch_info = f" (epoch {epoch})" if epoch is not None else ""
                logger.info(
                    "%s%s successfully saved - %.4f mean balanced accuracy.",
                    label,
                    epoch_info,
                    val_accuracy,
                )
            return "good_performance"

        # Not saved
        if logger:
            test_display = f"{test_accuracy:.4f}" if test_accuracy is not None else "N/A"
            logger.info(
                "%s not saved because performance (val: %.4f; test: %s) is below threshold (%.3f).",
                label,
                val_accuracy,
                test_display,
                threshold,
            )
        return "bad_performance"

    def consider_saving_best_model(
        self,
        name_info,
        args,
        num_classes,
        logger=None,
        model_info: Optional[SavedModelInfo] = None,
    ):
        perf = self.consider_saving_model(
            label="best_model",
            model=self.best_model,
            val_accuracy=self.best_model_val_accuracy,
            test_accuracy=self.best_model_test_accuracy,
            epoch=int(self.best_epoch) if self.best_epoch is not None and self.best_epoch >= 0 else None,
            name_info=name_info,
            args=args,
            num_classes=num_classes,
            logger=logger,
            model_info=model_info,
        )
        self.best_model_performance_str = perf

    def consider_saving_final_model(
        self,
        name_info,
        args,
        num_classes,
        logger=None,
        model_info: Optional[SavedModelInfo] = None,
    ):
        perf = self.consider_saving_model(
            label="final_model",
            model=self.final_model,
            val_accuracy=self.final_val_accuracy,
            test_accuracy=self.final_test_accuracy,
            epoch=self.final_epoch,
            name_info=name_info,
            args=args,
            num_classes=num_classes,
            logger=logger,
            model_info=model_info,
        )
        self.final_performance_str = perf
def save_model_w_yaml(
    model,
    args,
    name_info,
    output_dir,
    logger=None,
    model_info: Optional[SavedModelInfo] = None,
):
    os.makedirs(output_dir, exist_ok=True)
    if name_info in {"best_model", "final_model"}:
        model_path = os.path.join(output_dir, f"{name_info}.pt")
    else:
        model_path = os.path.join(output_dir, f"MODEL_{name_info}_{args.model_type}.pt")

    torch.save(model.state_dict(), model_path)
    if logger:
        logger.info("Performance is very good. Thus Model saved successfully in %s", model_path)

    if model_info is not None:
        yaml_path = model_path.replace(".pt", ".yml")
        model_info.save_yaml(yaml_path)
        if logger:
            logger.info("Corresponding model info saved in %s as %s", args.output_dir, yaml_path)




