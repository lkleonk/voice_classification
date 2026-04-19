import argparse
import json
import pprint
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

import yaml
from PIL import Image
from torch.utils.data import DataLoader

# Import the custom classes
import src.utils.toolkit.cuda_handling as cuda_handling
from src.schemas.dataclasses import (
    SavedModelInfo,
    TrainingArtefactsManager,
    TrainingResult,
)
from src.schemas.enums import RunMode
from src.settings import CONFIG, DATALOADER, FILES, MODELS
from src.utils.data_preprocessing.dataset_split_manager import DatasetManager
from src.utils.reporting.mlflow_helper import (
    init_training_mlflow_run,
    log_training_summary_to_mlflow,
)
from src.utils.reporting.training_reporting import (
    finalize_training_reporting,
    save_and_log_best_model_loss_artifacts,
)
from src.utils.reporting.training_metrics_artifacts import (
    save_metrics_json_files,
)
from src.utils.training.setup import build_training_setup
from src.model.pretrained.pretrained_models_params import overwrite_config_values
from src.utils.toolkit.eval_model import eval_model
from src.utils.reporting.logger_setup import setup_run_logger
from src.utils.toolkit.naming import (
    build_training_run_name,
)
from src.utils.training.loop import training_loop




device = cuda_handling.set_cuda_to_gpu_nr()




r"""
test_run:

python -m src_spectrogram.train_spectrogram --test_run --metadata_file= --config_file=hyperparameters_spectrogram.yml --output_dir=outputs/test_output_dir --model_type=test_cnn --epochs=2 --random_seed=42 --selected_classes='HC,PD'

"""




def filter_config_for_run(config: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    """
    Keep only:
    - core shared namespaces (spectrogram/features/loss/optimizer/scheduler/waveform)
    - {model_type}.* keys

    Drop everything else.
    """
    prefixes = (*CONFIG.SHARED_FLAT_KEY_PREFIXES, f"{model_type}.")

    return {
        k: v
        for k, v in config.items()
        if k.startswith(prefixes)
    }
def build_model_info(
    *,
    config: Dict[str, Any],
    args_obj: Any,
    report_info: Any,
) -> SavedModelInfo:
    return SavedModelInfo(
        configuration=config,
        arguments=vars(args_obj),
        test_report_info=report_info,
    )


def format_metric(metric, metric2=None):
    if metric2 is not None:
        metric_str = f"{metric:.3f}v_{metric2:.3f}t".replace(".", "")
    else:
        metric_str = f"{metric:.3f}".replace(".", "")
    return metric_str


def run_training(
    args: argparse.Namespace,
    config: Optional[Dict[str, Any]] = None,
    trial=None,
    logger=None,
    dataset_manager: Optional[DatasetManager] = None,
    run_mode=RunMode.NORMAL_TRAIN_ALWAYS_WITHOUT_RESULTS,
    training_run_name: Optional[str] = None,
    compute_test_metrics_in_trial: bool = True,
) -> TrainingResult:
    """
    for a test_run: script is executed locally - without snakefile context

    """
    # -------------------------------------------------------------------
    # 1. Setup
    # -------------------------------------------------------------------
    final_model_result_val_auroc = None
    final_model_result_test_auroc = None
    final_model_result_val_mba: Optional[float] = None
    final_model_result_test_mba: Optional[float] = None

    best_model_result_val_auroc = None
    best_model_result_test_auroc = None
    best_model_result_val_mba: Optional[float] = None
    best_model_result_test_mba: Optional[float] = None

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    result_pil_images: Optional[List[Image.Image]] = None
    result_report_info: Optional[str] = None

    train_start_dt = datetime.now(timezone.utc)
    train_start_t = time.perf_counter()

    if logger is None:
        logger = setup_run_logger(__name__, str(Path(args.output_dir) / FILES.LOGGER_FILENAME))
    model_type = args.model_type.lower()

    logger.info(
        f"TRAINING START | dataset={args.dataset_v} | classes={args.selected_classes} | "
        f"category={args.recording_category} | model={args.model_type} | run_mode={run_mode} | "
        f"trial_number={getattr(trial, 'number', None)} | study_id={getattr(args, 'study_id', None)} | "
        f"test_run={getattr(args, 'test_run', False)} | output_dir={args.output_dir} | "
        f"training_run_name={training_run_name}"
    )
    logger.info(f"Environment | cwd={Path.cwd()} | python={sys.executable}")


    ## LOAD CONFIG ##
    if config is None:
        with open(args.config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        # Guard caller-owned config against accidental in-place mutations.
        config = dict(config)


    if (
        model_type in MODELS.PRETRAINED_MODELS_LIST
        or model_type in MODELS.DEMO_ONLY_NO_AUDIO_MODELS_LIST
    ):
        logger.info("=" * 60)
        logger.info(f"MODEL-SPEC OVERRIDES DETECTED: {model_type}")
        logger.info(f"Config type before overwrite: {type(config)}")
        assert config is not None, "Config is None before overwrite_config_values()"
        config = overwrite_config_values(config, args.model_type)


    assert config is not None, "Config is None after loading. Check config loading branches."
    #set hyperparameters


    config = filter_config_for_run(config, model_type=args.model_type)


    if not trial: # args are already printed in the optuna_optimize.py
        # Print all args and config entries to logger
        logger.info("Arguments:\n%s", pprint.pformat(vars(args)))
    logger.info("YML configuration for 'spectrogram' and model_type:\n%s", pprint.pformat(config, indent=2))


    if not training_run_name:
        base_training_run_name = build_training_run_name(args, pretrained=False)
        study_id = getattr(args, "study_id", None)
        training_run_name = f"{study_id}_{base_training_run_name}" if study_id else base_training_run_name

    mlflow_enabled, run_context = init_training_mlflow_run(
        run_name=f"TRAIN_{training_run_name}",
        logger=logger,
    )
    with run_context:
        # -------------------------------------------------------------------
        # 2. Training setup & load dataset_manager
        # -------------------------------------------------------------------

        setup = build_training_setup(
            args=args,
            config=config,
            logger=logger,
            device=device,
            dataset_manager=dataset_manager,
        )
        dataset_manager = setup.dataset_manager
        model = setup.model
        criterion = setup.criterion
        criterion_w_weights = setup.criterion_w_weights
        optimizer = setup.optimizer
        train_dataloader = setup.train_dataloader
        val_dataloader = setup.val_dataloader


        # -------------------------------------------------------------------
        # 3. Training
        # -------------------------------------------------------------------

        lr_decay_mode = config[f'{model_type}.lr_decay']
        lr_decay_type = config[f'{model_type}.lr_decay_type']
        eta_min = float(config['scheduler.eta_min'])
        weighted_loss_mode = config['loss.weighted_loss']# during training only - loss given to optuna is not weighted
        add_demographic_data = args.add_demo_data

        name_info = training_run_name


        art: Optional[TrainingArtefactsManager] = None


        art = training_loop(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            criterion_w_weights=criterion_w_weights if criterion_w_weights else None,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            device=device,
            logger=logger,
            args=args,
            add_demographic_data=add_demographic_data,
            weighted_loss_mode=weighted_loss_mode,
            lr_decay_type=lr_decay_type,
            eta_min=eta_min,
            lr_decay_mode=lr_decay_mode,
            mlflow_enabled=mlflow_enabled,
            trial=trial,
        )
        model_m = art.model_m
        final_model_result_val_auroc = float(art.val_aurocs[-1]) if art.val_aurocs else 0.0



        # -------------------------------------------------------------------
        # 4. Evaluate on Test set (optional for Optuna trial runs)
        # -------------------------------------------------------------------

        assert dataset_manager is not None, "dataset_manager is None before eval_model().."

        should_compute_test_metrics = (trial is None) or compute_test_metrics_in_trial

        final_model_test_results = None
        best_model_testing_results = None
        report_info = ""
        report_info_bm = ""
        logits_list: List[torch.Tensor] = []
        all_targets = []
        all_preds = []
        best_model_test_accuracy = float("nan")
        final_result_test_mba = float("nan")

        if should_compute_test_metrics:
            # ======= EVALUATION: TEST DATA =======
            model.eval()
            test_dataloader = DataLoader(
                dataset_manager.test_dataset,
                batch_size=DATALOADER.EVAL_BATCH_SIZE,
                shuffle=False,
                num_workers=DATALOADER.NUM_WORKERS,
            )
            final_model_test_results = eval_model(
                model,
                device,
                test_dataloader,
                dataset_manager.label_encoder,
                criterion,
                logger,
                add_demographic_data_bool=add_demographic_data,
                info='training',
            )
            report_info = final_model_test_results.classification_report
            logits_list = [torch.tensor(final_model_test_results.logits)] #  if test_result.logits is not None else []
            all_targets = final_model_test_results.targets
            all_preds = final_model_test_results.preds
            result_report_info = report_info
            final_result_test_mba = float(final_model_test_results.balanced_accuracy)

            assert art is not None, "art must be returned by training_loop"
            ## FOR BEST MODEL (NOT FINAL MODEL EPOCH-WISE) ##
            best_model_testing_results = eval_model(
                model_m.best_model,
                device,
                test_dataloader,
                dataset_manager.label_encoder,
                criterion,
                logger,
                add_demographic_data_bool=add_demographic_data,
                info=f'Best model (epoch: {model_m.best_epoch} - val bal acc: {model_m.best_model_val_accuracy})',
            )  # bm stands for best model
            best_model_test_accuracy = float(best_model_testing_results.balanced_accuracy)
            report_info_bm = best_model_testing_results.classification_report
        else:
            logger.info("Optuna trial mode: test-set evaluation is skipped for faster tuning.")

        # give model instance from final executed epoch to model manager
        actual_last_epoch = len(art.val_losses)  # 1-based epoch count
        model_m.set_model_weights_from_final_epoch(
            model=model,
            final_model_val_mba=art.get_last_val_balanced_accuracy(),
            final_model_test_mba=final_result_test_mba,
            epoch=actual_last_epoch,
        )
        model_m.best_model_test_accuracy = best_model_test_accuracy

        best_model_metrics_str = format_metric(model_m.best_model_val_accuracy, model_m.best_model_test_accuracy)
        final_model_metrics_str = format_metric(model_m.final_val_accuracy, model_m.final_test_accuracy)
        # extended_name:info carries info from name_info (run settings) + performance (and the mba metrics for val and test)
        best_model_extended_name_info = f'{name_info}_{model_m.best_epoch}_{best_model_metrics_str}'
        final_model_extended_name_info = f'{name_info}_{model_m.final_epoch}_{final_model_metrics_str}'

        # Build the YAML sidecar payload for best/final checkpoint saves.
        last_epoch_model_info = build_model_info(
            config=config,
            args_obj=args,
            report_info=report_info,
        )
        best_epoch_model_info = build_model_info(
            config=config,
            args_obj=args,
            report_info=report_info_bm,
        )

        ## SET VAL AND TEST VALUES
        # final_model_test_results contains all I need for test values
        final_model_result_val_mba = float(model_m.final_val_accuracy) # correct
        final_model_result_test_mba = float(final_result_test_mba)
        final_model_result_val_auroc = float(art.get_last_auroc()) # correct
        final_model_result_test_auroc = float(final_model_test_results.auroc_macro_ovr) if final_model_test_results is not None else float("nan")

        ## best_model_testing_results contains all I need for test results.
        best_model_result_val_mba = float(model_m.best_model_val_accuracy) # correct
        best_model_result_test_mba = float(best_model_test_accuracy)
        best_model_result_val_auroc = float(model_m.best_model_auroc) # correct
        best_model_result_test_auroc = float(best_model_testing_results.auroc_macro_ovr) if best_model_testing_results is not None else float("nan")




        ##### MODEL MANAGEMENT ######
        if should_compute_test_metrics:
            model_m.consider_saving_best_model(
                best_model_extended_name_info,
                args,
                setup.num_classes,
                logger,
                best_epoch_model_info,
            )
            model_m.consider_saving_final_model(
                final_model_extended_name_info,
                args,
                setup.num_classes,
                logger,
                last_epoch_model_info,
            )

        mark_good_run = (
            model_m.best_model_performance_str == "good_performance"
            or run_mode == RunMode.NORMAL_TRAIN_ALWAYS_W_RESULTS
        )
        mlflow_enabled = log_training_summary_to_mlflow(
            mlflow_enabled=mlflow_enabled,
            logger=logger,
            name_info=name_info,
            cli_args=args,
            config=config,
            mark_good_run=mark_good_run,
            best_model_epoch=model_m.best_epoch,
            best_model_val_mba=model_m.best_model_val_accuracy,
            best_model_test_mba=model_m.best_model_test_accuracy,
            final_model_val_mba=model_m.final_val_accuracy,
            final_model_test_mba=model_m.final_test_accuracy,
        )


        if should_compute_test_metrics and final_model_test_results is not None and best_model_testing_results is not None:
            save_metrics_json_files(
                args.output_dir,
                best_model_extended_name_info,
                final_model_extended_name_info,
                art,
                final_model_test_results,
                best_model_testing_results,
                best_epoch=int(model_m.best_epoch) if model_m.best_epoch is not None and model_m.best_epoch >= 0 else None,
                final_epoch=int(model_m.final_epoch) if model_m.final_epoch is not None else None,
                args_obj=args,
                trial=trial,
                logger=logger,
            )


        # ------------------
        #   POSTPROCESSING
        # ------------------
        if run_mode == RunMode.NORMAL_TRAIN_ALWAYS_W_RESULTS:

            loss_artifacts = save_and_log_best_model_loss_artifacts(
                logger=logger,
                args=args,
                model_m=model_m,
                art=art,
                mlflow_enabled=mlflow_enabled,
                weighted_loss_mode=weighted_loss_mode,
            )
            mlflow_enabled = loss_artifacts.mlflow_enabled
            ######## POSTPROCESSING that is not learning rate related ########

            logger.info("Training run complete. Further files will be saved in the next steps. Output directory: %s", args.output_dir)
            artifacts = finalize_training_reporting(
                args=args,
                name_info=name_info,
                config=config,
                model=model,
                art=art,
                dataset_manager=dataset_manager,
                label_encoder=dataset_manager.label_encoder,
                report_info=report_info,
                all_targets=all_targets,
                all_preds=all_preds,
                logits_list=logits_list,
                weighted_loss_mode=weighted_loss_mode,
                lr_decay_mode=lr_decay_mode,
                model_type=model_type,
                model_m=model_m,
                logger=logger,
                mlflow_enabled=mlflow_enabled,
            )

            result_pil_images = artifacts.pil_images


        model = None
        dataset_manager = None

        payload = {
            "event": "training_run_completed",
            "mode": str(run_mode),

            "metrics": {
                "final_model": {
                    "val_mba": final_model_result_val_mba,
                    "test_mba": final_model_result_test_mba,
                    "val_auroc": final_model_result_val_auroc,
                    "test_auroc": final_model_result_test_auroc,
                },
                "best_model": {
                    "val_mba": best_model_result_val_mba,
                    "test_mba": best_model_result_test_mba,
                    "val_auroc": best_model_result_val_auroc,
                    "test_auroc": best_model_result_test_auroc,
                },
            },

            "artifacts": {
                "n_pil_images": len(result_pil_images) if result_pil_images else 0,
                "has_report_info": result_report_info is not None,
            },

            "metric_presence": {
                "final_model": {
                    "val_mba": final_model_result_val_mba is not None,
                    "test_mba": final_model_result_test_mba is not None,
                    "val_auroc": final_model_result_val_auroc is not None,
                    "test_auroc": final_model_result_test_auroc is not None,
                },
                "best_model": {
                    "val_mba": best_model_result_val_mba is not None,
                    "test_mba": best_model_result_test_mba is not None,
                    "val_auroc": best_model_result_val_auroc is not None,
                    "test_auroc": best_model_result_test_auroc is not None,
                },
            },
        }

        logger.info(
            "training_run_completed | %s",
            json.dumps(payload, ensure_ascii=False, default=str),
        )

        train_end_dt = datetime.now(timezone.utc)
        elapsed_seconds = time.perf_counter() - train_start_t
        duration_str = str(timedelta(seconds=round(elapsed_seconds)))
        logger.info(
            f"TRAINING FINISHED | start={train_start_dt.isoformat(timespec='seconds')} | "
            f"end={train_end_dt.isoformat(timespec='seconds')} | duration={duration_str} | "
            f"run_mode={run_mode} | test_metrics_computed={should_compute_test_metrics} | "
            f"training_run_name={training_run_name} | output_dir={args.output_dir}"
        )

        if not isinstance(result_pil_images, list):
            result_pil_images = []
        return TrainingResult(
            mode=run_mode,
            pil_images=result_pil_images,
            report_info=result_report_info,
            final_model_val_mba=final_model_result_val_mba,
            final_model_test_mba=final_model_result_test_mba,
            final_model_val_auroc=final_model_result_val_auroc,
            final_model_test_auroc=final_model_result_test_auroc,
            best_model_val_mba=best_model_result_val_mba,
            best_model_test_mba=best_model_result_test_mba,
            best_model_val_auroc=best_model_result_val_auroc,
            best_model_test_auroc=best_model_result_test_auroc,
        )

