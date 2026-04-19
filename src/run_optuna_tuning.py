import argparse
import copy
import pprint
import time
from contextlib import nullcontext
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, List, Optional
import mlflow
import optuna
import yaml

import src.train_autoencoder as train_autoencoder
import src.train_spectrogram as train_spectrogram

import src.utils.reporting.pdf_report as pdf_report
from src.utils.reporting.logger_setup import setup_run_logger
import src.utils.toolkit.time_utils as time_utils
from src.schemas.dataclasses import (
    OptunaStudyResult,
    TrainingResult,
)
from src.schemas.enums import RunMode
from src.settings import FILES, MLFLOW, PRUNING, STUDY_IDS
from src.utils.toolkit.config_utils import (
    compare_search_space_and_config,
    update_flat_dict_strict,
    validate_and_clean_tuning_config,
)
from src.utils.toolkit.naming import create_study_id
from src.utils.reporting.mlflow_helper import (
    log_optuna_study_to_mlflow,
    safe_mlflow_call,
)
from src.utils.tuning.optuna_reporting import (
    save_all_optuna_plots,
    save_optuna_summary_bundle,
    save_study_data_as_json,
    save_top_trials_config,
)

r"""
test_run:

python -m src_spectrogram.optuna_optimize --test_run --output_dir=outputs/test_output_dir --metadata_file= --config_file=hyperparameters_spectrogram.yml --selected_classes='HC,PD' --model_type=test_cnn --epochs=2 --trials=2 --random_seed=42

"""




def log_best_params(best_params, logger):
    """
    Logs the best parameters from an Optuna study using pretty print for cleaner output.

    Parameters:
    - best_params (dict): Dictionary containing the best parameters.
    - logger: Logger instance used for logging.
    """
    formatted_params = pprint.pformat(best_params, indent=2)
    logger.info("Best Parameters from Optuna Study:\n%s", formatted_params)


def save_best_params(filepath, best_params, logger):
    """
    Saves the best parameters from an Optuna study to a text file.

    Parameters:
    - filepath (str): Full path to the file where the parameters will be saved.
    - best_params (dict): Dictionary containing the best parameters.
    """
    target_path = Path(filepath)
    try:
        with target_path.open("w", encoding="utf-8") as f:
            f.write("Best Parameters from Optuna Study:\n")
            for key, value in best_params.items():
                f.write(f"{key}: {value}\n")
        logger.info(f"Best parameters successfully saved to {target_path}")
    except Exception as e:
        logger.info(f"Failed to save best parameters to {target_path}: {e}")


def load_dict_from_yaml_file(file_path):
  """
  Loads a dictionary from a YAML file.

  Args:
    file_path (str): The path to the YAML file.

  Returns:
    dict: The dictionary loaded from the file.
  """
  with Path(file_path).open("r", encoding="utf-8") as file:
    return yaml.safe_load(file)
def get_optuna_trial_suggest_values(trial, search_space_def):
    """
    Given a trial and a search space definition, returns a dict of sampled values.
    
    search_space_def format:
    {
        'lstm.layer_dropout': ['float', 0.1, 0.3, 0.01],  # last value is step size
        'lstm.final_dropout': ['float', 0.1, 0.3]
    }
    """
    sampled_values = {}
    
    for param_name, param_info in search_space_def.items():
        param_type = param_info[0]
        
        if param_type == 'float':
            low, high = param_info[1], param_info[2]
            step = param_info[3] if len(param_info) > 3 else None
            if step:
                sampled_values[param_name] = trial.suggest_float(param_name, low, high, step=step)
            else:
                sampled_values[param_name] = trial.suggest_float(param_name, low, high)
        elif param_type == 'log_float':
            low, high = param_info[1], param_info[2]
            step = param_info[3] if len(param_info) > 3 else None
            if step:
                raise ValueError("The parameter `step` is not supported when `log` is true.")
            else:
                sampled_values[param_name] = trial.suggest_float(param_name, low, high, log=True)
        elif param_type == 'int':
            low, high = param_info[1], param_info[2]
            step = param_info[3] if len(param_info) > 3 else None
            if step:
                sampled_values[param_name] = trial.suggest_int(param_name, low, high, step=step)
            else:
                sampled_values[param_name] = trial.suggest_int(param_name, low, high)
        elif param_type == 'categorical':
            choices = param_info[1]
            sampled_values[param_name] = trial.suggest_categorical(param_name, choices)
        else:
            raise ValueError(f"Unknown param type: {param_type} for param {param_name}")
    
    return sampled_values






r"""

              _                     
   ___  _ __ | |_ _   _ _ __   __ _ 
  / _ \| '_ \| __| | | | '_ \ / _` |
 | (_) | |_) | |_| |_| | | | | (_| |
  \___/| .__/ \__|\__,_|_| |_|\__,_|
       |_|                          

       
"""




search_space_def = None # to avoid errors in objective()


def objective(
    trial,
    args,
    base_config,
    search_space_def,
    compute_test_metrics_in_trial: bool = True,
): #, dataset_manager=None):

    trial_config = copy.deepcopy(base_config)

    # Get all sampled values for this trial
    param_space = get_optuna_trial_suggest_values(trial, search_space_def)

    updated_trial_config = update_flat_dict_strict(base_config= trial_config, target_values= param_space)
    print('\nupdated trial config:' + pprint.pformat(updated_trial_config, indent=2))    # get the losses from the training session
    # train_model requires a trial to perform optuna study
    if 'autoencoder' in args.model_type:
        avg_val_loss_last_epoch, avg_test_loss_last_epoch = train_autoencoder.run_training(args, updated_trial_config, trial = trial) #, dataset_manager=dataset_manager) #  --> VALIDATION LOSS
    else:
        _run_mode = RunMode.NORMAL_TRAIN_ALWAYS_WITHOUT_RESULTS
        study_id = getattr(args, "study_id", "OPT")
        trial_result = train_spectrogram.run_training(
            args,
            config=updated_trial_config,
            trial=trial,
            run_mode=_run_mode,
            training_run_name=f"{study_id}_trial_{trial.number + 1}",
            compute_test_metrics_in_trial=compute_test_metrics_in_trial,
        )  # --> VALIDATION LOSS
        avg_val_loss_last_epoch = trial_result.final_model_val_mba
        avg_test_loss_last_epoch = trial_result.final_model_test_mba

    print(f"------------ Finished Trial #{trial.number} - Validation Loss: {avg_val_loss_last_epoch:.4f} - Test Loss: {avg_test_loss_last_epoch:.4f}-----------\n\n")

    assert isinstance(avg_val_loss_last_epoch, float)
    assert isinstance(avg_test_loss_last_epoch, float) 
    return avg_val_loss_last_epoch


def _retrain_best_trial_and_save_report(
    *,
    args,
    logger,
    study_id: str,
    original_config: dict,
    best_params: dict,
    search_space_def: dict,
    output_dir: Path,
) -> TrainingResult | None:
    logger.info('--------------- Now training the model on the best params ----------------')
    logger.info(f"Config before update: {original_config}")

    best_trial_config = update_flat_dict_strict(original_config, best_params)

    logger.info(f"Config after update: {best_trial_config}")
    compare_search_space_and_config(original_config, best_params)
    logger.info("All best_params keys validated against config")

    final_result = train_spectrogram.run_training(
        args=args,
        config=best_trial_config,
        run_mode=RunMode.NORMAL_TRAIN_ALWAYS_WITHOUT_RESULTS,
        training_run_name=f"{study_id}_best_trial_retrain",
    )
    list_of_pil_imgs = final_result.pil_images
    rep_str = final_result.report_info

    report_doc = pdf_report.TuningReportPDF()
    report_doc.add_search_space(search_space_def)
    report_doc.add_best_params(best_params)
    assert isinstance(list_of_pil_imgs, list)
    report_doc.add_final_training_pil_imgs(list_of_pil_imgs)
    if rep_str is not None:
        report_doc.add_classification_report_str(rep_str)
    else:
        logger.info("No classification report string returned; skipping report section.")
    report_full_path = output_dir / f"PDF_rep_{args.recording_category}_{time_utils.get_month_day()}.pdf"
    report_doc.save_full_pdf(str(report_full_path))
    logger.info(f'PDF report complete - saved in {report_full_path}')
    logger.info("optuna_optimize: Final checkpoint: All operations completed")
    return final_result


def optimize(args, last_optuna_trial_after_tuning=False, study_id: Optional[str] = None) -> OptunaStudyResult: 
    
    if not study_id:
        study_id = create_study_id(STUDY_IDS.DIRECT_OPTUNA_PREFIX)

    
    
    optuna_start_dt = datetime.now()
    optuna_start_t = time.perf_counter()
    summary_timestamp = optuna_start_dt.strftime("%Y-%m-%d_%H-%M-%S")
    tuning_search_space = None
    search_space_def = None # out of paranoia


    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_run_logger(__name__, str(output_dir / FILES.LOGGER_FILENAME))
    
    logger.info("\n\n\n\n\n\n=== OPTUNA OPTIMIZER SCRIPT STARTED ===")

    original_config = load_dict_from_yaml_file(args.config_file)
    original_config = train_spectrogram.filter_config_for_run(original_config, model_type=args.model_type)


    mlflow_enabled = True
    mlflow_enabled, _ = safe_mlflow_call(
        mlflow_enabled,
        logger,
        "set_experiment",
        mlflow.set_experiment,
        MLFLOW.EXPERIMENT_NAME,
    )

    if study_id is None:
        study_id = create_study_id(STUDY_IDS.DIRECT_OPTUNA_PREFIX)
    args.study_id = study_id
    study_run_name = f"{study_id}_optuna_study_{args.model_type}_{args.dataset_v}_{args.recording_category}"
    logger.info("OPTUNA IDs | study_id=%s | run_name=%s", study_id, study_run_name)
    run_context = nullcontext()
    if mlflow_enabled:
        try:
            nested_run = mlflow.active_run() is not None
            run_context = mlflow.start_run(run_name=study_run_name, nested=nested_run)
        except Exception as exc:
            mlflow_enabled = False
            logger.warning("MLflow start_run failed. Continuing without MLflow: %s", exc)

    with run_context:

        #if dataset_manager is None:
        #    dataset_manager = train_spectrogram.load_dataset_instances(args, original_config['spectrogram'], logger)

        #global search_space_def
        
        if tuning_search_space is None:
            search_space_def = load_dict_from_yaml_file(args.tuning_file)
        else:
            search_space_def = tuning_search_space

        search_space_def = validate_and_clean_tuning_config(
            tuning_config=search_space_def,
            model_type=args.model_type,
            dataset_v=args.dataset_v,
            logger=logger,
            strict=True,
        )

        # orignal_config should stay untouched - no changes wanted --> deepcopy
        config = copy.deepcopy(original_config)



        logger.info("\nArguments:\n%s", pprint.pformat(vars(args)))
        logger.info("\noriginal YML configuration:\n%s", pprint.pformat(original_config, indent=2))
        logger.info("\nparam search space (after excluding irrelevant data):\n%s", pprint.pformat(search_space_def, indent=2))

        compare_search_space_and_config(config, search_space_def)
        logger.info("All search-space keys validated against config")

        #nr_trials = int(args.trials)

        # Define the pruner
        pruner_1 = optuna.pruners.MedianPruner(
            n_startup_trials=PRUNING.OPTUNA.MEDIAN.N_STARTUP_TRIALS,  # Number of unpruned trials at the beginning
            n_warmup_steps=PRUNING.OPTUNA.MEDIAN.N_WARMUP_STEPS,      # Number of steps before pruning begins for a single trial
            interval_steps=PRUNING.OPTUNA.MEDIAN.INTERVAL_STEPS,      # How often to check for pruning
        )

        n_trials = int(args.trials)
        direction='minimize' if 'autoencoder' in args.model_type else 'maximize'
        logger.info(f'model type: {args.model_type}; direction of optuna: {direction}')
        logger.info(
            "OPTUNA SETUP | dataset=%s | classes=%s | category=%s | model=%s | trials=%s | direction=%s | test_run=%s | retrain_best=%s",
            args.dataset_v,
            args.selected_classes,
            args.recording_category,
            args.model_type,
            n_trials,
            direction,
            bool(getattr(args, "test_run", False)),
            bool(last_optuna_trial_after_tuning),
        )
        logger.info("OPTUNA STARTING study.optimize | planned_trials=%s", n_trials)
        study = optuna.create_study(direction=direction, pruner=pruner_1) #, sampler=optuna.samplers.GridSampler)
        compute_test_metrics_in_trial = bool(getattr(args, "compute_test_metrics_in_trial", True))
        study.optimize(
            lambda trial: objective(
                trial,
                args,
                config,
                search_space_def,
                compute_test_metrics_in_trial,
            ),
            n_trials=n_trials,
        ) #, dataset_manager), n_trials= int(args.trials))

        total_trials = len(study.trials)
        complete_trials = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        pruned_trials = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
        failed_trials = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL)
        logger.info(
            "OPTUNA TRIAL SUMMARY | total=%s | complete=%s | pruned=%s | failed=%s",
            total_trials,
            complete_trials,
            pruned_trials,
            failed_trials,
        )

        best_trial = study.best_trial
        logger.info(
            "OPTUNA BEST TRIAL | number=%s | value=%s | n_params=%s | direction=%s",
            best_trial.number,
            best_trial.value,
            len(best_trial.params),
            direction,
        )
        logger.info(f"Best Trial - Number: {best_trial.number}, Loss: {best_trial.value}, \nParameters:\n{pprint.pformat(best_trial.params)}")
        
        ############      Saving data       ##############
        # SAVE ARGS to logger
        # Optuna study is done - PROCESSING OF OPTUNA RESULTS
        save_study_data_as_json(
            study,
            output_dir=output_dir,
            nr_of_trials=args.trials,
            prefix=f"{study_id}_",
        )
        
        name_info = f"{study_id}_{args.model_type}_{time_utils.get_month_day()}_{args.recording_category}"
        save_all_optuna_plots(study, str(output_dir / f"OPTUNA_graphs_{name_info}"), name_prefix = name_info)
        logger.info(f'OPTUNA data vis was successfully saved in {args.output_dir}')

        ############      Saving summary       ##############
        try:
            summary_bundle_paths = save_optuna_summary_bundle(
                study=study,
                direction=direction,
                study_id=study_id,
                summary_timestamp=summary_timestamp,
                search_space_def=search_space_def,
                args=args,
                output_dir=output_dir,
            )
            logger.info(
                "Optuna summary artifacts saved to %s",
                summary_bundle_paths["summary_dir"],
            )
            logger.info(
                "Optuna summary markdown: %s",
                summary_bundle_paths["summary_path"],
            )
        except Exception:
            logger.exception('Saving the Optuna summary bundle failed.')


        #####################################################
        #########        final Training           ###########
        #####################################################
        final_result: Optional[TrainingResult] = None
        if last_optuna_trial_after_tuning:
            final_result = _retrain_best_trial_and_save_report(
                args=args,
                logger=logger,
                study_id=study_id,
                original_config=original_config,
                best_params=best_trial.params,
                search_space_def=search_space_def,
                output_dir=output_dir,
            )
        else:
            logger.info('last_optuna_trial_after_tuning == False --> additional training will not be done')






        # Save best configs
        best_config_paths = save_top_trials_config(
            study,
            args,
            original_config,
            args.output_dir,
            study_id=study_id,
        )
        if best_config_paths:
            formatted_paths = "\n".join(f"{idx}. {path}" for idx, path in enumerate(best_config_paths, start=1))
            logger.info(
                "Saved top trial configs (%d):\n%s",
                len(best_config_paths),
                formatted_paths
            )

            mlflow_enabled, _ = safe_mlflow_call(
                mlflow_enabled,
                logger,
                "log_param(best_config_paths)",
                mlflow.log_param,
                "best_config_paths",
                "; ".join(best_config_paths),
            )

        else:
            logger.info("No top trial configs were saved.")


        # -------------------------------------------------------------------
        # MLflow Study Logging
        # -------------------------------------------------------------------
        mlflow_enabled = log_optuna_study_to_mlflow(
            enabled=mlflow_enabled,
            logger=logger,
            best_trial=best_trial,
            args=args,
            study_id=study_id,
            output_dir=output_dir,
        )




        optuna_end_dt = datetime.now()
        elapsed_seconds = time.perf_counter() - optuna_start_t
        duration_str = str(timedelta(seconds=round(elapsed_seconds)))
        logger.info(
            "OPTUNA FINISHED | start=%s | end=%s | duration=%s | study_id=%s | planned_trials=%s | total_trials=%s",
            optuna_start_dt.isoformat(timespec="seconds"),
            optuna_end_dt.isoformat(timespec="seconds"),
            duration_str,
            study_id,
            n_trials,
            total_trials,
        )
        return OptunaStudyResult(
            study_id=study_id,
            best_trial_number=best_trial.number,
            best_params=best_trial.params,
            best_value=best_trial.value if best_trial.value else 0.0,
            best_trial_config_path_list=best_config_paths,
            study=study,
            final_training_result_instance=final_result,
            output_dir=args.output_dir
        )







