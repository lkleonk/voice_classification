import argparse
import copy
import json
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
import optuna
import optuna.visualization as vis
import yaml

import src.utils.toolkit.time_utils as time_utils
from src.settings import OPTUNA, PATHS
from src.utils.toolkit.config_utils import update_flat_dict_strict
from src.utils.toolkit.naming import (
    build_best_config_filename,
    get_norm_str,
)


def save_study_data_as_json(study, output_dir, nr_of_trials, prefix=''):
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    trial_data = {}
    for trial in study.trials:
        trial_data[trial.number] = {
            'status': trial.state.name,
            'params': trial.params,
            'value': trial.value
        }

    json_file_name = output_dir_path / f"{prefix}OPTUNA_study_data_{time_utils.get_day_month_year_hour_minute()}_{nr_of_trials}trials.json"
    with json_file_name.open("w", encoding="utf-8") as f:
        json.dump(trial_data, f, indent=4)

    return trial_data


def save_all_optuna_plots(
    study,
    output_folder_name: str = OPTUNA.PLOTS_OUTPUT_DIRNAME,
    name_prefix: str = OPTUNA.PLOTS_NAME_PREFIX,
):
    output_folder = Path(output_folder_name)
    output_folder.mkdir(parents=True, exist_ok=True)

    plot_functions = {
        "parallel_coordinate": vis.plot_parallel_coordinate,
        "slice_plot": vis.plot_slice,
        "contour_plot": vis.plot_contour,
        "param_importances": vis.plot_param_importances,
        "intermediate_values": vis.plot_intermediate_values,
    }

    for func_name, plot_func in plot_functions.items():
        try:
            fig = plot_func(study)
            fig.write_html(str(output_folder / f"{name_prefix}_{func_name}.html"))
        except Exception as e:
            print(f"Skipping {func_name}: {e}")


def save_top_trials_config(
    study,
    args,
    original_config,
    output_dir,
    study_id: Optional[str] = None,
    top_n: int = OPTUNA.TOP_N_TRIALS,
) -> List[str]:
    saved_paths = []

    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    direction = study.direction
    if direction == optuna.study.StudyDirection.MINIMIZE:
        sorted_trials = sorted(complete_trials, key=lambda t: t.value)
    else:
        sorted_trials = sorted(complete_trials, key=lambda t: t.value, reverse=True)

    top_trials = sorted_trials[:top_n]

    del output_dir
    target_dir = PATHS.BEST_CONFIGS_SPECIFIC_DIR / time_utils.get_year_month()
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time_utils.get_year_month_day_hour_minute_second()

    for rank, trial in enumerate(top_trials, start=1):
        try:
            trial_config = copy.deepcopy(original_config)
            updated_config = update_flat_dict_strict(base_config=trial_config, target_values=trial.params)

            updated_config['info.timestamp'] = timestamp
            updated_config['info.trial_number'] = trial.number
            updated_config['info.trial_value'] = trial.value
            updated_config['info.model_type'] = args.model_type
            updated_config['info.dataset_version'] = args.dataset_v
            updated_config['info.recording_category'] = args.recording_category
            updated_config['info.norm_mode'] = get_norm_str(args.norm_mode)
            updated_config['info.lung_conditions'] = ",".join(
                [c.strip().lower() for c in str(args.selected_classes).split(",") if c.strip()]
            )
            updated_config['info.rank'] = rank
            if study_id is not None:
                updated_config["info.study_id"] = study_id

            filename = build_best_config_filename(
                model_type=args.model_type,
                dataset_v=args.dataset_v,
                selected_classes=args.selected_classes,
                recording_category=args.recording_category,
                timestamp=timestamp,
                rank=rank,
                study_id=study_id,
            )
            filepath = target_dir / filename

            with filepath.open("w", encoding="utf-8") as f:
                yaml.dump(updated_config, f, default_flow_style=False, sort_keys=False)

            saved_paths.append(str(filepath.resolve()))

        except Exception as e:
            print(f"Failed to save config for trial {trial.number}: {e}")

    return saved_paths


def _format_scalar(value: Any, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        if pd.isna(value):
            return "n/a"
    except TypeError:
        pass

    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)

    return f"{numeric_value:.{precision}f}"


def _duration_to_seconds(duration_value: Any) -> Optional[float]:
    if duration_value is None:
        return None
    try:
        if pd.isna(duration_value):
            return None
    except TypeError:
        pass

    if hasattr(duration_value, "total_seconds"):
        return float(duration_value.total_seconds())

    return None


def _format_duration(total_seconds: Optional[float]) -> str:
    if total_seconds is None:
        return "n/a"
    if total_seconds < 60:
        return f"{total_seconds:.1f}s"

    minutes, seconds = divmod(total_seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {seconds:.1f}s"

    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"


def _build_optuna_trials_dataframe(study) -> pd.DataFrame:
    df = study.trials_dataframe(attrs=("number", "value", "state", "duration", "params"))
    if df.empty:
        return df

    df = df.copy()
    if "duration" in df.columns:
        df["duration_seconds"] = df["duration"].apply(_duration_to_seconds)
        df["duration_readable"] = df["duration_seconds"].apply(_format_duration)
        df["duration"] = df["duration"].astype(str)
        df.loc[df["duration"] == "NaT", "duration"] = ""

    return df


def _sort_trials_df(df: pd.DataFrame, direction: str) -> pd.DataFrame:
    return df.sort_values("value", ascending=(direction == "minimize")).reset_index(drop=True)


def _format_trial_params(row: pd.Series, param_cols: List[str]) -> str:
    params_text = []
    for col in param_cols:
        value = row.get(col)
        try:
            if pd.isna(value):
                continue
        except TypeError:
            pass
        params_text.append(f"{col.replace('params_', '')}={_format_scalar(value)}")
    return ", ".join(params_text) if params_text else "no params recorded"


def generate_hyperparam_analysis_string(
    study,
    direction,
    *,
    study_id: Optional[str] = None,
    search_space_def: Optional[dict] = None,
    args: Optional[argparse.Namespace] = None,
    summary_timestamp: Optional[str] = None,
    summary_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    top_n: Optional[int] = None,
) -> str:
    all_trials_df = _build_optuna_trials_dataframe(study)
    if all_trials_df.empty:
        return "# Optuna Study Summary\n\nNo trials were recorded for this study."

    complete_df = all_trials_df[all_trials_df["state"] == "COMPLETE"].copy()
    if complete_df.empty:
        return (
            "# Optuna Study Summary\n\n"
            "Trials exist, but none completed successfully. Inspect pruning/failure logs."
        )

    df_sorted = _sort_trials_df(complete_df, direction)
    param_cols = [col for col in complete_df.columns if col.startswith("params_")]
    top_n = min(top_n or 10, len(df_sorted))
    top_trials_df = df_sorted.head(top_n)

    best_trial_row = df_sorted.iloc[0]
    best_value = float(best_trial_row["value"])
    median_value = float(complete_df["value"].median())
    mean_value = float(complete_df["value"].mean())
    worst_value = float(df_sorted.iloc[-1]["value"])
    std_value = float(complete_df["value"].std()) if len(complete_df) > 1 else 0.0

    total_trials = len(study.trials)
    complete_trials = len(complete_df)
    pruned_trials = sum(
        1 for trial in study.trials if trial.state == optuna.trial.TrialState.PRUNED
    )
    failed_trials = sum(
        1 for trial in study.trials if trial.state == optuna.trial.TrialState.FAIL
    )

    duration_seconds = all_trials_df.get("duration_seconds", pd.Series(dtype=float)).dropna()
    best_duration = _format_duration(best_trial_row.get("duration_seconds"))
    objective_name = (
        "validation loss" if direction == "minimize" else "validation balanced accuracy"
    )

    out = ["# Optuna Study Summary", ""]
    out.extend(
        [
            "## Study Context",
            f"- Study ID: {study_id or 'n/a'}",
            f"- Summary timestamp: {summary_timestamp or 'n/a'}",
            f"- Objective: {objective_name} ({direction})",
            f"- Total trials observed: {total_trials}",
            f"- Completed / pruned / failed: {complete_trials} / {pruned_trials} / {failed_trials}",
        ]
    )

    if args is not None:
        out.extend(
            [
                f"- Model: {getattr(args, 'model_type', 'n/a')}",
                f"- Dataset: {getattr(args, 'dataset_v', 'n/a')}",
                f"- Classes: {getattr(args, 'selected_classes', 'n/a')}",
                f"- Recording category: {getattr(args, 'recording_category', 'n/a')}",
                f"- Requested trials: {getattr(args, 'trials', 'n/a')}",
                f"- Test run: {bool(getattr(args, 'test_run', False))}",
                (
                    "- Compute test metrics in trial: "
                    f"{bool(getattr(args, 'compute_test_metrics_in_trial', True))}"
                ),
                f"- Config file: {getattr(args, 'config_file', 'n/a')}",
                f"- Tuning file: {getattr(args, 'tuning_file', 'n/a')}",
            ]
        )
    if output_dir is not None:
        out.append(f"- Optuna output dir: {output_dir}")
    if summary_dir is not None:
        out.append(f"- Summary bundle dir: {summary_dir}")

    out.extend(
        [
            "",
            "## Trial Outcome Overview",
            (
                "- Complete-trial objective distribution: "
                f"best={_format_scalar(best_value)}, median={_format_scalar(median_value)}, "
                f"mean={_format_scalar(mean_value)}, std={_format_scalar(std_value)}, "
                f"worst={_format_scalar(worst_value)}"
            ),
        ]
    )

    if not duration_seconds.empty:
        out.append(
            "- Trial duration (all recorded trials): "
            f"mean={_format_duration(float(duration_seconds.mean()))}, "
            f"min={_format_duration(float(duration_seconds.min()))}, "
            f"max={_format_duration(float(duration_seconds.max()))}"
        )

    best_params_block = yaml.safe_dump(
        study.best_trial.params,
        sort_keys=True,
        default_flow_style=False,
    ).strip()
    out.extend(
        [
            "",
            "## Best Trial Snapshot",
            f"- Trial number: {int(best_trial_row['number'])}",
            f"- Objective value: {_format_scalar(best_value)}",
            f"- Duration: {best_duration}",
            f"- Number of tuned parameters: {len(study.best_trial.params)}",
            "- Parameters:",
            "```yaml",
            best_params_block if best_params_block else "{}",
            "```",
        ]
    )

    out.extend(["", "## Top Completed Trials"])
    for rank, (_, row) in enumerate(top_trials_df.iterrows(), start=1):
        out.append(
            f"- Rank {rank} | trial={int(row['number'])} | value={_format_scalar(row['value'])} "
            f"| duration={row.get('duration_readable', 'n/a')} | {_format_trial_params(row, param_cols)}"
        )

    return "\n".join(out)


def save_optuna_summary_bundle(
    *,
    study,
    direction: str,
    study_id: str,
    summary_timestamp: str,
    search_space_def: dict,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Path]:
    summary_run_dir = (
        PATHS.OPTUNA_SUMMARY_DIR
        / f"{study_id}_{PATHS.TARGET_BEST_CONFIG_VERSION_DIR}"
    )
    summary_dir = summary_run_dir / f"{summary_timestamp}_{study_id}_optuna_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    all_trials_df = _build_optuna_trials_dataframe(study)
    complete_trials_df = all_trials_df[all_trials_df["state"] == "COMPLETE"].copy()
    ranked_complete_df = _sort_trials_df(complete_trials_df, direction)
    if not ranked_complete_df.empty:
        ranked_complete_df.insert(0, "rank", range(1, len(ranked_complete_df) + 1))

    summary_text = generate_hyperparam_analysis_string(
        study,
        direction,
        study_id=study_id,
        search_space_def=search_space_def,
        args=args,
        summary_timestamp=summary_timestamp,
        summary_dir=summary_dir,
        output_dir=output_dir,
    )

    summary_path = summary_dir / "00_optuna_study_summary.md"
    summary_path.write_text(summary_text, encoding="utf-8")

    ranked_trials_path = summary_dir / "01_complete_trials_ranked.csv"
    ranked_complete_df.to_csv(ranked_trials_path, index=False)

    all_trials_path = summary_dir / "02_all_trials.csv"
    all_trials_df.to_csv(all_trials_path, index=False)

    best_params_path = summary_dir / "03_best_trial_params.yaml"
    with best_params_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(study.best_trial.params, file, sort_keys=True)

    search_space_path = summary_dir / "04_search_space.yaml"
    with search_space_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(search_space_def, file, sort_keys=False)

    metadata = {
        "study_id": study_id,
        "summary_timestamp": summary_timestamp,
        "objective_direction": direction,
        "objective_name": (
            "validation loss"
            if direction == "minimize"
            else "validation balanced accuracy"
        ),
        "summary_dir": str(summary_dir.resolve()),
        "summary_run_dir": str(summary_run_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "counts": {
            "total_trials": len(study.trials),
            "complete_trials": int((all_trials_df["state"] == "COMPLETE").sum()),
            "pruned_trials": int((all_trials_df["state"] == "PRUNED").sum()),
            "failed_trials": int((all_trials_df["state"] == "FAIL").sum()),
        },
        "best_trial": {
            "number": study.best_trial.number,
            "value": study.best_trial.value,
            "params": study.best_trial.params,
        },
        "args": vars(args),
    }
    metadata_path = summary_dir / "05_study_context.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=str)

    return {
        "summary_run_dir": summary_run_dir,
        "summary_dir": summary_dir,
        "summary_path": summary_path,
        "ranked_trials_path": ranked_trials_path,
        "all_trials_path": all_trials_path,
        "best_params_path": best_params_path,
        "search_space_path": search_space_path,
        "metadata_path": metadata_path,
    }

