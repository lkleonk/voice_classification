from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path

from src.schemas.dataclasses import SheetResult
from src.settings import FILES, PATHS
from src.utils.reporting.excel_study import create_and_save_excel_file
from src.utils.reporting.logger_setup import setup_run_logger
from src.utils.toolkit.time_utils import get_year_month_day_hour_minute_second


def discover_sheet_results_pickle_paths(load_dir: str | Path) -> list[Path]:
    """Return sorted exhaustive-study pickle artifacts from the configured load dir."""
    load_dir = Path(load_dir)
    if not load_dir.exists():
        raise ValueError(f"Sheet-results load directory does not exist: {load_dir}")
    if not load_dir.is_dir():
        raise ValueError(f"Sheet-results load path is not a directory: {load_dir}")

    pickle_paths = sorted(
        path for path in load_dir.iterdir() if path.name.endswith(PATHS.SHEET_RESULTS_PICKLE_SUFFIX)
    )
    if not pickle_paths:
        raise ValueError(f"No '*{PATHS.SHEET_RESULTS_PICKLE_SUFFIX}' files found in {load_dir}")
    return pickle_paths


def load_pickled_sheet_results(pickle_path: str | Path) -> list[SheetResult]:
    """Load a single pickled exhaustive-study payload and validate its schema."""
    pickle_path = Path(pickle_path)

    with pickle_path.open("rb") as file_handle:
        loaded_payload = pickle.load(file_handle)

    if not isinstance(loaded_payload, list):
        raise ValueError(
            f"Expected pickled payload in {pickle_path} to be a list[SheetResult], "
            f"got {type(loaded_payload).__name__}"
        )
    if not loaded_payload:
        raise ValueError(f"Pickled sheet results list is empty in {pickle_path}")

    invalid_index = next(
        (index for index, item in enumerate(loaded_payload) if not isinstance(item, SheetResult)),
        None,
    )
    if invalid_index is not None:
        raise ValueError(
            f"Expected every pickled entry in {pickle_path} to be a SheetResult, "
            f"but entry {invalid_index} is {type(loaded_payload[invalid_index]).__name__}"
        )

    return loaded_payload


def load_sheet_results_from_dir(load_dir: str | Path) -> list[SheetResult]:
    """Load and concatenate all exhaustive-study pickles from one directory."""
    all_results: list[SheetResult] = []
    for pickle_path in discover_sheet_results_pickle_paths(load_dir):
        all_results.extend(load_pickled_sheet_results(pickle_path))
    return all_results


def filter_sheet_results_by_target_modes(
    *,
    sheet_results: list[SheetResult],
    norm_mode: str,
    add_demo_data: bool,
) -> tuple[list[SheetResult], int, int]:
    """Keep only rows that match the requested norm/demo target combination."""
    matching_results: list[SheetResult] = []
    skipped_norm_mode_count = 0
    skipped_demo_mode_count = 0

    for result in sheet_results:
        add_demo_data_value, add_demo_data_status = result.get_add_demographic_data_info()
        if add_demo_data_status != "present":
            raise ValueError(
                "Expected add_demographic_data_var to be present for every loaded SheetResult"
            )
        if result.norm_mode != norm_mode:
            skipped_norm_mode_count += 1
            continue
        if add_demo_data_value != add_demo_data:
            skipped_demo_mode_count += 1
            continue
        matching_results.append(result)

    if not matching_results:
        demo_token = "w_demo_data" if add_demo_data else "no_demo_data"
        raise ValueError(
            "No loaded SheetResult entries matched the requested filter "
            f"norm_mode={norm_mode}, demo_mode={demo_token}"
        )
    return matching_results, skipped_norm_mode_count, skipped_demo_mode_count


def build_excel_filename(
    *,
    model_variant: str,
    norm_mode: str,
    add_demo_data: bool,
    export_timestamp: str,
) -> str:
    """Build the combined rebuild workbook filename from loaded result metadata."""
    if model_variant not in ("final", "best"):
        raise ValueError('model_variant must be "final" or "best"')

    demo_token = "w_demo_data" if add_demo_data else "no_demo_data"
    return (
        f"{model_variant.upper()}_MODEL_{norm_mode}_{demo_token}_"
        f"pickled_sheet_results_excel_summary_{export_timestamp}.xlsx"
    )


def apply_duplicate_model_labels(sheet_results: list[SheetResult]) -> list[SheetResult]:
    """Rename later duplicate logical rows so the Excel pivot keeps all entries."""
    duplicate_counts: dict[tuple[str, str, str, str], int] = {}
    prepared_results: list[SheetResult] = []

    for result in sheet_results:
        identity = (
            result.dataset,
            result.lung_conditions,
            result.recording_category,
            result.model,
        )
        duplicate_counts[identity] = duplicate_counts.get(identity, 0) + 1
        duplicate_index = duplicate_counts[identity]

        if duplicate_index == 1:
            prepared_results.append(result)
            continue

        prepared_results.append(
            replace(
                result,
                model=f"Duplicate {duplicate_index}: {result.model}",
            )
        )

    return prepared_results


def save_excel_summaries_from_sheet_results(
    *,
    sheet_results: list[SheetResult],
    output_dir: str | Path,
    norm_mode: str,
    add_demo_data: bool,
    export_timestamp: str,
) -> tuple[Path, Path]:
    """Recreate both Excel study-sheet variants from pre-filtered pickled results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared_results = apply_duplicate_model_labels(sheet_results)

    final_model_out_xlsx = create_and_save_excel_file(
        prepared_results,
        output_dir,
        filename=build_excel_filename(
            model_variant="final",
            norm_mode=norm_mode,
            add_demo_data=add_demo_data,
            export_timestamp=export_timestamp,
        ),
        model_variant="final",
        best_configs_dict_path=None,
        add_demo_data_column=False,
    )
    best_model_out_xlsx = create_and_save_excel_file(
        prepared_results,
        output_dir,
        filename=build_excel_filename(
            model_variant="best",
            norm_mode=norm_mode,
            add_demo_data=add_demo_data,
            export_timestamp=export_timestamp,
        ),
        model_variant="best",
        best_configs_dict_path=None,
        add_demo_data_column=False,
    )
    return final_model_out_xlsx, best_model_out_xlsx


def main(cmd_args) -> int:
    output_dir = Path(cmd_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_run_logger(__name__, str(output_dir / FILES.LOGGER_FILENAME))

    load_dir = Path(PATHS.LOAD_SHEET_RESULTS_DIR)
    logger.info("Loading pickled sheet results from: %s", load_dir)
    logger.info(
        "Filtering loaded sheet results for norm_mode=%s and add_demo_data=%s",
        cmd_args.norm_mode,
        cmd_args.add_demo_data,
    )
    export_timestamp = get_year_month_day_hour_minute_second()

    sheet_results = load_sheet_results_from_dir(load_dir)
    logger.info("Loaded %s SheetResult rows before filtering", len(sheet_results))
    filtered_sheet_results, skipped_norm_mode_count, skipped_demo_mode_count = (
        filter_sheet_results_by_target_modes(
            sheet_results=sheet_results,
            norm_mode=cmd_args.norm_mode,
            add_demo_data=cmd_args.add_demo_data,
        )
    )
    logger.info(
        "Keeping %s rows after filtering; skipped %s for norm_mode mismatch and %s for demo_mode mismatch",
        len(filtered_sheet_results),
        skipped_norm_mode_count,
        skipped_demo_mode_count,
    )

    final_model_out_xlsx, best_model_out_xlsx = save_excel_summaries_from_sheet_results(
        sheet_results=filtered_sheet_results,
        output_dir=output_dir,
        norm_mode=cmd_args.norm_mode,
        add_demo_data=cmd_args.add_demo_data,
        export_timestamp=export_timestamp,
    )

    logger.info("Saved final-model Excel summary to: %s", final_model_out_xlsx)
    logger.info("Saved best-model Excel summary to: %s", best_model_out_xlsx)
    return 0
