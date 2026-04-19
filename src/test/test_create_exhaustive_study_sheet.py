import math
import pickle
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import src.create_exhaustive_study_sheet as exhaustive_study
import src.utils.reporting.mlflow_helper as mlflow_helper
from src.create_exhaustive_study_sheet import (
    discover_best_configs_dict_paths,
    load_and_merge_best_configs_dicts,
    save_best_configs_dict_artifacts,
    save_pickled_sheet_results,
    select_best_config_yaml,
    store_best_configs_dict_entry,
    simulate_cross_val_summary,
)
import src.utils.toolkit.naming as naming
from src.utils.toolkit.naming import (
    build_best_configs_dict_filename,
    build_sheet_results_pickle_filename,
)
from src.schemas.dataclasses import SheetResult
from src.settings import EXCEL, PATHS


MEAN_ATTRS = [
    "final_model_val_mba_mean",
    "final_model_test_mba_mean",
    "final_model_val_auroc_mean",
    "final_model_test_auroc_mean",
    "best_model_val_mba_mean",
    "best_model_test_mba_mean",
    "best_model_val_auroc_mean",
    "best_model_test_auroc_mean",
]

STD_ATTRS = [
    "final_model_val_mba_std",
    "final_model_test_mba_std",
    "final_model_val_auroc_std",
    "final_model_test_auroc_std",
    "best_model_val_mba_std",
    "best_model_test_mba_std",
    "best_model_val_auroc_std",
    "best_model_test_auroc_std",
]


def _build_args(
    *,
    dataset_v: str = "uk",
    model_type: str = "test_cnn",
    recording_category: str = "poem",
    selected_classes: str = "copd,control",
    norm_mode: str = "norm_off",
    add_demo_data: bool = False,
    test_run: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_v=dataset_v,
        model_type=model_type,
        recording_category=recording_category,
        selected_classes=selected_classes,
        norm_mode=norm_mode,
        add_demo_data=add_demo_data,
        test_run=test_run,
    )


def _build_best_config_payload(
    *,
    model: str = "cnn",
    add_demo_data: bool = False,
    norm_mode: str = "norm_off",
    config_path: str = "C:/tmp/best_trial.yml",
) -> dict:
    demo_data_key = "w_demo_data" if add_demo_data else "no_demo_data"
    return {
        norm_mode: {
            demo_data_key: {
                "uk": {
                    "cont_copd": {
                        "poem": {
                            model: {
                                "best_trial_config_path_list": [config_path],
                            }
                        }
                    }
                }
            }
        }
    }


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _build_logger() -> SimpleNamespace:
    return SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        exception=lambda *args, **kwargs: None,
    )


def _build_sheet_results() -> list[SheetResult]:
    args = _build_args()
    return [simulate_cross_val_summary(args, nan_probability=0.0)]


@pytest.mark.unit
class TestSimulateCrossValSummary:
    def test_returns_sheet_result(self) -> None:
        args = _build_args()
        summary = simulate_cross_val_summary(args, nan_probability=0.0)

        assert isinstance(summary, SheetResult)
        assert summary.dataset == args.dataset_v
        assert summary.recording_category == args.recording_category
        assert summary.lung_conditions == args.selected_classes
        assert summary.model == args.model_type
        assert summary.norm_mode == args.norm_mode

    def test_is_deterministic_for_same_identity(self) -> None:
        args = _build_args()

        summary_first = simulate_cross_val_summary(args, nan_probability=0.0)
        summary_second = simulate_cross_val_summary(args, nan_probability=0.0)

        assert summary_first == summary_second

    def test_changes_when_identity_changes(self) -> None:
        base_args = _build_args()
        changed_args = _build_args(model_type="cnn_lstm")

        base_summary = simulate_cross_val_summary(base_args, nan_probability=0.0)
        changed_summary = simulate_cross_val_summary(changed_args, nan_probability=0.0)

        assert base_summary != changed_summary

    def test_returns_all_nan_when_nan_probability_is_one(self) -> None:
        args = _build_args()
        summary = simulate_cross_val_summary(args, nan_probability=1.0)

        values = [getattr(summary, attr) for attr in [*MEAN_ATTRS, *STD_ATTRS]]
        assert all(math.isnan(value) for value in values)

    def test_non_nan_output_is_within_expected_ranges(self) -> None:
        args = _build_args()
        summary = simulate_cross_val_summary(args, nan_probability=0.0)

        mean_values = [getattr(summary, attr) for attr in MEAN_ATTRS]
        std_values = [getattr(summary, attr) for attr in STD_ATTRS]

        assert all(0.5 <= value <= 0.99 for value in mean_values)
        assert all(0.0 <= value <= 0.25 for value in std_values)
        assert summary.best_model_val_mba_mean >= summary.final_model_val_mba_mean


@pytest.mark.unit
class TestRunExhaustiveCombo:
    @pytest.mark.parametrize(
        ("only_change_random_seeds", "expected_dir_suffix"),
        [
            (True, PATHS.RSEEDS_ID),
            (False, PATHS.CROSS_VAL_ID),
        ],
    )
    def test_uses_mode_specific_output_dir_suffix(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        only_change_random_seeds: bool,
        expected_dir_suffix: str,
    ) -> None:
        args = _build_args(test_run=True)
        args.output_dir = str(tmp_path / "study_root")
        args.config_file = str(tmp_path / "config.yml")
        logger = _build_logger()
        captured_output_dirs: list[str] = []

        monkeypatch.setattr(EXCEL, "TUNING_MODE", EXCEL.TuningMode.NO_TUNING)
        monkeypatch.setattr(
            exhaustive_study.EVALUATION_PROTOCOL,
            "ONLY_CHANGE_RANDOM_SEEDS",
            only_change_random_seeds,
        )

        def _fake_perform_cross_val_run(cmd_args):
            captured_output_dirs.append(cmd_args.output_dir)
            return simulate_cross_val_summary(cmd_args, nan_probability=0.0)

        monkeypatch.setattr(
            exhaustive_study,
            "perform_cross_val_run",
            _fake_perform_cross_val_run,
        )

        combo_tag = "uk_cont_copd_poem_test_cnn"
        result = exhaustive_study.run_exhaustive_combo(
            cmd_args=args,
            logger=logger,
            original_output_dir=args.output_dir,
            norm_mode="norm_off",
            combo_tag=combo_tag,
            dataset_v="uk",
            lung_conditions="copd,control",
            rec_category="poem",
            model="test_cnn",
            loaded_best_configs_dict={},
        )

        assert isinstance(result.sheet_result, SheetResult)
        assert captured_output_dirs == [
            str(Path(args.output_dir) / f"{combo_tag}_{expected_dir_suffix}")
        ]


@pytest.mark.unit
class TestBestConfigDictArtifacts:
    def test_build_best_configs_dict_filename_uses_shared_descriptor_and_configured_suffix(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        args = _build_args(
            dataset_v="uk",
            norm_mode="norm_on",
            add_demo_data=True,
        )
        monkeypatch.setattr(EXCEL, "TUNING_MODE", EXCEL.TuningMode.RUN_OPTUNA_FIRST)
        monkeypatch.setattr(exhaustive_study.EVALUATION_PROTOCOL, "ONLY_CHANGE_RANDOM_SEEDS", False)
        monkeypatch.setattr(exhaustive_study.EVALUATION_PROTOCOL, "TEST_RUN_K_FOLDS", 2)

        filename = build_best_configs_dict_filename(
            cmd_args=args,
            is_test_mode=True,
            run_id="run_123",
        )

        assert filename == (
            "run_123_norm_on_uk_with_tuning_w_demo_data_2fold"
            f"{PATHS.BEST_CONFIGS_DICT_FILENAME_SUFFIX}"
        )

    def test_construct_excel_summary_filename_uses_shared_descriptor(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        args = _build_args(
            dataset_v="uk",
            norm_mode="norm_on",
            add_demo_data=True,
            test_run=True,
        )
        monkeypatch.setattr(EXCEL, "TUNING_MODE", EXCEL.TuningMode.NO_TUNING)
        monkeypatch.setattr(naming.EVALUATION_PROTOCOL, "ONLY_CHANGE_RANDOM_SEEDS", False)
        monkeypatch.setattr(naming.EVALUATION_PROTOCOL, "TEST_RUN_K_FOLDS", 2)

        filename = naming.construct_excel_summary_filename(
            model_variant="final",
            cmd_args=args,
        )

        assert filename == "FINAL_MODEL_norm_on_uk_no_tuning_w_demo_data_2fold_excel_summary.xlsx"

    def test_discover_best_configs_dict_paths_returns_sorted_matches_only(self, tmp_path: Path) -> None:
        version_dir = tmp_path / "v1"
        suffix = PATHS.BEST_CONFIGS_DICT_FILENAME_SUFFIX

        matching_b = version_dir / f"b{suffix}"
        matching_a = version_dir / f"a{suffix}"
        ignored_file = version_dir / "ignore.yml"
        nested_match = version_dir / "nested" / f"nested{suffix}"

        _write_yaml(matching_b, _build_best_config_payload(model="cnn"))
        _write_yaml(matching_a, _build_best_config_payload(model="pann"))
        ignored_file.write_text("ignored", encoding="utf-8")
        _write_yaml(nested_match, _build_best_config_payload(model="lstm"))

        discovered = discover_best_configs_dict_paths(version_dir)

        assert discovered == [matching_a, matching_b]

    def test_discover_best_configs_dict_paths_fails_for_missing_dir(self, tmp_path: Path) -> None:
        missing_dir = tmp_path / "missing"

        with pytest.raises(FileNotFoundError, match="Best-config version directory not found"):
            discover_best_configs_dict_paths(missing_dir)

    def test_discover_best_configs_dict_paths_fails_for_empty_or_nonmatching_dir(self, tmp_path: Path) -> None:
        version_dir = tmp_path / "v1"
        version_dir.mkdir()
        (version_dir / "ignore.yml").write_text("ignored", encoding="utf-8")

        with pytest.raises(FileNotFoundError, match="No best-config dict YAML files found"):
            discover_best_configs_dict_paths(version_dir)

    def test_load_and_merge_best_configs_dicts_fails_on_duplicate_discovered_combo(self, tmp_path: Path) -> None:
        version_dir = tmp_path / "v1"
        suffix = PATHS.BEST_CONFIGS_DICT_FILENAME_SUFFIX
        payload = _build_best_config_payload(model="cnn")

        _write_yaml(version_dir / f"a{suffix}", payload)
        _write_yaml(version_dir / f"b{suffix}", payload)

        discovered = discover_best_configs_dict_paths(version_dir)

        with pytest.raises(ValueError, match="Duplicate precomputed best-config entry"):
            load_and_merge_best_configs_dicts(discovered)

    def test_load_and_merge_best_configs_dicts_defers_invalid_unrelated_combo_until_selection(self, tmp_path: Path) -> None:
        version_dir = tmp_path / "v1"
        suffix = PATHS.BEST_CONFIGS_DICT_FILENAME_SUFFIX
        valid_config_path = tmp_path / "valid_best.yml"
        valid_config_path.write_text("model: test_cnn\n", encoding="utf-8")

        payload = {
            "norm_on": {
                "w_demo_data": {
                    "uk": {
                        "cont_copd": {
                            "poem": {
                                "test_cnn": {
                                    "best_trial_config_path_list": [str(valid_config_path)],
                                },
                                "test_cnn2": {
                                    "best_trial_config_path_list": [],
                                },
                            }
                        }
                    }
                }
            }
        }

        _write_yaml(version_dir / f"a{suffix}", payload)

        discovered = discover_best_configs_dict_paths(version_dir)
        merged = load_and_merge_best_configs_dicts(discovered)

        selected_path = select_best_config_yaml(
            merged,
            norm_mode="norm_on",
            add_demo_data=True,
            dataset_v="uk",
            lung_conditions="copd,control",
            rec_category="poem",
            model="test_cnn",
        )

        assert selected_path == str(valid_config_path)

        with pytest.raises(ValueError, match="best_trial_config_path_list missing/empty for combination"):
            select_best_config_yaml(
                merged,
                norm_mode="norm_on",
                add_demo_data=True,
                dataset_v="uk",
                lung_conditions="copd,control",
                rec_category="poem",
                model="test_cnn2",
            )

    def test_load_and_merge_best_configs_dicts_fails_when_demo_data_level_missing(self, tmp_path: Path) -> None:
        version_dir = tmp_path / "v1"
        suffix = PATHS.BEST_CONFIGS_DICT_FILENAME_SUFFIX

        payload = {
            "norm_on": {
                "uk": {
                    "cont_copd": {
                        "poem": {
                            "test_cnn": {
                                "best_trial_config_path_list": ["C:/tmp/best_trial.yml"],
                            }
                        }
                    }
                }
            }
        }

        _write_yaml(version_dir / f"a{suffix}", payload)

        discovered = discover_best_configs_dict_paths(version_dir)

        with pytest.raises(ValueError, match="missing the demo-data level under"):
            load_and_merge_best_configs_dicts(discovered)

    def test_select_best_config_yaml_uses_demo_data_key(self, tmp_path: Path) -> None:
        version_dir = tmp_path / "v1"
        suffix = PATHS.BEST_CONFIGS_DICT_FILENAME_SUFFIX
        with_demo_path = tmp_path / "with_demo.yml"
        no_demo_path = tmp_path / "no_demo.yml"
        with_demo_path.write_text("model: test_cnn\n", encoding="utf-8")
        no_demo_path.write_text("model: test_cnn\n", encoding="utf-8")

        payload = {
            "norm_on": {
                "w_demo_data": {
                    "uk": {
                        "cont_copd": {
                            "poem": {
                                "test_cnn": {
                                    "best_trial_config_path_list": [str(with_demo_path)],
                                }
                            }
                        }
                    }
                },
                "no_demo_data": {
                    "uk": {
                        "cont_copd": {
                            "poem": {
                                "test_cnn": {
                                    "best_trial_config_path_list": [str(no_demo_path)],
                                }
                            }
                        }
                    }
                },
            }
        }

        _write_yaml(version_dir / f"a{suffix}", payload)

        discovered = discover_best_configs_dict_paths(version_dir)
        merged = load_and_merge_best_configs_dicts(discovered)

        selected_with_demo = select_best_config_yaml(
            merged,
            norm_mode="norm_on",
            add_demo_data=True,
            dataset_v="uk",
            lung_conditions="copd,control",
            rec_category="poem",
            model="test_cnn",
        )
        selected_no_demo = select_best_config_yaml(
            merged,
            norm_mode="norm_on",
            add_demo_data=False,
            dataset_v="uk",
            lung_conditions="copd,control",
            rec_category="poem",
            model="test_cnn",
        )

        assert selected_with_demo == str(with_demo_path)
        assert selected_no_demo == str(no_demo_path)


    def test_save_best_configs_dict_artifacts_writes_to_target_version_dir(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        args = _build_args()
        logger = _build_logger()
        best_configs_root = tmp_path / "best_configs"
        version_dir = "v_target"
        best_params_dict: dict = {}

        store_best_configs_dict_entry(
            best_params_dict,
            norm_mode="norm_off",
            add_demo_data=False,
            dataset_v="uk",
            lung_conditions="copd,control",
            rec_category="poem",
            model="cnn",
            best_trial_config_path_list=["C:/tmp/best_trial.yml"],
            optuna_id="optuna_123",
        )

        monkeypatch.setattr(EXCEL, "TUNING_MODE", EXCEL.TuningMode.RUN_OPTUNA_FIRST)
        monkeypatch.setattr(PATHS, "BEST_CONFIGS_DIR", best_configs_root)
        monkeypatch.setattr(PATHS, "TARGET_BEST_CONFIG_VERSION_DIR", version_dir)

        saved_path = save_best_configs_dict_artifacts(
            best_params_dict=best_params_dict,
            cmd_args=args,
            test_mode=False,
            run_id="run_123",
            logger=logger,
        )

        expected_path = (
            best_configs_root
            / version_dir
            / build_best_configs_dict_filename(
                cmd_args=args,
                is_test_mode=False,
                run_id="run_123",
            )
        )
        saved_payload = yaml.safe_load(expected_path.read_text(encoding="utf-8"))

        assert saved_path == expected_path
        assert expected_path.exists()
        assert expected_path.parent == best_configs_root / version_dir
        assert (
            saved_payload["norm_off"]["no_demo_data"]["uk"]["cont_copd"]["poem"]["cnn"]
            ["best_trial_config_path_list"]
            == ["C:/tmp/best_trial.yml"]
        )

    def test_save_best_configs_dict_artifacts_writes_w_demo_data_branch(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        args = _build_args(add_demo_data=True)
        logger = _build_logger()
        best_configs_root = tmp_path / "best_configs"
        version_dir = "v_target"
        best_params_dict: dict = {}

        store_best_configs_dict_entry(
            best_params_dict,
            norm_mode="norm_on",
            add_demo_data=True,
            dataset_v="uk",
            lung_conditions="copd,control",
            rec_category="poem",
            model="cnn",
            best_trial_config_path_list=["C:/tmp/with_demo.yml"],
            optuna_id="optuna_456",
        )

        monkeypatch.setattr(EXCEL, "TUNING_MODE", EXCEL.TuningMode.RUN_OPTUNA_FIRST)
        monkeypatch.setattr(PATHS, "BEST_CONFIGS_DIR", best_configs_root)
        monkeypatch.setattr(PATHS, "TARGET_BEST_CONFIG_VERSION_DIR", version_dir)

        saved_path = save_best_configs_dict_artifacts(
            best_params_dict=best_params_dict,
            cmd_args=args,
            test_mode=False,
            run_id="run_123",
            logger=logger,
        )

        saved_payload = yaml.safe_load(saved_path.read_text(encoding="utf-8"))

        assert (
            saved_payload["norm_on"]["w_demo_data"]["uk"]["cont_copd"]["poem"]["cnn"]
            ["best_trial_config_path_list"]
            == ["C:/tmp/with_demo.yml"]
        )


@pytest.mark.unit
class TestPickledSheetResultsArtifacts:
    def test_build_sheet_results_pickle_filename_uses_shared_descriptor(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        args = _build_args(
            dataset_v="uk",
            norm_mode="norm_on",
            add_demo_data=True,
            test_run=True,
        )
        monkeypatch.setattr(EXCEL, "TUNING_MODE", EXCEL.TuningMode.NO_TUNING)
        monkeypatch.setattr(exhaustive_study.EVALUATION_PROTOCOL, "ONLY_CHANGE_RANDOM_SEEDS", False)
        monkeypatch.setattr(exhaustive_study.EVALUATION_PROTOCOL, "TEST_RUN_K_FOLDS", 2)

        filename = build_sheet_results_pickle_filename(
            "study_123",
            cmd_args=args,
            test_mode=True,
        )

        assert filename == (
            "study_123_uk_test_cnn_norm_on_no_tuning_w_demo_data_test_run_2fold"
            f"{PATHS.SHEET_RESULTS_PICKLE_SUFFIX}"
        )

    def test_build_sheet_results_pickle_filename_changes_with_variant_inputs(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        base_args = _build_args(test_run=True)
        norm_variant_args = _build_args(norm_mode="norm_on", test_run=True)
        model_variant_args = _build_args(model_type="cnn", test_run=True)
        demo_variant_args = _build_args(add_demo_data=True, test_run=True)
        monkeypatch.setattr(EXCEL, "TUNING_MODE", EXCEL.TuningMode.NO_TUNING)
        monkeypatch.setattr(exhaustive_study.EVALUATION_PROTOCOL, "ONLY_CHANGE_RANDOM_SEEDS", False)
        monkeypatch.setattr(exhaustive_study.EVALUATION_PROTOCOL, "TEST_RUN_K_FOLDS", 2)

        base_filename = build_sheet_results_pickle_filename(
            "study_123",
            cmd_args=base_args,
            test_mode=True,
        )

        assert base_filename != build_sheet_results_pickle_filename(
            "study_123",
            cmd_args=norm_variant_args,
            test_mode=True,
        )
        assert base_filename != build_sheet_results_pickle_filename(
            "study_123",
            cmd_args=model_variant_args,
            test_mode=True,
        )
        assert base_filename != build_sheet_results_pickle_filename(
            "study_123",
            cmd_args=demo_variant_args,
            test_mode=True,
        )

    def test_build_sheet_results_pickle_filename_omits_test_run_token_when_disabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        args = _build_args(
            dataset_v="uk",
            model_type="cnn",
            norm_mode="norm_on",
            add_demo_data=False,
            test_run=False,
        )
        monkeypatch.setattr(EXCEL, "TUNING_MODE", EXCEL.TuningMode.RUN_OPTUNA_FIRST)
        monkeypatch.setattr(exhaustive_study.EVALUATION_PROTOCOL, "ONLY_CHANGE_RANDOM_SEEDS", True)
        monkeypatch.setattr(exhaustive_study.EVALUATION_PROTOCOL, "RANDOM_SEEDS_RUNS", 8)

        filename = build_sheet_results_pickle_filename(
            "excel_study_145",
            cmd_args=args,
            test_mode=False,
        )

        assert filename == (
            "excel_study_145_uk_cnn_norm_on_with_tuning_no_demo_data_8seeds"
            f"{PATHS.SHEET_RESULTS_PICKLE_SUFFIX}"
        )

    def test_build_sheet_results_pickle_filename_uses_test_run_seed_count(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        args = _build_args(
            dataset_v="uk",
            model_type="cnn",
            norm_mode="norm_on",
            add_demo_data=False,
            test_run=True,
        )
        monkeypatch.setattr(EXCEL, "TUNING_MODE", EXCEL.TuningMode.RUN_OPTUNA_FIRST)
        monkeypatch.setattr(exhaustive_study.EVALUATION_PROTOCOL, "ONLY_CHANGE_RANDOM_SEEDS", True)
        monkeypatch.setattr(exhaustive_study.EVALUATION_PROTOCOL, "RANDOM_SEEDS_RUNS", 8)
        monkeypatch.setattr(exhaustive_study.EVALUATION_PROTOCOL, "TEST_RUN_RANDOM_SEEDS_RUNS", 2)

        filename = build_sheet_results_pickle_filename(
            "excel_study_145",
            cmd_args=args,
            test_mode=True,
        )

        assert filename == (
            "excel_study_145_uk_cnn_norm_on_with_tuning_no_demo_data_test_run_2seeds"
            f"{PATHS.SHEET_RESULTS_PICKLE_SUFFIX}"
        )

    def test_save_pickled_sheet_results_writes_pickle_to_target_dir(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        args = _build_args(
            dataset_v="uk",
            norm_mode="norm_on",
            add_demo_data=True,
            test_run=True,
        )
        logger = _build_logger()
        target_dir = tmp_path / "pickled_sheet_results" / "v_target"
        sheet_results = [simulate_cross_val_summary(args, nan_probability=0.0)]

        monkeypatch.setattr(PATHS, "TARGET_SHEET_RESULTS_DIR", target_dir)
        monkeypatch.setattr(EXCEL, "TUNING_MODE", EXCEL.TuningMode.NO_TUNING)
        monkeypatch.setattr(exhaustive_study.EVALUATION_PROTOCOL, "ONLY_CHANGE_RANDOM_SEEDS", False)
        monkeypatch.setattr(exhaustive_study.EVALUATION_PROTOCOL, "TEST_RUN_K_FOLDS", 2)

        pickle_path = save_pickled_sheet_results(
            all_results=sheet_results,
            study_id="study_123",
            cmd_args=args,
            test_mode=True,
            logger=logger,
        )

        expected_path = (
            target_dir
            / (
                "study_123_uk_test_cnn_norm_on_no_tuning_w_demo_data_test_run_2fold"
                f"{PATHS.SHEET_RESULTS_PICKLE_SUFFIX}"
            )
        )
        assert pickle_path == expected_path
        assert expected_path.exists()

        with expected_path.open("rb") as f:
            loaded_results = pickle.load(f)

        assert isinstance(loaded_results, list)
        assert loaded_results == sheet_results
        assert isinstance(loaded_results[0], SheetResult)

    def test_save_pickled_sheet_results_returns_none_on_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        args = _build_args()
        logger = _build_logger()
        target_dir = tmp_path / "pickled_sheet_results" / "v_target"
        sheet_results = _build_sheet_results()

        monkeypatch.setattr(PATHS, "TARGET_SHEET_RESULTS_DIR", target_dir)

        def _raise_pickle_error(*args, **kwargs):
            raise OSError("pickle write failed")

        monkeypatch.setattr(exhaustive_study.pickle, "dump", _raise_pickle_error)

        pickle_path = save_pickled_sheet_results(
            all_results=sheet_results,
            study_id="study_123",
            cmd_args=args,
            logger=logger,
        )

        assert pickle_path is None


@pytest.mark.unit
class TestStudySheetMlflowLogging:
    def test_logs_sheet_results_pickle_when_present(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        logger = _build_logger()
        logged_artifacts: list[str] = []

        yaml_path = tmp_path / "best_configs.yml"
        final_xlsx_path = tmp_path / "final.xlsx"
        best_xlsx_path = tmp_path / "best.xlsx"
        pickle_path = tmp_path / f"study_123{PATHS.SHEET_RESULTS_PICKLE_SUFFIX}"

        yaml_path.write_text("meta: 1", encoding="utf-8")
        final_xlsx_path.write_text("final", encoding="utf-8")
        best_xlsx_path.write_text("best", encoding="utf-8")
        pickle_path.write_bytes(b"pickle")

        monkeypatch.setattr(mlflow_helper.mlflow, "set_experiment", lambda *args, **kwargs: None)
        monkeypatch.setattr(mlflow_helper.mlflow, "active_run", lambda: None)
        monkeypatch.setattr(
            mlflow_helper.mlflow,
            "start_run",
            lambda **kwargs: nullcontext(),
        )
        monkeypatch.setattr(mlflow_helper.mlflow, "set_tag", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            mlflow_helper.mlflow,
            "log_artifact",
            lambda path: logged_artifacts.append(path),
        )

        mlflow_helper.log_study_sheet_artifacts_to_mlflow(
            logger=logger,
            experiment_name="test_experiment",
            study_id="study_123",
            yaml_save_path=yaml_path,
            final_model_out_xlsx=str(final_xlsx_path),
            best_model_out_xlsx=str(best_xlsx_path),
            sheet_results_pickle_path=pickle_path,
        )

        assert str(yaml_path) in logged_artifacts
        assert str(final_xlsx_path) in logged_artifacts
        assert str(best_xlsx_path) in logged_artifacts
        assert str(pickle_path) in logged_artifacts



