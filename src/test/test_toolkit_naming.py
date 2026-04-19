from types import SimpleNamespace

import pytest

from src.utils.toolkit.naming import (
    _canonicalize_csv_tokens,
    build_best_config_filename,
    build_cv_run_name,
    build_study_run_name,
    build_training_run_name,
    get_norm_str,
)


@pytest.mark.unit
class TestToolkitNaming:
    def test_get_norm_str_detects_expected_tokens(self) -> None:
        assert get_norm_str("norm_off") == "norm_off"
        assert get_norm_str("norm_on") == "norm_on"
        assert get_norm_str("something_else") == "unknown"

    def test_canonicalize_csv_tokens_handles_known_cases(self) -> None:
        assert _canonicalize_csv_tokens("control,copd") == "cont_copd"
        assert _canonicalize_csv_tokens("copd,control") == "cont_copd"
        assert _canonicalize_csv_tokens("control,copd,fibrosis") == "cont_copd_fibr"
        assert _canonicalize_csv_tokens("fibrosis,copd") == "copd_fibr"

    def test_build_training_run_name_is_deterministic_with_fixed_run_id(self) -> None:
        args = SimpleNamespace(
            output_dir="outputs/exp_norm_on",
            norm_mode="norm_on",
            dataset_v="uk",
            selected_classes="control,copd",
            test_run=True,
            model_type="test_cnn",
            recording_category="poem",
        )

        run_name = build_training_run_name(args, pretrained=True, run_id=123)

        assert run_name == "123_norm_on_cont_copd_uk_TEST_run_pretrained_test_cnn_poem"

    def test_build_cv_run_name_is_deterministic_with_fixed_run_id(self) -> None:
        args = SimpleNamespace(
            output_dir="outputs/exp_norm_off",
            norm_mode="norm_off",
            dataset_v="uk",
            selected_classes="control,copd",
            recording_category="a,b",
            model_type="cnn_lstm",
            test_run=False,
        )

        run_name = build_cv_run_name(args, effective_k=5, run_id=333)

        assert run_name == "CV_333_uk_control_copd_a_b_cnn_lstm_norm_off_k5_REAL"

    def test_build_study_run_name_and_best_config_filename(self) -> None:
        args = SimpleNamespace(
            output_dir="outputs/exp_norm_off",
            norm_mode="norm_off",
            dataset_v="uk",
            selected_classes="control,copd",
            recording_category="o,a",
            model_type="passt",
            test_run=False,
        )

        study_name = build_study_run_name(args, test_run=True, study_id="study_007")
        best_cfg_name = build_best_config_filename(
            model_type="passt",
            dataset_v="uk",
            selected_classes="control,copd",
            recording_category="o,a",
            timestamp="20260218_101500",
            rank=1,
            study_id="study_007",
        )

        assert study_name == "study_007_uk_control_copd_o_a_passt_norm_off_TEST"
        assert best_cfg_name == "study_007_best_config_passt_uk_cont_copd_a_o_20260218_101500_rank_1.yaml"


