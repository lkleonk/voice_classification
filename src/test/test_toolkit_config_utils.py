import logging

import pytest

from src.model.pretrained.pretrained_models_params import overwrite_config_values
from src.utils.toolkit.config_utils import (
    compare_search_space_and_config,
    update_flat_dict_strict,
    validate_and_clean_tuning_config,
)


@pytest.mark.unit
class TestToolkitConfigUtils:
    def test_validate_and_clean_tuning_config_drops_spectrogram_keys_for_tabular_mlp(self) -> None:
        tuning_config = {
            "tabular_mlp.final_dropout": [0.1, 0.3],
            "optimizer.lr": [1e-3, 1e-4],
            "spectrogram.n_mels": [64, 128],
            "spectrogram.norm": ["minmax_0_1"],
        }

        cleaned = validate_and_clean_tuning_config(
            tuning_config=tuning_config,
            model_type="tabular_mlp",
            dataset_v="uk",
            logger=logging.getLogger(__name__),
            strict=True,
        )

        assert "tabular_mlp.final_dropout" in cleaned
        assert "optimizer.lr" in cleaned
        assert "spectrogram.n_mels" not in cleaned
        assert "spectrogram.norm" not in cleaned

    def test_update_flat_dict_strict_updates_known_keys_without_mutating_base(self) -> None:
        base = {"a": 1, "b": {"nested": 2}}
        target_values = {"a": 99}

        updated = update_flat_dict_strict(base, target_values)

        assert updated["a"] == 99
        assert base["a"] == 1
        assert updated is not base

    def test_update_flat_dict_strict_raises_for_unknown_key(self) -> None:
        base = {"a": 1}

        with pytest.raises(KeyError, match="unknown key 'missing'"):
            update_flat_dict_strict(base, {"missing": 2})

    def test_compare_search_space_and_config_passes_for_known_keys(self) -> None:
        config = {"lr": 1e-3, "batch": 16, "dropout": 0.1}
        search_space = {"lr": [1e-3, 1e-4], "batch": [16, 32]}

        compare_search_space_and_config(config, search_space)

    def test_compare_search_space_and_config_raises_for_missing_keys(self) -> None:
        config = {"lr": 1e-3}
        search_space = {"lr": [1e-3], "batch": [16], "dropout": [0.1]}

        with pytest.raises(ValueError) as exc_info:
            compare_search_space_and_config(config, search_space)

        message = str(exc_info.value)
        assert "'batch'" in message
        assert "'dropout'" in message

    def test_overwrite_config_values_sets_expected_vggish_fields(self) -> None:
        config = {
            "spectrogram.add_chromagram": True,
            "spectrogram.duration": 10,
            "spectrogram.sample_rate": 32000,
            "spectrogram.window_length": 1024,
            "spectrogram.hop_length": 320,
            "spectrogram.n_mels": 128,
            "spectrogram.fmin": 0,
            "spectrogram.fmax": 15000,
            "spectrogram.norm": "minmax_0_1",
        }

        updated = overwrite_config_values(config, "vggish")

        assert updated["spectrogram.add_chromagram"] is False
        assert updated["spectrogram.duration"] == 0.95
        assert updated["spectrogram.sample_rate"] == 16000
        assert updated["spectrogram.window_length"] == 400
        assert updated["spectrogram.hop_length"] == 160
        assert updated["spectrogram.n_mels"] == 64
        assert updated["spectrogram.fmin"] == 125
        assert updated["spectrogram.fmax"] == 7500
        assert updated["spectrogram.norm"] == "audioset_mean_std"


