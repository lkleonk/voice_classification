import random
from pathlib import Path

import numpy as np
import pytest
import torch

import src.utils.data_preprocessing.spectrogram_file_dataset as spectrogram_file_dataset_module
from src.settings import SPECTROGRAM_NORM
from src.utils.data_preprocessing.spectrogram_file_dataset import (
    SpectrogramFileDataset,
    normalize_spectrogram_values,
)

TEST_RUN_DUMMY_FILES_DIR = Path(__file__).resolve().parent / "test_run_dummy_files"


@pytest.fixture
def dummy_wav_paths() -> list[str]:
    return [
        str(TEST_RUN_DUMMY_FILES_DIR / "100_a.wav"),
        str(TEST_RUN_DUMMY_FILES_DIR / "101_a.wav"),
        str(TEST_RUN_DUMMY_FILES_DIR / "102_a.wav"),
    ]


@pytest.fixture
def labels() -> list[int]:
    return [0, 1, 0]


@pytest.fixture
def demographics_tensor() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.5, 1.0],
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def base_config() -> dict[str, float | int | bool | str]:
    return {
        "spectrogram.sample_rate": 32000,
        "spectrogram.n_mels": 32,
        "spectrogram.n_fft": 512,
        "spectrogram.window_length": 512,
        "spectrogram.hop_length": 160,
        "spectrogram.fmin": 0,
        "spectrogram.fmax": 8000,
        "spectrogram.duration": 1.0,
        "spectrogram.trim_modality": "end",
        "spectrogram.add_chromagram": False,
        "spectrogram.add_mfcc": False,
        "spectrogram.add_delta_mfcc": False,
        "spectrogram.norm": "minmax_0_1",
        "spectrogram.max_noise_std": 0.0,
        "waveform.norm": "minmax_0_1",
    }


@pytest.mark.unit
class TestNormalizeSpectrogramValues:
    def test_minmax_0_1_output_is_bounded(self) -> None:
        values = np.array([-2.0, 0.0, 2.0], dtype=np.float32)

        normalized = normalize_spectrogram_values(values, norm_mode="minmax_0_1")

        assert normalized.min() == pytest.approx(0.0)
        assert normalized.max() == pytest.approx(1.0)
        assert np.all(normalized >= 0.0)
        assert np.all(normalized <= 1.0)

    def test_minmax_minus_1_1_output_is_bounded(self) -> None:
        values = np.array([-2.0, 0.0, 2.0], dtype=np.float32)

        normalized = normalize_spectrogram_values(values, norm_mode="minmax_-1_1")

        assert normalized.min() == pytest.approx(-1.0)
        assert normalized.max() == pytest.approx(1.0)
        assert np.all(normalized >= -1.0)
        assert np.all(normalized <= 1.0)

    def test_audioset_mean_std_uses_settings_constants(self) -> None:
        assert SPECTROGRAM_NORM.AUDIOSET_FBANK_MEAN == pytest.approx(15.41663)
        assert SPECTROGRAM_NORM.AUDIOSET_FBANK_STD == pytest.approx(6.55582)

        values = np.array(
            [
                SPECTROGRAM_NORM.AUDIOSET_FBANK_MEAN,
                SPECTROGRAM_NORM.AUDIOSET_FBANK_MEAN + SPECTROGRAM_NORM.AUDIOSET_FBANK_STD,
            ],
            dtype=np.float32,
        )

        normalized = normalize_spectrogram_values(values, norm_mode="audioset_mean_std")

        assert normalized[0] == pytest.approx(0.0)
        assert normalized[1] == pytest.approx(1.0)

    def test_unknown_norm_mode_raises_value_error(self) -> None:
        values = np.array([0.0, 1.0], dtype=np.float32)

        with pytest.raises(ValueError, match="norm_mode must be"):
            normalize_spectrogram_values(values, norm_mode="unknown_mode")


@pytest.mark.unit
class TestSpectrogramFileDataset:
    def test_getitem_returns_spectrogram_tensor_with_expected_shape(
        self,
        dummy_wav_paths: list[str],
        labels: list[int],
        demographics_tensor: torch.Tensor,
        base_config: dict[str, float | int | bool | str],
    ) -> None:
        dataset = SpectrogramFileDataset(
            filepaths=dummy_wav_paths,
            labels=labels,
            demographics_tensor=demographics_tensor,
            config=base_config,
        )

        tensor, label_tensor, demo_tensor = dataset[0]

        assert tensor.dtype == torch.float32
        assert tensor.ndim == 3
        assert tensor.shape[0] == 1
        assert tensor.shape[1] == base_config["spectrogram.n_mels"]
        assert tensor.shape[2] > 0
        assert label_tensor.dtype == torch.long
        assert label_tensor.item() == labels[0]
        assert torch.equal(demo_tensor, demographics_tensor[0])

    def test_return_dummy_audio_returns_zeros_without_disk_read(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dummy_wav_paths: list[str],
        labels: list[int],
        demographics_tensor: torch.Tensor,
        base_config: dict[str, float | int | bool | str],
    ) -> None:
        def _fail_if_called(*args, **kwargs):
            raise AssertionError("librosa.load should not be called for dummy audio mode")

        monkeypatch.setattr(spectrogram_file_dataset_module.librosa, "load", _fail_if_called)

        dataset = SpectrogramFileDataset(
            filepaths=dummy_wav_paths,
            labels=labels,
            demographics_tensor=demographics_tensor,
            config=base_config,
            return_dummy_audio_bool=True,
        )

        tensor, label_tensor, demo_tensor = dataset[1]

        assert torch.equal(tensor, torch.zeros(1, dtype=torch.float32))
        assert label_tensor.dtype == torch.long
        assert label_tensor.item() == labels[1]
        assert torch.equal(demo_tensor, demographics_tensor[1])

    def test_return_only_audio_returns_1d_waveform(
        self,
        dummy_wav_paths: list[str],
        labels: list[int],
        demographics_tensor: torch.Tensor,
        base_config: dict[str, float | int | bool | str],
    ) -> None:
        dataset = SpectrogramFileDataset(
            filepaths=dummy_wav_paths,
            labels=labels,
            demographics_tensor=demographics_tensor,
            config=base_config,
            return_only_audio_bool=True,
        )

        tensor, label_tensor, demo_tensor = dataset[0]
        target_length = int(
            base_config["spectrogram.duration"] * base_config["spectrogram.sample_rate"]
        )

        assert tensor.dtype == torch.float32
        assert tensor.ndim == 1
        assert tensor.shape[0] == target_length
        assert 0.0 <= tensor.min().item()
        assert tensor.max().item() <= 1.0
        assert label_tensor.dtype == torch.long
        assert torch.equal(demo_tensor, demographics_tensor[0])

    def test_noise_augmentation_changes_output(
        self,
        dummy_wav_paths: list[str],
        base_config: dict[str, float | int | bool | str],
    ) -> None:
        no_noise_config = dict(base_config)
        with_noise_config = dict(base_config)
        with_noise_config["spectrogram.max_noise_std"] = 0.05

        labels = [0]
        demographics_tensor = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

        no_noise_dataset = SpectrogramFileDataset(
            filepaths=[dummy_wav_paths[0]],
            labels=labels,
            demographics_tensor=demographics_tensor,
            config=no_noise_config,
        )
        noisy_dataset = SpectrogramFileDataset(
            filepaths=[dummy_wav_paths[0]],
            labels=labels,
            demographics_tensor=demographics_tensor,
            config=with_noise_config,
        )

        clean_tensor, _, _ = no_noise_dataset[0]

        random.seed(123)
        np.random.seed(123)
        noisy_tensor, _, _ = noisy_dataset[0]

        assert clean_tensor.shape == noisy_tensor.shape
        assert not torch.allclose(clean_tensor, noisy_tensor)

    def test_constructor_raises_for_mismatched_filepaths_and_labels(
        self,
        dummy_wav_paths: list[str],
        demographics_tensor: torch.Tensor,
        base_config: dict[str, float | int | bool | str],
    ) -> None:
        with pytest.raises(ValueError, match="Filepaths length .* labels length"):
            SpectrogramFileDataset(
                filepaths=dummy_wav_paths[:1],
                labels=[0, 1],
                demographics_tensor=demographics_tensor[:1],
                config=base_config,
            )

