from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from src.model.pretrained.pretrained_models_params import get_pretrained_model_spec
from src.settings import PATHS

try:
    from transformers import ASTModel
except ImportError as e:
    raise ImportError(
        "Please `pip install transformers` (and ideally accelerate). "
        "ASTEmbedder below uses Hugging Face ASTModel in local-only mode."
    ) from e

_AST_SPEC = get_pretrained_model_spec("ast")


class ASTEmbedder(nn.Module):
    """
    AST embedder that:
    - takes raw waveform
    - computes log-mel features
    - returns a [B, embed_dim] embedding
    """

    def __init__(
        self,
        device: str = "cuda",
        *,
        model_dir: Path = PATHS.AST_VARIANT_DIR,
        target_sr: int = int(_AST_SPEC["spectrogram.sample_rate"]),
        n_mels: int = int(_AST_SPEC["spectrogram.n_mels"]),
        n_fft: int = int(_AST_SPEC["spectrogram.n_fft"]),
        window_length: int = int(_AST_SPEC["spectrogram.window_length"]),
        hop_length: int = int(_AST_SPEC["spectrogram.hop_length"]),
        f_min: float = float(_AST_SPEC["spectrogram.fmin"]),
        f_max: float = float(_AST_SPEC["spectrogram.fmax"]),
        input_tdim: int = int(_AST_SPEC["ast.input_tdim"]),
        eps: float = 1e-10,
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.target_sr = target_sr
        self.eps = eps
        self.input_tdim = input_tdim
        self.spec_mean = float(_AST_SPEC["spectrogram.mean"])
        self.spec_std = float(_AST_SPEC["spectrogram.std"])

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=window_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=2.0,
            normalized=False,
            center=True,
            pad_mode="reflect",
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
            stype="power",
            top_db=None,
        )

        self._resampler: Optional[torchaudio.transforms.Resample] = None

        self.model = ASTModel.from_pretrained(
            PATHS.AST_HF_VARIANT_IDENTIFIER,
            local_files_only=False,
        )
        checkpoint_max_length = int(getattr(self.model.config, "max_length", self.input_tdim))
        if checkpoint_max_length != self.input_tdim:
            raise ValueError(
                "AST spec and checkpoint disagree on input_tdim: "
                f"spec={self.input_tdim}, checkpoint={checkpoint_max_length}"
            )

        self.embed_dim = getattr(self.model.config, "hidden_size", 768)
        self.to(self.device)

    @staticmethod
    def _fix_tdim(feats: torch.Tensor, tdim: int) -> torch.Tensor:
        """
        Ensure log-mel features have the fixed number of frames required by AST.

        feats: [B, T, mel_bins]
        """
        time_frames = feats.shape[1]
        if time_frames == tdim:
            return feats
        if time_frames > tdim:
            start = (time_frames - tdim) // 2
            return feats[:, start:start + tdim, :]

        pad_frames = tdim - time_frames
        return F.pad(feats, (0, 0, 0, pad_frames))

    def _ensure_resampler(self, input_sr: int) -> None:
        if input_sr == self.target_sr:
            self._resampler = None
            return

        if (
            self._resampler is None
            or self._resampler.orig_freq != input_sr
            or self._resampler.new_freq != self.target_sr
        ):
            self._resampler = torchaudio.transforms.Resample(
                orig_freq=input_sr,
                new_freq=self.target_sr,
            )

    def forward(
        self,
        wave: torch.Tensor,
        *,
        input_sr: int = int(_AST_SPEC["spectrogram.sample_rate"]),
    ) -> torch.Tensor:
        """
        wave: [B, T] raw waveform
        returns: [B, embed_dim]
        """
        wave = wave.to(self.device)

        self._ensure_resampler(input_sr)
        if self._resampler is not None:
            wave = self._resampler(wave)

        # [B, n_mels, frames]
        mel = self.mel(wave)
        mel_db = self.amplitude_to_db(torch.clamp(mel, min=self.eps))
        mel_db = (mel_db - self.spec_mean) / self.spec_std

        # HF AST expects [B, time, mel_bins]
        feats = mel_db.transpose(1, 2)
        feats = self._fix_tdim(feats, self.input_tdim)

        out = self.model(input_values=feats)

        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            emb = out.pooler_output
        else:
            emb = out.last_hidden_state[:, 0, :]

        return emb


