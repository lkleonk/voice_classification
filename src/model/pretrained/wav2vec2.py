from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchaudio

from src.model.pretrained.pretrained_models_params import get_pretrained_model_spec
from src.settings import PATHS
from src.model.pretrained.cache_env import configure_pretrained_cache_env





try:
    from transformers import Wav2Vec2Model
except ImportError as e:
    raise ImportError(
        "Please `pip install transformers` (and ideally accelerate). "
        "Wav2Vec2Embedder uses Hugging Face Wav2Vec2Model in local-only mode."
    ) from e

_WAV2VEC2_SPEC = get_pretrained_model_spec("wav2vec2.0")


class Wav2Vec2Embedder(nn.Module):
    """
    Wav2Vec2 embedder that:
    - takes raw waveform (expects 16 kHz by default)
    - loads weights only from work/model_weights (no home cache downloads)
    - returns pooled clip embeddings:
      - mean pooling: [B, H]
      - mean+std pooling: [B, 2H] when append_std=True
    """

    def __init__(
        self,
        device: str = "cuda",
        *,
        model_id: str,
        layer_index: Optional[int] = None,
        work_weights_root: Optional[str] = None,
        model_dirname: str = str(PATHS.WAV2VEC2_WEIGHTS_DIR.name),
        target_sr: int = int(_WAV2VEC2_SPEC["spectrogram.sample_rate"]),
        append_std: bool = False,
    ) -> None:
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.target_sr = target_sr
        self.append_std = append_std
        self.layer_index = layer_index

        if work_weights_root is None:
            work_weights_root = str(PATHS.MODEL_WEIGHTS_DIR)

        configure_pretrained_cache_env()

        self.model_path = str(Path(work_weights_root) / model_dirname)
        safe_name = model_id.replace("/", "__")
        mp = Path(self.model_path) / safe_name

        if not mp.exists():
            mp.mkdir(parents=True, exist_ok=True)

            # HuggingFace model id to download from (matches your dirname)
            model_id = model_id

            # 1) Download into the project-local directory (NOT ~/.cache)
            model = Wav2Vec2Model.from_pretrained(
                model_id,
                cache_dir=str(mp),
                local_files_only=False,
                use_safetensors=True,      # ← force safetensors again
            )

            # 2) Save a clean HF-style folder right here (config.json + weights)
            model.save_pretrained(str(mp))

        # Always load from the local folder afterward (offline-safe)
        self.model = Wav2Vec2Model.from_pretrained(
            str(mp),
            local_files_only=True,
            use_safetensors=True,      # ← force safetensors again
        )
        self.hidden_size = int(getattr(self.model.config, "hidden_size", 768))
        self.num_hidden_layers = int(getattr(self.model.config, "num_hidden_layers", 0))
        max_hidden_state_index = self.num_hidden_layers
        if self.layer_index is None:
            self.layer_index = max_hidden_state_index
        else:
            self.layer_index = int(self.layer_index)

        if self.layer_index < 0 or self.layer_index > max_hidden_state_index:
            raise ValueError(
                f"Invalid wav2vec2 layer_index={self.layer_index}. "
                f"Allowed range is 0..{max_hidden_state_index} "
                "(0=embedding output, last index=final transformer layer)."
            )

        self.embed_dim = self.hidden_size * (2 if self.append_std else 1)

        self._resampler: Optional[torchaudio.transforms.Resample] = None
        self.to(self.device)

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

    def _pool_embeddings(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # hidden: [B, T, H]
        if attention_mask is None:
            mean_emb = hidden.mean(dim=1)
            if not self.append_std:
                return mean_emb
            std_emb = hidden.std(dim=1, unbiased=False)
            return torch.cat([mean_emb, std_emb], dim=1)

        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # [B, T, 1]
        denom = mask.sum(dim=1).clamp_min(1.0)  # [B, 1]
        mean_emb = (hidden * mask).sum(dim=1) / denom
        if not self.append_std:
            return mean_emb

        var_emb = ((hidden - mean_emb.unsqueeze(1)) ** 2 * mask).sum(dim=1) / denom
        std_emb = torch.sqrt(var_emb.clamp_min(1e-12))
        return torch.cat([mean_emb, std_emb], dim=1)

    @torch.no_grad()
    def forward(
        self,
        wave: torch.Tensor,
        *,
        input_sr: int = int(_WAV2VEC2_SPEC["spectrogram.sample_rate"]),
    ) -> torch.Tensor:
        """
        wave: [B, T] raw waveform
        returns:
        - [B, H] if append_std=False
        - [B, 2H] if append_std=True
        """
        wave = wave.to(self.device)
        if wave.ndim == 1:
            wave = wave.unsqueeze(0)
        if wave.ndim != 2:
            raise ValueError(f"Expected wave shape [B, T] or [T], got {tuple(wave.shape)}")

        self._ensure_resampler(input_sr)
        if self._resampler is not None:
            wave = self._resampler(wave)

        out = self.model(
            input_values=wave,
            output_hidden_states=True,
        )
        hidden_states = out.hidden_states
        if hidden_states is None:
            raise RuntimeError("Wav2Vec2 output does not contain hidden_states.")

        hidden = hidden_states[self.layer_index]  # [B, T', H]
        return self._pool_embeddings(hidden)




