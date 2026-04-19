from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from speechbrain.lobes.models.beats import BEATs
from src.model.pretrained.checkpoint_utils import require_local_checkpoint
from src.model.pretrained.pretrained_models_params import get_pretrained_model_spec
import src.utils.toolkit.cuda_handling as cuda_handling
from src.settings import PATHS, SPECTROGRAM_NORM

device = cuda_handling.set_cuda_to_gpu_nr()
_BEATS_SPEC = get_pretrained_model_spec("beats")



class BEATSEmbedder(nn.Module):
    """
    SpeechBrain BEATs embedder. FROZEN

    Input:
      wave_16k: [B, T] float waveform @ 16kHz (recommended)
    Output:
      emb: [B, C] pooled embedding, where C is typically 768 (base) or 1024 (large)

    Notes:
      - BEATs computes fbank features internally + normalization internally.
      - You provide a BEATs checkpoint file yourself via PATHS.
    """

    def __init__(
        self,
        ):
        super().__init__()


        # Download + store the checkpoint yourself (e.g. BEATs_iter3_large.pt)
        weights_path = require_local_checkpoint(
            PATHS.BEATS_WEIGHTS_PATH,
            model_name="beats",
            source_hint="See README.md for the pinned checkpoint filename",
        )
        
        self.model = BEATs(
            ckp_path=str(weights_path),
            freeze=True,
            output_all_hiddens=False,
        )
        self.model.eval()

    @staticmethod
    def _mean_pool_time(
        feats: torch.Tensor,
        wav_lens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        feats: [B, T', C]
        wav_lens: [B] in [0, 1] relative lengths (SpeechBrain format)
        returns: [B, C]
        """
        if wav_lens is None:
            return feats.mean(dim=1)

        B, Tp, C = feats.shape
        # Convert relative wav lengths to relative feature lengths.
        # This is an approximation but works well in practice for pooling.
        feat_lens = torch.clamp((wav_lens * Tp).round().long(), min=1, max=Tp)

        # Mask: [B, T']
        idx = torch.arange(Tp, device=feats.device).unsqueeze(0).expand(B, Tp)
        mask = idx < feat_lens.unsqueeze(1)

        feats = feats * mask.unsqueeze(-1)
        denom = feat_lens.unsqueeze(1).to(feats.dtype)  # [B, 1]
        return feats.sum(dim=1) / denom

    def forward(self, wave_16k: torch.Tensor, wav_lens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        wave_16k: [B, T] waveform at 16kHz
        wav_lens: [B] relative lengths in [0, 1] (optional; if None assumes full length)

        returns:
          emb: [B, C]
        """
        # If caller doesn’t provide lens, assume all samples are full length.
        if wav_lens is None:
            wav_lens = torch.ones(wave_16k.shape[0], device=wave_16k.device, dtype=torch.float32)

        wave_16k = wave_16k.to(device)
        wav_lens = wav_lens.to(device)

        with torch.no_grad():
            # BEATs forward returns encoded featuresprint 
            with torch.no_grad():
                feats = self.model.extract_features(
                    wave_16k,
                    wav_lens=wav_lens,
                    fbank_mean=SPECTROGRAM_NORM.AUDIOSET_FBANK_MEAN,
                    fbank_std=SPECTROGRAM_NORM.AUDIOSET_FBANK_STD,
                )

            # Some versions/wrappers return (feats, padding_mask, ...) or a tuple/list.
            if isinstance(feats, (tuple, list)):
                feats = feats[0]

            # If output_all_hiddens=True, shape may be (L+1, B, T', C); take last layer.
            if feats.dim() == 4:
                feats = feats[-1]

            # Now expect feats: [B, T', C]
            emb = self._mean_pool_time(feats, wav_lens)

        return emb
