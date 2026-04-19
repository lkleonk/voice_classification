import torch
import torch.nn as nn

# from hear21passt.base import get_basic_model, get_scene_embeddings
from hear21passt.models.passt import get_model
from hear21passt.models.preprocess import AugmentMelSTFT

from src.model.pretrained.checkpoint_utils import require_local_checkpoint
from src.model.pretrained.pretrained_models_params import get_pretrained_model_spec
import src.utils.toolkit.cuda_handling as cuda_handling
from src.settings import PATHS

device = cuda_handling.set_cuda_to_gpu_nr()
_PASST_SPEC = get_pretrained_model_spec("passt")




class PASSTEmbedder(nn.Module):
    def __init__(
        self,
        *,
        sample_rate: int = int(_PASST_SPEC["spectrogram.sample_rate"]),
        window_size: int = int(_PASST_SPEC["spectrogram.window_length"]),
        hop_size: int = int(_PASST_SPEC["spectrogram.hop_length"]),
        mel_bins: int = int(_PASST_SPEC["spectrogram.n_mels"]),
        fmin: float = float(_PASST_SPEC["spectrogram.fmin"]),
        fmax: float = float(_PASST_SPEC["spectrogram.fmax"]),
        n_fft: int = int(_PASST_SPEC["passt.n_fft"]),
        input_tdim: int = int(_PASST_SPEC["passt.input_tdim"]),
    ):
        super().__init__()
        ARCH = "passt_s_kd_p16_128_ap486"
        self.input_tdim = input_tdim  # single source of truth


        self.mel = AugmentMelSTFT( # check settings again - compare with "overwrite_config"-function 
            n_mels=mel_bins,
            sr=sample_rate,          # <<< YES, 32k
            win_length=window_size,    # 25 ms at 32 kHz
            hopsize=hop_size,       # 10 ms at 32 kHz
            n_fft=n_fft,        # matches HEAR defaults
            fmin=fmin,
            fmax=fmax,        # HEAR uses higher fmax
            norm=1,
            htk=True,
            fmin_aug_range=10,
            fmax_aug_range=2000,
        )


        # 1) Build the model (NO pretrained weights)
        self.model = get_model(
            arch=ARCH,
            pretrained=False,
            n_classes=527,
            in_channels=1,
            fstride=10,
            tstride=10,
            input_fdim=mel_bins,
            input_tdim=self.input_tdim,
            u_patchout=0,
            s_patchout_t=0,
            s_patchout_f=0,
        )

        # load model backbone
        weights_path = require_local_checkpoint(
            PATHS.PASST_WEIGHTS_PATH,
            model_name="passt",
            source_hint="See README.md for the pinned checkpoint filename",
        )


        
        ckpt = torch.load(weights_path, map_location="cpu")

        self.model.load_state_dict(ckpt, strict=True)
        # self.model.eval()


    @staticmethod
    def _fix_tdim(x: torch.Tensor, tdim: int) -> torch.Tensor:
        """
        Ensure spectrogram has exactly tdim frames.
        x: (B, 1, F, T)
        """
        T = x.shape[-1]
        if T == tdim:
            return x
        if T > tdim:
            # center crop for determinism + symmetry
            start = (T - tdim) // 2
            return x[..., start:start + tdim]
        # pad on the right
        pad = tdim - T
        return torch.nn.functional.pad(x, (0, pad))
    

    def forward(self, wave_32k):
        """
        wave_32k: [B, T] waveform at 32kHz
        returns:  [B, 768] embedding
        """
        with torch.no_grad():                      # <-- IMPORTANT: no_grad, not inference_mode
            spec = self.mel(wave_32k)
            # input:  [B, T] waveform
            # output: [B, 128, frames]

            spec = spec.unsqueeze(1)
            # input:  [B, 128, frames]
            # output: [B, 1, 128, frames]

            spec = self._fix_tdim(spec, self.input_tdim)

            logits, emb = self.model(spec)
            # input:  [B, 1, 128, frames]
            # output:
            #   logits: [B, 527]
            #   emb:    [B, 768]

        return emb



