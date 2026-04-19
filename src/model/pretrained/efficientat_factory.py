import torch
import torch.nn as nn

from src.model.efficientat.EfficientAT.ef_at_models.dymn.model import (
    get_model as get_dymn,
)
from src.model.efficientat.EfficientAT.ef_at_models.preprocess import (
    AugmentMelSTFT,
)
from src.model.efficientat.EfficientAT.helpers.utils import NAME_TO_WIDTH
from src.model.pretrained.pretrained_models_params import get_pretrained_model_spec
import src.utils.toolkit.cuda_handling as cuda_handling

device = cuda_handling.set_cuda_to_gpu_nr()

_EFFICIENTAT_SPEC = get_pretrained_model_spec("efficientat")


class EfficientATEmbedder(nn.Module):
    def __init__(
        self,
        model_name: str = "dymn20_as",
        strides=(2, 2, 2, 2),
        sample_rate: int = int(_EFFICIENTAT_SPEC["spectrogram.sample_rate"]),
        window_size: int = int(_EFFICIENTAT_SPEC["spectrogram.window_length"]),
        hop_size: int = int(_EFFICIENTAT_SPEC["spectrogram.hop_length"]),
        n_mels: int = int(_EFFICIENTAT_SPEC["spectrogram.n_mels"]),
    ):
        super().__init__()

        self.model = get_dymn(
            width_mult=NAME_TO_WIDTH(model_name),
            pretrained_name=model_name,
            strides=strides,
        )

        self.model.to(device)

        self.mel = AugmentMelSTFT(
            n_mels=n_mels,
            sr=sample_rate,
            win_length=window_size,
            hopsize=hop_size,
        )
        self.mel.to(device)

    def forward(self, wave_32k: torch.Tensor) -> torch.Tensor:
        spec = self.mel(wave_32k)
        _, emb = self.model(spec.unsqueeze(1))
        return emb

