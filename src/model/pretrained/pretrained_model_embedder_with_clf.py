from typing import Optional

import torch
import torch.nn as nn

from src.model.pretrained.cache_env import configure_pretrained_cache_env
from src.model.model_utils import (
    _create_mlps,
    _verify_demographic_inputs,
    freeze_model,
    str_w_nrs_turn_to_int_list,
    unfreeze_last_layers,
)
from src.model.pretrained.pretrained_models_params import get_pretrained_model_spec

_DEMOGRAPHIC_DATA_MLP_OUTPUT_SIZE = 32


class ModelEmbedderWithClf(nn.Module):
    """Wrapper for pretrained backbones plus classifier head."""

    def __init__(
        self,
        base_model_name,
        num_classes,
        config,
        add_demographic_data,
        demographic_data_tensor_len=None,
        timesteps=None,
        freq_bins=None,
    ):
        super().__init__()
        model_spec = get_pretrained_model_spec(base_model_name)

        def _cfg(key: str):
            return config.get(key, model_spec[key])

        configure_pretrained_cache_env()

        if base_model_name == "vggish":
            from src.model.pretrained.vggish import VGGishEmbedder

            self.base_model = VGGishEmbedder()
        elif base_model_name == "pann":
            from src.model.pretrained.pann import PANNEmbedder

            self.base_model = PANNEmbedder()
        elif base_model_name == "passt":
            from src.model.pretrained.passt import PASSTEmbedder

            self.base_model = PASSTEmbedder(input_tdim=int(_cfg("passt.input_tdim")))
        elif base_model_name == "efficientat":
            from src.model.pretrained.efficientat_factory import (
                EfficientATEmbedder,
            )

            self.base_model = EfficientATEmbedder()
        elif base_model_name == "ast":
            from src.model.pretrained.ast import ASTEmbedder

            self.base_model = ASTEmbedder()
        elif base_model_name == "beats":
            from src.model.pretrained.beats import BEATSEmbedder

            self.base_model = BEATSEmbedder()
        elif base_model_name == "wav2vec2":
            from src.model.pretrained.wav2vec2 import Wav2Vec2Embedder

            wav2vec2_append_std = bool(config["wav2vec2.append_std"])
            wav2vec2_layer_size = int(config["wav2vec2.wav2vec2_layer_size"])
            self.base_model = Wav2Vec2Embedder(
                append_std=wav2vec2_append_std,
                model_id=config["wav2vec2.model_id"],
                layer_index=wav2vec2_layer_size,
            )
        else:
            raise ValueError(f"Unknown base_model_name: {base_model_name}")

        embed_dim = int(_cfg(f"{base_model_name}.embed_dim"))
        if base_model_name == "wav2vec2":
            embed_dim = int(self.base_model.embed_dim)

        self.add_demographic_data = add_demographic_data
        self.extra_demographic_data_mlp = config["features.extra_demographic_data_mlp"]
        self.demographic_data_tensor_len = demographic_data_tensor_len
        final_dropout = config[f"{base_model_name}.final_dropout"]
        mlp_layer_sizes = str_w_nrs_turn_to_int_list(config[f"{base_model_name}.mlp_layer_sizes"])
        self.unfreeze_last_layers_num = config[f"{base_model_name}.unfreeze_last_layers"]

        self.base_model = freeze_model(self.base_model)
        if self.unfreeze_last_layers_num != 0:
            unfreeze_last_layers(self.base_model, self.unfreeze_last_layers_num)

        _verify_demographic_inputs(self.add_demographic_data, self.demographic_data_tensor_len)
        self.demo_mlp, self.final_mlp = _create_mlps(
            config,
            self.add_demographic_data,
            num_classes,
            demographic_data_tensor_len,
            _DEMOGRAPHIC_DATA_MLP_OUTPUT_SIZE,
            mlp_layer_sizes,
            final_dropout,
            embed_dim,
        )

    def forward(self, x: torch.Tensor, demo_tensor: Optional[torch.Tensor] = None):
        if self.unfreeze_last_layers_num == 0:
            with torch.no_grad():
                x = self.base_model(x)
        else:
            x = self.base_model(x)

        x = x.view(x.size(0), -1)
        if self.add_demographic_data:
            if self.extra_demographic_data_mlp:
                demo_tensor = self.demo_mlp(demo_tensor)
            if demo_tensor is None:
                raise ValueError("demographic data must be provided when add_demographic_data=True")
            x = torch.cat((x, demo_tensor), dim=1)

        return self.final_mlp(x)

