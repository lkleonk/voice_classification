from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.model.model_utils import (
    _create_final_mlp,
    _verify_demographic_inputs,
    str_w_nrs_turn_to_int_list,
)


class TabularMLP(nn.Module):
    def __init__(
        self,
        num_classes: int,
        timesteps: int,
        freq_bins: int,
        config: dict,
        add_demographic_data: bool,
        demographic_data_tensor_len: Optional[int] = None,
    ) -> None:
        super().__init__()
        del timesteps, freq_bins # audio data is not needed here

        self.add_demographic_data = add_demographic_data
        self.demographic_data_tensor_len = demographic_data_tensor_len

        _verify_demographic_inputs(
            self.add_demographic_data,
            self.demographic_data_tensor_len,
        )
        if not self.add_demographic_data:
            raise ValueError("tabular_mlp requires --add_demo_data")

        mlp_layer_sizes = str_w_nrs_turn_to_int_list(config["tabular_mlp.mlp_layer_sizes"])
        final_dropout = float(config["tabular_mlp.final_dropout"])

        self.final_mlp = _create_final_mlp(
            input_size=int(self.demographic_data_tensor_len),
            mlp_layers=mlp_layer_sizes,
            output=num_classes,
            dropout=final_dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        demo_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del x
        if demo_tensor is None:
            raise ValueError("tabular_mlp requires demographic data input")
        return self.final_mlp(demo_tensor)
