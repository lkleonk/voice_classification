import torch
import torch.nn as nn

from src.model.model_utils import (
    _create_lstm_cells,
    _create_mlps,
    _verify_demographic_inputs,
    str_w_nrs_turn_to_int_list,
)

_DEMOGRAPHIC_DATA_MLP_OUTPUT_SIZE = 32


class LSTM(nn.Module):
    def __init__(
        self,
        num_classes,
        timesteps,
        freq_bins,
        config,
        add_demographic_data,
        demographic_data_tensor_len=None,
    ):
        super().__init__()
        self.freq_bins = freq_bins

        self.add_demographic_data = add_demographic_data
        self.extra_demographic_data_mlp = config["features.extra_demographic_data_mlp"]
        self.demographic_data_tensor_len = demographic_data_tensor_len
        _verify_demographic_inputs(self.add_demographic_data, self.demographic_data_tensor_len)

        lstm_layer_dropout = config["lstm.layer_dropout"]
        final_dropout = config["lstm.final_dropout"]
        lstm_cell_sizes = str_w_nrs_turn_to_int_list(config["lstm.lstm_cell_sizes"])
        mlp_layer_sizes = str_w_nrs_turn_to_int_list(config["lstm.mlp_layer_sizes"])
        bidirectional_lstm = config["lstm.bidirectional_lstm"]
        last_lstm_size = lstm_cell_sizes[-1] * 2 if bidirectional_lstm else lstm_cell_sizes[-1]

        self.lstm_cells = _create_lstm_cells(
            freq_bins,
            lstm_cell_sizes,
            lstm_layer_dropout,
            bidirectional=bidirectional_lstm,
        )
        self.demo_mlp, self.final_mlp = _create_mlps(
            config,
            self.add_demographic_data,
            num_classes,
            demographic_data_tensor_len,
            _DEMOGRAPHIC_DATA_MLP_OUTPUT_SIZE,
            mlp_layer_sizes,
            final_dropout,
            last_lstm_size,
        )

    def forward(self, x, demo_tensor=None):
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1)

        if x.size(1) == self.freq_bins:
            x = x.transpose(1, 2)

        for lstm in self.lstm_cells:
            x, _ = lstm(x)

        x = x[:, -1, :]
        if self.add_demographic_data:
            if self.extra_demographic_data_mlp:
                demo_tensor = self.demo_mlp(demo_tensor)
            if demo_tensor is None:
                raise ValueError("demographic data must be provided when add_demographic_data=True")
            x = torch.cat((x, demo_tensor), dim=1)

        return self.final_mlp(x)
