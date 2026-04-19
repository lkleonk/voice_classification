import torch
import torch.nn as nn

from src.model.model_utils import (
    _create_conv_layers,
    _create_lstm_cells,
    _create_mlps,
    _verify_demographic_inputs,
    str_w_nrs_turn_to_int_list,
)

_DEMOGRAPHIC_DATA_MLP_OUTPUT_SIZE = 32


class CNNLSTM(nn.Module):
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

        self.add_demographic_data = add_demographic_data
        self.extra_demographic_data_mlp = config["features.extra_demographic_data_mlp"]
        self.demographic_data_tensor_len = demographic_data_tensor_len

        cnn_layer_dropout = config["cnnlstm.cnn_layer_dropout"]
        lstm_layer_dropout = config["cnnlstm.lstm_layer_dropout"]
        final_dropout = config["cnnlstm.final_dropout"]
        cnn_layer_sizes = str_w_nrs_turn_to_int_list(config["cnnlstm.cnn_layer_sizes"])
        lstm_cell_sizes = str_w_nrs_turn_to_int_list(config["cnnlstm.lstm_cell_sizes"])
        mlp_layer_sizes = str_w_nrs_turn_to_int_list(config["cnnlstm.mlp_layer_sizes"])
        bidirectional_lstm = config["cnnlstm.bidirectional_lstm"]

        self.conv_layers = _create_conv_layers(cnn_layer_sizes, cnn_layer_dropout)

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, freq_bins, timesteps)
            cnn_output = self.conv_layers(dummy_input)
            _, channels, freq_prime, _ = cnn_output.shape
            lstm_input_size = channels * freq_prime

        _verify_demographic_inputs(self.add_demographic_data, self.demographic_data_tensor_len)

        self.lstm_cells = _create_lstm_cells(
            lstm_input_size,
            lstm_cell_sizes,
            lstm_layer_dropout,
            bidirectional=bidirectional_lstm,
        )
        size_lstm_output = lstm_cell_sizes[-1] * 2 if bidirectional_lstm else lstm_cell_sizes[-1]
        self.demo_mlp, self.final_mlp = _create_mlps(
            config,
            self.add_demographic_data,
            num_classes,
            demographic_data_tensor_len,
            _DEMOGRAPHIC_DATA_MLP_OUTPUT_SIZE,
            mlp_layer_sizes,
            final_dropout,
            size_lstm_output,
        )

    def forward(self, x, demo_tensor=None):
        if self.add_demographic_data and demo_tensor is None:
            raise ValueError("demographic data must be provided when add_demographic_data=True")

        x = self.conv_layers(x)
        batch_size, channels, freq_prime, timesteps_prime = x.shape
        x = x.permute(0, 3, 1, 2).reshape(batch_size, timesteps_prime, channels * freq_prime)

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

