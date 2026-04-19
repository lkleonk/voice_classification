import torch
import torch.nn as nn

from src.model.model_utils import (
    _auto_calculate_flattened_size,
    _create_extra_data_mlp,
    _create_mlps,
    _verify_demographic_inputs,
    str_w_nrs_turn_to_int_list,
)

_DEMOGRAPHIC_DATA_MLP_OUTPUT_SIZE = 32


class _BaseTestCNN(nn.Module):
    def __init__(
        self,
        num_classes,
        timesteps,
        freq_bins,
        config,
        add_demographic_data,
        demographic_data_tensor_len=None,
        conv_out_channels=2,
    ):
        super().__init__()

        self.add_demographic_data = add_demographic_data
        self.extra_demographic_data_mlp = config["features.extra_demographic_data_mlp"]
        self.demographic_data_tensor_len = demographic_data_tensor_len

        _verify_demographic_inputs(self.add_demographic_data, self.demographic_data_tensor_len)

        self.conv_layers = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(1, conv_out_channels, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, freq_bins, timesteps)
            self.flattened_size = self.conv_layers(dummy_input).view(1, -1).shape[1]

        mlp_input_size = self.flattened_size
        demographic_mlp_size = 1
        if self.add_demographic_data and demographic_data_tensor_len:
            if self.extra_demographic_data_mlp:
                mlp_input_size += demographic_mlp_size
            else:
                mlp_input_size += demographic_data_tensor_len

        self.demo_mlp = _create_extra_data_mlp(
            config,
            self.add_demographic_data,
            demographic_data_tensor_len,
            demographic_data_mlp_output_size=demographic_mlp_size,
        )
        self.fc_layers = nn.Sequential(nn.Linear(mlp_input_size, num_classes))

    def forward(self, x, demo_tensor=None):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)

        if self.add_demographic_data:
            if self.extra_demographic_data_mlp:
                demo_tensor = self.demo_mlp(demo_tensor)
            if demo_tensor is None:
                raise ValueError("demographic data must be provided when add_demographic_data=True")
            x = torch.cat((x, demo_tensor), dim=1)

        return self.fc_layers(x)


class test_CNN(_BaseTestCNN):
    def __init__(
        self,
        num_classes,
        timesteps,
        freq_bins,
        config,
        add_demographic_data,
        demographic_data_tensor_len=None,
    ):
        super().__init__(
            num_classes=num_classes,
            timesteps=timesteps,
            freq_bins=freq_bins,
            config=config,
            add_demographic_data=add_demographic_data,
            demographic_data_tensor_len=demographic_data_tensor_len,
            conv_out_channels=2,
        )


class test_CNN2(_BaseTestCNN):
    def __init__(
        self,
        num_classes,
        timesteps,
        freq_bins,
        config,
        add_demographic_data,
        demographic_data_tensor_len=None,
    ):
        super().__init__(
            num_classes=num_classes,
            timesteps=timesteps,
            freq_bins=freq_bins,
            config=config,
            add_demographic_data=add_demographic_data,
            demographic_data_tensor_len=demographic_data_tensor_len,
            conv_out_channels=3,
        )


class CNN(nn.Module):
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

        layer_dropout = config["cnn.layer_dropout"]
        final_dropout = config["cnn.final_dropout"]
        mlp_layer_sizes = str_w_nrs_turn_to_int_list(config["cnn.mlp_layer_sizes"])

        _verify_demographic_inputs(self.add_demographic_data, self.demographic_data_tensor_len)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(layer_dropout),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(layer_dropout),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(layer_dropout),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        flattened_size = _auto_calculate_flattened_size(freq_bins, timesteps, self.conv_layers)
        self.demo_mlp, self.final_mlp = _create_mlps(
            config,
            self.add_demographic_data,
            num_classes,
            demographic_data_tensor_len,
            _DEMOGRAPHIC_DATA_MLP_OUTPUT_SIZE,
            mlp_layer_sizes,
            final_dropout,
            flattened_size,
        )

    def forward(self, x, demo_tensor=None):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)

        if self.add_demographic_data and demo_tensor is None:
            raise ValueError("demographic data must be provided when add_demographic_data=True")

        if self.add_demographic_data:
            if self.extra_demographic_data_mlp:
                demo_tensor = self.demo_mlp(demo_tensor)
            if demo_tensor is None:
                raise ValueError("demographic data must be provided when add_demographic_data=True")
            x = torch.cat((x, demo_tensor), dim=1)

        return self.final_mlp(x)
