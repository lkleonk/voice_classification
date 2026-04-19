from typing import Optional

import torch.nn.functional as F
import torch.nn as nn
from torch.nn.init import xavier_normal_

from src.model.model_utils import get_activation_function


class Autoencoder(nn.Module):
    """Autoencoder model combining Encoder and Decoder."""

    _POOLING_FACTOR = 8

    def __init__(self, img_width, img_height, activation_function="relu"):
        super().__init__()

        self.activation = get_activation_function(activation_function)
        self.img_width = img_width
        self.img_height = img_height
        self.encoder = Encoder(activation_function=self.activation)
        self.decoder = Decoder(activation_function=self.activation)
        self._initialize_weights()

    def _initialize_weights(self):
        for param in self.encoder.parameters():
            if isinstance(param, nn.Conv2d):
                xavier_normal_(param)

        for param in self.decoder.parameters():
            if isinstance(param, nn.Conv2d):
                xavier_normal_(param)

    @classmethod
    def _get_required_padding(cls, x):
        pad_height = (-x.size(-2)) % cls._POOLING_FACTOR
        pad_width = (-x.size(-1)) % cls._POOLING_FACTOR
        return pad_height, pad_width

    def forward(self, x):
        original_height = x.size(-2)
        original_width = x.size(-1)
        pad_height, pad_width = self._get_required_padding(x)

        if pad_height or pad_width:
            x = F.pad(x, (0, pad_width, 0, pad_height), mode="reflect")

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded[..., :original_height, :original_width]

    def encode(self, x):
        return self.encoder(x)

    def decode(self, encoded_features):
        return self.decoder(encoded_features)


class Decoder(nn.Module):
    """Decoder network."""

    def __init__(self, activation_function: Optional[nn.Module] = None, nr_of_channels=1):
        super().__init__()
        if activation_function is None:
            activation_function = nn.LeakyReLU()

        self.reflecPad_1_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_1_1 = nn.Conv2d(512, 256, 3, 1, 0)
        self.relu_1_1 = activation_function
        self.unpool_1 = nn.UpsamplingNearest2d(scale_factor=2)

        self.reflecPad_2_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_1 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu_2_1 = activation_function
        self.reflecPad_2_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu_2_2 = activation_function
        self.reflecPad_2_3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu_2_3 = activation_function
        self.reflecPad_2_4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_4 = nn.Conv2d(256, 128, 3, 1, 0)
        self.relu_2_4 = activation_function
        self.unpool_2 = nn.UpsamplingNearest2d(scale_factor=2)

        self.reflecPad_3_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_1 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu_3_1 = activation_function
        self.reflecPad_3_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_2 = nn.Conv2d(128, 64, 3, 1, 0)
        self.relu_3_2 = activation_function
        self.unpool_3 = nn.UpsamplingNearest2d(scale_factor=2)

        self.reflecPad_4_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_4_1 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu_4_1 = activation_function
        self.reflecPad_4_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_4_2 = nn.Conv2d(64, nr_of_channels, 3, 1, 0)

    def forward(self, input):
        out = self.reflecPad_1_1(input)
        out = self.conv_1_1(out)
        out = self.relu_1_1(out)
        out = self.unpool_1(out)

        out = self.reflecPad_2_1(out)
        out = self.conv_2_1(out)
        out = self.relu_2_1(out)
        out = self.reflecPad_2_2(out)
        out = self.conv_2_2(out)
        out = self.relu_2_2(out)
        out = self.reflecPad_2_3(out)
        out = self.conv_2_3(out)
        out = self.relu_2_3(out)
        out = self.reflecPad_2_4(out)
        out = self.conv_2_4(out)
        out = self.relu_2_4(out)
        out = self.unpool_2(out)

        out = self.reflecPad_3_1(out)
        out = self.conv_3_1(out)
        out = self.relu_3_1(out)
        out = self.reflecPad_3_2(out)
        out = self.conv_3_2(out)
        out = self.relu_3_2(out)
        out = self.unpool_3(out)

        out = self.reflecPad_4_1(out)
        out = self.conv_4_1(out)
        out = self.relu_4_1(out)
        out = self.reflecPad_4_2(out)
        out = self.conv_4_2(out)
        return out


class Encoder(nn.Module):
    """Encoder network."""

    def __init__(self, activation_function: Optional[nn.Module] = None, nr_of_channels=1):
        super().__init__()
        if activation_function is None:
            activation_function = nn.LeakyReLU(inplace=True)

        self.conv_1_1 = nn.Conv2d(nr_of_channels, 3, 1, 1, 0)
        self.reflecPad_1_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_1_2 = nn.Conv2d(3, 64, 3, 1, 0)
        self.relu_1_2 = activation_function
        self.reflecPad_1_3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_1_3 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu_1_3 = activation_function
        self.maxPool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.reflecPad_2_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_1 = nn.Conv2d(64, 128, 3, 1, 0)
        self.relu_2_1 = activation_function
        self.reflecPad_2_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu_2_2 = activation_function
        self.maxPool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.reflecPad_3_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_1 = nn.Conv2d(128, 256, 3, 1, 0)
        self.relu_3_1 = activation_function
        self.reflecPad_3_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu_3_2 = activation_function
        self.reflecPad_3_3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu_3_3 = activation_function
        self.reflecPad_3_4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu_3_4 = activation_function
        self.maxPool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.reflecPad_4_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_4_1 = nn.Conv2d(256, 512, 3, 1, 0)
        self.relu_4_1 = activation_function

    def forward(self, input):
        out = self.conv_1_1(input)
        out = self.reflecPad_1_1(out)
        out = self.conv_1_2(out)
        out = self.relu_1_2(out)
        out = self.reflecPad_1_3(out)
        out = self.conv_1_3(out)
        out = self.relu_1_3(out)
        out = self.maxPool_1(out)

        out = self.reflecPad_2_1(out)
        out = self.conv_2_1(out)
        out = self.relu_2_1(out)
        out = self.reflecPad_2_2(out)
        out = self.conv_2_2(out)
        out = self.relu_2_2(out)
        out = self.maxPool_2(out)

        out = self.reflecPad_3_1(out)
        out = self.conv_3_1(out)
        out = self.relu_3_1(out)
        out = self.reflecPad_3_2(out)
        out = self.conv_3_2(out)
        out = self.relu_3_2(out)
        out = self.reflecPad_3_3(out)
        out = self.conv_3_3(out)
        out = self.relu_3_3(out)
        out = self.reflecPad_3_4(out)
        out = self.conv_3_4(out)
        out = self.relu_3_4(out)
        out = self.maxPool_3(out)

        out = self.reflecPad_4_1(out)
        out = self.conv_4_1(out)
        out = self.relu_4_1(out)
        return out

