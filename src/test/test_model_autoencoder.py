import torch

from src.model.self_supervised_autoencoder.autoencoder import Autoencoder


def test_autoencoder_preserves_shape_for_non_divisible_input_size():
    model = Autoencoder(img_width=466, img_height=132)
    x = torch.randn(2, 1, 132, 466)

    y = model(x)

    assert y.shape == x.shape


def test_autoencoder_preserves_shape_for_divisible_input_size():
    model = Autoencoder(img_width=464, img_height=128)
    x = torch.randn(2, 1, 128, 464)

    y = model(x)

    assert y.shape == x.shape
