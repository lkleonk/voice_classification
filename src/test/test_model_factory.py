import pytest
import torch

from src.model.model_factory import build_model


@pytest.mark.unit
def test_build_model_supports_test_cnn2() -> None:
    config = {
        "features.extra_demographic_data_mlp": False,
    }

    model = build_model(
        "test_cnn2",
        num_classes=4,
        img_width=64,
        img_height=32,
        config=config,
        demographic_data_tensor_length=0,
        add_demographic_data=False,
    )

    x = torch.randn(3, 1, 32, 64)
    logits = model(x)

    assert logits.shape == (3, 4)
    assert model.conv_layers[3].out_channels == 3
