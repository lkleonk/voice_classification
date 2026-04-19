from src.model.pretrained.pretrained_model_embedder_with_clf import (
    ModelEmbedderWithClf,
)
from src.model.scratch.scratch_cnn_models import (
    CNN,
    test_CNN,
    test_CNN2,
)
from src.model.scratch.scratch_cnnlstm import CNNLSTM
from src.model.scratch.scratch_lstm import LSTM
from src.model.scratch.scratch_vit import ViT
from src.model.tabular_mlp import TabularMLP


def build_model(
    model_type: str,
    *,
    num_classes: int,
    img_width: int,
    img_height: int,
    config: dict,
    demographic_data_tensor_length: int,
    add_demographic_data: bool,
):
    common_kwargs = dict(
        num_classes=num_classes,
        timesteps=img_width,
        freq_bins=img_height,
        config=config,
        add_demographic_data=add_demographic_data,
        demographic_data_tensor_len=demographic_data_tensor_length,
    )

    model_type = model_type.lower()

    registry = {
        "test_cnn": lambda: test_CNN(**common_kwargs),
        "test_cnn2": lambda: test_CNN2(**common_kwargs),
        "tabular_mlp": lambda: TabularMLP(**common_kwargs),
        "cnn": lambda: CNN(**common_kwargs),
        "lstm": lambda: LSTM(**common_kwargs),
        "cnnlstm": lambda: CNNLSTM(**common_kwargs),
        "vit": lambda: ViT(**common_kwargs),
        "vggish": lambda: ModelEmbedderWithClf(
            "vggish",
            num_classes,
            config,
            add_demographic_data=add_demographic_data,
            demographic_data_tensor_len=demographic_data_tensor_length,
            timesteps=img_width,
            freq_bins=img_height,
        ),
        "pann": lambda: ModelEmbedderWithClf(
            "pann",
            num_classes,
            config,
            add_demographic_data=add_demographic_data,
            demographic_data_tensor_len=demographic_data_tensor_length,
        ),
        "passt": lambda: ModelEmbedderWithClf(
            "passt",
            num_classes,
            config,
            add_demographic_data=add_demographic_data,
            demographic_data_tensor_len=demographic_data_tensor_length,
        ),
        "efficientat": lambda: ModelEmbedderWithClf(
            "efficientat",
            num_classes,
            config,
            add_demographic_data=add_demographic_data,
            demographic_data_tensor_len=demographic_data_tensor_length,
        ),
        "ast": lambda: ModelEmbedderWithClf(
            "ast",
            num_classes,
            config,
            add_demographic_data=add_demographic_data,
            demographic_data_tensor_len=demographic_data_tensor_length,
        ),
        "wav2vec2": lambda: ModelEmbedderWithClf(
            "wav2vec2",
            num_classes,
            config,
            add_demographic_data=add_demographic_data,
            demographic_data_tensor_len=demographic_data_tensor_length,
        ),
        "beats": lambda: ModelEmbedderWithClf(
            "beats",
            num_classes,
            config,
            add_demographic_data=add_demographic_data,
            demographic_data_tensor_len=demographic_data_tensor_length,
        ),
    }

    if model_type == "autoencoder":
        raise ValueError(f"Model type {model_type} not implemented here. Use train_autoencoder.py")

    try:
        return registry[model_type]()
    except KeyError as e:
        raise ValueError(f"Unknown model type: {model_type}") from e

