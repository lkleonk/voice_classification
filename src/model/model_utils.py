import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

import src.utils.toolkit.cuda_handling as cuda_handling

device = cuda_handling.set_cuda_to_gpu_nr()









def _verify_demographic_inputs(
    add_demographic_data: bool,
    demographic_data_tensor_len: Optional[int],
) -> None:
    """
    Ensure that if demographic data is enabled, the tensor length is specified.
    
    Args:
        add_demographic_data (bool): Whether demographic data is being used.
        demographic_data_tensor_len (int or None): Length of the demographic data tensor.

    Raises:
        ValueError: If add_demographic_data is True but demographic_data_tensor_len is not set.
    """
    if add_demographic_data and not demographic_data_tensor_len:
        raise ValueError(
            "When add_demographic_data is True, demographic_data_tensor_len must be specified. "
            f"Got add_demographic_data={add_demographic_data}, "
            f"demographic_data_tensor_len={demographic_data_tensor_len}"
        )






def get_activation_function(fn_name: str) -> nn.Module:
    """
    Returns the corresponding PyTorch activation function based on the input string.

    Args:okay nice thank
    fn_name (str): Name of the activation function as a string (e.g., "relu", "leakyrelu", "gelu").

    Returns:
    activation_function: A PyTorch activation function object (e.g., nn.ReLU(), nn.LeakyReLU(), nn.GELU()).
    
    Raises:
    ValueError: If the provided fn_name does not match any known activation function.
    """
    #activation_function = None
    
    fn_name = fn_name.lower()  # Convert the name to lowercase for comparison
    
    if fn_name == "leakyrelu":
        activation_function = nn.LeakyReLU()
    elif fn_name == "relu":
        activation_function = nn.ReLU()
    elif fn_name == "gelu":
        activation_function = nn.GELU()
    else:
        raise ValueError(f"Unknown activation function: {fn_name}")
    
    return activation_function





def _create_final_mlp(
    input_size: int,
    mlp_layers: Sequence[int],
    output: int,
    dropout: float = 0.0,
) -> nn.Sequential:
    """
    Creates an MLP (Multi-Layer Perceptron) for the final classification/regression head.
    
    Args:
        input_size (int): Size of the input features.
        mlp_layers (list): List of integers specifying hidden layer sizes (e.g., [128, 64]).
        output (int): Output dimension (e.g., 2 for binary classification).
        dropout (float): Dropout probability applied after each hidden layer.
    
    Returns:
        nn.Sequential: The MLP module.
    """
    layers = []
    prev_size = input_size
    
    # Add hidden layers
    for size in mlp_layers:
        layers.append(nn.Linear(prev_size, size))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_size = size
    
    # Add final output layer
    layers.append(nn.Linear(prev_size, output))
    
    return nn.Sequential(*layers)



def freeze_model(model: nn.Module) -> nn.Module: # freeze all model layers
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected nn.Module, got {type(model)}")

    for param in model.parameters():
        param.requires_grad = False

    return model


def get_work_model_weights_path() -> str:

    ### Get absolute path to current file (inside src)
    #current_dir = os.path.dirname(os.path.abspath(__file__))
    current_dir = os.path.dirname(os.path.abspath(__file__))

    ## Build path to weights
    parent_folder = os.path.dirname(current_dir)  # This goes back one level to models/ folder
    second_parent_folder = os.path.dirname(parent_folder)
    home_directory = os.path.dirname(os.path.dirname(second_parent_folder))
    weights_folder_path = os.path.join(home_directory, "work", "model_weights")
    return weights_folder_path




def unfreeze_last_layers(model: nn.Module, num_layers: int) -> nn.Module:
    """
    Unfreeze the last `num_layers` layers of a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.
        num_layers (int): Number of layers to unfreeze. Must be negative, e.g., -5.
    """

    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected nn.Module, got {type(model)}")
    if num_layers >= 0:
        raise ValueError("For now, num_layers must be negative. No scenario yet where it should be positive. This can be changed")

    params = list(model.named_parameters())

    # Freeze all layers first
    for _, param in params:
        param.requires_grad = False

    # Unfreeze last `abs(num_layers)` parameters
    for name, param in params[num_layers:]:
        param.requires_grad = True
        print(f"Unfroze: {name}")
        
    return model


def _get_final_mlp_input_size(
    config: Dict[str, Any],
    add_demographic_data: bool,
    mlp_input_size: int,
    demo_acoustic_data_tensor_len: int,
    demo_acoustic_data_mlp_output_size: int,
) -> int:
    

    if add_demographic_data and config['features.extra_demographic_data_mlp']:
        if config['features.extra_demographic_data_mlp']:
            # in case we have an extra MLP for the demographic and acoustic data
            mlp_input_size += demo_acoustic_data_mlp_output_size
        else:
            # in case we just add the demographic and acoustic data in raw form to the final MLP
            mlp_input_size += demo_acoustic_data_tensor_len

    return mlp_input_size



def _create_extra_data_mlp(
    config: Dict[str, Any],
    add_demographic_data: bool,
    demographic_data_tensor_len: int,
    demographic_data_mlp_output_size: int,
) -> Optional[nn.Sequential]:
    demo_mlp = None

    if add_demographic_data and config['features.extra_demographic_data_mlp']:
        demo_mlp_dropout = float(config["features.extra_demographic_data_mlp_dropout"])
        demo_mlp = nn.Sequential(
        nn.Linear(demographic_data_tensor_len, demographic_data_mlp_output_size),  # to project 5 demographics → 32 features
        nn.ReLU(),
        nn.Dropout(demo_mlp_dropout)
        )   

    return demo_mlp         





def _auto_calculate_flattened_size(
    height: int,
    width: int,
    conv_layers: nn.Module,
) -> int:
    with torch.no_grad():
        dummy_input = torch.zeros(1, height, width)
        conv_output = conv_layers(dummy_input)
        flattened_conv_layers_output_size = conv_output.view(1, -1).shape[1]
        #print(f"[DEBUG] Conv output shape: {conv_output.shape}, flattened={mlp_input_size}")
    return flattened_conv_layers_output_size
    



# Auto-calculate flattened size
#flattened_conv_layer_output_size = _auto_calculate_flattened_size(freq_bins, timesteps, self.conv_layers)

## calculate size of linear layer input to last MLP
#final_mlp_input_size = _get_final_mlp_input_size(config, flattened_conv_layer_output_size, demographic_data_tensor_len, _demographic_data_mlp_output_size)

## create MLP for demographic (and acoustic features)
#self.demo_mlp = _create_extra_data_mlp(config, demographic_data_tensor_len, _demographic_data_mlp_output_size)
#
## create MLP for final MLP
#self.final_mlp = _create_final_mlp(final_mlp_input_size, mlp_layer_sizes, output=num_classes, dropout = final_dropout)



def _create_mlps(
    config: Dict[str, Any],
    add_demographic_data: bool,
    num_classes: int,
    demographic_data_tensor_len: int,
    _demographic_data_mlp_output_size: int,
    mlp_layer_sizes: Sequence[int],
    final_dropout: float,
    initial_mlp_input_size: int,
) -> Tuple[Optional[nn.Sequential], nn.Sequential]:


    # calculate size of linear layer input to last MLP
    final_mlp_input_size = _get_final_mlp_input_size(
        config,
        add_demographic_data,
        initial_mlp_input_size,
        demographic_data_tensor_len,
        _demographic_data_mlp_output_size,
    )

    # create MLP for demographic (and acoustic features)
    demo_mlp = _create_extra_data_mlp(
        config,
        add_demographic_data,
        demographic_data_tensor_len,
        _demographic_data_mlp_output_size,
    )
    
    # create MLP for final MLP
    final_mlp = _create_final_mlp(final_mlp_input_size, mlp_layer_sizes, output=num_classes, dropout = final_dropout)

    return demo_mlp, final_mlp



def str_w_nrs_turn_to_int_list(s: str) -> List[int]:
    """
    Convert a comma-separated string of numbers into a list of integers.
    Example: '128, 64' -> [128, 64]
    """
    result = []
    parts = s.split(',')
    for part in parts:
        number = int(part.strip())
        result.append(number)
    return result






def _create_conv_layers(cnn_layer_sizes: Sequence[int], dropout: float) -> nn.Sequential:
    """
    Build a stack of Conv2d → ReLU → Dropout2d → MaxPool2d layers.
    cnn_layer_sizes: list of output channel sizes, e.g. [32, 64, 128]
    dropout: float, dropout probability
    """
    layers = []
    in_channels = 1  # spectrogram input has 1 channel (grayscale)
    for out_channels in cnn_layer_sizes:
        layers.extend([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        in_channels = out_channels
    return nn.Sequential(*layers)


def _create_lstm_cells(
    input_size: int,
    lstm_cell_sizes: Sequence[int],
    dropout: float,
    bidirectional: bool = False,
) -> nn.ModuleList:
    """
    Build stacked LSTMs with proper input sizes.
    """
    cells = nn.ModuleList()
    in_dim = input_size
    for i, hidden_size in enumerate(lstm_cell_sizes):
        lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=dropout if i < len(lstm_cell_sizes) - 1 else 0.0,
            bidirectional=bidirectional
        )
        cells.append(lstm)
        # next LSTM expects output size of previous one
        in_dim = hidden_size * (2 if bidirectional else 1)
    return cells

def freeze_layers(model: nn.Module, layer_name: str, logger: Any) -> None:
    layer = getattr(model, layer_name, None)
    if layer is None:
        logger.info(f"Layer {layer_name} not found in model.")
        return
    for param in layer.parameters():
        param.requires_grad = False
    logger.info(f"Frozen: {layer_name}")


def unfreeze_layers(model: nn.Module, layer_name: str, logger: Any) -> None:
    layer = getattr(model, layer_name, None)
    if layer is None:
        logger.info(f"Layer {layer_name} not found in model.")
        return
    for param in layer.parameters():
        param.requires_grad = True
    logger.info(f"Unfrozen: {layer_name}")


def check_ratio_trainable_and_freezed_params(model: nn.Module) -> str:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    ratio = trainable_params / total_params if total_params > 0 else 0

    return (
        f"Total parameters: {total_params:,}\n"
        f"Trainable parameters: {trainable_params:,}\n"
        f"Frozen parameters: {frozen_params:,}\n"
        f"Trainable ratio: {ratio:.2%}"
    )



