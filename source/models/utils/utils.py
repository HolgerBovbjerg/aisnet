import logging
import os

import torch
from torch import nn


logger = logging.getLogger(__name__)


def freeze_model_parameters(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model_parameters(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True


def compute_model_size(model):
    """
    Computes size of pytorch model in MB
    :param model: PyTorch model
    :return: size of model in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2

    return size_all_mb


def count_parameters(model, trainable: bool = False) -> int:
    """
    Computes size of pytorch model in MB
    :param model: PyTorch model
    :param trainable: If true, only trainable parameters are counted
    :return: number of parameters in model
    """
    if trainable:
        count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    else:
        count = sum(p.numel() for p in model.parameters())
    return count


def format_num(num):
    """Format a number with commas and round to 2 decimals if needed."""
    if num > 1e6:
        return f"{num / 1e6:.2f} M"
    elif num > 1e3:
        return f"{num / 1e3:.2f} K"
    return str(num)

def get_layer_info(layer):
    """
    Get detailed information about a PyTorch layer, including its parameters.

    Args:
        layer (nn.Module): A PyTorch layer.

    Returns:
        str: A string describing the layer's configuration.
    """
    if isinstance(layer, nn.LSTM) or isinstance(layer, nn.GRU):
        return (
            f"({layer.__class__.__name__}, input_dim={layer.input_size}, "
            f"hidden_dim={layer.hidden_size}, layers={layer.num_layers}, "
            f"bidirectional={layer.bidirectional})"
        )
    elif isinstance(layer, nn.Linear):
        return f"({layer.__class__.__name__}, in_features={layer.in_features}, out_features={layer.out_features})"
    elif isinstance(layer, nn.Conv2d):
        return (
            f"({layer.__class__.__name__}, in_channels={layer.in_channels}, "
            f"out_channels={layer.out_channels}, kernel_size={layer.kernel_size}, "
            f"stride={layer.stride}, padding={layer.padding})"
        )
    elif isinstance(layer, nn.Conv1d):
        return (
            f"({layer.__class__.__name__}, in_channels={layer.in_channels}, "
            f"out_channels={layer.out_channels}, kernel_size={layer.kernel_size}, "
            f"stride={layer.stride}, padding={layer.padding})"
        )
    elif isinstance(layer, nn.Embedding):
        return f"({layer.__class__.__name__}, num_embeddings={layer.num_embeddings}, embedding_dim={layer.embedding_dim})"
    else:
        return f"({layer.__class__.__name__})"


def print_module_structure(module, indent=0):
    """
    Prints the structure of a PyTorch module with layer sizes.

    Args:
        module (nn.Module): The module to analyze.
        indent (int): Current indentation level for nested submodules.
    """
    indent_space = "  " * indent
    module_info = get_layer_info(module)
    logger.info(f"{indent_space}{module.__class__.__name__}{module_info}:")

    for name, child in module.named_children():
        logger.info(f"{indent_space}  ({name}):")
        print_module_structure(child, indent + 2)

    # If no children, display this module's parameters if available
    if len(list(module.named_children())) == 0:
        params = list(module.parameters())
        if params:
            param_shapes = [tuple(p.size()) for p in params]
            logger.info(f"{indent_space}  Parameters: {param_shapes}")


def summarize_model(module):
    """
    Summarizes a PyTorch model with details about parameters and size.

    Args:
        module (nn.Module): The module to summarize.

    Returns:
        None
    """
    total_params = count_parameters(module)
    trainable_params = count_parameters(module, trainable=True)
    param_size = compute_model_size(module)

    for param in module.parameters():
        num_params = param.numel()
        total_params += num_params
        param_size += num_params * param.element_size()
        if param.requires_grad:
            trainable_params += num_params

    logger.info("Model summary:")
    logger.info(f"    Class Name: {module.__class__.__name__}")
    logger.info(f"    Total Number of model parameters: {format_num(total_params)}")
    logger.info(f"    Number of trainable parameters: {format_num(trainable_params)} ({(trainable_params / total_params) * 100:.1f}%)")
    logger.info(f"    Size: {param_size / (1024 ** 2):.1f} MB")
    logger.info(f"    Type: {next(module.parameters()).dtype}")


def load_partial_checkpoints(model, partial_checkpoints: dict):
    """
    Loads partial checkpoints for specific model modules.

    Args:
        partial_checkpoints (dict): A dictionary where keys are model module names
                                    (e.g., "encoder") and values are paths to their checkpoints.
    """
    for module_name, checkpoint_path in partial_checkpoints.items():
        if not hasattr(model, module_name):
            logger.warning(
                f"Model has no attribute '{module_name}'. Skipping loading checkpoint from {checkpoint_path}.")
            continue

        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint path '{checkpoint_path}' does not exist. Skipping.")
            continue

        module = getattr(model, module_name)
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if 'state_dict' in checkpoint:
                module.load_state_dict(checkpoint['state_dict'])
            else:
                module.load_state_dict(checkpoint)
            logger.info(f"Loaded checkpoint for '{module_name}' from {checkpoint_path}.")
        except Exception as e:
            logger.error(f"Failed to load checkpoint for '{module_name}' from {checkpoint_path}: {e}")

    return model