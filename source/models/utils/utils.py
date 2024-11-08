import os

from torch import nn


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



