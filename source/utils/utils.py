from math import sqrt, acos, atan2, degrees, sin, cos, radians
import os
import random
from typing import Optional
import logging
from logging.handlers import RotatingFileHandler
import sys

import torch
from torch import nn, optim
import numpy as np
import wandb
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


logger = logging.getLogger(__name__)

def cartesian_to_spherical(x, y, z):
    # Calculate the radius
    r = sqrt(x ** 2 + y ** 2 + z ** 2)

    # Calculate the polar angle (theta)
    theta = degrees(acos(z / r))

    # Calculate the azimuthal angle (phi)
    phi = degrees(atan2(y, x))

    return r, theta, phi


def spherical_to_cartesian(r, theta, phi):
    # Convert angles from degrees to radians
    theta = radians(theta)
    phi = radians(phi)

    # Calculate the Cartesian coordinates
    x = r * sin(theta) * cos(phi)
    y = r * sin(theta) * sin(phi)
    z = r * cos(theta)

    return x, y, z



def setup_logger(log_dir: str, log_name: str = "training.log", log_level: str = "info"):
    # Create handlers
    stdoutHandler = logging.StreamHandler(stream=sys.stdout)
    fileHandler = RotatingFileHandler(os.path.join(log_dir, log_name), backupCount=5, maxBytes=5000000)

    # Create formatter
    stdoutFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                        datefmt="%Y-%m-%dT%H:%M:%SZ")
    stdoutHandler.setFormatter(stdoutFormatter)
    fileHandler.setFormatter(stdoutFormatter)

    # Add handlers to logger
    logger.addHandler(stdoutHandler)
    logger.addHandler(fileHandler)

    # Set logging level
    if log_level == "debug":
        logger.setLevel(logging.DEBUG)
    elif log_level == "info":
        logger.setLevel(logging.INFO)
    else:
        logger.error("Invalid log level, setting to log level to DEBUG")
        logger.setLevel(logging.DEBUG)
    return logger


def get_device(device) -> torch.device:
    """
    Get a torch.device from a string.
    :param device: Name of device (auto, cpu, cuda or mps)"
    :return: torch.device
    """
    if device == "auto":
        logger.info("Automatically selecting device.")
        logger.info("Checking CUDA availability...")
        if torch.cuda.is_available():
            out = torch.device("cuda")
            logger.info(f"Using device: {out}")
            logger.info("CUDA Device Name: %s", torch.cuda.get_device_name(0))
            logger.info("CUDA Version: %s", torch.version.cuda)
            return out
        logger.info("No CUDA device available.")
        logger.info("Checking MPS availability...")
        if torch.torch.backends.mps.is_available():
            out = torch.device("mps")
            logger.info(f"Using device: {out}")
            return out
        else:
            logger.info("No CUDA or MPS Device available, using CPU.")
            out = torch.device("cpu")
            logger.info(f"Using device: {out}")
            return out
    else:
        try:
            out = torch.device(device)
            logger.info(f"Using device: {out}")
            return out
        except ValueError:
            logger.error(f"Unknown device name: {device}. Falling back to using cpu.", exc_info=True)
            return torch.device("cpu")


def freeze_model_parameters(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model_parameters(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        count += sum(f.endswith(extension) for f in files)
    return count


def nearest_interp(x_interpolate, x_original, y_original):
    x_distance = torch.abs(x_original - x_interpolate[:, None])
    y_interpolate = y_original[torch.argmin(x_distance, dim=1)]
    return y_interpolate


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


def seed_everything(seed: int) -> None:
    """
    Set manual seed of Python, NumPy, PyTorch, and CUDA.
    :param seed: Supplied seed.
    :return: None
    """

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Seed set to {seed}")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def save_model(epoch: int, score: float, save_path: str, net: nn.Module, optimizer: Optional[optim.Optimizer] = None,
               log_file: Optional[str] = None, **kwargs) -> None:
    """Saves checkpoint.
    Args:
        epoch (int): Current epoch.
        score (float): Validation accuracy.
        save_path (str): Checkpoint path.
        net (nn.Module): Model instance.
        optimizer (optim.Optimizer, optional): Optimizer. Defaults to None.
        log_file (str, optional): Log file. Defaults to None.
    """

    ckpt_dict = {
        "epoch": epoch,
        "score": score,
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else optimizer
    }
    ckpt_dict.update(kwargs)

    torch.save(ckpt_dict, save_path)

    log_message = f"Saved {save_path} with score {score}."
    print(log_message)

    if log_file is not None:
        with open(log_file, "a+", encoding="utf8") as file:
            file.write(log_message + "\n")


def plot_grad_flow(named_parameters):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(max_grads)), max_grads, alpha=0.3, lw=1, color="c", label="max gradient")
    ax.bar(np.arange(len(max_grads)), ave_grads, alpha=0.3, lw=1, color="b", label="mean gradient")
    ax.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k", label="zero gradient")
    ax.set_xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    ax.set_xlim(left=0, right=len(ave_grads))
    #ax.set_ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    ax.set_xlabel("Layers")
    ax.set_ylabel("Gradient")
    ax.set_title("Gradient flow")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig, ax

def get_local_rank():
    rank = int(os.environ["SLURM_PROCID"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    return rank - gpus_per_node * (rank // gpus_per_node)
