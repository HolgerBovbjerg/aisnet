import torch
from torch import nn

from .weighted_pairwise import WeightedPairwiseLoss
from .vicreg_loss import VICRegLoss


def get_loss(name: str, **kwargs):
    loss_classes = {
        "weighted_pairwise": WeightedPairwiseLoss,
        "cross_entropy": nn.CrossEntropyLoss,
        "l1": nn.L1Loss,
        "smooth_l1": nn.SmoothL1Loss,
        "MSE": nn.MSELoss,
        "VICReg": VICRegLoss
    }

    if name in loss_classes:
        return loss_classes[name](**kwargs)

    raise ValueError(f"Loss with name: {name}, not supported.")
