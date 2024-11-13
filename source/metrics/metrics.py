import torch
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import numpy as np


def angular_error(pred: torch.Tensor, true: torch.Tensor):
    cos_sim = torch.sum(pred * true, dim=-1) / (torch.norm(pred, dim=-1) * torch.norm(true, dim=-1))
    return torch.acos(torch.clamp(cos_sim, -1.0, 1.0))


def geodesic_distance(pred: torch.Tensor, true: torch.Tensor):
    dot_product = torch.sum(pred * true, dim=-1)
    return torch.acos(torch.clamp(dot_product, -1.0, 1.0))


def angular_precision(errors: torch.Tensor):
    return torch.var(errors, dim=-1)


def root_mean_square_angular_error(errors: torch.Tensor):
    return torch.sqrt(torch.mean(errors ** 2))


def doa_threshold_accuracy(errors: torch.Tensor, threshold_radians: float):
    within_threshold = errors < threshold_radians
    return torch.mean(within_threshold.float())


def percentile_error(errors: torch.Tensor, percentile=90):
    return torch.quantile(errors, percentile / 100.0)


def mse(pred: torch.Tensor, true: torch.Tensor):
    return torch.mean((pred - true) ** 2)


def equal_error_rate(pred: np.ndarray, target: np.ndarray):
    # EER (not backpropagated)

    # Snippet from https://yangcha.github.io/EER-ROC/
    fpr, tpr, thresholds = roc_curve(target.flatten(), pred.flatten())
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

def accuracy(predictions: torch.Tensor, targets: torch.Tensor):
    predicted_classes = predictions.argmax(dim=-1)
    correct = (predicted_classes == targets).sum().item()
    return correct / len(targets)