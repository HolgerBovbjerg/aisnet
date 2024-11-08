from typing import Optional
import torch
from torch import nn
from torch.nn.functional import relu, mse_loss


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def variance_loss(z_a: torch.Tensor, z_b: torch.Tensor):
    """
    Variance loss as defined in VICReg
    :param z_a: prediction
    :param z_b: target
    :return: variance loss
    """
    std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-6).mean()
    std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-6).mean()
    var_loss = torch.mean(relu(1 - std_z_a)) + torch.mean(relu(1 - std_z_b))
    return var_loss


def covariance_loss(x: torch.Tensor, y: torch.Tensor, batch_size: int):
    num_features = x.size(-1)
    cov_x = (x.T @ x) / (batch_size - 1)
    cov_y = (y.T @ y) / (batch_size - 1)
    cov_loss = (off_diagonal(cov_x).pow_(2).sum().div(num_features)
                + off_diagonal(cov_y).pow_(2).sum().div(num_features))
    return cov_loss


class VICRegLoss(nn.Module):
    def __init__(self, ignore_index: int = -1, weights: Optional[tuple] = None):
        super().__init__()
        self.ignore_index = ignore_index
        self.weights = weights

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = mse_loss(pred, target)
        var = variance_loss(pred, target)
        covar = 0
        if self.weights is not None:
            return mse * self.weights[0] + var * self.weights[1] + covar * self.weights[2]
        else:
            return mse + var + covar
