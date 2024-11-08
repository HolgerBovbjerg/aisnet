import torch
from torch.optim import lr_scheduler


class WarmUpLR(lr_scheduler._LRScheduler):
    """WarmUp learning rate scheduler.
    Args:
        optimizer (optim.Optimizer): Optimizer instance
        total_iters (int): steps_per_epoch * n_warmup_epochs
        last_epoch (int): Final epoch. Defaults to -1.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, total_iters: int, last_epoch: int = -1):
        """Initializer for WarmUpLR"""

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Learning rate will be set to base_lr * last_epoch / total_iters."""

        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
