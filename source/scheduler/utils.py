from torch import optim
from torch.optim import lr_scheduler

from .cosine_annealing_warm_restarts import CosineAnnealingWarmupRestarts
from .warmup_lr import WarmUpLR


def get_scheduler(optimizer: optim.Optimizer, scheduler_type: str, t_max: int, **kwargs) -> lr_scheduler._LRScheduler:
    """Gets scheduler.
    Args:
        optimizer (optim.Optimizer): Optimizer instance.
        scheduler_type (str): Specified scheduler.
        t_max (int): Number of steps.
    Raises:
        ValueError: Unsupported scheduler type.
    Returns:
        lr_scheduler._LRScheduler: Scheduler instance.
    """

    if scheduler_type == "reduce_on_plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, **kwargs)
    elif scheduler_type == "cosine_annealing":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=t_max, **kwargs)
    elif scheduler_type == "1cycle":
        scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer, total_steps=t_max, **kwargs)
    elif scheduler_type == "cosine_annealing_warmup_restarts":
        scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer, **kwargs)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    return scheduler


if __name__ == "__main__":
    import torch

    net = torch.nn.Linear(64, 10)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()))

    scheduler = get_scheduler(optimizer=optimizer, scheduler_type="cosine_annealing_warmup_restarts", t_max=50, max_lr=0.03,
                              first_cycle_steps=10)

    for _ in range(50):
        scheduler.step()

    print("done")

