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


if __name__ == "__main__":
    wpl = WeightedPairwiseLoss()

    z1 = torch.tensor([[0.2, 0.2, 0.6]])
    y1 = torch.tensor([0])

    loss1 = wpl(z1, y1).item()

    z2 = torch.tensor([[0.2, 0.2, 0.6]])
    y2 = torch.tensor([1])

    loss2 = wpl(z2, y2).item()

    z3 = torch.tensor([[0.2, 0.2, 0.6]])
    y3 = torch.tensor([2])

    loss3 = wpl(z3, y3).item()

    z4 = torch.tensor([[0.2, 0.6, 0.2]])
    y4 = torch.tensor([0])

    loss4 = wpl(z4, y4).item()

    z5 = torch.tensor([[0.2, 0.6, 0.2]])
    y5 = torch.tensor([1])

    loss5 = wpl(z5, y5).item()

    z6 = torch.tensor([[0.2, 0.6, 0.2]])
    y6 = torch.tensor([2])

    loss6 = wpl(z6, y6).item()

    z7 = torch.tensor([[0.6, 0.2, 0.2]])
    y7 = torch.tensor([0])

    loss7 = wpl(z7, y7).item()

    z8 = torch.tensor([[0.6, 0.2, 0.2]])
    y8 = torch.tensor([1])

    loss8 = wpl(z8, y8).item()

    z9 = torch.tensor([[0.6, 0.2, 0.2]])
    y9 = torch.tensor([2])

    loss9 = wpl(z9, y9).item()


    print("done")
