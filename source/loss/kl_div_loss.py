import torch
import torch.nn as nn

class KLDivLoss(nn.Module):
    def __init__(self, reduction='batchmean'):
        super().__init__()
        self.kl_loss = nn.KLDivLoss(reduction=reduction)

    def forward(self, input, target):
        """
        Forward pass for the KLDivLoss. Wrapper around Pytorch version which expects input in log domain.
        Args:
            input (Tensor): Predicted probability distribution (not in log domain).
            target (Tensor): Target probability distribution.
        Returns:
            Tensor: Computed KL divergence loss.
        """
        # Convert input to log domain
        log_input = torch.log(input + 1e-8)  # Add epsilon to avoid log(0)
        return self.kl_loss(log_input, target)
