import torch
import torch.nn as nn


class InputProjection(nn.Module):
    def __init__(self, input_dim, projected_dim, dropout_rate=0.1, dropout_first=True):
        """
        Initializes the InputProjection module.

        Args:
            input_dim (int): Dimension of the input features.
            projected_dim (int): Dimension of the projected features.
            dropout_rate (float): Dropout probability.
            dropout_first (bool): If True, apply dropout before projection, otherwise after.
        """
        super().__init__()
        self.projection = nn.Linear(input_dim, projected_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # Preconfigure the processing pipeline
        if dropout_first:
            self.process = nn.Sequential(
                self.dropout,
                self.projection
            )
        else:
            self.process = nn.Sequential(
                self.projection,
                self.dropout
            )

    def forward(self, x):
        """
        Forward pass through the InputProjection module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, input_dim).

        Returns:
            torch.Tensor: Projected tensor of shape (batch, projected_dim).
        """
        return self.process(x)
