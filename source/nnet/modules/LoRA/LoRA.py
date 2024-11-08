from typing import Tuple, Union, Any

from torch import nn, Tensor


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer for projecting inputs through a low-rank matrix.

    This class implements a LoRA layer that reduces the dimensionality of the input
    tensor using a low-rank matrix, then projects it back to the original output dimension.
    It also includes optional dropout for regularization.

    Attributes:
        lora_a (nn.Linear): Linear layer projecting inputs to a smaller rank.
        lora_b (nn.Linear): Linear layer projecting from the smaller rank to the output dimension.
        rank (int): Rank of the low-rank matrices.
        alpha (float): Scaling factor for the low-rank adaptation.
        dropout (nn.Dropout): Dropout layer for regularization.

    Methods:
        forward(x: Tensor) -> Tensor: Projects the input tensor through the low-rank
            matrices and returns the output.

    Args:
        in_dim (int, optional): Input dimension size for the LoRA layer. Default is 512.
        out_dim (int, optional): Output dimension size for the LoRA layer. Default is 512.
        rank (int, optional): Rank of the low-rank matrices used in the LoRA layer. Default is 16.
        alpha (float, optional): Scaling factor for the low-rank adaptation. Default is 1.0.
        dropout (float, optional): Dropout rate for regularization. Default is 0.0.
    """

    def __init__(self,
                 in_dim: int = 512,
                 out_dim: int = 512,
                 rank: int = 16,
                 alpha: float = 1.0,
                 dropout: float = 0.0):
        super().__init__()

        # These are the new LoRA params. In general rank << in_dim, out_dim
        self.lora_a = nn.Linear(in_dim, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_dim, bias=False)

        # Rank and alpha are commonly-tuned hyperparameters
        self.rank = rank
        self.alpha = alpha

        # Most implementations also include some dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Projects the input tensor through the low-rank matrices and returns the output.

        Args:
            x (Tensor): Input tensor to be projected.

        Returns:
            Tensor: The projected output tensor.
        """
        # lora_a projects inputs down to the much smaller self.rank,
        # then lora_b projects back up to the output dimension
        lora_out = self.lora_b(self.lora_a(self.dropout(x)))
        return lora_out


class LoRA(nn.Module):
    """
    Low-Rank Adaptation (LoRA) module for adapting pre-trained nnet with low-rank updates.

    This class implements the LoRA technique which adapts a pre-trained model by adding a low-rank
    layer to it. The parameters of the original model are frozen, and only the parameters of the
    LoRA layer are trainable.

    Attributes:
        model (nn.Module): The original pre-trained model to be adapted.
        rank (int): Rank of the low-rank matrix used in the LoRA layer.
        alpha (float): Scaling factor for the low-rank adaptation.
        lora_layer (LoRALayer): The low-rank adaptation layer.

    Methods:
        _freeze_pretrained_model(): Freezes the parameters of the original pre-trained model.
        forward(x: Tensor, lengths) -> tuple[Union[float, Any], Any]: Performs a forward pass
            through the adapted model, applying the low-rank adaptation to the original model's
            outputs.

    Args:
        model (nn.Module): The original pre-trained model to be adapted.
        in_dim (int, optional): Input dimension size for the LoRA layer. Default is 512.
        out_dim (int, optional): Output dimension size for the LoRA layer. Default is 512.
        rank (int, optional): Rank of the low-rank matrix used in the LoRA layer. Default is 16.
        alpha (float, optional): Scaling factor for the low-rank adaptation. Default is 1.0.
        dropout (float, optional): Dropout rate for the LoRA layer. Default is 0.0.
    """

    def __init__(self,
                 model: nn.Module,
                 in_dim: int = 512,
                 out_dim: int = 512,
                 rank: int = 16,
                 alpha: float = 1.0,
                 dropout: float = 0.0):
        super().__init__()
        # The model which is LoRA adapted
        self.model = model

        # Rank and alpha are commonly-tuned hyperparameters
        self.rank = rank
        self.alpha = alpha

        # LoRA layer
        self.lora_layer = LoRALayer(in_dim, out_dim, rank, alpha, dropout)

        # The original params are frozen, and only LoRA params are trainable.
        self._freeze_pretrained_model()
        self.lora_layer.lora_a.weight.requires_grad = True
        self.lora_layer.lora_b.weight.requires_grad = True

    def _freeze_pretrained_model(self):
        """
        Freezes the parameters of the original pre-trained model.
        """
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x: Tensor, lengths) -> tuple[Union[float, Any], Any]:
        """
        Performs a forward pass through the adapted model, applying the low-rank adaptation
        to the original model's outputs.

        Args:
            x (Tensor): Input tensor to the model.
            lengths: Additional input specifying the lengths (can vary depending on model type).

        Returns:
            tuple[Union[float, Any], Any]: The adapted model's output along with the lengths.
        """
        # This would be the output of the original pretrained model
        pretrained_out, lengths = self.model(x, lengths)

        # Generate the low-rank adaption
        lora_out = self.lora_layer(x)

        # Finally, scale the low-rank adaption by the alpha parameter (normalized by rank)
        # and add to the original model's outputs
        return pretrained_out + (self.alpha / (self.rank + 1.e-9)) * lora_out, lengths


if __name__ == '__main__':
    import copy
    import torch

    model = nn.Linear(512, 512)
    lora_model = LoRA(copy.copy(model))
    input = torch.randn(1, 512, 512)
    output = model(input)
    output_lora = lora_model(input)
    print("done")
