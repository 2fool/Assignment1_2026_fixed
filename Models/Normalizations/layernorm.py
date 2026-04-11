import torch
import torch.nn as nn
from typing import Union, List


class LayerNorm(nn.Module):
    """
    Channel-wise LayerNorm for channel-first tensors.

    For QANet we keep activations in shape [B, C, L]. Standard transformer-style
    layer normalization should normalize each position over the hidden/channel
    dimension only, not over both channel and sequence length together.

    This implementation therefore normalizes over dim=1 for inputs shaped
    [B, C, *], and applies one affine scale/bias per channel.
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int]],
        eps: float = 1e-5,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape]
        self.normalized_shape = list(normalized_shape)
        self.eps = eps

        if len(self.normalized_shape) != 1:
            raise ValueError(
                "This LayerNorm expects a single channel dimension, e.g. LayerNorm(C)."
            )

        channels = self.normalized_shape[0]
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError(f"Expected at least 2D input, got shape {tuple(x.shape)}")

        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        affine_shape = (1, self.weight.numel()) + (1,) * (x.ndim - 2)
        return x_norm * self.weight.view(affine_shape) + self.bias.view(affine_shape)
