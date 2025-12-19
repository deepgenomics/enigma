import re

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from .common import Activation


class ConvNorm(nn.Module):
    """Normalizations that are commonly used for 1D conv layers

    Args:
          name: Normalization type. Supported values:
              - 'batch': BatchNorm1d
              - 'group_N': GroupNorm with N groups (e.g., 'group_8')
          dim: Number of input channels/features
          affine: Whether to use learnable affine parameters
    """

    def __init__(self, name: str, dim: int, affine: bool = True) -> None:
        super().__init__()

        if name == "batch":
            self.norm = nn.BatchNorm1d(num_features=dim, affine=affine)
        elif match := re.match(r"group_(\d+)", name):
            num_groups = int(match.group(1))

            if dim % num_groups != 0:
                raise ValueError(
                    f"dim {dim} must be divisible by num_groups {num_groups}"
                )

            self.norm = nn.GroupNorm(
                num_groups=num_groups, num_channels=dim, affine=affine
            )
        else:
            raise ValueError(f"Unknown normalization type: {name}")

    def forward(self, x: Float[Tensor, "b d l"]) -> Float[Tensor, "b d l"]:
        return self.norm(x)


class ConvBlock(nn.Module):
    """Conv block with Pre-Norm architecture: Norm -> Activation -> Conv -> Dropout

    Args:
        dim_in: Input channels
        dim_out: Output channels
        kernel_size: Convolution kernel size
        norm: Normalization type (see ConvNorm for options)
        activation: Activation function name (see common.Activation)
        dropout: Dropout probability (0.0 to 1.0)
        separable: If True, use depthwise separable convolution
        bias: Whether to use bias in convolution layers
        norm_affine: Whether to use learnable parameters in normalization
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        kernel_size: int,
        norm: str,
        activation: str,
        dropout: float = 0.0,
        separable: bool = False,
        bias: bool = False,
        norm_affine: bool = True,
    ):
        super().__init__()

        self.norm = ConvNorm(name=norm, dim=dim_in, affine=norm_affine)
        self.activation = Activation(name=activation)

        if separable:
            self.conv = nn.Sequential(
                nn.Conv1d(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    groups=dim_in,
                    padding="same",
                    bias=bias,
                ),
                nn.Conv1d(dim_in, dim_out, kernel_size=1, bias=bias),
            )
        else:
            self.conv = nn.Conv1d(
                dim_in, dim_out, kernel_size=kernel_size, padding="same", bias=bias
            )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Float[Tensor, "b d l"]) -> Float[Tensor, "b d l"]:
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)

        x = self.dropout(x)

        return x
