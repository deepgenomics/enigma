import functools

import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


class Residual(nn.Module):
    """Residual connection wrapper: output = input + block(input)

    Args:
        block: nn.Module to wrap with residual connection
    """

    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return x + self.block(x, **kwargs)


class Activation(nn.Module):
    """Module for commonly used activation functions

    Converts activation function names to activation functions.

    Args:
        name: Activation function name. Supported: gelu, gelu_tanh, sigmoid, silu,
            softplus, relu, none
    """

    MAPPING = {
        "gelu": F.gelu,
        "gelu_tanh": functools.partial(F.gelu, approximate="tanh"),
        "sigmoid": F.sigmoid,
        "silu": F.silu,
        "softplus": F.softplus,
        "relu": F.relu,
        "none": nn.Identity(),
    }

    def __init__(self, name: str):
        super().__init__()

        if name not in self.MAPPING:
            raise ValueError(f"Unknown activation function: {name}")

        self.name = name
        self.activation = self.MAPPING[name]

    def forward(self, input: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return self.activation(input)

    def extra_repr(self) -> str:
        return f"activation={self.name}"


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit (SwiGLU)

    Implements SwiGLU activation: SiLU(W_gate @ x) ⊙ (W_up @ x)
    Uses single linear layer outputting 2*dim_out for efficiency.

    Args:
        dim_in: Input embedding dimension
        dim_out: Output embedding dimension
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        # Single linear layer outputs both gate and value projections
        self.linear = nn.Linear(dim_in, dim_out * 2, bias=False)

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        x = self.linear(x)
        x, x_gate = x.chunk(2, dim=-1)  # Split into value and gate
        return F.silu(x_gate) * x  # SwiGLU: SiLU(gate) * value


class MLP(nn.Module):
    """Multi-Layer Perceptron with configurable activation

    Standard two-layer MLP: Linear -> Activation -> Dropout -> Linear
    Supports both standard activations and SwiGLU with parameter matching.

    Args:
        dim: Input/output embedding dimension
        activation: Activation function name
        expansion: Hidden dimension multiplier (hidden_dim = dim * expansion)
        swiglu_match_params: If True and using SwiGLU, scale hidden_dim by 2/3 to
            match parameter count of standard MLP
        dropout: Dropout probability applied after first layer
        bias: Whether to use bias in linear layers
    """

    def __init__(
        self,
        dim: int,
        activation: str = "swiglu",
        expansion: int = 4,
        swiglu_match_params: bool = False,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()

        hidden_dim = dim * expansion

        if activation == "swiglu":
            if swiglu_match_params:
                # Scale hidden dimension by 2/3 to match parameter count of standard
                # FFN. SwiGLU uses 3 weight matrices vs 2 in standard FFN, so reducing
                # hidden_dim by 2/3 keeps total parameters approximately equal.
                #
                # 1. "GLU Variants Improve Transformer" (Shazeer, 2020):
                #    https://arxiv.org/abs/2002.05202
                #    "To keep the number of parameters and computation constant, we
                #    reduce the number of hidden units by a factor of 2/3"
                #
                # 2. Meta's LLaMA implementation:
                #    https://github.com/meta-llama/llama/blob/main/llama/model.py
                #    Uses: hidden_dim = int(2 * hidden_dim / 3)

                hidden_dim = int(2 / 3 * hidden_dim)

            self.fc_1 = SwiGLU(dim, hidden_dim)
        else:
            self.fc_1 = nn.Sequential(
                nn.Linear(dim, hidden_dim, bias=bias),
                Activation(activation),
            )

        self.fc_2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Float[Tensor, "... l d"]) -> Float[Tensor, "... l d"]:
        return self.fc_2(self.dropout(self.fc_1(x)))