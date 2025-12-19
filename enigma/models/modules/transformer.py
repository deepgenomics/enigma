from typing import Literal, Tuple

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from enigma.models.layers.attention import MHA
from enigma.models.layers.common import MLP


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture.

    Args:
        dim: Hidden dimension
        head_dim: Dimension per attention head
        expansion_factor: MLP hidden dimension multiplier
        norm: Normalization type ('ln' or 'rms')
        use_rope: Whether to use RoPE positional encoding
        rope_base: Base frequency for RoPE
        use_alibi: Whether to use ALiBi positional bias
        window_size: Attention window size (-1 for full attention)
        mlp_activation: MLP activation function
        mlp_dropout: Dropout rate in MLP
        mlp_swiglu_match_params: Match parameter count for SwiGLU
        attention_dropout: Dropout rate in attention
        post_attn_dropout: Dropout rate for attention residual
        post_mlp_dropout: Dropout rate for MLP residual
        use_qk_norm: Whether to use QK normalization
        use_flash_attn: Whether to use Flash Attention
        bias: Whether to use bias in linear layers
    """

    def __init__(
        self,
        dim: int,
        head_dim: int,
        expansion_factor: int = 4,
        norm: Literal["ln", "rms"] = "ln",
        use_rope: bool = True,
        rope_base: float = 10000.0,
        use_alibi: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
        mlp_activation: str = "gelu_tanh",
        mlp_dropout: float = 0.0,
        mlp_swiglu_match_params: bool = False,
        attention_dropout: float = 0.0,
        post_attn_dropout: float = 0.0,
        post_mlp_dropout: float = 0.0,
        use_qk_norm: bool = False,
        use_flash_attn: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.attn = MHA(
            dim=dim,
            head_dim=head_dim,
            use_flash_attn=use_flash_attn,
            use_rope=use_rope,
            rope_base=rope_base,
            use_alibi=use_alibi,
            window_size=window_size,
            dropout=attention_dropout,
            use_qk_norm=use_qk_norm,
            bias=bias,
        )
        self.post_attn_dropout = nn.Dropout(p=post_attn_dropout)
        self.post_mlp_dropout = nn.Dropout(p=post_mlp_dropout)

        self.mlp = MLP(
            dim=dim,
            activation=mlp_activation,
            expansion=expansion_factor,
            swiglu_match_params=mlp_swiglu_match_params,
            dropout=mlp_dropout,
        )

        if norm == "rms":
            norm_cls = nn.RMSNorm
        elif norm == "ln":
            norm_cls = nn.LayerNorm
        else:
            raise ValueError(f"Invalid norm: {norm}, must be one of 'ln' or 'rms'")

        self.attn_norm = norm_cls(dim)
        self.mlp_norm = norm_cls(dim)

    def forward(
        self,
        x: Float[Tensor, "b l d"],
    ) -> Float[Tensor, "b l d"]:
        h = x + self.post_attn_dropout(self.attn(self.attn_norm(x)))

        out = h + self.post_mlp_dropout(self.mlp(self.mlp_norm(h)))

        return out


class TransformerModule(nn.Module):
    """Stack of transformer blocks with optional gradient checkpointing.

    Args:
        num_blocks: Number of transformer blocks
        checkpointing: Whether to use gradient checkpointing for memory efficiency
        Other args: Passed through to each TransformerBlock (see TransformerBlock docs)
    """

    def __init__(
        self,
        num_blocks: int,
        dim: int,
        head_dim: int,
        expansion_factor: int = 4,
        norm: Literal["ln", "rms"] = "ln",
        use_rope: bool = True,
        rope_base: float = 10000.0,
        use_alibi: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
        mlp_activation: str = "gelu_tanh",
        mlp_dropout: float = 0.0,
        mlp_swiglu_match_params: bool = False,
        attention_dropout: float = 0.0,
        post_attn_dropout: float = 0.0,
        post_mlp_dropout: float = 0.0,
        use_qk_norm: bool = False,
        use_flash_attn: bool = True,
        bias: bool = False,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    head_dim=head_dim,
                    expansion_factor=expansion_factor,
                    norm=norm,
                    use_rope=use_rope,
                    rope_base=rope_base,
                    use_alibi=use_alibi,
                    window_size=window_size,
                    mlp_activation=mlp_activation,
                    mlp_dropout=mlp_dropout,
                    mlp_swiglu_match_params=mlp_swiglu_match_params,
                    attention_dropout=attention_dropout,
                    post_attn_dropout=post_attn_dropout,
                    post_mlp_dropout=post_mlp_dropout,
                    use_qk_norm=use_qk_norm,
                    use_flash_attn=use_flash_attn,
                    bias=bias,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self,
        x: Float[Tensor, "b l d"],
    ) -> Float[Tensor, "b l d"]:
        for block in self.blocks:
            x = block(x)

        return x
