import math
from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange

try:
    from flash_attn.modules.mha import FlashSelfAttention
    from flash_attn.ops.triton.rotary import apply_rotary
except ImportError:
    FlashSelfAttention = None
    apply_rotary = None

from jaxtyping import Bool, Float, Integer
from torch import Tensor

# Rotary embedding =====================================================================
# Adapted from flash_attn trition based rotary to support variable sequence length
# https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/layers/rotary.py


class ApplyRotaryEmbQKV_(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv: Float[Tensor, "b l 3 h d"] | Float[Tensor, "(b l) 3 h d"],
        cos: Float[Tensor, "l d"],
        sin: Float[Tensor, "l d"],
        interleaved: bool = False,
        cu_seqlens: Integer[Tensor, " b+1"] | None = None,
        max_seqlen: int | None = None,
    ):
        if cu_seqlens is not None:
            assert (
                qkv.ndim == 4 and qkv.shape[1] == 3
            ), "qkv must be of shape ((b l), 3, h, d)"
            assert (
                max_seqlen is not None
            ), "max_seqlen must be provided if cu_seqlens is provided"
            q, k = qkv[:, 0], qkv[:, 1]
        else:
            assert (
                qkv.ndim == 5 and qkv.shape[2] == 3
            ), "qkv must be of shape (b, l, 3, h, d)"
            q, k = qkv[:, :, 0], qkv[:, :, 1]

        apply_rotary(
            q,
            cos,
            sin,
            interleaved=interleaved,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            inplace=True,
        )
        apply_rotary(
            k,
            cos,
            sin,
            interleaved=interleaved,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            inplace=True,
        )

        ctx.save_for_backward(cos, sin, cu_seqlens)
        ctx.interleaved = interleaved
        ctx.max_seqlen = max_seqlen

        return qkv

    @staticmethod
    def backward(ctx, dqkv):
        max_seqlen = ctx.max_seqlen
        cos, sin, cu_seqlens = ctx.saved_tensors

        # Fix dimension handling to match the forward pass
        if cu_seqlens is not None:
            # For unpadded input case
            assert (
                dqkv.ndim == 4
            ), f"dqkv must be of shape ((b l), 3, h, d), got {dqkv.shape}"
            dq, dk = dqkv[:, 0], dqkv[:, 1]
        else:
            # For standard batched input case
            assert (
                dqkv.ndim == 5
            ), f"dqkv must be of shape (b, l, 3, h, d), got {dqkv.shape}"
            dq, dk = dqkv[:, :, 0], dqkv[:, :, 1]

        apply_rotary(
            dq,
            cos,
            sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            interleaved=ctx.interleaved,
            inplace=True,
            conjugate=True,
        )
        apply_rotary(
            dk,
            cos,
            sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            interleaved=ctx.interleaved,
            inplace=True,
            conjugate=True,
        )

        return dqkv, None, None, None, None, None


def apply_rotary_emb_qkv_(
    qkv: Float[Tensor, "b l 3 h d"] | Float[Tensor, "(b l) 3 h d"],
    cos: Float[Tensor, "l d"],
    sin: Float[Tensor, "l d"],
    interleaved: bool = False,
    cu_seqlens: Integer[Tensor, " b+1"] | None = None,
    max_seqlen: int | None = None,
) -> Float[Tensor, "b l 3 h d"] | Float[Tensor, "(b l) 3 h d"]:
    """Apply rotary embedding *inplace* to the first rotary_dim of Q and K.

    Args:
        qkv: query, key, value.
        cos, sin: (seqlen, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style)
            instead of 1st half and 2nd half (GPT-NeoX style).
        cu_seqlens: (batch_size + 1) the cumulative sum of the sequence lengths
        max_seqlen: int the maximum sequence length in the batch
    Return:
        qkv: query, key, value with rotary embedding applied.
    """
    return ApplyRotaryEmbQKV_.apply(qkv, cos, sin, interleaved, cu_seqlens, max_seqlen)


class RotaryEmbedding(torch.nn.Module):
    """The rotary position embeddings from RoFormer_ (Su et. al).

    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox

    If `scale_base` is not None, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).  # noqa E501
    A recommended value for `scale_base` is 512: https://github.com/HazyResearch/flash-attention/issues/96  # noqa E501
    Reference: https://github.com/sunyt32/torchscale/blob/main/torchscale/component/xpos_relative_position.py  # noqa E501

    If `pos_idx_in_fp32` is True, the position indices [0.0, ..., seqlen - 1] are in
    fp32, otherwise they might be in lower precision. This option was added because
    previously (before 2023-07-02), when we construct the position indices, we use the
    dtype of self.inv_freq. In most cases this would be fp32, but if the model is
    trained in pure bf16 (not mixed precision), then self.inv_freq would be bf16, and
    the position indices are also in bf16. Because of the limited precision of bf16
    (e.g. 1995.0 is rounded to 2000.0), the embeddings for some positions will coincide.
    To maintain compatibility with models previously trained in pure bf16, we add this
    option.
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        interleaved: bool = False,
        pos_idx_in_fp32: bool = True,
        device: str | torch.device | None = None,
    ) -> None:
        """ """
        super().__init__()

        self.dim = dim
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32

        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.interleaved = interleaved

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _compute_inv_freq(
        self, device: str | torch.device | None = None
    ) -> Float[Tensor, " d"]:
        return 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                / self.dim
            )
        )

    def _update_cos_sin_cache(
        self,
        seqlen: int,
        device: str | torch.device | None = None,
        dtype: str | torch.dtype | None = None,
    ) -> None:
        # Reset the tables if the sequence length has changed,
        # if we're on a new device (possibly due to tracing for instance),
        # or if we're switching from inference mode to training
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            # We want fp32 here, not self.inv_freq.dtype, since the model could be
            # loaded in bf16 and the output of arange can be quite large, so bf16
            # would lose a lot of precision.
            # However, for compatibility reason, we add an option to use the dtype of
            # self.inv_freq.
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                # We want fp32 here as well since inv_freq will be multiplied with t,
                # and the output will be large. Having it in bf16 will lose a lot of
                # precision and cause the cos & sin output to change significantly.
                # We want to recompute self.inv_freq if it was not loaded in fp32
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self._compute_inv_freq(device=device)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                inv_freq = self.inv_freq

            freqs = torch.outer(t, inv_freq)
            self._cos_cached = torch.cos(freqs).to(dtype)
            self._sin_cached = torch.sin(freqs).to(dtype)

    def forward(
        self,
        qkv: Float[Tensor, "b l 3 h d"] | Float[Tensor, "(b l) 3 h d"],
        cu_seqlens: Integer[Tensor, " b+1"] | None = None,
        max_seqlen: int | None = None,
    ) -> Float[Tensor, "b l 3 h d"] | Float[Tensor, "(b l) 3 h d"]:
        """Apply rotary embedding *inplace* to qkv.

        This implementation supports both padded and unpadded inputs.

        Args:
            qkv: query, key, value.
            cu_seqlens: (batch + 1) the cumulative sum of the sequence lengths.
            max_seqlen: int the maximum sequence length in the batch.
        Return:
            qkv: query, key, value with rotary embedding applied.
        """
        if max_seqlen is None:
            assert (
                qkv.ndim == 5 and qkv.shape[2] == 3
            ), f"Expected qkv to be of shape (b, l, 3, h, d), got {qkv.shape}"
            max_seqlen = qkv.shape[1]

        self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)

        return apply_rotary_emb_qkv_(
            qkv,
            self._cos_cached,
            self._sin_cached,
            interleaved=self.interleaved,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )


# End of rotary embedding ============================================================


# From
# https://github.com/ofirpress/attention_with_linear_biases/blob/4b92f28a005ead2567abe2359f633e73e08f3833/fairseq/models/transformer.py#L742
def get_alibi_slopes(nheads):
    def get_slopes_power_of_2(nheads):
        start = 2 ** (-(2 ** -(math.log2(nheads) - 3)))
        ratio = start
        return [start * ratio**i for i in range(nheads)]

    if math.log2(nheads).is_integer():
        return get_slopes_power_of_2(nheads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(nheads))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_alibi_slopes(2 * closest_power_of_2)[0::2][
                : nheads - closest_power_of_2
            ]
        )


class SelfAttention(nn.Module):
    """Implement the scaled dot product attention with softmax when FlashAttention
    is not used

    This is adapted from
    https://github.com/Dao-AILab/flash-attention/blob/98edb0d29bb1db336fef845fb5fd49bc98b04b96/flash_attn/modules/mha.py#L230

    Arguments
    ---------
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, attention_dropout=0.0):
        super().__init__()

        self.drop = nn.Dropout(attention_dropout)

    def forward(self, qkv, key_padding_mask=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D)
            key_padding_mask: boolean mask to apply to the attention weights. True means
                to keep, False means to mask out. (B, S)
        """
        batch_size, seqlen = qkv.shape[0], qkv.shape[1]
        q, k, v = qkv.unbind(dim=2)
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)

        if key_padding_mask is not None:
            padding_mask = torch.full(
                (batch_size, seqlen), -10000.0, dtype=scores.dtype, device=scores.device
            )
            padding_mask.masked_fill_(key_padding_mask, 0.0)
            # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)  # noqa E501
            scores = scores + rearrange(padding_mask, "b s -> b 1 1 s")

        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = self.drop(attention)

        output = torch.einsum("bhts,bshd->bthd", attention_drop, v)

        return output


class MHA(nn.Module):
    """Multi-head self-attention using FlashAttention.

    This is adapted from
    https://github.com/Dao-AILab/flash-attention/blob/934f6ad714691a21a09b78c3e19a2378917e9cba/flash_attn/modules/mha.py#L373
    to support unpadded variable length inputs with RoPE positional encoding.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int,
        use_flash_attn: bool = True,
        use_rope: bool = True,
        rope_base: float = 20000.0,
        rope_interleaved: bool = False,
        use_alibi: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
        dropout: float = 0.0,
        use_qk_norm: bool = False,
        bias: bool = False,
        device: str | torch.device | None = None,
    ):
        super().__init__()

        if use_flash_attn:
            if FlashSelfAttention is None or apply_rotary is None:
                raise ImportError(
                    "`flash_attn` must be installed when `use_flash_attn=True`"
                )
        else:
            if use_alibi:
                raise ValueError(
                    "`use_alibi=True` is only supported when `use_flash_attn=True`"
                )
            if window_size != (-1, -1):
                raise ValueError(
                    "`window_size` is only supported when `use_flash_attn=True`"
                )
        self.use_flash_attn = use_flash_attn

        if dim % head_dim != 0:
            raise ValueError(
                f"dim must be divisible by head_dim, got dim={dim} and "
                f"head_dim={head_dim}"
            )

        self.num_heads = dim // head_dim

        self.use_rope = use_rope
        if self.use_rope:
            rope_dim = head_dim  # use head_dim for RoPE
            self.rotary_emb = RotaryEmbedding(
                dim=rope_dim,
                base=rope_base,
                interleaved=rope_interleaved,
                device=device,
            )

        self.use_alibi = use_alibi
        if self.use_alibi:
            alibi_slopes = torch.tensor(get_alibi_slopes(self.num_heads), device=device)
        else:
            alibi_slopes = None

        if self.use_flash_attn:
            self.self_attn = FlashSelfAttention(
                attention_dropout=dropout,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
            )
        else:
            self.self_attn = SelfAttention(attention_dropout=dropout)

        qkv_dim = head_dim * self.num_heads * 3
        self.Wqkv = nn.Linear(dim, qkv_dim, bias=bias)

        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(head_dim)
            self.k_norm = nn.RMSNorm(head_dim)

        self.out_proj = nn.Linear(dim, dim, bias=bias)

    def forward(
        self,
        x: Float[Tensor, "b l d"] | Float[Tensor, "total_tokens d"],
        key_padding_mask: Bool[Tensor, "b l"] | None = None,
        cu_seqlens: Integer[Tensor, " b+1"] | None = None,
        max_seqlen: int | None = None,
    ):
        """
        If `cu_seqlens` and `max_seqlen` are provided, the input is assumed to
        be unpadded (i.e. shape of (total_tokens, d)).

        Args:
            x: (batch, seqlen, hidden_dim) or (total_tokens, hidden_dim) if unpadded
            key_padding_mask: boolean mask, True means to keep, False means to mask out.
                (batch, seqlen). Only applicable when not using FlashAttention.
            cu_seqlens: Tensor of shape (batch_size + 1,) containing the cumulative
                sequence lengths of the sequences in the batch, used to index into qkv.
            max_seqlen: Maximum sequence length in the batch.
        """
        # First check if the input is valid
        if cu_seqlens is not None:
            assert max_seqlen is not None
            assert key_padding_mask is None
            assert self.use_flash_attn
        if key_padding_mask is not None:
            assert cu_seqlens is None
            assert max_seqlen is None
            assert not self.use_flash_attn

        # Projection to qkv
        qkv = self.Wqkv(x)
        qkv = rearrange(
            qkv, "... (three h d) -> ... three h d", three=3, h=self.num_heads
        )

        if self.use_rope:
            qkv = self.rotary_emb(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        if self.use_qk_norm:
            qkv = self._apply_qk_norm(qkv)

        # To support both with and without FlashAttention, create a kwargs for self_attn
        kwargs = (
            {"cu_seqlens": cu_seqlens, "max_seqlen": max_seqlen}
            if self.use_flash_attn
            else {"key_padding_mask": key_padding_mask}
        )
        output = self.self_attn(qkv, **kwargs)

        # Projection to output
        out = rearrange(output, "... h d -> ... (h d)")
        out = self.out_proj(out)

        return out

    def _apply_qk_norm(
        self, qkv: Float[Tensor, "... three h d"]
    ) -> Float[Tensor, "... three h d"]:
        """Apply QK normalization to stacked `qkv`

        FlashAttention can be more efficient if it is already stacked [1]
        (although I have not tested whether this is true after the overhead of this
        operation). Therefore, rather than splitting `qkv` into `q`, `k`, and `v`,
        apply the normalization to the stacked `qkv` and return the stacked `qkv`
        again.

        [1] https://github.com/Dao-AILab/flash-attention/blob/04adaf0e9028d4bec7073f69e4dfa3f6d3357189/flash_attn/flash_attn_interface.py#L1236-L1238  # noqa E501
        """
        q, k, v = qkv.unbind(dim=-3)  # return views so shouldn't be much overhead
        q = self.q_norm(q)
        k = self.k_norm(k)

        return torch.stack(
            [q, k, v], dim=-3
        )  # stack's overhead should be minimal compared to the other operations
