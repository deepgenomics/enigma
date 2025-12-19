"""Utils for handling sequences"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

import torch
import torch.nn.functional as F
from jaxtyping import Float, Integer
from torch import Tensor

# Default encoding
BASE_ENCODING = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "N": 4,  # zero vector for unknown
}

BASE_COMPLEMENTS = {
    "A": "T",
    "C": "G",
    "G": "C",
    "T": "A",
    "N": "N",
}


# Tensor mapping of a base index to its complement index. Used for `reverse_complement`
COMPLEMENT_MAPPING = torch.zeros(len(BASE_ENCODING), dtype=torch.int8)
for base, comp_base in BASE_COMPLEMENTS.items():
    COMPLEMENT_MAPPING[BASE_ENCODING[base]] = BASE_ENCODING[comp_base]


# Functions for converting between different sequence representations =================


def str_to_idx(
    seq: str,
) -> Integer[Tensor, " l"]:
    """Convert a string to one-hot encoded indices"""
    return torch.tensor([BASE_ENCODING[base] for base in seq])


def idx_to_onehot(
    idx: Integer[Tensor, " l"] | Integer[Tensor, "b l"],
) -> Integer[Tensor, "l c"] | Integer[Tensor, "b l c"]:
    """Convert one-hot encoded indices to a one-hot encoded tensor

    Args:
        idx: Tensor of shape (l, ) or (b, l) containing the one-hot encoding
            indices of the sequence.

    Returns:
        One-hot encoded tensor of shape (l, c) or (b, l, c) where c is the number of
        classes.
    """
    num_classes = len(BASE_ENCODING) - 1  # number of bases, ignoring 'N'

    idx = idx.long()  # F.one_hot requires long integers

    # First, clamp to valid range for one-hot encoding and then applies zeroing out
    # for all 'N's and unknown indices.
    one_hot = F.one_hot(
        idx.clamp(
            min=0, max=num_classes - 1
        ),  # temporary clamp to valid one-hot encoding indices
        num_classes=num_classes,
    ).float()

    # For all indices that fall outside the temporary valid range, zero out their
    # one-hot vectors. This includes 'N's and unknown indices.
    mask = (idx >= 0) & (idx < num_classes)
    one_hot = one_hot * mask.unsqueeze(-1)

    return one_hot


def idx_to_str(
    idx: Integer[Tensor, " l"] | Integer[Tensor, "b l"],
) -> str | List[str]:
    """Convert one-hot encoded indices to a string or list of strings"""
    idx_to_str_mapping = list(BASE_COMPLEMENTS.keys())

    if idx.ndim == 1:
        return "".join([idx_to_str_mapping[i] for i in idx])
    else:
        return [idx_to_str(i) for i in idx]


def onehot_to_idx(
    one_hot: Integer[Tensor, "l c"] | Integer[Tensor, "b l c"],
) -> Integer[Tensor, " l"] | Integer[Tensor, "b l"]:
    # Get both maximum indices and values (for finding out zeros corresponding to 'N')
    max_vals, idx = torch.max(one_hot, dim=-1)
    idx = idx.where(max_vals > 0, BASE_ENCODING["N"])

    return idx


# =====================================================================================


@dataclass(frozen=True)
class Sequence:
    """A dataclass representing a genomic sequence in a one-hot encoding indices

    One-hot encoding indices are defined by `BASE_ENCODING`. This class provides
    helpful utilities for working with sequences, such as reverse complementing,
    batching, and slicing.

    Args:
        tensor: Tensor of shape (b, l) or (l, ) containing the one-hot encoding
            indices of the sequence. If unbatched sequence of shape (l, ), will
            be converted to (1, l) during __post_init__.
    """

    tensor: Integer[Tensor, " l"] | Integer[Tensor, "b l"]

    def __post_init__(self):
        # Unbatched sequence of shape (l, ), will be converted to (1, l)
        if self.tensor.ndim == 1:
            # Need this to override `frozen=True`
            object.__setattr__(self, "tensor", self.tensor.unsqueeze(0))
        elif self.tensor.ndim != 2:  # sanity check
            raise ValueError(
                f"Sequence must have shape [l] or [b, l], got {self.tensor.shape}"
            )

        # int8 is sufficient for one-hot encoding indices defined in `BASE_ENCODING`,
        # so convert to int8 if it is not already.
        if self.tensor.dtype != torch.int8:
            # Need this to override `frozen=True`
            object.__setattr__(self, "tensor", self.tensor.to(torch.int8))

        # Sanity check invalid indices
        if not torch.isin(
            self.tensor,
            torch.tensor(list(BASE_ENCODING.values()), device=self.tensor.device),
        ).all():
            raise ValueError(f"Sequence contains invalid indices: {self.tensor}")

    def __getitem__(self, idx: slice) -> Sequence:
        """Only support slicing along the sequence length (last dimension)"""
        if isinstance(idx, slice):
            return Sequence(self.tensor[:, idx])
        else:
            raise ValueError(f"Invalid slice: {idx}")

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Sequence) and torch.equal(self.tensor, other.tensor)

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the sequence tensor"""
        return self.tensor.shape

    def reverse_complement(self) -> Sequence:
        """Reverse complement the sequence

        Works with both batched sequences [b, l] and unbatched sequences [l].

        Returns:
            OneHotSequence: A new instance with the reverse complemented sequence
        """
        # Flip along the sequence length dimension
        sequence = torch.flip(self.tensor, dims=(-1,))

        # Apply complements via indexing - works for both batched and unbatched
        # This handles batched sequences properly by using advanced indexing
        # that respects the original tensor's shape
        sequence = COMPLEMENT_MAPPING.to(self.tensor.device)[
            sequence.int()
        ]  # indices must be int tensor

        return Sequence(sequence)

    def to(self, device: torch.device) -> Sequence:
        return Sequence(self.tensor.to(device))

    def pin_memory(self):
        return Sequence(self.tensor.pin_memory())

    @classmethod
    def from_string(cls, seq_string: str) -> Sequence:
        """Create a Sequence from a string representation

        Args:
            seq_string: String representation of the sequence (e.g., "ATCG")

        Returns:
            Sequence object containing the converted sequence
        """
        return cls(str_to_idx(seq_string))

    @classmethod
    def batch(cls, seq_list: List[Sequence]) -> Sequence:
        """Batch a list of Sequence objects

        Args:
            seq_list: List of Sequence objects to batch

        Returns:
            Sequence object containing the batched sequence
        """
        if len(seq_list) == 0:
            raise ValueError("Cannot batch an empty list of Sequence")

        sequence_batched = torch.cat([seq.tensor for seq in seq_list], dim=0)

        return cls(sequence_batched)


class RCSequences:
    """A wrapper dataclass holding a forward and reverse complement sequence

    Under the wrapper, the forward and reverse complement sequences are stored
    as a single tensor of shape (2 * b, l) where the first half of the tensor contains
    the forward sequence and the second half contains the reverse complement sequence.
    This helps batching for cases needing predictions from both forward and
    reverse complement sequences.

    To get the forward and reverse complement sequences, use the `forward` and
    `reverse` attributes.

    Args:
        forward: Sequence object containing the forward sequence
        reverse: Sequence object containing the reverse complement sequence. If
            not provided, will be computed from the forward sequence.
    """

    def __init__(self, forward: Sequence, reverse: Sequence | None = None):
        if not isinstance(forward, Sequence):
            raise ValueError("forward must be a Sequence")

        forward_rc = forward.reverse_complement()

        if reverse is None:
            reverse = forward_rc

        # Sanity check
        if forward_rc != reverse:
            raise ValueError("Forward and reverse sequences must be complements")

        self._tensor = torch.cat([forward.tensor, reverse.tensor], dim=0)

    def __getitem__(self, idx: slice) -> RCSequences:
        """Only support slicing along the sequence length (last dimension)"""
        if isinstance(idx, slice):
            forward = self.forward[idx]
            reverse = self.reverse[idx]

            return RCSequences(forward=forward, reverse=reverse)
        else:
            raise ValueError(f"Invalid slice: {idx}")

    def to(self, device: torch.device) -> RCSequences:
        return RCSequences(
            forward=self.forward.to(device),
            reverse=self.reverse.to(device),
        )

    def pin_memory(self):
        return RCSequences(
            forward=self.forward.pin_memory(),
            reverse=self.reverse.pin_memory(),
        )

    @classmethod
    def from_string(cls, seq_string: str) -> RCSequences:
        """Create an RCSequences from a string representation

        Args:
            seq_string: String representation of the forward sequence (e.g., "ATCG")

        Returns:
            RCSequences object containing the forward sequence and its reverse
            complement
        """
        forward = Sequence.from_string(seq_string)
        return cls(forward=forward)

    @classmethod
    def batch(cls, seq_list: List[RCSequences]) -> RCSequences:
        """Batch a list of RCSequences

        First need to batch forward and reverse sequences separately
        to ensure that first chunk is forward and second chunk is reverse
        """
        forward_batched = Sequence.batch([seq.forward for seq in seq_list])
        reverse_batched = Sequence.batch([seq.reverse for seq in seq_list])

        return cls(forward=forward_batched, reverse=reverse_batched)

    @property
    def tensor(self) -> Float[Tensor, "2b l"]:
        assert self._tensor.shape[0] % 2 == 0

        return self._tensor

    @property
    def forward(self) -> Sequence:
        assert self.tensor.shape[0] % 2 == 0

        return Sequence(torch.chunk(self.tensor, 2, dim=0)[0])

    @property
    def reverse(self) -> Sequence:
        assert self.tensor.shape[0] % 2 == 0

        return Sequence(torch.chunk(self.tensor, 2, dim=0)[1])
