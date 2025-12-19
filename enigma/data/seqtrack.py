from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Tuple

import torch
from jaxtyping import Float
from torch import Tensor

from enigma.data.sequences import RCSequences, Sequence
from enigma.data.tracks_mapping import TracksMapping


@dataclass(frozen=True, kw_only=True)
class TargetTracks:
    """Holds track values and mapping from track name to slice of the track

    Depending on the underlying dataset, TargetTracks can have `tracks` values
    in different spaces (e.g. raw counts, log2p1, etc.). `get_raw_counts` and
    `get_log2p1` methods are used to convert `tracks` to the appropriate space.

    Unbatched `tracks` of shape (l, c) will be converted to (1, l, c) during
    `__post_init__`.

    `example_id` is used for tracking which example the tracks belong to.
    """

    tracks: Float[Tensor, "b l c"]
    tracks_mapping: TracksMapping
    example_id: str | List[str] | None = None

    def __post_init__(self):
        if self.tracks.ndim == 2:
            object.__setattr__(self, "tracks", self.tracks.unsqueeze(0))
        elif self.tracks.ndim != 3:
            raise ValueError(
                f"Tracks must have shape [l, c] or [b, l, c], got {self.tracks.shape}"
            )

    def __getitem__(self, idx: slice) -> TargetTracks:
        """Get a slice along the length dimension of the tracks"""
        if not isinstance(idx, slice):
            raise ValueError(f"Only slice indexing is supported, got {type(idx)}")

        return self.update_tracks(tracks=self.tracks[..., idx, :])

    def get_tracks(self, track_name: str | None = None) -> Float[Tensor, "b l c"]:
        """Get a track by name. If `track_name` is None, all tracks are returned.

        Args:
            track_name: Name of the track to retrieve. If None, returns all tracks.

        Returns:
            Tensor of tracks with shape (b, l, c) where c depends on track_name
        """
        if track_name is None:
            return self.tracks
        else:
            return self.tracks[..., :, self.tracks_mapping.indices_by_type(track_name)]

    def reverse(self) -> TargetTracks:
        """Reverse the tracks along the length dimension and swaps forward and
        reverse tracks, while keeping unstranded tracks the same
        """
        # Get the complement tracks mapping that swaps forward and reverse
        # complement tracks while keeping unstranded tracks in the same position

        # Reverse the tracks along the length dimension
        tracks_rev = self.tracks.flip(-2)
        # Swap forward and reverse complement tracks
        tracks_rev = tracks_rev[:, :, self.tracks_mapping.strand_pairs]

        return self.update_tracks(tracks=tracks_rev)

    def update_tracks(self, tracks: Float[Tensor, "b l c"]) -> TargetTracks:
        """Create a new TargetTracks object with updated tracks tensor

        Args:
            tracks: New tracks tensor

        Returns:
            New TargetTracks instance with updated tracks
        """
        return replace(self, tracks=tracks)

    def to(self, device: torch.device) -> "TargetTracks":
        """Move tracks to a device

        Args:
            device: Target device

        Returns:
            New TargetTracks instance with tracks on the specified device
        """
        return self.update_tracks(tracks=self.tracks.to(device))

    def pin_memory(self):
        return self.update_tracks(tracks=self.tracks.pin_memory())

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Shape of the tracks tensor (b, l, c)"""
        return self.tracks.shape

    @property
    def track_names(self) -> Tuple[str, ...]:
        """Names of the loaded tracks"""
        return tuple(self.tracks_mapping.loaded_tracks)

    @classmethod
    def batch(cls, track_list: List[TargetTracks]) -> TargetTracks:
        """Batch a list of TargetTracks into a single TargetTracks object

        Args:
            track_list: List of TargetTracks to batch

        Returns:
            Batched TargetTracks object

        Raises:
            ValueError: If tracks are not compatible for batching
        """
        if len(track_list) == 0:
            raise ValueError("Cannot batch an empty list of TargetTracks.")

        # Extract shared attributes (excluding tracks and example_id)
        # Sanity check that all items have the same class type and attributes
        # except for `tracks` and `example_id`
        if any(not isinstance(track, cls) for track in track_list):
            raise ValueError("All items must be of the same class type")

        ref_track = track_list[0]
        ref_attributes = {
            key: getattr(ref_track, key)
            for key in ref_track.__dataclass_fields__
            if key not in ["tracks", "example_id"]
        }
        for track in track_list[1:]:
            attributes = {
                key: getattr(track, key)
                for key in track.__dataclass_fields__
                if key not in ["tracks", "example_id"]
            }
            if attributes != ref_attributes:
                raise ValueError("All tracks must have the same attributes")

        # Batch tracks and example_ids
        tracks_batched = torch.cat([track.tracks for track in track_list], dim=0)
        example_ids = [track.example_id for track in track_list]

        return cls(
            tracks=tracks_batched,
            example_id=example_ids,
            **ref_attributes,
        )


@dataclass(frozen=True)
class SequenceTracks:
    """Holds sequence and target tracks

    `sequence` can be Sequence or RCSequences (useful for batching forward and
    reverse complement sequences together)
    """

    sequence: Sequence | RCSequences
    tracks: TargetTracks

    def __getitem__(self, idx: slice) -> SequenceTracks:
        """Get a slice along the length dimension of the sequence and tracks

        Args:
            idx: Slice to apply to both sequence and tracks

        Returns:
            New SequenceTracks with sliced sequence and tracks
        """
        return SequenceTracks(
            sequence=self.sequence[idx],
            tracks=self.tracks[idx],
        )

    def reverse(self) -> SequenceTracks:
        """Reverse the sequence and tracks along the length dimension and swaps forward and
        reverse tracks, while keeping unstranded tracks the same
        """
        return SequenceTracks(
            sequence=self.sequence.reverse_complement(),
            tracks=self.tracks.reverse(),
        )

    def to(self, device: torch.device) -> SequenceTracks:
        """Move sequence and tracks to a device

        Args:
            device: Target device

        Returns:
            New SequenceTracks with sequence and tracks on the specified device
        """
        return SequenceTracks(
            sequence=self.sequence.to(device),
            tracks=self.tracks.to(device),
        )

    def pin_memory(self):
        return SequenceTracks(
            sequence=self.sequence.pin_memory(),
            tracks=self.tracks.pin_memory(),
        )

    @property
    def seq_len(self) -> int:
        """Length of the sequence"""
        return self.sequence.tensor.shape[-1]

    @property
    def track_len(self) -> int:
        """Length of the tracks"""
        return self.tracks.tracks.shape[-2]

    @classmethod
    def batch(cls, seqtrack_list: List[SequenceTracks]) -> SequenceTracks:
        """Batch a list of SequenceTracks into a single SequenceTracks object

        Args:
            seqtrack_list: List of SequenceTracks to batch

        Returns:
            Batched SequenceTracks object
        """
        if len(seqtrack_list) == 0:
            raise ValueError("Cannot batch an empty list of SequenceTracks.")

        # Separate sequences and tracks for batching
        sequence_list = [seqtrack.sequence for seqtrack in seqtrack_list]
        tracks_list = [seqtrack.tracks for seqtrack in seqtrack_list]

        # Batch using the respective class methods
        sequence_type = type(sequence_list[0])
        sequence_batched = sequence_type.batch(sequence_list)
        tracks_batched = type(tracks_list[0]).batch(tracks_list)

        return cls(sequence=sequence_batched, tracks=tracks_batched)
