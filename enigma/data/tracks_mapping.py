from pathlib import Path
from typing import Any, ClassVar, List, Set

import numpy as np
import pandas as pd
import torch
from jaxtyping import Bool, Float, Integer
from numpy.typing import NDArray
from torch import Tensor


def track_inverse_transform(
    targets: Float[Tensor, "l d"] | Float[Tensor, "b l d"],
    scale: Float[Tensor, " d"] | Float[NDArray, " d"],
    threshold: Float[Tensor, " d"] | Float[NDArray, " d"],
    apply_squashing: Bool[Tensor, " d"] | Bool[NDArray, " d"],
    post_transform_scale: Float[Tensor, " d"] | Float[NDArray, " d"],
    keep_float32: bool = True,
) -> Float[Tensor, "l d"] | Float[Tensor, "b l d"]:
    """Reverse the `track_transform` applied to the tracks

    Below outlines the order of operations:
    1. Apply inverse post-transform scaling
    2. Apply inverse thresholding with sqrt
    3. Apply inverse squashing
    4. Apply inverse scaling

    Args:
        targets: Target tracks tensor to be untransformed
        scale: Scale factor for each track
        threshold: Threshold for each track
        apply_squashing: Whether to apply the squash transform to the tracks
        post_transform_scale: Scale factor for the post-transform
        keep_float32: Whether to keep the output tensor in float32

    Returns:
        Untransformed tracks
    """
    assert targets.ndim == 2 or targets.ndim == 3, "`targets` must be a 2D or 3D tensor"
    assert (
        scale.ndim == 1 and threshold.ndim == 1 and apply_squashing.ndim == 1
    ), "`scale`, `threshold`, and `apply_squashing` must be 1D tensors"
    assert (
        targets.shape[-1] == scale.shape[0]
    ), "`targets` and `scale` must have the same number of tracks"
    assert (
        targets.shape[-1] == threshold.shape[0]
    ), "`targets` and `threshold` must have the same number of tracks"
    assert (
        targets.shape[-1] == apply_squashing.shape[0]
    ), "`targets` and `apply_squashing` must have the same number of tracks"

    # Convert to tensors if they are Numpy arrays
    if isinstance(scale, np.ndarray):
        scale = torch.from_numpy(scale)
    if isinstance(threshold, np.ndarray):
        threshold = torch.from_numpy(threshold)
    if isinstance(apply_squashing, np.ndarray):
        apply_squashing = torch.from_numpy(apply_squashing)
    if isinstance(post_transform_scale, np.ndarray):
        post_transform_scale = torch.from_numpy(post_transform_scale)

    # Keep operations in float32 for better precision, and move to the same device as
    # the targets
    original_dtype = targets.dtype
    targets = targets.float()

    scale = scale.to(dtype=torch.float32, device=targets.device)
    threshold = threshold.to(dtype=torch.float32, device=targets.device)
    apply_squashing = apply_squashing.bool().to(device=targets.device)
    post_transform_scale = post_transform_scale.to(
        dtype=torch.float32, device=targets.device
    )

    targets = targets / post_transform_scale

    targets = torch.where(
        targets > threshold, (targets - threshold) ** 2 + threshold, targets
    )

    # Apply inverse squashing selectively based on the boolean mask
    targets = torch.where(apply_squashing, targets ** (1.0 / 0.75), targets)

    targets = targets / scale

    if not keep_float32:
        targets = targets.to(original_dtype)

    return targets


class TracksMapping:
    """Mapping of tracks to indices in the merged tracks tensor

    This loads tracks metadata in a Pandas DataFrame and applies appropriate
    transformations to it, including filtering out tracks not in `tracks_to_load`
    and re-mapping indices.

    The underlying dataset can store all track types together or separately.
    `orginal_index` is the index of how the track is stored in the underlying dataset.
    `track_index` is the index within the same track type.

    From all the tracks in the underlying dataset, `loaded_tracks` specifies which
    tracks are actually loaded and stored in the merged tracks tensor.

    The mapping indices are stored in the underlying `metadata_df` DataFrame.
    Commonly used indices are conveniently accessible via methods, but custom
    indices need to be fetched directly from the DataFrame.

    Child classes can add additional transformations to the dataframe by extending
    `load_tracks_metadata` and `process_tracks_metadata` methods.

    Args:
        metadata_file: Path to metadata file. If None, the child class must define
            `metadata_file` as a class variable. If `metadata_file` is specified,
            it will override `self.metadata_file` in the child class.
        tracks_to_load: Tracks to load. If None, load all tracks. Default is
            ["rna", "dnase", "atac"], which are the tracks shown to be essential in
            the FlashRNA work.
    """

    # Columns in metadata file
    ORIGINAL_INDEX_COL: ClassVar[str] = "original_index"
    TRACK_INDEX_COL: ClassVar[str] = "track_index"
    IDENTIFIER_COL: ClassVar[str] = "identifier"
    TRACK_TYPE_COL: ClassVar[str] = "track_type"
    STRAND_COL: ClassVar[str] = "strand"
    STRAND_PAIR_COL: ClassVar[str] = "strand_pair"

    EXPECTED_COLUMNS: ClassVar[List[str]] = [
        ORIGINAL_INDEX_COL,
        IDENTIFIER_COL,
        TRACK_TYPE_COL,
        STRAND_COL,
        STRAND_PAIR_COL,
    ]

    def __init__(
        self,
        metadata_file: str | Path | None = None,
        tracks_to_load: Set[str] | List[str] | None = ["rna", "dnase", "atac"],
    ):
        if metadata_file is not None:
            metadata_file = Path(metadata_file)
        elif not hasattr(self, "DEFAULT_METADATA_FILE"):
            raise ValueError(
                "`DEFAULT_METADATA_FILE` must be specified in the child class or "
                "passed as an argument to the constructor"
            )
        else:
            metadata_file = self.DEFAULT_METADATA_FILE

        if not metadata_file.exists():
            raise FileNotFoundError(f"`metadata_file` not found: {metadata_file}")

        df = self.load_tracks_metadata(metadata_file)
        self._num_tracks_original = len(
            df
        )  # this is useful for keeping track of whether only subset is used
        df = self.process_tracks_metadata(df, tracks_to_load=tracks_to_load)

        self._metadata_df = df

        # Ensure `_loaded_tracks` preserves order of appearance specified in
        # `_metadata_df`
        self._loaded_tracks = df[self.TRACK_TYPE_COL].unique().tolist()

    def indices_by_gtex_rna(self, tissue: str) -> Integer[NDArray, " d"]:
        raise NotImplementedError("This method must be implemented in the child class")

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TracksMapping):
            return False

        # This can be quite restrictive. Can relax this depending on the
        # specific use case in the child class.
        return self.metadata_df.equals(other.metadata_df)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(loaded_tracks: {self.loaded_tracks}, "
            f"num_tracks: {self.num_tracks})"
        )

    @property
    def loaded_tracks(self) -> List[str]:
        """Tracks that are actually loaded and stored in the merged tensor"""
        return self._loaded_tracks.copy()

    @property
    def metadata_df(self) -> pd.DataFrame:
        """Metadata DataFrame"""
        if not hasattr(self, "_metadata_df"):
            raise AttributeError(
                f"{self.__class__.__name__} does not have metadata DataFrame loaded."
            )

        return self._metadata_df

    @property
    def num_tracks(self) -> int:
        """Number of tracks"""
        return len(self.metadata_df)

    @property
    def subset_loaded(self) -> bool:
        """Whether only a subset of tracks is used. This is useful for determining
        whether to subset the tracks when loading from the source dataset
        """
        return self._num_tracks_original != self.num_tracks

    @property
    def original_indices(self) -> Integer[NDArray, " d"]:
        """Original indices of the tracks"""
        return self.metadata_df[self.ORIGINAL_INDEX_COL].values

    @property
    def strand_pairs(self) -> Integer[NDArray, " d"]:
        """Strand pairs"""
        return self.metadata_df[self.STRAND_PAIR_COL].values

    def indices_by_type(self, track_type: str) -> Integer[NDArray, " d"]:
        """Get tracks indices by track type"""
        if track_type not in self.metadata_df[self.TRACK_TYPE_COL].unique():
            raise KeyError(
                f"Invalid track type: '{track_type}'. "
                f"Available types: {self.metadata_df[self.TRACK_TYPE_COL].unique()}"
            )
        return self.metadata_df[
            self.metadata_df[self.TRACK_TYPE_COL] == track_type
        ].index.values

    def load_tracks_metadata(self, metadata_file: str | Path) -> pd.DataFrame:
        """Load tracks metadata from underlying mapping file (e.g. a CSV file)

        This loads the raw metadata file without much processing (done in
        `process_tracks_metadata`).

        Returns:
            DataFrame with tracks metadata
        """
        df = pd.read_csv(metadata_file, sep=",", index_col=None)

        # Sanity check
        if not set(self.EXPECTED_COLUMNS).issubset(df.columns):
            raise ValueError(
                f"Expected columns {self.EXPECTED_COLUMNS} not found in DataFrame"
            )

        return df

    def process_tracks_metadata(
        self, df: pd.DataFrame, tracks_to_load: List[str] | None = None
    ) -> pd.DataFrame:
        """Process tracks metadata to filter out tracks not in `tracks_to_load` and
        update `strand_pair` accordingly.

        Args:
            df: DataFrame with tracks metadata
            tracks_to_load: List of tracks to load. If None, load all tracks

        Returns:
            DataFrame with the new tracks metadata with updated `strand_pair`
        """
        # If `tracks_to_load` is not specified, load all tracks
        if tracks_to_load is None:
            # If not specified, load all tracks
            tracks_to_load = df[self.TRACK_TYPE_COL].unique().tolist()
        else:
            if len(tracks_to_load) != len(set(tracks_to_load)):
                raise ValueError(
                    "`tracks_to_load` contains duplicates. Must contain unique tracks"
                )

            # Validate that all requested tracks exist
            available_tracks = set(df[self.TRACK_TYPE_COL].unique())
            missing_tracks = set(tracks_to_load) - available_tracks
            if missing_tracks:
                raise ValueError(
                    f"Requested tracks not found in DataFrame: {sorted(missing_tracks)}"
                )

        # Collect tracks of the same type together and update `strand_pair` with
        # the updated indices of the strand pair (otherwise, it will still point
        # to the original indices)

        df_track_list = []
        track_idx_offset = 0  # offset where each track type starts in the merged tensor
        for track_type in tracks_to_load:
            df_track = df[df[self.TRACK_TYPE_COL] == track_type].reset_index(drop=True)
            # `track_index` will be the absolute index in the merged tensor
            track_index = df_track.index + track_idx_offset

            # Sanity check before processing `strand_pair`: make sure all are valid
            self._validate_strand_pairs(df_track)

            # Map `strand_pair` to the `track_index` of the pair
            pair_idx_mapping = dict(zip(df_track[self.TRACK_INDEX_COL], track_index))
            df_track[self.STRAND_PAIR_COL] = df_track[self.STRAND_PAIR_COL].map(
                pair_idx_mapping
            )

            # Update `track_index`
            df_track[self.TRACK_INDEX_COL] = df_track.index

            df_track_list.append(df_track)
            track_idx_offset += len(df_track)  # update offset for the next track type

        df = pd.concat(df_track_list).reset_index(drop=True)

        return df

    def _validate_strand_pairs(self, df: pd.DataFrame) -> None:
        """Validate that strand pairing is consistent and correct.

        This validates only within a single track type. This is because depending on
        how tracks are stored in the source data, strand pair might be indices within
        a single track type, not global indices. Then, `df` from multiple track types
        will have clashing strand pairs.

        Validates:
        - All strand_pair values reference existing original_index values
        - Unstranded tracks (".") point to themselves
        - Stranded tracks ("+"/"-") have equal counts and point to each other
        - All pairing relationships are bidirectional

        Args:
            df: DataFrame of a single track type with required columns
        """
        # Initial sanity checks
        if df[self.TRACK_TYPE_COL].nunique() > 1:
            raise ValueError("DataFrame must contain a single track type")

        if set(df[self.STRAND_PAIR_COL]) != set(df[self.TRACK_INDEX_COL]):
            raise ValueError("`strand_pair` and `original_index` do not match")

        self._validate_unstranded_tracks(df[df[self.STRAND_COL] == "."])
        self._validate_stranded_tracks(df[df[self.STRAND_COL] != "."])

    def _validate_unstranded_tracks(self, df: pd.DataFrame) -> None:
        if df.empty:
            return

        if not df[self.STRAND_PAIR_COL].equals(df[self.TRACK_INDEX_COL]):
            raise ValueError(
                "Unstranded tracks must have strand pair pointing to themselves"
            )

    def _validate_stranded_tracks(self, df: pd.DataFrame) -> None:
        df_plus = df[df[self.STRAND_COL] == "+"]
        df_minus = df[df[self.STRAND_COL] == "-"]

        if not len(df_plus) == len(df_minus):
            raise ValueError(
                "Plus and minus strands must have the same number of tracks"
            )

        if df_plus.empty:
            return

        # Check that plus and minus strands point to each other
        # Create dictionaries for O(1) lookups instead of O(n) DataFrame searches
        track_idx_to_strand = dict(zip(df[self.TRACK_INDEX_COL], df[self.STRAND_COL]))
        track_idx_to_strand_pair = dict(
            zip(df[self.TRACK_INDEX_COL], df[self.STRAND_PAIR_COL])
        )

        for _, plus_row in df_plus.iterrows():
            plus_idx = plus_row[self.TRACK_INDEX_COL]
            paired_idx = plus_row[self.STRAND_PAIR_COL]

            # Check if the paired track exists
            if paired_idx not in track_idx_to_strand:
                raise ValueError(
                    f"Plus strand {plus_idx} points to non-existent " f"{paired_idx}"
                )

            # Check if the paired track is a minus strand
            paired_strand = track_idx_to_strand[paired_idx]
            if paired_strand != "-":
                raise ValueError(
                    f"Plus strand {plus_idx} should point to minus strand, "
                    f"but points to strand='{paired_strand}'"
                )

            # Check that the minus strand points back to the plus strand
            minus_strand_pair = track_idx_to_strand_pair[paired_idx]
            if minus_strand_pair != plus_idx:
                raise ValueError(
                    f"Plus strand {plus_idx} and minus strand {paired_idx} do not "
                    "point to each other"
                )


class TrackDatasetMapping(TracksMapping):
    ONTOLOGY_COL = "ontology"
    GTEX_TISSUE_COL = "gtex_tissue"
    SOURCE_COL = "source"
    TRANSFORM_SCALE_COL = "transform_scale"
    TRANSFORM_THRESHOLD_COL = "transform_threshold"
    SCALE_COL = "scale"
    APPLY_SQUASH_COL = "apply_squashing"
    PRE_TRANSFORM_SCALE_COL = "pre_transform_scale"

    EXPECTED_COLUMNS = TracksMapping.EXPECTED_COLUMNS + [
        ONTOLOGY_COL,
        GTEX_TISSUE_COL,
        SOURCE_COL,
        TRANSFORM_SCALE_COL,
        TRANSFORM_THRESHOLD_COL,
        SCALE_COL,
        APPLY_SQUASH_COL,
        PRE_TRANSFORM_SCALE_COL,
    ]

    def __init__(
        self,
        metadata_file: str | Path | None = None,
        tracks_to_load: Set[str] | List[str] | None = ["rna", "dnase", "atac"],
        gtex_only: bool = False,
    ):
        """Handle GTEX-only tracks

        If `gtex_only` is True, `rna` must be in `tracks_to_load`. Only GTEX
        tracks will be loaded for RNA-seq tracks.
        """
        if gtex_only and "rna" not in tracks_to_load:
            raise ValueError("`rna` must be in `tracks_to_load` if `gtex_only` is True")
        self.gtex_only = gtex_only

        super().__init__(metadata_file=metadata_file, tracks_to_load=tracks_to_load)

    def process_tracks_metadata(
        self, df: pd.DataFrame, tracks_to_load: List[str] | None = None
    ) -> pd.DataFrame:
        if self.gtex_only:
            df_non_gtex_rna = df[
                (df[self.TRACK_TYPE_COL] == "rna") & (df[self.SOURCE_COL] != "gtex")
            ]
            df = df.drop(df_non_gtex_rna.index)

        return super().process_tracks_metadata(df=df, tracks_to_load=tracks_to_load)

    def load_tracks_metadata(self, metadata_file: str | Path) -> pd.DataFrame:
        df = super().load_tracks_metadata(metadata_file=metadata_file)

        # Drop unnecessary columns
        df = df.drop(columns=["filename"])

        return df

    def reverse_transform(
        self,
        tracks: Float[Tensor, "b l d"],
        log2p1: bool = False,
    ) -> Float[Tensor, "b l d"]:
        """Reverse the transform applied to the tracks

        Args:
            tracks: Tracks tensor to be untransformed
            log2p1: Whether to apply log2p1 transform to the tracks

        Returns:
            Untransformed tracks
        """
        tracks_untransformed = track_inverse_transform(
            tracks,
            scale=self.transform_scale,
            threshold=self.transform_threshold,
            apply_squashing=self.apply_squash,
            post_transform_scale=self.scale,
        )

        if not log2p1:
            return tracks_untransformed

        should_log2p1 = self.should_log2p1.to(device=tracks_untransformed.device)

        tracks_untransformed = torch.where(
            should_log2p1, torch.log2(tracks_untransformed + 1), tracks_untransformed
        )

        return tracks_untransformed

    def indices_by_gtex_rna(self, tissue: str) -> Integer[NDArray, " n"]:
        if tissue not in self.gtex_tissues:
            raise ValueError(
                f"Invalid tissue: {tissue}. Available tissues: {self.gtex_tissues}"
            )

        tissue_tracks = self.metadata_df[
            (self.metadata_df[self.GTEX_TISSUE_COL] == tissue)
            & (self.metadata_df[self.TRACK_TYPE_COL] == "rna")
        ]

        return tissue_tracks.index.values

    def indices_by_gtex_junction(self, tissue: str) -> Integer[NDArray, " n"]:
        if tissue not in self.gtex_tissues:
            raise ValueError(
                f"Invalid tissue: {tissue}. Available tissues: {self.gtex_tissues}"
            )

        tissue_tracks = self.metadata_df[
            (self.metadata_df[self.GTEX_TISSUE_COL] == tissue)
            & (self.metadata_df[self.TRACK_TYPE_COL] == "junction")
        ]

        return tissue_tracks.index.values

    @property
    def gtex_tissues(self) -> List[str]:
        """GTEX tissues in the tracks mapping"""
        if not hasattr(self, "_gtex_tissues"):
            gtex_tissues = self.metadata_df[self.GTEX_TISSUE_COL].dropna().unique()

            if len(gtex_tissues) == 0:
                raise ValueError("No GTEX tissues found in the tracks mapping")

            self._gtex_tissues = gtex_tissues.tolist()

        return self._gtex_tissues

    @property
    def gtex_indices(self) -> Integer[NDArray, " d"]:
        if not hasattr(self, "_gtex_indices"):
            self._gtex_indices = self.metadata_df[
                (self.metadata_df[self.SOURCE_COL] == "gtex")
                & (self.metadata_df[self.TRACK_TYPE_COL] == "rna")
            ].index.values
        return self._gtex_indices

    @property
    def encode_rna_indices(self) -> Integer[NDArray, " d"]:
        if not hasattr(self, "_encode_rna_indices"):
            self._encode_rna_indices = self.metadata_df[
                (self.metadata_df[self.TRACK_TYPE_COL] == "rna")
                & (self.metadata_df[self.SOURCE_COL] == "encode")
            ].index.values
        return self._encode_rna_indices

    @property
    def transform_scale(self) -> Float[NDArray, " d"]:
        """Scale of the transform"""
        return self.metadata_df[self.TRANSFORM_SCALE_COL].values.astype(np.float32)

    @property
    def transform_threshold(self) -> Float[NDArray, " d"]:
        """Threshold of the transform"""
        return self.metadata_df[self.TRANSFORM_THRESHOLD_COL].values.astype(np.float32)

    @property
    def apply_squash(self) -> Bool[NDArray, " d"]:
        """Whether to apply the squash transform to the tracks"""
        return self.metadata_df[self.APPLY_SQUASH_COL].values

    @property
    def scale(self) -> Float[NDArray, " d"]:
        """Scale factor for each track"""
        return self.metadata_df[self.SCALE_COL].values.astype(np.float32)

    @property
    def pre_transform_scale(self) -> Float[NDArray, " d"]:
        """Scale of the pre-transform"""
        return self.metadata_df[self.PRE_TRANSFORM_SCALE_COL].values.astype(np.float32)

    @property
    def should_log2p1(self) -> Bool[Tensor, " d"]:
        """Whether to apply log2p1 transform to the tracks"""
        if not hasattr(self, "_should_log2p1"):
            _should_log2p1 = self.metadata_df[self.TRACK_TYPE_COL].isin(["rna"]).values
            self._should_log2p1 = torch.from_numpy(_should_log2p1)
        return self._should_log2p1
