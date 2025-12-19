from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal

import torch
import torch.nn as nn
import wandb
from jaxtyping import Float, Integer
from lightning.pytorch.core.mixins import HyperparametersMixin
from lightning.pytorch.utilities import rank_zero_info
from torch import Tensor

from enigma.config import MODELS_DIR
from enigma.data.seqtrack import TargetTracks
from enigma.data.sequences import RCSequences, Sequence
from enigma.data.tracks_mapping import TracksMapping
from enigma.models.utils import import_class_from_path


@dataclass(frozen=True, kw_only=True)
class BaseModelConfig(ABC):
    """Base model config class

    Every model config should have
    - `model_type` field to identify the model type.
    - `multi_species` field to specify whether the model is multi-species. If True,
        `tracks_mapping` should be a dictionary of species name to tracks mapping.
    - `tracks_mapping` field to keep track of which part of model predictions
        corresponds to which track.
    - `prediction_resolution` field to specify the resolution of the model predictions.
    - `prediction_crop_margin` field to specify how much of the model predictions
        to crop out near the edges of the sequence. Ideally, this should be consistent
        with the cropping used during training.
    """

    model_type: str

    species: Literal["hg38", "mm10", "multi-species"]

    tracks_mapping: (
        TracksMapping | Dict[str, TracksMapping]
    )  # dict for multi-species models
    prediction_resolution: int
    prediction_crop_margin: int | None = None


class BaseModel(ABC, HyperparametersMixin, nn.Module):
    """Base model class providing useful helper methods

    Each model should define
    - `predict_embeddings` method to compute prediction embedding (before the final
      output layers).
    - `forward` method to perform the forward pass.
    - `predict` method to perform the forward pass for inference (can be slightly
      different from `forward` for some models).
    - `predict_with_rc` method to perform predictions on both forward and reverse
      strands.
    """

    def __init__(self, config: BaseModelConfig):
        super().__init__()

        self.save_hyperparameters()

        self.config = self.hparams.config

    @abstractmethod
    def predict_embeddings(
        self,
        x: Integer[Tensor, "b l"],
        cropped_length: int | None = None,
    ) -> Float[Tensor, "b l_cropped d"]:
        """Predict embeddings before the final head layer

        Args:
            x: Input sequence tensor of shape (b, l)
            cropped_length: Margin to crop out of the model predictions near the
                edges of the sequence. If None, the default from the model
                config is used.
        """
        pass

    @abstractmethod
    def forward(
        self,
        x: Integer[Tensor, "b l"],
        cropped_length: int | None = None,
        example_id: str | None = None,
        **kwargs,
    ) -> TargetTracks:
        """Models should return a TrackOutput dataclass which makes it easy
        to keep track of indices for each target track

        Args:
            x: Input sequence tensor of shape (b, l)
            cropped_length: Margin to crop out of the model predictions near the
                edges of the sequence. If None, the default from the model
                config is used.
            example_id: Optional example ID to attach to the output.
        """
        pass

    def predict(
        self,
        x: Integer[Tensor, "b l"] | Sequence | RCSequences,
        cropped_length: int | None = None,
        example_id: str | None = None,
        pred_reverse_complement: bool = False,
        reverse_transform: bool = False,
        **kwargs,
    ) -> TargetTracks:
        """Forward pass for inference

        With `pred_reverse_complement` set to True, prediction will be merged from
        both forward and reverse complement of the input sequence.

        Args:
            x: Input sequence tensor of shape (b, l) or Sequence object or RCSequences
                object (only when `pred_reverse_complement` is True)
            cropped_length: Margin to crop out of the model predictions near the
                edges of the sequence. If None, the default from the model
                config is used.
            example_id: Optional example ID to attach to the output.
            pred_reverse_complement: Whether to predict the reverse complement of the
                input sequence.
        """
        if isinstance(x, RCSequences) and not pred_reverse_complement:
            raise ValueError(
                "RCSequences object provided but `pred_reverse_complement` is False"
            )

        if pred_reverse_complement:
            if isinstance(x, Sequence):
                x = RCSequences(x)
            elif isinstance(x, Tensor):
                x = RCSequences(Sequence(x))
            elif not isinstance(x, RCSequences):
                raise ValueError(
                    f"Invalid input type: {type(x)}, "
                    "must be Sequence, Tensor, or RCSequences"
                )

            return self._predict_with_rc(
                x,
                cropped_length=cropped_length,
                example_id=example_id,
                reverse_transform=reverse_transform,
                **kwargs,
            )
        else:
            preds = self.forward(
                x,
                cropped_length=cropped_length,
                example_id=example_id,
                **kwargs,
            )

            if not reverse_transform:
                return preds

            preds_tracks = preds.tracks_mapping.reverse_transform(preds.tracks.float())

            return preds.update_tracks(tracks=preds_tracks)

    def _predict_with_rc(
        self,
        x: RCSequences,
        cropped_length: int | None = None,
        example_id: str | None = None,
        reverse_transform: bool = False,
        **kwargs,
    ) -> TargetTracks:
        """Predict with both forward and reverse complement of the input sequence and
        merge the predictions

        Args:
            x: RCSequences object
            cropped_length: Margin to crop out of the model predictions near the
                edges of the sequence. If None, the default from the model
                config is used.
            example_id: Optional example ID to attach to the output.
        """
        if not isinstance(x, RCSequences):
            raise ValueError("x must be a RCSequences object")

        preds = self.forward(
            x.tensor,
            cropped_length=cropped_length,
            example_id=example_id,
            **kwargs,
        )

        preds_tracks = preds.tracks

        if reverse_transform:
            preds_tracks = preds.tracks_mapping.reverse_transform(preds_tracks.float())

        preds_fw, preds_rc = torch.chunk(preds_tracks, 2, dim=0)

        # Flip 'preds_rc' and using 'strand_pairs' to map stranded
        # tracks to its complement to align with 'preds_fw'.
        # 'strand_pairs' is a list of track indices that swaps
        # forward and reverse complement tracks while keeping unstranded tracks
        # in the same position
        preds_rc = preds_rc.flip(-2)[:, :, preds.tracks_mapping.strand_pairs]

        preds_merged = (preds_fw + preds_rc) * 0.5

        return preds.update_tracks(tracks=preds_merged)

    @classmethod
    def center_crop(
        cls, x: Float[Tensor, "b l d"], cropped_length: int | None
    ) -> Float[Tensor, "b l_cropped d"]:
        """Center crop a tensor by a given margin 'cropped_length',
        returning a tensor of shape '(b, l - 2 * crop_margin, d)'

        This is used to crop out model predictions near the edges of the
        sequence, where the model is expected to be worse due to limited
        sequence context.
        """
        if cropped_length is None or cropped_length == 0:
            return x

        assert len(x.shape) == 3, f"Expected (b l d), got {x.shape}"

        L = x.shape[1]
        assert (
            cropped_length > 0
        ), f"Cropped length must be positive, got {cropped_length}"
        assert (
            L - 2 * cropped_length > 0
        ), f"Cropped length must be positive, got {L - 2 * cropped_length}"

        return x[:, cropped_length:-cropped_length]

    @property
    def tracks_mapping(self) -> Dict[str, TracksMapping]:
        """Dictionary of species name to tracks mapping

        If `config.species` is "multi-species", the underlying `config.tracks_mapping`
        is a dictionary of species name to tracks mapping, but for single-species
        models, it is a TracksMapping object.

        This property always returns a dictionary of species name to tracks mapping to
        provide a consistent interface regardless of single- or multi-species training.
        """
        if self.config.species == "multi-species":
            # Should have already been validated in __init__
            assert isinstance(self.config.tracks_mapping, dict)

            return self.config.tracks_mapping

        return {self.config.species: self.config.tracks_mapping}

    @property
    def output_dim(self) -> Dict[str, int]:
        """Dictionary of species name to output dimension"""

        return {
            "hg38": self.config.output_dim_human,
            "mm10": self.config.output_dim_mouse,
        }

    @classmethod
    def from_ckpt(
        cls,
        ckpt_path: str | Path | None = None,
        wandb_artifact: str | None = None,
        wandb_filename: str = "model.ckpt",
        model_class_key: str = "model_class",
        model_config_key: str = "model_config",
        weights_prefix: str | None = "model.",
        load_weights: bool = True,
        device: str | torch.device | None = None,
    ) -> BaseModel:
        """Load a model from a checkpoint file or a wandb artifact

        This assumes the model class and model config were saved under
        `hyper_parameters`. `model_class_key` is the key under which the model
        class is found in the `hyper_parameters`, and `model_config_key` is
        the key under which the model config is found.

        A model is initialized using the model class with the model config.
        If `load_weights` is False, only the model is initialized according to
        the model config without loading saved weights.

        `weights_prefix` is the prefix added to the keys of the model weights in
        the checkpoint (e.g. when the model was saved as self.model in the
        LightningModule). This is removed before loading the model weights.

        Args:
            ckpt_path: Path to the checkpoint file
            wandb_artifact: Wandb artifact name
            wandb_filename: Wandb filename in the artifact
            model_class_key: Key for the model class in the checkpoint
            model_config_key: Key for the model config in the checkpoint
            weights_prefix: Prefix for the model weights in the checkpoint
            load_weights: Whether to load the saved model weights
        """
        if ckpt_path is None and wandb_artifact is None:
            raise ValueError("Either ckpt_path or wandb_artifact must be provided")
        elif ckpt_path is not None and wandb_artifact is not None:
            raise ValueError("Only one of ckpt_path or wandb_artifact must be provided")

        if wandb_artifact is not None:
            ckpt_path = MODELS_DIR / wandb_artifact / wandb_filename
            rank_zero_info(f"Loading from wandb artifact: {wandb_artifact}")
            if not ckpt_path.exists():
                rank_zero_info(
                    f"Local checkpoint not found at {ckpt_path}. "
                    f"Downloading from wandb artifact: {wandb_artifact}"
                )
                artifact = wandb.Api().artifact(wandb_artifact)
                artifact.download(root=ckpt_path.parent)

            # If there is a wandb run, track this model artifact is consumed during
            # this run
            if wandb.run is not None:
                wandb.run.use_artifact(wandb_artifact)
        else:
            ckpt_path = Path(ckpt_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

        rank_zero_info(
            f"Loading model from {ckpt_path.name} (loading weights: {load_weights})"
        )

        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        model_cls = ckpt["hyper_parameters"][model_class_key]
        if isinstance(model_cls, str):
            model_cls = import_class_from_path(model_cls)
        elif not issubclass(model_cls, BaseModel):
            raise ValueError(f"Model is not a subclass of BaseModel: {type(model_cls)}")

        model_config = ckpt["hyper_parameters"][model_config_key]

        if not isinstance(model_cls, type) or not issubclass(model_cls, BaseModel):
            raise ValueError(f"Model is not a subclass of BaseModel: {type(model_cls)}")

        if not isinstance(model_config, BaseModelConfig):
            raise ValueError(
                "Model config is not a subclass of "
                f"BaseModelConfig: {type(model_config)}"
            )

        model = model_cls(model_config)

        if load_weights:
            # If LightningModule is used, there can be additional weights_prefix added
            # depending on how the model is saved in the LightningModule.
            # e.g. if self.model = Model(...) inside the LightningModule,
            # weights_prefix will be "model."
            if weights_prefix:
                weights = {
                    k.removeprefix(weights_prefix): v
                    for k, v in ckpt["state_dict"].items()
                    if k.startswith(
                        weights_prefix
                    )  # only load weights starting with weights_prefix
                }
            else:
                weights = ckpt["state_dict"]

            model.load_state_dict(weights)

        return model
