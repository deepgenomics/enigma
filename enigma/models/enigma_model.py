"""Enigma model"""

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float, Integer
from torch import Tensor

from enigma.data.seqtrack import TargetTracks
from enigma.data.sequences import Sequence
from enigma.data.tracks_mapping import TracksMapping
from enigma.models.base import BaseModel, BaseModelConfig
from enigma.models.modules.transformer import TransformerModule
from enigma.models.modules.unet import UNetEncoder, UNetModule
from enigma.models.utils import OneHotEmbedding


@dataclass(frozen=True, kw_only=True)
class EnigmaConfig(BaseModelConfig):
    """Default values are chosen for single-bp modeling"""

    model_type: str = "enigma"

    species: Literal["hg38", "mm10", "multi-species"] = "multi-species"

    # Transformer config
    dim: int = 1536
    transformer_num_layers: int = 8
    head_dim: int = 192
    expansion_factor: int = 2
    transformer_norm: Literal["ln", "rms"] = "ln"
    use_rope: bool = True
    rope_base: float = 10000.0
    use_alibi: bool = False
    window_size: Tuple[int, int] = (-1, -1)
    mlp_activation: str = "gelu_tanh"
    mlp_swiglu_match_params: bool = False
    mlp_dropout: float = 0.3
    attention_dropout: float = 0.2
    post_attn_dropout: float = 0.3
    post_mlp_dropout: float = 0.3
    use_qk_norm: bool = True

    # Conv tower config
    conv_num_layers: int = 0  # no conv tower for single-bp model
    conv_dim_in: int = 256
    conv_dim_out: int = 256
    conv_kernel_size: int = 5
    conv_activation: str = "gelu_tanh"
    conv_norm: str = "group_32"
    stem_dropout: float = 0.0

    # UNet config
    unet_dim_out: int = 768
    unet_num_downsampling: int = 7
    encoder_kernel_size: int = 5
    decoder_kernel_size: int = 3
    unet_activation: str = "gelu_tanh"
    unet_norm: str = "group_32"
    unet_dropout: float = 0.0

    # Input embedding config
    embedding_conv_kernel_size: int = 15

    # Head config
    output_hidden_dim: int = 1024
    output_dropout: float = 0.1
    output_dim_human: int = 1190
    output_dim_mouse: int = 258
    output_no_weight_decay: bool = False

    # Configuration for how model is trained
    use_flash_attn: bool = True
    prediction_crop_margin: int | None = None
    prediction_resolution: int = 1


class DownsampleConvTower(nn.Module):
    def __init__(
        self,
        num_layers: int,
        dim_in: int,
        dim_out: int,
        kernel_size: int = 5,
        activation: str = "gelu_tanh",
        norm: str = "group_64",
        dropout: float = 0.0,
        # Generally not necessary to have bias, especially with normalization and
        # residual connections
        bias: bool = False,
    ):
        super().__init__()

        if num_layers == 0:
            self.conv_tower = nn.ModuleList([nn.Identity()])
        else:
            # Channels are changed in geometric progression from dim_input to dim_out,
            # rounded to the nearest multiple of 128. This is to ensure that at each
            # level, embedding dimension can be divided by 128 and be compatible with
            # group norm, which often uses a group dimension of 32, 64, or 128.
            conv_channels = np.geomspace(dim_in, dim_out, num=num_layers + 1)
            conv_channels = (
                (128 * np.round(conv_channels / 128)).astype(np.int32).tolist()
            )

            conv_block_kwargs = dict(
                kernel_size=kernel_size,
                norm=norm,
                activation=activation,
                bias=bias,
                dropout=dropout,
            )

            self.conv_tower = nn.ModuleList(
                [
                    UNetEncoder(d_in, d_out, **conv_block_kwargs)
                    for d_in, d_out in zip(conv_channels[:-1], conv_channels[1:])
                ]
            )

    def forward(self, x: Float[Tensor, "b d l"]) -> Float[Tensor, "b d l"]:
        for layer in self.conv_tower:
            x = layer(x)

        return x


class Enigma(BaseModel):
    def __init__(
        self,
        config: EnigmaConfig,
    ):
        super().__init__(config=config)

        # Initial sanity checks =======================================================
        if config.species == "multi-species":
            # Expect {"hg38": HumanTracksMapping, "mm10": MouseTracksMapping}
            if not isinstance(config.tracks_mapping, dict):
                raise ValueError(
                    f"{self.__class__.__name__} model with multi-species support "
                    "requires a dictionary of species name to tracks mapping."
                )

            if set(config.tracks_mapping.keys()) != {"hg38", "mm10"}:
                raise ValueError(
                    f"Currently, {self.__class__.__name__} only supports multi-species "
                    "models for hg38 and mm10."
                )
        elif config.species == "hg38" or config.species == "mm10":
            if not isinstance(config.tracks_mapping, TracksMapping):
                raise ValueError(
                    f"{self.__class__.__name__} model with hg38 or mm10 species "
                    "support requires a TracksMapping object."
                )
        else:
            raise ValueError(f"Invalid species: {config.species}")

        if config.species == "hg38" and config.output_dim_mouse != 0:
            raise ValueError(
                "hg38 model must have a zero output dimension for mouse tracks."
            )
        elif config.species == "mm10" and config.output_dim_human != 0:
            raise ValueError(
                "mm10 model must have a zero output dimension for human tracks."
            )

        # Model layers ================================================================
        self.embedding = OneHotEmbedding(num_classes=4)

        transformer = TransformerModule(
            num_blocks=config.transformer_num_layers,
            dim=config.dim,
            head_dim=config.head_dim,
            expansion_factor=config.expansion_factor,
            norm=config.transformer_norm,
            use_rope=config.use_rope,
            rope_base=config.rope_base,
            use_alibi=config.use_alibi,
            window_size=config.window_size,
            mlp_activation=config.mlp_activation,
            mlp_dropout=config.mlp_dropout,
            mlp_swiglu_match_params=config.mlp_swiglu_match_params,
            attention_dropout=config.attention_dropout,
            post_attn_dropout=config.post_attn_dropout,
            post_mlp_dropout=config.post_mlp_dropout,
            use_qk_norm=config.use_qk_norm,
            use_flash_attn=config.use_flash_attn,
        )

        self.conv_tower = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=config.conv_dim_in,
                kernel_size=config.embedding_conv_kernel_size,
                padding="same",
                bias=False,
            ),
            DownsampleConvTower(
                num_layers=config.conv_num_layers,
                dim_in=config.conv_dim_in,
                dim_out=config.conv_dim_out,
                kernel_size=config.conv_kernel_size,
                activation=config.conv_activation,
                norm=config.conv_norm,
                dropout=config.stem_dropout,
                bias=False,
            ),
        )

        self.core = UNetModule(
            trunk=transformer,
            num_downsampling=config.unet_num_downsampling,
            dim_input=config.conv_dim_out,
            dim_trunk=config.dim,
            dim_output=config.unet_dim_out,
            encoder_kernel_size=config.encoder_kernel_size,
            decoder_kernel_size=config.decoder_kernel_size,
            activation=config.unet_activation,
            norm=config.unet_norm,
            dropout=config.unet_dropout,
            bias=False,
        )

        # Final head
        self.final_joined_convs = nn.Sequential(
            nn.Linear(config.unet_dim_out, config.output_hidden_dim),
            nn.Dropout(config.output_dropout),
            nn.GELU(approximate="tanh"),
        )

        # Rather than having separate heads for each species, use a unified head and
        # slice appropriately. This is more efficient and easier to implement,
        # especially with DDP (otherwise conditional computation can cause issues and
        # need find_unused_parameters which can slow down computation).
        total_output_dim = config.output_dim_human + config.output_dim_mouse
        self.unified_head = nn.Linear(config.output_hidden_dim, total_output_dim)

        # Set _no_weight_decay on the parameters themselves, not the module
        if config.output_no_weight_decay:
            for param in self.unified_head.parameters():
                param._no_weight_decay = True

        # Store slice indices for each species
        self.human_slice = slice(0, config.output_dim_human)
        self.mouse_slice = slice(config.output_dim_human, total_output_dim)

        self.final_softplus = nn.Softplus()

    def predict_embeddings(
        self,
        x: Integer[Tensor, "b l"],
        cropped_length: int | None = None,
    ) -> Float[Tensor, "b l_cropped d"]:
        """Predict embeddings before the final head layer

        Args:
            x: Input sequence tensor
            cropped_length: Margin to crop out of the model predictions near the
                edges of the sequence. If None, the default from the model
                config is used. To explicitly disable cropping, set to 0.
        """
        if cropped_length is None:
            cropped_length = self.config.prediction_crop_margin

        x = self.embedding(x)

        x = rearrange(x, "b l d -> b d l")

        x = self.conv_tower(x)

        out = self.core(x, cropped_length=cropped_length)

        out = rearrange(out, "b d l -> b l d")

        return out

    def forward(
        self,
        x: Integer[Tensor, "b l"] | Sequence,
        species: Literal["hg38", "mm10"] = "hg38",
        cropped_length: int | None = None,
        example_id: str | None = None,
    ) -> TargetTracks:
        """Forward pass

        Returns a TargetTracks dataclass which makes it easy to keep track of indices
        for each target track.

        Args:
            x: Input sequence tensor of shape (b, l) or Sequence object
            species: Species for the output predictions. Defaults to "hg38" since
                this is expected to be the most common use case
            cropped_length: Margin to crop out of the model predictions near the
                edges of the sequence. If None, the default from the model
                config is used. To explicitly disable cropping, set to 0
            example_id: Optional example ID to attach to the output

        Returns:
            TargetTracks object with predictions.
        """
        if isinstance(x, Sequence):
            x = x.tensor

        # Initial sanity checks =======================================================
        if species not in ["hg38", "mm10"]:
            raise ValueError(f"Invalid species: {species}")

        if self.config.species != "multi-species" and species != self.config.species:
            raise ValueError(
                f"This model is for {self.config.species}, but `species` set to "
                f'"{species}". Please ensure `config.species` is consistent with '
                "`species`."
            )

        # Main forward pass ===========================================================
        x = self.predict_embeddings(x, cropped_length=cropped_length)

        x = self.final_joined_convs(x)

        # Disable autocast for full precision in the final layer
        with torch.amp.autocast(device_type="cuda", enabled=False):
            # Always compute full output, then slice appropriately
            full_output = self.unified_head(x.float())

            if species == "hg38":
                output = full_output[..., self.human_slice]
            else:
                output = full_output[..., self.mouse_slice]

            output_softplus = self.final_softplus(output)

        # Handle tracks_mapping access based on species
        if self.config.species == "multi-species":
            tracks_mapping = self.config.tracks_mapping[species]
        else:
            tracks_mapping = self.config.tracks_mapping

        return TargetTracks(
            tracks=output_softplus,
            tracks_mapping=tracks_mapping,
            example_id=example_id,
        )
