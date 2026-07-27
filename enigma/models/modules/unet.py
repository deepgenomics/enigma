from typing import List, Tuple

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from enigma.models.layers.conv import ConvBlock


def _crop_tensor(
    x: Float[Tensor, "b d l"], l_crop: int, r_crop: int
) -> Float[Tensor, "b d l_cropped"]:
    """Safely crop a tensor from both ends.

    Args:
        x: Input tensor of shape (batch, channels, length)
        l_crop: Number of tokens to crop from the left
        r_crop: Number of tokens to crop from the right

    Returns:
        Cropped tensor of shape (batch, channels, length - l_crop - r_crop)
    """
    input_length = x.shape[-1]
    # empty output is valid
    if l_crop + r_crop > input_length:
        raise ValueError(
            "Invalid crop sizes: l_crop + r_crop must be smaller or equal to input "
            f"length, got l_crop={l_crop}, r_crop={r_crop}, "
            f"input_length={input_length}"
        )

    end = input_length - r_crop
    return x[:, :, l_crop:end]


class UNetEncoder(nn.Module):
    """UNet encoder which applies a downsampling by a factor of 2 and a ConvBlock"""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        kernel_size: int,
        activation: str,
        norm: str,
        dropout: float = 0.0,
        separable: bool = False,
        bias: bool = False,
    ):
        super().__init__()

        # Max pooling is used for sharper features over avg pooling that can blur
        # features
        # NOTE: For very large tensors, the following error might occur:
        #   Expected output.numel() <= std::numeric_limits<int32_t>::max() to be true,
        #   but got false.  (Could this error message be improved?  If so, please
        #   report an enhancement request to PyTorch.)
        # Unfortunately, there is no easy fix for this without compromising the
        # performance.
        self.downsampling = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv = ConvBlock(
            dim_in=dim_in,
            dim_out=dim_out,
            kernel_size=kernel_size,
            norm=norm,
            activation=activation,
            dropout=dropout,
            separable=separable,
            bias=bias,
        )

    def forward(
        self, x: Float[Tensor, "b d_in l"]
    ) -> Float[Tensor, "b d_out l_downsampled"]:
        x = self.downsampling(x)
        x = self.conv(x)

        return x


class UNetEncoderBlocks(nn.Module):
    """U-Net encoder blocks

    Creates (num_downsampling - 1) encoder blocks that each downsample by 2x,
    followed by a final 2x downsampling for a total downsampling factor of
    2^num_downsampling.

    Channel dimensions follow a geometric progression from dim_in to dim_out,
    rounded to multiples of 128 for compatibility with group normalization.

    Args:
        num_downsampling: Number of downsampling
        dim_in: Input channel dimension
        dim_out: Output channel dimension
        kernel_size: Convolution kernel size for encoders
        activation: Activation function name
        norm: Normalization type (e.g., 'group_32')
        dropout: Dropout rate
        separable: Whether to use depthwise separable convolutions
        bias: Whether to use bias in convolutions
    """

    def __init__(
        self,
        num_downsampling: int,
        dim_in: int,
        dim_out: int,
        kernel_size: int,
        activation: str,
        norm: str,
        dropout: float = 0.0,
        separable: bool = False,
        bias: bool = False,
    ):
        super().__init__()

        self.num_downsampling = num_downsampling

        # UNet encoder channels are geometric progression rounded to the nearest
        # multiple of 128. This is to ensure that at each level, embedding dimension
        # can be divided by 128 and be compatible with group norm, which often uses a
        # group dimension of 32, 64, or 128.
        channels = np.geomspace(dim_in, dim_out, num=num_downsampling)
        channels = (128 * np.round(channels / 128)).astype(np.int32).tolist()
        self.channels = channels

        # Create encoder blocks (num_downsampling - 1 blocks)
        self.encoders = nn.ModuleList(
            [
                UNetEncoder(
                    dim_in=ch_in,
                    dim_out=ch_out,
                    kernel_size=kernel_size,
                    norm=norm,
                    activation=activation,
                    dropout=dropout,
                    separable=separable,
                    bias=bias,
                )
                for ch_in, ch_out in zip(channels[:-1], channels[1:])
            ]
        )

        # Final downsampling after the last encoder
        self.downsampling = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(
        self, x: Float[Tensor, "b d_in l"]
    ) -> Tuple[Float[Tensor, "b d_out l_downsampled"], List[Float[Tensor, "b d l"]]]:
        """Forward pass through the UNet encoder blocks

        Args:
            x: Input tensor

        Returns:
            x: Downsampled output
            skip_connections: List of skip connections (includes input + encoder
                outputs)
        """
        skip_connections = [x]

        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)

        x = self.downsampling(x)

        return x, skip_connections


class UNetDecoder(nn.Module):
    """UNet decoder which applies the following operations:
        1. Upsamples the input by a factor of 2
        2. Applies a pointwise ConvBlock. If dim_in != dim_out, this is where the
        dimension change occurs.
        3. Adds a skip connection after applying a linear projection to match the
        dimension
        4. Applies the final separable ConvBlock

    This decoder is unconventional in that a skip connection is added, rather than
    concatentation.

    Also, a pointwise convolution is applied before upsampling and a separable
    convolution is applied after the skip connection. This results in depthwise
    separable convolution when decoder blocks are stacked.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dim_skip: int,
        kernel_size: int,
        activation: str,
        norm: str,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        self.conv_pointwise = ConvBlock(
            dim_in=dim_in,
            dim_out=dim_out,
            kernel_size=1,
            norm=norm,
            activation=activation,
            dropout=dropout,
            bias=bias,
        )

        self.upsample = nn.Upsample(scale_factor=2)

        self.conv_skip = ConvBlock(
            dim_in=dim_skip,
            dim_out=dim_out,
            kernel_size=1,
            norm=norm,
            activation=activation,
            dropout=dropout,
            bias=bias,
        )

        self.conv_separable = ConvBlock(
            dim_in=dim_out,
            dim_out=dim_out,
            kernel_size=kernel_size,
            norm=norm,
            activation=activation,
            dropout=dropout,
            separable=True,
            bias=bias,
        )

    def forward(
        self, x: Float[Tensor, "b d l"], x_skip: Float[Tensor, "b d l"]
    ) -> Float[Tensor, "b d l"]:
        x = self.upsample(x)
        x = self.conv_pointwise(x)

        x += self.conv_skip(x_skip)

        x = self.conv_separable(x)

        return x


class UNetDecoderBlocks(nn.Module):
    """U-Net decoder blocks

    Creates num_upsampling decoder blocks that each upsample by 2x. This
    implementation only supports efficient cropping by removing edge tokens after the
    trunk, reducing computation in the decoder path.

    Channel dimensions follow a geometric progression from dim_in to dim_out,
    rounded to multiples of 128 for compatibility with group normalization.

    Args:
        num_upsampling: Number of upsampling
        skip_channels: List of skip channel dimensions (obtained from encoder blocks)
        dim_in: Input channel dimension
        dim_out: Output channel dimension
        kernel_size: Convolution kernel size for decoders
        activation: Activation function name
        norm: Normalization type (e.g., 'group_32')
        dropout: Dropout rate
        bias: Whether to use bias in convolutions
    """

    def __init__(
        self,
        num_upsampling: int,
        skip_channels: List[int],
        dim_in: int,
        dim_out: int,
        kernel_size: int,
        activation: str,
        norm: str,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        self.num_upsampling = num_upsampling

        # UNet decoder channels are geometric progression rounded to the nearest
        # multiple of 128. This is to ensure that at each level, embedding dimension
        # can be divided by 128 and be compatible with group norm, which often uses a
        # group dimension of 32, 64, or 128.
        channels = np.geomspace(dim_in, dim_out, num=num_upsampling + 1)
        channels = (128 * np.round(channels / 128)).astype(np.int32).tolist()
        self.channels = channels

        # Reverse skip channels to match decoder order (deepest to shallowest)
        skip_channels_reversed = list(reversed(skip_channels))

        # Create decoder blocks (num_upsampling blocks)
        self.decoders = nn.ModuleList(
            [
                UNetDecoder(
                    dim_in=ch_in,
                    dim_out=ch_out,
                    dim_skip=ch_skip,
                    kernel_size=kernel_size,
                    norm=norm,
                    activation=activation,
                    dropout=dropout,
                    bias=bias,
                )
                for ch_in, ch_out, ch_skip in zip(
                    channels[:-1], channels[1:], skip_channels_reversed
                )
            ]
        )

    def forward(
        self,
        x: Float[Tensor, "b d_in l"],
        skip_connections: List[Float[Tensor, "b d l"]],
        l_tokens_to_crop: int,
        r_tokens_to_crop: int,
    ) -> Float[Tensor, "b d_out l_cropped"]:
        """Forward pass with efficient early cropping.

        Crops tokens from the trunk output and progressively crops skip connections
        to match the upsampled resolution at each decoder stage.

        Args:
            x: Input tensor from trunk
            skip_connections: Skip connections from encoder blocks (deepest first)
            l_tokens_to_crop: Number of tokens to crop from left edge
            r_tokens_to_crop: Number of tokens to crop from right edge

        Returns:
            Upsampled and cropped output tensor
        """
        # Initial token cropping after trunk
        x = _crop_tensor(x, l_tokens_to_crop, r_tokens_to_crop)

        # Progressive upsampling with skip connections
        for i, decoder in enumerate(self.decoders):
            skip = skip_connections.pop()

            # Crop skip connections to match upsampled resolution
            # At each stage, resolution doubles: 2^(i+1)
            l_skip_crop = l_tokens_to_crop * 2 ** (i + 1)
            r_skip_crop = r_tokens_to_crop * 2 ** (i + 1)
            skip = _crop_tensor(skip, l_skip_crop, r_skip_crop)

            x = decoder(x, skip)

        assert not skip_connections, "All skip connections should be consumed"

        return x


class UNetModule(nn.Module):
    """UNet module wrapping around a trunk (often a transformer based module)

    The final output is of shape (batch, dim_output, L). Input is padded to ensure
    that L is a multiple of total_pool_size but the returned output is cropped to
    the original length L, removing the padded regions.
    """

    def __init__(
        self,
        trunk: nn.Module,
        num_downsampling: int,
        dim_input: int,
        dim_trunk: int,
        dim_output: int,
        encoder_kernel_size: int = 5,
        decoder_kernel_size: int = 3,
        activation: str = "gelu_tanh",
        norm: str = "group_32",
        dropout: float = 0.0,
        # Generally not necessary to have bias, especially with normalization and
        # residual connections
        bias: bool = False,
    ):
        super().__init__()

        if encoder_kernel_size % 2 == 0 or decoder_kernel_size % 2 == 0:
            raise ValueError(
                f"Only odd kernel sizes are supported, got {encoder_kernel_size} "
                f"and {decoder_kernel_size}. This constraint is applied to make "
                "the cropping logic easier to implement."
            )

        self.trunk = trunk

        # num_downsampling - 1 encoders apply downsampling and an additional
        # downsampling is applied right after the encoders, before the trunk
        self.total_pool_size = 2**num_downsampling

        self.encoder_blocks = UNetEncoderBlocks(
            num_downsampling=num_downsampling,
            dim_in=dim_input,
            dim_out=dim_trunk,
            kernel_size=encoder_kernel_size,
            activation=activation,
            norm=norm,
            dropout=dropout,
            bias=bias,
        )

        encoder_channels = self.encoder_blocks.channels

        self.decoder_blocks = UNetDecoderBlocks(
            num_upsampling=num_downsampling,
            skip_channels=encoder_channels,
            dim_in=dim_trunk,
            dim_out=dim_output,
            kernel_size=decoder_kernel_size,
            activation=activation,
            norm=norm,
            dropout=dropout,
            bias=bias,
        )

        # Store this for calculating cropped length for early cropping
        self.decoder_kernel_size = decoder_kernel_size

        assert len(self.encoder_blocks.encoders) + 1 == len(
            self.decoder_blocks.decoders
        ), "There should be one more decoder than encoder"

    def forward(
        self,
        input: Float[Tensor, "b d l"],
        cropped_length: int,
        transpose_for_trunk: bool = True,
    ) -> Float[Tensor, "b d l_cropped"]:
        """Forward pass for the UNet with efficient cropping.

        When the final output of the UNet is expected to be cropped, this method applies
        early cropping on the embedding right after the trunk. This saves unnecessary
        computation from decoders by removing tokens that will be cropped.

        When the final output is large and cropped_length is a significant portion of
        the input length (e.g. single-bp resolution), this can save quite a bit of
        computation and memory.
        """
        assert (
            cropped_length >= 0
        ), f"Cropped length must be non-negative, got {cropped_length}"

        assert (
            len(input.shape) == 3
        ), f"Input shape must be (batch_size, hidden_dim, length), got {input.shape}"
        L = input.shape[-1]

        # Pad input so that sequence length is divisible by total pool size
        padding_size = (
            0
            if L % self.total_pool_size == 0  # no padding needed if already divisible
            else self.total_pool_size - L % self.total_pool_size
        )
        x = F.pad(input, (0, padding_size), mode="constant", value=0)  # pad only right

        l_cropped_length = cropped_length
        r_cropped_length = cropped_length + padding_size

        # Calculate number of tokens after the trunk to crop. There are few
        # considerations:
        # 1. It would be preferable to have extra buffers to account for conv kernels
        #    near the edges.
        # 2. If cropping lengths are not divisible by total pool size, then there
        #    needs to be extra token at the edges that will be partially cropped after
        #    all the upsamplings.
        # 3. This means there will be some residual cropping that needs to occur at the
        #    very end.

        # Calculate number of tokens to crop, based on the final cropped length and
        # dividing by the total upsampling. As explain in 2) above, for any tokens
        # at the edge where part of the upsampled tokens are cropped, leave them
        # (i.e. round down the number of tokens to crop)
        l_tokens_to_crop = max(
            0,
            l_cropped_length // self.total_pool_size - self.decoder_kernel_size // 2
        )
        r_tokens_to_crop = max(
            0,
            r_cropped_length // self.total_pool_size - self.decoder_kernel_size // 2
        )

        # Now calculate the residual cropping that needs to occur at the end
        l_residual_crop = l_cropped_length - l_tokens_to_crop * self.total_pool_size
        r_residual_crop = r_cropped_length - r_tokens_to_crop * self.total_pool_size

        # Padded input x is used as a skip connection for reconstructing the final
        # upsampled resolution. The original input length is recovered by the final
        # residual cropping.
        x, skips = self.encoder_blocks(x)

        if transpose_for_trunk:
            x = x.mT  # (b, d, l) -> (b, l, d)

        x = self.trunk(x)

        if transpose_for_trunk:
            x = x.mT  # (b, l, d) -> (b, d, l)

        x = self.decoder_blocks(x, skips, l_tokens_to_crop, r_tokens_to_crop)

        # Apply residual cropping the get the original sequence length minus
        # the cropping lengths
        output = _crop_tensor(x, l_residual_crop, r_residual_crop)

        assert output.shape[-1] == L - cropped_length * 2, f"Output length after cropping {output.shape[-1]} does not match expected length {L - cropped_length * 2}"

        return output
