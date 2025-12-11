"""MeshNet architecture for stroke lesion segmentation.

This implements the REVISITED MeshNet architecture from:
"State-of-the-Art Stroke Lesion Segmentation at 1/1000th of Parameters"
Fedorov et al. (Emory, Georgia State, USC)

CRITICAL: This uses the NEW 10-layer symmetric dilation pattern:
    [1, 2, 4, 8, 16, 16, 8, 4, 2, 1]

DO NOT use the old BrainChop 8-layer pattern:
    [1, 1, 2, 4, 8, 16, 1, 1]
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

# Expected parameter counts from the paper (Table 1)
EXPECTED_PARAMS = {
    5: 5_682,
    16: 56_194,
    26: 147_474,
}

# NEW symmetric dilation pattern (10 layers)
DILATION_PATTERN = [1, 2, 4, 8, 16, 16, 8, 4, 2, 1]


def _init_weights(module: nn.Module) -> None:
    """Initialize weights using Xavier normal (verified from BrainChop).

    Args:
        module: PyTorch module to initialize.
    """
    if isinstance(module, nn.Conv3d):
        nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain("relu"))
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class ConvBNReLU(nn.Module):
    """Convolutional block: Conv3D -> BatchNorm3d -> ReLU -> Dropout3d.

    This is the building block for MeshNet layers 1-9.
    Verified from BrainChop reference implementation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout_p: float = 0.0,
    ) -> None:
        """Initialize ConvBNReLU block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Convolution kernel size.
            dilation: Dilation rate (padding = dilation for same output size).
            dropout_p: Dropout probability (0.0 = no dropout).
        """
        super().__init__()

        # Padding = dilation to maintain spatial dimensions
        padding = dilation if kernel_size == 3 else 0

        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=True,  # Verified from BrainChop
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, D, H, W).

        Returns:
            Output tensor of shape (B, C_out, D, H, W).
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class MeshNet(nn.Module):
    """MeshNet: 10-layer fully convolutional segmentation network.

    Uses symmetric dilation pattern to mimic encoder-decoder without
    downsampling, upsampling, or skip connections.

    Architecture:
        - Layers 1-9: Conv3d(k=3) -> BatchNorm3d -> ReLU -> Dropout3d
        - Layer 10: Conv3d(k=1) -> Output (no activation)

    Dilation pattern:
        1 -> 2 -> 4 -> 8 -> 16 -> 16 -> 8 -> 4 -> 2 -> 1
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        channels: int = 26,
        dropout_p: float = 0.0,
    ) -> None:
        """Initialize MeshNet.

        Args:
            in_channels: Number of input channels (1 for single MRI modality).
            num_classes: Number of output classes (2 for binary segmentation).
            channels: Number of intermediate channels (5, 16, or 26).
            dropout_p: Dropout probability (0.0 = no dropout).
        """
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.channels = channels
        self.dropout_p = dropout_p

        # Build layers
        layers: list[nn.Module] = []
        prev_ch = in_channels

        for i, dilation in enumerate(DILATION_PATTERN):
            is_last = i == len(DILATION_PATTERN) - 1

            if is_last:
                # Final layer: 1x1x1 conv, no BN/ReLU (verified from BrainChop)
                layers.append(
                    nn.Conv3d(
                        in_channels=prev_ch,
                        out_channels=num_classes,
                        kernel_size=1,
                        padding=0,
                        dilation=1,
                        bias=True,
                    )
                )
            else:
                # Standard layer: Conv -> BN -> ReLU -> Dropout
                out_ch = channels
                layers.append(
                    ConvBNReLU(
                        in_channels=prev_ch,
                        out_channels=out_ch,
                        kernel_size=3,
                        dilation=dilation,
                        dropout_p=dropout_p,
                    )
                )
                prev_ch = out_ch

        self.network = nn.Sequential(*layers)

        # Initialize weights (Xavier normal, verified from BrainChop)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 1, D, H, W).

        Returns:
            Logits tensor of shape (B, num_classes, D, H, W).
        """
        output: torch.Tensor = self.network(x)
        return output

    def count_parameters(self) -> int:
        """Count total trainable parameters.

        Returns:
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def verify_parameter_count(self) -> bool:
        """Verify parameter count matches paper.

        Returns:
            True if parameter count matches expected value.

        Raises:
            ValueError: If channel count not in expected values.
        """
        if self.channels not in EXPECTED_PARAMS:
            raise ValueError(
                f"No expected parameter count for channels={self.channels}. "
                f"Expected one of: {list(EXPECTED_PARAMS.keys())}"
            )

        expected = EXPECTED_PARAMS[self.channels]
        actual = self.count_parameters()

        return actual == expected


# =============================================================================
# Factory Functions
# =============================================================================


def meshnet_5(dropout_p: float = 0.0) -> MeshNet:
    """Create MeshNet-5 (5,682 parameters).

    Args:
        dropout_p: Dropout probability.

    Returns:
        MeshNet-5 model.
    """
    return MeshNet(channels=5, dropout_p=dropout_p)


def meshnet_16(dropout_p: float = 0.0) -> MeshNet:
    """Create MeshNet-16 (56,194 parameters).

    Args:
        dropout_p: Dropout probability.

    Returns:
        MeshNet-16 model.
    """
    return MeshNet(channels=16, dropout_p=dropout_p)


def meshnet_26(dropout_p: float = 0.0) -> MeshNet:
    """Create MeshNet-26 (147,474 parameters).

    Args:
        dropout_p: Dropout probability.

    Returns:
        MeshNet-26 model.
    """
    return MeshNet(channels=26, dropout_p=dropout_p)


MeshNetVariant = Literal["meshnet-5", "meshnet-16", "meshnet-26"]


def create_meshnet(
    variant: str,
    dropout_p: float = 0.0,
) -> MeshNet:
    """Create MeshNet model by variant name.

    Accepts both hyphenated ("meshnet-26") and underscored ("meshnet_26") names
    for compatibility with Hydra configs.

    Args:
        variant: Model variant (e.g., "meshnet-26" or "meshnet_26").
        dropout_p: Dropout probability.

    Returns:
        MeshNet model.

    Raises:
        ValueError: If variant is unknown.
    """
    # Normalize: underscore → hyphen for consistent lookup
    normalized = variant.replace("_", "-")

    factories = {
        "meshnet-5": meshnet_5,
        "meshnet-16": meshnet_16,
        "meshnet-26": meshnet_26,
    }

    if normalized not in factories:
        raise ValueError(
            f"Unknown variant: {variant}. Expected one of: {list(factories.keys())} "
            "(underscore variants also accepted)"
        )

    return factories[normalized](dropout_p=dropout_p)
