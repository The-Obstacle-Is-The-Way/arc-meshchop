# Spec 02: MeshNet Architecture Implementation

> **Phase 2 of 7** — Core neural network implementation
>
> **Goal:** Implement the 10-layer MeshNet architecture with exact parameter counts matching the paper.

---

## Overview

This spec covers:
- MeshNet architecture implementation with symmetric dilation pattern
- Parameter count verification (5,682 / 56,194 / 147,474)
- Weight initialization (Xavier normal)
- Unit tests for architecture correctness

---

## 1. Architecture Specification

### 1.1 Dilation Pattern (Critical)

**FROM PAPER (NEW 10-layer symmetric pattern):**
```
1 → 2 → 4 → 8 → 16 → 16 → 8 → 4 → 2 → 1
└───── encoder ─────┘  └───── decoder ─────┘
```

**DO NOT USE old BrainChop pattern:**
```
1 → 1 → 2 → 4 → 8 → 16 → 1 → 1  (8 layers, WRONG)
```

### 1.2 Layer Structure

| Layer | Input Ch | Output Ch | Kernel | Dilation | Padding | BatchNorm | ReLU |
|-------|----------|-----------|--------|----------|---------|-----------|------|
| 1 | 1 | X | 3×3×3 | 1 | 1 | ✅ | ✅ |
| 2 | X | X | 3×3×3 | 2 | 2 | ✅ | ✅ |
| 3 | X | X | 3×3×3 | 4 | 4 | ✅ | ✅ |
| 4 | X | X | 3×3×3 | 8 | 8 | ✅ | ✅ |
| 5 | X | X | 3×3×3 | 16 | 16 | ✅ | ✅ |
| 6 | X | X | 3×3×3 | 16 | 16 | ✅ | ✅ |
| 7 | X | X | 3×3×3 | 8 | 8 | ✅ | ✅ |
| 8 | X | X | 3×3×3 | 4 | 4 | ✅ | ✅ |
| 9 | X | X | 3×3×3 | 2 | 2 | ✅ | ✅ |
| 10 | X | 2 | 1×1×1 | 1 | 0 | ❌ | ❌ |

Where **X** = number of channels (5, 16, or 26)

### 1.3 Parameter Counts (Verification Targets)

| Variant | Channels | Expected Params | Source |
|---------|----------|-----------------|--------|
| MeshNet-5 | 5 | 5,682 | Paper Table 1 |
| MeshNet-16 | 16 | 56,194 | Paper Table 1 |
| MeshNet-26 | 26 | 147,474 | Paper Table 1 |

### 1.4 Component Details

> **NOTE: Implementation Details NOT Specified in Paper**
> The paper only describes the dilation pattern and layer count. The following details are
> **inferred from the BrainChop reference implementation**, not explicitly stated in the paper:
> - Normalization: BatchNorm3d (not GroupNorm, LayerNorm, etc.)
> - Activation: ReLU (not LeakyReLU, GELU, etc.)
> - Padding: padding = dilation (to maintain spatial dimensions)
> - Bias: Conv3d bias enabled
>
> These choices are standard for convolutional segmentation networks and match BrainChop's
> verified implementation. If metrics don't reproduce, these are candidates for investigation.

**INFERRED from BrainChop reference** (local read-only copy in `_references/brainchop/`):

| Component | Specification | Source |
|-----------|---------------|--------|
| Conv3D bias | True (initialized to 0.0) | BrainChop |
| Weight init | Xavier normal with gain=relu | BrainChop |
| BatchNorm | affine=True, track_running_stats=True | BrainChop |
| Dropout | Optional (Dropout3d, default p=0.0) | BrainChop |
| Activation | ReLU (inplace=True) | BrainChop |
| Padding | padding = dilation for k=3 layers | BrainChop |

---

## 2. Implementation

### 2.1 Core MeshNet Class

**File:** `src/arc_meshchop/models/meshnet.py`

```python
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
    """Convolutional block: Conv3D → BatchNorm3d → ReLU → Dropout3d.

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
        - Layers 1-9: Conv3d(k=3) → BatchNorm3d → ReLU → Dropout3d
        - Layer 10: Conv3d(k=1) → Output (no activation)

    Dilation pattern:
        1 → 2 → 4 → 8 → 16 → 16 → 8 → 4 → 2 → 1
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
                # Final layer: 1×1×1 conv, no BN/ReLU (verified from BrainChop)
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
                # Standard layer: Conv → BN → ReLU → Dropout
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
        return self.network(x)

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
    variant: MeshNetVariant,
    dropout_p: float = 0.0,
) -> MeshNet:
    """Create MeshNet model by variant name.

    Args:
        variant: Model variant ("meshnet-5", "meshnet-16", "meshnet-26").
        dropout_p: Dropout probability.

    Returns:
        MeshNet model.

    Raises:
        ValueError: If variant is unknown.
    """
    factories = {
        "meshnet-5": meshnet_5,
        "meshnet-16": meshnet_16,
        "meshnet-26": meshnet_26,
    }

    if variant not in factories:
        raise ValueError(f"Unknown variant: {variant}. Expected one of: {list(factories.keys())}")

    return factories[variant](dropout_p=dropout_p)
```

### 2.2 Module `__init__.py`

**File:** `src/arc_meshchop/models/__init__.py`

```python
"""MeshNet models for stroke lesion segmentation."""

from arc_meshchop.models.meshnet import (
    DILATION_PATTERN,
    EXPECTED_PARAMS,
    ConvBNReLU,
    MeshNet,
    MeshNetVariant,
    create_meshnet,
    meshnet_5,
    meshnet_16,
    meshnet_26,
)

__all__ = [
    "ConvBNReLU",
    "DILATION_PATTERN",
    "EXPECTED_PARAMS",
    "MeshNet",
    "MeshNetVariant",
    "create_meshnet",
    "meshnet_5",
    "meshnet_16",
    "meshnet_26",
]
```

---

## 3. Test-Driven Development

### 3.1 Parameter Count Tests (Critical)

**File:** `tests/test_models/test_meshnet.py`

```python
"""Tests for MeshNet architecture.

These tests verify that the implementation matches the paper exactly,
including parameter counts and architectural details.
"""

import pytest
import torch

from arc_meshchop.models import (
    DILATION_PATTERN,
    EXPECTED_PARAMS,
    MeshNet,
    create_meshnet,
    meshnet_5,
    meshnet_16,
    meshnet_26,
)


class TestDilationPattern:
    """Tests for dilation pattern correctness."""

    def test_dilation_pattern_is_symmetric(self) -> None:
        """Verify dilation pattern is symmetric (encoder-decoder)."""
        pattern = DILATION_PATTERN
        mid = len(pattern) // 2

        # First half (encoder) should mirror second half (decoder)
        encoder = pattern[:mid]
        decoder = pattern[mid:]

        assert encoder == list(reversed(decoder))

    def test_dilation_pattern_length(self) -> None:
        """Verify 10-layer pattern (not old 8-layer)."""
        assert len(DILATION_PATTERN) == 10

    def test_dilation_pattern_values(self) -> None:
        """Verify exact dilation values from paper."""
        expected = [1, 2, 4, 8, 16, 16, 8, 4, 2, 1]
        assert DILATION_PATTERN == expected

    def test_not_old_brainchop_pattern(self) -> None:
        """Ensure we're NOT using the old BrainChop 8-layer pattern."""
        old_pattern = [1, 1, 2, 4, 8, 16, 1, 1]
        assert DILATION_PATTERN != old_pattern


class TestParameterCounts:
    """Tests for exact parameter count verification."""

    @pytest.mark.parametrize(
        "channels,expected",
        [
            (5, 5_682),
            (16, 56_194),
            (26, 147_474),
        ],
    )
    def test_parameter_count_exact(self, channels: int, expected: int) -> None:
        """Verify parameter counts match paper Table 1 exactly."""
        model = MeshNet(channels=channels)
        actual = model.count_parameters()

        assert actual == expected, (
            f"MeshNet-{channels} parameter count mismatch: "
            f"expected {expected:,}, got {actual:,}"
        )

    def test_meshnet_5_params(self) -> None:
        """Verify MeshNet-5 has exactly 5,682 parameters."""
        model = meshnet_5()
        assert model.count_parameters() == 5_682

    def test_meshnet_16_params(self) -> None:
        """Verify MeshNet-16 has exactly 56,194 parameters."""
        model = meshnet_16()
        assert model.count_parameters() == 56_194

    def test_meshnet_26_params(self) -> None:
        """Verify MeshNet-26 has exactly 147,474 parameters."""
        model = meshnet_26()
        assert model.count_parameters() == 147_474

    def test_verify_parameter_count_method(self) -> None:
        """Test the verify_parameter_count method."""
        for channels in EXPECTED_PARAMS:
            model = MeshNet(channels=channels)
            assert model.verify_parameter_count()


class TestArchitecture:
    """Tests for architectural correctness."""

    def test_layer_count(self) -> None:
        """Verify model has 10 layers."""
        model = MeshNet(channels=16)

        # Count Conv3d layers
        conv_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv3d)]
        assert len(conv_layers) == 10

    def test_input_output_shape(self, tiny_volume: torch.Tensor) -> None:
        """Verify output shape matches input spatial dimensions."""
        model = MeshNet(channels=16)
        output = model(tiny_volume)

        # Input: (B, 1, D, H, W) -> Output: (B, 2, D, H, W)
        B, _, D, H, W = tiny_volume.shape
        assert output.shape == (B, 2, D, H, W)

    def test_full_volume_shape(self) -> None:
        """Verify 256³ input produces 256³ output."""
        model = MeshNet(channels=5)  # Use smallest for memory

        # Create minimal tensor for shape check
        x = torch.zeros(1, 1, 256, 256, 256)
        output = model(x)

        assert output.shape == (1, 2, 256, 256, 256)

    def test_final_layer_is_1x1_conv(self) -> None:
        """Verify final layer uses 1×1×1 kernel."""
        model = MeshNet(channels=16)

        # Get the last Conv3d layer
        conv_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv3d)]
        final_layer = conv_layers[-1]

        assert final_layer.kernel_size == (1, 1, 1)
        assert final_layer.padding == (0, 0, 0)
        assert final_layer.dilation == (1, 1, 1)

    def test_intermediate_layers_are_3x3_conv(self) -> None:
        """Verify layers 1-9 use 3×3×3 kernel."""
        model = MeshNet(channels=16)

        conv_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv3d)]

        # Check layers 1-9 (index 0-8)
        for i, layer in enumerate(conv_layers[:-1]):
            assert layer.kernel_size == (3, 3, 3), f"Layer {i+1} kernel size mismatch"

    def test_padding_equals_dilation(self) -> None:
        """Verify padding = dilation for 3×3×3 layers (maintains resolution)."""
        model = MeshNet(channels=16)

        conv_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv3d)]

        for i, (layer, dilation) in enumerate(zip(conv_layers[:-1], DILATION_PATTERN[:-1])):
            expected_padding = (dilation, dilation, dilation)
            assert layer.padding == expected_padding, f"Layer {i+1} padding mismatch"

    def test_no_batchnorm_after_final_layer(self) -> None:
        """Verify no BatchNorm after final 1×1×1 conv."""
        model = MeshNet(channels=16)

        # The final element in the sequential should be Conv3d, not BatchNorm
        final_module = model.network[-1]
        assert isinstance(final_module, torch.nn.Conv3d)


class TestFactoryFunctions:
    """Tests for model factory functions."""

    def test_create_meshnet_5(self) -> None:
        """Test create_meshnet with meshnet-5 variant."""
        model = create_meshnet("meshnet-5")
        assert model.channels == 5
        assert model.count_parameters() == 5_682

    def test_create_meshnet_16(self) -> None:
        """Test create_meshnet with meshnet-16 variant."""
        model = create_meshnet("meshnet-16")
        assert model.channels == 16
        assert model.count_parameters() == 56_194

    def test_create_meshnet_26(self) -> None:
        """Test create_meshnet with meshnet-26 variant."""
        model = create_meshnet("meshnet-26")
        assert model.channels == 26
        assert model.count_parameters() == 147_474

    def test_create_meshnet_invalid_variant(self) -> None:
        """Test create_meshnet raises error for invalid variant."""
        with pytest.raises(ValueError, match="Unknown variant"):
            create_meshnet("meshnet-100")  # type: ignore


class TestWeightInitialization:
    """Tests for weight initialization."""

    def test_conv_bias_initialized_to_zero(self) -> None:
        """Verify Conv3d biases are initialized to 0.0 (from BrainChop)."""
        model = MeshNet(channels=16)

        for module in model.modules():
            if isinstance(module, torch.nn.Conv3d) and module.bias is not None:
                assert torch.allclose(module.bias, torch.zeros_like(module.bias))

    def test_weights_not_all_zeros(self) -> None:
        """Verify weights are initialized (not zero)."""
        model = MeshNet(channels=16)

        for module in model.modules():
            if isinstance(module, torch.nn.Conv3d):
                assert not torch.allclose(module.weight, torch.zeros_like(module.weight))


class TestDropout:
    """Tests for dropout functionality."""

    def test_dropout_zero_by_default(self) -> None:
        """Verify no dropout by default."""
        model = MeshNet(channels=16)
        assert model.dropout_p == 0.0

    def test_dropout_configurable(self) -> None:
        """Verify dropout can be configured."""
        model = MeshNet(channels=16, dropout_p=0.5)
        assert model.dropout_p == 0.5

        # Check Dropout3d layers exist
        dropout_layers = [m for m in model.modules() if isinstance(m, torch.nn.Dropout3d)]
        assert len(dropout_layers) == 9  # One per ConvBNReLU block


class TestGradients:
    """Tests for gradient flow."""

    def test_gradients_flow(self, tiny_volume: torch.Tensor) -> None:
        """Verify gradients can flow through the network."""
        model = MeshNet(channels=5)
        tiny_volume.requires_grad = True

        output = model(tiny_volume)
        loss = output.sum()
        loss.backward()

        assert tiny_volume.grad is not None
        assert not torch.isnan(tiny_volume.grad).any()

    def test_no_nan_in_forward(self, tiny_volume: torch.Tensor) -> None:
        """Verify no NaN values in forward pass."""
        model = MeshNet(channels=16)
        output = model(tiny_volume)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


@pytest.mark.slow
class TestMemory:
    """Memory-related tests (marked slow)."""

    def test_meshnet_5_fits_in_memory(self, small_volume: torch.Tensor) -> None:
        """Test MeshNet-5 forward pass with 32³ volume."""
        model = MeshNet(channels=5)
        output = model(small_volume)
        assert output.shape[2:] == small_volume.shape[2:]

    def test_meshnet_16_fits_in_memory(self, small_volume: torch.Tensor) -> None:
        """Test MeshNet-16 forward pass with 32³ volume."""
        model = MeshNet(channels=16)
        output = model(small_volume)
        assert output.shape[2:] == small_volume.shape[2:]

    def test_meshnet_26_fits_in_memory(self, small_volume: torch.Tensor) -> None:
        """Test MeshNet-26 forward pass with 32³ volume."""
        model = MeshNet(channels=26)
        output = model(small_volume)
        assert output.shape[2:] == small_volume.shape[2:]
```

### 3.2 Additional Test Fixtures

**Add to:** `tests/conftest.py`

```python
# Add these fixtures for model tests

@pytest.fixture
def meshnet_5_model() -> "MeshNet":
    """Create MeshNet-5 model."""
    from arc_meshchop.models import meshnet_5
    return meshnet_5()


@pytest.fixture
def meshnet_16_model() -> "MeshNet":
    """Create MeshNet-16 model."""
    from arc_meshchop.models import meshnet_16
    return meshnet_16()


@pytest.fixture
def meshnet_26_model() -> "MeshNet":
    """Create MeshNet-26 model."""
    from arc_meshchop.models import meshnet_26
    return meshnet_26()
```

---

## 4. Implementation Checklist

### Phase 2.1: Create MeshNet Module

- [ ] Create `src/arc_meshchop/models/meshnet.py`
- [ ] Implement `ConvBNReLU` block
- [ ] Implement `MeshNet` class with 10-layer pattern
- [ ] Implement `_init_weights` function
- [ ] Implement factory functions

### Phase 2.2: Create Tests

- [ ] Create `tests/test_models/test_meshnet.py`
- [ ] Write parameter count tests
- [ ] Write architecture tests
- [ ] Write gradient flow tests

### Phase 2.3: Verify Parameter Counts

- [ ] MeshNet-5: exactly 5,682 params
- [ ] MeshNet-16: exactly 56,194 params
- [ ] MeshNet-26: exactly 147,474 params
- [ ] All tests pass: `uv run pytest tests/test_models/`

### Phase 2.4: Verify Architecture

- [ ] 10 Conv3d layers total
- [ ] Layers 1-9: 3×3×3 kernel with padding=dilation
- [ ] Layer 10: 1×1×1 kernel with no padding
- [ ] BatchNorm after layers 1-9 only
- [ ] ReLU after layers 1-9 only

---

## 5. Verification Commands

```bash
# Run model tests
uv run pytest tests/test_models/ -v

# Run specific test
uv run pytest tests/test_models/test_meshnet.py::TestParameterCounts -v

# Verify parameter counts manually
python -c "
from arc_meshchop.models import meshnet_5, meshnet_16, meshnet_26

for model, name in [(meshnet_5(), 'MeshNet-5'),
                    (meshnet_16(), 'MeshNet-16'),
                    (meshnet_26(), 'MeshNet-26')]:
    params = model.count_parameters()
    verified = model.verify_parameter_count()
    print(f'{name}: {params:,} params (verified: {verified})')
"
```

---

## 6. Parameter Count Calculation

For reference, here's how the parameter counts are calculated:

### Layer 1 (Input → channels)
```
Conv3d(1, X, k=3): 1 × X × 27 + X = 27X + X = 28X
BatchNorm3d(X): 2X (gamma, beta)
Total: 30X
```

### Layers 2-9 (channels → channels)
```
Conv3d(X, X, k=3): X × X × 27 + X = 27X² + X
BatchNorm3d(X): 2X
Total per layer: 27X² + 3X
8 layers: 8(27X² + 3X) = 216X² + 24X
```

### Layer 10 (channels → 2)
```
Conv3d(X, 2, k=1): X × 2 × 1 + 2 = 2X + 2
```

### Total
```
Total = 30X + 216X² + 24X + 2X + 2
      = 216X² + 56X + 2
```

### Verification
```
X = 5:  216(25) + 56(5) + 2 = 5400 + 280 + 2 = 5,682 ✓
X = 16: 216(256) + 56(16) + 2 = 55296 + 896 + 2 = 56,194 ✓
X = 26: 216(676) + 56(26) + 2 = 145816 + 1456 + 2 = 147,474 ✓
```

---

## 7. Common Issues

| Issue | Solution |
|-------|----------|
| Parameter count off by small amount | Check bias=True on all Conv3d layers |
| Parameter count way off | Verify 10-layer pattern, not 8-layer |
| Shape mismatch | Ensure padding=dilation for k=3 layers |
| NaN in forward pass | Check input normalization (0-1 range) |
| OOM on 256³ | Use FP16 or smaller volume for tests |

---

## 8. References

- Paper: "State-of-the-Art Stroke Lesion Segmentation at 1/1000th of Parameters"
- BrainChop: [github.com/neuroneural/brainchop](https://github.com/neuroneural/brainchop) (JS-based, local read-only copy in `_references/brainchop/`)
- Research docs: `docs/archive/research/01-architecture.md`, `docs/archive/research/04-model-variants.md`
