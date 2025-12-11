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
        assert expected == DILATION_PATTERN

    def test_not_old_brainchop_pattern(self) -> None:
        """Ensure we're NOT using the old BrainChop 8-layer pattern."""
        old_pattern = [1, 1, 2, 4, 8, 16, 1, 1]
        assert old_pattern != DILATION_PATTERN


class TestParameterCounts:
    """Tests for exact parameter count verification."""

    @pytest.mark.parametrize(
        ("channels", "expected"),
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
            f"MeshNet-{channels} parameter count mismatch: expected {expected:,}, got {actual:,}"
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

    @pytest.mark.slow
    def test_full_volume_shape(self) -> None:
        """Verify 256^3 input produces 256^3 output."""
        model = MeshNet(channels=5)  # Use smallest for memory

        # Create minimal tensor for shape check
        x = torch.zeros(1, 1, 256, 256, 256)
        output = model(x)

        assert output.shape == (1, 2, 256, 256, 256)

    def test_final_layer_is_1x1_conv(self) -> None:
        """Verify final layer uses 1x1x1 kernel."""
        model = MeshNet(channels=16)

        # Get the last Conv3d layer
        conv_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv3d)]
        final_layer = conv_layers[-1]

        assert final_layer.kernel_size == (1, 1, 1)
        assert final_layer.padding == (0, 0, 0)
        assert final_layer.dilation == (1, 1, 1)

    def test_intermediate_layers_are_3x3_conv(self) -> None:
        """Verify layers 1-9 use 3x3x3 kernel."""
        model = MeshNet(channels=16)

        conv_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv3d)]

        # Check layers 1-9 (index 0-8)
        for i, layer in enumerate(conv_layers[:-1]):
            assert layer.kernel_size == (3, 3, 3), f"Layer {i + 1} kernel size mismatch"

    def test_padding_equals_dilation(self) -> None:
        """Verify padding = dilation for 3x3x3 layers (maintains resolution).

        This enforces the full Conv3D(k=3, p=d, d=d) spec where both padding
        and dilation must match the pattern.
        """
        model = MeshNet(channels=16)

        conv_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv3d)]

        for i, (layer, dilation) in enumerate(
            zip(conv_layers[:-1], DILATION_PATTERN[:-1], strict=True)
        ):
            expected_padding = (dilation, dilation, dilation)
            expected_dilation = (dilation, dilation, dilation)
            assert layer.padding == expected_padding, f"Layer {i + 1} padding mismatch"
            assert layer.dilation == expected_dilation, f"Layer {i + 1} dilation mismatch"

    def test_no_batchnorm_after_final_layer(self) -> None:
        """Verify no BatchNorm after final 1x1x1 conv."""
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
            create_meshnet("meshnet-100")

    def test_create_meshnet_underscore_variants(self) -> None:
        """Test create_meshnet accepts underscore variants for Hydra compatibility."""
        # Hydra configs use underscores (meshnet_26.yaml), factory uses hyphens
        model = create_meshnet("meshnet_26")
        assert model.channels == 26
        assert model.count_parameters() == 147_474

        model = create_meshnet("meshnet_16")
        assert model.channels == 16

        model = create_meshnet("meshnet_5")
        assert model.channels == 5


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
        tiny_volume = tiny_volume.clone().detach().requires_grad_(True)

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
        """Test MeshNet-5 forward pass with 32^3 volume."""
        model = MeshNet(channels=5)
        output = model(small_volume)
        assert output.shape[2:] == small_volume.shape[2:]

    def test_meshnet_16_fits_in_memory(self, small_volume: torch.Tensor) -> None:
        """Test MeshNet-16 forward pass with 32^3 volume."""
        model = MeshNet(channels=16)
        output = model(small_volume)
        assert output.shape[2:] == small_volume.shape[2:]

    def test_meshnet_26_fits_in_memory(self, small_volume: torch.Tensor) -> None:
        """Test MeshNet-26 forward pass with 32^3 volume."""
        model = MeshNet(channels=26)
        output = model(small_volume)
        assert output.shape[2:] == small_volume.shape[2:]
