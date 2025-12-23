"""Tests for preprocessing functions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arc_meshchop.data.preprocessing import (
    compute_lesion_volume,
    crop_or_pad,
    get_lesion_quintile,
    get_spacing,
    normalize_intensity,
    preprocess_volume,
    resample_volume,
)


class TestNormalizeIntensity:
    """Tests for intensity normalization."""

    def test_output_range(self) -> None:
        """Verify output is in [0, 1] range."""
        data = np.random.randn(10, 10, 10).astype(np.float32) * 100 + 50
        normalized = normalize_intensity(data)

        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_min_is_zero(self) -> None:
        """Verify minimum value is 0."""
        data = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
        normalized = normalize_intensity(data)

        assert np.isclose(normalized.min(), 0.0)

    def test_max_is_one(self) -> None:
        """Verify maximum value is 1."""
        data = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
        normalized = normalize_intensity(data)

        assert np.isclose(normalized.max(), 1.0)

    def test_constant_volume(self) -> None:
        """Handle constant volume (all same value)."""
        data = np.ones((10, 10, 10), dtype=np.float32) * 5.0
        normalized = normalize_intensity(data)

        # Should return zeros for constant volume
        assert np.allclose(normalized, 0.0)

    def test_preserves_dtype(self) -> None:
        """Verify output is float32."""
        data = np.random.randn(5, 5, 5).astype(np.float32)
        normalized = normalize_intensity(data)

        assert normalized.dtype == np.float32


class TestCropOrPad:
    """Tests for crop_or_pad function."""

    def test_no_change_needed(self) -> None:
        """Test when shape already matches target."""
        data = np.ones((64, 64, 64), dtype=np.float32)
        result = crop_or_pad(data, (64, 64, 64))

        assert result.shape == (64, 64, 64)
        assert np.allclose(result, 1.0)

    def test_padding_small_volume(self) -> None:
        """Test padding smaller volume."""
        data = np.ones((32, 32, 32), dtype=np.float32)
        result = crop_or_pad(data, (64, 64, 64))

        assert result.shape == (64, 64, 64)
        # Check data is centered (zeros on edges)
        assert result[0, 0, 0] == 0.0  # Padded corner
        assert result[32, 32, 32] == 1.0  # Center (has data)

    def test_cropping_large_volume(self) -> None:
        """Test cropping larger volume."""
        data = np.ones((100, 100, 100), dtype=np.float32)
        result = crop_or_pad(data, (64, 64, 64))

        assert result.shape == (64, 64, 64)
        assert np.allclose(result, 1.0)  # All ones after cropping

    def test_asymmetric_crop_pad(self) -> None:
        """Test with asymmetric dimensions."""
        data = np.ones((50, 100, 30), dtype=np.float32)
        result = crop_or_pad(data, (64, 64, 64))

        assert result.shape == (64, 64, 64)

    def test_preserves_values(self) -> None:
        """Test that non-zero values are preserved."""
        data = np.zeros((32, 32, 32), dtype=np.float32)
        data[10:20, 10:20, 10:20] = 5.0
        result = crop_or_pad(data, (64, 64, 64))

        # The 5.0 values should be preserved somewhere in the result
        assert 5.0 in result


class TestResampleVolume:
    """Tests for volume resampling."""

    def test_output_shape(self) -> None:
        """Verify output has target shape."""
        data = np.random.rand(50, 50, 50).astype(np.float32)
        result = resample_volume(
            data,
            current_spacing=(2.0, 2.0, 2.0),
            target_spacing=(1.0, 1.0, 1.0),
            target_shape=(64, 64, 64),
        )

        assert result.shape == (64, 64, 64)

    def test_mask_uses_nearest_neighbor(self) -> None:
        """Verify masks use nearest-neighbor interpolation."""
        # Binary mask
        data = np.zeros((50, 50, 50), dtype=np.float32)
        data[20:30, 20:30, 20:30] = 1.0

        result = resample_volume(
            data,
            current_spacing=(2.0, 2.0, 2.0),
            target_spacing=(1.0, 1.0, 1.0),
            target_shape=(100, 100, 100),
            is_mask=True,
        )

        # Should only have 0 and 1 values (no interpolation artifacts)
        unique = np.unique(result)
        assert len(unique) == 2
        assert 0.0 in unique
        assert 1.0 in unique

    def test_preserves_dtype(self) -> None:
        """Verify output is float32."""
        data = np.random.rand(32, 32, 32).astype(np.float32)
        result = resample_volume(
            data,
            current_spacing=(1.0, 1.0, 1.0),
            target_spacing=(1.0, 1.0, 1.0),
            target_shape=(32, 32, 32),
        )

        assert result.dtype == np.float32

    def test_downsampling(self) -> None:
        """Test resampling to smaller size."""
        data = np.random.rand(100, 100, 100).astype(np.float32)
        result = resample_volume(
            data,
            current_spacing=(0.5, 0.5, 0.5),
            target_spacing=(1.0, 1.0, 1.0),
            target_shape=(50, 50, 50),
        )

        assert result.shape == (50, 50, 50)


class TestGetSpacing:
    """Tests for spacing extraction from affine."""

    def test_identity_affine(self) -> None:
        """Test with identity affine (1mm spacing)."""
        affine = np.eye(4, dtype=np.float64)
        spacing = get_spacing(affine)

        assert np.allclose(spacing, (1.0, 1.0, 1.0))

    def test_scaled_affine(self) -> None:
        """Test with scaled affine."""
        affine = np.diag([2.0, 0.5, 1.5, 1.0]).astype(np.float64)
        spacing = get_spacing(affine)

        assert np.allclose(spacing, (2.0, 0.5, 1.5))

    def test_returns_tuple(self) -> None:
        """Verify returns tuple of floats."""
        affine = np.eye(4, dtype=np.float64)
        spacing = get_spacing(affine)

        assert isinstance(spacing, tuple)
        assert len(spacing) == 3
        assert all(isinstance(s, float) for s in spacing)


class TestLesionVolume:
    """Tests for lesion volume computation."""

    def test_compute_lesion_volume(self) -> None:
        """Test lesion volume calculation."""
        mask = np.zeros((10, 10, 10), dtype=np.float32)
        mask[2:5, 2:5, 2:5] = 1.0  # 3x3x3 = 27 voxels

        volume = compute_lesion_volume(mask)
        assert volume == 27

    def test_empty_mask(self) -> None:
        """Test with empty mask."""
        mask = np.zeros((10, 10, 10), dtype=np.float32)
        volume = compute_lesion_volume(mask)
        assert volume == 0

    def test_full_mask(self) -> None:
        """Test with full mask."""
        mask = np.ones((10, 10, 10), dtype=np.float32)
        volume = compute_lesion_volume(mask)
        assert volume == 1000

    def test_returns_int(self) -> None:
        """Verify returns integer."""
        mask = np.ones((5, 5, 5), dtype=np.float32)
        volume = compute_lesion_volume(mask)
        assert isinstance(volume, int)


class TestLesionQuintile:
    """Tests for lesion quintile assignment."""

    @pytest.mark.parametrize(
        ("volume", "expected"),
        [
            (500, "Q1"),  # Small
            (50_000, "Q2"),  # Small-medium
            (100_000, "Q3"),  # Medium
            (200_000, "Q4"),  # Large
            (100, "Q1"),  # Very small (edge case)
            (400_000, "Q4"),  # Very large (edge case)
        ],
    )
    def test_quintile_assignment(self, volume: int, expected: str) -> None:
        """Test quintile assignment for various volumes."""
        result = get_lesion_quintile(volume)
        assert result == expected


class TestPreprocessVolume:
    """Tests for end-to-end preprocess_volume()."""

    def test_nibabel_conform_backend(self, tmp_path: Path) -> None:
        """Test nibabel_conform resampling produces valid output."""
        import nibabel as nib

        # Create a tiny synthetic volume with a non-identity affine to exercise
        # canonicalization + resampling.
        image = np.random.rand(10, 12, 14).astype(np.float32) * 1000
        mask = np.zeros((10, 12, 14), dtype=np.float32)
        mask[3:5, 4:7, 6:9] = 1.0

        affine = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        image_path = tmp_path / "image.nii.gz"
        mask_path = tmp_path / "mask.nii.gz"
        nib.save(nib.Nifti1Image(image, affine), str(image_path))  # type: ignore[attr-defined]
        nib.save(nib.Nifti1Image(mask, affine), str(mask_path))  # type: ignore[attr-defined]

        out_image, out_mask = preprocess_volume(
            image_path,
            mask_path,
            target_shape=(16, 16, 16),
            target_spacing=(1.0, 1.0, 1.0),
            resample_method="nibabel_conform",
        )

        assert out_image.shape == (16, 16, 16)
        assert out_mask is not None
        assert out_mask.shape == (16, 16, 16)
        assert out_image.dtype == np.float32
        assert out_mask.dtype == np.float32
        assert out_image.min() >= 0.0
        assert out_image.max() <= 1.0

        unique = np.unique(out_mask)
        assert 0.0 in unique
        assert 1.0 in unique

    def test_boundary_values(self) -> None:
        """Test exact boundary values."""
        # Q1 upper boundary
        assert get_lesion_quintile(33_619) == "Q1"

        # Q2 boundaries
        assert get_lesion_quintile(33_620) == "Q2"
        assert get_lesion_quintile(67_891) == "Q2"

        # Q3 boundaries
        assert get_lesion_quintile(67_892) == "Q3"
        assert get_lesion_quintile(128_314) == "Q3"

        # Q4 boundaries
        assert get_lesion_quintile(128_315) == "Q4"

    def test_returns_string(self) -> None:
        """Verify returns string label."""
        result = get_lesion_quintile(10_000)
        assert isinstance(result, str)
        assert result.startswith("Q")
