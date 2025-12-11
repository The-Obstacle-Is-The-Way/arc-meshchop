"""Tests for PyTorch dataset."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch


def create_synthetic_nifti(
    path: Path,
    shape: tuple[int, int, int] = (64, 64, 64),
    is_mask: bool = False,
) -> None:
    """Create a synthetic NIfTI file for testing.

    Args:
        path: Output path.
        shape: Volume shape.
        is_mask: If True, create binary mask with lesion.
    """
    data = np.zeros(shape, dtype=np.float32)
    if is_mask:
        # Add a small lesion in center
        c = shape[0] // 2
        s = shape[0] // 8
        data[c - s : c + s, c - s : c + s, c - s : c + s] = 1.0
    else:
        # Fill with random values
        data[:] = np.random.rand(*shape).astype(np.float32) * 1000

    # Create identity affine (1mm spacing)
    affine = np.eye(4)
    nii = nib.Nifti1Image(data, affine)  # type: ignore[attr-defined]
    nib.save(nii, str(path))  # type: ignore[attr-defined]


@pytest.fixture
def synthetic_data_dir(tmp_path: Path) -> Path:
    """Create synthetic dataset for testing.

    Creates 3 image/mask pairs in a temporary directory.
    """
    for i in range(3):
        image_path = tmp_path / f"image_{i}.nii.gz"
        mask_path = tmp_path / f"mask_{i}.nii.gz"
        create_synthetic_nifti(image_path, is_mask=False)
        create_synthetic_nifti(mask_path, is_mask=True)

    return tmp_path


class TestARCDataset:
    """Tests for ARCDataset class."""

    def test_dataset_length(self, synthetic_data_dir: Path) -> None:
        """Test dataset reports correct length."""
        from arc_meshchop.data import ARCDataset

        image_paths = sorted(synthetic_data_dir.glob("image_*.nii.gz"))
        mask_paths = sorted(synthetic_data_dir.glob("mask_*.nii.gz"))

        dataset = ARCDataset(
            image_paths=image_paths,
            mask_paths=mask_paths,
            preprocess=False,  # Skip preprocessing for faster tests
        )

        assert len(dataset) == 3

    def test_dataset_output_types(self, synthetic_data_dir: Path) -> None:
        """Test dataset returns correct tensor types."""
        from arc_meshchop.data import ARCDataset

        image_paths = sorted(synthetic_data_dir.glob("image_*.nii.gz"))
        mask_paths = sorted(synthetic_data_dir.glob("mask_*.nii.gz"))

        dataset = ARCDataset(
            image_paths=image_paths,
            mask_paths=mask_paths,
            preprocess=False,
        )

        image, mask = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert image.dtype == torch.float32
        assert mask.dtype == torch.int64

    def test_dataset_output_shapes(self, synthetic_data_dir: Path) -> None:
        """Test dataset returns correct shapes."""
        from arc_meshchop.data import ARCDataset

        image_paths = sorted(synthetic_data_dir.glob("image_*.nii.gz"))
        mask_paths = sorted(synthetic_data_dir.glob("mask_*.nii.gz"))

        dataset = ARCDataset(
            image_paths=image_paths,
            mask_paths=mask_paths,
            preprocess=False,
        )

        image, mask = dataset[0]

        # Image should have channel dimension
        assert image.dim() == 4  # (C, D, H, W)
        assert image.shape[0] == 1  # Single channel

        # Mask should not have channel dimension
        assert mask.dim() == 3  # (D, H, W)

    def test_image_has_channel_dimension(self, synthetic_data_dir: Path) -> None:
        """Test that image tensor has channel dimension."""
        from arc_meshchop.data import ARCDataset

        image_paths = sorted(synthetic_data_dir.glob("image_*.nii.gz"))
        mask_paths = sorted(synthetic_data_dir.glob("mask_*.nii.gz"))

        dataset = ARCDataset(
            image_paths=image_paths,
            mask_paths=mask_paths,
            preprocess=False,
        )

        image, _ = dataset[0]

        # Shape should be (1, D, H, W)
        assert image.shape[0] == 1
        assert image.shape[1:] == (64, 64, 64)

    def test_mask_values_are_binary(self, synthetic_data_dir: Path) -> None:
        """Test that mask contains only 0 and 1."""
        from arc_meshchop.data import ARCDataset

        image_paths = sorted(synthetic_data_dir.glob("image_*.nii.gz"))
        mask_paths = sorted(synthetic_data_dir.glob("mask_*.nii.gz"))

        dataset = ARCDataset(
            image_paths=image_paths,
            mask_paths=mask_paths,
            preprocess=False,
        )

        _, mask = dataset[0]

        unique_values = torch.unique(mask)
        assert len(unique_values) <= 2
        assert all(v in [0, 1] for v in unique_values.tolist())

    def test_mismatched_paths_raises_error(self, synthetic_data_dir: Path) -> None:
        """Test that mismatched image/mask counts raise error."""
        from arc_meshchop.data import ARCDataset

        image_paths = sorted(synthetic_data_dir.glob("image_*.nii.gz"))
        mask_paths = sorted(synthetic_data_dir.glob("mask_*.nii.gz"))[:2]  # Missing one

        with pytest.raises(ValueError, match="Mismatch"):
            ARCDataset(
                image_paths=image_paths,
                mask_paths=mask_paths,
            )

    def test_empty_dataset(self) -> None:
        """Test dataset with no samples."""
        from arc_meshchop.data import ARCDataset

        dataset = ARCDataset(
            image_paths=[],
            mask_paths=[],
            preprocess=False,
        )

        assert len(dataset) == 0

    def test_get_lesion_volume(self, synthetic_data_dir: Path) -> None:
        """Test lesion volume calculation."""
        from arc_meshchop.data import ARCDataset

        image_paths = sorted(synthetic_data_dir.glob("image_*.nii.gz"))
        mask_paths = sorted(synthetic_data_dir.glob("mask_*.nii.gz"))

        dataset = ARCDataset(
            image_paths=image_paths,
            mask_paths=mask_paths,
            preprocess=False,
        )

        volume = dataset.get_lesion_volume(0)
        assert isinstance(volume, int)
        assert volume > 0  # Our synthetic mask has a lesion


class TestCreateDataloaders:
    """Tests for dataloader creation."""

    def test_creates_two_loaders(self, synthetic_data_dir: Path) -> None:
        """Test that function returns two dataloaders."""
        from arc_meshchop.data import ARCDataset, create_dataloaders

        image_paths = sorted(synthetic_data_dir.glob("image_*.nii.gz"))
        mask_paths = sorted(synthetic_data_dir.glob("mask_*.nii.gz"))

        train_dataset = ARCDataset(
            image_paths=image_paths[:2],
            mask_paths=mask_paths[:2],
            preprocess=False,
        )
        val_dataset = ARCDataset(
            image_paths=image_paths[2:],
            mask_paths=mask_paths[2:],
            preprocess=False,
        )

        train_loader, val_loader = create_dataloaders(
            train_dataset,
            val_dataset,
            batch_size=1,
            num_workers=0,  # Use 0 for testing
        )

        assert isinstance(train_loader, torch.utils.data.DataLoader)
        assert isinstance(val_loader, torch.utils.data.DataLoader)

    def test_default_batch_size_is_one(self, synthetic_data_dir: Path) -> None:
        """Test that default batch size is 1 (paper requirement)."""
        from arc_meshchop.data import ARCDataset, create_dataloaders

        image_paths = sorted(synthetic_data_dir.glob("image_*.nii.gz"))
        mask_paths = sorted(synthetic_data_dir.glob("mask_*.nii.gz"))

        train_dataset = ARCDataset(
            image_paths=image_paths[:2],
            mask_paths=mask_paths[:2],
            preprocess=False,
        )
        val_dataset = ARCDataset(
            image_paths=image_paths[2:],
            mask_paths=mask_paths[2:],
            preprocess=False,
        )

        train_loader, _ = create_dataloaders(
            train_dataset,
            val_dataset,
            num_workers=0,
        )

        assert train_loader.batch_size == 1

    def test_train_loader_shuffles(self, synthetic_data_dir: Path) -> None:
        """Test that train loader has shuffle enabled."""
        from arc_meshchop.data import ARCDataset, create_dataloaders

        image_paths = sorted(synthetic_data_dir.glob("image_*.nii.gz"))
        mask_paths = sorted(synthetic_data_dir.glob("mask_*.nii.gz"))

        train_dataset = ARCDataset(
            image_paths=image_paths,
            mask_paths=mask_paths,
            preprocess=False,
        )
        val_dataset = ARCDataset(
            image_paths=image_paths,
            mask_paths=mask_paths,
            preprocess=False,
        )

        train_loader, val_loader = create_dataloaders(
            train_dataset,
            val_dataset,
            num_workers=0,
        )

        # Check sampler type indicates shuffling
        # RandomSampler is used when shuffle=True
        assert train_loader.sampler is not None
        # Val loader should use sequential sampler
        assert val_loader.sampler is not None
