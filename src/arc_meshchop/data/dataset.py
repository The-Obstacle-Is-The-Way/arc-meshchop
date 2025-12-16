"""PyTorch Dataset for ARC stroke lesion segmentation.

Implements the data loading described in the paper:
- Full 256³ volumes (no patching)
- Batch size 1 (memory constraints)
- No data augmentation (not mentioned in paper)

FROM PAPER (Section 2):
"Unlike standard approaches that use 3D subvolume sampling or 2D slices,
we use whole-brain 256³ cubes for both training and inference."
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import torch
from torch.utils.data import Dataset

from arc_meshchop.data.preprocessing import preprocess_volume

if TYPE_CHECKING:
    import numpy.typing as npt


class ARCDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset for ARC stroke lesion segmentation.

    Loads preprocessed 256³ T2-weighted volumes with lesion masks.

    FROM PAPER:
    "Unlike standard approaches that use 3D subvolume sampling or 2D slices,
    we use whole-brain 256³ cubes for both training and inference."

    Attributes:
        image_paths: List of paths to T2-weighted images.
        mask_paths: List of paths to lesion masks.
        preprocess: Whether to apply preprocessing.
        target_shape: Target volume shape.
        target_spacing: Target voxel spacing.
        cache_dir: Optional directory for caching.
    """

    def __init__(
        self,
        image_paths: Sequence[Path | str],
        mask_paths: Sequence[Path | str],
        preprocess: bool = True,
        target_shape: tuple[int, int, int] = (256, 256, 256),
        target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        cache_dir: Path | str | None = None,
    ) -> None:
        """Initialize ARC dataset.

        Args:
            image_paths: List of paths to T2-weighted images.
            mask_paths: List of paths to lesion masks.
            preprocess: Whether to apply preprocessing (resampling, normalization).
            target_shape: Target volume shape.
            target_spacing: Target voxel spacing.
            cache_dir: Optional directory for caching preprocessed data.

        Raises:
            ValueError: If number of images and masks don't match.
        """
        if len(image_paths) != len(mask_paths):
            raise ValueError(f"Mismatch: {len(image_paths)} images vs {len(mask_paths)} masks")

        self.image_paths = [Path(p) for p in image_paths]
        self.mask_paths = [Path(p) for p in mask_paths]
        self.preprocess = preprocess
        self.target_shape = target_shape
        self.target_spacing = target_spacing
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Create cache directory if specified
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, idx: int) -> str:
        """Generate cache key incorporating path and preprocessing params.

        This prevents cache collisions when:
        - Same cache_dir is used with different preprocessing parameters
        - Same cache_dir is used with different data splits

        Args:
            idx: Sample index.

        Returns:
            Cache filename (e.g., "sample_0000_a1b2c3d4.npz").
        """
        # Include source path and preprocessing params in hash
        image_path = str(self.image_paths[idx])
        mask_path = str(self.mask_paths[idx])
        params = (
            f"{image_path}|{mask_path}|{self.target_shape}|{self.target_spacing}|{self.preprocess}"
        )
        param_hash = hashlib.md5(params.encode()).hexdigest()[:8]
        return f"sample_{idx:04d}_{param_hash}.npz"

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image, mask) tensors.
            - image: (1, 256, 256, 256) float32 tensor, values in [0, 1]
            - mask: (256, 256, 256) int64 tensor, values in {0, 1}
        """
        image, mask = self._load_sample(idx)

        # Add channel dimension to image: (D, H, W) -> (1, D, H, W)
        image_tensor = torch.from_numpy(image).unsqueeze(0)

        # Mask as long tensor for CrossEntropyLoss
        mask_tensor = torch.from_numpy(mask.astype(np.int64))

        return image_tensor, mask_tensor

    def _load_sample(
        self,
        idx: int,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Load and optionally preprocess a sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image, mask) numpy arrays.

        Raises:
            ValueError: If mask is None after loading.
        """
        # Check cache first
        if self.cache_dir:
            cache_path = self.cache_dir / self._cache_key(idx)
            if cache_path.exists():
                data = np.load(cache_path)
                return data["image"], data["mask"]

        # Load and preprocess
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        if self.preprocess:
            image, mask = preprocess_volume(
                image_path,
                mask_path,
                self.target_shape,
                self.target_spacing,
            )
        else:
            # Load raw (for pre-preprocessed data)
            import nibabel as nib

            img_nii = nib.load(str(image_path))  # type: ignore[attr-defined]
            mask_nii = nib.load(str(mask_path))  # type: ignore[attr-defined]
            image = np.asarray(img_nii.get_fdata(), dtype=np.float32)  # type: ignore[attr-defined]
            mask = np.asarray(mask_nii.get_fdata(), dtype=np.float32)  # type: ignore[attr-defined]

        if mask is None:
            raise ValueError(f"Mask is None for sample {idx}")

        # Cache if directory specified
        if self.cache_dir:
            cache_path = self.cache_dir / self._cache_key(idx)
            np.savez_compressed(cache_path, image=image, mask=mask)

        return image, mask

    def get_lesion_volume(self, idx: int) -> int:
        """Get lesion volume for a sample.

        Args:
            idx: Sample index.

        Returns:
            Lesion volume in voxels.
        """
        _, mask = self._load_sample(idx)
        # Use > 0 for consistency with preprocessing binarization (see BUG-001).
        return int(np.sum(mask > 0))


DataLoaderType: TypeAlias = torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]]


def create_dataloaders(
    train_dataset: ARCDataset,
    val_dataset: ARCDataset,
    batch_size: int = 1,
    num_workers: int = 4,
    pin_memory: bool | None = None,
) -> tuple[DataLoaderType, DataLoaderType]:
    """Create training and validation dataloaders.

    FROM PAPER:
    "...with a batch size of 1" (required for 256³ volumes)

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        batch_size: Batch size (default 1 for full volumes).
        num_workers: Number of data loading workers.
        pin_memory: Pin memory for faster GPU transfer.
            If None, auto-detect (True for CUDA, False for MPS/CPU).
            MPS generates warnings with pin_memory=True.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    # Auto-detect pin_memory: only beneficial for CUDA
    # MPS and CPU don't benefit and MPS may generate warnings
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_loader: DataLoaderType = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader: DataLoaderType = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader
