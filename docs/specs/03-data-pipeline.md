# Spec 03: Data Pipeline (ARC Dataset + Preprocessing)

> **Phase 3 of 7** — Data loading and preprocessing pipeline
>
> **Goal:** Implement complete data pipeline from ARC dataset to training-ready tensors.

---

## Overview

This spec covers:
- ARC dataset access via HuggingFace Hub
- Preprocessing: resampling to 256³ @ 1mm isotropic
- Intensity normalization (0-1 scaling)
- Nested cross-validation split generation
- Stratification by lesion size and acquisition type
- PyTorch Dataset and DataLoader implementation

---

## 1. Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA PIPELINE                                  │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────┐
│  HuggingFace │     │   Filter &   │     │  Preprocess  │     │  PyTorch │
│     Hub      │ ──▶ │   Validate   │ ──▶ │    256³     │ ──▶ │ Dataset  │
│   (ARC)      │     │  (SPACE seq) │     │  (1mm iso)   │     │          │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────┘
                            │                    │
                            │                    │
                            ▼                    ▼
                     ┌──────────────┐     ┌──────────────┐
                     │   Compute    │     │   Normalize  │
                     │   Lesion     │     │    0-1       │
                     │   Volumes    │     │   Scaling    │
                     └──────────────┘     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │   Generate   │
                     │   CV Splits  │
                     │  (Stratified)│
                     └──────────────┘
```

---

## 2. Dataset Specifications

### 2.1 ARC Dataset (FROM PAPER)

| Property | Value | Source |
|----------|-------|--------|
| Total subjects | 230 | Paper Section 2 |
| SPACE with 2x acceleration | 115 scans | Paper Section 2 |
| SPACE without acceleration | 109 scans | Paper Section 2 |
| Excluded (turbo-spin echo) | 5 scans | Paper Section 2 |
| **Final dataset size** | **224 scans** | Paper Section 2 |
| Lesion masks | Available | Paper Section 2 |

> **NOTE: Dataset Source**
> The paper references OpenNeuro as the data source: "data made available in open access via OpenNeuro".
> The HuggingFace dataset `hugging-science/arc-aphasia-bids` is a mirror of the OpenNeuro ARC release.
> Before training, verify the HF dataset contains the same 230 subjects with 224 SPACE scans and expert masks.

### 2.2 Preprocessing (FROM PAPER)

| Step | Specification | Tool (Paper) | Tool (This Implementation) |
|------|---------------|--------------|----------------------------|
| Resampling | 256×256×256 @ 1mm isotropic | `mri_convert` | SciPy `ndimage.zoom` |
| Normalization | 0-1 min-max scaling | Custom | Custom |
| Orientation | Standard RAS+ | `mri_convert --conform` | NiBabel (assumed RAS+) |

> **NOTE: Preprocessing Toolchain**
> The paper uses FreeSurfer's `mri_convert` for resampling. This implementation uses SciPy as a
> Python-native equivalent. The outputs should be numerically equivalent for:
> - Linear interpolation (order=1) for images
> - Nearest-neighbor (order=0) for masks
>
> **IMPORTANT**: The paper does NOT mention skull stripping or MNI registration.
> Only resampling to 256³@1mm and 0-1 normalization are required.

### 2.3 Data Splits (FROM PAPER)

| Structure | Details |
|-----------|---------|
| Outer folds | 3 (train/test) |
| Inner folds | 3 per outer (train/val) |
| Total configs | 9 training configurations |
| Stratification | Lesion size quintiles + acquisition type |

### 2.4 Lesion Size Quintiles (FROM PAPER)

| Quintile | Voxel Range (Paper Notation) | Description |
|----------|------------------------------|-------------|
| Q1 | (203, 33,619] | Small |
| Q2 | (33,619, 67,891] | Small-medium |
| Q3 | (67,891, 128,314] | Medium |
| Q4 | (128,314, 363,885] | Large |

> **NOTE: Quintile Boundary Notation**
> The paper uses interval notation with exclusive lower bounds: "(203, 33619]" means > 203 and ≤ 33,619.
> The implementation uses inclusive lower bounds for simplicity. Edge cases (exactly on boundary)
> are assigned to the lower quintile, which has minimal impact since exact boundary values are rare.

---

## 3. Implementation

### 3.1 Dataset Configuration

**File:** `src/arc_meshchop/data/config.py`

```python
"""Data pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class DataConfig:
    """Configuration for data pipeline.

    Attributes:
        data_root: Root directory for data storage.
        cache_dir: Directory for caching preprocessed data.
        hf_repo: HuggingFace Hub repository ID.
        target_shape: Target volume shape after resampling.
        target_spacing: Target voxel spacing in mm.
        num_outer_folds: Number of outer CV folds.
        num_inner_folds: Number of inner CV folds per outer.
        random_seed: Random seed for reproducibility.
    """

    data_root: Path = field(default_factory=lambda: Path("data"))
    cache_dir: Path = field(default_factory=lambda: Path("data/processed"))
    hf_repo: str = "hugging-science/arc-aphasia-bids"  # ARC dataset on HuggingFace Hub

    # Preprocessing targets (FROM PAPER)
    target_shape: tuple[int, int, int] = (256, 256, 256)
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Cross-validation structure (FROM PAPER)
    num_outer_folds: int = 3
    num_inner_folds: int = 3
    random_seed: int = 42

    # Acquisition types to include (FROM PAPER)
    include_space_2x: bool = True
    include_space_no_accel: bool = True
    exclude_turbo_spin_echo: bool = True

    def __post_init__(self) -> None:
        """Convert paths to Path objects."""
        self.data_root = Path(self.data_root)
        self.cache_dir = Path(self.cache_dir)


# Lesion size quintile boundaries (FROM PAPER Section 2)
LESION_QUINTILES = {
    "Q1": (203, 33_619),
    "Q2": (33_619, 67_891),
    "Q3": (67_891, 128_314),
    "Q4": (128_314, 363_885),
}

# Acquisition type identifiers
AcquisitionType = Literal["space_2x", "space_no_accel", "turbo_spin_echo"]
```

### 3.2 Preprocessing Functions

**File:** `src/arc_meshchop/data/preprocessing.py`

```python
"""Preprocessing functions for MRI volumes.

Implements the preprocessing pipeline from the paper:
1. Resample to 256³ @ 1mm isotropic
2. Normalize intensity to 0-1 range

FROM PAPER:
"Using mri_convert tool, we resampled all images to a uniform 1mm isotropic
resolution with dimensions of 256×256×256 voxels and applied a 0-1 rescaling
to normalize intensity values across images."
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
from scipy import ndimage

if TYPE_CHECKING:
    import numpy.typing as npt


def load_nifti(path: Path | str) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float64]]:
    """Load NIfTI file and return data + affine.

    Args:
        path: Path to NIfTI file.

    Returns:
        Tuple of (data array, affine matrix).
    """
    nii = nib.load(str(path))
    data = np.asarray(nii.get_fdata(), dtype=np.float32)
    affine = np.asarray(nii.affine, dtype=np.float64)
    return data, affine


def save_nifti(
    data: npt.NDArray[np.float32],
    affine: npt.NDArray[np.float64],
    path: Path | str,
) -> None:
    """Save data as NIfTI file.

    Args:
        data: Volume data array.
        affine: Affine transformation matrix.
        path: Output file path.
    """
    nii = nib.Nifti1Image(data, affine)
    nib.save(nii, str(path))


def get_spacing(affine: npt.NDArray[np.float64]) -> tuple[float, float, float]:
    """Extract voxel spacing from affine matrix.

    Args:
        affine: 4x4 affine transformation matrix.

    Returns:
        Tuple of (spacing_x, spacing_y, spacing_z) in mm.
    """
    # Spacing is the norm of each column of the rotation/scaling part
    spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    return tuple(spacing.tolist())


def resample_volume(
    data: npt.NDArray[np.float32],
    current_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
    target_shape: tuple[int, int, int],
    order: int = 1,
    is_mask: bool = False,
) -> npt.NDArray[np.float32]:
    """Resample volume to target spacing and shape.

    This implements the resampling described in the paper using scipy,
    as an alternative to mri_convert for Python-native processing.

    Args:
        data: Input volume data.
        current_spacing: Current voxel spacing (x, y, z) in mm.
        target_spacing: Target voxel spacing (x, y, z) in mm.
        target_shape: Target volume shape (D, H, W).
        order: Interpolation order (0=nearest, 1=linear, 3=cubic).
        is_mask: If True, use nearest-neighbor interpolation.

    Returns:
        Resampled volume data.
    """
    if is_mask:
        order = 0  # Nearest neighbor for masks

    # Calculate zoom factors
    zoom_factors = [
        c / t for c, t in zip(current_spacing, target_spacing, strict=True)
    ]

    # First resample to target spacing
    resampled = ndimage.zoom(data, zoom_factors, order=order)

    # Then crop or pad to target shape
    result = crop_or_pad(resampled, target_shape)

    return result.astype(np.float32)


def crop_or_pad(
    data: npt.NDArray[np.float32],
    target_shape: tuple[int, int, int],
) -> npt.NDArray[np.float32]:
    """Crop or pad volume to target shape (centered).

    Args:
        data: Input volume.
        target_shape: Target shape (D, H, W).

    Returns:
        Volume with target shape.
    """
    result = np.zeros(target_shape, dtype=data.dtype)

    # Calculate crop/pad for each dimension
    for dim in range(3):
        if data.shape[dim] > target_shape[dim]:
            # Crop (centered)
            start = (data.shape[dim] - target_shape[dim]) // 2
            slc = slice(start, start + target_shape[dim])
            if dim == 0:
                data = data[slc, :, :]
            elif dim == 1:
                data = data[:, slc, :]
            else:
                data = data[:, :, slc]

    # Copy data into result (handles padding automatically)
    d, h, w = data.shape
    td, th, tw = target_shape

    # Center the data in the result
    d_start = max(0, (td - d) // 2)
    h_start = max(0, (th - h) // 2)
    w_start = max(0, (tw - w) // 2)

    result[d_start:d_start+d, h_start:h_start+h, w_start:w_start+w] = data

    return result


def normalize_intensity(data: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Normalize intensity to 0-1 range using min-max scaling.

    FROM PAPER: "applied a 0-1 rescaling to normalize intensity values"

    Args:
        data: Input volume data.

    Returns:
        Normalized volume with values in [0, 1].
    """
    data_min = data.min()
    data_max = data.max()

    # Avoid division by zero
    if data_max - data_min < 1e-8:
        return np.zeros_like(data)

    normalized = (data - data_min) / (data_max - data_min)
    return normalized.astype(np.float32)


def preprocess_volume(
    image_path: Path | str,
    mask_path: Path | str | None = None,
    target_shape: tuple[int, int, int] = (256, 256, 256),
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32] | None]:
    """Complete preprocessing pipeline for a single volume.

    Implements the full preprocessing from the paper:
    1. Load NIfTI
    2. Resample to target spacing/shape
    3. Normalize intensity to 0-1

    Args:
        image_path: Path to T2-weighted image.
        mask_path: Optional path to lesion mask.
        target_shape: Target volume shape.
        target_spacing: Target voxel spacing.

    Returns:
        Tuple of (preprocessed_image, preprocessed_mask).
        Mask is None if mask_path is None.
    """
    # Load image
    image_data, image_affine = load_nifti(image_path)
    current_spacing = get_spacing(image_affine)

    # Resample image
    image_resampled = resample_volume(
        image_data,
        current_spacing,
        target_spacing,
        target_shape,
        order=1,  # Linear interpolation
        is_mask=False,
    )

    # Normalize intensity
    image_normalized = normalize_intensity(image_resampled)

    # Process mask if provided
    mask_normalized = None
    if mask_path is not None:
        mask_data, mask_affine = load_nifti(mask_path)
        mask_spacing = get_spacing(mask_affine)

        mask_resampled = resample_volume(
            mask_data,
            mask_spacing,
            target_spacing,
            target_shape,
            order=0,  # Nearest neighbor for masks
            is_mask=True,
        )

        # Binarize mask (in case of interpolation artifacts)
        mask_normalized = (mask_resampled > 0.5).astype(np.float32)

    return image_normalized, mask_normalized


def compute_lesion_volume(mask: npt.NDArray[np.float32]) -> int:
    """Compute lesion volume in voxels.

    Args:
        mask: Binary lesion mask.

    Returns:
        Number of lesion voxels.
    """
    return int(np.sum(mask > 0.5))


def get_lesion_quintile(volume: int) -> str:
    """Determine lesion size quintile.

    Uses quintile boundaries from the paper:
    - Q1: 203 - 33,619
    - Q2: 33,619 - 67,891
    - Q3: 67,891 - 128,314
    - Q4: 128,314 - 363,885

    Args:
        volume: Lesion volume in voxels.

    Returns:
        Quintile label (Q1, Q2, Q3, or Q4).
    """
    from arc_meshchop.data.config import LESION_QUINTILES

    for quintile, (low, high) in LESION_QUINTILES.items():
        if low <= volume <= high:
            return quintile

    # Edge cases
    if volume < 203:
        return "Q1"  # Very small
    return "Q4"  # Very large
```

### 3.3 Cross-Validation Split Generation

**File:** `src/arc_meshchop/data/splits.py`

```python
"""Cross-validation split generation with stratification.

Implements nested 3-fold cross-validation from the paper:
- 3 outer folds (train/test)
- 3 inner folds per outer (train/val)
- Stratified by lesion size quintile and acquisition type
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.model_selection import StratifiedKFold

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclass
class CVSplit:
    """Single cross-validation split.

    Attributes:
        train_indices: Indices for training set.
        val_indices: Indices for validation set.
        test_indices: Indices for test set (outer fold only).
    """

    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int] = field(default_factory=list)


@dataclass
class NestedCVSplits:
    """Complete nested cross-validation structure.

    Structure:
        outer_folds[i] = {
            "test_indices": [...],
            "inner_folds": [
                {"train_indices": [...], "val_indices": [...]},
                {"train_indices": [...], "val_indices": [...]},
                {"train_indices": [...], "val_indices": [...]},
            ]
        }
    """

    outer_folds: list[dict]
    random_seed: int
    num_outer_folds: int
    num_inner_folds: int
    stratification_labels: list[str]

    def get_split(
        self,
        outer_fold: int,
        inner_fold: int | None = None,
    ) -> CVSplit:
        """Get a specific split configuration.

        Args:
            outer_fold: Outer fold index (0-2).
            inner_fold: Inner fold index (0-2) or None for outer test only.

        Returns:
            CVSplit with train/val/test indices.
        """
        outer = self.outer_folds[outer_fold]
        test_indices = outer["test_indices"]

        if inner_fold is None:
            # Return outer fold test set only
            return CVSplit(
                train_indices=[],
                val_indices=[],
                test_indices=test_indices,
            )

        inner = outer["inner_folds"][inner_fold]
        return CVSplit(
            train_indices=inner["train_indices"],
            val_indices=inner["val_indices"],
            test_indices=test_indices,
        )

    def save(self, path: Path | str) -> None:
        """Save splits to JSON file.

        Args:
            path: Output file path.
        """
        data = {
            "random_seed": self.random_seed,
            "num_outer_folds": self.num_outer_folds,
            "num_inner_folds": self.num_inner_folds,
            "stratification_labels": self.stratification_labels,
            "outer_folds": self.outer_folds,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path | str) -> NestedCVSplits:
        """Load splits from JSON file.

        Args:
            path: Input file path.

        Returns:
            NestedCVSplits object.
        """
        data = json.loads(Path(path).read_text())
        return cls(
            outer_folds=data["outer_folds"],
            random_seed=data["random_seed"],
            num_outer_folds=data["num_outer_folds"],
            num_inner_folds=data["num_inner_folds"],
            stratification_labels=data["stratification_labels"],
        )


def create_stratification_labels(
    lesion_quintiles: list[str],
    acquisition_types: list[str],
) -> list[str]:
    """Create combined stratification labels.

    Combines lesion size quintile and acquisition type for stratification.

    Args:
        lesion_quintiles: List of quintile labels (Q1-Q4) per sample.
        acquisition_types: List of acquisition types per sample.

    Returns:
        List of combined labels for stratification.
    """
    return [
        f"{q}_{a}" for q, a in zip(lesion_quintiles, acquisition_types, strict=True)
    ]


def generate_nested_cv_splits(
    n_samples: int,
    stratification_labels: list[str],
    num_outer_folds: int = 3,
    num_inner_folds: int = 3,
    random_seed: int = 42,
) -> NestedCVSplits:
    """Generate nested cross-validation splits with stratification.

    Implements the nested CV structure from the paper:
    - 3 outer folds for train/test
    - 3 inner folds per outer for train/val
    - Stratified by combined lesion size + acquisition type

    Args:
        n_samples: Total number of samples.
        stratification_labels: Combined stratification labels.
        num_outer_folds: Number of outer folds.
        num_inner_folds: Number of inner folds per outer.
        random_seed: Random seed for reproducibility.

    Returns:
        NestedCVSplits object with complete split structure.
    """
    indices = np.arange(n_samples)
    labels = np.array(stratification_labels)

    outer_splitter = StratifiedKFold(
        n_splits=num_outer_folds,
        shuffle=True,
        random_state=random_seed,
    )

    outer_folds = []

    for outer_train_idx, test_idx in outer_splitter.split(indices, labels):
        # Inner splits on outer training data
        inner_indices = indices[outer_train_idx]
        inner_labels = labels[outer_train_idx]

        inner_splitter = StratifiedKFold(
            n_splits=num_inner_folds,
            shuffle=True,
            random_state=random_seed,
        )

        inner_folds = []
        for train_idx, val_idx in inner_splitter.split(inner_indices, inner_labels):
            inner_folds.append({
                "train_indices": inner_indices[train_idx].tolist(),
                "val_indices": inner_indices[val_idx].tolist(),
            })

        outer_folds.append({
            "test_indices": indices[test_idx].tolist(),
            "inner_folds": inner_folds,
        })

    return NestedCVSplits(
        outer_folds=outer_folds,
        random_seed=random_seed,
        num_outer_folds=num_outer_folds,
        num_inner_folds=num_inner_folds,
        stratification_labels=stratification_labels,
    )
```

### 3.4 PyTorch Dataset

**File:** `src/arc_meshchop/data/dataset.py`

```python
"""PyTorch Dataset for ARC stroke lesion segmentation.

Implements the data loading described in the paper:
- Full 256³ volumes (no patching)
- Batch size 1 (memory constraints)
- No data augmentation (not mentioned in paper)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

from arc_meshchop.data.preprocessing import preprocess_volume

if TYPE_CHECKING:
    import numpy.typing as npt


class ARCDataset(Dataset):
    """PyTorch Dataset for ARC stroke lesion segmentation.

    Loads preprocessed 256³ T2-weighted volumes with lesion masks.

    FROM PAPER:
    "Unlike standard approaches that use 3D subvolume sampling or 2D slices,
    we use whole-brain 256³ cubes for both training and inference."
    """

    def __init__(
        self,
        image_paths: list[Path | str],
        mask_paths: list[Path | str],
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
        """
        if len(image_paths) != len(mask_paths):
            raise ValueError(
                f"Mismatch: {len(image_paths)} images vs {len(mask_paths)} masks"
            )

        self.image_paths = [Path(p) for p in image_paths]
        self.mask_paths = [Path(p) for p in mask_paths]
        self.preprocess = preprocess
        self.target_shape = target_shape
        self.target_spacing = target_spacing
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Create cache directory if specified
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

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
        """
        # Check cache first
        if self.cache_dir:
            cache_path = self.cache_dir / f"sample_{idx:04d}.npz"
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

            image = np.asarray(nib.load(str(image_path)).get_fdata(), dtype=np.float32)
            mask = np.asarray(nib.load(str(mask_path)).get_fdata(), dtype=np.float32)

        if mask is None:
            raise ValueError(f"Mask is None for sample {idx}")

        # Cache if directory specified
        if self.cache_dir:
            cache_path = self.cache_dir / f"sample_{idx:04d}.npz"
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
        return int(np.sum(mask > 0.5))


def create_dataloaders(
    train_dataset: ARCDataset,
    val_dataset: ARCDataset,
    batch_size: int = 1,
    num_workers: int = 4,
    pin_memory: bool | None = None,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
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

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader
```

### 3.5 Module `__init__.py`

**File:** `src/arc_meshchop/data/__init__.py`

```python
"""Data pipeline for ARC stroke lesion segmentation."""

from arc_meshchop.data.config import (
    AcquisitionType,
    DataConfig,
    LESION_QUINTILES,
)
from arc_meshchop.data.dataset import (
    ARCDataset,
    create_dataloaders,
)
from arc_meshchop.data.preprocessing import (
    compute_lesion_volume,
    crop_or_pad,
    get_lesion_quintile,
    get_spacing,
    load_nifti,
    normalize_intensity,
    preprocess_volume,
    resample_volume,
    save_nifti,
)
from arc_meshchop.data.splits import (
    CVSplit,
    NestedCVSplits,
    create_stratification_labels,
    generate_nested_cv_splits,
)

__all__ = [
    "AcquisitionType",
    "ARCDataset",
    "CVSplit",
    "DataConfig",
    "LESION_QUINTILES",
    "NestedCVSplits",
    "compute_lesion_volume",
    "create_dataloaders",
    "create_stratification_labels",
    "crop_or_pad",
    "generate_nested_cv_splits",
    "get_lesion_quintile",
    "get_spacing",
    "load_nifti",
    "normalize_intensity",
    "preprocess_volume",
    "resample_volume",
    "save_nifti",
]
```

---

## 4. Tests

### 4.1 Preprocessing Tests

**File:** `tests/test_data/test_preprocessing.py`

```python
"""Tests for preprocessing functions."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from arc_meshchop.data.preprocessing import (
    compute_lesion_volume,
    crop_or_pad,
    get_lesion_quintile,
    normalize_intensity,
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


class TestCropOrPad:
    """Tests for crop_or_pad function."""

    def test_no_change_needed(self) -> None:
        """Test when shape already matches target."""
        data = np.ones((256, 256, 256), dtype=np.float32)
        result = crop_or_pad(data, (256, 256, 256))

        assert result.shape == (256, 256, 256)

    def test_padding(self) -> None:
        """Test padding smaller volume."""
        data = np.ones((100, 100, 100), dtype=np.float32)
        result = crop_or_pad(data, (256, 256, 256))

        assert result.shape == (256, 256, 256)
        # Check data is centered (some zeros on edges)
        assert result[0, 0, 0] == 0.0  # Padded corner
        assert result[128, 128, 128] == 1.0  # Center (has data)

    def test_cropping(self) -> None:
        """Test cropping larger volume."""
        data = np.ones((300, 300, 300), dtype=np.float32)
        result = crop_or_pad(data, (256, 256, 256))

        assert result.shape == (256, 256, 256)


class TestResampleVolume:
    """Tests for volume resampling."""

    def test_output_shape(self) -> None:
        """Verify output has target shape."""
        data = np.random.rand(100, 100, 100).astype(np.float32)
        result = resample_volume(
            data,
            current_spacing=(2.0, 2.0, 2.0),
            target_spacing=(1.0, 1.0, 1.0),
            target_shape=(256, 256, 256),
        )

        assert result.shape == (256, 256, 256)

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


class TestLesionQuintile:
    """Tests for lesion quintile assignment."""

    @pytest.mark.parametrize(
        "volume,expected",
        [
            (500, "Q1"),       # Small
            (50_000, "Q2"),    # Small-medium
            (100_000, "Q3"),   # Medium
            (200_000, "Q4"),   # Large
            (100, "Q1"),       # Very small (edge case)
            (400_000, "Q4"),   # Very large (edge case)
        ],
    )
    def test_quintile_assignment(self, volume: int, expected: str) -> None:
        """Test quintile assignment for various volumes."""
        result = get_lesion_quintile(volume)
        assert result == expected
```

### 4.2 Dataset Tests

**File:** `tests/test_data/test_dataset.py`

```python
"""Tests for PyTorch dataset."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

import nibabel as nib
import torch


def create_synthetic_nifti(
    path: Path,
    shape: tuple[int, int, int] = (64, 64, 64),
    is_mask: bool = False,
) -> None:
    """Create a synthetic NIfTI file for testing."""
    if is_mask:
        data = np.zeros(shape, dtype=np.float32)
        # Add a small lesion
        c = shape[0] // 2
        s = shape[0] // 8
        data[c-s:c+s, c-s:c+s, c-s:c+s] = 1.0
    else:
        data = np.random.rand(*shape).astype(np.float32) * 1000

    # Create identity affine (1mm spacing)
    affine = np.eye(4)
    nii = nib.Nifti1Image(data, affine)
    nib.save(nii, str(path))


@pytest.fixture
def synthetic_data_dir(tmp_path: Path) -> Path:
    """Create synthetic dataset for testing."""
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

        image_paths = list(synthetic_data_dir.glob("image_*.nii.gz"))
        mask_paths = list(synthetic_data_dir.glob("mask_*.nii.gz"))

        dataset = ARCDataset(
            image_paths=sorted(image_paths),
            mask_paths=sorted(mask_paths),
            preprocess=False,  # Skip preprocessing for faster tests
        )

        assert len(dataset) == 3

    def test_dataset_output_types(self, synthetic_data_dir: Path) -> None:
        """Test dataset returns correct tensor types."""
        from arc_meshchop.data import ARCDataset

        image_paths = list(synthetic_data_dir.glob("image_*.nii.gz"))
        mask_paths = list(synthetic_data_dir.glob("mask_*.nii.gz"))

        dataset = ARCDataset(
            image_paths=sorted(image_paths),
            mask_paths=sorted(mask_paths),
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

        image_paths = list(synthetic_data_dir.glob("image_*.nii.gz"))
        mask_paths = list(synthetic_data_dir.glob("mask_*.nii.gz"))

        dataset = ARCDataset(
            image_paths=sorted(image_paths),
            mask_paths=sorted(mask_paths),
            preprocess=False,
        )

        image, mask = dataset[0]

        # Image should have channel dimension
        assert image.dim() == 4  # (C, D, H, W)
        assert image.shape[0] == 1  # Single channel

        # Mask should not have channel dimension
        assert mask.dim() == 3  # (D, H, W)

    def test_mismatched_paths_raises_error(self, synthetic_data_dir: Path) -> None:
        """Test that mismatched image/mask counts raise error."""
        from arc_meshchop.data import ARCDataset

        image_paths = list(synthetic_data_dir.glob("image_*.nii.gz"))
        mask_paths = list(synthetic_data_dir.glob("mask_*.nii.gz"))[:2]  # Missing one

        with pytest.raises(ValueError, match="Mismatch"):
            ARCDataset(
                image_paths=sorted(image_paths),
                mask_paths=sorted(mask_paths),
            )
```

### 4.3 Cross-Validation Tests

**File:** `tests/test_data/test_splits.py`

```python
"""Tests for cross-validation split generation."""

import pytest
import tempfile
from pathlib import Path

from arc_meshchop.data.splits import (
    create_stratification_labels,
    generate_nested_cv_splits,
    NestedCVSplits,
)


class TestStratificationLabels:
    """Tests for stratification label creation."""

    def test_creates_combined_labels(self) -> None:
        """Test combined label format."""
        quintiles = ["Q1", "Q2", "Q3"]
        acq_types = ["space_2x", "space_no_accel", "space_2x"]

        labels = create_stratification_labels(quintiles, acq_types)

        assert labels == ["Q1_space_2x", "Q2_space_no_accel", "Q3_space_2x"]


class TestNestedCVSplits:
    """Tests for nested cross-validation split generation."""

    def test_outer_fold_count(self) -> None:
        """Verify correct number of outer folds."""
        labels = ["Q1_a", "Q2_a", "Q1_b", "Q2_b"] * 10  # 40 samples

        splits = generate_nested_cv_splits(
            n_samples=len(labels),
            stratification_labels=labels,
            num_outer_folds=3,
            num_inner_folds=3,
        )

        assert len(splits.outer_folds) == 3

    def test_inner_fold_count(self) -> None:
        """Verify correct number of inner folds per outer."""
        labels = ["Q1_a", "Q2_a", "Q1_b", "Q2_b"] * 10

        splits = generate_nested_cv_splits(
            n_samples=len(labels),
            stratification_labels=labels,
            num_outer_folds=3,
            num_inner_folds=3,
        )

        for outer in splits.outer_folds:
            assert len(outer["inner_folds"]) == 3

    def test_no_data_leakage(self) -> None:
        """Verify test indices don't appear in train/val."""
        labels = ["Q1_a", "Q2_a", "Q1_b", "Q2_b"] * 10

        splits = generate_nested_cv_splits(
            n_samples=len(labels),
            stratification_labels=labels,
            num_outer_folds=3,
            num_inner_folds=3,
        )

        for outer in splits.outer_folds:
            test_set = set(outer["test_indices"])

            for inner in outer["inner_folds"]:
                train_set = set(inner["train_indices"])
                val_set = set(inner["val_indices"])

                # No overlap between test and train/val
                assert len(test_set & train_set) == 0
                assert len(test_set & val_set) == 0

                # No overlap between train and val
                assert len(train_set & val_set) == 0

    def test_all_indices_covered(self) -> None:
        """Verify all indices appear exactly once in outer test folds."""
        n_samples = 40
        labels = ["Q1_a", "Q2_a", "Q1_b", "Q2_b"] * 10

        splits = generate_nested_cv_splits(
            n_samples=n_samples,
            stratification_labels=labels,
            num_outer_folds=3,
            num_inner_folds=3,
        )

        all_test_indices = []
        for outer in splits.outer_folds:
            all_test_indices.extend(outer["test_indices"])

        # Each index appears exactly once across test folds
        assert sorted(all_test_indices) == list(range(n_samples))

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Test saving and loading splits."""
        labels = ["Q1_a", "Q2_a", "Q1_b", "Q2_b"] * 10

        splits = generate_nested_cv_splits(
            n_samples=len(labels),
            stratification_labels=labels,
            num_outer_folds=3,
            num_inner_folds=3,
            random_seed=42,
        )

        # Save
        save_path = tmp_path / "splits.json"
        splits.save(save_path)

        # Load
        loaded = NestedCVSplits.load(save_path)

        assert loaded.random_seed == splits.random_seed
        assert loaded.num_outer_folds == splits.num_outer_folds
        assert loaded.outer_folds == splits.outer_folds

    def test_reproducibility(self) -> None:
        """Test that same seed produces same splits."""
        labels = ["Q1_a", "Q2_a", "Q1_b", "Q2_b"] * 10

        splits1 = generate_nested_cv_splits(
            n_samples=len(labels),
            stratification_labels=labels,
            random_seed=42,
        )

        splits2 = generate_nested_cv_splits(
            n_samples=len(labels),
            stratification_labels=labels,
            random_seed=42,
        )

        assert splits1.outer_folds == splits2.outer_folds
```

---

## 5. Implementation Checklist

### Phase 3.1: Configuration

- [ ] Create `src/arc_meshchop/data/config.py`
- [ ] Define `DataConfig` dataclass
- [ ] Define lesion quintile boundaries

### Phase 3.2: Preprocessing

- [ ] Create `src/arc_meshchop/data/preprocessing.py`
- [ ] Implement `load_nifti` / `save_nifti`
- [ ] Implement `resample_volume`
- [ ] Implement `normalize_intensity`
- [ ] Implement `preprocess_volume`

### Phase 3.3: Cross-Validation

- [ ] Create `src/arc_meshchop/data/splits.py`
- [ ] Implement `generate_nested_cv_splits`
- [ ] Implement save/load functionality

### Phase 3.4: Dataset

- [ ] Create `src/arc_meshchop/data/dataset.py`
- [ ] Implement `ARCDataset`
- [ ] Implement `create_dataloaders`

### Phase 3.5: Tests

- [ ] Create preprocessing tests
- [ ] Create dataset tests
- [ ] Create split tests
- [ ] All tests pass

---

## 6. Verification Commands

```bash
# Run data pipeline tests
uv run pytest tests/test_data/ -v

# Test preprocessing specifically
uv run pytest tests/test_data/test_preprocessing.py -v

# Test with synthetic data
uv run python -c "
from arc_meshchop.data import ARCDataset, generate_nested_cv_splits

# Verify CV split generation
labels = ['Q1_a', 'Q2_a', 'Q1_b', 'Q2_b'] * 56  # 224 samples (paper size)
splits = generate_nested_cv_splits(len(labels), labels)
print(f'Generated {len(splits.outer_folds)} outer folds')
print(f'Each outer fold has {len(splits.outer_folds[0][\"inner_folds\"])} inner folds')
"
```

---

## 7. References

- Paper Section 2: Dataset and preprocessing details
- Research docs: `docs/research/02-dataset-preprocessing.md`

### ARC Dataset Access

The ARC dataset is accessed via **HuggingFace datasets** (native packages):

```python
# HuggingFace datasets for ARC data access
# NOTE: neuroimaging-go-brrrr git dependency removed due to submodule issues
# We implement our own data loading from hugging-science/arc-aphasia-bids

from datasets import load_dataset

# Load ARC dataset from HuggingFace Hub
dataset = load_dataset("hugging-science/arc-aphasia-bids")
```

HuggingFace Dataset: [hugging-science/arc-aphasia-bids](https://huggingface.co/datasets/hugging-science/arc-aphasia-bids)
