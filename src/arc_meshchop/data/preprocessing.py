"""Preprocessing functions for MRI volumes.

Implements the preprocessing pipeline from the paper:
1. Resample to 256³ @ 1mm isotropic
2. Normalize intensity to 0-1 range

FROM PAPER (Section 2):
"Using mri_convert tool, we resampled all images to a uniform 1mm isotropic
resolution with dimensions of 256x256x256 voxels and applied a 0-1 rescaling
to normalize intensity values across images."

NOTE: This implementation uses SciPy as a Python-native equivalent to
FreeSurfer's mri_convert. The outputs should be numerically equivalent for:
- Linear interpolation (order=1) for images
- Nearest-neighbor (order=0) for masks

IMPORTANT: The paper does NOT mention skull stripping or MNI registration.
Only resampling to 256³@1mm and 0-1 normalization are required.
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

    Automatically converts to canonical RAS+ orientation for consistency.
    Paper uses mri_convert --conform which enforces RAS+ orientation.

    Args:
        path: Path to NIfTI file.

    Returns:
        Tuple of (data array in RAS+ orientation, affine matrix).
    """
    nii = nib.load(str(path))  # type: ignore[attr-defined]
    # Enforce canonical RAS+ orientation (BUG-001: some ARC samples are non-RAS)
    # This matches paper's mri_convert --conform preprocessing
    nii = nib.as_closest_canonical(nii)  # type: ignore[attr-defined]
    data = np.asarray(nii.get_fdata(), dtype=np.float32)  # type: ignore[attr-defined]
    affine = np.asarray(nii.affine, dtype=np.float64)  # type: ignore[attr-defined]
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
    nii = nib.Nifti1Image(data, affine)  # type: ignore[attr-defined]
    nib.save(nii, str(path))  # type: ignore[attr-defined]


def get_spacing(affine: npt.NDArray[np.float64]) -> tuple[float, float, float]:
    """Extract voxel spacing from affine matrix.

    Args:
        affine: 4x4 affine transformation matrix.

    Returns:
        Tuple of (spacing_x, spacing_y, spacing_z) in mm.
    """
    # Spacing is the norm of each column of the rotation/scaling part
    spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    return (float(spacing[0]), float(spacing[1]), float(spacing[2]))


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
    zoom_factors = [c / t for c, t in zip(current_spacing, target_spacing, strict=True)]

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
    # Work with a copy to avoid modifying the input
    current = data.copy()

    # Crop each dimension if needed (centered cropping)
    for dim in range(3):
        if current.shape[dim] > target_shape[dim]:
            start = (current.shape[dim] - target_shape[dim]) // 2
            slices = [slice(None)] * 3
            slices[dim] = slice(start, start + target_shape[dim])
            current = current[tuple(slices)]

    # Create result array with target shape (zero-padded)
    result = np.zeros(target_shape, dtype=data.dtype)

    # Calculate where to place the data (centered)
    d, h, w = current.shape
    td, th, tw = target_shape

    d_start = max(0, (td - d) // 2)
    h_start = max(0, (th - h) // 2)
    w_start = max(0, (tw - w) // 2)

    result[d_start : d_start + d, h_start : h_start + h, w_start : w_start + w] = current

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

    # Avoid division by zero for constant volumes
    if data_max - data_min < 1e-8:
        return np.zeros_like(data)

    normalized = (data - data_min) / (data_max - data_min)
    result: npt.NDArray[np.float32] = normalized.astype(np.float32)
    return result


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

        # Binarize mask: any non-zero value is lesion
        # NOTE: Using > 0 instead of > 0.5 because nibabel's get_fdata() can apply
        # implicit scaling that results in values < 0.5 for valid lesion voxels.
        # See BUG-001: sub-M2269_ses-767 has max scaled value of 0.017.
        mask_normalized = (mask_resampled > 0).astype(np.float32)

    return image_normalized, mask_normalized


def compute_lesion_volume(mask: npt.NDArray[np.float32]) -> int:
    """Compute lesion volume in voxels.

    Args:
        mask: Binary lesion mask.

    Returns:
        Number of lesion voxels.
    """
    # Use > 0 for consistency with binarization (see BUG-001)
    return int(np.sum(mask > 0))


def get_lesion_quintile(volume: int) -> str:
    """Determine lesion size quintile.

    Uses quintile boundaries from the paper:
    - Q1: 203 - 33,619 (small)
    - Q2: 33,619 - 67,891 (small-medium)
    - Q3: 67,891 - 128,314 (medium)
    - Q4: 128,314 - 363,885 (large)

    Args:
        volume: Lesion volume in voxels.

    Returns:
        Quintile label (Q1, Q2, Q3, or Q4).
    """
    from arc_meshchop.data.config import LESION_QUINTILES

    for quintile, (low, high) in LESION_QUINTILES.items():
        if low <= volume <= high:
            return str(quintile)

    # Edge cases
    if volume < 203:
        return "Q1"  # Very small
    return "Q4"  # Very large
