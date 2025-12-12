"""HuggingFace dataset loader for ARC.

Loads the ARC dataset from HuggingFace Hub and extracts paths for use with ARCDataset.

FROM PAPER:
"The Aphasia Recovery Cohort (ARC) is an open-source neuroimaging dataset
comprising T2-weighted MRI scans from 230 unique individuals with chronic stroke."
"""

from __future__ import annotations

import hashlib
import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datasets import Dataset

logger = logging.getLogger(__name__)


@dataclass
class ARCSample:
    """Single ARC sample metadata.

    Attributes:
        subject_id: BIDS subject ID (e.g., "sub-M2001").
        session_id: BIDS session ID (e.g., "ses-1").
        image_path: Path to T2-weighted image.
        mask_path: Path to lesion mask (None if no mask).
        lesion_volume: Lesion volume in voxels.
        acquisition_type: "space_2x" or "space_no_accel".
    """

    subject_id: str
    session_id: str
    image_path: Path
    mask_path: Path | None  # None if no lesion mask exists
    lesion_volume: int
    acquisition_type: str


@dataclass
class ARCDatasetInfo:
    """Complete ARC dataset information.

    Attributes:
        samples: List of ARCSample objects.
        image_paths: List of image paths (convenience accessor).
        mask_paths: List of mask paths (convenience accessor).
        lesion_volumes: List of lesion volumes.
        acquisition_types: List of acquisition types.
        subject_ids: List of subject IDs.
    """

    samples: list[ARCSample]

    @property
    def image_paths(self) -> list[Path]:
        """Get all image paths."""
        return [s.image_path for s in self.samples]

    @property
    def mask_paths(self) -> list[Path]:
        """Get all mask paths (excludes samples without masks)."""
        return [s.mask_path for s in self.samples if s.mask_path is not None]

    @property
    def lesion_volumes(self) -> list[int]:
        """Get all lesion volumes."""
        return [s.lesion_volume for s in self.samples]

    @property
    def acquisition_types(self) -> list[str]:
        """Get all acquisition types."""
        return [s.acquisition_type for s in self.samples]

    @property
    def subject_ids(self) -> list[str]:
        """Get all subject IDs."""
        return [s.subject_id for s in self.samples]

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)


# Alias for backwards compatibility
load_arc_samples = None  # Will be assigned to load_arc_from_huggingface


def load_arc_from_huggingface(
    repo_id: str = "hugging-science/arc-aphasia-bids",
    cache_dir: Path | str | None = None,
    include_space_2x: bool = True,
    include_space_no_accel: bool = True,
    exclude_turbo_spin_echo: bool = True,
    require_lesion_mask: bool = True,
    strict_t2w: bool = True,
    verify_counts: bool = True,
) -> ARCDatasetInfo:
    """Load ARC dataset from HuggingFace Hub.

    Downloads/caches the ARC dataset and extracts file paths
    for use with ARCDataset.

    FROM PAPER Section 2:
    "We utilized SPACE sequences with x2 in plane acceleration (115 scans)
    and without acceleration (109 scans), while excluding the turbo-spin
    echo T2-weighted sequences (5 scans)."

    Args:
        repo_id: HuggingFace repository ID.
        cache_dir: Optional cache directory for downloaded data.
        include_space_2x: Include SPACE with 2x acceleration.
        include_space_no_accel: Include SPACE without acceleration.
        exclude_turbo_spin_echo: Exclude turbo-spin echo sequences.
        require_lesion_mask: Only include samples with lesion masks.
        strict_t2w: If True, only use T2w (no FLAIR fallback). Default True for paper parity.
        verify_counts: If True, verify sample counts match paper (224 = 115 + 109).
            Raises ValueError if counts don't match.

    Returns:
        ARCDatasetInfo with all sample metadata and paths.

    Raises:
        ImportError: If datasets library not installed.
        ValueError: If no samples match filters, or if verify_counts=True and counts don't match.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "HuggingFace datasets library required. Install with: pip install datasets"
        ) from e

    logger.info("Loading ARC dataset from HuggingFace: %s", repo_id)

    # Load dataset (will cache automatically)
    ds = load_dataset(
        repo_id,
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    # Get the train split (ARC only has train split)
    dataset = ds["train"]

    logger.info("Loaded %d sessions from HuggingFace", len(dataset))

    # Filter and extract samples
    samples = _extract_samples(
        dataset,
        include_space_2x=include_space_2x,
        include_space_no_accel=include_space_no_accel,
        exclude_turbo_spin_echo=exclude_turbo_spin_echo,
        require_lesion_mask=require_lesion_mask,
        strict_t2w=strict_t2w,
    )

    if not samples:
        raise ValueError(
            "No samples match the specified filters. Check filter settings and dataset contents."
        )

    logger.info(
        "Filtered to %d samples (space_2x=%s, space_no_accel=%s, "
        "exclude_tse=%s, require_mask=%s, strict_t2w=%s)",
        len(samples),
        include_space_2x,
        include_space_no_accel,
        exclude_turbo_spin_echo,
        require_lesion_mask,
        strict_t2w,
    )

    # Verify sample counts match paper requirements
    if verify_counts:
        verify_sample_counts(samples)
        logger.info("Sample count verification passed: 224 samples (115+109)")

    return ARCDatasetInfo(samples=samples)


def _extract_samples(
    dataset: Dataset,
    include_space_2x: bool,
    include_space_no_accel: bool,
    exclude_turbo_spin_echo: bool,
    require_lesion_mask: bool,
    strict_t2w: bool = True,
) -> list[ARCSample]:
    """Extract and filter samples from HuggingFace dataset.

    Args:
        dataset: HuggingFace Dataset object.
        include_space_2x: Include SPACE 2x acceleration.
        include_space_no_accel: Include SPACE no acceleration.
        exclude_turbo_spin_echo: Exclude TSE sequences.
        require_lesion_mask: Require lesion mask.
        strict_t2w: If True, ONLY use T2w (no FLAIR fallback). Default True for paper parity.

    Returns:
        List of ARCSample objects.
    """
    samples = []

    for idx in range(len(dataset)):
        row = dataset[idx]

        # Check for lesion mask
        lesion = row.get("lesion")
        if require_lesion_mask and lesion is None:
            continue

        # Get T2-weighted image (primary modality for lesion segmentation)
        # FROM PAPER: Uses T2-weighted images exclusively
        t2w = row.get("t2w")
        if t2w is None:
            if strict_t2w:
                # Paper parity mode: Skip sessions without T2w
                # DO NOT fall back to FLAIR - paper only uses T2w
                continue
            else:
                # Non-parity mode: Allow FLAIR fallback for experimentation
                t2w = row.get("flair")
                if t2w is None:
                    continue

        # Extract path first (needed for acquisition type inference)
        image_path = _get_nifti_path(t2w)
        if image_path is None:
            continue

        # Determine acquisition type from BIDS filename (primary) or metadata (fallback)
        acq_type = _determine_acquisition_type(row, image_path)

        # Reject unknown acquisition types in parity mode
        if acq_type == "unknown":
            logger.warning(
                "Unknown acquisition type for sample %d (path: %s). "
                "Skipping - cannot verify paper compliance.",
                idx,
                image_path,
            )
            continue

        # Apply acquisition type filters
        if exclude_turbo_spin_echo and acq_type == "turbo_spin_echo":
            continue
        if not include_space_2x and acq_type == "space_2x":
            continue
        if not include_space_no_accel and acq_type == "space_no_accel":
            continue

        # Extract mask path
        mask_path = _get_nifti_path(lesion) if lesion else None
        if require_lesion_mask and mask_path is None:
            continue

        # Compute lesion volume if mask available
        lesion_volume = 0
        if lesion is not None:
            lesion_volume = _compute_lesion_volume_from_nifti(lesion)

        samples.append(
            ARCSample(
                subject_id=row.get("subject_id", f"sub-{idx:04d}"),
                session_id=row.get("session_id", "ses-1"),
                image_path=Path(image_path),
                mask_path=Path(mask_path) if mask_path else None,
                lesion_volume=lesion_volume,
                acquisition_type=acq_type,
            )
        )

    return samples


def _determine_acquisition_type(row: dict, file_path: str | None) -> str:
    """Determine acquisition type from BIDS filename or row metadata.

    FROM PAPER:
    - SPACE with 2x in-plane acceleration: 115 scans
    - SPACE without acceleration: 109 scans
    - Turbo-spin echo (excluded): 5 scans

    Acquisition type is inferred from BIDS filename conventions:
    - `acq-space2x` or `acq-SPACE2x` -> space_2x
    - `acq-space` or `acq-SPACE` (no acceleration) -> space_no_accel
    - `acq-tse` or `acq-TSE` -> turbo_spin_echo

    Args:
        row: Dataset row with metadata.
        file_path: Cached file path (BIDS naming) for acquisition inference.

    Returns:
        Acquisition type string.
    """
    # Primary method: Parse BIDS filename for acquisition entity
    # BIDS format: sub-XXX_ses-X_acq-XXXXX_T2w.nii.gz
    if file_path:
        file_path_lower = str(file_path).lower()

        # Look for acq-* entity in filename
        acq_match = re.search(r"acq-([a-z0-9]+)", file_path_lower)
        if acq_match:
            acq_value = acq_match.group(1)

            if "tse" in acq_value or "turbo" in acq_value:
                return "turbo_spin_echo"
            elif "space2x" in acq_value or "2x" in acq_value:
                return "space_2x"
            elif "space" in acq_value:
                return "space_no_accel"

    # Fallback: Check row metadata (less reliable)
    acq = row.get("acquisition", row.get("acq", ""))
    acq_str = str(acq).lower()

    if "tse" in acq_str or "turbo" in acq_str:
        return "turbo_spin_echo"
    elif "2x" in acq_str:
        return "space_2x"
    elif "space" in acq_str:
        return "space_no_accel"

    # UNKNOWN - Must be verified manually
    # In parity mode, this should raise an error
    return "unknown"


def _get_nifti_path(nifti_obj: object, cache_dir: Path | None = None) -> str | None:
    """Extract file path from HuggingFace Nifti object.

    HuggingFace Nifti() feature type can provide:
    - A file path (string)
    - NIfTI bytes directly (will be written to cache)
    - A nibabel image object

    Args:
        nifti_obj: HuggingFace Nifti object.
        cache_dir: Directory to cache bytes data (if None, uses temp dir).

    Returns:
        File path string or None.
    """
    if nifti_obj is None:
        return None

    # If it's already a string path
    if isinstance(nifti_obj, str):
        return nifti_obj

    # If it's a dict with path or bytes
    if isinstance(nifti_obj, dict):
        # Prefer path over bytes
        if nifti_obj.get("path"):
            return nifti_obj["path"]

        # Handle bytes - write to cache file
        if nifti_obj.get("bytes"):
            data = nifti_obj["bytes"]
            # Create deterministic filename from content hash
            content_hash = hashlib.md5(data).hexdigest()[:12]
            if cache_dir:
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path = cache_dir / f"nifti_{content_hash}.nii.gz"
            else:
                cache_path = Path(tempfile.gettempdir()) / f"arc_nifti_{content_hash}.nii.gz"

            if not cache_path.exists():
                cache_path.write_bytes(data)

            return str(cache_path)

    # If it's a nibabel image, try to get filename
    if hasattr(nifti_obj, "get_filename"):
        filename = nifti_obj.get_filename()
        if filename:
            return filename

    # If it has a path attribute
    if hasattr(nifti_obj, "path"):
        return nifti_obj.path

    return None


def verify_sample_counts(samples: list[ARCSample]) -> None:
    """Verify sample counts match paper requirements.

    FROM PAPER Section 2:
    - 115 SPACE with 2x acceleration
    - 109 SPACE without acceleration
    - 5 TSE excluded
    - Total: 224 usable scans

    Args:
        samples: List of extracted samples.

    Raises:
        ValueError: If counts don't match expected values.
    """
    space_2x_count = sum(1 for s in samples if s.acquisition_type == "space_2x")
    space_no_accel_count = sum(1 for s in samples if s.acquisition_type == "space_no_accel")
    tse_count = sum(1 for s in samples if s.acquisition_type == "turbo_spin_echo")
    unknown_count = sum(1 for s in samples if s.acquisition_type == "unknown")
    total = len(samples)

    logger.info(
        "Sample counts: space_2x=%d, space_no_accel=%d, tse=%d, unknown=%d, total=%d",
        space_2x_count,
        space_no_accel_count,
        tse_count,
        unknown_count,
        total,
    )

    # Verify against paper
    expected_space_2x = 115
    expected_space_no_accel = 109
    expected_total = 224

    warnings = []

    if space_2x_count != expected_space_2x:
        warnings.append(
            f"SPACE 2x count mismatch: got {space_2x_count}, expected {expected_space_2x}"
        )

    if space_no_accel_count != expected_space_no_accel:
        warnings.append(
            f"SPACE no-accel: got {space_no_accel_count}, expected {expected_space_no_accel}"
        )

    if total != expected_total:
        warnings.append(f"Total count mismatch: got {total}, expected {expected_total}")

    if tse_count > 0:
        warnings.append(f"TSE samples should be excluded: found {tse_count}")

    if unknown_count > 0:
        warnings.append(f"Unknown acquisition types found: {unknown_count} samples")

    if warnings:
        for w in warnings:
            logger.warning(w)
        raise ValueError(
            f"Sample count verification failed. "
            f"Expected 224 (115 space_2x + 109 space_no_accel), got {total}. "
            f"Issues: {'; '.join(warnings)}"
        )


def _compute_lesion_volume_from_nifti(nifti_obj: object) -> int:
    """Compute lesion volume from NIfTI object.

    Args:
        nifti_obj: HuggingFace Nifti object containing mask.

    Returns:
        Lesion volume in voxels.
    """
    try:
        import tempfile

        import nibabel as nib
        import numpy as np

        # Try to load data
        if hasattr(nifti_obj, "get_fdata"):
            # Already a nibabel image
            data = nifti_obj.get_fdata()
        elif isinstance(nifti_obj, dict) and "bytes" in nifti_obj:
            # Load from bytes - write to temp file (nibabel doesn't support BytesIO)
            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
                f.write(nifti_obj["bytes"])
                temp_path = f.name
            nii = nib.load(temp_path)
            data = nii.get_fdata()
        elif isinstance(nifti_obj, str):
            nii = nib.load(nifti_obj)
            data = nii.get_fdata()
        else:
            return 0

        return int(np.sum(data > 0.5))

    except Exception:
        return 0


def download_arc_to_local(
    output_dir: Path | str,
    repo_id: str = "hugging-science/arc-aphasia-bids",
    **filter_kwargs: Any,
) -> ARCDatasetInfo:
    """Download ARC dataset and save NIfTI files locally.

    For workflows that prefer local file access over HuggingFace streaming.

    Args:
        output_dir: Directory to save NIfTI files.
        repo_id: HuggingFace repository ID.
        **filter_kwargs: Filters passed to load_arc_from_huggingface.

    Returns:
        ARCDatasetInfo with local file paths.
    """
    import shutil

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)

    # Load from HuggingFace
    info = load_arc_from_huggingface(repo_id=repo_id, **filter_kwargs)

    logger.info("Downloading %d samples to %s", len(info), output_dir)

    # Save files locally
    local_samples = []
    for i, sample in enumerate(info.samples):
        # Copy/save image
        local_image = images_dir / f"{sample.subject_id}_{sample.session_id}_t2w.nii.gz"
        local_mask = masks_dir / f"{sample.subject_id}_{sample.session_id}_lesion.nii.gz"

        # Copy files
        if sample.image_path.exists():
            shutil.copy2(sample.image_path, local_image)

        # Handle optional mask
        saved_mask: Path | None = None
        if sample.mask_path is not None and sample.mask_path.exists():
            shutil.copy2(sample.mask_path, local_mask)
            saved_mask = local_mask

        local_samples.append(
            ARCSample(
                subject_id=sample.subject_id,
                session_id=sample.session_id,
                image_path=local_image,
                mask_path=saved_mask,
                lesion_volume=sample.lesion_volume,
                acquisition_type=sample.acquisition_type,
            )
        )

        if (i + 1) % 50 == 0:
            logger.info("Downloaded %d/%d samples", i + 1, len(info))

    logger.info("Download complete: %d samples saved to %s", len(local_samples), output_dir)

    return ARCDatasetInfo(samples=local_samples)


# Alias for backwards compatibility and convenience
load_arc_samples = load_arc_from_huggingface
