"""HuggingFace dataset loader for ARC.

Loads the ARC dataset from HuggingFace Hub and extracts paths for use with ARCDataset.
Uses bids_hub (neuroimaging-go-brrrr) for validation constants.

FROM PAPER:
"The Aphasia Recovery Cohort (ARC) is an open-source neuroimaging dataset
comprising T2-weighted MRI scans from 230 unique individuals with chronic stroke."
"""

from __future__ import annotations

import hashlib
import logging
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

# Import validation constants from bids_hub (full dataset counts)
# NOTE: These are for the FULL ARC dataset (230 subjects, 902 sessions).
# Paper training subset is 224 samples (verified separately in verify_sample_counts).
from bids_hub.validation.arc import ARC_VALIDATION_CONFIG

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
    def mask_paths(self) -> list[Path | None]:
        """Get all mask paths, aligned 1:1 with image_paths.

        Returns None for samples without lesion masks.
        Use samples_with_masks() to get only samples that have masks.
        """
        return [s.mask_path for s in self.samples]

    @property
    def samples_with_masks(self) -> list[ARCSample]:
        """Get only samples that have lesion masks.

        Use this when you need filtered access. Note: indices will NOT
        align with image_paths, lesion_volumes, etc.
        """
        return [s for s in self.samples if s.mask_path is not None]

    @property
    def num_with_masks(self) -> int:
        """Count of samples that have lesion masks."""
        return sum(1 for s in self.samples if s.mask_path is not None)

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


_DATASET_INFO_REQUIRED_KEYS: tuple[str, ...] = (
    "image_paths",
    "mask_paths",
    "lesion_volumes",
    "acquisition_types",
    "subject_ids",
)


def parse_dataset_info(
    dataset_info: Mapping[str, Any],
    *,
    context: str,
) -> tuple[list[str], list[str | None], list[int], list[str], list[str]]:
    """Validate dataset_info schema and return typed lists.

    This is the SSOT for `dataset_info.json` validation across the codebase.
    """
    missing = [k for k in _DATASET_INFO_REQUIRED_KEYS if k not in dataset_info]
    if missing:
        raise ValueError(f"{context}: dataset_info missing keys: {missing}")

    def _require_list(value: Any, *, key: str) -> list[Any]:
        if not isinstance(value, list):
            raise ValueError(f"{context}: dataset_info[{key}] must be a list")
        return value

    image_paths_any = _require_list(dataset_info["image_paths"], key="image_paths")
    if not all(isinstance(p, str) for p in image_paths_any):
        raise ValueError(f"{context}: dataset_info[image_paths] must be list[str]")
    image_paths = cast(list[str], image_paths_any)

    mask_paths_any = _require_list(dataset_info["mask_paths"], key="mask_paths")
    if not all((m is None) or isinstance(m, str) for m in mask_paths_any):
        raise ValueError(f"{context}: dataset_info[mask_paths] must be list[str | None]")
    mask_paths = cast(list[str | None], mask_paths_any)

    lesion_volumes_any = _require_list(dataset_info["lesion_volumes"], key="lesion_volumes")
    if not all(isinstance(v, int) for v in lesion_volumes_any):
        raise ValueError(f"{context}: dataset_info[lesion_volumes] must be list[int]")
    lesion_volumes = cast(list[int], lesion_volumes_any)

    acquisition_types_any = _require_list(
        dataset_info["acquisition_types"], key="acquisition_types"
    )
    if not all(isinstance(a, str) for a in acquisition_types_any):
        raise ValueError(f"{context}: dataset_info[acquisition_types] must be list[str]")
    acquisition_types = cast(list[str], acquisition_types_any)

    subject_ids_any = _require_list(dataset_info["subject_ids"], key="subject_ids")
    if not all(isinstance(s, str) for s in subject_ids_any):
        raise ValueError(f"{context}: dataset_info[subject_ids] must be list[str]")
    subject_ids = cast(list[str], subject_ids_any)

    n = len(image_paths)
    if not (
        len(mask_paths) == n
        and len(lesion_volumes) == n
        and len(acquisition_types) == n
        and len(subject_ids) == n
    ):
        raise ValueError(
            f"{context}: dataset_info list lengths must match. "
            f"Got image_paths={len(image_paths)}, mask_paths={len(mask_paths)}, "
            f"lesion_volumes={len(lesion_volumes)}, acquisition_types={len(acquisition_types)}, "
            f"subject_ids={len(subject_ids)}."
        )

    if "num_samples" in dataset_info:
        num_samples_any = dataset_info["num_samples"]
        if not isinstance(num_samples_any, int):
            raise ValueError(f"{context}: dataset_info[num_samples] must be an int")
        num_samples = num_samples_any
        if num_samples != n:
            raise ValueError(
                f"{context}: dataset_info num_samples mismatch: "
                f"num_samples={num_samples} but len(image_paths)={n}."
            )

    return image_paths, mask_paths, lesion_volumes, acquisition_types, subject_ids


def validate_masks_present(
    mask_paths: Sequence[str | None],
    *,
    context: str,
) -> list[str]:
    """Validate all masks are present and return as non-None list.

    Args:
        mask_paths: List that may contain None values
        context: Description of what operation needs masks (for error message)

    Returns:
        Same list, but typed as list[str] (no None)

    Raises:
        ValueError: If any masks are None, with actionable error message
    """
    none_indices = [i for i, m in enumerate(mask_paths) if m is None]
    if none_indices:
        preview = none_indices[:5]
        suffix = f"... and {len(none_indices) - 5} more" if len(none_indices) > 5 else ""
        raise ValueError(
            f"{context} requires lesion masks, but {len(none_indices)} samples have None masks "
            f"(indices: {preview}{suffix}). "
            "Re-download dataset with 'arc-meshchop download' (default requires masks)."
        )
    return [cast(str, m) for m in mask_paths]


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
    paper_parity: bool = False,
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
        paper_parity: If True, use strict mode to match reproducible paper cohort (223 samples).
            Excludes TSE and enforces strict counts.

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

    # Log bids_hub expected counts for context (full dataset, not paper subset)
    logger.info(
        "bids_hub ARC_VALIDATION_CONFIG: %d subjects, %d sessions, %d t2w",
        ARC_VALIDATION_CONFIG.expected_counts["subjects"],
        ARC_VALIDATION_CONFIG.expected_counts["sessions"],
        ARC_VALIDATION_CONFIG.expected_counts["t2w"],
    )

    # Derive NIfTI cache directory from cache_dir
    # HuggingFace uses cache_dir for parquet files; we use a subdirectory for NIfTI extraction
    nifti_cache_dir = Path(cache_dir) / "nifti_cache" if cache_dir else None

    # Filter and extract samples
    samples = _extract_samples(
        dataset,
        include_space_2x=include_space_2x,
        include_space_no_accel=include_space_no_accel,
        exclude_turbo_spin_echo=exclude_turbo_spin_echo,
        require_lesion_mask=require_lesion_mask,
        strict_t2w=strict_t2w,
        cache_dir=nifti_cache_dir,
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
        # Pass paper_parity flag to verification logic
        # If paper_parity=True, strict is implied
        strict_verification = paper_parity
        verify_sample_counts(samples, strict=strict_verification, paper_parity=paper_parity)
        logger.info("Sample count verification passed: %d samples", len(samples))

    return ARCDatasetInfo(samples=samples)


def _extract_samples(
    dataset: Dataset,
    include_space_2x: bool,
    include_space_no_accel: bool,
    exclude_turbo_spin_echo: bool,
    require_lesion_mask: bool,
    strict_t2w: bool = True,
    cache_dir: Path | None = None,
) -> list[ARCSample]:
    """Extract and filter samples from HuggingFace dataset.

    Args:
        dataset: HuggingFace Dataset object.
        include_space_2x: Include SPACE 2x acceleration.
        include_space_no_accel: Include SPACE no acceleration.
        exclude_turbo_spin_echo: Exclude TSE sequences.
        require_lesion_mask: Require lesion mask.
        strict_t2w: If True, ONLY use T2w (no FLAIR fallback). Default True for paper parity.
        cache_dir: Directory to cache NIfTI files.

    Returns:
        List of ARCSample objects.
    """
    samples = []
    unknown_acq_count = 0

    # Determine cache directory
    if cache_dir is None:
        cache_dir = Path(tempfile.gettempdir()) / "arc_nifti_cache"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(len(dataset)):
        row = dataset[idx]

        # Get subject and session IDs
        subject_id = row.get("subject_id", f"sub-{idx:04d}")
        session_id = row.get("session_id", "ses-1")

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

        # Create identifier for cache filename
        image_identifier = f"{subject_id}_{session_id}_t2w"

        # Extract path (will cache Nifti1ImageWrapper to disk if needed)
        image_path = _get_nifti_path(t2w, cache_dir=cache_dir, identifier=image_identifier)
        if image_path is None:
            logger.debug("Could not extract path for sample %d (%s)", idx, subject_id)
            continue

        # Determine acquisition type from BIDS filename (primary) or metadata (fallback)
        acq_type = _determine_acquisition_type(row, image_path)

        if acq_type == "unknown":
            # Just count them, but don't reclassify.
            # Upstream fix provides t2w_acquisition, so unknowns should be rare/non-existent.
            unknown_acq_count += 1

        # Apply acquisition type filters (only effective when metadata available)
        if exclude_turbo_spin_echo and acq_type == "turbo_spin_echo":
            continue
        if not include_space_2x and acq_type == "space_2x":
            continue
        if not include_space_no_accel and acq_type == "space_no_accel":
            continue

        # Extract mask path
        mask_identifier = f"{subject_id}_{session_id}_lesion"
        mask_path = (
            _get_nifti_path(lesion, cache_dir=cache_dir, identifier=mask_identifier)
            if lesion
            else None
        )
        if require_lesion_mask and mask_path is None:
            continue

        # Compute lesion volume if mask available
        lesion_volume = 0
        if lesion is not None:
            lesion_volume = _compute_lesion_volume_from_nifti(lesion)

        samples.append(
            ARCSample(
                subject_id=subject_id,
                session_id=session_id,
                image_path=Path(image_path),
                mask_path=Path(mask_path) if mask_path else None,
                lesion_volume=lesion_volume,
                acquisition_type=acq_type,
            )
        )

        # Log progress every 50 samples
        if (idx + 1) % 50 == 0:
            logger.info(
                "Processed %d/%d sessions, extracted %d samples",
                idx + 1,
                len(dataset),
                len(samples),
            )

    if unknown_acq_count > 0:
        logger.warning(
            "Found %d samples with unknown acquisition type. "
            "This implies missing metadata in the source dataset.",
            unknown_acq_count,
        )

    return samples


def _determine_acquisition_type(row: dict, file_path: str | None) -> str:
    """Determine acquisition type from row metadata or BIDS filename.

    Priority:
    1. t2w_acquisition column (upstream fix - now available)
    2. BIDS filename parsing (fallback for local data)
    3. Legacy acquisition/acq columns (backwards compat)

    FROM PAPER:
    - SPACE with 2x in-plane acceleration: 115 scans
    - SPACE without acceleration: 109 scans
    - Turbo-spin echo (excluded): 5 scans

    Args:
        row: Dataset row with metadata.
        file_path: Cached file path (BIDS naming) for acquisition inference.

    Returns:
        Acquisition type string.
    """
    # PRIMARY: Use t2w_acquisition column from upstream fix
    t2w_acq = row.get("t2w_acquisition")
    if t2w_acq and t2w_acq in ("space_2x", "space_no_accel", "turbo_spin_echo"):
        return t2w_acq

    # FALLBACK: Parse BIDS filename for acquisition entity
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

    # LEGACY: Check old acquisition/acq columns
    acq = row.get("acquisition", row.get("acq", ""))
    acq_str = str(acq).lower()

    if "tse" in acq_str or "turbo" in acq_str:
        return "turbo_spin_echo"
    elif "2x" in acq_str:
        return "space_2x"
    elif "space" in acq_str:
        return "space_no_accel"

    # UNKNOWN - Must be verified manually
    return "unknown"


def _get_nifti_path(
    nifti_obj: object,
    cache_dir: Path | None = None,
    identifier: str = "nifti",
) -> str | None:
    """Extract file path from HuggingFace Nifti object.

    HuggingFace Nifti() feature type can provide:
    - A file path (string)
    - NIfTI bytes directly (will be written to cache)
    - A nibabel image object (Nifti1ImageWrapper)

    For Nifti1ImageWrapper objects (in-memory nibabel images), we save
    them to a cache directory to get a file path.

    Args:
        nifti_obj: HuggingFace Nifti object.
        cache_dir: Directory to cache bytes data (if None, uses temp dir).
        identifier: Identifier string for cache filename (e.g., "sub-M2001_ses-1_t2w").

    Returns:
        File path string or None.
    """
    import nibabel as nib

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

    # If it's a nibabel image, try to get filename first
    if hasattr(nifti_obj, "get_filename"):
        filename = nifti_obj.get_filename()
        if filename:
            return filename

        # Nifti1ImageWrapper: in-memory image, need to save to disk
        # This happens when HuggingFace datasets returns nibabel images directly
        if hasattr(nifti_obj, "get_fdata"):
            # Determine cache location
            if cache_dir:
                cache_dir.mkdir(parents=True, exist_ok=True)
            else:
                cache_dir = Path(tempfile.gettempdir()) / "arc_nifti_cache"
                cache_dir.mkdir(parents=True, exist_ok=True)

            # Create deterministic filename from data hash
            # Use a faster hash on shape + first few values to avoid loading full volume
            try:
                header = nifti_obj.header
                shape_str = str(tuple(header.get_data_shape()))
                # Create hash from shape + affine for uniqueness
                affine_str = str(nifti_obj.affine.tobytes()[:64])
                content_id = hashlib.md5((shape_str + affine_str).encode()).hexdigest()[:12]
            except Exception as e:
                # Fallback to identifier-based naming
                logger.debug(
                    "Failed to extract header/affine for hashing (using identifier-based hash): %s",
                    e,
                )
                content_id = hashlib.md5(identifier.encode()).hexdigest()[:12]

            cache_path = cache_dir / f"{identifier}_{content_id}.nii.gz"

            if not cache_path.exists():
                # Save the nibabel image to cache
                # Type ignore: nifti_obj is confirmed to be nibabel-like by hasattr check
                nib.save(nifti_obj, cache_path)  # type: ignore[arg-type]
                logger.debug("Cached NIfTI to: %s", cache_path)

            return str(cache_path)

    # If it has a path attribute
    if hasattr(nifti_obj, "path"):
        return nifti_obj.path

    return None


def verify_sample_counts(
    samples: list[ARCSample],
    strict: bool = False,
    paper_parity: bool = False,
) -> None:
    """Verify sample counts match paper requirements.

    FROM PAPER Section 2:
    - 115 SPACE with 2x acceleration
    - 109 SPACE without acceleration
    - 5 TSE excluded
    - Total: 224 usable scans

    NOTE: HuggingFace dataset has 228 lesion masks (4 more than paper).
    NOTE: OpenNeuro is missing 1 SPACE mask, so "paper parity" with available data is 223 samples.

    Args:
        samples: List of extracted samples.
        strict: If True, raise error on count mismatch. If False, just warn.
        paper_parity: If True, expect exactly 223 samples (available subset of paper).

    Raises:
        ValueError: If strict=True and counts don't match expected values.
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

    # Paper expected values (what the paper claims)
    expected_paper_claim = 224
    # Actual reproducible paper subset (missing 1 mask on OpenNeuro)
    expected_reproducible_parity = 223
    # HuggingFace dataset has more samples with lesion masks
    expected_total_huggingface = 228

    if paper_parity:
        # Strict mode for paper parity
        expected = expected_reproducible_parity
        if total != expected:
            raise ValueError(
                f"Paper parity mode: expected {expected} samples (available subset of paper), "
                f"got {total} (space_2x={space_2x_count}, space_no_accel={space_no_accel_count}). "
                "Ensure TSE is excluded and dataset is up to date."
            )
        if tse_count > 0:
            raise ValueError(f"Paper parity mode: Found {tse_count} TSE samples. Must be 0.")
        return

    # Normal mode (flexible)
    min_acceptable = expected_reproducible_parity
    max_acceptable = expected_total_huggingface

    if total < min_acceptable:
        msg = (
            f"Too few samples: got {total}, expected at least {min_acceptable}. "
            f"Check dataset loading and filtering."
        )
        logger.error(msg)
        raise ValueError(msg)

    if total > max_acceptable:
        logger.warning(
            "More samples than expected: got %d, expected at most %d. "
            "This may indicate dataset changes.",
            total,
            max_acceptable,
        )

    if total != expected_paper_claim and total != expected_reproducible_parity:
        logger.warning(
            "Sample count differs from paper claim (224): got %d. "
            "This is expected if including extra samples found on HuggingFace.",
            total,
        )

    if tse_count > 0:
        logger.info(
            "Found %d TSE samples (paper excluded these). "
            "Included because exclude_turbo_spin_echo=False or they are useful extra data.",
            tse_count,
        )

    # In strict mode (non-parity), likely expecting specific experimental setup
    if strict and total not in (expected_paper_claim, expected_reproducible_parity):
        raise ValueError(
            f"Strict mode: expected {expected_paper_claim} or "
            f"{expected_reproducible_parity} samples, got {total}"
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

            try:
                nii = nib.load(temp_path)
                data = nii.get_fdata()
            finally:
                # Clean up temp file
                Path(temp_path).unlink(missing_ok=True)
        elif isinstance(nifti_obj, str):
            nii = nib.load(nifti_obj)
            data = nii.get_fdata()
        else:
            return 0

        # Use > 0 because nibabel scaling can result in values < 0.5 (BUG-001)
        return int(np.sum(data > 0))

    except Exception as e:
        logger.warning("Failed to compute lesion volume: %s", e)
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
