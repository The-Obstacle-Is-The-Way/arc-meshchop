# Local Spec 01: HuggingFace Data Loader

> **Prerequisite for Training** — Wire up actual ARC dataset access
>
> **Goal:** Implement the missing link between HuggingFace Hub and ARCDataset class.

---

## Overview

This spec bridges the gap between:
- **HuggingFace Hub** (`hugging-science/arc-aphasia-bids`) where ARC data lives
- **ARCDataset class** which expects `(image_paths, mask_paths)` lists

Currently, the infrastructure assumes paths exist. This spec implements the loader that fetches real data.

---

## IMPORTANT: Data Pipeline Architecture

### Native HuggingFace Approach

We use the **native HuggingFace `datasets` and `huggingface-hub`** libraries directly.
This avoids git submodule issues and provides more control over the data loading process.

**Dependencies** (already in `pyproject.toml`):
```toml
dependencies = [
    "datasets>=3.4.0",
    "huggingface-hub>=0.32.0",
]
```

### How We Use It

```python
from datasets import load_dataset
from huggingface_hub import hf_hub_download

# Load dataset metadata from HuggingFace Hub
ds = load_dataset("hugging-science/arc-aphasia-bids", split="train")

# Files are cached locally after first download
# Access via row["t2w"], row["lesion"], etc.
```

### What We DO

- ✅ Use native `datasets` library for HuggingFace access
- ✅ Use `huggingface_hub` for direct file downloads when needed
- ✅ Infer acquisition type from **cached file paths** (BIDS naming convention)
- ✅ Pass extracted paths to our `ARCDataset` class
- ✅ **Enforce 224 sample count** (115 SPACE 2x + 109 SPACE no accel) for paper parity

### What We Do NOT Do

- ❌ Fall back to FLAIR when T2w missing (paper requires T2w only)
- ❌ Guess acquisition types without verification
- ❌ Accept sample counts that don't match paper (must be exactly 224)

---

## 1. The Gap

### What Exists

```python
# ARCDataset expects file paths
from arc_meshchop.data import ARCDataset

dataset = ARCDataset(
    image_paths=["/path/to/image1.nii.gz", ...],  # WHERE DO THESE COME FROM?
    mask_paths=["/path/to/mask1.nii.gz", ...],    # WHERE DO THESE COME FROM?
)
```

### What's Missing

```python
# Need this function
from arc_meshchop.data import load_arc_from_huggingface

image_paths, mask_paths, metadata = load_arc_from_huggingface()
# Returns paths to NIfTI files ready for ARCDataset
```

---

## 2. Dataset Facts (FROM PAPER)

| Property | Value | Source |
|----------|-------|--------|
| HuggingFace Repo | `hugging-science/arc-aphasia-bids` | User-provided |
| Total subjects | 230 | Paper Section 2 |
| SPACE with 2x accel | 115 scans | Paper Section 2 |
| SPACE without accel | 109 scans | Paper Section 2 |
| Excluded (turbo-spin echo) | 5 scans | Paper Section 2 |
| **Usable scans** | **224** | Paper Section 2 |
| Sessions with lesion masks | 228 | HuggingFace dataset |
| Image modality for segmentation | T2-weighted | Paper Section 2 |

> **NOTE:** The HuggingFace dataset has 902 sessions (longitudinal), but only 228 have expert lesion masks.
> The paper uses 224 SPACE scans with masks for training.

---

## 3. Implementation

### 3.1 HuggingFace Loader

**File:** `src/arc_meshchop/data/huggingface_loader.py`

```python
"""HuggingFace dataset loader for ARC.

Loads the ARC dataset from HuggingFace Hub using neuroimaging-go-brrrr
utilities and extracts paths for use with ARCDataset.

Dependencies:
    neuroimaging-go-brrrr @ git+https://github.com/The-Obstacle-Is-The-Way/neuroimaging-go-brrrr.git@v0.2.1

FROM PAPER:
"The Aphasia Recovery Cohort (ARC) is an open-source neuroimaging dataset
comprising T2-weighted MRI scans from 230 unique individuals with chronic stroke."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

# Import from neuroimaging-go-brrrr (git dependency)
from bids_hub.datasets.arc import build_arc_file_table, get_arc_features

if TYPE_CHECKING:
    from datasets import Dataset
    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ARCSample:
    """Single ARC sample metadata.

    Attributes:
        subject_id: BIDS subject ID (e.g., "sub-M2001").
        session_id: BIDS session ID (e.g., "ses-1").
        image_path: Path to T2-weighted image.
        mask_path: Path to lesion mask.
        lesion_volume: Lesion volume in voxels.
        acquisition_type: "space_2x" or "space_no_accel".
    """

    subject_id: str
    session_id: str
    image_path: Path
    mask_path: Path
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
        """Get all mask paths."""
        return [s.mask_path for s in self.samples]

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
            "HuggingFace datasets library required. "
            "Install with: pip install datasets"
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
            "No samples match the specified filters. "
            "Check filter settings and dataset contents."
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
        logger.info("✅ Sample count verification passed: 224 samples (115 space_2x + 109 space_no_accel)")

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
                idx, image_path
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

        samples.append(ARCSample(
            subject_id=row.get("subject_id", f"sub-{idx:04d}"),
            session_id=row.get("session_id", "ses-1"),
            image_path=Path(image_path),
            mask_path=Path(mask_path) if mask_path else Path(""),
            lesion_volume=lesion_volume,
            acquisition_type=acq_type,
        ))

    return samples


def _determine_acquisition_type(row: dict, file_path: str | None) -> str:
    """Determine acquisition type from BIDS filename or row metadata.

    FROM PAPER:
    - SPACE with 2x in-plane acceleration: 115 scans
    - SPACE without acceleration: 109 scans
    - Turbo-spin echo (excluded): 5 scans

    Acquisition type is inferred from BIDS filename conventions:
    - `acq-space2x` or `acq-SPACE2x` → space_2x
    - `acq-space` or `acq-SPACE` (no acceleration) → space_no_accel
    - `acq-tse` or `acq-TSE` → turbo_spin_echo

    Args:
        row: Dataset row with metadata.
        file_path: Cached file path (BIDS naming) for acquisition inference.

    Returns:
        Acquisition type string.
    """
    import re

    # Primary method: Parse BIDS filename for acquisition entity
    # BIDS format: sub-XXX_ses-X_acq-XXXXX_T2w.nii.gz
    if file_path:
        file_path_lower = str(file_path).lower()

        # Look for acq-* entity in filename
        acq_match = re.search(r'acq-([a-z0-9]+)', file_path_lower)
        if acq_match:
            acq_value = acq_match.group(1)

            if 'tse' in acq_value or 'turbo' in acq_value:
                return "turbo_spin_echo"
            elif 'space2x' in acq_value or '2x' in acq_value:
                return "space_2x"
            elif 'space' in acq_value:
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


def _get_nifti_path(nifti_obj, cache_dir: Path | None = None) -> str | None:
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
    import tempfile
    import hashlib

    if nifti_obj is None:
        return None

    # If it's already a string path
    if isinstance(nifti_obj, str):
        return nifti_obj

    # If it's a dict with path or bytes
    if isinstance(nifti_obj, dict):
        # Prefer path over bytes
        if "path" in nifti_obj and nifti_obj["path"]:
            return nifti_obj["path"]

        # Handle bytes - write to cache file
        if "bytes" in nifti_obj and nifti_obj["bytes"]:
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
        space_2x_count, space_no_accel_count, tse_count, unknown_count, total
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
            f"SPACE no-accel count mismatch: got {space_no_accel_count}, expected {expected_space_no_accel}"
        )

    if total != expected_total:
        warnings.append(
            f"Total count mismatch: got {total}, expected {expected_total}"
        )

    if tse_count > 0:
        warnings.append(
            f"TSE samples should be excluded: found {tse_count}"
        )

    if unknown_count > 0:
        warnings.append(
            f"Unknown acquisition types found: {unknown_count} samples"
        )

    if warnings:
        for w in warnings:
            logger.warning(w)
        raise ValueError(
            f"Sample count verification failed. "
            f"Expected 224 (115 space_2x + 109 space_no_accel), got {total}. "
            f"Issues: {'; '.join(warnings)}"
        )


def _compute_lesion_volume_from_nifti(nifti_obj) -> int:
    """Compute lesion volume from NIfTI object.

    Args:
        nifti_obj: HuggingFace Nifti object containing mask.

    Returns:
        Lesion volume in voxels.
    """
    try:
        import nibabel as nib
        import numpy as np

        # Try to load data
        if hasattr(nifti_obj, "get_fdata"):
            data = nifti_obj.get_fdata()
        elif isinstance(nifti_obj, dict) and "bytes" in nifti_obj:
            # Load from bytes
            import io
            nii = nib.load(io.BytesIO(nifti_obj["bytes"]))
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
    **filter_kwargs,
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

        # Copy files (implementation depends on HuggingFace data format)
        _copy_nifti(sample.image_path, local_image)
        _copy_nifti(sample.mask_path, local_mask)

        local_samples.append(ARCSample(
            subject_id=sample.subject_id,
            session_id=sample.session_id,
            image_path=local_image,
            mask_path=local_mask,
            lesion_volume=sample.lesion_volume,
            acquisition_type=sample.acquisition_type,
        ))

        if (i + 1) % 50 == 0:
            logger.info("Downloaded %d/%d samples", i + 1, len(info))

    logger.info("Download complete: %d samples saved to %s", len(local_samples), output_dir)

    return ARCDatasetInfo(samples=local_samples)


def _copy_nifti(src: Path, dst: Path) -> None:
    """Copy NIfTI file to destination.

    Args:
        src: Source path.
        dst: Destination path.
    """
    import shutil

    if src.exists():
        shutil.copy2(src, dst)
    else:
        logger.warning("Source file not found: %s", src)
```

### 3.2 Integration with ARCDataset

**Update:** `src/arc_meshchop/data/__init__.py`

```python
# Add to existing exports
from arc_meshchop.data.huggingface_loader import (
    ARCDatasetInfo,
    ARCSample,
    download_arc_to_local,
    load_arc_from_huggingface,
)

__all__ = [
    # ... existing exports ...
    "ARCDatasetInfo",
    "ARCSample",
    "download_arc_to_local",
    "load_arc_from_huggingface",
]
```

---

## 4. Usage Example

### Option A: Load from HuggingFace Hub (Recommended for Training)

```python
from datasets import load_dataset
from arc_meshchop.data import (
    ARCDataset,
    generate_nested_cv_splits,
    create_stratification_labels,
    get_lesion_quintile,
)

# Step 1: Load from HuggingFace Hub
ds = load_dataset("hugging-science/arc-aphasia-bids", split="train")

# Step 2: Filter sessions with lesion masks
sessions_with_masks = [
    i for i in range(len(ds))
    if ds[i]["lesion"] is not None
]
print(f"Sessions with masks: {len(sessions_with_masks)}")
# Expected: ~228 sessions
```

### Option B: Load from Local BIDS (For Development)

```python
from pathlib import Path
from bids_hub.datasets.arc import build_arc_file_table

# Step 1: Build file table from local BIDS directory
bids_root = Path("data/openneuro/ds004884")
file_table = build_arc_file_table(bids_root)

print(f"Found {len(file_table)} sessions")
# Each row has: subject_id, session_id, t1w, t2w, flair, lesion, ...
```

### Full Training Pipeline Example

```python
from arc_meshchop.data import (
    ARCDataset,
    load_arc_from_huggingface,
    generate_nested_cv_splits,
    create_stratification_labels,
    get_lesion_quintile,
)

# Step 1: Load from HuggingFace
arc_info = load_arc_from_huggingface(
    include_space_2x=True,
    include_space_no_accel=True,
    exclude_turbo_spin_echo=True,
    require_lesion_mask=True,
)

print(f"Loaded {len(arc_info)} samples")
# Expected: ~224 samples (matching paper)

# Step 2: Compute stratification labels
quintiles = [get_lesion_quintile(v) for v in arc_info.lesion_volumes]
strat_labels = create_stratification_labels(quintiles, arc_info.acquisition_types)

# Step 3: Generate CV splits
splits = generate_nested_cv_splits(
    n_samples=len(arc_info),
    stratification_labels=strat_labels,
    num_outer_folds=3,
    num_inner_folds=3,
    random_seed=42,
)

# Step 4: Create dataset for specific fold
outer_fold = 0
inner_fold = 0
split = splits.get_split(outer_fold, inner_fold)

train_dataset = ARCDataset(
    image_paths=[arc_info.image_paths[i] for i in split.train_indices],
    mask_paths=[arc_info.mask_paths[i] for i in split.train_indices],
)

val_dataset = ARCDataset(
    image_paths=[arc_info.image_paths[i] for i in split.val_indices],
    mask_paths=[arc_info.mask_paths[i] for i in split.val_indices],
)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
```

---

## 5. Tests

**File:** `tests/test_data/test_huggingface_loader.py`

```python
"""Tests for HuggingFace loader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestLoadARCFromHuggingFace:
    """Tests for load_arc_from_huggingface function."""

    def test_imports_datasets_library(self) -> None:
        """Verify datasets import error is informative."""
        with patch.dict("sys.modules", {"datasets": None}):
            from arc_meshchop.data.huggingface_loader import load_arc_from_huggingface

            # Should raise ImportError with helpful message
            with pytest.raises(ImportError, match="datasets"):
                load_arc_from_huggingface()

    def test_returns_arc_dataset_info(self) -> None:
        """Verify return type is ARCDatasetInfo."""
        # Mock the HuggingFace dataset
        mock_dataset = _create_mock_dataset(n_samples=10)

        with patch("datasets.load_dataset", return_value={"train": mock_dataset}):
            from arc_meshchop.data.huggingface_loader import load_arc_from_huggingface

            info = load_arc_from_huggingface()

            assert hasattr(info, "samples")
            assert hasattr(info, "image_paths")
            assert hasattr(info, "mask_paths")

    def test_filters_by_acquisition_type(self) -> None:
        """Verify acquisition type filtering works."""
        mock_dataset = _create_mock_dataset(
            n_samples=10,
            acquisition_types=["space_2x"] * 5 + ["turbo_spin_echo"] * 5,
        )

        with patch("datasets.load_dataset", return_value={"train": mock_dataset}):
            from arc_meshchop.data.huggingface_loader import load_arc_from_huggingface

            info = load_arc_from_huggingface(exclude_turbo_spin_echo=True)

            # Should exclude turbo-spin echo
            assert all(s.acquisition_type != "turbo_spin_echo" for s in info.samples)

    def test_requires_lesion_mask_by_default(self) -> None:
        """Verify samples without masks are excluded by default."""
        mock_dataset = _create_mock_dataset(
            n_samples=10,
            has_mask=[True] * 5 + [False] * 5,
        )

        with patch("datasets.load_dataset", return_value={"train": mock_dataset}):
            from arc_meshchop.data.huggingface_loader import load_arc_from_huggingface

            info = load_arc_from_huggingface(require_lesion_mask=True)

            # Should only have samples with masks
            assert len(info) == 5


class TestARCDatasetInfo:
    """Tests for ARCDatasetInfo dataclass."""

    def test_property_accessors(self) -> None:
        """Verify property accessors work correctly."""
        from arc_meshchop.data.huggingface_loader import ARCDatasetInfo, ARCSample

        samples = [
            ARCSample(
                subject_id="sub-001",
                session_id="ses-1",
                image_path=Path("/path/image.nii.gz"),
                mask_path=Path("/path/mask.nii.gz"),
                lesion_volume=1000,
                acquisition_type="space_2x",
            ),
        ]

        info = ARCDatasetInfo(samples=samples)

        assert len(info) == 1
        assert info.image_paths == [Path("/path/image.nii.gz")]
        assert info.mask_paths == [Path("/path/mask.nii.gz")]
        assert info.lesion_volumes == [1000]
        assert info.acquisition_types == ["space_2x"]


def _create_mock_dataset(
    n_samples: int = 10,
    acquisition_types: list[str] | None = None,
    has_mask: list[bool] | None = None,
) -> MagicMock:
    """Create mock HuggingFace dataset for testing."""
    if acquisition_types is None:
        acquisition_types = ["space_2x"] * n_samples
    if has_mask is None:
        has_mask = [True] * n_samples

    mock = MagicMock()
    mock.__len__ = MagicMock(return_value=n_samples)

    def getitem(idx):
        return {
            "subject_id": f"sub-{idx:04d}",
            "session_id": "ses-1",
            "t2w": f"/path/t2w_{idx}.nii.gz",
            "lesion": f"/path/mask_{idx}.nii.gz" if has_mask[idx] else None,
            "acquisition": acquisition_types[idx],
        }

    mock.__getitem__ = getitem
    return mock
```

---

## 6. Implementation Checklist

- [ ] Create `src/arc_meshchop/data/huggingface_loader.py`
- [ ] Implement `ARCSample` dataclass
- [ ] Implement `ARCDatasetInfo` dataclass
- [ ] Implement `load_arc_from_huggingface()` function
- [ ] Implement acquisition type filtering
- [ ] Implement lesion volume computation
- [ ] Implement `download_arc_to_local()` for offline workflows
- [ ] Update `src/arc_meshchop/data/__init__.py` with exports
- [ ] Create tests in `tests/test_data/test_huggingface_loader.py`
- [ ] Test with real HuggingFace dataset (integration test)
- [ ] Verify ~224 samples after filtering (matching paper)

---

## 7. Verification Commands

```bash
# Run HuggingFace loader tests
uv run pytest tests/test_data/test_huggingface_loader.py -v

# Test loading from HuggingFace (requires network)
uv run python -c "
from arc_meshchop.data import load_arc_from_huggingface

info = load_arc_from_huggingface(
    include_space_2x=True,
    include_space_no_accel=True,
    exclude_turbo_spin_echo=True,
    require_lesion_mask=True,
)

print(f'Total samples: {len(info)}')
print(f'Expected: ~224 (from paper)')
print(f'Acquisition types: {set(info.acquisition_types)}')
"
```

---

## 8. Notes

### HuggingFace Dataset Structure

The actual structure of `hugging-science/arc-aphasia-bids` depends on how neuroimaging-go-brrrr
uploaded the data. Key fields expected:

- `subject_id`: BIDS subject identifier
- `session_id`: BIDS session identifier
- `t2w`: T2-weighted NIfTI (Nifti() type)
- `lesion`: Lesion mask NIfTI (Nifti() type, nullable)
- Acquisition metadata for filtering

### Adjustments May Be Needed

The filtering logic (especially `_determine_acquisition_type`) may need adjustment based on
the actual metadata structure in the HuggingFace dataset. Test with real data and adjust as needed.

### Acquisition Parsing Validation

**IMPORTANT:** Before running full experiments, validate that acquisition type parsing works
correctly on the real HuggingFace dataset:

```bash
# Quick validation of acquisition parsing
uv run python -c "
from arc_meshchop.data import load_arc_samples

samples = load_arc_samples(verify_counts=False)  # Disable strict count check
acq_counts = {}
for s in samples:
    acq_counts[s.acquisition_type] = acq_counts.get(s.acquisition_type, 0) + 1
print('Acquisition type counts:', acq_counts)
print('Expected: space_2x=115, space_no_accel=109, turbo_spin_echo=5')
"
```

If counts don't match paper expectations:
1. Check if HuggingFace cached filenames preserve BIDS `acq-*` entity
2. Inspect actual filename format with `samples[0].image_path`
3. Update `_determine_acquisition_type()` regex patterns as needed
4. Re-run with `verify_counts=True` after fixing

The `verify_counts=True` flag (default) will fail fast if counts don't match,
preventing silent errors from propagating to training.
