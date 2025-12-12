# Local Spec 01: HuggingFace Data Loader (Corrected)

> **Prerequisite for Training** — Wire up ARC dataset access via bids_hub
>
> **Supersedes:** `_archived/local-01-hf-data-loader.md.CORRUPTED`
>
> **Related:** `FIX-001-bids-hub-integration.md`

---

## Overview

This spec bridges the gap between:
- **HuggingFace Hub** (`hugging-science/arc-aphasia-bids`) where ARC data lives
- **ARCDataset class** which expects `(image_paths, mask_paths)` lists

---

## Data Pipeline Architecture

### Using neuroimaging-go-brrrr (bids_hub)

We use the `neuroimaging-go-brrrr` package as a **git dependency** for:
1. ARC dataset validation utilities
2. Expected count constants from the ARC paper
3. HuggingFace schema definitions

**Dependency in pyproject.toml:**
```toml
dependencies = [
    "neuroimaging-go-brrrr @ git+https://github.com/The-Obstacle-Is-The-Way/neuroimaging-go-brrrr.git@v0.2.1",
]
```

### What We Import from bids_hub

```python
from bids_hub import validate_arc_download  # Validation for local BIDS directories
from bids_hub.validation.arc import (
    ARC_VALIDATION_CONFIG,   # Full dataset counts: subjects=230, sessions=902, t2w=447
)
```

**NOT imported** (these are for UPLOAD, not consumption):
- `build_arc_file_table` - for building HF datasets from local BIDS
- `get_arc_features` - schema definition for upload (HF infers schema on load)

### What We Implement Ourselves

Our domain-specific logic for the MeshNet paper:
- `ARCSample` / `ARCDatasetInfo` dataclasses
- Acquisition type filtering (space_2x, space_no_accel, TSE exclusion)
- Paper-specific sample counts (224 = 115 + 109)
- Stratification utilities

### Count Distinction (CRITICAL)

| Source | Count | Purpose |
|--------|-------|---------|
| `ARC_VALIDATION_CONFIG` | 230 subjects, 902 sessions, 447 t2w | Full ARC dataset integrity |
| **Paper training subset** | **224 samples** (115 + 109) | MeshNet training data |

These are DIFFERENT. The paper uses a SUBSET of ARC (only T2w with SPACE acquisition, with lesion masks).

### Parity-Critical Behaviors (MUST PRESERVE)

These behaviors ensure paper replication accuracy:

1. **`strict_t2w=True`** (default): Only use T2w images, NO FLAIR fallback
   - Paper explicitly uses T2-weighted images for lesion segmentation

2. **Filename-first acquisition parsing**: Extract `acq-*` entity from BIDS filename
   - Primary: `acq-space2x` → `space_2x`, `acq-space` → `space_no_accel`, `acq-tse` → `turbo_spin_echo`
   - Fallback: Row metadata (less reliable)

3. **Paper-specific count verification**: 224 = 115 + 109
   - `verify_counts=True` (default) raises `ValueError` if mismatch
   - These counts are HARDCODED (from paper), not from `ARC_VALIDATION_CONFIG`

4. **Unknown acquisition rejection**: Skip samples with unrecognizable acquisition type
   - Ensures only verified SPACE sequences are included

---

## Dataset Facts

| Property | Value | Source |
|----------|-------|--------|
| HuggingFace Repo | `hugging-science/arc-aphasia-bids` | OpenNeuro ds004884 |
| Total subjects | 230 | ARC paper (Gibson et al., 2024) |
| Total sessions | 902 | bids_hub.validation.arc |
| Sessions with lesion masks | ~228 | HuggingFace dataset |
| SPACE with 2x accel | 115 scans | MeshNet paper Section 2 |
| SPACE without accel | 109 scans | MeshNet paper Section 2 |
| TSE excluded | 5 scans | MeshNet paper Section 2 |
| **Usable for training** | **224** | MeshNet paper Section 2 |

---

## Implementation

### File: `src/arc_meshchop/data/huggingface_loader.py`

```python
"""HuggingFace dataset loader for ARC.

Uses bids_hub (neuroimaging-go-brrrr) for validation constants.
Implements paper-specific filtering and parity-critical behaviors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

# Import validation constants from bids_hub (full dataset counts)
from bids_hub.validation.arc import ARC_VALIDATION_CONFIG

if TYPE_CHECKING:
    from datasets import Dataset

logger = logging.getLogger(__name__)


@dataclass
class ARCSample:
    """Single ARC sample metadata."""
    subject_id: str
    session_id: str
    image_path: Path
    mask_path: Path | None
    lesion_volume: int
    acquisition_type: str


@dataclass
class ARCDatasetInfo:
    """Complete ARC dataset information."""
    samples: list[ARCSample]

    @property
    def image_paths(self) -> list[Path]:
        return [s.image_path for s in self.samples]

    @property
    def mask_paths(self) -> list[Path]:
        return [s.mask_path for s in self.samples if s.mask_path is not None]

    @property
    def lesion_volumes(self) -> list[int]:
        return [s.lesion_volume for s in self.samples]

    @property
    def acquisition_types(self) -> list[str]:
        return [s.acquisition_type for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)


def load_arc_from_huggingface(
    repo_id: str = "hugging-science/arc-aphasia-bids",
    cache_dir: Path | str | None = None,
    include_space_2x: bool = True,
    include_space_no_accel: bool = True,
    exclude_turbo_spin_echo: bool = True,
    require_lesion_mask: bool = True,
    strict_t2w: bool = True,  # PARITY-CRITICAL: No FLAIR fallback
    verify_counts: bool = True,
) -> ARCDatasetInfo:
    """Load ARC dataset from HuggingFace Hub.

    FROM PAPER Section 2:
    "We utilized SPACE sequences with x2 in plane acceleration (115 scans)
    and without acceleration (109 scans), while excluding the turbo-spin
    echo T2-weighted sequences (5 scans)."

    Args:
        repo_id: HuggingFace repository ID.
        cache_dir: Optional cache directory.
        include_space_2x: Include SPACE with 2x acceleration.
        include_space_no_accel: Include SPACE without acceleration.
        exclude_turbo_spin_echo: Exclude turbo-spin echo sequences.
        require_lesion_mask: Only include samples with lesion masks.
        strict_t2w: If True, ONLY use T2w (no FLAIR fallback). Default True for paper parity.
        verify_counts: Verify counts match paper expectations (224 = 115 + 109).

    Returns:
        ARCDatasetInfo with filtered samples.
    """
    from datasets import load_dataset

    logger.info("Loading ARC dataset from HuggingFace: %s", repo_id)

    # Load dataset (HuggingFace handles caching)
    ds = load_dataset(repo_id, cache_dir=str(cache_dir) if cache_dir else None)
    dataset = ds["train"]

    logger.info("Loaded %d sessions", len(dataset))
    logger.info(
        "bids_hub expected: %d subjects, %d sessions",
        ARC_VALIDATION_CONFIG.expected_counts["subjects"],
        ARC_VALIDATION_CONFIG.expected_counts["sessions"],
    )

    # Extract and filter samples (with parity-critical behaviors)
    samples = _extract_samples(
        dataset,
        include_space_2x=include_space_2x,
        include_space_no_accel=include_space_no_accel,
        exclude_turbo_spin_echo=exclude_turbo_spin_echo,
        require_lesion_mask=require_lesion_mask,
        strict_t2w=strict_t2w,
    )

    if verify_counts:
        _verify_paper_counts(samples)

    return ARCDatasetInfo(samples=samples)


def _verify_paper_counts(samples: list[ARCSample]) -> None:
    """Verify sample counts match MeshNet paper requirements."""
    # Paper-specific counts (not from bids_hub - these are training subset)
    EXPECTED_SPACE_2X = 115
    EXPECTED_SPACE_NO_ACCEL = 109
    EXPECTED_TOTAL = 224

    space_2x = sum(1 for s in samples if s.acquisition_type == "space_2x")
    space_no_accel = sum(1 for s in samples if s.acquisition_type == "space_no_accel")
    total = len(samples)

    if total != EXPECTED_TOTAL:
        raise ValueError(
            f"Sample count mismatch: got {total}, expected {EXPECTED_TOTAL} "
            f"(space_2x={space_2x}/{EXPECTED_SPACE_2X}, "
            f"space_no_accel={space_no_accel}/{EXPECTED_SPACE_NO_ACCEL})"
        )

    logger.info("Paper count verification passed: %d samples", total)


# ... rest of implementation (see current file) ...
```

---

## Usage Examples

### Load for Training

```python
from arc_meshchop.data import (
    ARCDataset,
    load_arc_from_huggingface,
    generate_nested_cv_splits,
)

# Step 1: Load from HuggingFace
arc_info = load_arc_from_huggingface(verify_counts=True)
print(f"Loaded {len(arc_info)} samples")  # Expected: 224

# Step 2: Generate CV splits
splits = generate_nested_cv_splits(
    n_samples=len(arc_info),
    stratification_labels=...,
    num_outer_folds=3,
    num_inner_folds=3,
)

# Step 3: Create PyTorch dataset
train_dataset = ARCDataset(
    image_paths=[arc_info.image_paths[i] for i in split.train_indices],
    mask_paths=[arc_info.mask_paths[i] for i in split.train_indices],
)
```

### Validate Local BIDS Directory

```python
from bids_hub import validate_arc_download
from pathlib import Path

result = validate_arc_download(
    Path("data/openneuro/ds004884"),
    tolerance=0.0,
)

if result.all_passed:
    print("Dataset valid!")
else:
    print(result.summary())
```

---

## Verification Commands

```bash
# Verify bids_hub imports work
uv run python -c "
from bids_hub.validation.arc import ARC_VALIDATION_CONFIG
print('Expected subjects:', ARC_VALIDATION_CONFIG.expected_counts['subjects'])
print('Expected sessions:', ARC_VALIDATION_CONFIG.expected_counts['sessions'])
"

# Run loader tests
uv run pytest tests/test_data/test_huggingface_loader.py -v
```

---

## Implementation Checklist

- [ ] Refactor huggingface_loader.py to import from bids_hub
- [ ] Use ARC_VALIDATION_CONFIG for expected counts logging
- [ ] Keep paper-specific counts (224) for training verification
- [ ] Update tests to mock bids_hub imports
- [ ] Run CI to verify all integrations work
