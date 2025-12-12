# FIX-001: bids_hub Integration Restoration

> **Priority:** HIGH - Foundation for data pipeline
>
> **Status:** Ready for Implementation
>
> **Branch:** `refactor/bids-hub-integration`

---

## Executive Summary

The `neuroimaging-go-brrrr` git dependency exists in `pyproject.toml` but is **completely unused**.
The HuggingFace data loader was hand-rolled instead of using `bids_hub` utilities.
This document specifies the required fixes.

---

## 1. What Went Wrong

### 1.1 Root Cause: Git Submodule Confusion

During initial development, there was a git submodule issue with `neuroimaging-go-brrrr`.
The spec (local-01) was incorrectly modified to say "Native HuggingFace Approach...
avoids git submodule issues" while simultaneously keeping `bids_hub` import examples.

**Result:** An internally contradictory spec that led to a hand-rolled implementation.

### 1.2 The Contradiction in Archived Spec

```
# From local-01-hf-data-loader.md.CORRUPTED (now archived)

Lines 19-31:
"### Native HuggingFace Approach
We use the **native HuggingFace `datasets` and `huggingface-hub`** libraries directly.
This avoids git submodule issues..."

Line 137:
"from bids_hub.datasets.arc import build_arc_file_table, get_arc_features"
```

These statements are **mutually exclusive**. The implementation followed the first
paragraph and ignored the second, resulting in ~500 lines of hand-rolled code that
duplicates functionality already in `bids_hub`.

### 1.3 What Should Have Happened

The `neuroimaging-go-brrrr` package was specifically created to:
1. Handle BIDS-to-HuggingFace conversion
2. Work around the SIGKILL bug (huggingface/datasets#7894)
3. Provide validation utilities for ARC dataset

It's listed as a git dependency for a reason:

```toml
# pyproject.toml (EXISTING - CORRECT)
"neuroimaging-go-brrrr @ git+https://github.com/The-Obstacle-Is-The-Way/neuroimaging-go-brrrr.git@v0.2.1"
```

But **nothing imports from `bids_hub`**.

---

## 2. Current State Analysis

### 2.1 What Exists (pyproject.toml)

```toml
dependencies = [
    # ...
    "neuroimaging-go-brrrr @ git+https://github.com/The-Obstacle-Is-The-Way/neuroimaging-go-brrrr.git@v0.2.1",
    # ...
]
```

### 2.2 What Was Built (huggingface_loader.py)

**567 lines** of hand-rolled code that:
- Does NOT import from `bids_hub`
- Re-implements NIfTI path extraction
- Re-implements acquisition type parsing
- Does NOT use the SIGKILL workaround from `bids_hub`

### 2.3 What bids_hub Provides

From `_references/neuroimaging-go-brrrr/docs/reference/api.md`:

```python
from bids_hub import (
    # Core
    DatasetBuilderConfig,
    build_hf_dataset,
    push_dataset_to_hub,
    # ARC
    build_arc_file_table,
    build_and_push_arc,
    get_arc_features,
    validate_arc_download,
    # Validation
    ValidationResult,
)
```

### 2.4 The SIGKILL Bug Workaround

From `_references/neuroimaging-go-brrrr/src/bids_hub/core/builder.py` lines 176-193:

```python
# WORKAROUND for huggingface/datasets#7894
# Sequence(Nifti()) columns crash embed_table_storage() after shard()
# due to Arrow slice references. Pandas roundtrip breaks the references.
if nifti_sequence_columns:
    logger.warning(
        "Applying workaround for huggingface/datasets#7894: "
        "Converting %s to pandas/back to break Arrow slice references",
        nifti_sequence_columns,
    )
    # This breaks the Arrow slice references that cause SIGKILL
    df = ds.to_pandas()
    ds = Dataset.from_pandas(df, features=ds.features)
```

**This workaround is NOT in our hand-rolled code.**

---

## 3. Required Changes

### 3.1 File: `src/arc_meshchop/data/huggingface_loader.py`

**Action:** Refactor to use `bids_hub` utilities

**Before (current):**
```python
# NO bids_hub imports
from datasets import load_dataset
# ... 567 lines of hand-rolled code
```

**After:**
```python
"""HuggingFace dataset loader for ARC.

Uses bids_hub (neuroimaging-go-brrrr) utilities for ARC dataset access.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

# Import from neuroimaging-go-brrrr (git dependency)
from bids_hub import get_arc_features, validate_arc_download
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

    # ... rest of properties ...


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

    Uses bids_hub utilities with SIGKILL workaround baked in.
    """
    from datasets import load_dataset

    logger.info("Loading ARC dataset from HuggingFace: %s", repo_id)

    # Load via datasets library (bids_hub's workaround is in push, not load)
    ds = load_dataset(repo_id, cache_dir=str(cache_dir) if cache_dir else None)
    dataset = ds["train"]

    # Use our extraction logic but leverage bids_hub for validation
    samples = _extract_samples(
        dataset,
        include_space_2x=include_space_2x,
        include_space_no_accel=include_space_no_accel,
        exclude_turbo_spin_echo=exclude_turbo_spin_echo,
        require_lesion_mask=require_lesion_mask,
        strict_t2w=strict_t2w,
    )

    if verify_counts:
        _verify_sample_counts_against_bids_hub(samples)

    return ARCDatasetInfo(samples=samples)


def _verify_sample_counts_against_bids_hub(samples: list[ARCSample]) -> None:
    """Verify counts match bids_hub's ARC_VALIDATION_CONFIG."""
    # Use bids_hub's expected counts instead of hardcoding
    expected = ARC_VALIDATION_CONFIG.expected_counts
    # ... validation logic using bids_hub constants ...
```

### 3.2 Key Imports to Add

```python
# For CONSUMPTION (training), only import validation constants:
from bids_hub.validation.arc import ARC_VALIDATION_CONFIG

# Optional: For validating local BIDS directories
from bids_hub import validate_arc_download
```

**NOT imported** (for UPLOAD, not consumption):
- `get_arc_features` - Schema definition for building HF datasets
- `build_arc_file_table` - For converting local BIDS to HF format

### 3.3 What to Keep vs Replace

| Component | Action | Reason |
|-----------|--------|--------|
| `ARCSample` dataclass | Keep | Our domain model |
| `ARCDatasetInfo` dataclass | Keep | Our domain model |
| `load_arc_from_huggingface()` | Refactor | Use bids_hub validation |
| `_extract_samples()` | Keep | Our filtering logic |
| `_determine_acquisition_type()` | Keep | Our parsing logic |
| `_get_nifti_path()` | Simplify | HF handles caching |
| `verify_sample_counts()` | Replace | Use ARC_VALIDATION_CONFIG |
| `_compute_lesion_volume_from_nifti()` | Keep | Our utility |
| `download_arc_to_local()` | Keep | Our utility |

---

## 4. Validation Requirements

### 4.1 Expected Counts (from bids_hub)

```python
# From bids_hub.validation.arc
ARC_VALIDATION_CONFIG = DatasetValidationConfig(
    name="arc",
    expected_counts={
        "subjects": 230,
        "sessions": 902,
        "t1w": 441,
        "t2w": 447,
        "flair": 235,
        "bold": 850,
        "dwi": 613,
        "sbref": 88,
        "lesion": 230,  # All subjects have lesion masks
    },
    # ...
)
```

### 4.2 Paper Requirements (our filtering)

From the MeshNet paper:
- 115 SPACE with 2x acceleration
- 109 SPACE without acceleration
- 5 TSE excluded
- **Total: 224 usable scans**

These counts are for TRAINING, not raw dataset counts.

---

## 5. Implementation Checklist

- [ ] Add `bids_hub` imports to huggingface_loader.py
- [ ] Use `ARC_VALIDATION_CONFIG.expected_counts` instead of hardcoded values
- [ ] Keep our `ARCSample`/`ARCDatasetInfo` domain models
- [ ] Keep our acquisition type filtering logic
- [ ] Update tests to verify bids_hub integration
- [ ] Run `uv sync` to ensure git dependency resolves
- [ ] Run full CI (`make ci`)

---

## 6. Testing

### 6.1 Verify bids_hub Import Works

```bash
uv run python -c "from bids_hub import get_arc_features, validate_arc_download; print('bids_hub imports OK')"
```

### 6.2 Verify ARC_VALIDATION_CONFIG

```bash
uv run python -c "
from bids_hub.validation.arc import ARC_VALIDATION_CONFIG
print('Expected counts:', ARC_VALIDATION_CONFIG.expected_counts)
"
```

### 6.3 Run Unit Tests

```bash
uv run pytest tests/test_data/test_huggingface_loader.py -v
```

---

## 7. References

- **neuroimaging-go-brrrr repo:** https://github.com/The-Obstacle-Is-The-Way/neuroimaging-go-brrrr
- **Version pinned:** v0.2.1
- **Local reference copy:** `_references/neuroimaging-go-brrrr/` (read-only, no git history)
- **SIGKILL bug:** huggingface/datasets#7894
- **Workaround PR:** huggingface/datasets#7896
- **Archived corrupted spec:** `docs/specs/_archived/local-01-hf-data-loader.md.CORRUPTED`

---

## 8. Notes

### Why Not Fully Replace Our Code?

The `bids_hub` package is designed for **uploading** BIDS datasets to HuggingFace.
For **consuming** the dataset (loading for training), we still need:

1. Our `ARCSample`/`ARCDatasetInfo` domain models
2. Our acquisition type filtering (paper-specific)
3. Our stratification logic (paper-specific)

We use `bids_hub` for:
1. Validation constants (expected counts)
2. Schema definitions (get_arc_features)
3. Future: validation utilities

### The SIGKILL Workaround Location

The workaround in `bids_hub/core/builder.py` is for **push operations**.
For **load operations**, the workaround isn't needed because we're not sharding.
However, if we ever need to re-shard or process large datasets, we should
use `bids_hub`'s builder utilities.
