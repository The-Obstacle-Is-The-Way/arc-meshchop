# BUG-004: Mask Paths Alignment & None-Safety Issues

**Date**: 2025-12-14
**Status**: ✅ FIXED (commit d940415)
**Priority**: P2 (Low impact - default config works correctly)
**Affects**: `experiment/runner.py`, `training/hpo.py`, `data/huggingface_loader.py`

---

## Executive Summary

The `--no-require-mask` code path has robustness issues:

| Component | Current State | Impact |
|-----------|---------------|--------|
| CLI download | ✅ Correctly writes aligned lists with None | None |
| CLI train | ✅ Validates masks, fails fast with clear message | None |
| CLI evaluate | ✅ Validates masks, fails fast with clear message | None |
| experiment runner | ❌ Crashes with TypeError on None | P2 |
| hpo.py | ❌ Crashes with TypeError on None | P2 |
| ARCDatasetInfo.mask_paths | ❌ Filters out None (misaligned accessor) | P3 |
| Test dataset cache | ⚠️ No cache_dir (repeated preprocessing) | P2 |

**Default workflow (with masks) is UNAFFECTED.** These bugs only manifest if someone:
1. Downloads with `--no-require-mask`
2. Then runs `experiment` or `hpo` commands

---

## Validated Issues

### [P2] Issue 1: experiment/runner.py Crashes on None Masks

**Location**: `src/arc_meshchop/experiment/runner.py:466-467`

**Current Code**:
```python
train_dataset = ARCDataset(
    image_paths=[Path(image_paths[i]) for i in train_indices],
    mask_paths=[Path(mask_paths[i]) for i in train_indices],  # ← TypeError if None
    ...
)
```

**Problem**: If `mask_paths[i]` is `None`, `Path(None)` raises:
```
TypeError: expected str, bytes or os.PathLike object, not NoneType
```

**User Experience**: Confusing stack trace instead of actionable error message.

**Same issue at line 507** for test_dataset.

---

### [P2] Issue 2: hpo.py Crashes on None Masks

**Location**: `src/arc_meshchop/training/hpo.py:219`

**Current Code**:
```python
mask_paths = [Path(p) for p in dataset_info["mask_paths"]]  # ← TypeError if None
```

**Problem**: Same as above - `Path(None)` raises TypeError.

---

### [P3] Issue 3: ARCDatasetInfo.mask_paths Accessor Breaks Alignment

**Location**: `src/arc_meshchop/data/huggingface_loader.py:74-76`

**Current Code**:
```python
@property
def mask_paths(self) -> list[Path]:
    """Get all mask paths (excludes samples without masks)."""
    return [s.mask_path for s in self.samples if s.mask_path is not None]
```

**Problem**: This filters out `None` values, so:
- `len(arc_info.image_paths)` = 228
- `len(arc_info.mask_paths)` = 223 (if 5 have no masks)
- **Indexing is no longer aligned**

**Why It Matters**: Any code that assumes `arc_info.mask_paths[i]` corresponds to `arc_info.image_paths[i]` will silently compute wrong results.

**Note**: CLI already works around this at `cli.py:144`:
```python
mask_paths_aligned = [str(s.mask_path) if s.mask_path else None for s in arc_info.samples]
```

---

### [P2] Issue 4: Test Dataset Has No Cache

**Location**: `src/arc_meshchop/experiment/runner.py:505-508`

**Current Code**:
```python
test_dataset = ARCDataset(
    image_paths=[Path(image_paths[i]) for i in test_indices],
    mask_paths=[Path(mask_paths[i]) for i in test_indices],
    # No cache_dir!
)
```

**Problem**: Test preprocessing (256³ resampling) is repeated 10 times per outer fold (once per restart), even though test set is identical.

**Impact**: Adds ~5-10 minutes per restart × 10 restarts × 3 folds = **~2.5-5 hours** of wasted compute for full experiment.

**Note**: Train dataset correctly uses cache at line 468:
```python
cache_dir=self.config.data_dir / "cache" / f"outer_{outer_fold}" / "train",
```

---

### [P3] Issue 5: Per-Subject Scores Use Index, Not Subject ID

**Location**: `src/arc_meshchop/experiment/runner.py:545`

**Current Code**:
```python
subject_indices=list(test_indices),  # Dataset indices
```

**Problem**: For cross-experiment Wilcoxon pairing, dataset indices are fragile if:
- Dataset ordering changes
- Different filtering is applied

**Better**: Store `subject_id` alongside indices for stable pairing.

---

## Proposed Fix Plan

### Fix 1: Add None-Mask Validation to experiment/runner.py

**File**: `src/arc_meshchop/experiment/runner.py`

**After line 444** (after loading dataset_info), add:
```python
# Validate all masks exist (training/evaluation requires ground truth)
mask_paths_raw = [p if p else None for p in mask_paths]
none_mask_count = sum(1 for m in mask_paths_raw if m is None)
if none_mask_count > 0:
    raise ValueError(
        f"{none_mask_count} samples have no lesion masks. "
        "Experiment requires masks for training and evaluation. "
        "Re-download with default --require-mask option."
    )
```

**Then at line 466** and **507**, the existing `Path(mask_paths[i])` is safe.

---

### Fix 2: Add None-Mask Validation to hpo.py

**File**: `src/arc_meshchop/training/hpo.py`

**Replace line 219** with:
```python
# Validate all masks exist (HPO requires ground truth)
mask_paths_raw = dataset_info["mask_paths"]
none_mask_count = sum(1 for m in mask_paths_raw if m is None)
if none_mask_count > 0:
    raise ValueError(
        f"{none_mask_count} samples have no lesion masks. "
        "HPO requires masks for training and evaluation. "
        "Re-download with default --require-mask option."
    )
mask_paths = [Path(p) for p in mask_paths_raw]
```

---

### Fix 3: Fix ARCDatasetInfo.mask_paths Accessor

**File**: `src/arc_meshchop/data/huggingface_loader.py`

**Option A (Recommended)**: Change `mask_paths` to be aligned, add new filtered accessor:

```python
@property
def mask_paths(self) -> list[Path | None]:
    """Get all mask paths, aligned with image_paths (may contain None)."""
    return [s.mask_path for s in self.samples]

@property
def mask_paths_present(self) -> list[Path]:
    """Get mask paths for samples that have masks (filtered, NOT aligned)."""
    return [s.mask_path for s in self.samples if s.mask_path is not None]
```

**Option B (Conservative)**: Keep current behavior, add aligned accessor:

```python
@property
def mask_paths_aligned(self) -> list[Path | None]:
    """Get all mask paths, aligned with image_paths (may contain None)."""
    return [s.mask_path for s in self.samples]
```

**Recommendation**: Option A is cleaner but requires updating any code that assumes `mask_paths` is all non-None. Option B is safer for backwards compatibility.

---

### Fix 4: Add Cache to Test Dataset

**File**: `src/arc_meshchop/experiment/runner.py`

**At line 505-508**, change to:
```python
test_dataset = ARCDataset(
    image_paths=[Path(image_paths[i]) for i in test_indices],
    mask_paths=[Path(mask_paths[i]) for i in test_indices],
    cache_dir=self.config.data_dir / "cache" / f"outer_{outer_fold}" / "test",
)
```

Test cache is shared across all 10 restarts (same test set per fold).

---

### Fix 5: Store Subject IDs for Stable Pairing (Optional Enhancement)

**File**: `src/arc_meshchop/experiment/runner.py`

**At line 545**, also store subject IDs:
```python
subject_indices=list(test_indices),
subject_ids=[dataset_info["subject_ids"][i] for i in test_indices],  # NEW
```

**Also update RunResult dataclass** to include `subject_ids: list[str]`.

---

## Implementation Checklist

- [ ] Fix 1: Add validation in runner.py (after line 444)
- [ ] Fix 2: Add validation in hpo.py (replace line 219)
- [ ] Fix 3: Update ARCDatasetInfo.mask_paths accessor
- [ ] Fix 4: Add cache_dir to test dataset in runner.py
- [ ] Fix 5: Store subject_ids in RunResult (optional)
- [ ] Add tests for None-mask error messages
- [ ] Update CLI download code if Fix 3 changes accessor behavior
- [ ] Run full CI

---

## Testing Plan

### Test 1: Verify None-Mask Validation
```python
def test_runner_rejects_none_masks():
    """Runner should fail fast with clear message if masks are None."""
    dataset_info = {"mask_paths": ["/path/a.nii", None, "/path/c.nii"], ...}
    # Should raise ValueError with actionable message
```

### Test 2: Verify Test Cache Reduces Runtime
```bash
# Time first restart (creates cache)
time uv run arc-meshchop experiment --dry-run --restarts 1

# Time with existing cache (should be much faster)
time uv run arc-meshchop experiment --dry-run --restarts 1
```

### Test 3: Verify Accessor Alignment
```python
def test_mask_paths_aligned_with_image_paths():
    """mask_paths should have same length as image_paths."""
    arc_info = load_arc_from_huggingface(..., require_lesion_mask=False)
    assert len(arc_info.mask_paths) == len(arc_info.image_paths)
```

---

## Why This Isn't Blocking Training

1. **Default config requires masks**: `require_lesion_mask=True` by default
2. **CLI already handles this**: `train` and `evaluate` commands validate masks
3. **Paper replication uses default**: We need masks for supervised training

These fixes improve robustness for edge cases (HPO, experiment runner with unusual configs).

---

## References

- Senior review feedback (2025-12-14)
- `src/arc_meshchop/cli.py:276-286` (correct validation pattern)
- `src/arc_meshchop/experiment/runner.py:441-468` (missing validation)
- `src/arc_meshchop/training/hpo.py:212-222` (missing validation)
- `src/arc_meshchop/data/huggingface_loader.py:74-76` (misaligned accessor)

---

## Last Updated

2025-12-14
