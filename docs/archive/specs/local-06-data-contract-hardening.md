# SPEC: Data Contract Hardening

> **Status:** ✅ IMPLEMENTED (commit d940415)
> **Priority:** P2 (Robustness + performance hardening)
> **Author:** Senior Review
> **Date:** 2025-12-14

---

## Executive Summary

This spec defines the ONE correct implementation for data contract consistency across the codebase. No options.

This is primarily **robustness hardening**: the default supervised training/evaluation workflow (which requires masks) works today, but the current API has a footgun and multiple call sites will crash with unhelpful `TypeError` if `dataset_info.json` contains `null` masks.

**Core Principle:** All list accessors on a data structure must have identical indexing semantics.

---

## Problem Statement

The `ARCDatasetInfo` dataclass has inconsistent accessor behavior:

| Property | Current Behavior | Aligned? |
|----------|------------------|----------|
| `image_paths` | Returns all paths | ✅ Yes |
| `mask_paths` | **Filters out None** | ❌ NO |
| `lesion_volumes` | Returns all volumes | ✅ Yes |
| `acquisition_types` | Returns all types | ✅ Yes |
| `subject_ids` | Returns all IDs | ✅ Yes |

This inconsistency is a **design defect**. When `image_paths[i]` refers to sample `i`, but `mask_paths[i]` does NOT refer to sample `i` (because some were filtered), the API violates the principle of least surprise and creates subtle bugs.

Additionally, code that consumes datasets with None masks crashes with unhelpful `TypeError` instead of actionable error messages.

---

## Design Principles

These are non-negotiable:

1. **Consistent Contracts**: All list properties on a dataclass have the same length and indexing
2. **Explicit Types**: Type signatures communicate nullability (`list[Path | None]` vs `list[Path]`)
3. **Fail-Fast Validation**: Invalid states are caught early with actionable error messages
4. **No Hidden Footguns**: The API surface doesn't allow silent misuse
5. **Canonical Data Flow**: One path for data, validated at boundaries

---

## Implementation Specification

### 1. Fix `ARCDatasetInfo.mask_paths` to be Aligned

**File:** `src/arc_meshchop/data/huggingface_loader.py`

**Current (WRONG):**
```python
@property
def mask_paths(self) -> list[Path]:
    """Get all mask paths (excludes samples without masks)."""
    return [s.mask_path for s in self.samples if s.mask_path is not None]
```

**Correct:**
```python
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
```

**Rationale:**
- `mask_paths` is now consistent with all other accessors
- Type signature `list[Path | None]` explicitly communicates nullability
- `samples_with_masks` provides filtered access when needed (returns full samples, not just paths)
- `num_with_masks` provides quick count without iteration

---

### 2. Define the `dataset_info.json` Contract + Add Validation Helpers

**File:** `src/arc_meshchop/data/huggingface_loader.py`

**SSOT Contract:** `dataset_info.json` is the interface between `download` and all training/evaluation commands. It must satisfy:

- Required keys: `image_paths`, `mask_paths`, `lesion_volumes`, `acquisition_types`, `subject_ids`
- All list fields are **the same length** and share indexing semantics
- `mask_paths[i]` may be `null` *only* if downloaded with `--no-require-mask`
- `num_samples` (if present) must match the list lengths

Add this function after the `ARCDatasetInfo` class:

```python
from collections.abc import Mapping, Sequence
from typing import Any, cast

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

    acquisition_types_any = _require_list(dataset_info["acquisition_types"], key="acquisition_types")
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
```

---

### 3. Add Validation to experiment/runner.py

**File:** `src/arc_meshchop/experiment/runner.py`

**Location:** Immediately after loading `dataset_info`, before any indexing/splits, add:

```python
from arc_meshchop.data.huggingface_loader import parse_dataset_info, validate_masks_present

# ... existing code to load dataset_info ...

image_paths, mask_paths_raw, lesion_volumes, acquisition_types, subject_ids = parse_dataset_info(
    dataset_info,
    context="Experiment training/evaluation",
)

# Validate masks (experiment requires ground truth for all samples)
mask_paths = validate_masks_present(mask_paths_raw, context="Experiment training/evaluation")
```

The rest of the code remains unchanged, but now:
- list indexing is safe (length-aligned)
- `Path(mask_paths[i])` is safe (no `None`)
- `subject_ids` is available for stable pairing (see Step 6)

---

### 4. Add Validation to training/hpo.py

**File:** `src/arc_meshchop/training/hpo.py`

**Location:** Immediately after loading `dataset_info`, add:

```python
from arc_meshchop.data.huggingface_loader import parse_dataset_info, validate_masks_present

# ... existing code to load dataset_info ...

image_paths_raw, mask_paths_raw, lesion_volumes, acquisition_types, _subject_ids = parse_dataset_info(
    dataset_info,
    context="Hyperparameter optimization",
)
image_paths = [Path(p) for p in image_paths_raw]
mask_paths_raw = validate_masks_present(mask_paths_raw, context="Hyperparameter optimization")
mask_paths = [Path(p) for p in mask_paths_raw]
```

---

### 5. Add Cache to Test Dataset in runner.py

**File:** `src/arc_meshchop/experiment/runner.py`

**Location:** Lines 505-508, change to:

```python
test_dataset = ARCDataset(
    image_paths=[Path(image_paths[i]) for i in test_indices],
    mask_paths=[Path(mask_paths[i]) for i in test_indices],
    cache_dir=self.config.data_dir / "cache" / f"outer_{outer_fold}" / "test",
)
```

**Rationale:** Test set is identical across all 10 restarts within an outer fold. Caching saves ~5 hours of redundant preprocessing in a full experiment.

---

### 6. Store Subject IDs in RunResult

**File:** `src/arc_meshchop/experiment/runner.py`

**Location:** Update `RunResult` dataclass (around line 58):

```python
@dataclass
class RunResult:
    """Result of a single training run."""

    outer_fold: int
    restart: int
    seed: int
    test_dice: float
    test_avd: float
    test_mcc: float
    final_train_loss: float
    checkpoint_path: str
    duration_seconds: float
    per_subject_dice: list[float]
    per_subject_avd: list[float]
    per_subject_mcc: list[float]
    subject_indices: list[int]
    subject_ids: list[str]  # NEW: stable identifiers for cross-experiment pairing
```

**Location:** Update result creation (around line 545):

```python
run_result = RunResult(
    outer_fold=outer_fold,
    restart=restart,
    seed=seed,
    test_dice=float(np.mean(dice_scores)) if dice_scores else 0.0,
    test_avd=float(np.mean(avd_scores)) if avd_scores else 0.0,
    test_mcc=float(np.mean(mcc_scores)) if mcc_scores else 0.0,
    final_train_loss=float(train_results["final_train_loss"]),
    checkpoint_path=str(run_dir / "final.pt"),
    duration_seconds=duration,
    per_subject_dice=dice_scores,
    per_subject_avd=avd_scores,
    per_subject_mcc=mcc_scores,
    subject_indices=list(test_indices),
    subject_ids=[subject_ids[i] for i in test_indices],  # NEW
)
```

**Rationale:** Dataset indices are fragile if ordering changes. Subject IDs are stable identifiers that enable reliable cross-experiment Wilcoxon pairing.

---

### 7. Update CLI Download (Already Correct, Verify)

**File:** `src/arc_meshchop/cli.py`

The CLI already works around the accessor bug at line 144. After fixing `ARCDatasetInfo.mask_paths`, this workaround becomes unnecessary but harmless.

**No change needed** — the code is correct and will continue to work.

---

## Test Requirements

### Test 1: Accessor Alignment

**File:** `tests/test_data/test_huggingface_loader.py`

```python
def test_mask_paths_aligned_with_image_paths():
    """mask_paths must have same length as image_paths, even with None values."""
    samples = [
        ARCSample(
            subject_id="sub-001",
            session_id="ses-1",
            image_path=Path("/a.nii.gz"),
            mask_path=Path("/a_mask.nii.gz"),
            lesion_volume=100,
            acquisition_type="space_2x",
        ),
        ARCSample(
            subject_id="sub-002",
            session_id="ses-1",
            image_path=Path("/b.nii.gz"),
            mask_path=None,  # No mask
            lesion_volume=0,
            acquisition_type="space_2x",
        ),
        ARCSample(
            subject_id="sub-003",
            session_id="ses-1",
            image_path=Path("/c.nii.gz"),
            mask_path=Path("/c_mask.nii.gz"),
            lesion_volume=200,
            acquisition_type="space_2x",
        ),
    ]

    info = ARCDatasetInfo(samples=samples)

    # All accessors must have same length
    assert len(info.image_paths) == 3
    assert len(info.mask_paths) == 3  # Was 2 before fix!
    assert len(info.lesion_volumes) == 3
    assert len(info.acquisition_types) == 3
    assert len(info.subject_ids) == 3

    # Indexing must be aligned
    assert info.mask_paths[0] == Path("/a_mask.nii.gz")
    assert info.mask_paths[1] is None
    assert info.mask_paths[2] == Path("/c_mask.nii.gz")

    # samples_with_masks provides filtered access
    assert len(info.samples_with_masks) == 2
    assert info.num_with_masks == 2
```

### Test 2: Validation Helper

**File:** `tests/test_data/test_huggingface_loader.py`

```python
def test_validate_masks_present_passes_when_all_present():
    """validate_masks_present returns list unchanged when no None values."""
    mask_paths = ["/a.nii.gz", "/b.nii.gz", "/c.nii.gz"]
    result = validate_masks_present(mask_paths, context="test")
    assert result == mask_paths


def test_validate_masks_present_raises_on_none():
    """validate_masks_present raises ValueError with actionable message."""
    mask_paths = ["/a.nii.gz", None, "/c.nii.gz", None]

    with pytest.raises(ValueError) as exc_info:
        validate_masks_present(mask_paths, context="Training")

    error_msg = str(exc_info.value)
    assert "Training requires lesion masks" in error_msg
    assert "2 samples have None masks" in error_msg
    assert "indices: [1, 3]" in error_msg
    assert "Re-download" in error_msg
```

### Test 2b: dataset_info Schema Validation

**File:** `tests/test_data/test_huggingface_loader.py`

```python
def test_parse_dataset_info_rejects_length_mismatch():
    dataset_info = {
        "num_samples": 2,
        "image_paths": ["a", "b"],
        "mask_paths": ["a_mask"],  # mismatch
        "lesion_volumes": [1, 2],
        "acquisition_types": ["space_2x", "space_2x"],
        "subject_ids": ["sub-001", "sub-002"],
    }

    with pytest.raises(ValueError, match="list lengths must match"):
        parse_dataset_info(dataset_info, context="test")
```

### Test 3: Runner Validation

**File:** `tests/test_experiment/test_runner_logic.py`

```python
def test_runner_rejects_none_masks():
    """Runner must fail fast with clear message if any masks are None."""
    dataset_info = {
        "num_samples": 2,
        "image_paths": ["/a.nii.gz", "/b.nii.gz"],
        "mask_paths": ["/a_mask.nii.gz", None],  # One missing
        "lesion_volumes": [100, 200],
        "acquisition_types": ["space_2x", "space_2x"],
        "subject_ids": ["sub-001", "sub-002"],
    }

    runner = ExperimentRunner(config)
    with patch.object(runner, "_load_dataset_info", return_value=dataset_info):
        with pytest.raises(ValueError, match="requires lesion masks"):
            runner._run_single(outer_fold=0, restart=0)
```

### Test 4: Subject IDs in RunResult

**File:** `tests/test_experiment/test_runner_logic.py`

```python
def test_run_result_contains_subject_ids():
    """RunResult must include subject_ids for stable cross-experiment pairing."""
    # ... setup mock training ...

    result = runner._run_single(outer_fold=0, restart=0)

    assert hasattr(result, "subject_ids")
    assert len(result.subject_ids) == len(result.subject_indices)
    assert all(isinstance(sid, str) for sid in result.subject_ids)
```

---

## Existing Test Updates

### Update `test_huggingface_loader.py:109`

The existing test creates a sample WITH a mask, so `mask_paths` returns `[Path("/path/mask.nii.gz")]`. This test continues to pass unchanged because the sample has a mask.

**No change needed.**

### Update `test_runner_logic.py` dataset_info mocks

`ExperimentRunner._run_single()` will now call `parse_dataset_info()`, which requires `subject_ids` (and optionally `num_samples`).

Update any tests that patch `_load_dataset_info` (e.g., `tests/test_experiment/test_runner_logic.py`) to include:

```python
{
    "num_samples": 1,
    "image_paths": ["a"],
    "mask_paths": ["a"],
    "lesion_volumes": [1],
    "acquisition_types": ["t1"],
    "subject_ids": ["sub-0000"],
}
```

---

## Migration Notes

### Breaking Change Analysis

The `mask_paths` property return type changes from `list[Path]` to `list[Path | None]`.

**Impact assessment:**
- Internal code: CLI already works around this at line 144
- External code: Anyone using `arc_info.mask_paths` and assuming no None values

**Migration path:** Code that previously used `mask_paths` assuming no None values should either:
1. Use `samples_with_masks` property for filtered access
2. Filter explicitly: `[p for p in info.mask_paths if p is not None]`

This is the correct behavior - the type system now accurately reflects the data.

---

## Implementation Order

Execute in this exact order:

1. **Fix `ARCDatasetInfo` accessor** (huggingface_loader.py) - establishes correct contract
2. **Add `validate_masks_present` helper** (huggingface_loader.py) - reusable validation
3. **Add validation to runner.py** - fail-fast for experiment
4. **Add validation to hpo.py** - fail-fast for HPO
5. **Add test cache to runner.py** - performance optimization
6. **Add subject_ids to RunResult** - data completeness
7. **Add tests** - verify all changes
8. **Run CI** - ensure nothing breaks

---

## Verification

After implementation:

```bash
# All tests pass
make ci

# Specific new tests pass
uv run pytest tests/test_data/test_huggingface_loader.py -v -k "aligned or validate_masks"
uv run pytest tests/test_experiment/test_runner_logic.py -v -k "none_masks or subject_ids"
```

---

## References

- Senior review feedback (2025-12-14)
- Google API Design Guide: "Be consistent" principle
- Python typing best practices: explicit nullability

---

## Approval

This spec is ready for senior review. Once approved, implement it end-to-end (including tests) before merging to `main`.

**Last Updated:** 2025-12-14
