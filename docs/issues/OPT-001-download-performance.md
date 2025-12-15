# OPT-001: Download Performance Optimization

> **Status:** DEFERRED (P3 - Post-Training Polish)
> **Impact:** First-run download takes ~25-30 minutes instead of ~5 minutes
> **Severity:** Low (only affects first run, cached afterward)

---

## Problem

The `arc-meshchop download` command takes ~25-30 minutes on first run due to:

1. **HuggingFace returns in-memory images** (`Nifti1ImageWrapper`)
2. **Must save each to disk** via `nib.save()` for training
3. **Lesion volume computed eagerly** during download

### Observed Performance

```
2025-12-14 20:12:10 | Loaded 902 sessions from HuggingFace
2025-12-14 20:15:13 | Processed 100/902 sessions, extracted 17 samples  (3 min)
2025-12-14 20:20:11 | Processed 200/902 sessions, extracted 38 samples  (5 min)
```

Rate: ~100 sessions / 5 minutes = 20 sessions/min
Total: 902 sessions / 20 per min = ~45 minutes worst case

Actual: ~25-30 minutes (most sessions are filtered out early)

---

## Root Cause

In `huggingface_loader.py`:

### 1. NIfTI Caching (lines 582-622)

```python
# For Nifti1ImageWrapper, we must save to disk
if hasattr(nifti_obj, "get_fdata"):
    # ... hash computation ...
    if not cache_path.exists():
        nib.save(nifti_obj, cache_path)  # SLOW: ~2-5 seconds per file
```

### 2. Lesion Volume Computation (lines 732-762)

```python
def _compute_lesion_volume_from_nifti(nifti_obj):
    data = nifti_obj.get_fdata()  # SLOW: loads entire 3D volume
    return int((data > 0).sum())
```

---

## Potential Optimizations

### Option A: Parallel Caching (Medium Effort)

Use `concurrent.futures.ThreadPoolExecutor` to cache multiple NIfTIs in parallel.

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(_cache_sample, row) for row in dataset]
    samples = [f.result() for f in futures if f.result()]
```

**Expected speedup:** 3-4x

### Option B: Lazy Lesion Volume (Low Effort)

Skip lesion volume computation during download, compute lazily on first access.

```python
@cached_property
def lesion_volume(self) -> int:
    if self._lesion_volume is None:
        self._lesion_volume = _compute_from_mask(self.mask_path)
    return self._lesion_volume
```

**Expected speedup:** ~20% (saves one `get_fdata()` per sample)

### Option C: Upstream Fix (High Effort, Best Long-Term)

Fix neuroimaging-go-brrrr to return file paths instead of `Nifti1ImageWrapper`:

- Use `streaming=False` and return actual cached paths
- Requires changes to `bids_hub` NIfTI feature type

**Expected speedup:** 10x+ (no re-caching needed)

---

## Current Workaround

None needed. The cache persists, so subsequent runs are instant:

```bash
# First run: ~25-30 minutes
uv run arc-meshchop download --paper-parity

# Second run: ~5 seconds (reads dataset_info.json)
uv run arc-meshchop download --paper-parity
```

---

## Decision

**DEFERRED** - Not blocking training. Optimize post-paper-replication if needed.

The 25-30 minute download is acceptable for a one-time setup cost. The cache at
`/var/folders/.../arc_nifti_cache/` persists across runs.

---

## Related

- neuroimaging-go-brrrr lazy loading discussion
- HuggingFace datasets `Nifti()` feature type behavior
- `_get_nifti_path()` caching logic

---

## Last Updated

2025-12-14
