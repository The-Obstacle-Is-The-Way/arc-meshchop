# OPT-001: Download Performance Optimization

> **Status:** DEFERRED (P3 - Post-Training Polish)
>
> **Impact:** Download takes ~25-45 minutes (I/O + CPU bound)
>
> **Severity:** Low (one-time cost per machine)

---

## Problem

The `arc-meshchop download` command takes ~25-45 minutes due to:

1. **HuggingFace returns in-memory images** (`Nifti1ImageWrapper` from `Nifti(decode=True)`)
2. **Must save each to disk** via `nib.save()` (gzip compression)
3. **Lesion volume computed eagerly** via `get_fdata()` + full-volume scan

### Observed Performance (2025-12-14)

```
20:12:10 | Loaded 902 sessions from HuggingFace
20:15:13 | Processed 100/902 sessions, extracted 17 samples   (+3 min)
20:20:11 | Processed 200/902 sessions, extracted 38 samples   (+5 min)
20:31:44 | Processed 550/902 sessions, extracted 103 samples  (+11 min)
```

Rate: ~100 sessions / 5 minutes = 20 sessions/min
Total: 902 sessions → ~45 minutes worst case
Actual: ~25-35 minutes (many sessions filtered out early)

---

## Root Cause Analysis

### 1. NIfTI Caching (`huggingface_loader.py:582-622`)

```python
# For Nifti1ImageWrapper, we must save to disk
if hasattr(nifti_obj, "get_fdata"):
    # ... hash computation ...
    if not cache_path.exists():
        nib.save(nifti_obj, cache_path)  # SLOW: gzip + disk I/O
```

Each `nib.save()` writes a gzipped NIfTI (~50-200MB uncompressed → ~20-80MB gzipped).
Cost: ~2-5 seconds per file × 2 files per sample × 222 samples = ~15-40 minutes.

### 2. Lesion Volume Computation (`huggingface_loader.py:732-762`)

```python
def _compute_lesion_volume_from_nifti(nifti_obj):
    data = nifti_obj.get_fdata()  # Loads entire 3D volume into RAM
    return int((data > 0).sum())  # Full numpy scan
```

Each `get_fdata()` decompresses and loads the full 256³ volume (~134MB float64).
Cost: ~1-3 seconds per sample × 222 samples = ~4-10 minutes.

### 3. No Short-Circuit on Re-run

**IMPORTANT:** The `download` command has NO check for existing `dataset_info.json`.
Re-running `arc-meshchop download` will:
- Load the HuggingFace dataset again
- Iterate all 902 sessions again
- Recompute lesion volumes (even if NIfTI cache exists)
- Overwrite `dataset_info.json`

---

## Cache Locations

| Component | Location | Purpose |
|-----------|----------|---------|
| HuggingFace dataset | `~/.cache/huggingface/hub/datasets--hugging-science--arc-aphasia-bids/` | Raw blobs (273GB) |
| NIfTI file cache | `data/arc/cache/nifti_cache/` | Extracted .nii.gz files |
| Dataset manifest | `data/arc/dataset_info.json` | Sample paths for training |

**Note:** The temp path `/var/folders/.../arc_nifti_cache/` is only used when
`cache_dir=None` is passed to `_get_nifti_path()`. The CLI always passes
`cache_dir=output_dir / "cache"`, so NIfTIs go to `data/arc/cache/nifti_cache/`.

---

## Actual Fast Path

The "fast path" is NOT re-running download. It's:

```bash
# SLOW: First-time download (~25-45 minutes)
uv run arc-meshchop download --paper-parity

# FAST: Training reads dataset_info.json directly (~instant)
uv run arc-meshchop train --data-dir data/arc
uv run arc-meshchop experiment --data-dir data/arc
```

Once `dataset_info.json` exists, `train` and `experiment` commands read it directly
without touching HuggingFace or recomputing anything.

---

## Potential Optimizations

### Option A: Add Download Short-Circuit (Low Effort, High Impact)

Check if `dataset_info.json` exists and skip re-processing:

```python
def download(...):
    info_path = output_dir / "dataset_info.json"
    if info_path.exists() and not force:
        console.print(f"[yellow]Dataset already exists: {info_path}[/yellow]")
        console.print("Use --force to re-download.")
        return
    # ... existing logic ...
```

**Impact:** Prevents accidental re-downloads. Zero runtime cost.

### Option B: Lazy Lesion Volume (Low Effort, ~20% Speedup)

Skip lesion volume computation during download. Compute lazily or remove entirely
(lesion volume is only used for stratification, which can use file size as proxy).

**Impact:** Saves one `get_fdata()` call per sample (~4-10 minutes).

### Option C: Use `Nifti(decode=False)` (Medium Effort, 10x Speedup)

The HuggingFace `datasets` library supports `Nifti(decode=False)`, which returns
a `{path, bytes}` dict instead of a decoded `Nifti1ImageWrapper`.

If we restructure the loader to:
1. Accept raw bytes or `gzip://...::...` paths
2. Write bytes directly to cache (skip `nib.save()` recompression)
3. Defer `get_fdata()` to training time

**Impact:** Skip most decompression/recompression. Potentially 10x speedup.

### Option D: Parallel Caching (Medium Effort, Uncertain Gains)

Use `ThreadPoolExecutor` or `ProcessPoolExecutor` for parallel I/O.

```python
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(_cache_sample, row) for row in dataset]
```

**Caveat:** `nib.save()` is CPU-bound (gzip compression) and GIL-constrained.
Threads may give modest gains or regress under memory pressure. The real lever
is avoiding recompression (Option C), not parallelizing it.

**Expected:** 1.5-2x speedup (not 3-4x as previously claimed).

---

## Decision

**DEFERRED** - Not blocking training.

The 25-45 minute download is acceptable as a one-time setup cost per machine.
The NIfTI cache at `data/arc/cache/nifti_cache/` persists, and training reads
from `dataset_info.json` without re-downloading.

**Recommended post-training:**
1. Add download short-circuit (Option A) - prevents user error
2. Consider `Nifti(decode=False)` (Option C) - for upstream improvement

---

## Related

- neuroimaging-go-brrrr: Upstream dataset builder
- HuggingFace `datasets.features.Nifti()` feature type
- `_get_nifti_path()` caching logic in `huggingface_loader.py`

---

## Last Updated

2025-12-14 (validated against actual run + code inspection)
