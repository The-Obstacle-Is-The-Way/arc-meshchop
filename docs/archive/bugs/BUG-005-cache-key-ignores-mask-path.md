# BUG-005: Dataset Cache Key Ignores Mask Path (Stale/Wrong Labels)

> **Severity:** P1 (silent label corruption)
>
> **Status:** FIXED (fix/bug-005-006-007)
>
> **Date:** 2025-12-16
>
> **Affected:** `src/arc_meshchop/data/dataset.py`

## Summary

`ARCDataset` caches **both** the preprocessed image and mask to `.npz`, but the cache key/hash only includes the **image path** and preprocessing params, not the **mask path**. If the mask for an image changes (new annotation version, re-download, different cohort) and the same `cache_dir` is reused, the dataset can silently load an **old cached mask** for a **new mask path**.

## Why This Can Ruin Results

This is a silent failure mode: training/evaluation still runs, but labels may not correspond to the intended ground truth. Any resulting metrics become untrustworthy.

## Evidence

- Cache key hash only includes `image_path` (not `mask_path`): `ARCDataset._cache_key()`
- Cache load returns stored `image` and `mask` without validating against current `mask_path`: `ARCDataset._load_sample()`

## Reproduction (Conceptual)

1. Run training once with `cache_dir=...` to populate `*.npz` caches.
2. Change lesion masks for one or more samples (new mask files, or updated dataset), keeping the same images.
3. Re-run training using the same `cache_dir`.
4. Samples with cached entries will load the **old** mask from cache (no warning).

## Mitigations

- **Operational:** delete/rotate `cache_dir` whenever masks can change.
- **Code fix:** include `mask_path` (or a content hash / mtime+size) in the cache key so caches invalidate when labels change.
