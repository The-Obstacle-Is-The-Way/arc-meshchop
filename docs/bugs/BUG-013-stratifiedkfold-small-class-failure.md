# BUG-013: StratifiedKFold Fails When Any Stratum Has < n_splits (P2)

**Priority:** P2
**Status:** Open
**Impacts:** `arc-meshchop train`, `arc-meshchop experiment`, `arc-meshchop hpo`, `arc-meshchop evaluate`

---

## Symptom

Training or evaluation can fail with:

```
ValueError: The least populated class in y has only 1 members, which is too few.
```

This happens when any (lesion_quintile, acquisition_type) stratum has fewer samples
than `n_splits` (default 3).

---

## Root Cause

`create_stratification_labels()` combines lesion quintile and acquisition type into
labels. `generate_*_cv_splits()` uses `StratifiedKFold` without validating that each
label has at least `n_splits` members.

This can occur when:

- The dataset is filtered (e.g., SPACE-2x only).
- Unknown acquisition types are present.
- A rare lesion size quintile has too few samples in a subset.

---

## Suggested Fix

1. Add a preflight count check for stratification labels.
2. If any class has < n_splits:
   - Option A: Reduce `n_splits` to the minimum viable value.
   - Option B: Merge rare labels into an `other` bucket.
   - Option C: Fall back to non-stratified KFold with a clear warning.
3. Add a `--strict-stratification` flag to force failure if strict parity is required.

---

## Verification

- Add tests with a toy dataset where a stratum has < 3 samples and confirm:
  - Fallback path works, or
  - The error message is explicit and actionable.
- Ensure default ARC dataset still produces stratified splits.
