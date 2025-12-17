# Known Issues

## Issue #1: Missing Acquisition Metadata (RESOLVED)

The HuggingFace dataset (`hugging-science/arc-aphasia-bids`) has been updated with the `t2w_acquisition`
column, enabling filtering by MRI sequence type.

- **Status:** Resolved (Upstream fix complete)
- **Resolution:** `t2w_acquisition` column added with values: `space_2x`, `space_no_accel`, `turbo_spin_echo`.
- **Action:** Use `--paper-parity` flag to strictly filter for the paper's cohort.

## Issue #2: 228 vs 224 Sample Count

We have 228 samples; paper used 224.

| Acquisition | OpenNeuro | Paper |
|-------------|-----------|-------|
| SPACE 2x (`acq-spc3p2`) | 115 | 115 |
| SPACE no-accel (`acq-spc3`) | 108 | 109 |
| TSE (`acq-tse3`) | 5 | 0 (excluded) |
| **Total** | **228** | **224** |

The discrepancy:
- Paper cites 224 SPACE samples (115 + 109); OpenNeuro has 223 (115 + 108)
- This is likely a typo in the paper (off by 1 on space_no_accel count)
- Paper excluded 5 TSE samples; we also exclude them by default
- Ground truth: 223 SPACE samples (115 SPACE 2x + 108 SPACE no-accel)

**Solution:**
- **Default mode:** Uses 223 samples (excludes TSE). This matches the paper methodology.
- **Extended mode (`--include-tse`):** Uses 228 samples (includes TSE). Maximizes data utility.
- **Paper Parity mode (`--paper-parity`):** Uses strict 223 samples with count verification.
  - Fails if count doesn't exactly match 223.
  - Recommended for replication studies.

For paper-exact parity, the loader automatically excludes these TSE subjects:
- sub-M2002/ses-1441
- sub-M2007/ses-6330
- sub-M2015/ses-409
- sub-M2016/ses-2721
- sub-M2017/ses-1141


## Fixed Issues (Historical)

These are resolved but documented for reference:

- **Missing Acquisition Metadata** - Fixed: Upstream dataset updated
- **NIfTI cache in temp dir** - Fixed: `cache_dir` now passed through properly
- **Nifti1ImageWrapper objects** - Fixed: in-memory images saved to disk cache
- **MallocStackLogging warning** - Non-issue: cosmetic macOS noise from PyTorch workers


## Current State

- 223 samples ready for training (default, excludes TSE)
- 228 samples available with `--include-tse` flag
- MeshNet-26: 147,474 parameters
- Protocol: 3 outer folds × 10 restarts = 30 runs
- Expected DICE: ~0.876

Verified against OpenNeuro ds004884 (commit `0885e593`) and paper Section 2.
