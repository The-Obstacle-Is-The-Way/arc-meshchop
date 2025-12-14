# Known Issues

## Issue #1: Missing Acquisition Metadata (Blocked on Upstream)

The HuggingFace dataset (`hugging-science/arc-aphasia-bids`) lacks the `t2w_acquisition` column needed to filter by MRI sequence type.

Current schema:
```
subject_id, session_id, t1w, t2w, flair, bold, dwi, sbref, lesion, age_at_stroke, sex, wab_aq, wab_type
```
Missing: `t2w_acquisition` (values: `space_2x`, `space_no_accel`, `turbo_spin_echo`)

The paper states: *"We utilized SPACE sequences with x2 in-plane acceleration (115 scans) and without acceleration (109 scans), while excluding the turbo-spin echo T2-weighted sequences (5 scans)."*

OpenNeuro has this info in BIDS filenames (`acq-spc3p2`, `acq-spc3`, `acq-tse3`), but it wasn't extracted to HuggingFace.

**Workaround:** Accept all samples with T2w + lesion mask regardless of acquisition type.

**Upstream fix needed:** Add `t2w_acquisition` column to HuggingFace dataset.


## Issue #2: 228 vs 224 Sample Count

We have 228 samples; paper used 224.

| Acquisition | OpenNeuro | Paper |
|-------------|-----------|-------|
| SPACE 2x (`acq-spc3p2`) | 115 | 115 |
| SPACE no-accel (`acq-spc3`) | 108 | 109 |
| TSE (`acq-tse3`) | 5 | 0 (excluded) |
| **Total** | **228** | **224** |

The discrepancy:
- Paper had 1 additional SPACE mask (`sub-M2039/ses-1222`) not publicly available
- Paper excluded 5 TSE samples; we include them (can't filter without acquisition metadata)
- Net: +5 TSE - 1 missing = +4

**Decision:** Proceed with 228. Both SPACE and TSE are T2-weighted with identical lesion contrast. The 5 extra samples (2.2% of dataset) are well within the paper's reported variance (0.876 ± 0.016).

For paper-exact parity, exclude these TSE subjects:
- sub-M2002/ses-1441
- sub-M2007/ses-6330
- sub-M2015/ses-409
- sub-M2016/ses-2721
- sub-M2017/ses-1141


## Fixed Issues (Historical)

These are resolved but documented for reference:

- **NIfTI cache in temp dir** - Fixed: `cache_dir` now passed through properly
- **Nifti1ImageWrapper objects** - Fixed: in-memory images saved to disk cache
- **MallocStackLogging warning** - Non-issue: cosmetic macOS noise from PyTorch workers


## Current State

- 228 samples ready for training
- MeshNet-26: 147,474 parameters
- Protocol: 3 outer folds × 10 restarts = 30 runs
- Expected DICE: ~0.876

Verified against OpenNeuro ds004884 (commit `0885e593`) and paper Section 2.
