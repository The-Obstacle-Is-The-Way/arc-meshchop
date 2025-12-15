# Local Spec 05: Upstream Dataset Fixes

> **Status:** ✅ IMPLEMENTED
>
> **Goal:** Enable exact paper replication with 223 SPACE-only samples (excluding 5 TSE).

---

## Overview

This spec documents the required upstream fixes to enable exact paper replication:

1. **Add `t2w_acquisition` column to HuggingFace dataset**
2. **Update loader to filter by acquisition type**
3. **Document the 5 TSE exclusions in dataset card**

Currently training proceeds with 228 samples (includes 5 TSE), which is clinically valid 
but not exact paper replication.

---

## 1. Current State

### 1.1 Sample Counts

| Source | Count | Composition |
|--------|-------|-------------|
| **Paper** | 224 | 115 SPACE 2x + 109 SPACE no-accel (excludes 5 TSE) |
| **HuggingFace** | 228 | 115 + 108 + 5 TSE (all included) |
| **Discrepancy** | +4 | 4 extra samples (TSE included + 1 missing SPACE mask) |

### 1.2 OpenNeuro Verification

Verified against OpenNeuro ds004884 commit `0885e5939abc8f909a175dd782369b7afc3fdd08`:

**5 TSE samples (should be excluded for paper parity):**

| Subject | Session | Acquisition |
|---------|---------|-------------|
| sub-M2002 | ses-1441 | `acq-tse3` |
| sub-M2007 | ses-6330 | `acq-tse3` |
| sub-M2015 | ses-409 | `acq-tse3` |
| sub-M2016 | ses-2721 | `acq-tse3` |
| sub-M2017 | ses-1141 | `acq-tse3` |

Full OpenNeuro paths follow pattern: `derivatives/lesion_masks/{subject}/{session}/anat/{subject}_{session}_acq-tse3_run-*_T2w_desc-lesion_mask.nii.gz`

**1 missing SPACE mask:**
- `sub-M2039/ses-1222` has `acq-spc3` T2w but NO lesion mask directory

---

## 2. Root Cause Analysis

### 2.1 neuroimaging-go-brrrr Bug

Location: `src/bids_hub/datasets/arc.py`

**Line 146:** Finds T2w files but doesn't parse `acq-*` entity from BIDS filename
```python
t2w_path = find_single_nifti(session_dir / "anat", "*_T2w.nii.gz")
# Should extract: acq-spc3p2, acq-spc3, acq-tse3 from filename
```

**Lines 226-246:** Schema missing `t2w_acquisition` field
```python
def get_arc_features() -> Features:
    return Features({
        "subject_id": Value("string"),
        "session_id": Value("string"),
        "t1w": Nifti(),  # Optional
        "t2w": Nifti(),  # MISSING: t2w_acquisition
        # ...
    })
```

### 2.2 Required Fix in neuroimaging-go-brrrr

```python
# In build_arc_file_table():
def _parse_acquisition_type(t2w_path: Path) -> str:
    """Extract acquisition type from BIDS filename.

    Examples:
        sub-M2002_ses-1441_acq-spc3p2_run-4_T2w.nii.gz -> space_2x
        sub-M2002_ses-1441_acq-spc3_run-4_T2w.nii.gz -> space_no_accel
        sub-M2002_ses-1441_acq-tse3_run-4_T2w.nii.gz -> turbo_spin_echo
    """
    name = t2w_path.stem.lower()
    if "acq-spc3p2" in name:
        return "space_2x"
    elif "acq-spc3" in name:
        return "space_no_accel"
    elif "acq-tse3" in name:
        return "turbo_spin_echo"
    return "unknown"

# In get_arc_features():
def get_arc_features() -> Features:
    return Features({
        # ... existing fields ...
        "t2w_acquisition": Value("string"),  # ADD THIS
    })
```

---

## 3. Implementation Steps

### 3.1 Upstream (neuroimaging-go-brrrr)

- [x] Add `_parse_acquisition_type()` function to extract `acq-*` from BIDS filenames
- [x] Add `t2w_acquisition` field to `get_arc_features()` schema
- [x] Update `build_arc_file_table()` to populate the new field
- [x] Re-upload HuggingFace dataset with acquisition metadata
- [x] Update dataset card to document:
  - Acquisition type field meanings
  - 5 TSE samples that paper excludes
  - Missing mask for sub-M2039/ses-1222

### 3.2 This Codebase (arc-meshchop)

After upstream is fixed:

- [x] Update `load_arc_from_huggingface()` to use `t2w_acquisition` column
- [x] Change default filtering: `exclude_turbo_spin_echo=True`
- [x] Add `--paper-parity` flag to CLI for strict 223-sample mode
- [x] Remove workaround code that defaults unknown acquisitions

---

## 4. Testing Plan

### 4.1 Verification Tests

```python
def test_paper_parity_sample_count():
    """Verify exactly 223 samples with paper-parity filtering."""
    info = load_arc_from_huggingface(
        exclude_turbo_spin_echo=True,
        strict_t2w=True,
    )
    assert len(info.samples) == 223  # 115 + 108, not 224 (mask missing)

def test_acquisition_type_distribution():
    """Verify correct acquisition type counts."""
    info = load_arc_from_huggingface(exclude_turbo_spin_echo=False)

    acq_counts = {}
    for sample in info.samples:
        acq = sample.acquisition_type
        acq_counts[acq] = acq_counts.get(acq, 0) + 1

    assert acq_counts["space_2x"] == 115
    assert acq_counts["space_no_accel"] == 108
    assert acq_counts["turbo_spin_echo"] == 5
```

---

## 5. Historical Workaround (No Longer Needed)

This section documents the workaround that was in place before the upstream fix.

**RESOLVED:** The HuggingFace dataset now has `t2w_acquisition` column and the loader uses it.

Previously:
1. Accepted 228 samples (TSE included)
2. Defaulted unknown acquisitions to `space_no_accel`
3. Logged warnings about discrepancy

Now:
1. Default mode uses 223 samples (excludes TSE, matches paper)
2. `--paper-parity` flag enables strict count verification
3. `--include-tse` flag allows 228 samples for maximum data utilization

---

## 6. References

- KNOWN_ISSUES.md: Issue #1 (Acquisition Type Metadata) and Issue #2 (Sample Count)
- OpenNeuro ds004884: [ds004884](https://openneuro.org/datasets/ds004884)
- Paper Section 2: "We utilized SPACE sequences with x2 in-plane acceleration (115 scans)
 and without acceleration (109 scans), while excluding the turbo-spin echo T2-weighted se
quences (5 scans)"

---

## Last Updated

2025-12-14
