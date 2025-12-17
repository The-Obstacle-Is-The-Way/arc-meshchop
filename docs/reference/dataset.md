# Dataset & Preprocessing Pipeline

> Reference: "State-of-the-Art Stroke Lesion Segmentation at 1/1000th of Parameters"

## Dataset: Aphasia Recovery Cohort (ARC)

### Overview

| Property | Value |
|----------|-------|
| Source | OpenNeuro (open-source) |
| Subjects | 230 unique individuals |
| Condition | Chronic stroke |
| Modality | T2-weighted MRI |
| Ground Truth | Expert lesion delineations |

### Dataset Source

- **Repository:** OpenNeuro
- **Reference:** Gibson et al., "The aphasia recovery cohort, an open-source chronic stroke repository," Scientific Data, 2024
- **License:** Open access (no ethical approval required for secondary use)

---

## Acquisition Protocol Filtering

The paper uses specific acquisition sequences and excludes others:

### Included Sequences

| Sequence Type | Count | Description |
|---------------|-------|-------------|
| SPACE with 2x acceleration | 115 scans | Sampling Perfection with Application optimized Contrast using different flip angle Evolution |
| SPACE without acceleration | 108 scans | Same sequence, no in-plane acceleration |
| **Total included** | **223 scans** | |

> **Note:** Paper cites 224 (115+109), but OpenNeuro ground truth has 223 (115+108). Likely a typo in the paper.

### Excluded Sequences

| Sequence Type | Count | Reason |
|---------------|-------|--------|
| Turbo-spin echo T2-weighted | 5 scans | Maintain homogeneity in imaging protocols |

---

## Preprocessing Pipeline

### Step 1: Resampling with mri_convert

**Tool:** `mri_convert` from FreeSurfer

**Operations:**
```bash
mri_convert input.nii output.nii \
    --conform \           # Conform to standard orientation
    -vs 1 1 1 \           # Voxel size: 1mm isotropic
    -ds 256 256 256       # Dimensions: 256x256x256
```

**Target Specifications:**
| Property | Value |
|----------|-------|
| Resolution | 1mm x 1mm x 1mm (isotropic) |
| Dimensions | 256 x 256 x 256 voxels |
| Total voxels | 16,777,216 per volume |

### Step 2: Intensity Normalization

**Operation:** 0-1 rescaling (min-max normalization)

```python
normalized = (image - image.min()) / (image.max() - image.min())
```

---

## Data Split Strategy

### Nested Cross-Validation Structure

```
Outer Folds (3 total)
+-- Outer Fold 1
|   +-- Train Set -> Inner Folds (3)
|   |   +-- Inner Fold 1 (train/val)
|   |   +-- Inner Fold 2 (train/val)
|   |   +-- Inner Fold 3 (train/val)
|   +-- Hold-out Test Set
+-- Outer Fold 2
|   +-- [Same structure]
+-- Outer Fold 3
    +-- [Same structure]
```

**Purpose of Nested CV:**
- Outer folds: Unbiased performance estimation
- Inner folds: Hyperparameter tuning without leaking test data
- Robust statistical comparisons between models

### Stratification Strategy

Data splits are stratified by:

1. **Lesion Volume** (quintile-based)
2. **Acquisition Type** (SPACE with/without acceleration)

#### Lesion Volume Quartiles

| Quartile | Voxel Range | Description |
|----------|-------------|-------------|
| Q1 | 203 - 33,619 | Small lesions |
| Q2 | 33,619 - 67,891 | Small-medium |
| Q3 | 67,891 - 128,314 | Medium |
| Q4 | 128,314 - 363,885 | Large lesions |

**Percentile cutoffs:** 0th, 25th, 50th, 75th, 100th

---

## Whole-Brain Processing (Key Distinction)

### Standard Approaches (Not Used)

Most segmentation models use:
- **3D subvolume sampling:** Extract patches (e.g., 64^3, 128^3) from volumes
- **2D slice processing:** Treat each 2D slice independently

### This Paper's Approach

**Full-volume processing:** 256^3 cubes for both training and inference

**Advantages:**
- Complete brain context in every forward pass
- No stitching artifacts
- Consistent lesion boundaries

**Challenges:**
- Requires memory-efficient architecture (hence MeshNet)
- Batch size limited to 1 on most GPUs
- Need half-precision training (CUDA) or FP32 (MPS/CPU)

---

## Ground Truth Labels

| Class | Value | Description |
|-------|-------|-------------|
| Background | 0 | Non-lesion tissue |
| Lesion | 1 | Stroke lesion voxels |

**Note:** Binary segmentation (2-class output)

---

## Data Augmentation

> **NOT IN PAPER:** The paper does NOT mention any data augmentation. For strict reproduction of the paper's results, do NOT use augmentation.

---

## Memory Requirements

| Precision | Single Volume | Batch of 1 |
|-----------|---------------|------------|
| Float32 | 64 MB | 64 MB |
| Float16 | 32 MB | 32 MB |

**Note:** With input + output + gradients, expect 4-6x memory multiplier during training.

---

## References

- ARC Dataset: Gibson et al., Scientific Data, 2024
- mri_convert: Dale et al., "Cortical surface-based analysis: I. segmentation and surface reconstruction," NeuroImage, 1999
- OpenNeuro: https://openneuro.org
