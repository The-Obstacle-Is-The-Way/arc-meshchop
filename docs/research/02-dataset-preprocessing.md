# Dataset & Preprocessing Pipeline

> Extracted from: "State-of-the-Art Stroke Lesion Segmentation at 1/1000th of Parameters"

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
| SPACE without acceleration | 109 scans | Same sequence, no in-plane acceleration |
| **Total included** | **224 scans** | |

### Excluded Sequences

| Sequence Type | Count | Reason |
|---------------|-------|--------|
| Turbo-spin echo T2-weighted | 5 scans | Maintain homogeneity in imaging protocols |

**Note:** Filtering is critical for reproducibility - mixed acquisition protocols can introduce confounding variables.

---

## Preprocessing Pipeline

### Step 1: Resampling with mri_convert

**Tool:** `mri_convert` from FreeSurfer

**Operations:**
```bash
mri_convert input.nii output.nii \
    --conform \           # Conform to standard orientation
    -vs 1 1 1 \           # Voxel size: 1mm isotropic
    -ds 256 256 256       # Dimensions: 256×256×256
```

**Target Specifications:**
| Property | Value |
|----------|-------|
| Resolution | 1mm × 1mm × 1mm (isotropic) |
| Dimensions | 256 × 256 × 256 voxels |
| Total voxels | 16,777,216 per volume |

### Step 2: Intensity Normalization

**Operation:** 0-1 rescaling (min-max normalization)

```python
# Pseudocode
normalized = (image - image.min()) / (image.max() - image.min())
```

**Purpose:**
- Normalize intensity values across different scanners/protocols
- Standardize input range for neural network
- Improve training stability

---

## Data Split Strategy

### Nested Cross-Validation Structure

```
Outer Folds (3 total)
├── Outer Fold 1
│   ├── Train Set → Inner Folds (3)
│   │   ├── Inner Fold 1 (train/val)
│   │   ├── Inner Fold 2 (train/val)
│   │   └── Inner Fold 3 (train/val)
│   └── Hold-out Test Set
├── Outer Fold 2
│   └── [Same structure]
└── Outer Fold 3
    └── [Same structure]
```

**Purpose of Nested CV:**
- Outer folds: Unbiased performance estimation
- Inner folds: Hyperparameter tuning without leaking test data
- Robust statistical comparisons between models

### Stratification Strategy

Data splits are stratified by:

1. **Lesion Volume** (quintile-based)
2. **Acquisition Type** (SPACE with/without acceleration)

#### Lesion Volume Quintiles

| Quintile | Voxel Range | Description |
|----------|-------------|-------------|
| Q1 | 203 - 33,619 | Small lesions |
| Q2 | 33,619 - 67,891 | Small-medium |
| Q3 | 67,891 - 128,314 | Medium |
| Q4 | 128,314 - 363,885 | Large lesions |

**Percentile cutoffs:** 0th, 25th, 50th, 75th, 100th

**Why stratify by lesion size:**
- Ensures each fold has representative distribution of lesion sizes
- Prevents model from seeing only small or only large lesions during training
- Critical for generalization across lesion severity

---

## Whole-Brain Processing (Key Distinction)

### Standard Approaches (Not Used)

Most segmentation models use:
- **3D subvolume sampling:** Extract patches (e.g., 64³, 128³) from volumes
- **2D slice processing:** Treat each 2D slice independently

**Problems with these approaches:**
- Loss of 3D context across slice/patch boundaries
- Need for post-processing to stitch results
- Potential boundary artifacts

### This Paper's Approach

**Full-volume processing:** 256³ cubes for both training and inference

**Advantages:**
- Complete brain context in every forward pass
- No stitching artifacts
- Consistent lesion boundaries

**Challenges:**
- Requires memory-efficient architecture (hence MeshNet)
- Batch size limited to 1 on most GPUs
- Need half-precision training

---

## Ground Truth Labels

### Label Structure

| Class | Value | Description |
|-------|-------|-------------|
| Background | 0 | Non-lesion tissue |
| Lesion | 1 | Stroke lesion voxels |

**Note:** Binary segmentation (2-class output)

### Label Source

- Expert manual delineation from the ARC dataset
- Provided as part of the OpenNeuro release

---

## Data Augmentation

The paper does not explicitly detail augmentation strategies. For reproduction, consider:

### Recommended Augmentations (Standard Practice)

| Augmentation | Parameters | Purpose |
|--------------|------------|---------|
| Random flip | All 3 axes | Orientation invariance |
| Random rotation | ±15° per axis | Rotation invariance |
| Intensity scaling | ±10% | Scanner variability |
| Gaussian noise | σ=0.01-0.05 | Noise robustness |
| Elastic deformation | Optional | Shape variability |

**Caution:** 3D augmentations are memory-intensive; apply on-the-fly with GPU.

---

## Implementation Checklist for Preprocessing

- [ ] Download ARC dataset from OpenNeuro
- [ ] Install FreeSurfer (for mri_convert)
- [ ] Filter to SPACE sequences only (exclude turbo-spin echo)
- [ ] Resample all volumes to 256³ @ 1mm isotropic
- [ ] Apply 0-1 intensity normalization
- [ ] Compute lesion volumes for stratification
- [ ] Implement nested 3-fold cross-validation splits
- [ ] Stratify by lesion quintile and acquisition type
- [ ] Save preprocessed data in efficient format (NIfTI or HDF5)

---

## Data Loading Considerations

### Memory Requirements

| Precision | Single Volume | Batch of 1 |
|-----------|---------------|------------|
| Float32 | 64 MB | 64 MB |
| Float16 | 32 MB | 32 MB |

**Note:** With input + output + gradients, expect 4-6x memory multiplier during training.

### Recommended Data Format

```python
# PyTorch DataLoader example
class ARCDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, transform=None):
        self.files = file_list
        self.transform = transform

    def __getitem__(self, idx):
        # Load NIfTI
        img = nib.load(self.files[idx]['image']).get_fdata()
        mask = nib.load(self.files[idx]['mask']).get_fdata()

        # Normalize to 0-1
        img = (img - img.min()) / (img.max() - img.min())

        # Add channel dimension: (256,256,256) → (1,256,256,256)
        img = img[np.newaxis, ...]
        mask = mask[np.newaxis, ...]

        if self.transform:
            img, mask = self.transform(img, mask)

        return torch.FloatTensor(img), torch.LongTensor(mask)
```

---

## References

- ARC Dataset: Gibson et al., Scientific Data, 2024
- mri_convert: Dale et al., "Cortical surface-based analysis: I. segmentation and surface reconstruction," NeuroImage, 1999
- OpenNeuro: https://openneuro.org
