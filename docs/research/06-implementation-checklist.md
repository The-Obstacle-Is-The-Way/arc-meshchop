# Implementation Checklist

> Complete technical checklist for reproducing the MeshNet stroke lesion segmentation paper

## Phase 1: Environment Setup

### Hardware Requirements

**From paper:** NVIDIA A100 GPU with 80 GB memory

**Estimated minimums (not in paper):**
- [ ] GPU with ~4-8 GB VRAM (MeshNet only, FP16) or 80 GB (full benchmark)
- [ ] 32+ GB system RAM (estimated)
- [ ] 50+ GB storage for dataset and checkpoints (estimated)

### Software Dependencies

- [ ] Python 3.8+
- [ ] PyTorch 2.0+ with CUDA support
- [ ] FreeSurfer (for `mri_convert` preprocessing)
- [ ] MONAI (for baseline models)
- [ ] Hydra (configuration management)
- [ ] Optuna (hyperparameter optimization - replaces paper's Orion for Python 3.12+ compatibility)
- [ ] NiBabel (NIfTI file handling)
- [ ] SciPy, NumPy, Pandas

```bash
# Example environment setup
conda create -n meshnet python=3.10
conda activate meshnet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install monai nibabel hydra-core optuna scipy pandas
```

---

## Phase 2: Data Acquisition & Preprocessing

### Download ARC Dataset

- [ ] Access OpenNeuro: https://openneuro.org
- [ ] Download Aphasia Recovery Cohort (ARC) dataset
- [ ] Verify 230 subjects downloaded
- [ ] Verify both T2-weighted images and lesion masks present

### Filter Acquisition Types

- [ ] Identify SPACE sequences with 2x acceleration (n=115)
- [ ] Identify SPACE sequences without acceleration (n=109)
- [ ] **Exclude** turbo-spin echo T2-weighted sequences (n=5)
- [ ] Final count: 224 scans

### Preprocessing Pipeline

For each subject:

- [ ] Resample to 256×256×256 @ 1mm isotropic using `mri_convert`
- [ ] Apply 0-1 intensity normalization (min-max scaling)
- [ ] Apply same transformations to lesion masks
- [ ] Verify output dimensions and value ranges

```bash
# mri_convert example
mri_convert input.nii output_256.nii \
    --conform \
    -vs 1 1 1 \
    -ds 256 256 256
```

### Compute Lesion Statistics

- [ ] Calculate lesion volume (voxel count) for each subject
- [ ] Compute quintile boundaries:
  - Q1: 203 - 33,619 voxels
  - Q2: 33,619 - 67,891 voxels
  - Q3: 67,891 - 128,314 voxels
  - Q4: 128,314 - 363,885 voxels

### Create Cross-Validation Splits

- [ ] Implement nested 3-fold cross-validation
- [ ] Stratify by lesion size quintile
- [ ] Stratify by acquisition type (SPACE with/without acceleration)
- [ ] Outer folds: 3 train/test splits
- [ ] Inner folds: 3 train/val splits per outer fold
- [ ] Save split indices to JSON/CSV

---

## Phase 3: Model Implementation

### MeshNet Architecture

**FROM PAPER:**
- [ ] Implement 10-layer fully convolutional network
- [ ] Implement dilation pattern: `[1, 2, 4, 8, 16, 16, 8, 4, 2, 1]`
- [ ] Support variable channel count (X in MeshNet-X)

**✅ VERIFIED from BrainChop reference (`_references/brainchop/py2tfjs/meshnet.py`):**
- [ ] Kernel size: 3×3×3 (final layer: 1×1×1)
- [ ] BatchNorm3d after each conv (except final)
- [ ] ReLU activation after BatchNorm
- [ ] Dropout3d optional
- [ ] Padding = dilation to maintain 256³ dimensions
- [ ] Bias = True (initialized to 0.0)
- [ ] Xavier normal weight initialization

```python
# Layer specifications (VERIFIED from BrainChop)
Layer 1:  Conv3d(1, X, 3, d=1, p=1) + BN + ReLU
Layer 2:  Conv3d(X, X, 3, d=2, p=2) + BN + ReLU
Layer 3:  Conv3d(X, X, 3, d=4, p=4) + BN + ReLU
Layer 4:  Conv3d(X, X, 3, d=8, p=8) + BN + ReLU
Layer 5:  Conv3d(X, X, 3, d=16, p=16) + BN + ReLU
Layer 6:  Conv3d(X, X, 3, d=16, p=16) + BN + ReLU
Layer 7:  Conv3d(X, X, 3, d=8, p=8) + BN + ReLU
Layer 8:  Conv3d(X, X, 3, d=4, p=4) + BN + ReLU
Layer 9:  Conv3d(X, X, 3, d=2, p=2) + BN + ReLU
Layer 10: Conv3d(X, 2, 1, d=1, p=0)  # Final: 1×1×1, no BN/ReLU
```

> **VALIDATION:** Use parameter counts (5,682 / 56,194 / 147,474) to verify your architecture matches.
> **⚠️ NOTE:** BrainChop uses OLD 8-layer pattern. Adapt to NEW 10-layer symmetric pattern above.

### Verify Parameter Counts

- [ ] MeshNet-5: ~5,682 parameters
- [ ] MeshNet-16: ~56,194 parameters
- [ ] MeshNet-26: ~147,474 parameters

### Baseline Models (Optional)

If reproducing full benchmark:

- [ ] SegResNet (MONAI)
- [ ] V-Net (MONAI)
- [ ] UNETR (MONAI)
- [ ] Swin-UNETR (MONAI)
- [ ] Residual U-Net (MONAI)
- [ ] MedNeXt-S, -M, -B (official repo)
- [ ] U-MAMBA-BOT, -ENC (official repo)
- [ ] U-KAN (official repo)

---

## Phase 4: Training Configuration

### Data Loading

- [ ] Implement PyTorch Dataset for ARC
- [ ] Load full 256³ volumes (no patching)
- [ ] Batch size = 1 (required for 256³ at full resolution)

> **⚠️ NOT IN PAPER:** Data augmentation is NOT mentioned. For strict reproduction, do NOT use augmentation.

### Loss Function

- [ ] Cross-entropy loss
- [ ] Class weights: `[0.5, 1.0]` (background, lesion)
- [ ] Label smoothing: 0.01

```python
criterion = nn.CrossEntropyLoss(
    weight=torch.tensor([0.5, 1.0]),
    label_smoothing=0.01
)
```

### Optimizer (Baselines)

- [ ] AdamW optimizer
- [ ] Learning rate: 0.001
- [ ] Weight decay: 3e-5
- [ ] Epsilon: 1e-4

### Learning Rate Scheduler

- [ ] OneCycleLR
- [ ] Max LR: 0.001
- [ ] Warmup: 1% of training (pct_start=0.01)
- [ ] Gradual decrease after peak (paper does not specify annealing strategy; PyTorch default is cosine)

### Training Loop

- [ ] Half-precision (FP16) training with GradScaler
- [ ] 50 epochs
- [ ] Layer checkpointing for large models
- [ ] Save best model by validation DICE

---

## Phase 5: Hyperparameter Optimization (MeshNet Only)

### Optuna Setup (replaces paper's Orion)

- [ ] Install Optuna: `pip install optuna`
- [ ] Configure SuccessiveHalvingPruner for ASHA algorithm
- [ ] Define search space:

| Parameter | Distribution | Range |
|-----------|--------------|-------|
| channels | Uniform int | [5, 21] |
| lr | Log-uniform | [1e-4, 4e-2] |
| weight_decay | Log-uniform | [1e-4, 4e-2] |
| bg_weight | Uniform | [0, 1] |
| warmup_pct | Categorical | {0.02, 0.1, 0.2} |
| epochs | Fidelity | [15, 50] |

### Run HPO

- [ ] Run on inner folds of first outer fold only
- [ ] Use ASHA for efficient early stopping
- [ ] Select best hyperparameters by validation DICE

### Apply Optimized Hyperparameters

- [ ] Train MeshNet with optimal hyperparameters on all outer folds
- [ ] **CRITICAL: Use 10 random restarts** - Paper explicitly states "we trained the model with 10 restarts"
- [ ] Select best model from restarts based on validation performance

---

## Phase 6: Evaluation

### Metrics Implementation

- [ ] DICE coefficient
- [ ] Average Volume Difference (AVD)
- [ ] Matthews Correlation Coefficient (MCC)

### Inference

- [ ] Run inference on hold-out test sets
- [ ] Use FP16 for memory efficiency
- [ ] Save predictions for qualitative analysis

### Statistical Testing

- [ ] Wilcoxon signed-rank test (vs. MeshNet-26 reference)
- [ ] Holm-Bonferroni correction for multiple comparisons
- [ ] Report p-values and significance

### Results Reporting

- [ ] Mean ± std for each metric
- [ ] Per-fold results for transparency
- [ ] Parameter efficiency comparisons
- [ ] Qualitative boundary visualization

---

## Phase 7: Deployment Considerations

### Browser Deployment (BrainChop Integration)

- [ ] Export model to ONNX format
- [ ] Quantize weights (INT8 or FP16)
- [ ] Test in TensorFlow.js or ONNX.js
- [ ] Verify memory footprint < browser limits
- [ ] Profile inference time

### Edge Device Deployment

- [ ] Test on target hardware
- [ ] Profile memory and latency
- [ ] Consider MeshNet-5 or -16 for tighter constraints

---

## Verification Checklist

### Architecture Verification

- [ ] Input: (1, 1, 256, 256, 256) produces output: (1, 2, 256, 256, 256)
- [ ] Parameter count matches expected values
- [ ] No NaN/Inf in forward pass

### Training Verification

- [ ] Loss decreases over epochs
- [ ] Validation DICE improves and plateaus
- [ ] No gradient explosion/vanishing
- [ ] Checkpoints save correctly

### Results Verification

- [ ] DICE scores in expected range (0.84-0.88 for MeshNet variants)
- [ ] Statistical tests run correctly
- [ ] Results reproducible with same seeds

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| OOM during training | Enable gradient checkpointing, reduce batch size to 1, use FP16 |
| DICE not improving | Check class weights, verify label encoding, increase epochs |
| Inference OOM | Use FP16, clear cache between batches, use sliding window (fallback) |
| Slow training | Enable mixed precision, use compiled model (PyTorch 2.0), optimize data loading |
| Reproducibility issues | Set all random seeds, use deterministic algorithms |

---

## File Structure Template

```
arc-meshchop/
├── data/
│   ├── raw/                  # Original ARC download
│   ├── processed/            # Preprocessed 256³ volumes
│   └── splits.json           # CV fold assignments
├── src/
│   ├── models/
│   │   └── meshnet.py        # MeshNet implementation
│   ├── data/
│   │   └── dataset.py        # ARC dataset class
│   ├── training/
│   │   ├── train.py          # Training loop
│   │   └── loss.py           # Loss functions
│   ├── evaluation/
│   │   ├── metrics.py        # DICE, AVD, MCC
│   │   └── stats.py          # Statistical tests
│   └── utils/
│       └── preprocessing.py  # mri_convert wrapper
├── configs/
│   ├── model/                # Hydra model configs
│   ├── training/             # Hydra training configs
│   └── experiment/           # Full experiment configs
├── scripts/
│   ├── preprocess.sh         # Data preprocessing
│   ├── train.sh              # Training launcher
│   └── evaluate.sh           # Evaluation launcher
└── experiments/
    └── logs/                 # Training logs and checkpoints
```

---

## References

- Original MeshNet: https://github.com/neuroneural/brainchop (reference implementation)
- ARC Dataset: https://openneuro.org (search for Aphasia Recovery Cohort)
- MONAI: https://monai.io
- Optuna: https://optuna.readthedocs.io (replaces paper's Orion for Python 3.12+ compatibility)
- FreeSurfer: https://surfer.nmr.mgh.harvard.edu
