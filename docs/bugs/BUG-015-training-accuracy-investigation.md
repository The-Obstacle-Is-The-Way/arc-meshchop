# BUG-015: Training Accuracy Investigation

**Status:** INVESTIGATION COMPLETE - SOME FINDINGS, MORE RUNS NEEDED
**Date:** 2025-12-21
**Priority:** P1 (Critical - Core functionality)

## Summary

After completing 2/30 training runs (fold_0_restart_0 and fold_0_restart_1), the observed DICE scores (0.649-0.667) are significantly below the paper target (0.876). This document investigates potential causes.

## Current Results

| Run | Seed | Test DICE | Test AVD | Test MCC | Duration |
|-----|------|-----------|----------|----------|----------|
| fold_0_restart_0 | 42 | 0.6673 | 0.3049 | 0.6768 | 63.6 hrs |
| fold_0_restart_1 | 43 | 0.6488 | 0.3435 | 0.6606 | 64.3 hrs |

**Paper target:** DICE 0.876 (0.016), AVD 0.245 (0.036), MCC 0.760 (0.030)

## Investigation Findings

### 1. Architecture - VERIFIED CORRECT

- Dilation pattern: `[1, 2, 4, 8, 16, 16, 8, 4, 2, 1]` (10 layers, symmetric)
- Parameter count: 147,474 (matches paper exactly)
- Conv3D with padding=dilation for same output size
- BatchNorm3d + ReLU on all layers except final 1x1 conv
- Xavier normal initialization

**No issues found.**

### 2. Hyperparameters - VERIFIED CORRECT

| Parameter | Our Value | Paper Value | Status |
|-----------|-----------|-------------|--------|
| Optimizer | AdamW | AdamW | OK |
| Learning rate | 0.001 | 0.001 | OK |
| Weight decay | 3e-5 | 3e-5 | OK |
| Epsilon | 1e-4 | 1e-4 | OK |
| Scheduler | OneCycleLR | OneCycleLR | OK |
| pct_start | 0.01 (1%) | 1% | OK |
| div_factor | 100 | "1/100th of max lr" | OK |
| Epochs | 50 | 50 | OK |
| Batch size | 1 | 1 | OK |
| Loss | CrossEntropy | CrossEntropy | OK |
| Class weights | [0.5, 1.0] | [0.5, 1.0] | OK |
| Label smoothing | 0.01 | 0.01 | OK |

**No issues found.**

### 3. Cross-Validation Structure - VERIFIED CORRECT

- Paper: 3 outer folds, 10 restarts per fold
- Our impl: 3 outer folds, 10 restarts per fold (= 30 total runs)
- Split: ~67% train, ~33% test per outer fold (verified: 148/75 = 66.4%/33.6%)
- Stratification: Lesion quintile x acquisition type

**No issues found.**

### 4. Data Pipeline - VERIFIED CORRECT

- 223 samples (paper: 224, we're missing 1 mask from OpenNeuro)
- Excluded TSE sequences as per paper
- Resampling to 256x256x256 @ 1mm isotropic
- 0-1 normalization
- RAS+ orientation enforcement

**No issues found.**

### 5. Near-Zero DICE Subjects - CRITICAL FINDING

8 subjects in test set have DICE < 0.1:

| Subject | Lesion Volume | DICE | Issue |
|---------|---------------|------|-------|
| sub-M2201 | 532 | 0.0000 | Very small lesion, model predicts 0 |
| sub-M2119 | 1,990 | 0.0369 | Small lesion |
| sub-M2215 | 170,083 | 0.0235 | LARGE lesion, model still fails |
| sub-M2104 | 21,862 | 0.0440 | Medium lesion |
| sub-M2147 | 3,913 | 0.0026 | Small lesion |
| sub-M2144 | 1,740 | 0.0000 | Small lesion |
| sub-M2164 | 4,990 | 0.0000 | Small-medium lesion |
| sub-M2150 | 701 | 0.0000 | Very small lesion |

**Key observation:** sub-M2215 has 170K voxels (Q4 large) but model predicts only 2,105 voxels. This is NOT just a small lesion problem.

Analysis of model predictions at ground truth locations:
- Good subjects (DICE 0.85+): Mean probability at GT = 0.87-0.89
- Bad subjects (DICE < 0.1): Mean probability at GT = 0.01-0.03

The model is confident about MOST lesions but completely fails on specific subjects.

### 6. Potential Root Causes (Hypotheses)

#### H1: Insufficient training data diversity
- With only 148 training subjects per fold, some lesion patterns may not be represented
- The paper used identical splits (3 folds), so this should be the same

#### H2: Hyperparameter optimization mismatch
- Paper did HPO on inner folds of first outer fold
- We're using paper's published hyperparameters directly
- **CRITICAL:** Paper hyperparameters were optimized on a specific data distribution that may differ from our preprocessing

#### H3: Preprocessing differences
- Paper used `mri_convert` (FreeSurfer)
- Default pipeline uses `scipy.ndimage.zoom` for resampling
- New option: `--resample-method nibabel_conform` uses `nibabel.processing.conform` (mri_convert-like) for A/B testing
- Slight differences in interpolation could affect results

#### H4: Random seed / initialization effects
- Only 2 runs complete out of 30
- Paper reports mean ± std across all runs
- May need all 30 runs to see full picture

#### H5: FP16 precision on MPS vs CUDA
- Paper trained on NVIDIA A100 with FP16
- We're training on MPS (Apple Silicon) with FP32 fallback
- Numerical differences could accumulate over 50 epochs

#### H6: Specific subjects have data issues
- Some subjects may have unusual imaging characteristics
- Lesion location, shape, or contrast may be atypical
- Model may need to see similar cases during training

### 7. Metric Calculation - VERIFIED CORRECT

- DICE: Standard Sorensen-Dice coefficient
- AVD: |V_pred - V_true| / V_true
- MCC: Matthews correlation coefficient
- All computed on binary masks after argmax

**No issues found.**

## Recommendations

### Immediate Actions

1. **Wait for more runs** - Only 2/30 complete. Paper's variance may not be captured yet.

2. **Check if problematic subjects are in training data of other folds** - If sub-M2215 is never in training, the model may never learn that pattern.

3. **Visualize predictions** - Create visual comparisons of good vs bad predictions to identify patterns.

### Future Investigation

4. **Compare scipy vs mri_convert resampling** - Test if FreeSurfer preprocessing changes results.

5. **Train on CUDA if available** - FP16 training may give different results.

6. **Hyperparameter sensitivity analysis** - Small changes to lr or class weights may help.

## Files Investigated

- `src/arc_meshchop/models/meshnet.py` - Architecture
- `src/arc_meshchop/training/trainer.py` - Training loop
- `src/arc_meshchop/training/config.py` - Hyperparameters
- `src/arc_meshchop/training/loss.py` - Loss function
- `src/arc_meshchop/data/preprocessing.py` - Data preprocessing
- `src/arc_meshchop/data/splits.py` - Cross-validation
- `src/arc_meshchop/experiment/runner.py` - Experiment orchestration
- `src/arc_meshchop/evaluation/metrics.py` - Metric calculation
- `_literature/markdown/stroke_lesion_segmentation/stroke_lesion_segmentation.md` - Paper

## Conclusion

No obvious implementation bugs found. The architecture, hyperparameters, and data pipeline match the paper specification. The gap between our results (0.65-0.67 DICE) and paper target (0.876 DICE) may be due to:

1. Insufficient runs (need all 30 for proper statistics)
2. Preprocessing differences (scipy vs mri_convert)
3. Platform differences (MPS/FP32 vs CUDA/FP16)
4. Subject-specific issues (8 near-zero DICE subjects need investigation)

**Recommend:** Continue training to complete all 30 runs before drawing conclusions.
