# BUG-001: Codebase Audit Findings

**Date**: 2025-12-13
**Status**: VALIDATED
**Impact**: 1 P0, 2 P1, 3 P2, 1 P3

## Summary

An external audit identified 7 issues requiring attention. All claims have been validated against the actual codebase and data.

**Training Impact**: Current training (epoch 6/50) can continue. The P0 issue affects only 1 sample (sub-M2269, 0.44% of data). Other issues don't affect current training.

---

## [P0] Mask Binarization Zeros Valid Lesions

**CRITICAL - VALIDATED**

### Location
- `src/arc_meshchop/data/preprocessing.py:237`
- `src/arc_meshchop/data/huggingface_loader.py:620`

### Problem
Current binarization uses `mask_resampled > 0.5`, but nibabel's `get_fdata()` can apply implicit scaling that results in mask values < 0.5 for actual lesion voxels.

### Evidence
```python
# sub-M2269_ses-767 mask analysis:
Raw uint8 values:      [0, 255]  # Correct binary mask
After get_fdata():     [0, 0.017]  # Scaled down!
Non-zero voxels:       216,912
Voxels > 0.5:          0  # ALL LESION VOXELS LOST
```

### Root Cause
The NIfTI file has unusual scl_slope header value causing nibabel to scale uint8 [0,255] to float [0,0.017] instead of [0,1].

### Impact
- 1 out of 228 masks affected (sub-M2269_ses-767)
- 0.44% data corruption
- Model learns incorrect labels for this sample
- Also breaks lesion volume calculation for stratification

### Fix
```python
# preprocessing.py:237
# OLD: mask_normalized = (mask_resampled > 0.5).astype(np.float32)
# NEW:
mask_normalized = (mask_resampled > 0).astype(np.float32)

# huggingface_loader.py:620
# OLD: return int(np.sum(data > 0.5))
# NEW:
return int(np.sum(data > 0))
```

---

## [P1] Missing RAS+ Canonical Orientation

**HIGH - VALIDATED**

### Location
- `src/arc_meshchop/data/preprocessing.py:43`

### Problem
`load_nifti()` loads data "as-is" without enforcing canonical RAS+ orientation. Some ARC samples have non-standard orientations.

### Evidence
```python
# 2 samples with non-RAS orientation:
sub-M2105_ses-964: ('P', 'S', 'R')  # Should be ('R', 'A', 'S')
sub-M2269_ses-767: ('P', 'S', 'R')
```

### Impact
- 2 out of 228 samples affected (0.88%)
- Network may learn inconsistent spatial features
- Diverges from paper's `mri_convert --conform` preprocessing

### Fix
```python
# preprocessing.py:43
def load_nifti(path: Path | str) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float64]]:
    nii = nib.load(str(path))
    # ADD: Enforce canonical RAS+ orientation
    nii = nib.as_closest_canonical(nii)
    data = np.asarray(nii.get_fdata(), dtype=np.float32)
    affine = np.asarray(nii.affine, dtype=np.float64)
    return data, affine
```

---

## [P1] Resume Does Not Restore Training State

**HIGH - VALIDATED**

### Location
- `src/arc_meshchop/training/trainer.py:166, 393-410`

### Problem
`load_checkpoint()` only loads scheduler state if `self.scheduler` already exists, but scheduler is created in `train()` after checkpoint loading. Also, `train()` always starts from epoch 0.

### Evidence
```python
# trainer.py:124 - scheduler is None at init
self.scheduler = None

# trainer.py:393 - load_checkpoint only loads scheduler if exists
if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
    self.scheduler.load_state_dict(...)

# trainer.py:166 - always starts from 0
for epoch in range(self.config.epochs):
```

### Impact
- Resume functionality is broken
- OneCycleLR schedule resets on resume
- Epoch counter restarts from 0
- Does NOT affect current training (no resume used)

### Fix
1. Store scheduler state on load even if scheduler is None
2. Defer scheduler state loading to after scheduler creation in `train()`
3. Start training from `self.state.epoch + 1` instead of 0
4. Ensure `last_epoch` is set correctly for OneCycleLR

---

## [P2] Insecure Checkpoint Deserialization

**MEDIUM - VALIDATED**

### Location
- `src/arc_meshchop/training/trainer.py:393`

### Problem
`torch.load(..., weights_only=False)` uses pickle which allows arbitrary code execution if loading untrusted checkpoints.

### Evidence
```python
# trainer.py:393
checkpoint = torch.load(path, map_location=self.device, weights_only=False)

# Compare with cli.py:455 which does it correctly:
checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=True)
```

### Impact
- Security risk if loading untrusted checkpoints
- CLI `--resume` exposes this

### Fix
```python
# trainer.py:393
checkpoint = torch.load(path, map_location=self.device, weights_only=True)
```

---

## [P2] --no-require-mask Creates Misaligned dataset_info.json

**MEDIUM - VALIDATED**

### Location
- `src/arc_meshchop/cli.py:145`
- `src/arc_meshchop/data/huggingface_loader.py:75-76`

### Problem
`mask_paths` property excludes None masks, but `image_paths` includes all samples. When `--no-require-mask` is used, the lists have different lengths.

### Evidence
```python
# huggingface_loader.py:75-76
def mask_paths(self) -> list[Path]:
    return [s.mask_path for s in self.samples if s.mask_path is not None]

# cli.py:145 - writes these misaligned lists
"mask_paths": [str(p) for p in arc_info.mask_paths],
```

### Impact
- Training will crash or use wrong masks if `--no-require-mask` used
- Default is `require_lesion_mask=True`, so rarely triggered

### Fix
Write aligned lists with None placeholders, or disallow `--no-require-mask` for training workflows.

---

## [P2] Nested-CV Uses Inner-Fold Model Instead of Retrained Model

**MEDIUM - VALIDATED (PROTOCOL DEVIATION)**

### Location
- `src/arc_meshchop/experiment/runner.py:458-464`

### Problem
After selecting the best inner fold model, it's evaluated directly on the outer test set without retraining on the full outer-training split.

### Evidence
```python
# runner.py:458-464
# Loads checkpoint from best inner fold run
checkpoint = torch.load(best_run.checkpoint_path, ...)
model.load_state_dict(checkpoint["model_state_dict"])
# Then evaluates directly on test set
```

### Impact
- Deviates from strict nested-CV protocol
- May produce LOWER scores than paper (fewer training samples)
- Many papers do this anyway; not necessarily wrong

### Discussion
Strict nested-CV:
1. Select best model/hyperparams from inner folds
2. **Retrain on full outer-train (all inner fold data combined)**
3. Evaluate retrained model on outer test set

Current implementation skips step 2. This is a valid simplification used in many papers, but may underperform the "proper" protocol.

### Fix (Optional)
Add retrain step:
```python
# After selecting best model, compute outer-train indices
outer_train_indices = all_indices - test_indices
# Create combined dataset from outer-train
# Retrain model from scratch
# Then evaluate on test set
```

---

## [P3] MPS Autocast Context When Disabled

**LOW - PARTIALLY VALIDATED**

### Location
- `src/arc_meshchop/training/trainer.py:228`

### Problem
The `autocast` context is always entered even when `enabled=False`, which may cause warnings on MPS.

### Evidence
```python
# trainer.py:228
with autocast(
    device_type=self.device.type,
    dtype=self._amp_dtype,
    enabled=self._amp_enabled,  # False on MPS
):
```

### Impact
- Log noise on MPS, not functional issue
- Lower priority

### Fix
```python
# Conditional context manager
if self._amp_enabled:
    with autocast(...):
        outputs = self.model(images)
        loss = self.loss_fn(outputs, masks)
else:
    outputs = self.model(images)
    loss = self.loss_fn(outputs, masks)
```

---

## Summary Table

| Priority | Issue | File:Line | Impact | Fix Effort |
|----------|-------|-----------|--------|------------|
| P0 | Mask binarization | preprocessing.py:237 | 1 sample (0.44%) | Trivial |
| P1 | RAS+ orientation | preprocessing.py:43 | 2 samples (0.88%) | Small |
| P1 | Resume broken | trainer.py:166,393 | Resume unusable | Medium |
| P2 | Checkpoint security | trainer.py:393 | Security risk | Trivial |
| P2 | mask_paths alignment | cli.py:145 | --no-require-mask | Small |
| P2 | Nested-CV protocol | runner.py:458 | Protocol deviation | Medium |
| P3 | MPS autocast | trainer.py:228 | Log noise | Trivial |

---

## Recommended Action

1. **Immediate**: Fix P0 (mask binarization) - trivial change, prevents data corruption
2. **High**: Fix P1s before next training run
3. **Medium**: Fix P2s for robustness
4. **Low**: P3 can wait

**Current training should continue** - impact is minimal (1 sample affected).
