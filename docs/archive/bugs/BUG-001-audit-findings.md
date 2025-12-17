# BUG-001: Codebase Audit Findings

**Date**: 2025-12-13
**Status**: FIXED (6 of 7 issues resolved)
**Last Updated**: 2025-12-14

## Summary

An external audit identified 7 issues. **6 have been fixed**, 1 remains (low-impact).

| Priority | Issue | Status |
|----------|-------|--------|
| P0 | Mask binarization | ✅ FIXED |
| P1 | RAS+ orientation | ✅ FIXED |
| P1 | Resume broken | ✅ FIXED |
| P2 | Checkpoint security | ✅ FIXED |
| P2 | mask_paths alignment | ⚠️ NOT FIXED (low impact) |
| P2 | Nested-CV protocol | ✅ FIXED (see NESTED-CV-PROTOCOL.md) |
| P3 | MPS autocast | ✅ FIXED |

---

## [P0] Mask Binarization Zeros Valid Lesions

**CRITICAL - ✅ FIXED**

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

### Fix Applied
```python
# preprocessing.py:246 - NOW USES > 0
mask_normalized = (mask_resampled > 0).astype(np.float32)

# huggingface_loader.py:621 - NOW USES > 0
return int(np.sum(data > 0))
```

**Verified**: 2025-12-14

---

## [P1] Missing RAS+ Canonical Orientation

**HIGH - ✅ FIXED**

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

### Fix Applied
```python
# preprocessing.py:49 - NOW ENFORCES RAS+
nii = nib.as_closest_canonical(nii)
```

**Verified**: 2025-12-14

---

## [P1] Resume Does Not Restore Training State

**HIGH - ✅ FIXED**

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

### Fix Applied
```python
# trainer.py:127 - Pending state storage
self._pending_scheduler_state: dict | None = None

# trainer.py:172-176 - Deferred scheduler state loading
if self._pending_scheduler_state is not None:
    self.scheduler.load_state_dict(self._pending_scheduler_state)
    self._pending_scheduler_state = None

# trainer.py:183 - Start from correct epoch
start_epoch = self.state.epoch + 1 if self.state.global_step > 0 else 0

# trainer.py:197 - Loop from start_epoch
for epoch in range(start_epoch, self.config.epochs):
```

**Verified**: 2025-12-14

---

## [P2] Insecure Checkpoint Deserialization

**MEDIUM - ✅ FIXED**

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

### Fix Applied
```python
# trainer.py:445-446 - Now uses weights_only=True
# Use weights_only=True for security (BUG-001: P2 fix)
checkpoint = torch.load(path, map_location=self.device, weights_only=True)
```

**Verified**: 2025-12-14

---

## [P2] --no-require-mask Creates Misaligned dataset_info.json

**MEDIUM - ⚠️ NOT FIXED (Low Impact)**

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

### Status
**NOT FIXED** - Low impact because `--no-require-mask` is rarely used. Default is `require_lesion_mask=True`.

### Recommended Fix
Write aligned lists with None placeholders, or disallow `--no-require-mask` for training workflows.

---

## [P2] Nested-CV Uses Inner-Fold Model Instead of Retrained Model

**MEDIUM - ✅ FIXED (Protocol Changed)**

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

### Resolution
The nested CV protocol was completely redesigned based on paper analysis. See `docs/archive/bugs/NESTED-CV-PROTOCOL.md` for full details.

**Key changes:**
- Inner folds removed from final evaluation (only used for HP search on fold 1)
- Now trains on full outer-train (67% of data) per fold
- 3 outer folds × 10 restarts = 30 runs (was incorrectly 90)

**Verified**: 2025-12-14

---

## [P3] MPS Autocast Context When Disabled

**LOW - ✅ FIXED**

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

### Fix Applied
```python
# trainer.py:268,331 - Now conditional
if self._amp_enabled:
    with autocast(
        device_type=self.device.type,
        dtype=self._amp_dtype,
        enabled=True,
    ):
        # forward pass
else:
    # forward pass without autocast
```

**Verified**: 2025-12-14

---

## Summary Table (Updated 2025-12-14)

| Priority | Issue | Status | Verified |
|----------|-------|--------|----------|
| P0 | Mask binarization | ✅ FIXED | 2025-12-14 |
| P1 | RAS+ orientation | ✅ FIXED | 2025-12-14 |
| P1 | Resume broken | ✅ FIXED | 2025-12-14 |
| P2 | Checkpoint security | ✅ FIXED | 2025-12-14 |
| P2 | mask_paths alignment | ⚠️ NOT FIXED | Low impact |
| P2 | Nested-CV protocol | ✅ FIXED | 2025-12-14 |
| P3 | MPS autocast | ✅ FIXED | 2025-12-14 |

---

## Remaining Work

Only **1 issue remains unfixed**:

- **P2 mask_paths alignment**: The `mask_paths` property filters out None masks, causing misalignment with `image_paths` when `--no-require-mask` is used. Low impact because this flag is rarely used (default is `require_lesion_mask=True`).

---

## Last Updated

2025-12-14
