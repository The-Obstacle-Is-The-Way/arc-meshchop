# BLOCKERS.md — Pre-Training Audit SSOT

> **Purpose:** Single source of truth for all blocking issues that must be fixed before training locally.
>
> **Status:** FIXED — All blockers resolved (2025-12-12).
>
> **Created:** 2025-12-12
>
> **Last Validated:** Each claim verified against actual code and paper.

---

## Executive Summary

The core pipeline is implemented and largely paper-aligned:
- ✅ MeshNet architecture matches paper (10-layer, symmetric dilation)
- ✅ Training config matches paper (AdamW, OneCycleLR div_factor=100, etc.)
- ✅ Nested CV with stratification implemented
- ✅ Paper parity validation infrastructure in place
- ✅ bids_hub integration correct (only imports validation constants)

**All 5 blocking issues have been fixed:**

| # | Issue | Severity | Impact | Status |
|---|-------|----------|--------|--------|
| 1 | Memory OOM on evaluation | **CRITICAL** | Will crash on typical 16GB machines | **FIXED** |
| 2 | Docs say `--checkpoint`, CLI uses positional | MEDIUM | User confusion, broken examples | **FIXED** |
| 3 | Docs say global HF cache, CLI uses project-local | MEDIUM | User confusion about data location | **FIXED** |
| 4 | Gradient checkpointing flag exists but does nothing | LOW | Misleading config option | **FIXED** |
| 5 | HPO uses MedianPruner, docs/config say ASHA | LOW | Incorrect terminology | **FIXED** |

**Quality risks fixed:**
- ✅ Experiment runner now caches preprocessed data (faster runtime)
- ✅ Temp file leaks fixed
- ✅ Silent error swallowing fixed

---

## BLOCKING ISSUE #1: Memory OOM on Evaluation

### The Problem

All evaluation paths accumulate **full 256³ predictions and targets in RAM** before computing metrics:

```python
# trainer.py:278-298 (_validate_epoch)
all_preds: list[torch.Tensor] = []
all_targets: list[torch.Tensor] = []
for images, masks in tqdm(val_loader, desc="Validation"):
    ...
    all_preds.append(preds.cpu())
    all_targets.append(masks.cpu())
# Then concatenates: torch.cat(all_preds, dim=0)
```

Same pattern in:
- `cli.py:462-480` (evaluate command)
- `evaluator.py:104-127` (Evaluator.evaluate)
- `experiment/runner.py:476-491` (_evaluate_on_test_sets)

### Why This Fails

Each 256³ volume = 16.7 million voxels.
- Predictions: int64 tensor = 16.7M × 8 bytes = 134 MB per sample
- Targets: same = 134 MB per sample
- For 75 test samples (typical outer fold): 2 × 75 × 134 MB = **20+ GB just for eval**

**A 16GB MacBook will OOM during validation/evaluation.**

### Paper Context

The paper used "NVIDIA A100 GPU with 80 GB memory." They could accumulate everything in RAM.
We target consumer hardware (M1 MacBook with 16-32GB).

### Required Fix

Compute metrics incrementally (sample-by-sample) instead of batch accumulation:

```python
# Instead of accumulating, compute running metrics
dice_scores = []
for images, masks in val_loader:
    preds = model(images).argmax(dim=1)
    dice = compute_dice(preds, masks)  # Per-sample
    dice_scores.append(dice)
# dice_scores is just a list of floats, not 256³ tensors
```

### Files to Modify

1. `src/arc_meshchop/training/trainer.py` - `_validate_epoch()`
2. `src/arc_meshchop/cli.py` - `evaluate()` command
3. `src/arc_meshchop/evaluation/evaluator.py` - `evaluate()`
4. `src/arc_meshchop/experiment/runner.py` - `_evaluate_on_test_sets()`

---

## BLOCKING ISSUE #2: Docs vs CLI Mismatch (`--checkpoint` vs positional)

### The Problem

Documentation shows:
```bash
# TRAINING.md:133
arc-meshchop evaluate --checkpoint outputs/meshnet26/best.pt --data-dir data/arc

# ARCHITECTURE.md:213
arc-meshchop evaluate --checkpoint outputs/train/best.pt --data-dir data/arc
```

But CLI actually uses a **positional argument**:
```python
# cli.py:372-377
@app.command()
def evaluate(
    checkpoint: Annotated[
        Path,
        typer.Argument(help="Path to model checkpoint"),  # <-- POSITIONAL
    ],
```

Correct usage:
```bash
arc-meshchop evaluate outputs/meshnet26/best.pt --data-dir data/arc
```

### Required Fix

Update `TRAINING.md:133` and `ARCHITECTURE.md:213` to use positional argument syntax.

---

## BLOCKING ISSUE #3: Cache Location Mismatch

### The Problem

Documentation claims data goes to global HuggingFace cache:
```markdown
# README.md:35
# Download dataset (goes to ~/.cache/huggingface/hub/)

# DATA.md:273
# 3. Data goes to ~/.cache/huggingface/hub/
```

But CLI forces a **project-local cache**:
```python
# cli.py:128-130
arc_info = load_arc_from_huggingface(
    repo_id=repo_id,
    cache_dir=output_dir / "cache",  # <-- Forces project-local!
```

So data actually goes to `data/arc/cache/`, not `~/.cache/huggingface/hub/`.

### Why This Matters

- Users may not realize ~25GB is being stored in their project directory
- Users who expect global caching will download the data twice
- Docs are misleading about storage requirements

### Required Fix

Either:
1. **Option A:** Change CLI to use global cache (remove `cache_dir=output_dir / "cache"`)
2. **Option B:** Update docs to accurately reflect project-local cache behavior

Recommend **Option B** — project-local is actually better for reproducibility and cleanup.

---

## BLOCKING ISSUE #4: Gradient Checkpointing Does Nothing

### The Problem

The config has a flag:
```python
# config.py:53
use_gradient_checkpointing: bool = False
```

And the trainer docstring mentions it:
```python
# trainer.py:7
"Layer checkpointing for large models"
```

**But there is no implementation.** The flag is never read. No `torch.utils.checkpoint` calls exist.

### Paper Context

The paper says: "We used layer checkpointing for models that could not fit into GPU memory."

But this was for **baseline models** (U-MAMBA, Swin-UNETR, etc.), not MeshNet.
MeshNet is designed to NOT need checkpointing due to its small size (147K params).

### Required Fix

Either:
1. **Option A:** Remove the flag and docstring reference (MeshNet doesn't need it)
2. **Option B:** Implement it properly with `torch.utils.checkpoint`

Recommend **Option A** — MeshNet doesn't need checkpointing, and implementing it adds complexity for no benefit.

---

## BLOCKING ISSUE #5: HPO Terminology Mismatch

### The Problem

Config and docs say ASHA via SuccessiveHalvingPruner:
```python
# config.py:87
algorithm: Literal["asha", "tpe"] = "asha"  # ASHA via SuccessiveHalvingPruner
```

But implementation uses MedianPruner:
```python
# hpo.py:139
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=5,
    n_warmup_steps=10,
    interval_steps=5,
)
```

### Paper Context

The paper used Orion with ASHA. We use Optuna because Orion is broken on Python 3.12.

MedianPruner is NOT equivalent to SuccessiveHalvingPruner (ASHA):
- **ASHA/SuccessiveHalving:** Promotes top 1/η fraction at each rung
- **MedianPruner:** Prunes trials below median at intermediate points

### Required Fix

Either:
1. **Option A:** Change to `optuna.pruners.SuccessiveHalvingPruner` (actual ASHA)
2. **Option B:** Update docs/config to accurately describe MedianPruner behavior

Recommend **Option A** — for paper parity, use actual ASHA.

---

## QUALITY RISKS (Non-Blocking)

### Risk 1: Experiment Runner Doesn't Cache Preprocessed Data

**Problem:**
```python
# cli.py:316 (train command) - HAS cache_dir
train_dataset = ARCDataset(
    cache_dir=data_dir / "cache" / f"fold_{outer_fold}" / "train",
)

# experiment/runner.py:309 - NO cache_dir
train_dataset = ARCDataset(
    image_paths=[Path(image_paths[i]) for i in split.train_indices],
    mask_paths=[Path(mask_paths[i]) for i in split.train_indices],
    # No cache_dir!
)
```

**Impact:** Full 30-run experiment will re-preprocess every volume every time.
256³ resampling is expensive. This will add hours to experiment runtime.

**Status:** FIXED — runner.py now uses `cache_dir=...f"outer_{outer_fold}"...` which
shares cache across restarts within each fold (BUG-003 fix).

### Risk 2: Temp File Leaks in Lesion Volume Computation

**Problem:**
```python
# huggingface_loader.py:494-498
with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
    f.write(nifti_obj["bytes"])
    temp_path = f.name
nii = nib.load(temp_path)
# temp_path is NEVER deleted!
```

**Impact:** Each call leaks a temp file. For 224 samples, that's 224 orphaned files.

**Fix:** Add `os.unlink(temp_path)` after loading, or use `delete=True`.

### Risk 3: Silent Error Swallowing

**Problem:**
```python
# huggingface_loader.py:507-508
except Exception:
    return 0
```

**Impact:** Any error in lesion volume computation is silently ignored.
User has no idea if volumes are wrong.

**Fix:** At minimum, log a warning. Better: raise or re-raise with context.

---

## Files Requiring Updates

### Code Changes Required

| File | Issue | Change |
|------|-------|--------|
| `src/arc_meshchop/training/trainer.py` | #1 Memory | Incremental metric computation |
| `src/arc_meshchop/cli.py` | #1 Memory | Incremental metric computation |
| `src/arc_meshchop/evaluation/evaluator.py` | #1 Memory | Incremental metric computation |
| `src/arc_meshchop/experiment/runner.py` | #1 Memory + Risk 1 | Incremental metrics + add cache_dir |
| `src/arc_meshchop/training/config.py` | #4 or #5 | Remove checkpointing flag OR fix HPO comment |
| `src/arc_meshchop/training/hpo.py` | #5 | Change to SuccessiveHalvingPruner |
| `src/arc_meshchop/data/huggingface_loader.py` | Risk 2, 3 | Fix temp file leak, add error logging |

### Doc Changes Required

| File | Issue | Change |
|------|-------|--------|
| `TRAINING.md:133` | #2 | Change `--checkpoint` to positional |
| `ARCHITECTURE.md:213` | #2 | Change `--checkpoint` to positional |
| `README.md:35` | #3 | Update cache location description |
| `DATA.md:273` | #3 | Update cache location description |

---

## Implementation Priority

1. **FIRST:** Fix memory issue (#1) — blocks all evaluation
2. **SECOND:** Fix HPO to use actual ASHA (#5) — for paper parity
3. **THIRD:** Fix docs (#2, #3) — user experience
4. **FOURTH:** Clean up config (#4) — reduce confusion
5. **FIFTH:** Fix quality risks — polish

---

## Verification After Fixes

```bash
# After fixing memory issue, this should work on 16GB machine:
uv run arc-meshchop train --channels 26 --epochs 5 --outer-fold 0

# After fixing docs, these examples should work:
uv run arc-meshchop evaluate outputs/train/best.pt --data-dir data/arc

# Full CI should pass:
make ci
```

---

## References

- Paper: [arxiv.org/html/2503.05531v1](https://arxiv.org/html/2503.05531v1)
- Paper memory: "NVIDIA A100 GPU with 80 GB memory"
- Paper HPO: "Orion... employed the Asynchronous Successive Halving Algorithm (ASHA)"
- Paper checkpointing: "We used layer checkpointing for models that could not fit into GPU memory" (for baselines, not MeshNet)
