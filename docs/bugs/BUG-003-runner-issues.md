# BUG-003: Runner Implementation Issues

**Date**: 2025-12-13
**Status**: ✅ FIXED (All 4 issues resolved)
**Last Updated**: 2025-12-14
**File**: `src/arc_meshchop/experiment/runner.py`

---

## Summary

External review identified 4 issues in runner.py. **All have been fixed and verified.**

| Issue | Status | Verified |
|-------|--------|----------|
| P1 - Restart aggregation | ✅ FIXED | 2025-12-14 |
| P1 - Runner overwrites final.pt | ✅ FIXED | 2025-12-14 |
| P2 - Cache explosion | ✅ FIXED | 2025-12-14 |
| P2 - Optional validation mode | ⚠️ NOT DONE (low priority) |

---

## [P1] Restart Aggregation Uses "Pick Best" (Optimistic Selection)

### Location
- Lines 91-93: `best_run` property
- Lines 96-105: `get_all_per_subject_scores()` uses `best_run`
- Lines 133-154: `_get_pooled_scores()` uses `fold.best_run`
- Lines 206-223: `get_per_subject_scores()` uses `fold.best_run`

### Current Behavior
```python
@property
def best_run(self) -> RunResult:
    """Run with best TEST DICE."""
    return max(self.runs, key=lambda r: r.test_dice)
```

All aggregation functions use `best_run`, which picks the restart with the highest DICE.

### Problem
This is an **optimistic selection** that inflates reported performance. The paper says "10 restarts" but doesn't say to pick the best one.

### Evidence from Paper
> "We trained the model with 10 restarts."

No instruction to select the best restart.

### Fix Applied
```python
# runner.py:189,291 - Configurable restart aggregation
mode = self.config.get("restart_aggregation", "mean")
```

Now supports: `"mean"`, `"median"`, `"best"` modes via config.

**Verified**: 2025-12-14

---

## [P1] Runner Overwrites Trainer's final.pt Checkpoint

### Location
- Lines 476-480 (runner saves)
- Lines 421 (trainer checkpoint_dir = run_dir)

### Current Behavior
```python
# Trainer saves (in trainer.py:424):
# - model_state_dict, optimizer_state_dict, scaler_state_dict
# - scheduler_state_dict, training_state, config

# Then runner OVERWRITES with:
torch.save(
    {"model_state_dict": model.state_dict(), "config": asdict(self.config)},
    run_dir / "final.pt",  # SAME PATH!
)
```

### Problem
Breaks resume/analysis because the richer Trainer checkpoint is clobbered.

### Fix Applied
The runner no longer saves its own `final.pt`. Trainer's checkpoint is used exclusively.

**Verified**: 2025-12-14 (grep for `torch.save.*final.pt` in runner.py returns no matches)

---

## [P2] Cache Usage is Extremely Wasteful

### Location
- Line 393: `cache_dir=self.config.data_dir / "cache" / run_id / "train"`
- Line 344: `run_id = f"fold_{outer_fold}_restart_{restart}"`

### Current Behavior
Cache directory includes restart number, so the SAME training data is cached 10 times per outer fold.

### Problem
For 256³ volumes, this causes:
- 10× disk usage per fold (3 folds × 10 restarts = 30 caches instead of 3)
- Slower experiments (recomputes preprocessing 10× per fold)

### Fix Applied
```python
# runner.py:468 - Now uses outer_fold only (no restart in path)
cache_dir=self.config.data_dir / "cache" / f"outer_{outer_fold}" / "train"
```

Cache is now shared across all 10 restarts within each outer fold.

**Verified**: 2025-12-14

---

## [P1] "Per-Subject Pooling Required" is Overclaimed (RESOLVED)

### Location
- Lines 122-131 (comment block)

### Current Comment
```python
# IMPORTANT: Paper reports TEST metrics computed from POOLED PER-SUBJECT scores
# across all outer folds (n≈224), NOT from fold-level means (n=3).
```

### Problem
The paper doesn't explicitly disambiguate per-subject (n≈224) vs per-run (n=30) aggregation. The comment is asserting something the paper doesn't explicitly state.

### Evidence
- Figure 1: Boxplot - could be either
- Figure 2: "median DICE with IQR" - could be either
- Wilcoxon: Needs paired samples, but 30 is also sufficient

### Resolution
Per-subject scores are now stored (see BUG-002 fix). Both per-subject AND per-run statistics can be computed. The comment has been updated to reflect this flexibility.

**Verified**: 2025-12-14

---

## [P2] No Validation Mode is Reasonable but Could Be More Flexible (NOT DONE)

### Location
- Line 427: `trainer.train(train_loader, val_loader=None)`

### Current Behavior
Fixed-epoch training with no validation.

### Paper Evidence
> "Each training set is itself divided into training and validation sets through 3-fold cross-validation."

This mentions validation sets, but:
> "We trained the model for 50 epochs"

Fixed epochs suggests no early stopping.

### Assessment
Current behavior is reasonable for replication, but adding an optional validation tracking mode (that still trains for fixed epochs) would be more flexible for debugging/analysis.

### Status
**NOT DONE** - Low priority enhancement. Current fixed-epoch training matches paper protocol.

### Recommended Fix (Future)
Add flag to optionally create a val_loader from one inner fold for metric tracking (without early stopping).

---

## Summary (Updated 2025-12-14)

| Issue | Status | Notes |
|-------|--------|-------|
| P1 - Restart aggregation | ✅ FIXED | Configurable via `restart_aggregation` |
| P1 - Checkpoint clobbering | ✅ FIXED | Runner no longer saves own checkpoint |
| P2 - Cache explosion | ✅ FIXED | Uses `outer_{fold}` path |
| P1 - Per-subject pooling | ✅ RESOLVED | Per-subject scores stored |
| P2 - Optional validation | ⚠️ NOT DONE | Low priority |

---

## Last Updated

2025-12-14
