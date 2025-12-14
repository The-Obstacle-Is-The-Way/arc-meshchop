# BUG-003: Runner Implementation Issues

**Date**: 2025-12-13
**Status**: DOCUMENTED (for tomorrow)
**File**: `src/arc_meshchop/experiment/runner.py`

---

## Summary

External review identified 4 issues in runner.py that weren't addressed during the nested CV protocol refactor. These are ADDITIONAL bugs, not contradictions to the protocol change.

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

### Fix
Add `restart_aggregation: Literal["mean", "median", "all", "best", "ensemble"]` parameter:
- Default to `"mean"` (non-optimistic)
- For per-subject scores, either:
  - Average each subject's score across restarts, OR
  - Treat each (fold, restart) as one replicate (n=30)

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

### Fix
Either:
1. Delete the runner's `torch.save()` call (use Trainer's checkpoint)
2. Save to a different filename (e.g., `model.pt` or `run_config.pt`)

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

### Fix
Use cache dirs shared per outer fold:
```python
# Before:
cache_dir=self.config.data_dir / "cache" / run_id / "train"  # run_id includes restart

# After:
cache_dir=self.config.data_dir / "cache" / f"outer_{outer_fold}" / "train"
```

---

## [P1] "Per-Subject Pooling Required" is Overclaimed

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

### Fix
1. Soften the comment to "likely" rather than "required"
2. Compute BOTH per-subject AND per-run statistics
3. Match paper empirically when we run experiments

---

## [P2] No Validation Mode is Reasonable but Could Be More Flexible

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

### Fix (Optional)
Add flag to optionally create a val_loader from one inner fold for metric tracking (without early stopping).

---

## Priority Order for Fixes

1. **[P1] Restart aggregation** - Most impactful on reported metrics
2. **[P1] Checkpoint clobbering** - Simple 1-line fix
3. **[P2] Cache explosion** - Practical issue for storage/speed
4. **[P1] Per-subject overclaim** - Comment fix + add per-run stats
5. **[P2] Optional validation** - Nice-to-have

---

## Not Addressed Yet

These issues don't contradict the nested CV protocol fix. They're orthogonal improvements that should be addressed on top of the correct protocol.
