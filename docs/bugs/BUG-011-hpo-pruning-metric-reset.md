# BUG-011: HPO Pruning Uses Per-Fold Metrics Sequentially (P2)

**Priority:** P2
**Status:** Fixed
**Impacts:** `arc-meshchop hpo` (Optuna ASHA pruning)

---

## Symptom

- HPO results are unstable across runs with the same seed.
- Trials with strong early performance can be pruned when moving to the next inner fold.
- ASHA pruning decisions depend on fold order rather than overall trial quality.

---

## Root Cause

In `run_hpo_trial()` (see `src/arc_meshchop/training/hpo.py`):

- `best_dice` is reset for each inner fold.
- `trial.report(best_dice, step)` is called across folds with a single, monotonically
  increasing `step` counter.
- This produces a metric time series that *resets* at fold boundaries, which violates
  ASHA's assumption that metric curves represent increasing resource usage for a
  single training run.

As a result, pruning decisions compare apples and oranges (fold 0 epoch 10 vs fold 1 epoch 1).

---

## Fix Implementation

Refactored `run_hpo_trial` to use interleaved training:
1. Initialize trainers and data loaders for all 3 inner folds upfront.
2. Iterate through epochs (0 to 50).
3. Inside each epoch, train and validate all 3 folds sequentially.
4. Calculate the mean validation DICE across all 3 folds for that epoch.
5. Report the per-epoch mean DICE to Optuna.
6. Track per-fold best DICE separately for the final trial objective.
7. Store per-fold DICE history in Optuna trial attributes for analysis.

This ensures that `step` corresponds to "epochs trained on all folds" and the metric curve reflects actual per-epoch progress.

---

## Verification

- Added unit test to verify that `trial.report` is called once per epoch with aggregated metrics.
- Verified that per-epoch reporting does not reset at fold boundaries and respects decreasing epochs.
