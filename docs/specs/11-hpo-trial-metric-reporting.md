# Spec 11: HPO Trial Metric Reporting

**Status:** Implemented

## Summary

Fix Optuna ASHA reporting to avoid fold-by-fold metric resets and improve
pruning stability during HPO.

---

## Goals

- Ensure reported metrics reflect increasing resource usage.
- Make ASHA pruning decisions consistent across folds.
- Preserve fold-level metrics for analysis.

---

## Non-Goals

- Changing the HPO search space.
- Replacing Optuna.

---

## Implementation

### Interleaved Per-Epoch Aggregation

- Initialize trainers and data loaders for all inner folds up front.
- For each epoch:
  - Train one epoch on each inner fold.
  - Validate each fold and collect DICE values.
  - Report the mean DICE across folds via `trial.report(mean_dice, epoch)`.

### Data Capture

- Track per-fold best DICE for the final objective value.
- Store `fold_dice_history` and `mean_dice_history` in Optuna trial attributes.

---

## Verification

- Added a unit test that simulates fold metrics and confirms reporting is per-epoch
  (no fold-boundary resets, decreasing epochs allowed).
