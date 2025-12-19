# Spec 11: HPO Trial Metric Reporting

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

## Proposed Design

### Option A (Preferred): Aggregate Per Epoch Across Folds

- For each epoch:
  - Train one epoch on each inner fold.
  - Compute mean (or median) validation DICE across folds.
  - Report the aggregated metric via `trial.report()`.

### Option B: Report Only After All Folds

- Disable pruning during fold training.
- Report the final aggregated score only once.

### Data Capture

- Store per-fold DICE history in the trial attributes or a sidecar JSON file.

---

## Implementation Notes

- Refactor `run_hpo_trial` to separate:
  - fold training loop
  - aggregation
  - Optuna reporting

---

## Verification

- Add a unit test that simulates two folds and confirms the reported
  metric sequence does not reset at fold boundaries.
- Run HPO twice with the same seed and compare top-ranked trials.
