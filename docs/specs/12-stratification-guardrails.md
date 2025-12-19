# Spec 12: Stratification Guardrails

## Summary

Add preflight checks and fallbacks to prevent StratifiedKFold failures when
any stratum has fewer samples than `n_splits`.

---

## Goals

- Avoid hard failures on small or filtered datasets.
- Preserve stratification where possible.
- Provide actionable error messages when strict stratification is required.

---

## Non-Goals

- Changing the default split logic for the full ARC dataset.

---

## Proposed Design

1. **Preflight count check**
   - Compute counts for each stratification label.
   - Detect if any count < `n_splits`.

2. **Fallback behavior (configurable)**
   - Option A: Reduce `n_splits` to the minimum viable value.
   - Option B: Merge rare labels into `other` and retry stratification.
   - Option C: Fall back to non-stratified KFold.

3. **CLI flag**
   - `--strict-stratification` forces a hard error if constraints are not met.

---

## Verification

- Add tests with a toy dataset where a stratum has < 3 samples:
  - Confirm fallback behavior works.
  - Confirm strict mode errors with clear diagnostics.
