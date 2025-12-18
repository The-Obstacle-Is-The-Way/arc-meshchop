# BUG-009: Stale References to `docs/archive/bugs/*` (P4)

**Priority:** P4
**Status:** Fixed
**Impact:** Developer confusion / dead links (no runtime impact)

---

## Symptom

Several module docstrings/comments reference documentation paths that do not exist in this repo:

- `docs/archive/bugs/NESTED-CV-PROTOCOL.md`
- `docs/archive/bugs/BUG-002-metric-aggregation.md`

This makes it harder to verify protocol details (nested CV, per-subject vs per-run aggregation).

---

## Where It Shows Up

- `src/arc_meshchop/data/splits.py`
- `src/arc_meshchop/experiment/config.py`
- `src/arc_meshchop/experiment/runner.py`

---

## Suggested Fix

Either:

1. Add/move the referenced docs into the repo (e.g., `docs/bugs/`), **or**
2. Update references to point at existing docs:
   - `docs/REPRODUCIBILITY.md`
   - `docs/reference/training.md`

---

## Resolution

Updated all references to point to `docs/REPRODUCIBILITY.md`:
- `src/arc_meshchop/data/splits.py`: Updated docstring
- `src/arc_meshchop/experiment/config.py`: Updated docstring
- `src/arc_meshchop/experiment/runner.py`: Updated module docstring and comments
