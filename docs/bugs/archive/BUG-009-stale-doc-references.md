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
3. Remove internal doc-path references entirely by making docstrings self-contained (preferred; see `docs/bugs/BUG-010-docstring-doc-coupling.md`).

---

## Resolution

- Removed all stale `docs/archive/bugs/*` references from code.
- Removed remaining internal doc-path references by making docstrings self-contained (see `docs/bugs/BUG-010-docstring-doc-coupling.md`).
