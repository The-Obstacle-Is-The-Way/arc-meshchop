# BUG-010: Docstrings Coupled to Documentation File Paths (P4)

**Priority:** P4 (Code smell, no runtime impact)
**Status:** Fixed
**Impact:** Maintenance burden, stale references when docs reorganized

---

## Symptom

Module docstrings reference documentation file paths.

**Before (historical - fixed in this PR):**

```python
# src/arc_meshchop/experiment/runner.py:17
"""See docs/REPRODUCIBILITY.md for full protocol details."""

# src/arc_meshchop/experiment/config.py:24
"""See docs/REPRODUCIBILITY.md for full protocol details."""

# src/arc_meshchop/data/splits.py:13
"""See docs/REPRODUCIBILITY.md for full protocol details."""
```

This creates **tight coupling** between code and filesystem paths.

---

## Why This Is Problematic

### 1. Violates DRY (Don't Repeat Yourself)

The same explanation exists in TWO places:
- The docstring contains a partial explanation
- The doc file contains the full explanation

When protocol changes, both must be updated.

### 2. Violates Single Source of Truth (SSOT)

Where is the canonical protocol definition?
- `runner.py` docstring says "3 outer folds x 10 restarts"
- `docs/REPRODUCIBILITY.md` says the same thing

If they diverge, which is correct?

### 3. Tight Coupling to Filesystem

When docs are reorganized (as happened in BUG-009), references break:
- Old path: `docs/archive/bugs/NESTED-CV-PROTOCOL.md` (deleted)
- New path: `docs/REPRODUCIBILITY.md`

This is the EXACT failure mode that caused BUG-009.

### 4. Violates PEP 257 Spirit

From [PEP 257](https://peps.python.org/pep-0257/):
> "The docstring should describe the function's calling syntax and its semantics"

Docstrings should be **self-contained**. A developer reading the docstring shouldn't need to chase down external files to understand the code.

---

## What IS Appropriate

### Scientific Citations (GOOD)

```python
# FROM PAPER Section 2:
# "Hyperparameter optimization was conducted on the inner folds..."
```

This is **citation** - attributing the source of truth (the published paper). The paper is external, immutable, and authoritative. This is correct.

### Self-Contained Docstrings (GOOD)

```python
"""Generate nested CV splits.

Protocol:
- 3 outer folds for train/test
- 3 inner folds per outer for train/val
- Stratified by lesion size + acquisition type
"""
```

The docstring IS the documentation. No external reference needed.

---

## Root Cause Analysis

This pattern emerged from a reasonable intention:
1. Developer wanted to avoid duplicating long explanations
2. Created external doc file with full details
3. Added "See docs/..." to point developers there

**But this inverts the correct relationship:**
- Docs should be GENERATED FROM or SUMMARIZE code
- Code should NOT reference docs (except for truly external resources)

---

## Recommended Fix

### Option A: Make Docstrings Self-Contained (Recommended)

Remove doc file references. Make each docstring complete:

```python
"""Experiment runner for full paper replication.

Protocol (from paper Section 2):
- 3 outer folds × 10 restarts = 30 runs
- HP search was on Outer Fold 1 only (we skip, use paper's HPs)
- Training uses FULL outer-train (no inner fold split)
- Fixed 50 epochs, no early stopping
"""
```

The `docs/REPRODUCIBILITY.md` file becomes a USER GUIDE, not a developer reference.

### Option B: Use URL/Permalink (Acceptable for External Resources)

If you MUST reference external docs, use stable URLs:

```python
"""See: https://github.com/org/repo/blob/main/docs/PROTOCOL.md"""
```

But for internal docs, Option A is cleaner.

### Option C: Keep As-Is (Not Recommended)

Accept the maintenance burden. Every doc reorganization requires code changes.

---

## Affected Files

| File | Line | Reference |
|------|------|-----------|
| `src/arc_meshchop/experiment/runner.py` | 17 | `docs/REPRODUCIBILITY.md` |
| `src/arc_meshchop/experiment/config.py` | 24 | `docs/REPRODUCIBILITY.md` |
| `src/arc_meshchop/data/splits.py` | 13 | `docs/REPRODUCIBILITY.md` |

---

## Distinction: Citations vs References

| Pattern | Example | Verdict |
|---------|---------|---------|
| Paper citation | `FROM PAPER: "lr=0.001"` | ✅ GOOD - External, immutable source |
| Doc file reference | `See docs/REPRODUCIBILITY.md` | ❌ BAD - Internal, mutable coupling |
| URL permalink | `See: https://...#section` | ⚠️ OK for external resources |

---

## Software Engineering Principles

| Principle | Status |
|-----------|--------|
| DRY | ❌ Violated |
| SSOT | ❌ Violated |
| Loose Coupling | ❌ Violated |
| PEP 257 | ⚠️ Spirit violated |
| Clean Code (Martin) | ❌ Comments should not require chasing files |

---

## Severity Assessment

**P4 - Low Priority** because:
- No runtime impact
- No correctness impact
- Affects developer experience only
- Manifests only during doc reorganization

This is a code smell that should be addressed for long-term maintainability.

---

## Resolution

Applied **Option A: Make Docstrings Self-Contained**.

All `See docs/REPRODUCIBILITY.md` references removed. Docstrings now contain complete protocol information inline:

| File | Before | After |
|------|--------|-------|
| `runner.py` | "See docs/REPRODUCIBILITY.md" | Complete protocol + hyperparameters inline |
| `config.py` | "See docs/REPRODUCIBILITY.md" | Complete protocol + target metrics inline |
| `splits.py` | "See docs/REPRODUCIBILITY.md" | Complete CV structure inline |

**Example: splits.py after fix:**

```python
"""Cross-validation split generation with stratification.

Protocol (FROM PAPER Section 2):
    "Hyperparameter optimization was conducted on the inner folds of the first
    outer fold. The optimized hyperparameters were then applied to train models
    on all outer folds."

Structure:
    - 3 outer folds: 67% train / 33% test per fold
    - 3 inner folds per outer: Only used for HP search (first outer fold)
    - Stratification: Lesion size quintile x acquisition type (SPACE, SPACE-2x)

For replication (using paper's published HPs):
    - Use outer folds only via `generate_outer_cv_splits()`
    - Train on FULL outer-train (no inner validation holdout)
    - Evaluate on outer-test after fixed 50 epochs
"""
```

The `docs/REPRODUCIBILITY.md` file now serves as a **user guide** (how to run experiments) rather than a developer reference that code points to.

**Distinction preserved:**
- `FROM PAPER:` citations → ✅ Kept (external immutable source)
- `See docs/...` references → ❌ Removed (internal mutable coupling)

---

## References

- [PEP 257 - Docstring Conventions](https://peps.python.org/pep-0257/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Documentation Best Practices](https://google.github.io/styleguide/docguide/best_practices.html)
- BUG-009: The exact failure mode this pattern causes
