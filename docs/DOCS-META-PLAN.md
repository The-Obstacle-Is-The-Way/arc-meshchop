# Documentation Meta-Plan: Canonical Documentation Strategy

> **Purpose:** Phased plan to audit existing docs, extract valuable content, and create Google DeepMind-quality canonical documentation.
>
> **Status:** PLANNING (v3 - Post Senior Review)
>
> **Date:** 2025-12-16

---

## Executive Summary

This repo has accumulated documentation across multiple phases of development. This plan defines a systematic approach to:

1. **Audit** all existing docs (root + docs/) and classify by current relevance
2. **Extract** valuable information into a canonical structure
3. **Archive** outdated docs (don't delete—preserve history)
4. **Create** a clean, accurate documentation set
5. **Fix** stale references (export command, Hydra claims)

---

## Current State Analysis

### What Exists: Root-Level Docs

```
(repo root)
├── README.md           # Project intro - HAS STALE REFS (export, Hydra)
├── ARCHITECTURE.md     # Data pipeline architecture - PARTIAL (checkpoint paths, test count)
├── TRAINING.md         # Training guide - PARTIAL (checkpoint paths, runtime estimate)
├── DATA.md             # Data pipeline explained - ACTIVE/ACCURATE
├── CONTRIBUTING.md     # Developer guide - PARTIAL (Hydra reference)
└── CLAUDE.md           # Agent instructions - PARTIAL (Hydra reference, docs pointers)
```

### What Exists: docs/ Directory

```
docs/
├── DOCS-META-PLAN.md   # This file
│
├── research/           # Paper extraction (6 files)
│   ├── 00-overview.md      # Good overview, BrainChop section partially stale
│   ├── 01-architecture.md  # Core reference, accurate
│   ├── 02-dataset-preprocessing.md  # Core reference, accurate
│   ├── 03-training-configuration.md # HAS STALE REFS (Hydra, CUDA-only AMP)
│   ├── 04-model-variants.md         # Core reference, accurate
│   └── 05-evaluation-metrics.md     # Core reference, accurate
│
├── specs/              # TDD specs
│   ├── 00-index.md           # Index, references archived specs
│   ├── 07-huggingface-spaces.md  # SPEC (future work) - preserve as-is
│   └── IO-DIRECTORY-STRUCTURE.md # HAS STALE REFS (export command)
│
├── bugs/               # Bug tracking (3 files) - ALL FIXED
│   ├── BUG-005-*.md    # FIXED - archive
│   ├── BUG-006-*.md    # FIXED - archive
│   └── BUG-007-*.md    # FIXED - archive
│
├── issues/             # Open issues (2 files) - CURRENT
│   ├── TRAIN-001-runtime-estimates.md  # OPEN - preserve
│   └── OPT-001-download-performance.md # DEFERRED - preserve
│
├── KNOWN_ISSUES.md     # Dataset issues - PARTIAL (cohort discrepancy still relevant)
└── TODO.md             # Work tracker - STALE (references old bugs)
```

### Source Code Structure (For Verification)

```
src/arc_meshchop/
├── cli.py              # Commands: version, info, download, train, evaluate,
│                       #           experiment, hpo, validate
│                       # NOTE: NO export command (docs are stale)
├── data/
│   ├── huggingface_loader.py
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── splits.py
│   └── config.py
├── models/
│   └── meshnet.py
├── training/
│   ├── trainer.py
│   ├── loss.py
│   ├── optimizer.py
│   ├── hpo.py
│   └── config.py
├── evaluation/
│   ├── metrics.py
│   ├── evaluator.py
│   └── statistics.py
├── experiment/
│   ├── runner.py
│   └── config.py
├── export/
│   ├── onnx_export.py  # EXISTS but not exposed via CLI
│   ├── tfjs_export.py
│   └── pipeline.py
├── validation/
│   └── parity.py
└── utils/
    ├── device.py
    └── paths.py
```

### configs/ Directory (Hydra NOT Used)

```
configs/                # EXISTS but Hydra is NOT imported in src/
├── config.yaml
├── experiment/
├── model/
└── training/
```

**Finding:** Docs reference "Hydra configuration" but Hydra is not actually used. The CLI uses Typer with direct argument parsing.

---

## Classification Schema

| Status | Meaning | Action |
|--------|---------|--------|
| `ACTIVE` | Content matches current codebase, in active use | Keep at current location |
| `ACCURATE` | Content matches codebase, may need integration | Extract into canonical |
| `PARTIAL` | Some content accurate, some stale | Extract accurate parts, fix stale |
| `STALE` | Superseded or contains false claims | Archive (after extracting any value) |
| `SPEC` | Future work specification | Preserve as-is |
| `REFERENCE` | Historical value for context | Archive with note |

---

## Audit Results

### Root-Level Docs

| Document | Status | Notes |
|----------|--------|-------|
| `README.md` | PARTIAL | Good intro but claims `arc-meshchop export` (doesn't exist) and "Hydra configuration" (not used) |
| `ARCHITECTURE.md` | PARTIAL | CLI example uses wrong checkpoint path; hard-codes test count |
| `TRAINING.md` | PARTIAL | CLI example uses wrong checkpoint path; runtime estimate conflicts with TRAIN-001 |
| `DATA.md` | ACTIVE | Accurate data pipeline explanation |
| `CONTRIBUTING.md` | PARTIAL | Good but references "Hydra config" which doesn't exist |
| `CLAUDE.md` | PARTIAL | Mostly accurate, but references Hydra configuration; documentation pointers will change |

### docs/ Directory

| Document | Status | Notes |
|----------|--------|-------|
| `research/00-overview.md` | PARTIAL | Good overview, BrainChop section mentions implementation strategy that's complete |
| `research/01-architecture.md` | ACCURATE | Core reference |
| `research/02-dataset-preprocessing.md` | ACCURATE | Core reference |
| `research/03-training-configuration.md` | PARTIAL | Mentions "Hydra" and CUDA-only AMP snippets vs actual cross-platform |
| `research/04-model-variants.md` | ACCURATE | Core reference |
| `research/05-evaluation-metrics.md` | ACCURATE | Core reference |
| `specs/00-index.md` | PARTIAL | References archived files |
| `specs/07-huggingface-spaces.md` | SPEC | Future work, preserve |
| `specs/IO-DIRECTORY-STRUCTURE.md` | PARTIAL | References `arc-meshchop export` which doesn't exist |
| `bugs/BUG-005-*.md` | REFERENCE | Fixed, archive |
| `bugs/BUG-006-*.md` | REFERENCE | Fixed, archive |
| `bugs/BUG-007-*.md` | REFERENCE | Fixed, archive |
| `issues/TRAIN-001-*.md` | ACTIVE | Open issue |
| `issues/OPT-001-*.md` | ACTIVE | Deferred issue |
| `KNOWN_ISSUES.md` | PARTIAL | Cohort discrepancy explanation still valuable |
| `TODO.md` | STALE | References old bugs, mostly complete |

---

## Stale References to Fix

### 1. Non-Existent `export` Command

**Files affected:**
- `README.md` line ~103: `uv run arc-meshchop export`
- `docs/specs/IO-DIRECTORY-STRUCTURE.md` lines ~191-268: Full export CLI docs

**Reality check:** Export is supported via Python API (`arc_meshchop.export`), but is not exposed via the CLI.

**Fix:** Choose one:
- A) Replace CLI export references with Python API usage (recommended; no code changes)
- B) Add an `export` CLI command that wraps `arc_meshchop.export` (code change; optional)

**Decision:** Replace CLI references with Python API usage. Track CLI `export` as a future enhancement if desired.

### 2. "Hydra Configuration" Claims

**Files affected:**
- `README.md` line ~129: `configs/            # Hydra configuration`
- `CONTRIBUTING.md` line ~207: `configs/                       # Hydra config`
- `CLAUDE.md` line ~105: `configs/` table entry claims Hydra configuration
- `docs/research/03-training-configuration.md`: Mentions Hydra for config

**Fix:** Remove Hydra references. CLI uses Typer with direct arguments.

**Decision:** Update to say "Configuration presets" or remove. The configs/ directory exists but isn't used via Hydra.

### 3. Incorrect Checkpoint Paths in CLI Examples

**Files affected:**
- `ARCHITECTURE.md` CLI workflow: `outputs/train/best.pt` (actual: `outputs/train/fold_0_0/best.pt`)
- `TRAINING.md` evaluate example: `outputs/meshnet26/best.pt` (actual: `outputs/meshnet26/fold_0_0/best.pt`)

**Fix:** Update examples to match `TrainingConfig.checkpoint_dir` behavior (fold-scoped directories).

### 4. Conflicting Runtime Estimates

**Files affected:**
- `README.md` hardware table: `~6-15 hours (tested)` (conflicts with `docs/issues/TRAIN-001-runtime-estimates.md`)
- `TRAINING.md` runtime estimate: `~6-15 hours` (conflicts with `docs/issues/TRAIN-001-runtime-estimates.md`)

**Fix:** Make `docs/issues/TRAIN-001-runtime-estimates.md` the measurement SSOT and update root docs to:
- Avoid hard-coded promises, or
- Link to TRAIN-001 (and later to `docs/REPRODUCIBILITY.md`) with clear caveats.

### 5. Hard-Coded Test Counts

**Files affected:**
- `ARCHITECTURE.md`: "293 tests verify the implementation" (test count drifts over time)

**Fix:** Replace exact numbers with a stable statement ("CI passes" / `make ci`).

### 6. CUDA-Only AMP Snippets

**Files affected:**
- `docs/research/03-training-configuration.md`: Shows `torch.cuda.amp` examples

**Fix:** Update to show cross-platform `torch.amp` approach per actual implementation.

---

## Target State: Canonical Documentation

### Decision: Keep Root-Level Structure

The existing root-level docs (ARCHITECTURE.md, TRAINING.md, DATA.md, CONTRIBUTING.md) are **already canonical**. We should:

1. **Fix** stale references in these files
2. **Preserve** their root location (standard practice)
3. **Use docs/** for reference material and specs

### Final Structure

```
(repo root)
├── README.md           # Fixed: no CLI export, no Hydra, no hard runtime promises
├── ARCHITECTURE.md     # Fixed: checkpoint paths + remove hard-coded test count
├── TRAINING.md         # Fixed: checkpoint paths + runtime section synced to TRAIN-001
├── DATA.md             # Keep as-is (already accurate)
├── CONTRIBUTING.md     # Fixed: no Hydra reference
├── CLAUDE.md           # Fixed: no Hydra reference; docs pointers updated
│
├── CHANGELOG.md        # NEW: Track user-visible changes
└── SECURITY.md         # NEW: Disclosure policy

docs/
├── README.md           # Docs navigation index
│
├── reference/          # Deep technical reference
│   ├── meshnet.md      # ← from research/01-architecture.md
│   ├── dataset.md      # ← from research/02-dataset-preprocessing.md
│   ├── training.md     # ← from research/03-training-configuration.md (fixed)
│   ├── metrics.md      # ← from research/05-evaluation-metrics.md
│   └── variants.md     # ← from research/04-model-variants.md
│
├── REPRODUCIBILITY.md  # NEW: Exact replication protocol
├── TROUBLESHOOTING.md  # NEW: ← from KNOWN_ISSUES.md + IO troubleshooting
│
├── specs/              # Future work specs (preserve as-is)
│   ├── 00-index.md     # Updated to reflect current state
│   └── 07-huggingface-spaces.md  # Preserve
│
├── issues/             # Open issues (keep active)
│   ├── TRAIN-001-runtime-estimates.md
│   └── OPT-001-download-performance.md
│
└── archive/            # Historical docs
    ├── README.md       # Archive index
    ├── bugs/           # BUG-005, BUG-006, BUG-007
    ├── research/       # Original paper extraction
    ├── specs/          # IO-DIRECTORY-STRUCTURE.md (after extraction)
    └── legacy/         # TODO.md, KNOWN_ISSUES.md
```

---

## Phase 1: Fix Stale References

**Goal:** Make existing docs accurate before reorganization.

### 1.1 Fix README.md

Remove/update:
- Line ~103: Replace `uv run arc-meshchop export` with Python export API usage (export is not CLI-exposed)
- Line ~129: Change `configs/            # Hydra configuration` to remove Hydra claim
- Hardware/runtime table: remove hard promises; link to `docs/issues/TRAIN-001-runtime-estimates.md`

### 1.2 Fix CONTRIBUTING.md

- Line ~207: Remove "Hydra config" reference

### 1.3 Fix CLAUDE.md

- Remove "Hydra configuration" claim for `configs/` (Hydra not used)

### 1.4 Fix ARCHITECTURE.md

- Fix CLI workflow checkpoint path example (`best.pt` location)
- Replace hard-coded test count with stable CI statement

### 1.5 Fix TRAINING.md

- Fix evaluate checkpoint path example (fold-scoped output directory)
- Update runtime estimate section to match `docs/issues/TRAIN-001-runtime-estimates.md` (or link)

### 1.6 Fix docs/specs/IO-DIRECTORY-STRUCTURE.md

- Keep Section 3 (Export Directory), but remove/replace CLI export commands with Python API usage
- Explicitly note: export is available as `arc_meshchop.export`, not as `arc-meshchop export`

### 1.7 Fix docs/research/03-training-configuration.md

- Update AMP examples to show cross-platform `torch.amp`
- Remove Hydra references

---

## Phase 2: Create Archive Structure

```bash
mkdir -p docs/archive/{bugs,research,specs,legacy}
```

### Move Fixed Bugs
```bash
mv docs/bugs/*.md docs/archive/bugs/
rmdir docs/bugs
```

### Move Stale Trackers
```bash
mv docs/TODO.md docs/archive/legacy/
mv docs/KNOWN_ISSUES.md docs/archive/legacy/  # After extraction
```

### Create Archive Index

`docs/archive/README.md`:
```markdown
# Archived Documentation

Historical documentation preserved for reference. May be outdated.

## Contents
- `bugs/` - Resolved bug documentation (BUG-005 through BUG-007)
- `research/` - Original paper extraction (superseded by docs/reference/)
- `specs/` - Completed TDD specifications
- `legacy/` - Old trackers (TODO.md, KNOWN_ISSUES.md)
```

---

## Phase 3: Create New Canonical Docs

### 3.1 docs/README.md (Docs Index)

Content:
- Navigation guide to all documentation
- Quick links to key docs
- "Start here" guidance

### 3.2 docs/reference/*.md (Refactored from research/)

| New File | Source | Changes |
|----------|--------|---------|
| `reference/meshnet.md` | `research/01-architecture.md` | Minor cleanup |
| `reference/dataset.md` | `research/02-dataset-preprocessing.md` | Minor cleanup |
| `reference/training.md` | `research/03-training-configuration.md` | Fix Hydra/AMP |
| `reference/metrics.md` | `research/05-evaluation-metrics.md` | Minor cleanup |
| `reference/variants.md` | `research/04-model-variants.md` | Minor cleanup |

### 3.3 docs/REPRODUCIBILITY.md (NEW)

Content:
- Exact 30-run protocol (3 folds × 10 restarts)
- Aggregation method (pooled per-subject scores)
- Seed handling
- Expected runtimes by hardware
- Success criteria

### 3.4 docs/TROUBLESHOOTING.md (NEW)

Sources: `KNOWN_ISSUES.md`, `docs/issues/TRAIN-001-runtime-estimates.md`, `docs/issues/OPT-001-download-performance.md`, `IO-DIRECTORY-STRUCTURE.md` troubleshooting sections

Content:
- Download slowness (OPT-001 summary)
- Sample count discrepancy (223 vs 224)
- MPS vs CUDA performance
- Cache/path pitfalls
- Common errors and fixes

### 3.5 CHANGELOG.md (NEW - at repo root)

Content:
- v0.1.0: Initial paper replication implementation
- Track user-visible changes going forward

### 3.6 SECURITY.md (NEW - at repo root)

Content:
- Security disclosure policy
- No known vulnerabilities
- Dependency update policy

---

## Phase 4: Verify and Sync

### 4.1 Verification Checklist

For each doc:

1. [ ] All CLI commands exist and work as documented
2. [ ] All file paths reference existing files (`src/arc_meshchop/...`)
3. [ ] No references to non-existent commands (export, etc.)
4. [ ] No references to unused systems (Hydra, etc.)
5. [ ] Sample counts match code (`--paper-parity`, `--include-tse`)
6. [ ] All code examples run successfully

### 4.2 Cross-Reference Verification

| Doc | Must Match |
|-----|------------|
| `README.md` | `src/arc_meshchop/cli.py` commands |
| `ARCHITECTURE.md` | Actual module structure |
| `TRAINING.md` | `src/arc_meshchop/training/trainer.py` |
| `DATA.md` | `src/arc_meshchop/data/huggingface_loader.py` |
| `CONTRIBUTING.md` | `pyproject.toml`, `Makefile` |
| `reference/meshnet.md` | `src/arc_meshchop/models/meshnet.py` |
| `reference/training.md` | `src/arc_meshchop/training/config.py` |
| `reference/metrics.md` | `src/arc_meshchop/evaluation/metrics.py` |

### 4.3 CLI Command Verification

Run each documented command:
```bash
uv run arc-meshchop --help
uv run arc-meshchop version
uv run arc-meshchop info
uv run arc-meshchop download --help
uv run arc-meshchop train --help
uv run arc-meshchop evaluate --help
uv run arc-meshchop experiment --help
uv run arc-meshchop hpo --help
uv run arc-meshchop validate --help
```

Verify NO export command exists:
```bash
uv run arc-meshchop export --help  # Should fail
```

---

## Phase 5: Cleanup and Finalize

### 5.1 Move research/ to archive

After extraction to reference/:
```bash
mv docs/research docs/archive/research
```

### 5.2 Update CLAUDE.md Documentation Section

```markdown
## Documentation

- `README.md` - Project overview and quick start
- `ARCHITECTURE.md` - System architecture and data flow
- `TRAINING.md` - Training guide
- `DATA.md` - Data pipeline
- `CONTRIBUTING.md` - Developer guide
- `docs/reference/` - Deep technical reference
- `docs/specs/` - Future work specifications
```

### 5.3 Final Review Checklist

- [ ] All canonical docs written
- [ ] All stale references fixed
- [ ] All stale docs archived
- [ ] CLAUDE.md updated
- [ ] No broken cross-references
- [ ] CI still passes

---

## Execution Order

| Phase | Description | Effort |
|-------|-------------|--------|
| 1 | Fix stale references | Medium |
| 2 | Create archive structure | Quick |
| 3 | Create new canonical docs | Main effort |
| 4 | Verify and sync | Validation |
| 5 | Cleanup and finalize | Quick |

### Dependencies

```
Phase 1 (fix stale) → Phase 2 (archive) → Phase 3 (create) → Phase 4 (verify) → Phase 5 (cleanup)
```

---

## What NOT to Touch

1. **`docs/specs/07-huggingface-spaces.md`** - Preserve as-is (future work)
2. **`docs/issues/`** - Keep active issues
3. **`_references/`** - Read-only external references
4. **`_literature/`** - Read-only source paper
5. **`configs/`** - Leave directory (may be used later), just fix doc claims

---

## Success Criteria

A developer should be able to:

1. Read `README.md` and understand the project in 2 minutes
2. Follow `CONTRIBUTING.md` and have a working dev environment in 10 minutes
3. Follow `TRAINING.md` and start training in 5 minutes
4. Find any technical detail in `docs/reference/` without source diving
5. **Never encounter a stale or contradictory doc**
6. **Never see a command that doesn't exist**

---

## Summary of Senior Review Fixes

| Issue | Fix |
|-------|-----|
| Missing root docs in audit | Added ARCHITECTURE.md, TRAINING.md, DATA.md, CONTRIBUTING.md to inventory |
| `export` command doesn't exist | Will replace CLI references with Python API usage (track CLI wrapper as optional future) |
| Hydra not actually used | Will remove claims from README.md, CONTRIBUTING.md, CLAUDE.md, research docs |
| Path inaccuracies | All paths now use `src/arc_meshchop/...` |
| VERIFY not defined | Removed; using consistent 6-status schema |
| KNOWN_ISSUES.md has value | Extract cohort discrepancy to TROUBLESHOOTING.md before archiving |
| Missing canonical docs | Added CHANGELOG.md, SECURITY.md, REPRODUCIBILITY.md, TROUBLESHOOTING.md |
| Checkpoint-path examples wrong | Will fix ARCHITECTURE.md + TRAINING.md examples to match fold-scoped outputs |
| Runtime estimates conflict | Will treat TRAIN-001 as measurement SSOT and update root docs accordingly |

---

## Next Steps

1. **Approve this plan**
2. **Execute Phase 1** - Fix stale references (can do immediately)
3. **Execute Phases 2-5** sequentially
4. **Final review** before merging
