# Archived Documentation

Historical documentation preserved for reference. May be outdated.

## Contents

### `BLOCKERS.md`
Original project blockers (historical, all resolved):
- `BLOCKERS.md`

### `bugs/`
Resolved bug documentation:
- `BUG-001-audit-findings.md` - Early audit findings (historical)
- `BUG-002-metric-aggregation.md` - Fixed: Metrics computed from pooled per-subject scores
- `BUG-003-runner-issues.md` - Fixed: Nested CV protocol + runner correctness
- `BUG-004-mask-paths-alignment.md` - Fixed: Mask path alignment (paper parity)
- `BUG-005-cache-key-ignores-mask-path.md` - Fixed: Cache key now includes mask_path
- `BUG-006-dataset-info-paths-are-cwd-relative.md` - Fixed: Paths resolved relative to data_dir
- `BUG-007-cli-experiment-variant-not-validated.md` - Fixed: Variant validated before experiment runs
- `NESTED-CV-PROTOCOL.md` - Protocol analysis (90 → 30 runs)

### `research/`
Original paper extraction (superseded by `docs/reference/`):
- `00-overview.md` - Paper overview and goals
- `01-architecture.md` - MeshNet architecture details
- `02-dataset-preprocessing.md` - Dataset and preprocessing
- `03-training-configuration.md` - Training hyperparameters
- `04-model-variants.md` - MeshNet-5/16/26 variants
- `05-evaluation-metrics.md` - DICE, AVD, MCC metrics
- `06-implementation-checklist.md` - Implementation checklist (historical)
- `audit_report.md` - Internal audit report (historical)

### `specs/`
Completed TDD specifications (implementation done):
- `IO-DIRECTORY-STRUCTURE.md` - Input/output directory structure
- `01-project-setup.md` - Initial project setup spec (historical)
- `02-meshnet-architecture.md` - MeshNet architecture spec (historical)
- `03-data-pipeline.md` - Data pipeline spec (historical)
- `04-training-infrastructure.md` - Training infrastructure spec (historical)
- `05-evaluation-metrics.md` - Evaluation metrics spec (historical)
- `06-model-export.md` - Model export spec (historical)
- `08-cross-platform.md` - Cross-platform support spec (historical)
- `FIX-001-bids-hub-integration.md` - BIDS Hub integration spec (historical)
- `local-01-hf-data-loader.md` - Local spec (historical)
- `local-02-training-cli.md` - Local spec (historical)
- `local-03-experiment-runner.md` - Local spec (historical)
- `local-04-paper-parity-validation.md` - Local spec (historical)
- `local-05-upstream-dataset-fixes.md` - Local spec (historical)
- `local-06-data-contract-hardening.md` - Local spec (historical)

### `legacy/`
Old work trackers:
- `TODO.md` - Original work tracker (mostly complete)
- `KNOWN_ISSUES.md` - Dataset issues (cohort discrepancy extracted to TROUBLESHOOTING.md)

---

## Note

These documents are preserved for historical context. For current documentation:
- Start at the repo root `README.md`
- See `docs/reference/` for technical details
- See `docs/TROUBLESHOOTING.md` for common issues
