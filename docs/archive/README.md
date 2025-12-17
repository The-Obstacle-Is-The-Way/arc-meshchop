# Archived Documentation

Historical documentation preserved for reference. May be outdated.

## Contents

### `bugs/`
Resolved bug documentation:
- `BUG-005-cache-key-ignores-mask-path.md` - Fixed: Cache key now includes mask_path
- `BUG-006-dataset-info-paths-are-cwd-relative.md` - Fixed: Paths resolved relative to data_dir
- `BUG-007-cli-experiment-variant-not-validated.md` - Fixed: Variant validated before experiment runs

### `research/`
Original paper extraction (superseded by `docs/reference/`):
- `00-overview.md` - Paper overview and goals
- `01-architecture.md` - MeshNet architecture details
- `02-dataset-preprocessing.md` - Dataset and preprocessing
- `03-training-configuration.md` - Training hyperparameters
- `04-model-variants.md` - MeshNet-5/16/26 variants
- `05-evaluation-metrics.md` - DICE, AVD, MCC metrics

### `specs/`
Completed TDD specifications (implementation done):
- `IO-DIRECTORY-STRUCTURE.md` - Input/output directory structure

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
