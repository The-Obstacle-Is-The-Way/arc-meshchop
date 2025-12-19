# IO Registry

This document maps the main input/output paths used by ARC-MeshChop.
All paths are repo-relative by default unless overridden by CLI flags.

---

## Conventions

- Paths are relative to the repo root unless absolute paths are provided.
- `data/arc/` is the default dataset root.
- `outputs/` is for single-run training.
- `experiments/` is for multi-run paper replication.

---

## Inputs

| Path | Producer | Consumer | Notes |
|------|----------|----------|-------|
| `data/arc/` | `arc-meshchop download --output` | all commands | Dataset root |
| `data/arc/dataset_info.json` | download | train/eval/experiment/hpo | Manifest of paths and metadata |
| `data/arc/cache/nifti_cache/` | download | data loader | Cached NIfTI files |
| `data/arc/cache/fold_*` | train | train | Preprocessed training cache |
| `data/arc/cache/outer_*` | experiment | experiment | Preprocessed train/test cache |

---

## Outputs

| Path | Producer | Consumer | Notes |
|------|----------|----------|-------|
| `outputs/train/` | `arc-meshchop train` | user | Default single-run output root |
| `outputs/<name>/fold_*/` | train | evaluate | Checkpoints + results |
| `experiments/` | `arc-meshchop experiment` | validate | Default replication output root |
| `experiments/<run>/fold_*_restart_*/` | experiment | validate | Per-run outputs |
| `experiments/hpo/` | `arc-meshchop hpo` | user | HPO results (best_params.json) |
| `checkpoints/` | TrainingConfig default | user | Used when TrainingConfig is created without CLI overrides |
| `exports/<name>/` | export pipeline | user | Exported ONNX/TFJS artifacts (path is configurable) |

---

## Logs

- Current behavior: logs go to stdout/stderr only.
- Ad-hoc files like `download.log` and `training.log` may exist but are not produced by code.
- Planned: a dedicated `logs/` directory with per-run log files (see Spec 09).

---

## CLI Path Overrides

Common flags that change IO roots:

- `arc-meshchop download --output <dir>`
- `arc-meshchop train --data-dir <dir> --output <dir>`
- `arc-meshchop evaluate --data-dir <dir> --output <file>`
- `arc-meshchop experiment --data-dir <dir> --output <dir>`
- `arc-meshchop hpo --data-dir <dir> --output <dir>`

---

## Known Gaps

- Output paths are defined in multiple places (CLI defaults, TrainingConfig default).
- There is no centralized registry in code, so paths can drift over time.
- No run manifest ties together config, dataset info, and outputs.

See `docs/specs/09-io-registry-and-logging.md` for the proposed consolidation.
