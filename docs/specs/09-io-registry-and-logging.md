# Spec 09: IO Registry and Logging

## Summary

Create a canonical IO registry, standardize output directories, and persist
per-run logs and manifests.

---

## Goals

- Single source of truth for input/output paths.
- Consistent default output roots across CLI and programmatic APIs.
- Persistent log files per run.
- A run manifest that ties together config, dataset info, and outputs.

---

## Non-Goals

- Changing preprocessing or model behavior.
- Rewriting the experiment protocol.

---

## Current State

- CLI defaults write to `outputs/` and `experiments/`.
- TrainingConfig default writes to `checkpoints/`.
- HPO uses temporary directories for checkpoints.
- Logging is stdout-only; no log files or run manifests.

---

## Proposed Design

### 1. IO Registry

Introduce `arc_meshchop.utils.io_registry`:

- Central definitions for:
  - dataset root
  - cache root
  - run output root
  - logs root
- Helper APIs to resolve paths (with overrides from CLI flags).

### 2. Run ID and Manifest

- Add a `run_id` generated per command (timestamp + command + short hash).
- Write `run.json` alongside results with:
  - command name
  - config parameters
  - seeds
  - data_dir and dataset_info hash
  - output paths

### 3. Persistent Logs

- Add `--log-dir` option to CLI commands.
- Create a log file per run: `logs/<run_id>.log`.
- Keep stdout logging, but also attach a file handler.

---

## Data and IO

- New directory: `logs/` (configurable)
- New file: `run.json` (per run)
- Existing outputs remain supported, but new defaults should use registry paths.

---

## Migration Plan

- Keep old paths as valid inputs.
- New runs use registry paths; old runs remain readable.
- Update docs (`docs/reference/io-registry.md`) to reflect canonical paths.

---

## Verification

- Run `train` and `experiment` and confirm:
  - Log files are created.
  - `run.json` exists and includes expected fields.
- Add unit tests that validate IO registry path resolution.
