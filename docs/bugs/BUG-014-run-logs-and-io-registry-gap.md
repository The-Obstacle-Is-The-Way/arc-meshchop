# BUG-014: Run Artifacts and Logs Are Not Persisted Consistently (P3)

**Priority:** P3
**Status:** Open
**Impacts:** `arc-meshchop train`, `arc-meshchop experiment`, `arc-meshchop hpo`, `arc-meshchop evaluate`

---

## Symptom

- Logs are only emitted to stdout, so diagnostics are lost after long runs.
- Output locations are inconsistent (`outputs/`, `experiments/`, `checkpoints/`, tmp dirs).
- There is no run manifest tying together config, dataset version, and outputs.

This makes debugging, cleanup, and reproducibility harder than necessary.

---

## Root Cause

- Logging is configured only via `logging.basicConfig` in the CLI (stdout only).
- Output paths are defined in multiple modules without a shared registry.
- HPO uses `tempfile.mkdtemp()` for checkpoints, which are not tracked or cleaned.

---

## Suggested Fix

1. Introduce a canonical IO registry (documented and used by code).
2. Add persistent log files per run (`logs/<run-id>.log`).
3. Write a `run.json` manifest containing:
   - command/config
   - seed(s)
   - data_dir and dataset_info hash
   - output paths
4. Provide a cleanup script or documented cleanup procedure.

---

## Verification

- Run `train` and `experiment` and confirm:
  - Log files are written to a predictable directory.
  - Each run produces a manifest with config and output locations.
  - No artifacts are left in temporary directories.
