# BUG-006: `dataset_info.json` Paths Are CWD-Relative (Fragile Launch Context)

> **Severity:** P2 (run-stopper; usually fails loudly)
>
> **Status:** FIXED (fix/bug-005-006-007)
>
> **Date:** 2025-12-16
>
> **Affected:** `src/arc_meshchop/cli.py`, `src/arc_meshchop/experiment/runner.py`, `src/arc_meshchop/training/hpo.py`

## Summary

`arc-meshchop download` writes `dataset_info.json` with **relative** `image_paths` / `mask_paths` (e.g., `data/arc/cache/...`). Training/eval code constructs `Path(p)` directly from these strings, which makes them relative to the **current working directory**, not relative to `--data-dir` or the `dataset_info.json` location.

## Why This Can Ruin Runs

Launching `arc-meshchop train/experiment/evaluate/hpo` from a different CWD (common in tmux panes, schedulers, or scripts) can cause `FileNotFoundError` when the dataset is first accessed, wasting time and potentially invalidating automation.

## Evidence

- `dataset_info.json` generation: `src/arc_meshchop/cli.py` (`download()` command) stores paths as returned by the loader.
- Path use sites treat them as CWD-relative: `Path(p)` without resolving against `data_dir` / `info_path.parent`.

## Mitigations

- **Operational:** run commands from repo root (where `data/arc/...` resolves correctly).
- **Code fix (recommended):**
  - Store paths relative to `output_dir` (e.g., `cache/nifti_cache/...`) during download, and resolve relative to `info_path.parent`.
  - Add backward-compatible resolution for existing files (try CWD, then `info_path.parent`, then parents).
