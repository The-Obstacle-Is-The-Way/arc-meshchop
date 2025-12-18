# BUG-008: `dataset_info.json` Paths Fail to Resolve (P0)

**Priority:** P0
**Status:** Fixed
**Impacts:** `arc-meshchop train`, `arc-meshchop experiment`, `arc-meshchop hpo`, `arc-meshchop evaluate`

---

## Symptom

Training/evaluation can fail with file-not-found errors when loading NIfTI files, because paths get resolved to a duplicated prefix like:

```text
.../data/arc/data/arc/cache/nifti_cache/...
```

This prevents runs from starting (or breaks on the first batch).

---

## Root Cause

`arc-meshchop download` historically wrote `dataset_info.json` entries as repo-root relative paths like:

```text
data/arc/cache/nifti_cache/<file>.nii.gz
```

Meanwhile, `resolve_dataset_path(data_dir, path)` assumed all relative paths are relative to `data_dir` (the directory containing `dataset_info.json`) and always joined:

```py
data_dir / path
```

So with `data_dir=/.../data/arc` and `path=data/arc/cache/...`, the join produced `/.../data/arc/data/arc/cache/...` which does not exist.

---

## Fix

1. **Backward-compatible resolution**
   - `src/arc_meshchop/utils/paths.py`: resolve relative to `data_dir` first, then fall back to searching parent directories of `data_dir` to support repo-root relative entries.

2. **Write portable paths going forward**
   - `src/arc_meshchop/cli.py`: `download` now writes paths **relative to `--output`** (e.g., `cache/nifti_cache/...`) so downstream code can reliably resolve them.

---

## Verification

- Added regression test: `tests/test_cli_bug_008.py`
- `uv run pytest` passes.
- Resolving real `data/arc/dataset_info.json` entries now yields `0` missing paths.

---

## Migration Notes

- Existing `dataset_info.json` files in the old format are still supported.
- New downloads will produce the new (portable) path format.
