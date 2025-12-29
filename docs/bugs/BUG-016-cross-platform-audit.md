# BUG-016: Cross-Platform Audit (Mac MPS to Windows/NVIDIA CUDA)

**Status:** AUDIT COMPLETE - ISSUES DOCUMENTED
**Date:** 2025-12-29
**Priority:** P2 (Pre-emptive - no active failures yet)

## Summary

Pre-emptive audit of the codebase before running on Windows/NVIDIA RTX 4090 after all prior development was done on Mac MPS. Documents all potential cross-platform friction points.

## Test Environments

| Environment | Device | OS | Status |
|-------------|--------|-----|--------|
| Mac MPS | Apple Silicon M1/M2/M3/M4 | macOS | Tested (3 runs complete) |
| Windows CUDA | RTX 4090 24GB | WSL2 Ubuntu | **NEW - Testing now** |
| Linux CUDA | A100 80GB | Ubuntu | Paper reference (not tested) |

---

## CRITICAL ISSUES (Must Fix Before Production)

### None found

The codebase is fundamentally cross-platform safe. No platform-specific hard failures were found in the core training loop.

---

## MEDIUM ISSUES (Should Fix)

### M1: Unconditional CUDA Seeding

**File:** `src/arc_meshchop/utils/seeding.py:33-37`

```python
torch.cuda.manual_seed_all(seed)

if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Problem:** These lines run unconditionally even on MPS/CPU where CUDA isn't available.

**Actual Impact:**
- `torch.cuda.manual_seed_all()` is a no-op on non-CUDA (safe but confusing)
- `torch.backends.cudnn.*` settings are no-ops on non-CUDA (safe)

**Risk Level:** Low - no runtime errors, but misleading code

**Fix:**
```python
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

if deterministic and torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

### M2: Loss Weight Dtype Mismatch (Potential)

**File:** `src/arc_meshchop/training/loss.py:60-61`

```python
if self._loss_fn is None or self.class_weights.device != logits.device:
    self.class_weights = self.class_weights.to(logits.device)
```

**Problem:** Moves weights to correct device but does NOT convert dtype. When using FP16 (autocast), logits are FP16 but weights stay FP32.

**Actual Impact:**
- If called *outside* autocast with `logits.dtype == float16` and `class_weights.dtype == float32`, PyTorch raises:
  `RuntimeError: expected scalar type Half but found Float`
- In the current training loop, loss is computed inside `torch.amp.autocast(...)`, and `cross_entropy` runs in FP32 under autocast, so this does not currently crash.

**Risk Level:** Low-Medium - safe in current training loop, but brittle if loss is reused outside autocast

**Fix:**
```python
self.class_weights = self.class_weights.to(logits.device, dtype=logits.dtype)
```

---

### M3: CI/CD Only Tests Ubuntu

**File:** `.github/workflows/ci.yml:24,50,73`

```yaml
runs-on: ubuntu-latest
```

**Problem:** All CI jobs run on Ubuntu only. Windows and macOS not tested in CI.

**Actual Impact:**
- Windows-specific bugs would slip through CI
- GPU tests not run (no CUDA in GitHub Actions runners)

**Risk Level:** Medium - no automated cross-platform regression testing

**Fix:** Add Windows matrix:
```yaml
strategy:
  matrix:
    os: [ubuntu-latest, windows-latest, macos-latest]
    python-version: ["3.10", "3.11", "3.12"]
```

---

### M4: Makefile Uses Bash Commands

**File:** `Makefile:75-85`

```makefile
clean:
    rm -rf build/
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
```

**Problem:** `rm -rf` and `find` are Bash commands that don't work in Windows cmd.exe.

**Actual Impact:**
- `make clean` fails on native Windows CMD
- Works fine in WSL2 (which we're using)

**Risk Level:** Low - WSL2 handles this fine

**Fix (optional):** Create `clean.py` script or use uv/Python for cross-platform cleaning.

---

### M5: Pre-commit Mypy Hook Uses `language: system` (Host `uv` Dependency)

**File:** `.pre-commit-config.yaml:27-28`

```yaml
entry: uv run mypy
language: system
```

**Problem:** `language: system` runs `uv run mypy` from the host environment. This requires `uv` to be installed and on `PATH` on every dev machine (including Windows).

**Actual Impact:**
- Pre-commit runs can fail on fresh machines if `uv` isn't installed globally
- Works fine in WSL2 (and in Windows if `uv` is installed and discoverable)

**Risk Level:** Low - WSL2 handles this

---

### M6: HuggingFace Hub Cache Location Can Ignore `cache_dir` (Disk-Fill Risk on WSL2)

**Evidence:** `huggingface_hub` warnings during download indicated writes to the default hub cache under:
`~/.cache/huggingface/hub/.../blobs` even when `arc-meshchop download` is passed a project-local `cache_dir`.

**Actual Impact:**
- Large downloads can fill the WSL root filesystem (`ENOSPC`), causing partial/corrupt writes and cascading failures
- Confusing because the CLI appears to cache to `data/arc/cache/`, but the hub snapshot cache may still go to `~/.cache/huggingface/hub/`

**Risk Level:** Medium - can hard-fail downloads/training due to disk full

**Fix (recommended):**
- Add an explicit CLI option (or documented env var) to set `HF_HOME` / `HUGGINGFACE_HUB_CACHE` during downloads so hub artifacts land next to `data/arc/cache/` (or on a large disk).
- Add a preflight disk-space check before starting download.

---

### M7: Docs Disagree on Dataset Size / Disk Requirements

**Evidence:**
- `DATA.md:308` says: “full ARC dataset is ~50GB”
- `docs/TROUBLESHOOTING.md:23` says: “Fetches ~273GB from HuggingFace Hub”

**Actual Impact:**
- Users under-provision disk and hit `ENOSPC` during download (especially on WSL2 root filesystem)
- Conflicting expectations across docs creates setup friction

**Risk Level:** Medium - directly causes broken first-run experience

**Fix (docs):**
- Normalize the story into a single breakdown:
  - Hub download size (parquet + metadata)
  - Extracted/NIfTI cache size (project cache)
  - Preprocessed cache size (256³ volumes)

---

## LOW ISSUES (Nice to Have)

### L1: TFJS Export Disabled on Mac AND Windows

**File:** `tests/test_export/test_pipeline.py:322-329`

```python
@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="TFJS export requires TensorFlow which has no ARM Mac wheels",
)
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="TFJS export requires TensorFlow which may have issues on Windows",
)
```

**Status:** Intentional - TensorFlow ecosystem issues. TFJS export only works on Linux.

---

### L2: Hardcoded Test Paths

**File:** `tests/test_data/test_huggingface_loader.py:116-227`

```python
image_path=Path("/path/image.nii.gz"),
```

**Status:** Safe - Using `Path()` constructor handles separators. These are mock data paths never accessed on disk.

---

### L3: No GPU Tests in CI

**File:** `.github/workflows/ci.yml:94`

```yaml
run: uv run pytest -m "not slow and not gpu"
```

**Status:** Intentional - GitHub Actions doesn't have GPU runners by default. GPU tests require self-hosted runners or paid GPU runners.

---

## VERIFIED SAFE (No Action Needed)

### Device Detection
**File:** `src/arc_meshchop/utils/device.py`

Properly handles CUDA > MPS > CPU fallback with guards:
```python
if preferred == "cuda" and torch.cuda.is_available():
    # CUDA path
elif preferred == "mps" and torch.backends.mps.is_available():
    if _mps_is_functional():
        # MPS path
else:
    # CPU fallback
```

---

### Mixed Precision (AMP)
**File:** `src/arc_meshchop/training/trainer.py:99-113`

Properly uses device type checks:
```python
if self.device.type == "cuda":
    self.scaler = GradScaler("cuda", enabled=config.use_fp16)
    self._amp_dtype = torch.float16
elif self.device.type == "mps":
    # FP32 fallback
else:
    # FP32 fallback
```

---

### Autocast Context
**File:** `src/arc_meshchop/training/trainer.py:269-271`

Uses `self.device.type` for cross-platform autocast:
```python
with autocast(
    device_type=self.device.type,
    dtype=self._amp_dtype,
```

---

### Path Handling
**All production code** uses `pathlib.Path` - no hardcoded `/` separators.

---

### Tempfile Handling
**Files:** `src/arc_meshchop/data/huggingface_loader.py`, `src/arc_meshchop/training/hpo.py`

Uses `tempfile.gettempdir()` and `tempfile.mkdtemp()` - cross-platform safe.

---

### pin_memory Auto-Detection
**File:** `src/arc_meshchop/data/dataset.py:242`

```python
if pin_memory is None:
    pin_memory = torch.cuda.is_available()
```

Only enables pin_memory on CUDA (correct).

---

### MPS Functional Test
**Files:** `src/arc_meshchop/utils/device.py:74-85`, `tests/conftest.py:15-22`

MPS tensor creation is wrapped in try-except and only runs when MPS is available. Safe on Windows/CUDA.

---

## Summary Table

| ID | Issue | File | Severity | Status |
|----|-------|------|----------|--------|
| M1 | Unconditional CUDA seeding | `utils/seeding.py:33-37` | Medium | **FIXED** |
| M2 | Loss weight dtype | `training/loss.py:60-61` | Low-Med | **FIXED** |
| M3 | Ubuntu-only CI | `.github/workflows/ci.yml` | Medium | Backlog |
| M4 | Makefile bash commands | `Makefile:75-85` | Low | OK for WSL2 |
| M5 | Pre-commit mypy uses host `uv` | `.pre-commit-config.yaml` | Low | OK if `uv` installed |
| M6 | HF hub cache may ignore `cache_dir` | `cli.py`, `huggingface_loader.py` | Medium | **FIXED** |
| M7 | Docs disagree on dataset size | `DATA.md`, `docs/TROUBLESHOOTING.md` | Medium | **FIXED** |
| L1 | TFJS disabled on Win/Mac | `tests/test_export/` | Low | Intentional |
| L2 | Test hardcoded paths | `tests/test_data/` | Low | Safe |
| L3 | No GPU tests in CI | `.github/workflows/` | Low | Intentional |

---

## Recommendations

### Remaining Work
- M3: Consider adding Windows/macOS CI matrix for broader test coverage

### Not Required
- M4: We're using WSL2, so bash commands work fine
- M5: OK if contributors have `uv` installed
- L1/L2/L3: These are intentional design decisions

---

## Files Investigated

### Key Modules/Configs
- `src/arc_meshchop/utils/device.py` - Device detection
- `src/arc_meshchop/utils/seeding.py` - RNG seeding
- `src/arc_meshchop/training/trainer.py` - Mixed precision + training loop
- `src/arc_meshchop/training/loss.py` - Loss function
- `src/arc_meshchop/data/huggingface_loader.py` - HuggingFace integration
- `src/arc_meshchop/data/dataset.py` - DataLoader settings
- `src/arc_meshchop/cli.py` - CLI entrypoints
- `.github/workflows/ci.yml`, `Makefile`, `.pre-commit-config.yaml`, `pyproject.toml`

---

## Conclusion

The codebase is **cross-platform ready**. No critical blockers for Windows/CUDA. The identified medium issues are code quality improvements, not functional bugs.

**Expected behavior on RTX 4090:**
- Device: CUDA (auto-detected)
- Precision: FP16 with GradScaler (enabled)
