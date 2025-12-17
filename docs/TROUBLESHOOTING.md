# Troubleshooting Guide

Common issues and solutions for ARC-MeshChop.

---

## Download Issues

### "Dataset info not found"

```
FileNotFoundError: Dataset info not found: data/arc/dataset_info.json
```

**Fix:** Run the download command first:
```bash
uv run arc-meshchop download --output data/arc
```

### Download is Slow (~25-45 minutes)

This is normal for first-time setup. The download:
1. Fetches ~273GB from HuggingFace Hub
2. Extracts and caches NIfTI files
3. Computes lesion volumes for stratification

**Note:** Subsequent `train` and `experiment` commands read from the cached `dataset_info.json` instantly.

### "HuggingFace quota exceeded"

**Fix:** Wait for rate limit reset or authenticate:
```bash
huggingface-cli login
```

---

## Sample Count Issues

### 223 vs 224 Samples

**Expected:** 223 samples (115 SPACE-2x + 108 SPACE no-accel)

The paper cites 224 (115 + 109), but OpenNeuro has 223 (115 + 108). This is likely a typo in the paper.

**Ground truth:**
- SPACE with 2x acceleration: 115 samples
- SPACE without acceleration: 108 samples
- Turbo-spin echo (excluded): 5 samples

### Including TSE Samples

By default, turbo-spin echo samples are excluded per paper methodology. To include them:
```bash
uv run arc-meshchop download --include-tse
```

This gives 228 total samples (223 + 5 TSE).

---

## Training Issues

### "Checkpoint not found"

```
FileNotFoundError: No such file: experiments/meshnet26/fold_0_restart_0/final.pt
```

**Fix:** Training did not complete. Check:
1. Logs for errors during training
2. Disk space for checkpoint storage
3. Re-run training for that fold/restart

### NaN or Inf Loss

Possible causes:
1. **Learning rate too high**: Reduce from 0.001
2. **Bad sample**: Check data preprocessing
3. **Numerical instability**: Enable gradient clipping

### Low DICE Score

If DICE is significantly below 0.876:
1. **Check class weights**: Should be [0.5, 1.0] for [background, lesion]
2. **Check preprocessing**: Must be 256^3 @ 1mm isotropic
3. **Check epochs**: Need 50 full epochs
4. **Run more restarts**: Variance is expected (std ~0.016)

### Training Too Slow

| Platform | Expected Per-Run Time |
|----------|----------------------|
| NVIDIA A100 | ~1-2 hours |
| NVIDIA RTX 4090 | ~6-12 hours |
| Apple M-series | ~62 hours |

See [TRAIN-001](issues/TRAIN-001-runtime-estimates.md) for detailed estimates.

**Speedup options:**
- Use CUDA if available (FP16 automatic)
- Reduce epochs for debugging (--epochs 5)
- Use MeshNet-16 instead of MeshNet-26 (faster, similar accuracy)

---

## MPS (Apple Silicon) Issues

### "MPS backend not available"

```python
RuntimeError: MPS backend not available
```

**Fix:**
1. Update PyTorch: `uv pip install --upgrade torch`
2. Update macOS to 12.3+
3. Check `torch.backends.mps.is_available()`

### "MPS op not implemented"

Some PyTorch ops aren't supported on MPS yet.

**Workaround:** Fall back to CPU:
```bash
export TORCH_DEVICE=cpu
uv run arc-meshchop train ...
```

### FP16 on MPS

MPS doesn't benefit from FP16 for this workload. The code automatically uses FP32 on MPS.

---

## Cache Issues

### Stale Cache

If preprocessing changes, delete the cache:
```bash
rm -rf data/arc/cache/
```

The cache will regenerate on next training run.

### Disk Full

Cache is the largest component:
```bash
du -sh data/arc/cache/   # Check size
rm -rf data/arc/cache/   # Clear (will regenerate)
```

**Expected sizes:**
- `data/arc/cache/`: ~50GB (preprocessed 256^3 volumes)
- `experiments/`: ~150MB (30 runs with checkpoints)

---

## Evaluation Issues

### Results Don't Match Paper

Check these in order:
1. **Sample count**: `uv run python -c "import json; print(json.load(open('data/arc/dataset_info.json'))['num_samples'])"` should print `223`
2. **Preprocessing**: 256^3 @ 1mm isotropic (no skull stripping)
3. **Hyperparameters**: lr=0.001, weight_decay=3e-5, eps=1e-4
4. **Loss weights**: [0.5, 1.0] for [background, lesion]
5. **Epochs**: 50 full epochs
6. **Restarts**: 10 per fold (30 total)

### Statistical Significance

With only 3 outer folds, variance is expected:
- DICE std: ~0.016
- AVD std: ~0.036
- MCC std: ~0.030

Run all 30 training runs for statistically meaningful results.

---

## Export Issues

### No CLI Export Command

Export is available via Python API only, not CLI:

```python
from pathlib import Path

import torch

from arc_meshchop.export import export_to_onnx
from arc_meshchop.models import meshnet_26

checkpoint = torch.load(
    "outputs/train/fold_0_0/best.pt",
    map_location="cpu",
    weights_only=True,
)

model = meshnet_26()
model.load_state_dict(checkpoint["model_state_dict"])

export_to_onnx(model, Path("exports/meshnet_26.onnx"))
```

---

## Environment Issues

### "ModuleNotFoundError: arc_meshchop"

**Fix:** Install in development mode:
```bash
uv sync --all-extras
```

Or:
```bash
pip install -e .
```

### "Command not found: arc-meshchop"

**Fix:** Use `uv run`:
```bash
uv run arc-meshchop info
```

Or activate the virtual environment:
```bash
source .venv/bin/activate
arc-meshchop info
```

---

## Getting Help

1. Check existing documentation:
   - [README.md](../README.md)
   - [TRAINING.md](../TRAINING.md)
   - [docs/reference/](reference/)

2. Open an issue: https://github.com/The-Obstacle-Is-The-Way/arc-meshchop/issues

3. Check the paper: "State-of-the-Art Stroke Lesion Segmentation at 1/1000th of Parameters"
