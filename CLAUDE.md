# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This repository replicates the MeshNet stroke lesion segmentation paper:
**"State-of-the-Art Stroke Lesion Segmentation at 1/1000th of Parameters"** (Fedorov et al.)

The goal is to achieve 0.876 DICE score with only 147K parameters on the ARC dataset.

## Commands

```bash
# Install dependencies
uv sync --all-extras

# Check device availability (CUDA/MPS/CPU)
uv run arc-meshchop info

# Run tests
uv run pytest

# Run single test
uv run pytest tests/test_models/test_meshnet.py::test_parameter_count -v

# Lint
uv run ruff check src tests

# Type check
uv run mypy src tests

# Format
uv run ruff format src tests

# Pre-commit hooks
uv run pre-commit install
uv run pre-commit run --all-files

# Full CI locally
make ci
```

## Architecture

### MeshNet (NEW 10-layer symmetric pattern)

```
Dilation: 1 → 2 → 4 → 8 → 16 → 16 → 8 → 4 → 2 → 1
          └─── encoder ───┘     └─── decoder ───┘

Each layer: Conv3D(k=3, p=d, d=d) → BatchNorm3d → ReLU
Final layer: Conv3D(k=1) → Output (2 classes)
```

**DO NOT** use the old BrainChop 8-layer pattern: `[1, 1, 2, 4, 8, 16, 1, 1]`

### Model Variants

| Variant | Channels | Parameters |
|---------|----------|------------|
| MeshNet-5 | 5 | 5,682 |
| MeshNet-16 | 16 | 56,194 |
| MeshNet-26 | 26 | 147,474 |

### Training

- **Optimizer:** AdamW (lr=0.001, weight_decay=3e-5, eps=1e-4)
- **Scheduler:** OneCycleLR (1% warmup)
- **Loss:** CrossEntropy(weights=[0.5, 1.0], label_smoothing=0.01)
- **Precision:** FP16 on CUDA, FP32 on MPS/CPU (auto-detected)
- **Batch size:** 1 (full 256³ volumes)

### Cross-Platform Support

MeshNet is small enough (147K params) to train on any platform:

| Platform | Device | Precision | Notes |
|----------|--------|-----------|-------|
| Linux/Windows | CUDA | FP16 | Full support |
| Mac (M1/M2/M3/M4) | MPS | FP32 | Auto-fallback, still fast |
| Any | CPU | FP32 | Fallback |

**Key utilities:**
- `from arc_meshchop.utils.device import get_device` - Auto-detects best device
- Training uses `torch.amp` (not `torch.cuda.amp`) for cross-platform AMP
- `GradScaler` disabled on MPS/CPU (not needed)

## Testing

Tests use TDD approach with synthetic data:
- Minimal NIfTI files (2×2×2 voxels) for fast execution
- Parameter count verification (must match paper exactly)
- Shape verification (256³ → 256³)
- Memory estimation tests

## Key Files

| File | Purpose |
|------|---------|
| `src/arc_meshchop/models/meshnet.py` | MeshNet architecture |
| `src/arc_meshchop/data/dataset.py` | ARC dataset loader |
| `src/arc_meshchop/training/trainer.py` | Training loop |
| `src/arc_meshchop/evaluation/metrics.py` | DICE, AVD, MCC |
| `configs/` | Configuration presets (unused) |

## Reference Implementations

> **Note:** `_references/` contains local read-only copies for study. Do NOT import from there.

- `_references/brainchop/` - VERIFIED architecture (JS-based, read-only reference)
- `_references/niivue/` - WebGL visualization (JS-based, read-only reference)

## Documentation

- `README.md` - Project overview and quick start
- `ARCHITECTURE.md` - System architecture and data flow
- `TRAINING.md` - Training guide
- `DATA.md` - Data pipeline
- `CONTRIBUTING.md` - Developer guide
- `docs/README.md` - Documentation index
- `docs/reference/` - Deep technical reference (MeshNet, dataset, training, metrics)
- `docs/REPRODUCIBILITY.md` - Exact paper replication protocol
- `docs/TROUBLESHOOTING.md` - Common issues and solutions
- `docs/specs/` - Future work specifications
- `docs/archive/` - Historical documentation (bugs, research notes)
- `_literature/markdown/stroke_lesion_segmentation/` - The actual paper

## Important Notes

1. **NO GRADIO** - Deployment uses Docker-based HF Spaces, not Gradio
2. **NO SKULL STRIPPING** - Paper only uses mri_convert resampling
3. **4 BINS, NOT 5** - Quintile terminology but 4 ranges from 5 cutoffs
4. **BN/ReLU/padding** - Inferred from BrainChop, not stated in paper
