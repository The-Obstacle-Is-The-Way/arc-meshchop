# ARC-MeshChop

**MeshNet stroke lesion segmentation on the Aphasia Recovery Cohort (ARC)**

Replicating ["State-of-the-Art Stroke Lesion Segmentation at 1/1000th of Parameters"](https://arxiv.org/html/2503.05531v1) (Fedorov et al.) — a tiny 3D CNN that beats transformers.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         WHY THIS EXISTS                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   MeshNet-26 (147K params)  vs  Swin-UNETR (18M params)                 │
│                                                                         │
│   DICE: 0.876                    DICE: 0.859                            │
│   Size: 0.6 MB                   Size: 70+ MB                           │
│                                                                         │
│   Runs in browser (brainchop.org)   Needs GPU server                    │
│                                                                         │
│   The tiny model WINS.                                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install
git clone https://github.com/The-Obstacle-Is-The-Way/arc-meshchop.git
cd arc-meshchop
uv sync --all-extras

# Check your hardware
uv run arc-meshchop info

# Download dataset (goes to ~/.cache/huggingface/hub/)
uv run arc-meshchop download

# Train MeshNet-26
uv run arc-meshchop train --channels 26 --epochs 50
```

## Hardware Requirements

| Platform | Device | Works? | Notes |
|----------|--------|--------|-------|
| **Mac M1/M2/M3/M4** | MPS | ✅ | FP32, ~6-15 hours |
| **Linux/Windows** | CUDA | ✅ | FP16, faster |
| **Any** | CPU | ✅ | Slow but works |

**This model is designed to be tiny.** You don't need an A100.

## The Model

```
MeshNet-26: 10-layer dilated 3D CNN

Dilation: 1 → 2 → 4 → 8 → 16 → 16 → 8 → 4 → 2 → 1
          └─── encoder ───┘     └─── decoder ───┘

Input:  256 × 256 × 256 × 1  (T2-weighted MRI)
Output: 256 × 256 × 256 × 2  (background + lesion)

Parameters: 147,474  (~0.6 MB)
```

## The Dataset

We use the [Aphasia Recovery Cohort (ARC)](https://huggingface.co/datasets/hugging-science/arc-aphasia-bids):

| Metric | Full Dataset | Paper Subset |
|--------|--------------|--------------|
| Subjects | 230 | — |
| Sessions | 902 | — |
| T2w scans | 447 | 224 |
| With lesion masks | — | 224 |

The paper filters to 224 samples: 115 SPACE-2x + 109 SPACE (no turbo-spin-echo).

## Documentation

| Document | Purpose |
|----------|---------|
| [TRAINING.md](TRAINING.md) | What we're training and why |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design and data flow |
| [DATA.md](DATA.md) | Data pipeline explained |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |
| [docs/specs/](docs/specs/) | TDD specifications |
| [docs/research/](docs/research/) | Paper analysis |

## Commands

```bash
# Development
uv run pytest                    # Run tests
uv run ruff check src tests      # Lint
uv run mypy src tests            # Type check
make ci                          # Run all checks

# Training
uv run arc-meshchop download     # Get dataset
uv run arc-meshchop train        # Train model
uv run arc-meshchop evaluate     # Evaluate checkpoint
uv run arc-meshchop export       # Export to ONNX
```

## Target Performance

From the paper (Table 1):

| Metric | MeshNet-26 | Swin-UNETR | U-MAMBA |
|--------|------------|------------|---------|
| **DICE** | **0.876** | 0.859 | 0.870 |
| Parameters | 147K | 18M | 7M |
| Ratio | 1× | 124× | 50× |

## Project Structure

```
arc-meshchop/
├── src/arc_meshchop/
│   ├── data/           # Dataset loading & preprocessing
│   ├── models/         # MeshNet architecture
│   ├── training/       # Training loop, loss, optimizer
│   ├── evaluation/     # Metrics (DICE, AVD, MCC)
│   ├── export/         # ONNX export
│   └── cli.py          # Command-line interface
├── tests/              # Test suite
├── configs/            # Hydra configuration
└── docs/               # Documentation
```

## Citation

```bibtex
@article{fedorov2024meshnet,
  title={State-of-the-Art Stroke Lesion Segmentation at 1/1000th of Parameters},
  author={Fedorov, A. and others},
  journal={arXiv preprint},
  year={2024}
}

@article{gibson2024arc,
  title={A large-scale longitudinal multimodal neuroimaging dataset for aphasia},
  author={Gibson, M. and others},
  journal={Scientific Data},
  volume={11},
  year={2024},
  doi={10.1038/s41597-024-03819-7}
}
```

## License

Apache 2.0. Dataset is CC0 (public domain).
