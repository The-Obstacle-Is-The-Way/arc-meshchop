# Contributing to ARC-MeshChop

Thanks for your interest in contributing! This document explains how to get started.

---

## Development Setup

```bash
# Clone the repo
git clone https://github.com/The-Obstacle-Is-The-Way/arc-meshchop.git
cd arc-meshchop

# Install with dev dependencies
uv sync --all-extras

# Install pre-commit hooks
uv run pre-commit install

# Verify setup
uv run arc-meshchop info
make ci
```

---

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

Follow the existing code style. Key points:
- Type hints on all functions
- Google-style docstrings
- 100 character line limit

### 3. Run Checks

```bash
# Run all checks (recommended)
make ci

# Or individually:
uv run ruff check src tests     # Lint
uv run ruff format src tests    # Format
uv run mypy src tests           # Type check
uv run pytest                   # Test
```

### 4. Commit

```bash
git add .
git commit -m "feat: add your feature"
```

Commit message format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `refactor:` Code change that neither fixes nor adds
- `test:` Adding tests
- `chore:` Maintenance

### 5. Push and PR

```bash
git push origin feature/your-feature-name
gh pr create
```

---

## Code Style

### Python

```python
def calculate_dice(
    prediction: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
) -> float:
    """Calculate DICE coefficient between prediction and target.

    Args:
        prediction: Binary prediction tensor.
        target: Binary ground truth tensor.
        smooth: Smoothing factor to avoid division by zero.

    Returns:
        DICE coefficient between 0 and 1.
    """
    intersection = (prediction * target).sum()
    return (2.0 * intersection + smooth) / (prediction.sum() + target.sum() + smooth)
```

### Documentation

- Use ASCII diagrams for architecture explanations
- Include "why" not just "what"
- Link to relevant docs/specs

---

## Testing

### Running Tests

```bash
# All tests
uv run pytest

# Specific test file
uv run pytest tests/test_models/test_meshnet.py

# Specific test
uv run pytest tests/test_models/test_meshnet.py::test_parameter_count -v

# With coverage
uv run pytest --cov=src/arc_meshchop --cov-report=html
```

### Writing Tests

Tests use TDD approach with synthetic data:

```python
import pytest
import torch

from arc_meshchop.models.meshnet import MeshNet


class TestMeshNet:
    """Tests for MeshNet architecture."""

    def test_parameter_count(self) -> None:
        """Verify parameter count matches paper (147,474 for MeshNet-26)."""
        model = MeshNet(in_channels=1, out_channels=2, channels=26)
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count == 147474

    def test_output_shape(self) -> None:
        """Verify output shape matches input shape."""
        model = MeshNet(in_channels=1, out_channels=2, channels=26)
        x = torch.randn(1, 1, 64, 64, 64)  # Small for fast test
        y = model(x)
        assert y.shape == (1, 2, 64, 64, 64)
```

### Test Markers

```python
@pytest.mark.slow
def test_full_volume() -> None:
    """Test with full 256³ volume (slow)."""
    ...

@pytest.mark.gpu
def test_cuda_training() -> None:
    """Test training on GPU."""
    ...

@pytest.mark.integration
def test_full_pipeline() -> None:
    """End-to-end test."""
    ...
```

Run without slow tests:
```bash
uv run pytest -m "not slow"
```

---

## Project Structure

```
arc-meshchop/
├── src/arc_meshchop/
│   ├── data/               # Data loading
│   │   ├── huggingface_loader.py  # HuggingFace → ARCDatasetInfo
│   │   ├── dataset.py             # PyTorch Dataset
│   │   ├── preprocessing.py       # Resampling, normalization
│   │   └── splits.py              # Cross-validation
│   ├── models/
│   │   └── meshnet.py             # MeshNet architecture
│   ├── training/
│   │   ├── trainer.py             # Training loop
│   │   ├── loss.py                # Loss functions
│   │   └── optimizer.py           # AdamW + OneCycleLR
│   ├── evaluation/
│   │   ├── metrics.py             # DICE, AVD, MCC
│   │   └── evaluator.py           # Batch evaluation
│   ├── export/
│   │   └── onnx_export.py         # ONNX export
│   └── cli.py                     # Typer CLI
├── tests/                         # Mirror src/ structure
├── configs/                       # Configuration presets (unused)
└── docs/
    ├── specs/                     # TDD specifications
    └── research/                  # Paper analysis
```

---

## Adding a New Feature

1. **Write the spec** in `docs/specs/` describing expected behavior
2. **Write tests** that verify the spec
3. **Implement** the feature
4. **Document** in relevant .md files
5. **PR** with tests passing

---

## Key Constraints

### Architecture (from paper)

- MeshNet uses **10-layer symmetric dilation**: `[1,2,4,8,16,16,8,4,2,1]`
- **NOT** the old BrainChop 8-layer pattern
- Parameter counts must match exactly (147,474 for MeshNet-26)

### Training (from paper)

- AdamW: lr=0.001, weight_decay=3e-5, eps=1e-4
- OneCycleLR: 1% warmup
- CrossEntropy: weights=[0.5, 1.0], label_smoothing=0.01
- Batch size: 1 (full 256³ volumes)

### Data (OpenNeuro ground truth)

- 223 samples (115 SPACE-2x + 108 SPACE)
- Exclude turbo-spin-echo (5 samples)
- Require lesion masks
- 256³ @ 1mm isotropic

---

## Questions?

Open an issue or check existing documentation:
- [DATA.md](DATA.md) - Data pipeline
- [ARCHITECTURE.md](ARCHITECTURE.md) - System design
- [TRAINING.md](TRAINING.md) - Training details
- [docs/specs/](docs/specs/) - TDD specifications
