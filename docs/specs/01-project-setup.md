# Spec 01: Project Setup

> **Phase 1 of 7** — Foundation infrastructure for MeshNet paper replication
>
> **Goal:** Establish a production-ready Python project with modern tooling, CI/CD, and TDD infrastructure.

---

## Overview

This spec covers the complete project setup including:
- UV package manager with `pyproject.toml`
- Development dependencies (pytest, mypy, ruff)
- Makefile for common commands
- GitHub Actions CI/CD pipeline
- Pre-commit hooks
- Directory structure

---

## 1. Directory Structure

```
arc-meshchop/
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions CI/CD
├── src/
│   └── arc_meshchop/
│       ├── __init__.py
│       ├── py.typed              # PEP 561 marker for type stubs
│       ├── models/               # MeshNet implementation (Spec 02)
│       │   └── __init__.py
│       ├── data/                 # Data pipeline (Spec 03)
│       │   └── __init__.py
│       ├── training/             # Training infrastructure (Spec 04)
│       │   └── __init__.py
│       ├── evaluation/           # Metrics & evaluation (Spec 05)
│       │   └── __init__.py
│       └── export/               # Model export (Spec 06)
│           └── __init__.py
├── tests/
│   ├── conftest.py               # Shared pytest fixtures
│   ├── test_models/
│   ├── test_data/
│   ├── test_training/
│   ├── test_evaluation/
│   └── test_export/
├── configs/                      # Hydra configuration files
│   ├── config.yaml
│   ├── model/
│   ├── training/
│   └── experiment/
├── scripts/                      # Standalone scripts
│   └── preprocess.py
├── docs/
│   ├── research/                 # Existing research docs
│   └── specs/                    # These spec documents
├── _references/                  # Local reference copies (READ-ONLY, not imported)
├── _literature/                  # Existing paper markdown
├── pyproject.toml
├── Makefile
├── .pre-commit-config.yaml
├── .gitignore
├── .python-version               # Python version pinning
├── CLAUDE.md                     # Claude Code guidance
└── README.md
```

---

## 2. pyproject.toml

```toml
[project]
name = "arc-meshchop"
version = "0.1.0"
description = "MeshNet stroke lesion segmentation - paper replication"
readme = "README.md"
license = "Apache-2.0"
requires-python = ">=3.10"
authors = [
    { name = "arc-meshchop contributors" },
]
keywords = [
    "meshnet",
    "stroke",
    "lesion",
    "segmentation",
    "mri",
    "neuroimaging",
    "deep-learning",
    "pytorch",
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Medical Science Apps.",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

dependencies = [
    # Core ML
    "torch>=2.0.0",
    "torchvision>=0.15.0",

    # Medical imaging
    "nibabel>=5.0.0",
    "monai>=1.3.0",

    # ARC dataset pipeline (pinned Git dependency)
    "neuroimaging-go-brrrr @ git+https://github.com/The-Obstacle-Is-The-Way/neuroimaging-go-brrrr.git@v0.2.0",

    # Data handling
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scipy>=1.10.0",

    # Configuration
    "hydra-core>=1.3.0",
    "omegaconf>=2.3.0",

    # Hyperparameter optimization
    "orion>=0.2.7",

    # Logging & tracking
    "tensorboard>=2.14.0",
    "tqdm>=4.65.0",

    # CLI
    "typer>=0.12.0",
    "rich>=13.0.0",
]

[project.optional-dependencies]
dev = [
    # Testing
    "pytest>=8.0.0",
    "pytest-cov>=4.0.0",
    "pytest-xdist>=3.5.0",       # Parallel test execution
    "pytest-sugar>=1.0.0",       # Pretty test output
    "pytest-mock>=3.12.0",       # Mocking utilities

    # Type checking
    "mypy>=1.11.0",
    "types-tqdm>=4.66.0",
    "pandas-stubs>=2.0.0",

    # Linting & formatting
    "ruff>=0.8.0",

    # Pre-commit
    "pre-commit>=3.0.0",
]

export = [
    # Model export (Spec 06)
    "onnx>=1.14.0",
    "onnxruntime>=1.16.0",
    "tf2onnx>=1.16.0",
]

huggingface = [
    # HuggingFace integration (Spec 07)
    "datasets>=3.4.0",
    "huggingface-hub>=0.32.0",
]

all = [
    "arc-meshchop[dev,export,huggingface]",
]

[project.scripts]
arc-meshchop = "arc_meshchop.cli:app"

[project.urls]
Homepage = "https://github.com/username/arc-meshchop"
Repository = "https://github.com/username/arc-meshchop"
Issues = "https://github.com/username/arc-meshchop/issues"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/arc_meshchop"]

# =============================================================================
# RUFF - Linting & Formatting
# =============================================================================
[tool.ruff]
line-length = 100
target-version = "py310"
extend-exclude = ["*.ipynb", "_references/*", "_literature/*"]

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "F",      # pyflakes
    "I",      # isort
    "B",      # flake8-bugbear
    "UP",     # pyupgrade
    "SIM",    # flake8-simplify
    "PTH",    # flake8-use-pathlib
    "RUF",    # ruff-specific
    "D",      # pydocstyle (docstrings)
    "ANN",    # flake8-annotations
]
ignore = [
    "D100",   # Missing docstring in public module
    "D104",   # Missing docstring in public package
    "D107",   # Missing docstring in __init__
    "ANN101", # Missing type annotation for self
    "ANN102", # Missing type annotation for cls
]

[tool.ruff.lint.isort]
known-first-party = ["arc_meshchop"]

[tool.ruff.lint.pydocstyle]
convention = "google"

# =============================================================================
# PYTEST - Testing
# =============================================================================
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
python_classes = ["Test*"]
addopts = [
    "-v",
    "--tb=short",
    "--strict-markers",
    "--cov=src/arc_meshchop",
    "--cov-report=term-missing",
    "--cov-report=xml:coverage.xml",
    "--cov-fail-under=80",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "gpu: marks tests that require GPU",
    "integration: marks integration tests",
]
filterwarnings = [
    "ignore::DeprecationWarning",
]

# =============================================================================
# MYPY - Type Checking
# =============================================================================
[tool.mypy]
python_version = "3.10"
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
strict_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_return_any = true
no_implicit_reexport = true
show_error_codes = true
exclude = [
    "_references/",
    "_literature/",
]

[[tool.mypy.overrides]]
module = [
    "nibabel.*",
    "monai.*",
    "orion.*",
    "hydra.*",
    "omegaconf.*",
    "tensorboard.*",
]
ignore_missing_imports = true

# =============================================================================
# COVERAGE - Test Coverage
# =============================================================================
[tool.coverage.run]
source = ["src/arc_meshchop"]
branch = true
omit = [
    "*/__init__.py",
    "*/cli.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
    "if __name__ == .__main__.:",
]
```

---

## 3. Makefile

```makefile
.PHONY: help install install-dev sync lint format typecheck test test-fast test-cov clean pre-commit

# Default Python version
PYTHON_VERSION := 3.10

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# Installation
# =============================================================================
install:  ## Install production dependencies
	uv sync

install-dev:  ## Install all dependencies including dev
	uv sync --all-extras

sync:  ## Sync dependencies with lockfile
	uv sync --all-extras

# =============================================================================
# Code Quality
# =============================================================================
lint:  ## Run ruff linter
	uv run ruff check src tests

lint-fix:  ## Run ruff linter with auto-fix
	uv run ruff check src tests --fix

format:  ## Run ruff formatter
	uv run ruff format src tests

format-check:  ## Check formatting without changes
	uv run ruff format src tests --check

typecheck:  ## Run mypy type checker
	uv run mypy src tests

quality:  ## Run all code quality checks (lint + format-check + typecheck)
	$(MAKE) lint
	$(MAKE) format-check
	$(MAKE) typecheck

# =============================================================================
# Testing
# =============================================================================
test:  ## Run all tests
	uv run pytest

test-fast:  ## Run tests excluding slow markers
	uv run pytest -m "not slow"

test-cov:  ## Run tests with coverage report
	uv run pytest --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

test-parallel:  ## Run tests in parallel
	uv run pytest -n auto

test-verbose:  ## Run tests with verbose output
	uv run pytest -vvs

# =============================================================================
# Pre-commit
# =============================================================================
pre-commit-install:  ## Install pre-commit hooks
	uv run pre-commit install

pre-commit:  ## Run pre-commit on all files
	uv run pre-commit run --all-files

# =============================================================================
# Development
# =============================================================================
clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf coverage.xml
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# =============================================================================
# CI/CD Simulation
# =============================================================================
ci:  ## Run full CI pipeline locally
	$(MAKE) quality
	$(MAKE) test

# =============================================================================
# Project Initialization
# =============================================================================
init:  ## Initialize project (first-time setup)
	uv sync --all-extras
	uv run pre-commit install
	@echo "Project initialized successfully!"
```

---

## 4. GitHub Actions CI/CD

### `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  # ===========================================================================
  # Lint and Format Check
  # ===========================================================================
  lint:
    name: Lint & Format
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: "latest"
          enable-cache: true
          cache-dependency-glob: "uv.lock"

      - name: Set up Python
        run: uv python install 3.10

      - name: Install dependencies
        run: uv sync --all-extras

      - name: Ruff lint
        run: uv run ruff check src tests --output-format=github

      - name: Ruff format check
        run: uv run ruff format src tests --check

  # ===========================================================================
  # Type Checking
  # ===========================================================================
  typecheck:
    name: Type Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: "latest"
          enable-cache: true
          cache-dependency-glob: "uv.lock"

      - name: Set up Python
        run: uv python install 3.10

      - name: Install dependencies
        run: uv sync --all-extras

      - name: Mypy
        run: uv run mypy src tests --cache-dir=.mypy_cache

  # ===========================================================================
  # Tests
  # ===========================================================================
  test:
    name: Test (Python ${{ matrix.python-version }})
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: "latest"
          enable-cache: true
          cache-dependency-glob: "uv.lock"

      - name: Set up Python ${{ matrix.python-version }}
        run: uv python install ${{ matrix.python-version }}

      - name: Install dependencies
        run: uv sync --all-extras

      - name: Run tests
        run: uv run pytest -m "not slow and not gpu" --cov-report=xml

      - name: Upload coverage to Codecov
        if: matrix.python-version == '3.10'
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
          fail_ci_if_error: false

  # ===========================================================================
  # GPU Tests (Optional - requires self-hosted runner)
  # ===========================================================================
  # test-gpu:
  #   name: GPU Tests
  #   runs-on: [self-hosted, gpu]
  #   if: github.event_name == 'push' && github.ref == 'refs/heads/main'
  #   steps:
  #     - uses: actions/checkout@v4
  #     - name: Run GPU tests
  #       run: |
  #         uv sync --all-extras
  #         uv run pytest -m "gpu" -v

  # ===========================================================================
  # All Checks Pass Gate
  # ===========================================================================
  all-checks:
    name: All Checks Pass
    needs: [lint, typecheck, test]
    runs-on: ubuntu-latest
    steps:
      - name: All checks passed
        run: echo "All CI checks passed!"
```

---

## 5. Pre-commit Configuration

### `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-toml
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-merge-conflict
      - id: debug-statements
      - id: detect-private-key

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.0
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  - repo: local
    hooks:
      - id: mypy
        name: mypy
        entry: uv run mypy
        language: system
        types: [python]
        pass_filenames: false
        args: [src, tests]
```

---

## 6. Supporting Files

### `.python-version`

```
3.10
```

### `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.env
.venv
env/
venv/
ENV/

# UV
.uv/

# Testing
.pytest_cache/
.coverage
htmlcov/
coverage.xml
*.cover

# Type checking
.mypy_cache/

# Linting
.ruff_cache/

# IDE
.idea/
.vscode/
*.swp
*.swo
*~

# Jupyter
.ipynb_checkpoints/

# Data (large files)
data/raw/
data/processed/
*.nii
*.nii.gz
*.mgz

# Model checkpoints
checkpoints/
*.pt
*.pth
*.ckpt
*.onnx

# Logs
logs/
*.log
tensorboard/

# OS
.DS_Store
Thumbs.db

# Secrets
.env.local
*.pem
*.key
```

### `CLAUDE.md`

```markdown
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

- **Optimizer:** AdamW (lr=0.001, weight_decay=3e-5)
- **Scheduler:** OneCycleLR (1% warmup)
- **Loss:** CrossEntropy(weights=[0.5, 1.0], label_smoothing=0.01)
- **Precision:** FP16 (half-precision)
- **Batch size:** 1 (full 256³ volumes)

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
| `configs/` | Hydra configuration |

## Reference Implementations

> **Note:** `_references/` contains local read-only copies for study. Do NOT import from there.

- `_references/brainchop/` - VERIFIED architecture (JS-based, read-only reference)
- `_references/niivue/` - WebGL visualization (JS-based, read-only reference)

### Pinned Python Dependencies (installed via Git)

- **neuroimaging-go-brrrr** @ `git+https://github.com/The-Obstacle-Is-The-Way/neuroimaging-go-brrrr.git@v0.2.0`
  - Provides: ARC dataset access via HuggingFace Hub
  - Import as: `from bids_hub.datasets import arc`

## Documentation

- `docs/research/` - Paper extraction and verified facts
- `docs/specs/` - TDD specification documents
```

---

## 7. Initial Source Files

### `src/arc_meshchop/__init__.py`

```python
"""ARC MeshChop: MeshNet stroke lesion segmentation."""

__version__ = "0.1.0"
```

### `src/arc_meshchop/py.typed`

```
# PEP 561 marker file - this package supports type checking
```

### `src/arc_meshchop/cli.py`

```python
"""Command-line interface for arc-meshchop."""

import typer
from rich.console import Console

app = typer.Typer(
    name="arc-meshchop",
    help="MeshNet stroke lesion segmentation - paper replication",
    add_completion=False,
)
console = Console()


@app.command()
def version() -> None:
    """Show version information."""
    from arc_meshchop import __version__

    console.print(f"arc-meshchop v{__version__}")


@app.command()
def info() -> None:
    """Show project information."""
    console.print("[bold]ARC MeshChop[/bold]")
    console.print("MeshNet stroke lesion segmentation - paper replication")
    console.print()
    console.print("[bold]Paper:[/bold]")
    console.print("  State-of-the-Art Stroke Lesion Segmentation at 1/1000th of Parameters")
    console.print("  Fedorov et al. (Emory, Georgia State, USC)")
    console.print()
    console.print("[bold]Model Variants:[/bold]")
    console.print("  MeshNet-5:  5,682 params  (0.848 DICE)")
    console.print("  MeshNet-16: 56,194 params (0.873 DICE)")
    console.print("  MeshNet-26: 147,474 params (0.876 DICE)")


if __name__ == "__main__":
    app()
```

### `tests/conftest.py`

```python
"""Shared pytest fixtures for arc-meshchop tests."""

import numpy as np
import pytest
import torch


@pytest.fixture
def device() -> torch.device:
    """Get available device (GPU if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def random_seed() -> int:
    """Fixed random seed for reproducibility."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture
def sample_volume() -> torch.Tensor:
    """Create a sample 256³ volume tensor."""
    return torch.randn(1, 1, 256, 256, 256)


@pytest.fixture
def small_volume() -> torch.Tensor:
    """Create a small 32³ volume for fast tests."""
    return torch.randn(1, 1, 32, 32, 32)


@pytest.fixture
def tiny_volume() -> torch.Tensor:
    """Create a tiny 8³ volume for unit tests."""
    return torch.randn(1, 1, 8, 8, 8)
```

---

## 8. Test-Driven Development: Initial Tests

### `tests/test_smoke.py`

```python
"""Smoke tests to verify basic project setup."""

import pytest


def test_import_package() -> None:
    """Test that the main package can be imported."""
    import arc_meshchop

    assert arc_meshchop.__version__ == "0.1.0"


def test_import_submodules() -> None:
    """Test that all submodules can be imported."""
    import arc_meshchop.models
    import arc_meshchop.data
    import arc_meshchop.training
    import arc_meshchop.evaluation
    import arc_meshchop.export


def test_cli_import() -> None:
    """Test that CLI can be imported."""
    from arc_meshchop.cli import app

    assert app is not None


def test_torch_available() -> None:
    """Test that PyTorch is available."""
    import torch

    assert torch.__version__ is not None


def test_cuda_detection() -> None:
    """Test CUDA detection (should not fail even without GPU)."""
    import torch

    # Just verify the check doesn't error
    _ = torch.cuda.is_available()


@pytest.mark.slow
def test_monai_available() -> None:
    """Test that MONAI is available."""
    import monai

    assert monai.__version__ is not None
```

---

## 9. Implementation Checklist

### Phase 1.1: Initialize Project Structure

- [ ] Create directory structure
- [ ] Create `pyproject.toml`
- [ ] Create `Makefile`
- [ ] Create `.gitignore`
- [ ] Create `.python-version`
- [ ] Create `CLAUDE.md`

### Phase 1.2: Install Dependencies

- [ ] Run `uv sync --all-extras`
- [ ] Verify `uv.lock` is created
- [ ] Verify all packages install correctly

### Phase 1.3: Configure CI/CD

- [ ] Create `.github/workflows/ci.yml`
- [ ] Create `.pre-commit-config.yaml`
- [ ] Run `make pre-commit-install`
- [ ] Verify pre-commit hooks work

### Phase 1.4: Create Initial Source Files

- [ ] Create `src/arc_meshchop/__init__.py`
- [ ] Create `src/arc_meshchop/py.typed`
- [ ] Create `src/arc_meshchop/cli.py`
- [ ] Create all subpackage `__init__.py` files

### Phase 1.5: Create Initial Tests

- [ ] Create `tests/conftest.py`
- [ ] Create `tests/test_smoke.py`
- [ ] Run `make test` - all tests pass
- [ ] Run `make quality` - all checks pass

### Phase 1.6: Verify Complete Setup

- [ ] Run `make ci` - full CI passes locally
- [ ] Push to GitHub - CI workflow passes
- [ ] Coverage report generates correctly

---

## 10. Verification Commands

After completing this spec, these commands should all succeed:

```bash
# Installation
uv sync --all-extras

# Code quality
make lint          # No errors
make format-check  # No changes needed
make typecheck     # No type errors

# Tests
make test          # All tests pass
make test-fast     # Fast tests pass

# Full CI
make ci            # Everything passes

# CLI
uv run arc-meshchop version   # Shows version
uv run arc-meshchop info      # Shows project info
```

---

## Dependencies Reference

### Core Production Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >=2.0.0 | Deep learning framework |
| torchvision | >=0.15.0 | Vision utilities |
| nibabel | >=5.0.0 | NIfTI file I/O |
| monai | >=1.3.0 | Medical imaging utilities |
| numpy | >=1.24.0 | Numerical computing |
| pandas | >=2.0.0 | Data manipulation |
| scipy | >=1.10.0 | Scientific computing |
| hydra-core | >=1.3.0 | Configuration management |
| orion | >=0.2.7 | Hyperparameter optimization |
| tensorboard | >=2.14.0 | Training visualization |
| typer | >=0.12.0 | CLI framework |

### Development Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pytest | >=8.0.0 | Testing framework |
| pytest-cov | >=4.0.0 | Coverage reporting |
| pytest-xdist | >=3.5.0 | Parallel testing |
| pytest-sugar | >=1.0.0 | Pretty output |
| mypy | >=1.11.0 | Type checking |
| ruff | >=0.8.0 | Linting & formatting |
| pre-commit | >=3.0.0 | Git hooks |

---

## Sources

- [UV Package Manager - Real Python](https://realpython.com/python-uv/)
- [pyproject.toml Guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
- [GitHub Actions for Python 2025](https://ber2.github.io/posts/2025_github_actions_python/)
- [Ruff Configuration](https://docs.astral.sh/ruff/configuration/)
- [MONAI Medical Imaging](https://learnopencv.com/monai-medical-imaging-pytorch/)
