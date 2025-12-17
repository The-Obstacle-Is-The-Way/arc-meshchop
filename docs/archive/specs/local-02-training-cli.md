# Local Spec 02: Training CLI

> **Entry Point for Training** — CLI commands to run the full pipeline
>
> **Status:** ✅ Implemented in `src/arc_meshchop/cli.py`

---

## Overview

The CLI provides commands for the full MeshNet training pipeline:

1. `arc-meshchop download` - Download ARC dataset from HuggingFace
2. `arc-meshchop train` - Train a single model configuration
3. `arc-meshchop evaluate` - Evaluate a trained model
4. `arc-meshchop experiment` - Run full nested CV experiment
5. `arc-meshchop validate` - Validate results against paper

---

## 1. Implementation Status

```bash
# All commands implemented in src/arc_meshchop/cli.py
arc-meshchop version   # ✅ Implemented
arc-meshchop info      # ✅ Implemented
arc-meshchop download  # ✅ Implemented
arc-meshchop train     # ✅ Implemented
arc-meshchop evaluate  # ✅ Implemented
arc-meshchop experiment # ✅ Implemented
arc-meshchop validate  # ✅ Implemented
arc-meshchop export    # ✅ Implemented
```

---

## 2. CLI Design

### 2.1 Download Command

```bash
# Download ARC dataset from HuggingFace
arc-meshchop download \
    --output data/arc \
    --include-space-2x \
    --include-space-no-accel \
    --exclude-tse \
    --require-mask

# Result: Downloads ~224 samples to data/arc/
```

### 2.2 Train Command

```bash
# Train single configuration
arc-meshchop train \
    --data-dir data/arc \
    --output outputs/train_001 \
    --channels 26 \
    --outer-fold 0 \
    --inner-fold 0 \
    --epochs 50 \
    --lr 0.001 \
    --weight-decay 3e-5 \
    --div-factor 100 \
    --seed 42

# Result: Trains MeshNet-26, saves checkpoints and metrics
```

**Key flags (FROM PAPER):**
- `--lr` (not `--learning-rate`): Max learning rate for OneCycleLR
- `--div-factor 100`: Initial LR = max_lr / 100 (paper requirement)
- `--bg-weight 0.5`: Background class weight (paper default)
- `--warmup 0.01`: 1% warmup (paper default)

### 2.3 Evaluate Command

```bash
# Evaluate trained model (checkpoint is positional argument)
arc-meshchop evaluate outputs/train_001/best.pt \
    --data-dir data/arc \
    --outer-fold 0 \
    --channels 26 \
    --output results/eval_001.json

# Result: Computes DICE, AVD, MCC on test fold
```

**Note:** Checkpoint path is a **positional argument**, not `--checkpoint`.

---

## 3. Implementation

### 3.1 CLI Module Structure

**File:** `src/arc_meshchop/cli.py` (single file, not a package)

The CLI is implemented as a single Typer application with all commands in one file.

```python
"""Command-line interface for arc-meshchop."""

import typer

app = typer.Typer(
    name="arc-meshchop",
    help="MeshNet stroke lesion segmentation - paper replication",
)

@app.command()
def download(...): ...

@app.command()
def train(...): ...

@app.command()
def evaluate(...): ...

@app.command()
def experiment(...): ...

@app.command()
def validate(...): ...
```

### 3.2 Entry Point

**File:** `pyproject.toml`

```toml
[project.scripts]
arc-meshchop = "arc_meshchop.cli:app"
```

---

## 4. Command Reference

### download

| Flag | Default | Description |
|------|---------|-------------|
| `--output, -o` | `data/arc` | Output directory |
| `--repo` | `hugging-science/arc-aphasia-bids` | HuggingFace repo |
| `--include-space-2x/--no-space-2x` | True | Include SPACE 2x |
| `--include-space-no-accel/--no-space-no-accel` | True | Include SPACE no accel |
| `--exclude-tse/--include-tse` | True | Exclude turbo-spin-echo |
| `--require-mask/--no-require-mask` | True | Require lesion mask |
| `--verify-counts/--no-verify-counts` | True | Verify paper counts (224) |

### train

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir, -d` | `data/arc` | Dataset directory |
| `--output, -o` | `outputs/train` | Output directory |
| `--channels, -c` | 26 | MeshNet channels |
| `--outer-fold` | 0 | Outer CV fold (0-2) |
| `--inner-fold` | 0 | Inner CV fold (0-2) |
| `--epochs, -e` | 50 | Number of epochs |
| `--lr` | 0.001 | Max learning rate |
| `--weight-decay, --wd` | 3e-5 | Weight decay |
| `--bg-weight` | 0.5 | Background class weight |
| `--warmup` | 0.01 | Warmup percentage |
| `--div-factor` | 100.0 | OneCycleLR div_factor |
| `--fp16/--no-fp16` | True | Use FP16 (CUDA only) |
| `--seed` | 42 | Random seed |
| `--workers` | 4 | DataLoader workers |
| `--resume` | None | Resume from checkpoint |

### evaluate

| Argument/Flag | Default | Description |
|---------------|---------|-------------|
| `checkpoint` (positional) | Required | Path to model checkpoint |
| `--data-dir, -d` | `data/arc` | Dataset directory |
| `--outer-fold` | 0 | Outer fold for test set |
| `--channels, -c` | 26 | MeshNet channels |
| `--output, -o` | None | Output JSON file |

---

## 5. Usage Examples

### Quick Test (MeshNet-5, 2 epochs)

```bash
# Download data
arc-meshchop download --output data/arc

# Quick training test
arc-meshchop train \
    --data-dir data/arc \
    --channels 5 \
    --epochs 2 \
    --outer-fold 0 \
    --inner-fold 0

# Evaluate
arc-meshchop evaluate outputs/train/fold_0_0/best.pt --channels 5
```

### Full Paper Replication (MeshNet-26, 50 epochs)

```bash
# Download data
arc-meshchop download --output data/arc

# Train MeshNet-26 on fold 0.0
arc-meshchop train \
    --data-dir data/arc \
    --output outputs/meshnet26 \
    --channels 26 \
    --epochs 50 \
    --lr 0.001 \
    --weight-decay 3e-5 \
    --bg-weight 0.5 \
    --warmup 0.01 \
    --div-factor 100 \
    --outer-fold 0 \
    --inner-fold 0 \
    --seed 42

# Evaluate on test set
arc-meshchop evaluate \
    outputs/meshnet26/fold_0_0/best.pt \
    --data-dir data/arc \
    --outer-fold 0 \
    --channels 26 \
    --output results/meshnet26_fold0.json
```

---

## 6. Verification Commands

```bash
# Test CLI commands
uv run arc-meshchop --help
uv run arc-meshchop download --help
uv run arc-meshchop train --help
uv run arc-meshchop evaluate --help

# Run smoke tests
uv run pytest tests/test_smoke.py -v
```

---

## 7. Implementation Checklist

- [x] Implement `download` command
- [x] Implement `train` command
- [x] Implement `evaluate` command
- [x] Implement `experiment` command
- [x] Implement `validate` command
- [x] Add Rich console output
- [x] Add proper error handling
- [x] Update `pyproject.toml` entry point
