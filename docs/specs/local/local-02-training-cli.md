# Local Spec 02: Training CLI

> **Entry Point for Training** — CLI commands to run the full pipeline
>
> **Goal:** Implement CLI commands that let users train MeshNet on real ARC data.

---

## Overview

The existing CLI has only `info` and `version` commands. This spec adds:

1. `arc-meshchop download` - Download ARC dataset from HuggingFace
2. `arc-meshchop train` - Train a single model configuration
3. `arc-meshchop evaluate` - Evaluate a trained model

---

## 1. Current State

```bash
# What exists now
arc-meshchop version   # ✅ Works
arc-meshchop info      # ✅ Works

# What's missing
arc-meshchop download  # ❌ Not implemented
arc-meshchop train     # ❌ Not implemented
arc-meshchop evaluate  # ❌ Not implemented
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
    --exclude-turbo-spin-echo \
    --require-lesion-mask

# Result: Downloads ~224 samples to data/arc/
```

### 2.2 Train Command

```bash
# Train single configuration
arc-meshchop train \
    --data-dir data/arc \
    --output-dir outputs/train_001 \
    --channels 26 \
    --outer-fold 0 \
    --inner-fold 0 \
    --epochs 50 \
    --learning-rate 0.001 \
    --weight-decay 3e-5 \
    --seed 42

# Result: Trains MeshNet-26, saves checkpoints and metrics
```

### 2.3 Evaluate Command

```bash
# Evaluate trained model
arc-meshchop evaluate \
    --checkpoint outputs/train_001/best.pt \
    --data-dir data/arc \
    --outer-fold 0 \
    --output results/eval_001.json

# Result: Computes DICE, AVD, MCC on test fold
```

---

## 3. Implementation

### 3.1 CLI Module Structure

**File:** `src/arc_meshchop/cli/__init__.py`

```python
"""CLI entry points for arc-meshchop."""

from arc_meshchop.cli.main import app

__all__ = ["app"]
```

### 3.2 Main CLI Application

**File:** `src/arc_meshchop/cli/main.py`

```python
"""Main CLI application for arc-meshchop.

Provides commands for downloading data, training models,
and evaluating results.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

from arc_meshchop import __version__

app = typer.Typer(
    name="arc-meshchop",
    help="MeshNet stroke lesion segmentation - paper replication",
    add_completion=False,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@app.command()
def version() -> None:
    """Show version information."""
    typer.echo(f"arc-meshchop version {__version__}")


@app.command()
def info() -> None:
    """Show system information and device availability."""
    from arc_meshchop.utils.device import get_device_info

    info = get_device_info()
    typer.echo("System Information:")
    typer.echo(f"  Device: {info['device']}")
    typer.echo(f"  Device Type: {info['device_type']}")

    if info.get("cuda_available"):
        typer.echo(f"  CUDA Version: {info.get('cuda_version', 'N/A')}")
        typer.echo(f"  GPU Memory: {info.get('cuda_memory_gb', 'N/A')} GB")

    if info.get("mps_available"):
        typer.echo("  MPS: Available (Apple Silicon)")


@app.command()
def download(
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory for dataset"),
    ] = Path("data/arc"),
    repo_id: Annotated[
        str,
        typer.Option("--repo", help="HuggingFace repository ID"),
    ] = "hugging-science/arc-aphasia-bids",
    include_space_2x: Annotated[
        bool,
        typer.Option("--include-space-2x/--no-space-2x", help="Include SPACE 2x accel"),
    ] = True,
    include_space_no_accel: Annotated[
        bool,
        typer.Option("--include-space-no-accel/--no-space-no-accel", help="Include SPACE no accel"),
    ] = True,
    exclude_turbo_spin_echo: Annotated[
        bool,
        typer.Option("--exclude-tse/--include-tse", help="Exclude turbo-spin echo"),
    ] = True,
    require_lesion_mask: Annotated[
        bool,
        typer.Option("--require-mask/--no-require-mask", help="Require lesion mask"),
    ] = True,
) -> None:
    """Download ARC dataset from HuggingFace Hub.

    Downloads the ARC dataset with specified filters and saves
    file paths for training. By default, downloads ~224 samples
    matching the paper's methodology.
    """
    from arc_meshchop.data import load_arc_from_huggingface

    typer.echo(f"Downloading ARC dataset to {output_dir}...")

    try:
        info = load_arc_from_huggingface(
            repo_id=repo_id,
            cache_dir=output_dir / "cache",
            include_space_2x=include_space_2x,
            include_space_no_accel=include_space_no_accel,
            exclude_turbo_spin_echo=exclude_turbo_spin_echo,
            require_lesion_mask=require_lesion_mask,
        )

        # Save dataset info for later use
        output_dir.mkdir(parents=True, exist_ok=True)
        info_path = output_dir / "dataset_info.json"

        dataset_info = {
            "num_samples": len(info),
            "image_paths": [str(p) for p in info.image_paths],
            "mask_paths": [str(p) for p in info.mask_paths],
            "lesion_volumes": info.lesion_volumes,
            "acquisition_types": info.acquisition_types,
            "subject_ids": info.subject_ids,
        }

        info_path.write_text(json.dumps(dataset_info, indent=2))

        typer.echo(f"Downloaded {len(info)} samples")
        typer.echo(f"Dataset info saved to {info_path}")
        typer.echo(f"Expected: ~224 samples (from paper)")

    except Exception as e:
        typer.echo(f"Error downloading dataset: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def train(
    # Data options
    data_dir: Annotated[
        Path,
        typer.Option("--data-dir", "-d", help="Directory with dataset_info.json"),
    ] = Path("data/arc"),
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory for checkpoints"),
    ] = Path("outputs/train"),
    # Model options
    channels: Annotated[
        int,
        typer.Option("--channels", "-c", help="MeshNet channels (5, 16, or 26)"),
    ] = 26,
    # Fold options
    outer_fold: Annotated[
        int,
        typer.Option("--outer-fold", help="Outer CV fold (0-2)"),
    ] = 0,
    inner_fold: Annotated[
        int,
        typer.Option("--inner-fold", help="Inner CV fold (0-2)"),
    ] = 0,
    # Training options (FROM PAPER)
    epochs: Annotated[
        int,
        typer.Option("--epochs", "-e", help="Number of epochs"),
    ] = 50,
    learning_rate: Annotated[
        float,
        typer.Option("--lr", help="Max learning rate (OneCycleLR max_lr)"),
    ] = 0.001,
    weight_decay: Annotated[
        float,
        typer.Option("--weight-decay", "--wd", help="Weight decay"),
    ] = 3e-5,
    background_weight: Annotated[
        float,
        typer.Option("--bg-weight", help="Background class weight"),
    ] = 0.5,
    warmup_pct: Annotated[
        float,
        typer.Option("--warmup", help="Warmup percentage for OneCycleLR"),
    ] = 0.01,
    div_factor: Annotated[
        float,
        typer.Option("--div-factor", help="OneCycleLR div_factor (initial_lr = max_lr/div_factor)"),
    ] = 100.0,  # FROM PAPER: "starts at 1/100th of the max learning rate"
    use_fp16: Annotated[
        bool,
        typer.Option("--fp16/--no-fp16", help="Use FP16 mixed precision"),
    ] = True,
    # Reproducibility
    seed: Annotated[
        int,
        typer.Option("--seed", help="Random seed"),
    ] = 42,
    # Other
    num_workers: Annotated[
        int,
        typer.Option("--workers", help="DataLoader workers"),
    ] = 4,
    resume: Annotated[
        Optional[Path],
        typer.Option("--resume", help="Resume from checkpoint"),
    ] = None,
) -> None:
    """Train MeshNet on ARC dataset.

    Trains a single model configuration on specified CV folds.
    Use arc-meshchop experiment to run full nested CV.

    FROM PAPER:
    - AdamW optimizer (lr=0.001, weight_decay=3e-5, eps=1e-4)
    - OneCycleLR scheduler (1% warmup, starts at 1/100th of max_lr)
    - CrossEntropyLoss (weights=[0.5, 1.0], label_smoothing=0.01)
    - 50 epochs, batch size 1
    """
    import torch

    from arc_meshchop.data import (
        ARCDataset,
        create_dataloaders,
        create_stratification_labels,
        generate_nested_cv_splits,
        get_lesion_quintile,
    )
    from arc_meshchop.models import MeshNet
    from arc_meshchop.training import Trainer, TrainingConfig
    from arc_meshchop.utils.device import get_device

    # Set seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load dataset info
    info_path = data_dir / "dataset_info.json"
    if not info_path.exists():
        typer.echo(f"Dataset info not found: {info_path}", err=True)
        typer.echo("Run 'arc-meshchop download' first.", err=True)
        raise typer.Exit(1)

    with open(info_path) as f:
        dataset_info = json.load(f)

    image_paths = [Path(p) for p in dataset_info["image_paths"]]
    mask_paths = [Path(p) for p in dataset_info["mask_paths"]]
    lesion_volumes = dataset_info["lesion_volumes"]
    acquisition_types = dataset_info["acquisition_types"]

    typer.echo(f"Loaded {len(image_paths)} samples from {info_path}")

    # Generate CV splits
    quintiles = [get_lesion_quintile(v) for v in lesion_volumes]
    strat_labels = create_stratification_labels(quintiles, acquisition_types)

    splits = generate_nested_cv_splits(
        n_samples=len(image_paths),
        stratification_labels=strat_labels,
        num_outer_folds=3,
        num_inner_folds=3,
        random_seed=42,  # Fixed seed for reproducible splits
    )

    # Get fold indices
    split = splits.get_split(outer_fold, inner_fold)
    train_indices = split.train_indices
    val_indices = split.val_indices

    typer.echo(f"Fold {outer_fold}.{inner_fold}: {len(train_indices)} train, {len(val_indices)} val")

    # Create datasets
    train_dataset = ARCDataset(
        image_paths=[image_paths[i] for i in train_indices],
        mask_paths=[mask_paths[i] for i in train_indices],
        cache_dir=data_dir / "cache" / f"fold_{outer_fold}_{inner_fold}" / "train",
    )

    val_dataset = ARCDataset(
        image_paths=[image_paths[i] for i in val_indices],
        mask_paths=[mask_paths[i] for i in val_indices],
        cache_dir=data_dir / "cache" / f"fold_{outer_fold}_{inner_fold}" / "val",
    )

    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=1,  # FROM PAPER: batch size 1
        num_workers=num_workers,
    )

    # Create model
    device = get_device()
    typer.echo(f"Using device: {device}")

    model = MeshNet(channels=channels)
    typer.echo(f"MeshNet-{channels}: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create training config
    # NOTE: div_factor=100 is critical for paper parity
    # FROM PAPER: "starts at 1/100th of the max learning rate"
    config = TrainingConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        max_lr=learning_rate,
        weight_decay=weight_decay,
        pct_start=warmup_pct,
        div_factor=div_factor,  # Paper requires 100 (start at max_lr/100)
        background_weight=background_weight,
        use_fp16=use_fp16,
        checkpoint_dir=output_dir / f"fold_{outer_fold}_{inner_fold}",
        random_seed=seed,
    )

    # Create trainer
    trainer = Trainer(model, config, device=device)

    # Resume if specified
    if resume and resume.exists():
        typer.echo(f"Resuming from checkpoint: {resume}")
        trainer.load_checkpoint(resume)

    # Train
    typer.echo(f"Training for {epochs} epochs...")
    results = trainer.train(train_loader, val_loader)

    # Save results
    results_path = output_dir / f"fold_{outer_fold}_{inner_fold}" / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    results_data = {
        "channels": channels,
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "epochs": epochs,
        "seed": seed,
        **results,
    }

    results_path.write_text(json.dumps(results_data, indent=2))

    typer.echo(f"Training complete!")
    typer.echo(f"Best DICE: {results['best_dice']:.4f} (epoch {results['best_epoch']})")
    typer.echo(f"Results saved to {results_path}")


@app.command()
def evaluate(
    checkpoint: Annotated[
        Path,
        typer.Argument(help="Path to model checkpoint"),
    ],
    data_dir: Annotated[
        Path,
        typer.Option("--data-dir", "-d", help="Directory with dataset_info.json"),
    ] = Path("data/arc"),
    outer_fold: Annotated[
        int,
        typer.Option("--outer-fold", help="Outer CV fold for test set"),
    ] = 0,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output JSON file for results"),
    ] = None,
    channels: Annotated[
        int,
        typer.Option("--channels", "-c", help="MeshNet channels"),
    ] = 26,
) -> None:
    """Evaluate trained model on test set.

    Loads a checkpoint and evaluates on the held-out test fold.
    Reports DICE, AVD, and MCC metrics.
    """
    import torch

    from arc_meshchop.data import (
        ARCDataset,
        create_stratification_labels,
        generate_nested_cv_splits,
        get_lesion_quintile,
    )
    from arc_meshchop.evaluation import SegmentationMetrics
    from arc_meshchop.models import MeshNet
    from arc_meshchop.utils.device import get_device

    # Load dataset info
    info_path = data_dir / "dataset_info.json"
    if not info_path.exists():
        typer.echo(f"Dataset info not found: {info_path}", err=True)
        raise typer.Exit(1)

    with open(info_path) as f:
        dataset_info = json.load(f)

    image_paths = [Path(p) for p in dataset_info["image_paths"]]
    mask_paths = [Path(p) for p in dataset_info["mask_paths"]]
    lesion_volumes = dataset_info["lesion_volumes"]
    acquisition_types = dataset_info["acquisition_types"]

    # Generate splits to get test indices
    quintiles = [get_lesion_quintile(v) for v in lesion_volumes]
    strat_labels = create_stratification_labels(quintiles, acquisition_types)

    splits = generate_nested_cv_splits(
        n_samples=len(image_paths),
        stratification_labels=strat_labels,
        num_outer_folds=3,
        num_inner_folds=3,
        random_seed=42,
    )

    # Get test indices (outer fold test set)
    split = splits.get_split(outer_fold, inner_fold=None)
    test_indices = split.test_indices

    typer.echo(f"Evaluating on {len(test_indices)} test samples (outer fold {outer_fold})")

    # Create test dataset
    test_dataset = ARCDataset(
        image_paths=[image_paths[i] for i in test_indices],
        mask_paths=[mask_paths[i] for i in test_indices],
        preprocess=True,
    )

    # Load model
    device = get_device()
    model = MeshNet(channels=channels)
    model = model.to(device)

    checkpoint_data = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()

    typer.echo(f"Loaded checkpoint from {checkpoint}")

    # Evaluate
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            image, mask = test_dataset[idx]
            image = image.unsqueeze(0).to(device)

            output = model(image)
            pred = output.argmax(dim=1).squeeze(0).cpu()

            all_preds.append(pred)
            all_targets.append(mask)

            if (idx + 1) % 10 == 0:
                typer.echo(f"Processed {idx + 1}/{len(test_dataset)} samples")

    # Stack and compute metrics
    preds = torch.stack(all_preds)
    targets = torch.stack(all_targets)

    # Use SegmentationMetrics class (not the non-existent calculate_metrics function)
    metrics_calculator = SegmentationMetrics()
    metrics = metrics_calculator.compute_batch(preds, targets)

    typer.echo(f"\nResults:")
    typer.echo(f"  DICE: {metrics['dice']:.4f}")
    typer.echo(f"  AVD:  {metrics['avd']:.4f}")
    typer.echo(f"  MCC:  {metrics['mcc']:.4f}")

    # Save results if output specified
    if output:
        results = {
            "checkpoint": str(checkpoint),
            "outer_fold": outer_fold,
            "num_samples": len(test_indices),
            **metrics,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2))
        typer.echo(f"\nResults saved to {output}")


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
```

---

## 4. Tests

**File:** `tests/test_cli/test_commands.py`

```python
"""Tests for CLI commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from arc_meshchop.cli.main import app

runner = CliRunner()


class TestVersionCommand:
    """Tests for version command."""

    def test_shows_version(self) -> None:
        """Verify version is displayed."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "arc-meshchop version" in result.output


class TestInfoCommand:
    """Tests for info command."""

    def test_shows_device_info(self) -> None:
        """Verify device info is displayed."""
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "Device:" in result.output


class TestDownloadCommand:
    """Tests for download command."""

    def test_requires_network(self) -> None:
        """Download requires network access."""
        # This is an integration test - skip if no network
        pass

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        """Verify output directory is created."""
        with patch("arc_meshchop.cli.main.load_arc_from_huggingface") as mock:
            mock_info = MagicMock()
            mock_info.__len__ = MagicMock(return_value=10)
            mock_info.image_paths = [Path(f"/path/img_{i}.nii.gz") for i in range(10)]
            mock_info.mask_paths = [Path(f"/path/mask_{i}.nii.gz") for i in range(10)]
            mock_info.lesion_volumes = [1000] * 10
            mock_info.acquisition_types = ["space_2x"] * 10
            mock_info.subject_ids = [f"sub-{i}" for i in range(10)]
            mock.return_value = mock_info

            result = runner.invoke(app, [
                "download",
                "--output", str(tmp_path / "data"),
            ])

            assert result.exit_code == 0
            assert (tmp_path / "data" / "dataset_info.json").exists()


class TestTrainCommand:
    """Tests for train command."""

    def test_requires_dataset_info(self, tmp_path: Path) -> None:
        """Verify error when dataset_info.json missing."""
        result = runner.invoke(app, [
            "train",
            "--data-dir", str(tmp_path),
            "--output", str(tmp_path / "output"),
        ])

        assert result.exit_code == 1
        assert "Dataset info not found" in result.output


class TestEvaluateCommand:
    """Tests for evaluate command."""

    def test_requires_checkpoint(self, tmp_path: Path) -> None:
        """Verify checkpoint path is required."""
        result = runner.invoke(app, ["evaluate"])
        assert result.exit_code != 0
```

---

## 5. Implementation Checklist

- [ ] Update `src/arc_meshchop/cli/main.py` with new commands
- [ ] Implement `download` command
- [ ] Implement `train` command
- [ ] Implement `evaluate` command
- [ ] Add progress bars for long operations
- [ ] Add proper error handling and messages
- [ ] Create CLI tests
- [ ] Update `pyproject.toml` entry point if needed
- [ ] Test end-to-end workflow

---

## 6. Verification Commands

```bash
# Test CLI commands
uv run arc-meshchop --help
uv run arc-meshchop download --help
uv run arc-meshchop train --help
uv run arc-meshchop evaluate --help

# Run CLI tests
uv run pytest tests/test_cli/ -v

# Full workflow test
uv run arc-meshchop download --output data/arc
uv run arc-meshchop train \
    --data-dir data/arc \
    --output outputs/test_run \
    --channels 5 \
    --epochs 2 \
    --outer-fold 0 \
    --inner-fold 0

uv run arc-meshchop evaluate \
    outputs/test_run/fold_0_0/best.pt \
    --data-dir data/arc \
    --outer-fold 0 \
    --channels 5
```

---

## 7. Usage Examples

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
