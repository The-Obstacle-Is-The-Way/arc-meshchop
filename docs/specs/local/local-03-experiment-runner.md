# Local Spec 03: Experiment Runner

> **Full Paper Replication** — Run all 90 training configurations
>
> **Goal:** Orchestrate the complete nested CV experiment with 10 restarts.

---

## Overview

The paper's experimental setup:
- **3 outer folds** (train/test splits)
- **3 inner folds** per outer (train/val splits)
- **10 restarts** per configuration
- **Total:** 3 × 3 × 10 = **90 training runs**

This spec implements the orchestration layer to run all configurations,
aggregate results, and produce the final benchmark table.

---

## 1. Experiment Structure (FROM PAPER)

### 1.1 Nested Cross-Validation

```
Outer Fold 0 ─┬─ Inner Fold 0 ─── 10 restarts → Mean DICE
              ├─ Inner Fold 1 ─── 10 restarts → Mean DICE
              └─ Inner Fold 2 ─── 10 restarts → Mean DICE
              └─ TEST SET (held out)

Outer Fold 1 ─┬─ Inner Fold 0 ─── 10 restarts → Mean DICE
              ├─ Inner Fold 1 ─── 10 restarts → Mean DICE
              └─ Inner Fold 2 ─── 10 restarts → Mean DICE
              └─ TEST SET (held out)

Outer Fold 2 ─┬─ Inner Fold 0 ─── 10 restarts → Mean DICE
              ├─ Inner Fold 1 ─── 10 restarts → Mean DICE
              └─ Inner Fold 2 ─── 10 restarts → Mean DICE
              └─ TEST SET (held out)
```

### 1.2 Paper Protocol (FROM PAPER Section 2)

| Parameter | Value | Source |
|-----------|-------|--------|
| Outer folds | 3 | Paper: "nested cross-validation approach with three outer folds" |
| Inner folds | 3 per outer | Paper: "divided each outer fold's training data into three inner folds" |
| Restarts | 10 | Paper: "we trained the model with 10 restarts" |
| HPO location | Inner folds of first outer fold | Paper: "HPO was conducted on the inner folds of the first outer fold" |
| Epochs | 50 | Paper: "trained the model for 50 epochs" |

### 1.3 Results Aggregation

FROM PAPER Table 1:
- Report **mean (std)** across all configurations
- Statistical significance via **Wilcoxon signed-rank test**
- Multiple comparison correction via **Holm-Bonferroni**

---

## 2. Implementation

### 2.1 Experiment Configuration

**File:** `src/arc_meshchop/experiment/config.py`

```python
"""Experiment configuration for paper replication."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class ExperimentConfig:
    """Configuration for full experiment.

    FROM PAPER Section 2:
    - 3 outer folds × 3 inner folds × 10 restarts = 90 runs
    - MeshNet-26 with 147,474 parameters
    - Target: 0.876 DICE
    """

    # Data
    data_dir: Path = field(default_factory=lambda: Path("data/arc"))
    output_dir: Path = field(default_factory=lambda: Path("experiments"))

    # Model
    model_variant: Literal["meshnet_5", "meshnet_16", "meshnet_26"] = "meshnet_26"
    channels: int = 26

    # Cross-validation structure (FROM PAPER)
    num_outer_folds: int = 3
    num_inner_folds: int = 3
    num_restarts: int = 10

    # Training (FROM PAPER)
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 3e-5
    background_weight: float = 0.5
    warmup_pct: float = 0.01
    use_fp16: bool = True

    # Execution
    parallel_runs: int = 1  # Number of parallel training runs
    skip_completed: bool = True  # Skip runs that already have results
    save_all_checkpoints: bool = False  # Save checkpoint for every restart

    # Random seeds (for restarts)
    base_seed: int = 42

    def __post_init__(self) -> None:
        """Convert paths and validate."""
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)

        if self.model_variant == "meshnet_5":
            self.channels = 5
        elif self.model_variant == "meshnet_16":
            self.channels = 16
        else:
            self.channels = 26

    @property
    def total_runs(self) -> int:
        """Total number of training runs."""
        return self.num_outer_folds * self.num_inner_folds * self.num_restarts

    def get_restart_seed(self, restart: int) -> int:
        """Get seed for specific restart."""
        return self.base_seed + restart
```

### 2.2 Experiment Runner

**File:** `src/arc_meshchop/experiment/runner.py`

```python
"""Experiment runner for full paper replication.

Orchestrates 90 training runs:
- 3 outer folds × 3 inner folds × 10 restarts

FROM PAPER:
"we trained the model with 10 restarts"
"nested cross-validation approach with three outer folds"
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from arc_meshchop.experiment.config import ExperimentConfig

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Result from a single training run."""

    outer_fold: int
    inner_fold: int
    restart: int
    seed: int
    best_dice: float
    best_epoch: int
    final_train_loss: float
    checkpoint_path: str
    duration_seconds: float


@dataclass
class FoldResult:
    """Aggregated results for a single fold (across restarts)."""

    outer_fold: int
    inner_fold: int
    runs: list[RunResult]

    @property
    def mean_dice(self) -> float:
        """Mean DICE across restarts."""
        return sum(r.best_dice for r in self.runs) / len(self.runs)

    @property
    def std_dice(self) -> float:
        """Standard deviation of DICE across restarts."""
        import numpy as np
        return float(np.std([r.best_dice for r in self.runs]))

    @property
    def best_dice(self) -> float:
        """Best DICE among restarts."""
        return max(r.best_dice for r in self.runs)

    @property
    def best_run(self) -> RunResult:
        """Run with best DICE."""
        return max(self.runs, key=lambda r: r.best_dice)


@dataclass
class ExperimentResult:
    """Complete experiment results."""

    config: dict
    folds: list[FoldResult]
    test_results: list[dict]  # Per outer fold test set evaluation
    start_time: str
    end_time: str
    total_duration_hours: float

    @property
    def mean_dice(self) -> float:
        """Overall mean DICE."""
        all_dices = [r.best_dice for fold in self.folds for r in fold.runs]
        return sum(all_dices) / len(all_dices)

    @property
    def std_dice(self) -> float:
        """Overall standard deviation."""
        import numpy as np
        all_dices = [r.best_dice for fold in self.folds for r in fold.runs]
        return float(np.std(all_dices))


class ExperimentRunner:
    """Run full nested CV experiment.

    Implements the paper's experimental protocol:
    1. Train on inner folds of each outer fold
    2. 10 restarts per inner fold
    3. Select best model per outer fold
    4. Evaluate on outer fold test set
    5. Report mean ± std across all runs
    """

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize experiment runner.

        Args:
            config: Experiment configuration.
        """
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Track results
        self.results: list[RunResult] = []
        self.start_time = datetime.now()

    def run(self) -> ExperimentResult:
        """Run complete experiment.

        Returns:
            ExperimentResult with all training and test results.
        """
        logger.info(
            "Starting experiment: %d outer × %d inner × %d restarts = %d runs",
            self.config.num_outer_folds,
            self.config.num_inner_folds,
            self.config.num_restarts,
            self.config.total_runs,
        )

        # Run all training configurations
        for outer_fold in range(self.config.num_outer_folds):
            for inner_fold in range(self.config.num_inner_folds):
                for restart in range(self.config.num_restarts):
                    result = self._run_single(outer_fold, inner_fold, restart)
                    self.results.append(result)

                    # Log progress
                    completed = len(self.results)
                    logger.info(
                        "Progress: %d/%d (%.1f%%) - Fold %d.%d Restart %d: DICE=%.4f",
                        completed,
                        self.config.total_runs,
                        100 * completed / self.config.total_runs,
                        outer_fold,
                        inner_fold,
                        restart,
                        result.best_dice,
                    )

        # Aggregate results by fold
        folds = self._aggregate_by_fold()

        # Evaluate best models on test sets
        test_results = self._evaluate_on_test_sets(folds)

        # Create final result
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds() / 3600

        experiment_result = ExperimentResult(
            config=asdict(self.config),
            folds=folds,
            test_results=test_results,
            start_time=self.start_time.isoformat(),
            end_time=end_time.isoformat(),
            total_duration_hours=duration,
        )

        # Save results
        self._save_results(experiment_result)

        logger.info(
            "Experiment complete: DICE = %.4f ± %.4f",
            experiment_result.mean_dice,
            experiment_result.std_dice,
        )

        return experiment_result

    def _run_single(
        self,
        outer_fold: int,
        inner_fold: int,
        restart: int,
    ) -> RunResult:
        """Run single training configuration.

        Args:
            outer_fold: Outer fold index (0-2).
            inner_fold: Inner fold index (0-2).
            restart: Restart index (0-9).

        Returns:
            RunResult with training metrics.
        """
        import time

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

        run_id = f"fold_{outer_fold}_{inner_fold}_restart_{restart}"
        run_dir = self.config.output_dir / run_id

        # Check if already completed
        results_file = run_dir / "results.json"
        if self.config.skip_completed and results_file.exists():
            logger.info("Skipping completed run: %s", run_id)
            with open(results_file) as f:
                data = json.load(f)
            return RunResult(**data)

        start_time = time.time()
        seed = self.config.get_restart_seed(restart)

        # Set seeds
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Load dataset
        dataset_info = self._load_dataset_info()
        image_paths = dataset_info["image_paths"]
        mask_paths = dataset_info["mask_paths"]
        lesion_volumes = dataset_info["lesion_volumes"]
        acquisition_types = dataset_info["acquisition_types"]

        # Generate splits
        quintiles = [get_lesion_quintile(v) for v in lesion_volumes]
        strat_labels = create_stratification_labels(quintiles, acquisition_types)

        splits = generate_nested_cv_splits(
            n_samples=len(image_paths),
            stratification_labels=strat_labels,
            num_outer_folds=self.config.num_outer_folds,
            num_inner_folds=self.config.num_inner_folds,
            random_seed=42,  # Fixed for reproducible splits
        )

        split = splits.get_split(outer_fold, inner_fold)

        # Create datasets
        train_dataset = ARCDataset(
            image_paths=[Path(image_paths[i]) for i in split.train_indices],
            mask_paths=[Path(mask_paths[i]) for i in split.train_indices],
        )

        val_dataset = ARCDataset(
            image_paths=[Path(image_paths[i]) for i in split.val_indices],
            mask_paths=[Path(mask_paths[i]) for i in split.val_indices],
        )

        train_loader, val_loader = create_dataloaders(
            train_dataset,
            val_dataset,
            batch_size=1,
            num_workers=4,
        )

        # Create model
        device = get_device()
        model = MeshNet(channels=self.config.channels)

        # Training config
        training_config = TrainingConfig(
            epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            max_lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            background_weight=self.config.background_weight,
            pct_start=self.config.warmup_pct,
            use_fp16=self.config.use_fp16,
            checkpoint_dir=run_dir,
            random_seed=seed,
        )

        # Train
        trainer = Trainer(model, training_config, device=device)
        results = trainer.train(train_loader, val_loader)

        duration = time.time() - start_time

        # Create result
        run_result = RunResult(
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            restart=restart,
            seed=seed,
            best_dice=results["best_dice"],
            best_epoch=results["best_epoch"],
            final_train_loss=results["final_train_loss"],
            checkpoint_path=str(run_dir / "best.pt"),
            duration_seconds=duration,
        )

        # Save individual result
        run_dir.mkdir(parents=True, exist_ok=True)
        results_file.write_text(json.dumps(asdict(run_result), indent=2))

        return run_result

    def _aggregate_by_fold(self) -> list[FoldResult]:
        """Aggregate results by fold.

        Returns:
            List of FoldResult objects.
        """
        folds = []

        for outer in range(self.config.num_outer_folds):
            for inner in range(self.config.num_inner_folds):
                fold_runs = [
                    r for r in self.results
                    if r.outer_fold == outer and r.inner_fold == inner
                ]
                folds.append(FoldResult(
                    outer_fold=outer,
                    inner_fold=inner,
                    runs=fold_runs,
                ))

        return folds

    def _evaluate_on_test_sets(self, folds: list[FoldResult]) -> list[dict]:
        """Evaluate best models on held-out test sets.

        For each outer fold, select best model from inner folds
        and evaluate on the outer fold's test set.

        Args:
            folds: Aggregated fold results.

        Returns:
            Test results per outer fold.
        """
        from arc_meshchop.data import (
            ARCDataset,
            create_stratification_labels,
            generate_nested_cv_splits,
            get_lesion_quintile,
        )
        from arc_meshchop.evaluation import calculate_metrics
        from arc_meshchop.models import MeshNet
        from arc_meshchop.utils.device import get_device

        test_results = []
        device = get_device()

        dataset_info = self._load_dataset_info()
        image_paths = dataset_info["image_paths"]
        mask_paths = dataset_info["mask_paths"]
        lesion_volumes = dataset_info["lesion_volumes"]
        acquisition_types = dataset_info["acquisition_types"]

        quintiles = [get_lesion_quintile(v) for v in lesion_volumes]
        strat_labels = create_stratification_labels(quintiles, acquisition_types)

        splits = generate_nested_cv_splits(
            n_samples=len(image_paths),
            stratification_labels=strat_labels,
            num_outer_folds=self.config.num_outer_folds,
            num_inner_folds=self.config.num_inner_folds,
            random_seed=42,
        )

        for outer_fold in range(self.config.num_outer_folds):
            # Find best model for this outer fold
            outer_folds = [f for f in folds if f.outer_fold == outer_fold]
            best_fold = max(outer_folds, key=lambda f: f.mean_dice)
            best_run = best_fold.best_run

            logger.info(
                "Outer fold %d: Best model from inner fold %d restart %d (DICE=%.4f)",
                outer_fold,
                best_run.inner_fold,
                best_run.restart,
                best_run.best_dice,
            )

            # Load model
            model = MeshNet(channels=self.config.channels)
            model = model.to(device)
            checkpoint = torch.load(best_run.checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            # Get test indices
            split = splits.get_split(outer_fold, inner_fold=None)
            test_indices = split.test_indices

            # Create test dataset
            test_dataset = ARCDataset(
                image_paths=[Path(image_paths[i]) for i in test_indices],
                mask_paths=[Path(mask_paths[i]) for i in test_indices],
            )

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

            preds = torch.stack(all_preds)
            targets = torch.stack(all_targets)

            metrics = calculate_metrics(preds, targets)

            test_results.append({
                "outer_fold": outer_fold,
                "best_inner_fold": best_run.inner_fold,
                "best_restart": best_run.restart,
                "best_val_dice": best_run.best_dice,
                "test_dice": metrics["dice"],
                "test_avd": metrics["avd"],
                "test_mcc": metrics["mcc"],
                "num_test_samples": len(test_indices),
            })

            logger.info(
                "Outer fold %d test: DICE=%.4f, AVD=%.4f, MCC=%.4f",
                outer_fold,
                metrics["dice"],
                metrics["avd"],
                metrics["mcc"],
            )

        return test_results

    def _load_dataset_info(self) -> dict:
        """Load dataset info from JSON."""
        info_path = self.config.data_dir / "dataset_info.json"
        if not info_path.exists():
            raise FileNotFoundError(
                f"Dataset info not found: {info_path}. "
                "Run 'arc-meshchop download' first."
            )
        with open(info_path) as f:
            return json.load(f)

    def _save_results(self, result: ExperimentResult) -> None:
        """Save experiment results to JSON."""
        results_path = self.config.output_dir / "experiment_results.json"

        # Convert to serializable format
        data = {
            "config": result.config,
            "folds": [
                {
                    "outer_fold": f.outer_fold,
                    "inner_fold": f.inner_fold,
                    "mean_dice": f.mean_dice,
                    "std_dice": f.std_dice,
                    "best_dice": f.best_dice,
                    "runs": [asdict(r) for r in f.runs],
                }
                for f in result.folds
            ],
            "test_results": result.test_results,
            "summary": {
                "mean_dice": result.mean_dice,
                "std_dice": result.std_dice,
                "target_dice": 0.876,  # FROM PAPER
                "paper_parity": result.mean_dice >= 0.86,
            },
            "timing": {
                "start_time": result.start_time,
                "end_time": result.end_time,
                "total_duration_hours": result.total_duration_hours,
            },
        }

        results_path.write_text(json.dumps(data, indent=2))
        logger.info("Results saved to %s", results_path)


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run complete experiment with given config.

    Convenience function for running experiments.

    Args:
        config: Experiment configuration.

    Returns:
        ExperimentResult with all results.
    """
    runner = ExperimentRunner(config)
    return runner.run()
```

### 2.3 CLI Command

**Add to:** `src/arc_meshchop/cli/main.py`

```python
@app.command()
def experiment(
    data_dir: Annotated[
        Path,
        typer.Option("--data-dir", "-d", help="Directory with dataset_info.json"),
    ] = Path("data/arc"),
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory for experiment"),
    ] = Path("experiments"),
    variant: Annotated[
        str,
        typer.Option("--variant", "-v", help="Model variant: meshnet_5, meshnet_16, meshnet_26"),
    ] = "meshnet_26",
    num_restarts: Annotated[
        int,
        typer.Option("--restarts", "-r", help="Number of restarts per fold"),
    ] = 10,
    epochs: Annotated[
        int,
        typer.Option("--epochs", "-e", help="Number of epochs per run"),
    ] = 50,
    skip_completed: Annotated[
        bool,
        typer.Option("--skip-completed/--no-skip", help="Skip completed runs"),
    ] = True,
) -> None:
    """Run full nested CV experiment.

    Runs all 90 configurations (3 outer × 3 inner × 10 restarts)
    and produces paper-comparable results.

    FROM PAPER:
    - MeshNet-26: 0.876 (0.016) DICE
    - MeshNet-16: 0.873 (0.007) DICE
    - MeshNet-5: 0.848 (0.023) DICE
    """
    from arc_meshchop.experiment.config import ExperimentConfig
    from arc_meshchop.experiment.runner import run_experiment

    config = ExperimentConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        model_variant=variant,
        num_restarts=num_restarts,
        epochs=epochs,
        skip_completed=skip_completed,
    )

    typer.echo(f"Running experiment: {variant}")
    typer.echo(f"Total runs: {config.total_runs}")
    typer.echo(f"Output: {output_dir}")

    result = run_experiment(config)

    typer.echo("\n" + "=" * 60)
    typer.echo("EXPERIMENT COMPLETE")
    typer.echo("=" * 60)
    typer.echo(f"Model: MeshNet-{config.channels}")
    typer.echo(f"DICE: {result.mean_dice:.4f} ± {result.std_dice:.4f}")
    typer.echo(f"Paper Target: 0.876 (MeshNet-26)")
    typer.echo(f"Duration: {result.total_duration_hours:.1f} hours")
    typer.echo(f"Results: {output_dir / 'experiment_results.json'}")
```

---

## 3. Tests

**File:** `tests/test_experiment/test_runner.py`

```python
"""Tests for experiment runner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestExperimentConfig:
    """Tests for ExperimentConfig."""

    def test_total_runs(self) -> None:
        """Verify total run calculation."""
        from arc_meshchop.experiment.config import ExperimentConfig

        config = ExperimentConfig(
            num_outer_folds=3,
            num_inner_folds=3,
            num_restarts=10,
        )

        assert config.total_runs == 90

    def test_channel_mapping(self) -> None:
        """Verify model variant to channels mapping."""
        from arc_meshchop.experiment.config import ExperimentConfig

        config5 = ExperimentConfig(model_variant="meshnet_5")
        config16 = ExperimentConfig(model_variant="meshnet_16")
        config26 = ExperimentConfig(model_variant="meshnet_26")

        assert config5.channels == 5
        assert config16.channels == 16
        assert config26.channels == 26

    def test_restart_seeds(self) -> None:
        """Verify restart seeds are unique."""
        from arc_meshchop.experiment.config import ExperimentConfig

        config = ExperimentConfig(base_seed=42, num_restarts=10)

        seeds = [config.get_restart_seed(i) for i in range(10)]

        # All unique
        assert len(set(seeds)) == 10
        # Sequential from base
        assert seeds == list(range(42, 52))


class TestFoldResult:
    """Tests for FoldResult aggregation."""

    def test_mean_dice(self) -> None:
        """Verify mean DICE calculation."""
        from arc_meshchop.experiment.runner import FoldResult, RunResult

        runs = [
            RunResult(
                outer_fold=0,
                inner_fold=0,
                restart=i,
                seed=42 + i,
                best_dice=0.8 + i * 0.01,
                best_epoch=50,
                final_train_loss=0.1,
                checkpoint_path="/path",
                duration_seconds=100,
            )
            for i in range(5)
        ]

        fold = FoldResult(outer_fold=0, inner_fold=0, runs=runs)

        # Mean of [0.8, 0.81, 0.82, 0.83, 0.84] = 0.82
        assert fold.mean_dice == pytest.approx(0.82)

    def test_best_run(self) -> None:
        """Verify best run selection."""
        from arc_meshchop.experiment.runner import FoldResult, RunResult

        runs = [
            RunResult(
                outer_fold=0,
                inner_fold=0,
                restart=i,
                seed=42,
                best_dice=0.8 if i != 2 else 0.9,  # Restart 2 is best
                best_epoch=50,
                final_train_loss=0.1,
                checkpoint_path="/path",
                duration_seconds=100,
            )
            for i in range(5)
        ]

        fold = FoldResult(outer_fold=0, inner_fold=0, runs=runs)

        assert fold.best_run.restart == 2
        assert fold.best_dice == 0.9
```

---

## 4. Implementation Checklist

- [ ] Create `src/arc_meshchop/experiment/__init__.py`
- [ ] Create `src/arc_meshchop/experiment/config.py`
- [ ] Create `src/arc_meshchop/experiment/runner.py`
- [ ] Implement `ExperimentConfig` dataclass
- [ ] Implement `ExperimentRunner` class
- [ ] Implement `_run_single()` for individual training
- [ ] Implement `_aggregate_by_fold()` for results aggregation
- [ ] Implement `_evaluate_on_test_sets()` for final evaluation
- [ ] Add `experiment` CLI command
- [ ] Add resume capability for interrupted experiments
- [ ] Create tests in `tests/test_experiment/`
- [ ] Test with quick run (2 epochs, 1 restart)
- [ ] Document expected runtime and resource requirements

---

## 5. Verification Commands

```bash
# Quick test (1 restart, 2 epochs)
uv run arc-meshchop experiment \
    --data-dir data/arc \
    --output experiments/test \
    --variant meshnet_5 \
    --restarts 1 \
    --epochs 2

# Full MeshNet-26 experiment (paper replication)
uv run arc-meshchop experiment \
    --data-dir data/arc \
    --output experiments/meshnet26 \
    --variant meshnet_26 \
    --restarts 10 \
    --epochs 50

# View results
cat experiments/meshnet26/experiment_results.json | jq '.summary'
```

---

## 6. Expected Runtime

| Configuration | Runs | Time per Run | Total Time |
|---------------|------|--------------|------------|
| Quick test (2 epochs) | 9 | ~5 min | ~45 min |
| MeshNet-5 (50 epochs) | 90 | ~30 min | ~45 hours |
| MeshNet-16 (50 epochs) | 90 | ~45 min | ~67 hours |
| MeshNet-26 (50 epochs) | 90 | ~60 min | ~90 hours |

> **NOTE:** Times estimated for single A100 GPU. Actual times depend on hardware.
> For faster iteration, use `--skip-completed` to resume interrupted experiments.

---

## 7. Expected Results (FROM PAPER)

| Variant | Parameters | DICE (Paper) | Target |
|---------|------------|--------------|--------|
| MeshNet-5 | 5,682 | 0.848 (0.023) | > 0.84 |
| MeshNet-16 | 56,194 | 0.873 (0.007) | > 0.86 |
| MeshNet-26 | 147,474 | 0.876 (0.016) | **0.876** |

Success criteria:
- MeshNet-26 should achieve **DICE ≥ 0.86** (within 2% of paper)
- Standard deviation should be **< 0.03** (comparable to paper)
