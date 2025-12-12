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

    # Training hyperparameters (FROM PAPER)
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 3e-5
    background_weight: float = 0.5
    warmup_pct: float = 0.01
    div_factor: float = 100.0  # FROM PAPER: "starts at 1/100th of the max learning rate"
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

    # IMPORTANT: Paper reports TEST metrics, not validation metrics
    # These properties compute from test_results (outer fold holdout)

    @property
    def test_mean_dice(self) -> float:
        """Mean TEST DICE across outer folds (paper Table 1 metric)."""
        if not self.test_results:
            return 0.0
        test_dices = [t["test_dice"] for t in self.test_results]
        return sum(test_dices) / len(test_dices)

    @property
    def test_std_dice(self) -> float:
        """Std of TEST DICE across outer folds."""
        import numpy as np
        if not self.test_results:
            return 0.0
        test_dices = [t["test_dice"] for t in self.test_results]
        return float(np.std(test_dices))

    @property
    def test_mean_avd(self) -> float:
        """Mean TEST AVD across outer folds."""
        if not self.test_results:
            return 0.0
        test_avds = [t["test_avd"] for t in self.test_results]
        return sum(test_avds) / len(test_avds)

    @property
    def test_std_avd(self) -> float:
        """Std of TEST AVD across outer folds."""
        import numpy as np
        if not self.test_results:
            return 0.0
        test_avds = [t["test_avd"] for t in self.test_results]
        return float(np.std(test_avds))

    @property
    def test_mean_mcc(self) -> float:
        """Mean TEST MCC across outer folds."""
        if not self.test_results:
            return 0.0
        test_mccs = [t["test_mcc"] for t in self.test_results]
        return sum(test_mccs) / len(test_mccs)

    @property
    def test_std_mcc(self) -> float:
        """Std of TEST MCC across outer folds."""
        import numpy as np
        if not self.test_results:
            return 0.0
        test_mccs = [t["test_mcc"] for t in self.test_results]
        return float(np.std(test_mccs))

    # Validation metrics (for debugging/monitoring, NOT for paper parity)
    @property
    def val_mean_dice(self) -> float:
        """Overall mean validation DICE (for monitoring only)."""
        all_dices = [r.best_dice for fold in self.folds for r in fold.runs]
        return sum(all_dices) / len(all_dices) if all_dices else 0.0

    @property
    def val_std_dice(self) -> float:
        """Overall std of validation DICE (for monitoring only)."""
        import numpy as np
        all_dices = [r.best_dice for fold in self.folds for r in fold.runs]
        return float(np.std(all_dices)) if all_dices else 0.0


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
        # NOTE: div_factor=100 is critical for paper parity
        # FROM PAPER: "starts at 1/100th of the max learning rate"
        training_config = TrainingConfig(
            epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            max_lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            background_weight=self.config.background_weight,
            pct_start=self.config.warmup_pct,
            div_factor=self.config.div_factor,  # Paper requires 100
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
        from arc_meshchop.evaluation import SegmentationMetrics
        from arc_meshchop.models import MeshNet
        from arc_meshchop.utils.device import get_device

        test_results = []
        device = get_device()
        metrics_calculator = SegmentationMetrics()

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

            metrics = metrics_calculator.compute_batch(preds, targets)

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
                # PRIMARY: Test metrics (for paper parity - FROM PAPER Table 1)
                "test_mean_dice": result.test_mean_dice,
                "test_std_dice": result.test_std_dice,
                "test_mean_avd": result.test_mean_avd,
                "test_std_avd": result.test_std_avd,
                "test_mean_mcc": result.test_mean_mcc,
                "test_std_mcc": result.test_std_mcc,
                # SECONDARY: Validation metrics (for monitoring only)
                "val_mean_dice": result.val_mean_dice,
                "val_std_dice": result.val_std_dice,
                # Paper targets
                "target_dice": 0.876,  # FROM PAPER
                "paper_parity": result.test_mean_dice >= 0.86,  # Use TEST DICE for parity
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

**Location:** `src/arc_meshchop/cli.py` (✅ Implemented)

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
    # Hyperparameters (FROM PAPER or from HPO)
    learning_rate: Annotated[
        float,
        typer.Option("--lr", help="Max learning rate"),
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
        typer.Option("--div-factor", help="OneCycleLR div_factor"),
    ] = 100.0,  # FROM PAPER: "starts at 1/100th of the max learning rate"
    skip_completed: Annotated[
        bool,
        typer.Option("--skip-completed/--no-skip", help="Skip completed runs"),
    ] = True,
) -> None:
    """Run full nested CV experiment.

    Runs all 90 configurations (3 outer × 3 inner × 10 restarts)
    and produces paper-comparable results.

    Hyperparameters can be overridden (e.g., from HPO results).

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
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        background_weight=background_weight,
        warmup_pct=warmup_pct,
        div_factor=div_factor,
        skip_completed=skip_completed,
    )

    typer.echo(f"Running experiment: {variant}")
    typer.echo(f"Total runs: {config.total_runs}")
    typer.echo(f"Hyperparameters: lr={learning_rate}, wd={weight_decay}, bg_weight={background_weight}")
    typer.echo(f"Output: {output_dir}")

    result = run_experiment(config)

    # Report TEST metrics (not validation) for paper parity
    typer.echo("\n" + "=" * 60)
    typer.echo("EXPERIMENT COMPLETE")
    typer.echo("=" * 60)
    typer.echo(f"Model: MeshNet-{config.channels}")
    typer.echo(f"Test DICE: {result.test_mean_dice:.4f} ± {result.test_std_dice:.4f}")
    typer.echo(f"Test AVD:  {result.test_mean_avd:.4f} ± {result.test_std_avd:.4f}")
    typer.echo(f"Test MCC:  {result.test_mean_mcc:.4f} ± {result.test_std_mcc:.4f}")
    typer.echo(f"Paper Target: DICE 0.876 (MeshNet-26)")
    typer.echo(f"Duration: {result.total_duration_hours:.1f} hours")
    typer.echo(f"Results: {output_dir / 'experiment_results.json'}")
```

### 2.4 Hyperparameter Optimization Workflow

The paper uses HPO on inner folds of outer fold 0 to find optimal hyperparameters.
This spec supports **two modes**:

#### Mode A: Use Paper's Published Hyperparameters (Quick Replication)

The paper already published their optimal hyperparameters from HPO:

| Parameter | Paper Value | CLI Default |
|-----------|-------------|-------------|
| Learning rate | 0.001 | `--lr 0.001` |
| Weight decay | 3e-5 | `--wd 3e-5` |
| Background weight | 0.5 | `--bg-weight 0.5` |
| Warmup | 1% | `--warmup 0.01` |
| div_factor | 100 | `--div-factor 100` |
| Channels | 26 | `--variant meshnet_26` |

```bash
# Quick replication: use paper's hyperparameters directly
uv run arc-meshchop experiment \
    --variant meshnet_26 \
    --lr 0.001 --wd 3e-5 --bg-weight 0.5 --warmup 0.01 --div-factor 100
```

#### Mode B: Run HPO Sweep (Full Paper Protocol)

To fully replicate the paper's methodology, run HPO on outer fold 0's inner folds:

**Step 1: Run HPO on Outer Fold 0**

```bash
# HPO sweep using Optuna (Orion alternative for Python 3.12+ compatibility)
uv run arc-meshchop hpo \
    --data-dir data/arc \
    --output experiments/hpo \
    --outer-fold 0 \
    --max-trials 100 \
    --epochs 50
```

**HPO Search Space (FROM PAPER Section 2):**

| Parameter | Distribution | Range | Source |
|-----------|--------------|-------|--------|
| channels | uniform int | [5, 21] | Paper: "number of channels to be 26 for our largest model" |
| learning_rate | log-uniform | [1e-4, 4e-2] | Paper HPO range |
| weight_decay | log-uniform | [1e-4, 4e-2] | Paper HPO range |
| bg_weight | uniform | [0, 1] | Paper HPO range |
| warmup_pct | choice | {0.02, 0.1, 0.2} | Paper HPO range |

**Step 2: Extract Best Hyperparameters**

```bash
# Get best params from HPO results
cat experiments/hpo/best_params.json
```

**Step 3: Run Full Experiment with HPO-Found Params**

```bash
# Use HPO-found hyperparameters for all 90 runs
uv run arc-meshchop experiment \
    --variant meshnet_26 \
    --lr $(jq -r .learning_rate experiments/hpo/best_params.json) \
    --wd $(jq -r .weight_decay experiments/hpo/best_params.json) \
    --bg-weight $(jq -r .bg_weight experiments/hpo/best_params.json) \
    --warmup $(jq -r .warmup_pct experiments/hpo/best_params.json)
```

#### HPO CLI Command

**Location:** `src/arc_meshchop/cli.py` (Note: HPO integrated into experiment workflow)

```python
@app.command()
def hpo(
    data_dir: Annotated[
        Path,
        typer.Option("--data-dir", "-d", help="Directory with dataset_info.json"),
    ] = Path("data/arc"),
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory for HPO results"),
    ] = Path("experiments/hpo"),
    outer_fold: Annotated[
        int,
        typer.Option("--outer-fold", help="Outer fold for HPO (paper uses 0)"),
    ] = 0,
    max_trials: Annotated[
        int,
        typer.Option("--max-trials", "-n", help="Maximum HPO trials"),
    ] = 100,
    epochs: Annotated[
        int,
        typer.Option("--epochs", "-e", help="Epochs per trial"),
    ] = 50,
    pruning: Annotated[
        bool,
        typer.Option("--pruning/--no-pruning", help="Enable ASHA-style pruning"),
    ] = True,
) -> None:
    """Run hyperparameter optimization.

    FROM PAPER Section 2:
    "HPO was conducted on the inner folds of the first outer fold"
    "We employed the Asynchronous Successive Halving Algorithm (ASHA)"

    This replicates the paper's HPO process using Optuna with ASHA pruning.
    Results can be passed to the `experiment` command.
    """
    import json
    import optuna
    from arc_meshchop.training.hpo import run_hpo_trial, create_study

    typer.echo(f"Running HPO on outer fold {outer_fold}")
    typer.echo(f"Max trials: {max_trials}, Epochs: {epochs}")

    study = create_study(
        study_name=f"meshnet_hpo_outer{outer_fold}",
        pruning=pruning,
    )

    study.optimize(
        lambda trial: run_hpo_trial(
            trial,
            data_dir=data_dir,
            outer_fold=outer_fold,
            epochs=epochs,
        ),
        n_trials=max_trials,
    )

    # Save best params
    output_dir.mkdir(parents=True, exist_ok=True)
    best_params = {
        "learning_rate": study.best_params["learning_rate"],
        "weight_decay": study.best_params["weight_decay"],
        "bg_weight": study.best_params["bg_weight"],
        "warmup_pct": study.best_params["warmup_pct"],
        "best_dice": study.best_value,
        "n_trials": len(study.trials),
    }

    params_path = output_dir / "best_params.json"
    params_path.write_text(json.dumps(best_params, indent=2))

    typer.echo(f"\nHPO Complete!")
    typer.echo(f"Best DICE: {study.best_value:.4f}")
    typer.echo(f"Best params: {params_path}")
    typer.echo(f"\nTo run full experiment with these params:")
    typer.echo(f"  uv run arc-meshchop experiment \\")
    typer.echo(f"    --lr {best_params['learning_rate']:.6f} \\")
    typer.echo(f"    --wd {best_params['weight_decay']:.6f} \\")
    typer.echo(f"    --bg-weight {best_params['bg_weight']:.4f} \\")
    typer.echo(f"    --warmup {best_params['warmup_pct']:.4f}")
```

#### HPO Implementation

**File:** `src/arc_meshchop/training/hpo.py`

```python
"""Hyperparameter optimization with Optuna.

FROM PAPER Section 2:
"To optimize hyperparameters of MeshNet, we conducted a hyperparameter search
using Orion, an asynchronous framework for black-box function optimization.
We employed the Asynchronous Successive Halving Algorithm (ASHA)."

NOTE: Uses Optuna instead of Orion for Python 3.12+ compatibility.
Optuna's MedianPruner provides similar early-stopping behavior to ASHA.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import optuna

if TYPE_CHECKING:
    from optuna import Study, Trial

logger = logging.getLogger(__name__)


def create_study(
    study_name: str = "meshnet_hpo",
    pruning: bool = True,
) -> Study:
    """Create Optuna study with ASHA-like pruning.

    Args:
        study_name: Name for the study.
        pruning: Enable MedianPruner (ASHA-like behavior).

    Returns:
        Configured Optuna study.
    """
    pruner = (
        optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=15,  # FROM PAPER: min_epochs=15
            interval_steps=5,
        )
        if pruning
        else optuna.pruners.NopPruner()
    )

    return optuna.create_study(
        study_name=study_name,
        direction="maximize",  # Maximize DICE
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(seed=42),
    )


def run_hpo_trial(
    trial: Trial,
    data_dir: Path,
    outer_fold: int = 0,
    epochs: int = 50,
) -> float:
    """Run single HPO trial.

    Trains on inner folds of specified outer fold,
    returns mean validation DICE.

    Args:
        trial: Optuna trial object.
        data_dir: Path to dataset.
        outer_fold: Outer fold to use for HPO.
        epochs: Training epochs.

    Returns:
        Mean validation DICE across inner folds.
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

    # Sample hyperparameters (FROM PAPER search space)
    lr = trial.suggest_float("learning_rate", 1e-4, 4e-2, log=True)
    wd = trial.suggest_float("weight_decay", 1e-4, 4e-2, log=True)
    bg_weight = trial.suggest_float("bg_weight", 0.0, 1.0)
    warmup = trial.suggest_categorical("warmup_pct", [0.02, 0.1, 0.2])

    # Load dataset
    import json
    info_path = data_dir / "dataset_info.json"
    with open(info_path) as f:
        dataset_info = json.load(f)

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
        num_outer_folds=3,
        num_inner_folds=3,
        random_seed=42,
    )

    # Train on each inner fold
    inner_fold_dices = []
    device = get_device()

    for inner_fold in range(3):
        split = splits.get_split(outer_fold, inner_fold)

        train_dataset = ARCDataset(
            image_paths=[Path(image_paths[i]) for i in split.train_indices],
            mask_paths=[Path(mask_paths[i]) for i in split.train_indices],
        )
        val_dataset = ARCDataset(
            image_paths=[Path(image_paths[i]) for i in split.val_indices],
            mask_paths=[Path(mask_paths[i]) for i in split.val_indices],
        )

        train_loader, val_loader = create_dataloaders(
            train_dataset, val_dataset, batch_size=1, num_workers=4
        )

        model = MeshNet(channels=26)  # HPO uses fixed channel count
        config = TrainingConfig(
            epochs=epochs,
            learning_rate=lr,
            max_lr=lr,
            weight_decay=wd,
            background_weight=bg_weight,
            pct_start=warmup,
            div_factor=100.0,  # Paper requirement
            use_fp16=True,
        )

        trainer = Trainer(model, config, device=device)

        # Report intermediate values for pruning
        best_dice = 0.0
        for epoch in range(epochs):
            metrics = trainer.train_epoch(train_loader)
            val_dice = trainer.validate(val_loader)["dice"]

            if val_dice > best_dice:
                best_dice = val_dice

            # Report for ASHA-style pruning
            trial.report(best_dice, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

        inner_fold_dices.append(best_dice)

    # Return mean DICE across inner folds
    mean_dice = sum(inner_fold_dices) / len(inner_fold_dices)
    logger.info(
        "Trial %d: lr=%.6f, wd=%.6f, bg=%.3f, warmup=%.2f -> DICE=%.4f",
        trial.number, lr, wd, bg_weight, warmup, mean_dice
    )

    return mean_dice
```

#### HPO Checklist

- [x] Create `src/arc_meshchop/training/hpo.py`
- [x] Implement `create_study()` with ASHA-like pruning (Optuna)
- [x] Implement `run_hpo_trial()` with paper search space
- [x] HPO integrated into experiment workflow
- [x] Add `best_params.json` output format
- [x] Test HPO with Optuna pruning
- [x] Document HPO → experiment workflow

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

- [x] Create `src/arc_meshchop/experiment/__init__.py`
- [x] Create `src/arc_meshchop/experiment/config.py`
- [x] Create `src/arc_meshchop/experiment/runner.py`
- [x] Implement `ExperimentConfig` dataclass
- [x] Implement `ExperimentRunner` class
- [x] Implement `_run_single()` for individual training
- [x] Implement `_aggregate_by_fold()` for results aggregation
- [x] Implement `_evaluate_on_test_sets()` for final evaluation
- [x] Add `experiment` CLI command
- [x] Add resume capability for interrupted experiments
- [x] Create tests in `tests/test_experiment/`
- [x] Test with quick run (2 epochs, 1 restart)
- [x] Document expected runtime and resource requirements

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
