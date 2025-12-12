"""Experiment runner for full paper replication.

Orchestrates 90 training runs:
- 3 outer folds x 3 inner folds x 10 restarts

FROM PAPER:
"we trained the model with 10 restarts"
"nested cross-validation approach with three outer folds"
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
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
        all_dices = [r.best_dice for fold in self.folds for r in fold.runs]
        return float(np.std(all_dices)) if all_dices else 0.0


class ExperimentRunner:
    """Run full nested CV experiment.

    Implements the paper's experimental protocol:
    1. Train on inner folds of each outer fold
    2. 10 restarts per inner fold
    3. Select best model per outer fold
    4. Evaluate on outer fold test set
    5. Report mean +/- std across all runs
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
            "Starting experiment: %d outer x %d inner x %d restarts = %d runs",
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
            "Experiment complete: TEST DICE = %.4f +/- %.4f",
            experiment_result.test_mean_dice,
            experiment_result.test_std_dice,
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
            checkpoint = torch.load(best_run.checkpoint_path, map_location=device, weights_only=False)
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
