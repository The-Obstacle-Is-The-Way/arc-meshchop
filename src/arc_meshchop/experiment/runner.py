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

    # IMPORTANT: Paper reports TEST metrics computed from POOLED PER-SUBJECT scores
    # across all outer folds (n≈224), NOT from fold-level means (n=3).
    #
    # Evidence from paper:
    # - Figure 1: Boxplot showing per-subject distribution (median, IQR, outliers)
    # - Figure 2: "median DICE score with interquartile range (IQR) error bars"
    # - Table 1: Wilcoxon signed-rank test requires paired per-subject data
    # - Std values (0.005-0.02) consistent with per-subject variation, not fold variation
    #
    # See docs/bugs/BUG-002-metric-aggregation.md for full analysis.

    def _get_pooled_scores(self, metric: str) -> list[float]:
        """Get pooled per-subject scores across all outer folds.

        Args:
            metric: One of 'dice', 'avd', 'mcc'.

        Returns:
            List of all per-subject scores pooled from outer test folds.
        """
        key = f"per_subject_{metric}"
        all_scores: list[float] = []
        for t in self.test_results:
            if key in t:
                all_scores.extend(t[key])
            else:
                # Fallback for legacy results without per-subject data
                all_scores.append(t.get(f"test_{metric}", 0.0))
        return all_scores

    @property
    def test_mean_dice(self) -> float:
        """Mean TEST DICE across all subjects (paper Table 1 metric)."""
        scores = self._get_pooled_scores("dice")
        return float(np.mean(scores)) if scores else 0.0

    @property
    def test_std_dice(self) -> float:
        """Std of TEST DICE across all subjects (paper Table 1 metric)."""
        scores = self._get_pooled_scores("dice")
        return float(np.std(scores)) if scores else 0.0

    @property
    def test_median_dice(self) -> float:
        """Median TEST DICE (for Figure 2 central point)."""
        scores = self._get_pooled_scores("dice")
        return float(np.median(scores)) if scores else 0.0

    @property
    def test_iqr_dice(self) -> tuple[float, float]:
        """IQR of TEST DICE (for Figure 2 error bars)."""
        scores = self._get_pooled_scores("dice")
        if not scores:
            return (0.0, 0.0)
        return (float(np.percentile(scores, 25)), float(np.percentile(scores, 75)))

    @property
    def test_mean_avd(self) -> float:
        """Mean TEST AVD across all subjects."""
        scores = self._get_pooled_scores("avd")
        return float(np.mean(scores)) if scores else 0.0

    @property
    def test_std_avd(self) -> float:
        """Std of TEST AVD across all subjects."""
        scores = self._get_pooled_scores("avd")
        return float(np.std(scores)) if scores else 0.0

    @property
    def test_mean_mcc(self) -> float:
        """Mean TEST MCC across all subjects."""
        scores = self._get_pooled_scores("mcc")
        return float(np.mean(scores)) if scores else 0.0

    @property
    def test_std_mcc(self) -> float:
        """Std of TEST MCC across all subjects."""
        scores = self._get_pooled_scores("mcc")
        return float(np.std(scores)) if scores else 0.0

    def get_per_subject_scores(self) -> dict[int, dict[str, float]]:
        """Get per-subject scores indexed by subject ID for Wilcoxon pairing.

        Returns:
            Dict mapping subject index to metrics dict.
        """
        scores_by_subject: dict[int, dict[str, float]] = {}
        for t in self.test_results:
            indices = t.get("subject_indices", [])
            dices = t.get("per_subject_dice", [])
            avds = t.get("per_subject_avd", [])
            mccs = t.get("per_subject_mcc", [])
            for i, idx in enumerate(indices):
                scores_by_subject[idx] = {
                    "dice": dices[i] if i < len(dices) else 0.0,
                    "avd": avds[i] if i < len(avds) else 0.0,
                    "mcc": mccs[i] if i < len(mccs) else 0.0,
                }
        return scores_by_subject

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

        # Create run directory before training (checkpointing needs this)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Check if already completed
        results_file = run_dir / "results.json"
        if self.config.skip_completed and results_file.exists():
            logger.info("Skipping completed run: %s", run_id)
            with results_file.open() as f:
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
            cache_dir=self.config.data_dir / "cache" / run_id / "train",
        )

        val_dataset = ARCDataset(
            image_paths=[Path(image_paths[i]) for i in split.val_indices],
            mask_paths=[Path(mask_paths[i]) for i in split.val_indices],
            cache_dir=self.config.data_dir / "cache" / run_id / "val",
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
            best_dice=float(results["best_dice"]),
            best_epoch=int(results["best_epoch"]),
            final_train_loss=float(results["final_train_loss"]),
            checkpoint_path=str(run_dir / "best.pt"),
            duration_seconds=duration,
        )

        # Save individual result (run_dir already created above)
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
                    r for r in self.results if r.outer_fold == outer and r.inner_fold == inner
                ]
                # Fail fast if no runs collected (prevents div-by-zero in FoldResult)
                if not fold_runs:
                    raise RuntimeError(
                        f"No runs collected for fold {outer}.{inner}. "
                        f"Check num_restarts ({self.config.num_restarts}) and run execution."
                    )
                folds.append(
                    FoldResult(
                        outer_fold=outer,
                        inner_fold=inner,
                        runs=fold_runs,
                    )
                )

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
            checkpoint = torch.load(
                best_run.checkpoint_path, map_location=device, weights_only=True
            )
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
            dice_scores = []
            avd_scores = []
            mcc_scores = []

            with torch.no_grad():
                for idx in range(len(test_dataset)):
                    image, mask = test_dataset[idx]
                    image = image.unsqueeze(0).to(device)

                    output = model(image)
                    pred = output.argmax(dim=1).squeeze(0).cpu()

                    scores = metrics_calculator.compute_single(pred, mask)
                    dice_scores.append(scores["dice"])
                    avd_scores.append(scores["avd"])
                    mcc_scores.append(scores["mcc"])

            # Compute fold-level means (for backward compatibility and logging)
            test_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
            test_avd = float(np.mean(avd_scores)) if avd_scores else 0.0
            test_mcc = float(np.mean(mcc_scores)) if mcc_scores else 0.0

            # Store both fold-level means AND per-subject scores
            # Per-subject scores are required for:
            # - Paper-parity std computation (n≈224, not n=3)
            # - Boxplot generation (Figure 1)
            # - IQR error bars (Figure 2)
            # - Wilcoxon paired testing (Table 1)
            # See docs/bugs/BUG-002-metric-aggregation.md
            test_results.append(
                {
                    "outer_fold": outer_fold,
                    "best_inner_fold": best_run.inner_fold,
                    "best_restart": best_run.restart,
                    "best_val_dice": best_run.best_dice,
                    # Fold-level means (for logging/backward compat)
                    "test_dice": test_dice,
                    "test_avd": test_avd,
                    "test_mcc": test_mcc,
                    "num_test_samples": len(test_indices),
                    # Per-subject scores (for paper-parity aggregation)
                    "subject_indices": list(test_indices),
                    "per_subject_dice": dice_scores,
                    "per_subject_avd": avd_scores,
                    "per_subject_mcc": mcc_scores,
                }
            )

            logger.info(
                "Outer fold %d test: DICE=%.4f, AVD=%.4f, MCC=%.4f",
                outer_fold,
                test_dice,
                test_avd,
                test_mcc,
            )

        return test_results

    def _load_dataset_info(self) -> dict:
        """Load dataset info from JSON."""
        info_path = self.config.data_dir / "dataset_info.json"
        if not info_path.exists():
            raise FileNotFoundError(
                f"Dataset info not found: {info_path}. Run 'arc-meshchop download' first."
            )
        with info_path.open() as f:
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
                # PRIMARY: Test metrics computed from POOLED per-subject scores (n≈224)
                # This matches paper protocol - see docs/bugs/BUG-002-metric-aggregation.md
                "test_mean_dice": result.test_mean_dice,
                "test_std_dice": result.test_std_dice,
                "test_median_dice": result.test_median_dice,  # For Figure 2
                "test_iqr_dice": result.test_iqr_dice,  # For Figure 2 error bars
                "test_mean_avd": result.test_mean_avd,
                "test_std_avd": result.test_std_avd,
                "test_mean_mcc": result.test_mean_mcc,
                "test_std_mcc": result.test_std_mcc,
                # Total subjects in pooled test set
                "num_test_subjects": len(result._get_pooled_scores("dice")),
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
