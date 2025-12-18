"""Experiment runner for full paper replication.

Orchestrates 30 training runs:
- 3 outer folds x 10 restarts = 30 runs

FROM PAPER:
"Hyperparameter optimization was conducted on the inner folds of the first
outer fold. The optimized hyperparameters were then applied to train models
on all outer folds."

This means:
- HP search was only on Outer Fold 1 (we skip this, using paper's final HPs)
- Final training uses FULL outer-train data (no inner fold split)
- Fixed epochs (50) - no validation-based early stopping needed
- 10 restarts for stability/averaging

See docs/REPRODUCIBILITY.md for full protocol details.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from arc_meshchop.utils.paths import resolve_dataset_path

if TYPE_CHECKING:
    from arc_meshchop.experiment.config import ExperimentConfig

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Result from a single training run.

    In the paper's protocol, we train on FULL outer-train (no inner fold split)
    for fixed epochs. The "best_dice" here is the TEST DICE on outer-test,
    evaluated after training completes.
    """

    outer_fold: int
    restart: int
    seed: int
    test_dice: float  # DICE on outer test set
    test_avd: float  # AVD on outer test set
    test_mcc: float  # MCC on outer test set
    final_train_loss: float
    checkpoint_path: str
    duration_seconds: float
    # Per-subject scores for paper-parity aggregation
    per_subject_dice: list[float]
    per_subject_avd: list[float]
    per_subject_mcc: list[float]
    subject_indices: list[int]
    subject_ids: list[str]


@dataclass
class FoldResult:
    """Aggregated results for a single outer fold (across restarts).

    In the paper's protocol, we train 10 restarts per outer fold on FULL
    outer-train data. Each restart evaluates on outer-test immediately.
    """

    outer_fold: int
    runs: list[RunResult]

    @property
    def mean_dice(self) -> float:
        """Mean TEST DICE across restarts."""
        return sum(r.test_dice for r in self.runs) / len(self.runs)

    @property
    def std_dice(self) -> float:
        """Standard deviation of TEST DICE across restarts."""
        return float(np.std([r.test_dice for r in self.runs]))

    @property
    def best_dice(self) -> float:
        """Best TEST DICE among restarts."""
        return max(r.test_dice for r in self.runs)

    @property
    def best_run(self) -> RunResult:
        """Run with best TEST DICE.

        Note: This uses optimistic selection (picking best restart).
        For paper parity, prefer using get_aggregated_scores() with 'mean'.
        """
        return max(self.runs, key=lambda r: r.test_dice)

    def get_aggregated_scores(self, mode: str) -> dict[int, dict[str, float]]:
        """Get per-subject scores aggregated across restarts.

        Args:
            mode: Aggregation mode ("mean", "median", "best").

        Returns:
            Dict mapping subject index to metrics dict.
        """
        if mode == "best":
            return self.get_all_per_subject_scores()

        # Group scores by subject index
        # Assumes all restarts in this fold have same subject indices (same test split)
        scores_by_subject: dict[int, dict[str, list[float]]] = {}

        for run in self.runs:
            for i, idx in enumerate(run.subject_indices):
                if idx not in scores_by_subject:
                    scores_by_subject[idx] = {"dice": [], "avd": [], "mcc": []}

                scores_by_subject[idx]["dice"].append(run.per_subject_dice[i])
                scores_by_subject[idx]["avd"].append(run.per_subject_avd[i])
                scores_by_subject[idx]["mcc"].append(run.per_subject_mcc[i])

        # Aggregate
        final_scores: dict[int, dict[str, float]] = {}
        for idx, metrics in scores_by_subject.items():
            final_scores[idx] = {}
            for metric, values in metrics.items():
                if mode == "mean":
                    final_scores[idx][metric] = float(np.mean(values))
                elif mode == "median":
                    final_scores[idx][metric] = float(np.median(values))
                else:
                    raise ValueError(f"Unknown aggregation mode: {mode}")

        return final_scores

    def get_all_per_subject_scores(self) -> dict[int, dict[str, float]]:
        """Get per-subject scores from best restart for aggregation."""
        best = self.best_run
        scores: dict[int, dict[str, float]] = {}
        for i, idx in enumerate(best.subject_indices):
            scores[idx] = {
                "dice": best.per_subject_dice[i],
                "avd": best.per_subject_avd[i],
                "mcc": best.per_subject_mcc[i],
            }
        return scores


@dataclass
class ExperimentResult:
    """Complete experiment results.

    In the paper's protocol, each run trains on FULL outer-train and evaluates
    on outer-test immediately. Results are stored in FoldResult objects.
    """

    config: dict
    folds: list[FoldResult]
    start_time: str
    end_time: str
    total_duration_hours: float

    # IMPORTANT: Paper likely reports TEST metrics computed from POOLED PER-SUBJECT scores
    # across all outer folds (n≈224), rather than fold-level means (n=3).
    # However, per-run aggregation (n=30) is also valuable for analysis.
    #
    # Evidence from paper:
    # - Figure 1: Boxplot showing per-subject distribution (median, IQR, outliers)
    # - Figure 2: "median DICE score with interquartile range (IQR) error bars"
    # - Table 1: Wilcoxon signed-rank test requires paired per-subject data
    # - Std values (0.005-0.02) consistent with per-subject variation, not fold variation
    #
    # Metrics are pooled per-subject across folds for paper-parity statistics.

    def _get_pooled_scores(self, metric: str) -> list[float]:
        """Get pooled per-subject scores across all outer folds.

        Uses aggregated scores across restarts (mean/median/best) based on config.
        Pools all ~224 subjects for paper-parity statistics.

        Args:
            metric: One of 'dice', 'avd', 'mcc'.

        Returns:
            List of all per-subject scores pooled from outer test folds.
        """
        mode = self.config.get("restart_aggregation", "mean")
        all_scores: list[float] = []

        for fold in self.folds:
            scores_map = fold.get_aggregated_scores(mode)
            # Extract values for the requested metric
            # Sort by subject ID for deterministic ordering (though pooling doesn't care)
            for idx in sorted(scores_map.keys()):
                all_scores.append(scores_map[idx][metric])

        return all_scores

    def _get_pooled_run_scores(self, metric: str) -> list[float]:
        """Get pooled run-level scores across all outer folds (n=30).

        Args:
            metric: One of 'dice', 'avd', 'mcc'.

        Returns:
            List of run scores.
        """
        all_scores: list[float] = []
        for fold in self.folds:
            for run in fold.runs:
                if metric == "dice":
                    all_scores.append(run.test_dice)
                elif metric == "avd":
                    all_scores.append(run.test_avd)
                elif metric == "mcc":
                    all_scores.append(run.test_mcc)
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
    def test_mean_dice_per_run(self) -> float:
        """Mean TEST DICE across all 30 runs."""
        scores = self._get_pooled_run_scores("dice")
        return float(np.mean(scores)) if scores else 0.0

    @property
    def test_std_dice_per_run(self) -> float:
        """Std of TEST DICE across all 30 runs."""
        scores = self._get_pooled_run_scores("dice")
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

        Uses aggregated scores across restarts (mean/median/best).

        Returns:
            Dict mapping subject index to metrics dict.
        """
        mode = self.config.get("restart_aggregation", "mean")
        scores_by_subject: dict[int, dict[str, float]] = {}

        for fold in self.folds:
            fold_scores = fold.get_aggregated_scores(mode)
            scores_by_subject.update(fold_scores)

        return scores_by_subject


class ExperimentRunner:
    """Run full nested CV experiment.

    Implements the paper's experimental protocol:
    1. For each outer fold, train on FULL outer-train data (no inner split)
    2. 10 restarts per outer fold for stability
    3. Evaluate on outer-test immediately after training
    4. Pool per-subject scores across all folds for final statistics

    FROM PAPER:
    "Hyperparameter optimization was conducted on the inner folds of the first
    outer fold. The optimized hyperparameters were then applied to train models
    on all outer folds."

    We skip HP search (using paper's final HPs) and go directly to final evaluation.
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
            "Starting experiment: %d outer folds x %d restarts = %d runs",
            self.config.num_outer_folds,
            self.config.num_restarts,
            self.config.total_runs,
        )

        # Run all training configurations
        # Paper protocol: Train on FULL outer-train, no inner fold loop
        for outer_fold in range(self.config.num_outer_folds):
            for restart in range(self.config.num_restarts):
                result = self._run_single(outer_fold, restart)
                self.results.append(result)

                # Log progress
                completed = len(self.results)
                logger.info(
                    "Progress: %d/%d (%.1f%%) - Fold %d Restart %d: TEST DICE=%.4f",
                    completed,
                    self.config.total_runs,
                    100 * completed / self.config.total_runs,
                    outer_fold,
                    restart,
                    result.test_dice,
                )

        # Aggregate results by fold
        folds = self._aggregate_by_fold()

        # Create final result
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds() / 3600

        experiment_result = ExperimentResult(
            config=asdict(self.config),
            folds=folds,
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
        restart: int,
    ) -> RunResult:
        """Run single training configuration.

        Paper protocol:
        1. Train on FULL outer-train data (no inner fold split)
        2. Train for fixed 50 epochs (no validation-based early stopping)
        3. Evaluate on outer-test immediately after training

        Args:
            outer_fold: Outer fold index (0-2).
            restart: Restart index (0-9).

        Returns:
            RunResult with test metrics.
        """
        from arc_meshchop.data import (
            ARCDataset,
            create_stratification_labels,
            get_lesion_quintile,
        )
        from arc_meshchop.data.splits import generate_outer_cv_splits
        from arc_meshchop.evaluation import SegmentationMetrics
        from arc_meshchop.models import MeshNet
        from arc_meshchop.training import Trainer, TrainingConfig
        from arc_meshchop.utils.device import get_device

        run_id = f"fold_{outer_fold}_restart_{restart}"
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

        from arc_meshchop.data.huggingface_loader import parse_dataset_info, validate_masks_present

        (
            image_paths_str,
            mask_paths_raw,
            lesion_volumes,
            acquisition_types,
            subject_ids,
        ) = parse_dataset_info(
            dataset_info,
            context="Experiment training/evaluation",
        )

        # Validate masks (experiment requires ground truth for all samples)
        mask_paths_str = validate_masks_present(
            mask_paths_raw, context="Experiment training/evaluation"
        )

        # Resolve paths
        data_dir = self.config.data_dir.resolve()
        image_paths = [cast(Path, resolve_dataset_path(data_dir, p)) for p in image_paths_str]
        mask_paths = [cast(Path, resolve_dataset_path(data_dir, p)) for p in mask_paths_str]

        # Generate OUTER-only splits (no inner folds needed for replication)
        quintiles = [get_lesion_quintile(v) for v in lesion_volumes]
        strat_labels = create_stratification_labels(quintiles, acquisition_types)

        splits = generate_outer_cv_splits(
            n_samples=len(image_paths),
            stratification_labels=strat_labels,
            num_folds=self.config.num_outer_folds,
            random_seed=42,  # Fixed for reproducible splits
        )

        split = splits.get_split(outer_fold)
        train_indices = split.train_indices
        test_indices = split.test_indices

        # Create FULL outer-train dataset (no validation holdout)
        # Paper: "Train on full outer-train for 50 fixed epochs"
        train_dataset = ARCDataset(
            image_paths=[image_paths[i] for i in train_indices],
            mask_paths=[mask_paths[i] for i in train_indices],
            cache_dir=self.config.data_dir / "cache" / f"outer_{outer_fold}" / "train",
        )

        # No validation dataset - we use fixed epochs, no early stopping
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
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

        # Train (no validation loader - fixed epochs, no early stopping)
        trainer = Trainer(model, training_config, device=device)
        train_results = trainer.train(train_loader, val_loader=None)

        # Immediately evaluate on outer-test
        test_dataset = ARCDataset(
            image_paths=[image_paths[i] for i in test_indices],
            mask_paths=[mask_paths[i] for i in test_indices],
            cache_dir=self.config.data_dir / "cache" / f"outer_{outer_fold}" / "test",
        )

        metrics_calculator = SegmentationMetrics()
        dice_scores: list[float] = []
        avd_scores: list[float] = []
        mcc_scores: list[float] = []

        model.eval()
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

        duration = time.time() - start_time

        # Create result with per-subject scores
        run_result = RunResult(
            outer_fold=outer_fold,
            restart=restart,
            seed=seed,
            test_dice=float(np.mean(dice_scores)) if dice_scores else 0.0,
            test_avd=float(np.mean(avd_scores)) if avd_scores else 0.0,
            test_mcc=float(np.mean(mcc_scores)) if mcc_scores else 0.0,
            final_train_loss=float(train_results["final_train_loss"]),
            checkpoint_path=str(run_dir / "final.pt"),
            duration_seconds=duration,
            per_subject_dice=dice_scores,
            per_subject_avd=avd_scores,
            per_subject_mcc=mcc_scores,
            subject_indices=list(test_indices),
            subject_ids=[subject_ids[i] for i in test_indices],
        )

        # Save individual result
        results_file.write_text(json.dumps(asdict(run_result), indent=2))

        return run_result

    def _aggregate_by_fold(self) -> list[FoldResult]:
        """Aggregate results by outer fold.

        Returns:
            List of FoldResult objects (one per outer fold).
        """
        folds = []

        for outer in range(self.config.num_outer_folds):
            fold_runs = [r for r in self.results if r.outer_fold == outer]
            # Fail fast if no runs collected (prevents div-by-zero in FoldResult)
            if not fold_runs:
                raise RuntimeError(
                    f"No runs collected for outer fold {outer}. "
                    f"Check num_restarts ({self.config.num_restarts}) and run execution."
                )
            folds.append(
                FoldResult(
                    outer_fold=outer,
                    runs=fold_runs,
                )
            )

        return folds

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
                    "mean_dice": f.mean_dice,  # Mean across restarts
                    "std_dice": f.std_dice,  # Std across restarts
                    "best_dice": f.best_dice,  # Best restart
                    "num_restarts": len(f.runs),
                    "runs": [asdict(r) for r in f.runs],
                }
                for f in result.folds
            ],
            "summary": {
                # PRIMARY: Test metrics computed from POOLED per-subject scores (n≈224)
                # Pooled per-subject scores match paper protocol (see docs/REPRODUCIBILITY.md)
                "test_mean_dice": result.test_mean_dice,
                "test_std_dice": result.test_std_dice,
                "test_median_dice": result.test_median_dice,  # For Figure 2
                "test_iqr_dice": result.test_iqr_dice,  # For Figure 2 error bars
                # SECONDARY: Test metrics computed from per-run scores (n=30)
                "test_mean_dice_per_run": result.test_mean_dice_per_run,
                "test_std_dice_per_run": result.test_std_dice_per_run,
                "test_mean_avd": result.test_mean_avd,
                "test_std_avd": result.test_std_avd,
                "test_mean_mcc": result.test_mean_mcc,
                "test_std_mcc": result.test_std_mcc,
                # Total subjects in pooled test set
                "num_test_subjects": len(result._get_pooled_scores("dice")),
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
