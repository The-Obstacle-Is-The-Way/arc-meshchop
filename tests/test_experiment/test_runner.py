"""Tests for experiment runner."""

from __future__ import annotations

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

    def test_div_factor_default(self) -> None:
        """Verify div_factor defaults to 100 (paper requirement)."""
        from arc_meshchop.experiment.config import ExperimentConfig

        config = ExperimentConfig()

        assert config.div_factor == 100.0


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

    def test_std_dice(self) -> None:
        """Verify std DICE calculation."""
        import numpy as np

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

        expected_std = np.std([0.8, 0.81, 0.82, 0.83, 0.84])
        assert fold.std_dice == pytest.approx(expected_std)


class TestExperimentResult:
    """Tests for ExperimentResult aggregation."""

    def test_test_mean_dice_legacy(self) -> None:
        """Verify backward compat: mean from fold-level means when no per-subject data."""
        from arc_meshchop.experiment.runner import ExperimentResult

        result = ExperimentResult(
            config={},
            folds=[],
            test_results=[
                {"test_dice": 0.85, "test_avd": 0.25, "test_mcc": 0.75},
                {"test_dice": 0.87, "test_avd": 0.24, "test_mcc": 0.76},
                {"test_dice": 0.86, "test_avd": 0.26, "test_mcc": 0.74},
            ],
            start_time="2024-01-01T00:00:00",
            end_time="2024-01-02T00:00:00",
            total_duration_hours=24.0,
        )

        # Mean of [0.85, 0.87, 0.86] = 0.86
        assert result.test_mean_dice == pytest.approx(0.86)

    def test_returns_zero_with_no_test_results(self) -> None:
        """Verify zero returned when no test results."""
        from arc_meshchop.experiment.runner import ExperimentResult

        result = ExperimentResult(
            config={},
            folds=[],
            test_results=[],
            start_time="2024-01-01T00:00:00",
            end_time="2024-01-02T00:00:00",
            total_duration_hours=24.0,
        )

        assert result.test_mean_dice == 0.0
        assert result.test_std_dice == 0.0

    def test_pooled_per_subject_scores(self) -> None:
        """Verify stats computed from pooled per-subject scores (BUG-002 fix).

        Paper reports mean±std computed from ALL per-subject test scores pooled
        across outer folds (n≈224), not from fold-level means (n=3).
        """
        import numpy as np

        from arc_meshchop.experiment.runner import ExperimentResult

        # Simulate 3 outer folds with per-subject scores
        result = ExperimentResult(
            config={},
            folds=[],
            test_results=[
                {
                    "outer_fold": 0,
                    "test_dice": 0.85,  # Fold-level mean (for backward compat)
                    "test_avd": 0.25,
                    "test_mcc": 0.75,
                    "per_subject_dice": [0.80, 0.85, 0.90],  # 3 subjects
                    "per_subject_avd": [0.20, 0.25, 0.30],
                    "per_subject_mcc": [0.70, 0.75, 0.80],
                    "subject_indices": [0, 1, 2],
                },
                {
                    "outer_fold": 1,
                    "test_dice": 0.87,
                    "test_avd": 0.24,
                    "test_mcc": 0.76,
                    "per_subject_dice": [0.82, 0.87, 0.92],  # 3 subjects
                    "per_subject_avd": [0.19, 0.24, 0.29],
                    "per_subject_mcc": [0.71, 0.76, 0.81],
                    "subject_indices": [3, 4, 5],
                },
                {
                    "outer_fold": 2,
                    "test_dice": 0.86,
                    "test_avd": 0.26,
                    "test_mcc": 0.74,
                    "per_subject_dice": [0.81, 0.86, 0.91],  # 3 subjects
                    "per_subject_avd": [0.21, 0.26, 0.31],
                    "per_subject_mcc": [0.69, 0.74, 0.79],
                    "subject_indices": [6, 7, 8],
                },
            ],
            start_time="2024-01-01T00:00:00",
            end_time="2024-01-02T00:00:00",
            total_duration_hours=24.0,
        )

        # All 9 per-subject DICE scores pooled
        all_dice = [0.80, 0.85, 0.90, 0.82, 0.87, 0.92, 0.81, 0.86, 0.91]

        # Mean/std should be from pooled scores (n=9), NOT fold means (n=3)
        assert result.test_mean_dice == pytest.approx(np.mean(all_dice))
        assert result.test_std_dice == pytest.approx(np.std(all_dice))

        # Median and IQR for Figure 2 reproduction
        assert result.test_median_dice == pytest.approx(np.median(all_dice))
        q25, q75 = result.test_iqr_dice
        assert q25 == pytest.approx(np.percentile(all_dice, 25))
        assert q75 == pytest.approx(np.percentile(all_dice, 75))

    def test_per_subject_score_retrieval_for_wilcoxon(self) -> None:
        """Verify per-subject scores can be retrieved for Wilcoxon test."""
        from arc_meshchop.experiment.runner import ExperimentResult

        result = ExperimentResult(
            config={},
            folds=[],
            test_results=[
                {
                    "outer_fold": 0,
                    "test_dice": 0.85,
                    "per_subject_dice": [0.80, 0.85, 0.90],
                    "per_subject_avd": [0.20, 0.25, 0.30],
                    "per_subject_mcc": [0.70, 0.75, 0.80],
                    "subject_indices": [0, 1, 2],
                },
                {
                    "outer_fold": 1,
                    "test_dice": 0.87,
                    "per_subject_dice": [0.82, 0.87],
                    "per_subject_avd": [0.19, 0.24],
                    "per_subject_mcc": [0.71, 0.76],
                    "subject_indices": [3, 4],
                },
            ],
            start_time="2024-01-01T00:00:00",
            end_time="2024-01-02T00:00:00",
            total_duration_hours=24.0,
        )

        scores = result.get_per_subject_scores()

        # Should have 5 subjects indexed by their IDs
        assert len(scores) == 5
        assert scores[0]["dice"] == pytest.approx(0.80)
        assert scores[2]["dice"] == pytest.approx(0.90)
        assert scores[4]["dice"] == pytest.approx(0.87)

    def test_std_computed_from_n_subjects_not_n_folds(self) -> None:
        """Verify std reflects per-subject variation, not fold variation (BUG-002).

        This is the critical test: std from n≈224 subjects should be ~0.01-0.02
        (matching paper Table 1), not the much larger fold-to-fold variation.
        """
        import numpy as np

        from arc_meshchop.experiment.runner import ExperimentResult

        # Simulate realistic data: 3 folds with ~75 subjects each
        # Per-subject DICE typically varies ~0.6-1.0 with std ~0.15
        np.random.seed(42)
        fold1_scores = np.clip(np.random.normal(0.87, 0.15, 75), 0, 1).tolist()
        fold2_scores = np.clip(np.random.normal(0.87, 0.15, 75), 0, 1).tolist()
        fold3_scores = np.clip(np.random.normal(0.87, 0.15, 74), 0, 1).tolist()

        result = ExperimentResult(
            config={},
            folds=[],
            test_results=[
                {
                    "outer_fold": 0,
                    "test_dice": float(np.mean(fold1_scores)),
                    "per_subject_dice": fold1_scores,
                    "subject_indices": list(range(75)),
                },
                {
                    "outer_fold": 1,
                    "test_dice": float(np.mean(fold2_scores)),
                    "per_subject_dice": fold2_scores,
                    "subject_indices": list(range(75, 150)),
                },
                {
                    "outer_fold": 2,
                    "test_dice": float(np.mean(fold3_scores)),
                    "per_subject_dice": fold3_scores,
                    "subject_indices": list(range(150, 224)),
                },
            ],
            start_time="2024-01-01T00:00:00",
            end_time="2024-01-02T00:00:00",
            total_duration_hours=24.0,
        )

        all_scores = fold1_scores + fold2_scores + fold3_scores

        # Std should be from all 224 subjects
        assert result.test_std_dice == pytest.approx(np.std(all_scores), rel=0.01)

        # Std should be ~0.15 (per-subject variation), NOT ~0.01 (fold variation)
        # This is the key insight: paper reports per-subject std
        assert result.test_std_dice > 0.1  # Per-subject variation is large
        assert len(result._get_pooled_scores("dice")) == 224


class TestRunResult:
    """Tests for RunResult dataclass."""

    def test_serialization(self) -> None:
        """Verify RunResult can be serialized to dict."""
        from dataclasses import asdict

        from arc_meshchop.experiment.runner import RunResult

        run = RunResult(
            outer_fold=0,
            inner_fold=1,
            restart=5,
            seed=47,
            best_dice=0.876,
            best_epoch=42,
            final_train_loss=0.05,
            checkpoint_path="/path/to/checkpoint.pt",
            duration_seconds=3600.0,
        )

        d = asdict(run)

        assert d["outer_fold"] == 0
        assert d["inner_fold"] == 1
        assert d["restart"] == 5
        assert d["best_dice"] == 0.876
