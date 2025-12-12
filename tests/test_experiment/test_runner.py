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
        from arc_meshchop.experiment.runner import FoldResult, RunResult
        import numpy as np

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

    def test_test_mean_dice(self) -> None:
        """Verify test mean DICE from outer folds."""
        from arc_meshchop.experiment.runner import ExperimentResult, FoldResult

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
