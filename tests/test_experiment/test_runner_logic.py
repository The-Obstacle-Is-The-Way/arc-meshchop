"""Tests for experiment runner logic (mocked)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from arc_meshchop.experiment.config import ExperimentConfig
from arc_meshchop.experiment.runner import ExperimentRunner

if TYPE_CHECKING:
    from pathlib import Path


class TestExperimentRunnerLogic:
    """Tests for ExperimentRunner logic."""

    @pytest.fixture
    def config(self, tmp_path: Path) -> ExperimentConfig:
        """Create experiment config."""
        return ExperimentConfig(
            output_dir=tmp_path / "experiments",
            data_dir=tmp_path / "data",
        )

    @pytest.fixture
    def runner(self, config: ExperimentConfig) -> ExperimentRunner:
        """Create experiment runner."""
        return ExperimentRunner(config)

    def test_run_single_does_not_overwrite_checkpoint(self, runner: ExperimentRunner) -> None:
        """Bug 2: Verify runner does not overwrite Trainer's final.pt."""
        with (
            patch("arc_meshchop.data.ARCDataset"),
            patch("arc_meshchop.data.create_stratification_labels"),
            patch("arc_meshchop.data.get_lesion_quintile"),
            patch("arc_meshchop.data.splits.generate_outer_cv_splits"),
            patch("arc_meshchop.evaluation.SegmentationMetrics"),
            patch("arc_meshchop.models.MeshNet"),
            patch("arc_meshchop.training.Trainer") as MockTrainer,
            patch("arc_meshchop.utils.device.get_device"),
            patch("torch.utils.data.DataLoader"),
            patch("torch.save") as mock_save,
            patch.object(runner, "_load_dataset_info") as mock_info,
            patch("torch.cuda.is_available", return_value=False),
            patch("time.time", return_value=0),
        ):
            mock_info.return_value = {
                "image_paths": ["a"],
                "mask_paths": ["a"],
                "lesion_volumes": [1],
                "acquisition_types": ["t1"],
                "subject_ids": ["sub-0000"],
            }

            MockTrainer.return_value.train.return_value = {"final_train_loss": 0.1}

            # Run
            runner._run_single(0, 0)

            # Check torch.save calls
            # We assume Trainer handles saving (which is mocked here).
            # Runner should NOT save to final.pt

            # Get all calls to torch.save
            save_calls = mock_save.call_args_list
            for call in save_calls:
                args, _ = call
                path = args[1]
                # If path ends with final.pt, fail
                if str(path).endswith("final.pt"):
                    pytest.fail(f"Runner overwrote final.pt at {path}")

    def test_cache_shared_across_restarts(self, runner: ExperimentRunner) -> None:
        """Bug 3: Verify cache dir uses outer fold, not restart."""
        with (
            patch("arc_meshchop.data.ARCDataset") as MockDataset,
            patch("arc_meshchop.data.create_stratification_labels"),
            patch("arc_meshchop.data.get_lesion_quintile"),
            patch("arc_meshchop.data.splits.generate_outer_cv_splits"),
            patch("arc_meshchop.evaluation.SegmentationMetrics"),
            patch("arc_meshchop.models.MeshNet"),
            patch("arc_meshchop.training.Trainer"),
            patch("arc_meshchop.utils.device.get_device"),
            patch("torch.utils.data.DataLoader"),
            patch("torch.save"),
            patch.object(runner, "_load_dataset_info") as mock_info,
            patch("torch.cuda.is_available", return_value=False),
            patch("time.time", return_value=0),
        ):
            mock_info.return_value = {
                "image_paths": ["a"],
                "mask_paths": ["a"],
                "lesion_volumes": [1],
                "acquisition_types": ["t1"],
                "subject_ids": ["sub-0000"],
            }

            # Run Restart 0
            runner._run_single(0, 0)

            # Check first initialization of ARCDataset (train)
            # The first call is for train_dataset
            call_args = MockDataset.call_args_list[0]
            kwargs = call_args.kwargs
            cache_dir = kwargs.get("cache_dir")

            # Should contain "outer_0" but NOT "restart_0"
            assert "outer_0" in str(cache_dir)
            assert "restart_0" not in str(cache_dir)
