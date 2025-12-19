"""Tests for BUG-011: HPO Pruning Metric Reset.

Ensures that HPO trials report aggregated metrics per epoch across folds,
rather than resetting the metric curve at each fold boundary.
"""

import types
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from arc_meshchop.training.hpo import run_hpo_trial


@pytest.fixture
def mock_dependencies() -> Generator[dict[str, Any], None, None]:
    """Mock external dependencies for run_hpo_trial."""
    # Stub huggingface_loader locally to avoid global pollution
    stub_hf = types.ModuleType("arc_meshchop.data.huggingface_loader")
    for name in [
        "ARCDatasetInfo",
        "ARCSample",
        "download_arc_to_local",
        "load_arc_from_huggingface",
        "load_arc_samples",
        "verify_sample_counts",
        "parse_dataset_info",
        "validate_masks_present",
    ]:
        setattr(stub_hf, name, MagicMock())

    with (
        patch.dict("sys.modules", {"arc_meshchop.data.huggingface_loader": stub_hf}),
        patch("arc_meshchop.training.Trainer") as MockTrainer,
        patch("arc_meshchop.data.ARCDataset"),
        patch("arc_meshchop.data.create_dataloaders") as mock_loaders,
        patch("arc_meshchop.utils.paths.resolve_dataset_path") as mock_resolve,
        patch("arc_meshchop.models.MeshNet"),
        patch("arc_meshchop.training.hpo.json.load"),
        patch("pathlib.Path.open"),
        patch("pathlib.Path.resolve"),
        patch("arc_meshchop.data.generate_nested_cv_splits") as mock_splits,
        patch("arc_meshchop.utils.seeding.seed_everything"),
    ):
        # Setup mock dataset info
        stub_hf.parse_dataset_info.return_value = (
            ["img1", "img2"],  # image paths
            ["mask1", "mask2"],  # mask paths
            [100, 200],  # volumes
            ["t1", "t2"],  # acquisition types
            ["sub1", "sub2"],  # subject ids
        )
        stub_hf.validate_masks_present.return_value = ["mask1", "mask2"]
        mock_resolve.side_effect = lambda d, p: p

        # Setup mock loaders
        mock_loader = MagicMock()
        mock_loaders.return_value = (mock_loader, mock_loader)

        # Setup mock splits
        mock_split = MagicMock()
        mock_split.train_indices = [0]
        mock_split.val_indices = [1]

        mock_splits_obj = MagicMock()
        mock_splits_obj.get_split.return_value = mock_split
        mock_splits.return_value = mock_splits_obj

        # Setup mock trainer
        trainer_instance = MockTrainer.return_value

        # Simulate validation scores (3 folds per epoch):
        # epoch 0: 0.6, 0.5, 0.4 -> mean 0.5
        # epoch 1: 0.3, 0.4, 0.5 -> mean 0.4
        trainer_instance.validate.side_effect = [
            {"dice": 0.6},
            {"dice": 0.5},
            {"dice": 0.4},  # fold 0, 1, 2 epoch 0
            {"dice": 0.3},
            {"dice": 0.4},
            {"dice": 0.5},  # fold 0, 1, 2 epoch 1
        ]

        yield {
            "trainer": MockTrainer,
            "loaders": mock_loaders,
            "splits": mock_splits,
        }


def test_hpo_reporting_per_epoch_mean(mock_dependencies: dict[str, Any]) -> None:
    """Verify that HPO reporting is aggregated per epoch across folds."""
    mock_trial = MagicMock()
    mock_trial.suggest_float.return_value = 0.001
    mock_trial.suggest_categorical.return_value = 0.1
    mock_trial.number = 0
    mock_trial.should_prune.return_value = False

    # Run for 2 epochs
    mean_dice = run_hpo_trial(
        trial=mock_trial,
        data_dir="dummy/path",
        epochs=2,
    )

    # Check trial.report calls
    # Should be called once per epoch (total 2 calls)
    assert mock_trial.report.call_count == 2

    # Call args: (value, step)
    calls = mock_trial.report.call_args_list

    # Epoch 0: mean of [0.6, 0.5, 0.4] = 0.5
    assert calls[0][0][0] == pytest.approx(0.5)
    assert calls[0][0][1] == 0  # step 0

    # Epoch 1: mean of [0.3, 0.4, 0.5] = 0.4
    assert calls[1][0][0] == pytest.approx(0.4)
    assert calls[1][0][1] == 1  # step 1

    # Final result
    expected_best_mean = (0.6 + 0.5 + 0.5) / 3
    assert mean_dice == pytest.approx(expected_best_mean)
