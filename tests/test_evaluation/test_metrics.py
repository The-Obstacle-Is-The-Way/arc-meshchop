"""Tests for evaluation metrics."""

import pytest
import torch

from arc_meshchop.evaluation.metrics import (
    MetricResult,
    SegmentationMetrics,
    average_volume_difference,
    compute_confusion_matrix,
    dice_coefficient,
    matthews_correlation_coefficient,
)


class TestDiceCoefficient:
    """Tests for DICE coefficient."""

    def test_perfect_overlap(self) -> None:
        """Verify DICE = 1 for perfect overlap."""
        pred = torch.ones(10, 10, 10)
        target = torch.ones(10, 10, 10)

        dice = dice_coefficient(pred, target)
        assert dice == pytest.approx(1.0, abs=1e-5)

    def test_no_overlap(self) -> None:
        """Verify DICE ~ 0 for no overlap."""
        pred = torch.zeros(10, 10, 10)
        pred[:5, :, :] = 1
        target = torch.zeros(10, 10, 10)
        target[5:, :, :] = 1

        dice = dice_coefficient(pred, target)
        assert dice < 0.01  # Should be very small (smoothing prevents exact 0)

    def test_partial_overlap(self) -> None:
        """Verify DICE is in (0, 1) for partial overlap."""
        pred = torch.zeros(10, 10, 10)
        pred[2:8, 2:8, 2:8] = 1  # 6x6x6 = 216 voxels

        target = torch.zeros(10, 10, 10)
        target[4:10, 4:10, 4:10] = 1  # 6x6x6 = 216 voxels

        # Overlap: 4x4x4 = 64 voxels
        # DICE = 2*64 / (216 + 216) = 128/432 ~ 0.296

        dice = dice_coefficient(pred, target)
        assert 0 < dice < 1

    def test_empty_masks(self) -> None:
        """Verify DICE ~ 1 for both empty (no lesion case)."""
        pred = torch.zeros(10, 10, 10)
        target = torch.zeros(10, 10, 10)

        dice = dice_coefficient(pred, target)
        # With smoothing, empty masks give ~1.0
        assert dice > 0.99

    def test_range(self) -> None:
        """Verify DICE is always in [0, 1]."""
        for _ in range(10):
            pred = (torch.rand(8, 8, 8) > 0.5).float()
            target = (torch.rand(8, 8, 8) > 0.5).float()

            dice = dice_coefficient(pred, target)
            assert 0 <= dice <= 1

    def test_symmetric(self) -> None:
        """Verify DICE is symmetric (pred, target) == (target, pred)."""
        pred = (torch.rand(8, 8, 8) > 0.5).float()
        target = (torch.rand(8, 8, 8) > 0.5).float()

        dice1 = dice_coefficient(pred, target)
        dice2 = dice_coefficient(target, pred)

        assert dice1 == pytest.approx(dice2)


class TestAverageVolumeDifference:
    """Tests for AVD."""

    def test_perfect_volume(self) -> None:
        """Verify AVD = 0 for identical volumes."""
        pred = torch.zeros(10, 10, 10)
        pred[3:7, 3:7, 3:7] = 1

        target = torch.zeros(10, 10, 10)
        target[3:7, 3:7, 3:7] = 1

        avd = average_volume_difference(pred, target)
        assert avd == pytest.approx(0.0)

    def test_double_volume(self) -> None:
        """Verify AVD = 1 for double volume."""
        pred = torch.ones(10, 10, 10)  # 1000 voxels

        target = torch.zeros(10, 10, 10)
        target[:5, :, :] = 1  # 500 voxels

        avd = average_volume_difference(pred, target)
        assert avd == pytest.approx(1.0)  # |1000-500|/500 = 1.0

    def test_half_volume(self) -> None:
        """Verify AVD = 0.5 for half volume."""
        pred = torch.zeros(10, 10, 10)
        pred[:5, :, :] = 1  # 500 voxels

        target = torch.ones(10, 10, 10)  # 1000 voxels

        avd = average_volume_difference(pred, target)
        assert avd == pytest.approx(0.5)  # |500-1000|/1000 = 0.5

    def test_empty_target(self) -> None:
        """Verify AVD = inf for empty target with non-empty pred."""
        pred = torch.ones(10, 10, 10)
        target = torch.zeros(10, 10, 10)

        avd = average_volume_difference(pred, target)
        assert avd == float("inf")

    def test_both_empty(self) -> None:
        """Verify AVD = 0 for both empty masks."""
        pred = torch.zeros(10, 10, 10)
        target = torch.zeros(10, 10, 10)

        avd = average_volume_difference(pred, target)
        assert avd == 0.0

    def test_non_negative(self) -> None:
        """Verify AVD is always non-negative."""
        for _ in range(10):
            pred = (torch.rand(8, 8, 8) > 0.3).float()
            target = (torch.rand(8, 8, 8) > 0.3).float()

            avd = average_volume_difference(pred, target)
            assert avd >= 0


class TestMatthewsCorrelationCoefficient:
    """Tests for MCC."""

    def test_perfect_prediction(self) -> None:
        """Verify MCC = 1 for perfect prediction."""
        pred = torch.zeros(10, 10, 10)
        pred[3:7, 3:7, 3:7] = 1

        target = pred.clone()

        mcc = matthews_correlation_coefficient(pred, target)
        assert mcc == pytest.approx(1.0)

    def test_inverse_prediction(self) -> None:
        """Verify MCC = -1 for inverse prediction."""
        target = torch.zeros(10, 10, 10)
        target[3:7, 3:7, 3:7] = 1

        pred = 1 - target  # Inverted

        mcc = matthews_correlation_coefficient(pred, target)
        assert mcc == pytest.approx(-1.0)

    def test_random_prediction(self) -> None:
        """Verify MCC ~ 0 for random prediction."""
        torch.manual_seed(42)
        pred = (torch.rand(100, 100, 100) > 0.5).float()
        target = (torch.rand(100, 100, 100) > 0.5).float()

        mcc = matthews_correlation_coefficient(pred, target)
        assert -0.1 < mcc < 0.1  # Should be close to 0

    def test_range(self) -> None:
        """Verify MCC is always in [-1, 1]."""
        for _ in range(10):
            pred = (torch.rand(8, 8, 8) > 0.5).float()
            target = (torch.rand(8, 8, 8) > 0.5).float()

            mcc = matthews_correlation_coefficient(pred, target)
            assert -1 <= mcc <= 1


class TestComputeConfusionMatrix:
    """Tests for confusion matrix computation."""

    def test_all_true_positives(self) -> None:
        """Verify confusion matrix for all TP case."""
        pred = torch.ones(10, 10, 10)
        target = torch.ones(10, 10, 10)

        cm = compute_confusion_matrix(pred, target)

        assert cm["tp"] == 1000
        assert cm["tn"] == 0
        assert cm["fp"] == 0
        assert cm["fn"] == 0

    def test_all_true_negatives(self) -> None:
        """Verify confusion matrix for all TN case."""
        pred = torch.zeros(10, 10, 10)
        target = torch.zeros(10, 10, 10)

        cm = compute_confusion_matrix(pred, target)

        assert cm["tp"] == 0
        assert cm["tn"] == 1000
        assert cm["fp"] == 0
        assert cm["fn"] == 0

    def test_mixed_case(self) -> None:
        """Verify confusion matrix sums to total voxels."""
        pred = (torch.rand(8, 8, 8) > 0.5).float()
        target = (torch.rand(8, 8, 8) > 0.5).float()

        cm = compute_confusion_matrix(pred, target)

        total = cm["tp"] + cm["tn"] + cm["fp"] + cm["fn"]
        assert total == 8 * 8 * 8  # 512 voxels


class TestMetricResult:
    """Tests for MetricResult dataclass."""

    def test_str_format(self) -> None:
        """Verify string format is 'mean (std)'."""
        result = MetricResult(
            name="DICE",
            mean=0.876,
            std=0.016,
            values=[0.85, 0.87, 0.89],
        )

        assert str(result) == "0.876 (0.016)"

    def test_values_stored(self) -> None:
        """Verify individual values are stored."""
        values = [0.85, 0.87, 0.89]
        result = MetricResult(
            name="DICE",
            mean=0.87,
            std=0.02,
            values=values,
        )

        assert result.values == values


class TestSegmentationMetrics:
    """Tests for SegmentationMetrics class."""

    def test_compute_single(self) -> None:
        """Test computing metrics for single sample."""
        metrics = SegmentationMetrics()

        pred = (torch.rand(8, 8, 8) > 0.5).float()
        target = (torch.rand(8, 8, 8) > 0.5).float()

        result = metrics.compute_single(pred, target)

        assert "dice" in result
        assert "avd" in result
        assert "mcc" in result

    def test_compute_single_with_batch_dim(self) -> None:
        """Test computing metrics handles batch dimension."""
        metrics = SegmentationMetrics()

        pred = (torch.rand(1, 8, 8, 8) > 0.5).float()
        target = (torch.rand(1, 8, 8, 8) > 0.5).float()

        result = metrics.compute_single(pred, target)

        assert "dice" in result
        assert "avd" in result
        assert "mcc" in result

    def test_compute_batch(self) -> None:
        """Test computing metrics for batch."""
        metrics = SegmentationMetrics()

        preds = (torch.rand(4, 8, 8, 8) > 0.5).float()
        targets = (torch.rand(4, 8, 8, 8) > 0.5).float()

        result = metrics.compute_batch(preds, targets)

        assert "dice" in result
        assert "avd" in result
        assert "mcc" in result

    def test_compute_with_stats(self) -> None:
        """Test computing metrics with statistics."""
        metrics = SegmentationMetrics()

        preds = (torch.rand(4, 8, 8, 8) > 0.5).float()
        targets = (torch.rand(4, 8, 8, 8) > 0.5).float()

        result = metrics.compute_with_stats(preds, targets)

        assert result["dice"].mean is not None
        assert result["dice"].std is not None
        assert len(result["dice"].values) == 4

    def test_compute_with_stats_values_count(self) -> None:
        """Verify per-sample values match batch size."""
        metrics = SegmentationMetrics()

        batch_size = 7
        preds = (torch.rand(batch_size, 8, 8, 8) > 0.5).float()
        targets = (torch.rand(batch_size, 8, 8, 8) > 0.5).float()

        result = metrics.compute_with_stats(preds, targets)

        assert len(result["dice"].values) == batch_size
        assert len(result["avd"].values) == batch_size
        assert len(result["mcc"].values) == batch_size
