"""Tests for loss functions."""

import pytest
import torch

from arc_meshchop.training.loss import WeightedCrossEntropyLoss, create_loss_function


class TestWeightedCrossEntropyLoss:
    """Tests for WeightedCrossEntropyLoss."""

    def test_output_is_scalar(self) -> None:
        """Verify loss output is a scalar tensor."""
        loss_fn = create_loss_function()

        logits = torch.randn(1, 2, 8, 8, 8)
        targets = torch.randint(0, 2, (1, 8, 8, 8))

        loss = loss_fn(logits, targets)

        assert loss.dim() == 0  # Scalar
        assert loss.dtype == torch.float32

    def test_loss_is_positive(self) -> None:
        """Verify loss is positive."""
        loss_fn = create_loss_function()

        logits = torch.randn(1, 2, 8, 8, 8)
        targets = torch.randint(0, 2, (1, 8, 8, 8))

        loss = loss_fn(logits, targets)

        assert loss.item() > 0

    def test_perfect_prediction_has_low_loss(self) -> None:
        """Verify perfect prediction has low loss."""
        loss_fn = create_loss_function()

        # Create "perfect" predictions
        targets = torch.zeros(1, 8, 8, 8, dtype=torch.long)
        targets[0, 3:5, 3:5, 3:5] = 1  # Small lesion

        logits = torch.zeros(1, 2, 8, 8, 8)
        logits[0, 0] = 10.0  # High confidence background
        logits[0, 0, 3:5, 3:5, 3:5] = -10.0
        logits[0, 1, 3:5, 3:5, 3:5] = 10.0  # High confidence lesion

        loss = loss_fn(logits, targets)

        assert loss.item() < 0.1  # Should be very low

    def test_class_weights_from_paper(self) -> None:
        """Verify default class weights match paper."""
        loss_fn = create_loss_function()

        assert loss_fn.class_weights[0].item() == pytest.approx(0.5)  # Background
        assert loss_fn.class_weights[1].item() == pytest.approx(1.0)  # Lesion

    def test_label_smoothing_from_paper(self) -> None:
        """Verify default label smoothing matches paper."""
        loss_fn = create_loss_function()

        assert loss_fn.label_smoothing == pytest.approx(0.01)

    def test_custom_weights(self) -> None:
        """Verify custom weights are applied."""
        loss_fn = WeightedCrossEntropyLoss(
            background_weight=0.3,
            lesion_weight=0.7,
            label_smoothing=0.05,
        )

        assert loss_fn.class_weights[0].item() == pytest.approx(0.3)
        assert loss_fn.class_weights[1].item() == pytest.approx(0.7)
        assert loss_fn.label_smoothing == pytest.approx(0.05)

    def test_loss_is_differentiable(self) -> None:
        """Verify loss is differentiable for backprop."""
        loss_fn = create_loss_function()

        logits = torch.randn(1, 2, 8, 8, 8, requires_grad=True)
        targets = torch.randint(0, 2, (1, 8, 8, 8))

        loss = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_loss_handles_batch(self) -> None:
        """Verify loss handles batch dimension."""
        loss_fn = create_loss_function()

        # Batch size of 4
        logits = torch.randn(4, 2, 8, 8, 8)
        targets = torch.randint(0, 2, (4, 8, 8, 8))

        loss = loss_fn(logits, targets)

        assert loss.dim() == 0  # Still scalar
        assert loss.item() > 0


class TestCreateLossFunction:
    """Tests for create_loss_function factory."""

    def test_returns_weighted_ce_loss(self) -> None:
        """Verify factory returns correct type."""
        loss_fn = create_loss_function()

        assert isinstance(loss_fn, WeightedCrossEntropyLoss)

    def test_default_values_match_paper(self) -> None:
        """Verify factory defaults match paper."""
        loss_fn = create_loss_function()

        # FROM PAPER: weights=[0.5, 1.0], label_smoothing=0.01
        assert loss_fn.class_weights[0].item() == pytest.approx(0.5)
        assert loss_fn.class_weights[1].item() == pytest.approx(1.0)
        assert loss_fn.label_smoothing == pytest.approx(0.01)
