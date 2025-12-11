"""Tests for optimizer and scheduler."""

import pytest
import torch
import torch.nn as nn

from arc_meshchop.training.optimizer import create_optimizer, create_scheduler


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        result: torch.Tensor = self.linear(x)
        return result


class TestCreateOptimizer:
    """Tests for create_optimizer."""

    def test_creates_adamw(self) -> None:
        """Verify creates AdamW optimizer."""
        model = SimpleModel()
        optimizer = create_optimizer(model)

        assert isinstance(optimizer, torch.optim.AdamW)

    def test_default_params_from_paper(self) -> None:
        """Verify default params match paper."""
        model = SimpleModel()
        optimizer = create_optimizer(model)

        # Check defaults match paper (FROM PAPER Section 2)
        assert optimizer.defaults["lr"] == pytest.approx(0.001)
        assert optimizer.defaults["weight_decay"] == pytest.approx(3e-5)
        assert optimizer.defaults["eps"] == pytest.approx(1e-4)

    def test_custom_learning_rate(self) -> None:
        """Verify custom learning rate is applied."""
        model = SimpleModel()
        optimizer = create_optimizer(model, learning_rate=0.01)

        assert optimizer.defaults["lr"] == pytest.approx(0.01)

    def test_custom_weight_decay(self) -> None:
        """Verify custom weight decay is applied."""
        model = SimpleModel()
        optimizer = create_optimizer(model, weight_decay=1e-4)

        assert optimizer.defaults["weight_decay"] == pytest.approx(1e-4)

    def test_custom_eps(self) -> None:
        """Verify custom epsilon is applied."""
        model = SimpleModel()
        optimizer = create_optimizer(model, eps=1e-8)

        assert optimizer.defaults["eps"] == pytest.approx(1e-8)

    def test_optimizer_has_model_params(self) -> None:
        """Verify optimizer tracks model parameters."""
        model = SimpleModel()
        optimizer = create_optimizer(model)

        # Optimizer should have param groups
        assert len(optimizer.param_groups) > 0
        # Parameters should be from the model
        assert len(list(optimizer.param_groups[0]["params"])) == len(list(model.parameters()))


class TestCreateScheduler:
    """Tests for create_scheduler."""

    def test_creates_onecycle(self) -> None:
        """Verify creates OneCycleLR scheduler."""
        model = SimpleModel()
        optimizer = create_optimizer(model)
        # Use higher total_steps to avoid edge case with 1% warmup (needs at least 100 steps)
        scheduler = create_scheduler(optimizer, max_lr=0.001, total_steps=1000, pct_start=0.1)

        assert isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)

    def test_warmup_fraction(self) -> None:
        """Verify 1% warmup as in paper."""
        model = SimpleModel()
        optimizer = create_optimizer(model)
        scheduler = create_scheduler(
            optimizer,
            max_lr=0.001,
            total_steps=1000,
            pct_start=0.01,
        )

        # First LR should be ~1/25 of max (OneCycleLR default div_factor)
        initial_lr = scheduler.get_last_lr()[0]
        assert initial_lr < 0.001  # Less than max

    def test_lr_increases_then_decreases(self) -> None:
        """Verify LR follows OneCycle pattern."""
        model = SimpleModel()
        optimizer = create_optimizer(model)
        scheduler = create_scheduler(
            optimizer,
            max_lr=0.001,
            total_steps=100,
            pct_start=0.1,  # 10% warmup for easier testing
        )

        lrs = []
        for _ in range(100):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()

        # LR should increase then decrease
        max_idx = lrs.index(max(lrs))
        assert max_idx > 0  # Not at start
        assert max_idx < 99  # Not at end

    def test_lr_reaches_max(self) -> None:
        """Verify LR reaches max_lr."""
        model = SimpleModel()
        optimizer = create_optimizer(model)
        max_lr = 0.01
        scheduler = create_scheduler(
            optimizer,
            max_lr=max_lr,
            total_steps=100,
            pct_start=0.1,
        )

        lrs = []
        for _ in range(100):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()

        # Max LR should be approximately max_lr
        assert max(lrs) == pytest.approx(max_lr, rel=0.01)

    def test_paper_configuration(self) -> None:
        """Verify paper configuration works correctly."""
        model = SimpleModel()
        optimizer = create_optimizer(model)

        # Paper: max_lr=0.001, pct_start=0.01 (1% warmup)
        scheduler = create_scheduler(
            optimizer,
            max_lr=0.001,
            total_steps=50 * 224,  # 50 epochs * 224 samples
            pct_start=0.01,
        )

        # Initial LR should be very low
        initial_lr = scheduler.get_last_lr()[0]
        assert initial_lr < 0.0001  # Much less than max

        # After warmup steps, should be at max
        warmup_steps = int(50 * 224 * 0.01)  # 1% of total
        for _ in range(warmup_steps):
            scheduler.step()

        post_warmup_lr = scheduler.get_last_lr()[0]
        assert post_warmup_lr == pytest.approx(0.001, rel=0.1)  # Near max


class TestOptimizerSchedulerIntegration:
    """Tests for optimizer and scheduler working together."""

    def test_optimizer_and_scheduler_work_together(self) -> None:
        """Verify optimizer and scheduler integrate correctly."""
        model = SimpleModel()
        optimizer = create_optimizer(model)
        scheduler = create_scheduler(
            optimizer,
            max_lr=0.001,
            total_steps=10,
            pct_start=0.2,
        )

        # Simulate training loop
        for _ in range(10):
            # Zero grad
            optimizer.zero_grad()

            # Forward
            x = torch.randn(4, 10)
            y = model(x)
            loss = y.sum()

            # Backward
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Should complete without error
        assert True

    def test_parameters_update(self) -> None:
        """Verify optimizer updates model parameters."""
        model = SimpleModel()
        optimizer = create_optimizer(model)

        # Store initial weights
        initial_weights = model.linear.weight.clone()

        # Training step
        x = torch.randn(4, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        # Weights should have changed
        assert not torch.allclose(model.linear.weight, initial_weights)
