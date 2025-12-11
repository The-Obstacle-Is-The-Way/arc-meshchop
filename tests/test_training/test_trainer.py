"""Tests for training loop."""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from arc_meshchop.models import meshnet_5
from arc_meshchop.training import Trainer, TrainingConfig, TrainingState


@pytest.fixture
def tiny_train_loader() -> DataLoader:
    """Create tiny training data loader for tests."""
    # 2 samples, 8³ volumes
    images = torch.randn(2, 1, 8, 8, 8)
    masks = torch.randint(0, 2, (2, 8, 8, 8))
    dataset = TensorDataset(images, masks)
    return DataLoader(dataset, batch_size=1, shuffle=True)


@pytest.fixture
def tiny_val_loader() -> DataLoader:
    """Create tiny validation data loader for tests."""
    images = torch.randn(2, 1, 8, 8, 8)
    masks = torch.randint(0, 2, (2, 8, 8, 8))
    dataset = TensorDataset(images, masks)
    return DataLoader(dataset, batch_size=1, shuffle=False)


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values_from_paper(self) -> None:
        """Verify default values match paper."""
        config = TrainingConfig()

        # FROM PAPER Section 2
        assert config.learning_rate == pytest.approx(0.001)
        assert config.weight_decay == pytest.approx(3e-5)
        assert config.eps == pytest.approx(1e-4)
        assert config.max_lr == pytest.approx(0.001)
        assert config.pct_start == pytest.approx(0.01)  # 1% warmup
        assert config.background_weight == pytest.approx(0.5)
        assert config.lesion_weight == pytest.approx(1.0)
        assert config.label_smoothing == pytest.approx(0.01)
        assert config.epochs == 50
        assert config.batch_size == 1

    def test_checkpoint_dir_is_path(self) -> None:
        """Verify checkpoint_dir is a Path object."""
        config = TrainingConfig()
        assert isinstance(config.checkpoint_dir, Path)

    def test_checkpoint_dir_string_conversion(self) -> None:
        """Verify string checkpoint_dir is converted to Path."""
        config = TrainingConfig(checkpoint_dir="my_checkpoints")  # type: ignore[arg-type]
        assert isinstance(config.checkpoint_dir, Path)
        assert str(config.checkpoint_dir) == "my_checkpoints"


class TestTrainingState:
    """Tests for TrainingState dataclass."""

    def test_initial_state(self) -> None:
        """Verify initial training state."""
        state = TrainingState(
            epoch=0,
            best_dice=0.0,
            best_epoch=0,
            global_step=0,
        )

        assert state.epoch == 0
        assert state.best_dice == 0.0
        assert state.best_epoch == 0
        assert state.global_step == 0


class TestTrainer:
    """Tests for Trainer class."""

    def test_trainer_initializes(self) -> None:
        """Verify trainer initializes correctly."""
        model = meshnet_5()
        config = TrainingConfig(epochs=1, use_fp16=False)
        trainer = Trainer(model, config)

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.loss_fn is not None

    def test_trainer_device_assignment(self) -> None:
        """Verify trainer assigns device correctly."""
        model = meshnet_5()
        config = TrainingConfig(epochs=1, use_fp16=False)
        trainer = Trainer(model, config)

        # Device should be set
        assert trainer.device is not None
        # Model should be on same device type (mps:0 == mps, cuda:0 == cuda)
        param_device = next(trainer.model.parameters()).device
        assert param_device.type == trainer.device.type

    def test_trainer_runs_one_epoch(
        self,
        tiny_train_loader: DataLoader,
        tiny_val_loader: DataLoader,
        tmp_path: Path,
    ) -> None:
        """Verify trainer can complete one epoch."""
        model = meshnet_5()
        config = TrainingConfig(
            epochs=1,
            use_fp16=False,
            checkpoint_dir=tmp_path / "checkpoints",
        )
        trainer = Trainer(model, config)

        results = trainer.train(tiny_train_loader, tiny_val_loader)

        assert "best_dice" in results
        assert results["best_dice"] >= 0.0
        assert results["best_dice"] <= 1.0

    def test_trainer_saves_checkpoint(
        self,
        tiny_train_loader: DataLoader,
        tiny_val_loader: DataLoader,
        tmp_path: Path,
    ) -> None:
        """Verify trainer saves checkpoints."""
        model = meshnet_5()
        checkpoint_dir = tmp_path / "checkpoints"
        config = TrainingConfig(
            epochs=1,
            use_fp16=False,
            checkpoint_dir=checkpoint_dir,
        )
        trainer = Trainer(model, config)

        trainer.train(tiny_train_loader, tiny_val_loader)

        # Check checkpoints exist
        assert (checkpoint_dir / "best.pt").exists()
        assert (checkpoint_dir / "final.pt").exists()

    def test_trainer_checkpoint_contains_state(
        self,
        tiny_train_loader: DataLoader,
        tiny_val_loader: DataLoader,
        tmp_path: Path,
    ) -> None:
        """Verify checkpoint contains all necessary state."""
        model = meshnet_5()
        checkpoint_dir = tmp_path / "checkpoints"
        config = TrainingConfig(
            epochs=1,
            use_fp16=False,
            checkpoint_dir=checkpoint_dir,
        )
        trainer = Trainer(model, config)

        trainer.train(tiny_train_loader, tiny_val_loader)

        # Load and verify checkpoint contents
        checkpoint = torch.load(checkpoint_dir / "final.pt", weights_only=False)

        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "training_state" in checkpoint
        assert "config" in checkpoint

    def test_trainer_loads_checkpoint(
        self,
        tiny_train_loader: DataLoader,
        tiny_val_loader: DataLoader,
        tmp_path: Path,
    ) -> None:
        """Verify trainer can load checkpoint."""
        model = meshnet_5()
        checkpoint_dir = tmp_path / "checkpoints"
        config = TrainingConfig(
            epochs=1,
            use_fp16=False,
            checkpoint_dir=checkpoint_dir,
        )
        trainer = Trainer(model, config)

        # Train and save
        trainer.train(tiny_train_loader, tiny_val_loader)

        # Create new trainer and load
        model2 = meshnet_5()
        trainer2 = Trainer(model2, config)
        trainer2.train(tiny_train_loader, tiny_val_loader)  # Initialize scheduler
        trainer2.load_checkpoint(checkpoint_dir / "final.pt")

        # State should be loaded
        assert trainer2.state.epoch == trainer.state.epoch

    def test_trainer_updates_best_dice(
        self,
        tiny_train_loader: DataLoader,
        tiny_val_loader: DataLoader,
        tmp_path: Path,
    ) -> None:
        """Verify trainer tracks best DICE score."""
        model = meshnet_5()
        config = TrainingConfig(
            epochs=2,
            use_fp16=False,
            checkpoint_dir=tmp_path / "checkpoints",
        )
        trainer = Trainer(model, config)

        results = trainer.train(tiny_train_loader, tiny_val_loader)

        # Should have tracked some best DICE
        assert "best_dice" in results
        assert "best_epoch" in results

    def test_trainer_periodic_checkpoint(
        self,
        tiny_train_loader: DataLoader,
        tiny_val_loader: DataLoader,
        tmp_path: Path,
    ) -> None:
        """Verify trainer saves periodic checkpoints."""
        model = meshnet_5()
        checkpoint_dir = tmp_path / "checkpoints"
        config = TrainingConfig(
            epochs=4,
            use_fp16=False,
            checkpoint_dir=checkpoint_dir,
            save_every_n_epochs=2,
        )
        trainer = Trainer(model, config)

        trainer.train(tiny_train_loader, tiny_val_loader)

        # Should have epoch checkpoints
        assert (checkpoint_dir / "epoch_2.pt").exists()
        assert (checkpoint_dir / "epoch_4.pt").exists()

    def test_trainer_fp32_on_cpu(
        self,
        tiny_train_loader: DataLoader,
        tiny_val_loader: DataLoader,
        tmp_path: Path,
    ) -> None:
        """Verify trainer uses FP32 on CPU even if FP16 requested."""
        model = meshnet_5()
        config = TrainingConfig(
            epochs=1,
            use_fp16=True,  # Request FP16
            checkpoint_dir=tmp_path / "checkpoints",
        )
        trainer = Trainer(model, config, device=torch.device("cpu"))

        # On CPU, FP16 should be disabled
        assert trainer._amp_enabled is False
        assert trainer._amp_dtype == torch.float32

        # Should still run without error
        results = trainer.train(tiny_train_loader, tiny_val_loader)
        assert "best_dice" in results


class TestTrainerValidation:
    """Tests for trainer validation functionality."""

    def test_compute_dice(self) -> None:
        """Test DICE computation."""
        model = meshnet_5()
        config = TrainingConfig(epochs=1, use_fp16=False)
        trainer = Trainer(model, config)

        # Perfect match
        preds = torch.ones(1, 8, 8, 8)
        targets = torch.ones(1, 8, 8, 8)
        dice = trainer._compute_dice(preds, targets)
        assert dice == pytest.approx(1.0, rel=0.01)

        # No overlap
        preds = torch.ones(1, 8, 8, 8)
        targets = torch.zeros(1, 8, 8, 8)
        dice = trainer._compute_dice(preds, targets)
        assert dice < 0.01  # Near zero

    def test_compute_dice_partial_overlap(self) -> None:
        """Test DICE with partial overlap."""
        model = meshnet_5()
        config = TrainingConfig(epochs=1, use_fp16=False)
        trainer = Trainer(model, config)

        # 50% overlap
        preds = torch.zeros(1, 8, 8, 8)
        preds[0, :4, :, :] = 1  # Half is predicted

        targets = torch.zeros(1, 8, 8, 8)
        targets[0, 2:6, :, :] = 1  # Half is target, 25% overlap with preds

        dice = trainer._compute_dice(preds, targets)
        assert 0.0 < dice < 1.0  # Partial overlap


class TestTrainerIntegration:
    """Integration tests for trainer."""

    def test_full_training_loop(
        self,
        tiny_train_loader: DataLoader,
        tiny_val_loader: DataLoader,
        tmp_path: Path,
    ) -> None:
        """Test complete training loop."""
        model = meshnet_5()
        config = TrainingConfig(
            epochs=3,
            use_fp16=False,
            checkpoint_dir=tmp_path / "checkpoints",
        )
        trainer = Trainer(model, config)

        results = trainer.train(tiny_train_loader, tiny_val_loader)

        # Verify all expected outputs
        assert "best_dice" in results
        assert "best_epoch" in results
        assert "final_train_loss" in results

        # Verify state was updated
        assert trainer.state.global_step > 0
        assert trainer.state.epoch == 2  # 0-indexed, last epoch

    def test_training_loss_is_finite(
        self,
        tiny_train_loader: DataLoader,
        tiny_val_loader: DataLoader,
        tmp_path: Path,
    ) -> None:
        """Verify training loss stays finite."""
        model = meshnet_5()
        config = TrainingConfig(
            epochs=2,
            use_fp16=False,
            checkpoint_dir=tmp_path / "checkpoints",
        )
        trainer = Trainer(model, config)

        results = trainer.train(tiny_train_loader, tiny_val_loader)

        assert not torch.isnan(torch.tensor(results["final_train_loss"]))
        assert not torch.isinf(torch.tensor(results["final_train_loss"]))
