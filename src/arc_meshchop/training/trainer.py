"""Training loop for MeshNet stroke lesion segmentation.

FROM PAPER Section 2:
- Half-precision (FP16) training
- Batch size 1 (full 256³ volumes)
- 50 epochs
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast  # type: ignore[attr-defined]
from tqdm import tqdm

from arc_meshchop.training.config import TrainingConfig
from arc_meshchop.training.loss import create_loss_function
from arc_meshchop.training.optimizer import create_optimizer, create_scheduler

if TYPE_CHECKING:
    from typing import Any

    from torch.utils.data import DataLoader

    # SegmentationMetrics will be implemented in Phase 5
    SegmentationMetrics = Any

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """Current training state for checkpointing."""

    epoch: int
    best_dice: float
    best_epoch: int
    global_step: int


class Trainer:
    """Trainer for MeshNet stroke lesion segmentation.

    Implements the training procedure from the paper:
    - FP16 mixed-precision training
    - AdamW optimizer with OneCycleLR scheduler
    - Weighted cross-entropy loss with label smoothing
    - Checkpoint management
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device | None = None,
    ) -> None:
        """Initialize trainer.

        Args:
            model: Model to train.
            config: Training configuration.
            device: Device to train on (default: auto-detect).
        """
        from arc_meshchop.utils.device import get_device

        self.model = model
        self.config = config

        # Cross-platform device selection (CUDA > MPS > CPU)
        self.device = device or get_device()

        # Move model to device
        self.model = self.model.to(self.device)

        # Create loss function (FROM PAPER)
        self.loss_fn = create_loss_function(
            background_weight=config.background_weight,
            lesion_weight=config.lesion_weight,
            label_smoothing=config.label_smoothing,
        )

        # Create optimizer (FROM PAPER)
        self.optimizer = create_optimizer(
            self.model,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=config.eps,
        )

        # Platform-aware mixed precision setup
        # CUDA: Full FP16 support with GradScaler
        # MPS: Limited FP16 - use FP32 (still fast due to unified memory)
        # CPU: No mixed precision (FP32)
        if self.device.type == "cuda":
            self.scaler = GradScaler("cuda", enabled=config.use_fp16)
            self._amp_dtype = torch.float16
            self._amp_enabled = config.use_fp16
        elif self.device.type == "mps":
            # MPS has limited FP16 support - disable scaling
            self.scaler = GradScaler("cpu", enabled=False)
            self._amp_dtype = torch.float32
            self._amp_enabled = False
            if config.use_fp16:
                logger.warning("FP16 not fully supported on MPS, using FP32")
        else:
            self.scaler = GradScaler("cpu", enabled=False)
            self._amp_dtype = torch.float32
            self._amp_enabled = False

        # Training state
        self.state = TrainingState(
            epoch=0,
            best_dice=0.0,
            best_epoch=0,
            global_step=0,
        )

        # Scheduler will be created in train() when we know total_steps
        self.scheduler: torch.optim.lr_scheduler.OneCycleLR | None = None

        # Create checkpoint directory
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        metrics_calculator: SegmentationMetrics | None = None,
    ) -> dict[str, float]:
        """Train the model.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            metrics_calculator: Optional metrics calculator for validation.

        Returns:
            Dictionary with final metrics.
        """
        # Create scheduler with total steps
        # NOTE: div_factor=100 is critical for paper parity
        # FROM PAPER: "starts at 1/100th of the max learning rate"
        total_steps = self.config.epochs * len(train_loader)
        self.scheduler = create_scheduler(
            self.optimizer,
            max_lr=self.config.max_lr,
            total_steps=total_steps,
            pct_start=self.config.pct_start,
            div_factor=self.config.div_factor,
        )

        logger.info(
            "Starting training: %d epochs, %d steps/epoch, %d total steps",
            self.config.epochs,
            len(train_loader),
            total_steps,
        )

        train_loss = 0.0

        for epoch in range(self.config.epochs):
            self.state.epoch = epoch

            # Training epoch
            train_loss = self._train_epoch(train_loader)

            # Validation epoch
            val_metrics = self._validate_epoch(val_loader, metrics_calculator)
            val_dice = val_metrics.get("dice", 0.0)

            logger.info(
                "Epoch %d/%d - Train Loss: %.4f, Val DICE: %.4f",
                epoch + 1,
                self.config.epochs,
                train_loss,
                val_dice,
            )

            # Check for best model
            if val_dice > self.state.best_dice:
                self.state.best_dice = val_dice
                self.state.best_epoch = epoch
                if self.config.save_best_only:
                    self._save_checkpoint("best")

            # Periodic checkpoint
            if (
                self.config.save_every_n_epochs
                and (epoch + 1) % self.config.save_every_n_epochs == 0
            ):
                self._save_checkpoint(f"epoch_{epoch + 1}")

        # Save final checkpoint
        self._save_checkpoint("final")

        return {
            "best_dice": self.state.best_dice,
            "best_epoch": self.state.best_epoch,
            "final_train_loss": train_loss,
        }

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Run single training epoch.

        Args:
            train_loader: Training data loader.

        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {self.state.epoch + 1}")
        for images, masks in pbar:
            # Move to device
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass with platform-aware mixed precision
            self.optimizer.zero_grad()

            with autocast(
                device_type=self.device.type,
                dtype=self._amp_dtype,
                enabled=self._amp_enabled,
            ):
                outputs = self.model(images)
                loss = self.loss_fn(outputs, masks)

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update scheduler (step per batch for OneCycleLR)
            if self.scheduler is not None:
                self.scheduler.step()

            # Track loss
            total_loss += loss.item()
            self.state.global_step += 1

            # Update progress bar
            current_lr = (
                self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate
            )
            pbar.set_postfix(
                loss=loss.item(),
                lr=current_lr,
            )

        return total_loss / len(train_loader)

    @torch.no_grad()
    def _validate_epoch(
        self,
        val_loader: DataLoader,
        metrics_calculator: SegmentationMetrics | None = None,
    ) -> dict[str, float]:
        """Run validation epoch.

        Args:
            val_loader: Validation data loader.
            metrics_calculator: Optional metrics calculator.

        Returns:
            Dictionary with validation metrics.
        """
        self.model.eval()
        total_loss = 0.0

        # Accumulate scores, not tensors
        dice_scores: list[float] = []
        avd_scores: list[float] = []
        mcc_scores: list[float] = []

        for images, masks in tqdm(val_loader, desc="Validation"):
            images = images.to(self.device)
            masks = masks.to(self.device)

            with autocast(
                device_type=self.device.type,
                dtype=self._amp_dtype,
                enabled=self._amp_enabled,
            ):
                outputs = self.model(images)
                loss = self.loss_fn(outputs, masks)

            total_loss += loss.item()

            # Get predictions
            preds = outputs.argmax(dim=1)

            # Compute metrics for this batch (incrementally)
            if metrics_calculator:
                batch_metrics = metrics_calculator.compute_with_stats(preds.cpu(), masks.cpu())
                dice_scores.extend(batch_metrics["dice"].values)
                avd_scores.extend(batch_metrics["avd"].values)
                mcc_scores.extend(batch_metrics["mcc"].values)
            else:
                # Basic DICE only
                # Iterate batch items if batch_size > 1
                batch_size = preds.shape[0]
                preds_cpu = preds.cpu()
                masks_cpu = masks.cpu()
                for i in range(batch_size):
                    dice = self._compute_dice(preds_cpu[i], masks_cpu[i])
                    dice_scores.append(dice)

        metrics: dict[str, float] = {"loss": total_loss / len(val_loader)}

        # Aggregate metrics
        if metrics_calculator:
            import numpy as np

            metrics["dice"] = float(np.mean(dice_scores)) if dice_scores else 0.0
            metrics["avd"] = float(np.mean(avd_scores)) if avd_scores else 0.0
            metrics["mcc"] = float(np.mean(mcc_scores)) if mcc_scores else 0.0
        else:
            import numpy as np

            metrics["dice"] = float(np.mean(dice_scores)) if dice_scores else 0.0

        return metrics

    def _compute_dice(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        smooth: float = 1e-6,
    ) -> float:
        """Compute DICE coefficient.

        Args:
            preds: Predicted segmentation.
            targets: Ground truth segmentation.
            smooth: Smoothing factor to avoid division by zero.

        Returns:
            DICE coefficient.
        """
        preds_flat = preds.flatten()
        targets_flat = targets.flatten()

        intersection = (preds_flat * targets_flat).sum()
        union = preds_flat.sum() + targets_flat.sum()

        dice = (2.0 * intersection + smooth) / (union + smooth)
        return float(dice.item())

    def _save_checkpoint(self, name: str) -> None:
        """Save model checkpoint.

        Args:
            name: Checkpoint name (e.g., "best", "final", "epoch_10").
        """
        checkpoint: dict[str, object] = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "training_state": {
                "epoch": self.state.epoch,
                "best_dice": self.state.best_dice,
                "best_epoch": self.state.best_epoch,
                "global_step": self.state.global_step,
            },
            "config": {
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "epochs": self.config.epochs,
            },
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        path = self.config.checkpoint_dir / f"{name}.pt"
        torch.save(checkpoint, path)
        logger.info("Saved checkpoint: %s", path)

    def load_checkpoint(self, path: Path | str) -> None:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if "training_state" in checkpoint:
            state = checkpoint["training_state"]
            self.state = TrainingState(
                epoch=state["epoch"],
                best_dice=state["best_dice"],
                best_epoch=state["best_epoch"],
                global_step=state["global_step"],
            )

        logger.info("Loaded checkpoint: %s", path)

    def train_epoch(self, train_loader: DataLoader) -> dict[str, float]:
        """Run single training epoch (public API for HPO).

        This is a public wrapper around _train_epoch for use in
        hyperparameter optimization where epoch-level control is needed.

        Args:
            train_loader: Training data loader.

        Returns:
            Dictionary with training metrics (loss).
        """
        loss = self._train_epoch(train_loader)
        self.state.epoch += 1
        return {"loss": loss}

    def validate(
        self,
        val_loader: DataLoader,
        metrics_calculator: SegmentationMetrics | None = None,
    ) -> dict[str, float]:
        """Run validation (public API for HPO).

        This is a public wrapper around _validate_epoch for use in
        hyperparameter optimization where epoch-level control is needed.

        Args:
            val_loader: Validation data loader.
            metrics_calculator: Optional metrics calculator.

        Returns:
            Dictionary with validation metrics (dice, loss).
        """
        return self._validate_epoch(val_loader, metrics_calculator)

    def setup_scheduler(self, train_loader: DataLoader) -> None:
        """Set up OneCycleLR scheduler for epoch-level training.

        Must be called before using train_epoch() for HPO.

        Args:
            train_loader: Training data loader (needed for total_steps).
        """
        total_steps = self.config.epochs * len(train_loader)
        self.scheduler = create_scheduler(
            self.optimizer,
            max_lr=self.config.max_lr,
            total_steps=total_steps,
            pct_start=self.config.pct_start,
            div_factor=self.config.div_factor,
        )
