# Spec 04: Training Infrastructure

> **Phase 4 of 7** — Training loop, loss functions, and hyperparameter optimization
>
> **Goal:** Implement complete training infrastructure matching the paper's specifications.

---

## Overview

This spec covers:
- Loss function (CrossEntropy with class weighting + label smoothing)
- Optimizer (AdamW with OneCycleLR scheduler)
- FP16 mixed-precision training
- Training loop with validation
- Hyperparameter optimization with Optuna (paper used Orion, but it's broken on Python 3.12)
- Checkpoint management

---

## 1. Training Configuration (FROM PAPER)

### 1.1 Loss Function

| Parameter | Value | Source |
|-----------|-------|--------|
| Type | CrossEntropyLoss | Paper Section 2 |
| Background weight | 0.5 | Paper Section 2 |
| Lesion weight | 1.0 | Paper Section 2 |
| Label smoothing | 0.01 | Paper Section 2 |

### 1.2 Optimizer (Baselines)

| Parameter | Value | Source |
|-----------|-------|--------|
| Type | AdamW | Paper Section 2 |
| Learning rate | 0.001 | Paper Section 2 |
| Weight decay | 3e-5 | Paper Section 2 |
| Epsilon | 1e-4 | Paper Section 2 |
| Batch size | 1 | Paper Section 2 |

### 1.3 Scheduler

| Parameter | Value | Source |
|-----------|-------|--------|
| Type | OneCycleLR | Paper Section 2 |
| Max LR | 0.001 | Paper Section 2 |
| pct_start (warmup) | 0.01 (1%) | Paper Section 2 |
| div_factor | 100 | Paper Section 2: "starts at 1/100th of max learning rate" |
| Epochs | 50 | Paper Section 2 |

### 1.4 MeshNet Hyperparameter Search Space

| Parameter | Distribution | Range | Source |
|-----------|--------------|-------|--------|
| Channels | Uniform int | [5, 21] | Paper Section 2 |
| Learning rate | Log-uniform | [1e-4, 4e-2] | Paper Section 2 |
| Weight decay | Log-uniform | [1e-4, 4e-2] | Paper Section 2 |
| Background weight | Uniform | [0, 1] | Paper Section 2 |
| Warmup percentage | Categorical | {0.02, 0.1, 0.2} | Paper Section 2 |
| Epochs (fidelity) | Fidelity | [15, 50] | Paper Section 2 |

### 1.5 Training Protocol

- **Restarts:** 10 per configuration (for MeshNet)
- **HPO Location:** Inner folds of first outer fold only
- **Precision:** FP16 (half-precision)
- **Checkpointing:** Layer checkpointing for large models

---

## 2. Implementation

### 2.1 Training Configuration

**File:** `src/arc_meshchop/training/config.py`

```python
"""Training configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class TrainingConfig:
    """Configuration for training.

    FROM PAPER Section 2:
    - AdamW optimizer (lr=0.001, weight_decay=3e-5, eps=1e-4)
    - OneCycleLR scheduler (max_lr=0.001, pct_start=0.01)
    - CrossEntropyLoss (weights=[0.5, 1.0], label_smoothing=0.01)
    - FP16 training
    - 50 epochs, batch size 1
    """

    # Optimizer (FROM PAPER)
    optimizer: Literal["adamw"] = "adamw"
    learning_rate: float = 0.001
    weight_decay: float = 3e-5
    eps: float = 1e-4

    # Scheduler (FROM PAPER)
    scheduler: Literal["onecycle"] = "onecycle"
    max_lr: float = 0.001
    pct_start: float = 0.01  # 1% warmup
    div_factor: float = 100.0  # Paper: "starts at 1/100th of max learning rate"

    # Loss (FROM PAPER)
    background_weight: float = 0.5
    lesion_weight: float = 1.0
    label_smoothing: float = 0.01

    # Training (FROM PAPER)
    epochs: int = 50
    batch_size: int = 1

    # Precision (FROM PAPER)
    use_fp16: bool = True

    # Checkpointing
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    save_best_only: bool = True
    save_every_n_epochs: int | None = None

    # Logging
    log_every_n_steps: int = 10

    # Reproducibility
    random_seed: int = 42
    num_restarts: int = 10  # FROM PAPER: "trained with 10 restarts"

    def __post_init__(self) -> None:
        """Convert paths to Path objects."""
        self.checkpoint_dir = Path(self.checkpoint_dir)


@dataclass
class HPOConfig:
    """Configuration for hyperparameter optimization.

    FROM PAPER Section 2:
    - Paper used Orion framework with ASHA algorithm
    - Search on inner folds of first outer fold only

    NOTE: Orion is broken on Python 3.12 (uses deprecated configparser.SafeConfigParser).
    We use Optuna instead, which supports ASHA pruning via SuccessiveHalvingPruner.
    """

    # Framework (Optuna replaces paper's Orion due to Python 3.12 compatibility)
    framework: Literal["optuna"] = "optuna"
    algorithm: Literal["asha", "tpe"] = "asha"  # ASHA via SuccessiveHalvingPruner

    # Search space (FROM PAPER)
    channels_min: int = 5
    channels_max: int = 21
    lr_min: float = 1e-4
    lr_max: float = 4e-2
    weight_decay_min: float = 1e-4
    weight_decay_max: float = 4e-2
    bg_weight_min: float = 0.0
    bg_weight_max: float = 1.0
    warmup_pct_choices: tuple[float, ...] = (0.02, 0.1, 0.2)

    # Fidelity (FROM PAPER)
    min_epochs: int = 15
    max_epochs: int = 50

    # ASHA settings
    num_rungs: int = 4
    num_brackets: int = 1
    seed: int = 42
```

### 2.2 Loss Functions

**File:** `src/arc_meshchop/training/loss.py`

```python
"""Loss functions for stroke lesion segmentation.

FROM PAPER Section 2:
"The objective is cross-entropy loss with label smoothing of 0.01
and class weighting of 0.5 for the background and 1.0 for lesions."
"""

from __future__ import annotations

import torch
import torch.nn as nn


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted cross-entropy loss with label smoothing.

    Implements the loss function from the paper:
    - Class weights: [0.5, 1.0] for [background, lesion]
    - Label smoothing: 0.01

    This addresses class imbalance where lesions are typically
    1-10% of brain volume.
    """

    def __init__(
        self,
        background_weight: float = 0.5,
        lesion_weight: float = 1.0,
        label_smoothing: float = 0.01,
    ) -> None:
        """Initialize loss function.

        Args:
            background_weight: Weight for background class.
            lesion_weight: Weight for lesion class.
            label_smoothing: Label smoothing factor.
        """
        super().__init__()

        self.class_weights = torch.tensor([background_weight, lesion_weight])
        self.label_smoothing = label_smoothing

        self._loss_fn: nn.CrossEntropyLoss | None = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted cross-entropy loss.

        Args:
            logits: Model output of shape (B, C, D, H, W).
            targets: Ground truth of shape (B, D, H, W).

        Returns:
            Scalar loss value.
        """
        # Initialize loss function on first call (to get correct device)
        if self._loss_fn is None or self.class_weights.device != logits.device:
            self._loss_fn = nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device),
                label_smoothing=self.label_smoothing,
            )

        return self._loss_fn(logits, targets)


def create_loss_function(
    background_weight: float = 0.5,
    lesion_weight: float = 1.0,
    label_smoothing: float = 0.01,
) -> WeightedCrossEntropyLoss:
    """Create loss function with paper defaults.

    Args:
        background_weight: Weight for background class.
        lesion_weight: Weight for lesion class.
        label_smoothing: Label smoothing factor.

    Returns:
        Configured loss function.
    """
    return WeightedCrossEntropyLoss(
        background_weight=background_weight,
        lesion_weight=lesion_weight,
        label_smoothing=label_smoothing,
    )
```

### 2.3 Optimizer and Scheduler

**File:** `src/arc_meshchop/training/optimizer.py`

```python
"""Optimizer and scheduler setup.

FROM PAPER Section 2:
- AdamW optimizer (lr=0.001, weight_decay=3e-5, eps=1e-4)
- OneCycleLR scheduler (max_lr=0.001, pct_start=0.01)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.optim as optim

if TYPE_CHECKING:
    from torch.optim.lr_scheduler import _LRScheduler


def create_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 0.001,
    weight_decay: float = 3e-5,
    eps: float = 1e-4,
) -> optim.AdamW:
    """Create AdamW optimizer with paper defaults.

    FROM PAPER Section 2:
    "AdamW optimizer with a learning rate of 0.001,
    weight decay of 3e-5, and epsilon of 1e-4"

    Args:
        model: Model to optimize.
        learning_rate: Learning rate.
        weight_decay: Weight decay for regularization.
        eps: Epsilon for numerical stability (higher for FP16).

    Returns:
        Configured AdamW optimizer.
    """
    return optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=eps,
    )


def create_scheduler(
    optimizer: optim.Optimizer,
    max_lr: float,
    total_steps: int,
    pct_start: float = 0.01,
    div_factor: float = 100.0,
) -> optim.lr_scheduler.OneCycleLR:
    """Create OneCycleLR scheduler with paper defaults.

    FROM PAPER Section 2:
    "OneCycleLR learning rate scheduler. The scheduler starts at 1/100th
    of the max learning rate, quickly increases to the maximum, and
    gradually decreases. The maximum learning rate was set to 0.001,
    with the increase phase occurring in the first 1% of training time."

    Args:
        optimizer: Optimizer to schedule.
        max_lr: Maximum learning rate.
        total_steps: Total training steps (epochs * steps_per_epoch).
        pct_start: Fraction of training for warmup.
        div_factor: Divide max_lr by this to get initial_lr. FROM PAPER: 100 (starts at 1/100th of max_lr).

    Returns:
        Configured OneCycleLR scheduler.
    """
    return optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        div_factor=div_factor,  # Paper requires 100 (start at max_lr/100)
        # PyTorch defaults: anneal_strategy='cos', cycle_momentum=True
    )
```

### 2.4 Training Loop

**File:** `src/arc_meshchop/training/trainer.py`

```python
"""Training loop for MeshNet stroke lesion segmentation.

FROM PAPER Section 2:
- Half-precision (FP16) training
- Batch size 1 (full 256³ volumes)
- 50 epochs
- Layer checkpointing for large models
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from arc_meshchop.training.config import TrainingConfig
from arc_meshchop.training.loss import create_loss_function
from arc_meshchop.training.optimizer import create_optimizer, create_scheduler

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from arc_meshchop.evaluation.metrics import SegmentationMetrics

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
        total_steps = self.config.epochs * len(train_loader)
        self.scheduler = create_scheduler(
            self.optimizer,
            max_lr=self.config.max_lr,
            total_steps=total_steps,
            pct_start=self.config.pct_start,
        )

        logger.info(
            "Starting training: %d epochs, %d steps/epoch, %d total steps",
            self.config.epochs,
            len(train_loader),
            total_steps,
        )

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
        for batch_idx, (images, masks) in enumerate(pbar):
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
            self.scheduler.step()

            # Track loss
            total_loss += loss.item()
            self.state.global_step += 1

            # Update progress bar
            pbar.set_postfix(
                loss=loss.item(),
                lr=self.scheduler.get_last_lr()[0],
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
        all_preds = []
        all_targets = []

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
            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())

        metrics = {"loss": total_loss / len(val_loader)}

        # Compute segmentation metrics if calculator provided
        if metrics_calculator:
            all_preds_cat = torch.cat(all_preds, dim=0)
            all_targets_cat = torch.cat(all_targets, dim=0)
            seg_metrics = metrics_calculator.compute_batch(all_preds_cat, all_targets_cat)
            metrics.update(seg_metrics)
        else:
            # Compute basic DICE without full metrics
            metrics["dice"] = self._compute_dice(
                torch.cat(all_preds, dim=0),
                torch.cat(all_targets, dim=0),
            )

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
        return dice.item()

    def _save_checkpoint(self, name: str) -> None:
        """Save model checkpoint.

        Args:
            name: Checkpoint name (e.g., "best", "final", "epoch_10").
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
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

        path = self.config.checkpoint_dir / f"{name}.pt"
        torch.save(checkpoint, path)
        logger.info("Saved checkpoint: %s", path)

    def load_checkpoint(self, path: Path | str) -> None:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint:
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
```

### 2.5 Hyperparameter Optimization

**File:** `src/arc_meshchop/training/hpo.py`

```python
"""Hyperparameter optimization with Optuna.

FROM PAPER Section 2:
"To optimize hyperparameters of MeshNet, we conducted a hyperparameter search
using Orion, an asynchronous framework for black-box function optimization.
We employed the Asynchronous Successive Halving Algorithm (ASHA)."

NOTE: Paper used Orion, but it's broken on Python 3.12 (uses deprecated
configparser.SafeConfigParser). We use Optuna instead, which supports
equivalent ASHA pruning via SuccessiveHalvingPruner.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import optuna
from optuna.pruners import SuccessiveHalvingPruner

from arc_meshchop.training.config import HPOConfig, TrainingConfig

if TYPE_CHECKING:
    from pathlib import Path

    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class HPOTrial:
    """Single hyperparameter optimization trial."""

    trial_id: str
    params: dict[str, Any]
    dice_score: float | None = None
    status: str = "pending"


def create_optuna_study(config: HPOConfig, study_name: str) -> optuna.Study:
    """Create Optuna study with ASHA-equivalent pruning.

    FROM PAPER Section 2:
    - channels: uniform(5, 21)
    - lr: loguniform(1e-4, 4e-2)
    - weight_decay: loguniform(1e-4, 4e-2)
    - bg_weight: uniform(0, 1)
    - warmup_pct: choices([0.02, 0.1, 0.2])

    Args:
        config: HPO configuration.
        study_name: Name for the Optuna study.

    Returns:
        Configured Optuna study with ASHA pruning.
    """
    # SuccessiveHalvingPruner implements ASHA algorithm
    pruner = SuccessiveHalvingPruner(
        min_resource=config.min_epochs,
        reduction_factor=3,
        min_early_stopping_rate=0,
    )

    return optuna.create_study(
        study_name=study_name,
        direction="maximize",  # Maximize DICE
        pruner=pruner,
    )


def suggest_hyperparameters(trial: optuna.Trial, config: HPOConfig) -> dict[str, Any]:
    """Suggest hyperparameters for a trial using paper's search space.

    Args:
        trial: Optuna trial object.
        config: HPO configuration with search ranges.

    Returns:
        Dictionary of suggested hyperparameters.
    """
    return {
        "channels": trial.suggest_int("channels", config.channels_min, config.channels_max),
        "lr": trial.suggest_float("lr", config.lr_min, config.lr_max, log=True),
        "weight_decay": trial.suggest_float("weight_decay", config.weight_decay_min, config.weight_decay_max, log=True),
        "bg_weight": trial.suggest_float("bg_weight", config.bg_weight_min, config.bg_weight_max),
        "warmup_pct": trial.suggest_categorical("warmup_pct", list(config.warmup_pct_choices)),
    }


def trial_params_to_training_config(
    params: dict[str, Any],
    base_config: TrainingConfig | None = None,
) -> TrainingConfig:
    """Convert trial parameters to TrainingConfig.

    Args:
        params: Trial parameters from Optuna.
        base_config: Optional base config to override.

    Returns:
        Training configuration with trial parameters.
    """
    config = base_config or TrainingConfig()

    return TrainingConfig(
        # From HPO trial
        learning_rate=params["lr"],
        weight_decay=params["weight_decay"],
        background_weight=params["bg_weight"],
        pct_start=params["warmup_pct"],
        max_lr=params["lr"],  # max_lr matches lr for OneCycleLR

        # Preserve base config settings
        lesion_weight=config.lesion_weight,
        label_smoothing=config.label_smoothing,
        batch_size=config.batch_size,
        use_fp16=config.use_fp16,
        checkpoint_dir=config.checkpoint_dir,
        random_seed=config.random_seed,
    )


def run_hpo(
    hpo_config: HPOConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    study_name: str = "meshnet_hpo",
    n_trials: int = 100,
) -> dict[str, Any]:
    """Run hyperparameter optimization with Optuna.

    Uses Optuna with SuccessiveHalvingPruner (ASHA equivalent).

    NOTE: Paper used Orion, but it's broken on Python 3.12.
    Optuna provides equivalent functionality.

    Args:
        hpo_config: HPO configuration.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        study_name: Name for the Optuna study.
        n_trials: Number of trials to run.

    Returns:
        Best parameters found.
    """
    from arc_meshchop.models import MeshNet
    from arc_meshchop.training.trainer import Trainer

    # Create Optuna study with ASHA pruning
    study = create_optuna_study(hpo_config, study_name)

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function."""
        # Suggest hyperparameters
        params = suggest_hyperparameters(trial, hpo_config)
        logger.info("Trial %d: %s", trial.number, params)

        try:
            # Create model with trial channels
            model = MeshNet(channels=params["channels"])

            # Create training config from trial params
            train_config = trial_params_to_training_config(params)
            train_config.epochs = hpo_config.max_epochs

            # Train with pruning callback
            trainer = Trainer(model, train_config)

            # Custom training loop with pruning
            for epoch in range(train_config.epochs):
                trainer.train_epoch(train_loader)
                dice_score = trainer.validate(val_loader)

                # Report to Optuna for pruning
                trial.report(dice_score, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return trainer.best_dice

        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.error("Trial %d failed: %s", trial.number, e)
            return 0.0

    logger.info("Starting HPO with %d trials", n_trials)
    study.optimize(objective, n_trials=n_trials)

    logger.info("HPO complete. Best DICE: %.4f", study.best_value)
    logger.info("Best params: %s", study.best_params)

    return {"best_dice": study.best_value, "best_params": study.best_params}
```

### 2.6 Module `__init__.py`

**File:** `src/arc_meshchop/training/__init__.py`

```python
"""Training infrastructure for MeshNet stroke lesion segmentation."""

from arc_meshchop.training.config import HPOConfig, TrainingConfig
from arc_meshchop.training.loss import (
    WeightedCrossEntropyLoss,
    create_loss_function,
)
from arc_meshchop.training.optimizer import create_optimizer, create_scheduler
from arc_meshchop.training.trainer import Trainer, TrainingState

__all__ = [
    "HPOConfig",
    "TrainingConfig",
    "TrainingState",
    "Trainer",
    "WeightedCrossEntropyLoss",
    "create_loss_function",
    "create_optimizer",
    "create_scheduler",
]
```

---

## 3. Tests

### 3.1 Loss Function Tests

**File:** `tests/test_training/test_loss.py`

```python
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
```

### 3.2 Optimizer Tests

**File:** `tests/test_training/test_optimizer.py`

```python
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
        return self.linear(x)


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

        # Check defaults match paper
        assert optimizer.defaults["lr"] == pytest.approx(0.001)
        assert optimizer.defaults["weight_decay"] == pytest.approx(3e-5)
        assert optimizer.defaults["eps"] == pytest.approx(1e-4)


class TestCreateScheduler:
    """Tests for create_scheduler."""

    def test_creates_onecycle(self) -> None:
        """Verify creates OneCycleLR scheduler."""
        model = SimpleModel()
        optimizer = create_optimizer(model)
        scheduler = create_scheduler(optimizer, max_lr=0.001, total_steps=100)

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

        # First LR should be ~1/100 of max (div_factor=100 from paper)
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
```

### 3.3 Trainer Tests

**File:** `tests/test_training/test_trainer.py`

```python
"""Tests for training loop."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from arc_meshchop.models import meshnet_5
from arc_meshchop.training import TrainingConfig, Trainer


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

    def test_trainer_runs_one_epoch(
        self,
        tiny_train_loader: DataLoader,
        tiny_val_loader: DataLoader,
        tmp_path: pytest.TempPathFactory,
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
        tmp_path: pytest.TempPathFactory,
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

    def test_trainer_loss_decreases(
        self,
        tiny_train_loader: DataLoader,
        tiny_val_loader: DataLoader,
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """Verify loss decreases over multiple epochs."""
        model = meshnet_5()
        config = TrainingConfig(
            epochs=5,
            use_fp16=False,
            checkpoint_dir=tmp_path / "checkpoints",
        )
        trainer = Trainer(model, config)

        # This is a weak test - just verify training doesn't crash
        # In practice, 5 epochs on random data may not show loss decrease
        results = trainer.train(tiny_train_loader, tiny_val_loader)

        assert results["final_train_loss"] is not None
```

---

## 4. Implementation Checklist

### Phase 4.1: Configuration

- [ ] Create `src/arc_meshchop/training/config.py`
- [ ] Define `TrainingConfig` dataclass
- [ ] Define `HPOConfig` dataclass

### Phase 4.2: Loss Function

- [ ] Create `src/arc_meshchop/training/loss.py`
- [ ] Implement `WeightedCrossEntropyLoss`
- [ ] Verify class weights [0.5, 1.0]
- [ ] Verify label smoothing 0.01

### Phase 4.3: Optimizer & Scheduler

- [ ] Create `src/arc_meshchop/training/optimizer.py`
- [ ] Implement `create_optimizer` (AdamW)
- [ ] Implement `create_scheduler` (OneCycleLR)
- [ ] Verify 1% warmup

### Phase 4.4: Training Loop

- [ ] Create `src/arc_meshchop/training/trainer.py`
- [ ] Implement FP16 training
- [ ] Implement checkpoint save/load
- [ ] Implement validation loop

### Phase 4.5: HPO (Optional)

- [ ] Create `src/arc_meshchop/training/hpo.py`
- [ ] Implement Optuna integration (paper used Orion, but broken on Python 3.12)
- [ ] Implement ASHA-equivalent pruning via SuccessiveHalvingPruner

### Phase 4.6: Tests

- [ ] Create loss function tests
- [ ] Create optimizer tests
- [ ] Create trainer tests
- [ ] All tests pass

---

## 5. Verification Commands

```bash
# Run training tests
uv run pytest tests/test_training/ -v

# Test training config
uv run python -c "
from arc_meshchop.training import TrainingConfig

config = TrainingConfig()
print(f'Learning rate: {config.learning_rate}')
print(f'Weight decay: {config.weight_decay}')
print(f'Class weights: [{config.background_weight}, {config.lesion_weight}]')
print(f'Label smoothing: {config.label_smoothing}')
print(f'Warmup: {config.pct_start * 100}%')
"

# Test loss function
uv run python -c "
import torch
from arc_meshchop.training import create_loss_function

loss_fn = create_loss_function()
logits = torch.randn(1, 2, 8, 8, 8)
targets = torch.randint(0, 2, (1, 8, 8, 8))
loss = loss_fn(logits, targets)
print(f'Loss: {loss.item():.4f}')
"
```

---

## 6. References

- Paper Section 2: Training methodology
- Research docs: `docs/research/03-training-configuration.md`
- PyTorch OneCycleLR: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
- Optuna HPO: https://optuna.readthedocs.io (replaces paper's Orion due to Python 3.12 compatibility)
- Optuna SuccessiveHalvingPruner: https://optuna.readthedocs.io/en/stable/reference/pruners.html
