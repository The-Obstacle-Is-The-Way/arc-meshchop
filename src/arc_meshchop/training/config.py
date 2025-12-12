"""Training configuration dataclasses.

FROM PAPER Section 2:
- AdamW optimizer (lr=0.001, weight_decay=3e-5, eps=1e-4)
- OneCycleLR scheduler (max_lr=0.001, pct_start=0.01)
- CrossEntropyLoss (weights=[0.5, 1.0], label_smoothing=0.01)
- FP16 training
- 50 epochs, batch size 1
"""

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
    div_factor: float = 100.0  # FROM PAPER: "starts at 1/100th of the max learning rate"

    # Loss (FROM PAPER)
    background_weight: float = 0.5
    lesion_weight: float = 1.0
    label_smoothing: float = 0.01

    # Training (FROM PAPER)
    epochs: int = 50
    batch_size: int = 1

    # Precision (FROM PAPER)
    use_fp16: bool = True
    use_gradient_checkpointing: bool = False

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
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)


@dataclass
class HPOConfig:
    """Configuration for hyperparameter optimization.

    FROM PAPER Section 2:
    - Paper used Orion framework with ASHA algorithm
    - Search on inner folds of first outer fold only

    NOTE: Orion is broken on Python 3.12 (uses deprecated configparser.SafeConfigParser).
    We use Optuna instead, which supports ASHA-equivalent pruning via SuccessiveHalvingPruner.
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
