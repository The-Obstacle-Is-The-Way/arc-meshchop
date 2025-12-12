"""Experiment configuration for paper replication."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class ExperimentConfig:
    """Configuration for full experiment.

    FROM PAPER Section 2:
    - 3 outer folds x 3 inner folds x 10 restarts = 90 runs
    - MeshNet-26 with 147,474 parameters
    - Target: 0.876 DICE
    """

    # Data
    data_dir: Path = field(default_factory=lambda: Path("data/arc"))
    output_dir: Path = field(default_factory=lambda: Path("experiments"))

    # Model
    model_variant: Literal["meshnet_5", "meshnet_16", "meshnet_26"] = "meshnet_26"
    channels: int = 26

    # Cross-validation structure (FROM PAPER)
    num_outer_folds: int = 3
    num_inner_folds: int = 3
    num_restarts: int = 10

    # Training hyperparameters (FROM PAPER)
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 3e-5
    background_weight: float = 0.5
    warmup_pct: float = 0.01
    div_factor: float = 100.0  # FROM PAPER: "starts at 1/100th of the max learning rate"
    use_fp16: bool = True

    # Execution
    parallel_runs: int = 1  # Number of parallel training runs
    skip_completed: bool = True  # Skip runs that already have results
    save_all_checkpoints: bool = False  # Save checkpoint for every restart

    # Random seeds (for restarts)
    base_seed: int = 42

    def __post_init__(self) -> None:
        """Convert paths and validate."""
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)

        if self.model_variant == "meshnet_5":
            self.channels = 5
        elif self.model_variant == "meshnet_16":
            self.channels = 16
        else:
            self.channels = 26

    @property
    def total_runs(self) -> int:
        """Total number of training runs."""
        return self.num_outer_folds * self.num_inner_folds * self.num_restarts

    def get_restart_seed(self, restart: int) -> int:
        """Get seed for specific restart."""
        return self.base_seed + restart
