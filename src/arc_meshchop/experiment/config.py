"""Experiment configuration for paper replication."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class ExperimentConfig:
    """Configuration for full experiment.

    FROM PAPER Section 2:
    "Hyperparameter optimization was conducted on the inner folds of the first
    outer fold. The optimized hyperparameters were then applied to train models
    on all outer folds."

    This means:
    - HP search: Only on Outer Fold 1 (we skip this, using paper's final HPs)
    - Final evaluation: 3 outer folds x 10 restarts = 30 runs
    - Training data: Full outer-train (67% of total), not inner-train (44%)

    See NESTED-CV-PROTOCOL.md for full analysis.
    """

    # Data
    data_dir: Path = field(default_factory=lambda: Path("data/arc"))
    output_dir: Path = field(default_factory=lambda: Path("experiments"))

    # Model
    model_variant: Literal["meshnet_5", "meshnet_16", "meshnet_26"] = "meshnet_26"
    channels: int = 26

    # Cross-validation structure (FROM PAPER)
    num_outer_folds: int = 3
    num_restarts: int = 10
    # num_inner_folds is NOT used for replication - inner folds were only for HP search
    # which was done only on the first outer fold. We use the paper's final HPs.

    # Training hyperparameters (FROM PAPER - these are the optimized values)
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
    # How to aggregate across restarts
    restart_aggregation: Literal["mean", "median", "best"] = "mean"

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
        """Total number of training runs.

        Paper protocol: 3 outer folds x 10 restarts = 30 runs.
        We train on FULL outer-train (no inner fold split).
        """
        return self.num_outer_folds * self.num_restarts

    def get_restart_seed(self, restart: int) -> int:
        """Get seed for specific restart."""
        return self.base_seed + restart
