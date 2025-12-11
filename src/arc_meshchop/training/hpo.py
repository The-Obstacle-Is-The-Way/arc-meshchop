"""Hyperparameter optimization with Orion/ASHA.

FROM PAPER Section 2:
"To optimize hyperparameters of MeshNet, we conducted a hyperparameter search
using Orion, an asynchronous framework for black-box function optimization.
We employed the Asynchronous Successive Halving Algorithm (ASHA)."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from arc_meshchop.training.config import HPOConfig, TrainingConfig

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class HPOTrial:
    """Single hyperparameter optimization trial."""

    trial_id: str
    params: dict[str, Any]
    dice_score: float | None = None
    status: str = "pending"


def get_orion_search_space(config: HPOConfig) -> dict[str, str]:
    """Get Orion search space definition.

    FROM PAPER Section 2:
    - channels: uniform(5, 21)
    - lr: loguniform(1e-4, 4e-2)
    - weight_decay: loguniform(1e-4, 4e-2)
    - bg_weight: uniform(0, 1)
    - warmup_pct: choices([0.02, 0.1, 0.2])
    - epochs: fidelity(15, 50)

    Args:
        config: HPO configuration.

    Returns:
        Orion-compatible search space dictionary.
    """
    return {
        "channels": f"uniform({config.channels_min}, {config.channels_max}, discrete=True)",
        "lr": f"loguniform({config.lr_min}, {config.lr_max})",
        "weight_decay": f"loguniform({config.weight_decay_min}, {config.weight_decay_max})",
        "bg_weight": f"uniform({config.bg_weight_min}, {config.bg_weight_max})",
        "warmup_pct": f"choices({list(config.warmup_pct_choices)})",
        "epochs": f"fidelity({config.min_epochs}, {config.max_epochs}, base=2)",
    }


def trial_params_to_training_config(
    params: dict[str, Any],
    base_config: TrainingConfig | None = None,
) -> TrainingConfig:
    """Convert trial parameters to TrainingConfig.

    Args:
        params: Trial parameters from Orion.
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
    experiment_name: str = "meshnet_hpo",
    max_trials: int = 100,
) -> dict[str, Any]:
    """Run hyperparameter optimization.

    Uses Orion with ASHA for efficient HPO.

    Args:
        hpo_config: HPO configuration.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        experiment_name: Name for the experiment.
        max_trials: Maximum number of trials.

    Returns:
        Best parameters found.
    """
    try:
        from orion.client import create_experiment  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError("Orion is required for HPO. Install with: pip install orion") from e

    from arc_meshchop.models import MeshNet
    from arc_meshchop.training.trainer import Trainer

    # Create Orion experiment
    experiment = create_experiment(
        name=experiment_name,
        space=get_orion_search_space(hpo_config),
        algorithms={
            "asha": {
                "seed": hpo_config.seed,
                "num_rungs": hpo_config.num_rungs,
                "num_brackets": hpo_config.num_brackets,
            }
        },
    )

    logger.info("Starting HPO with max %d trials", max_trials)

    best_dice = 0.0
    best_params: dict[str, Any] = {}

    trial_count = 0
    while not experiment.is_done and trial_count < max_trials:
        trial = experiment.suggest()
        if trial is None:
            break

        trial_count += 1
        logger.info("Trial %d: %s", trial_count, trial.params)

        try:
            # Create model with trial channels
            model = MeshNet(channels=trial.params["channels"])

            # Create training config from trial params
            epochs = trial.params.get("epochs", hpo_config.max_epochs)
            train_config = trial_params_to_training_config(trial.params)
            train_config.epochs = epochs

            # Train
            trainer = Trainer(model, train_config)
            results = trainer.train(train_loader, val_loader)

            dice_score = results["best_dice"]
            logger.info("Trial %d: DICE = %.4f", trial_count, dice_score)

            # Report to Orion (minimize negative DICE)
            experiment.observe(
                trial,
                [{"name": "dice", "type": "objective", "value": -dice_score}],
            )

            # Track best
            if dice_score > best_dice:
                best_dice = dice_score
                best_params = dict(trial.params)

        except Exception as e:
            logger.error("Trial %d failed: %s", trial_count, e)
            # Report worst possible value (1.0) since we minimize -dice
            # A failed trial should never be selected over a successful one
            experiment.observe(
                trial,
                [{"name": "dice", "type": "objective", "value": 1.0}],
            )

    logger.info("HPO complete. Best DICE: %.4f", best_dice)
    logger.info("Best params: %s", best_params)

    return {"best_dice": best_dice, "best_params": best_params}
