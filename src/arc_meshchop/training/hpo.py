"""Hyperparameter optimization with Optuna/ASHA.

FROM PAPER Section 2:
"To optimize hyperparameters of MeshNet, we conducted a hyperparameter search
using Orion, an asynchronous framework for black-box function optimization.
We employed the Asynchronous Successive Halving Algorithm (ASHA)."

This implementation uses Optuna with ASHAMedianPruner for the same effect.
Optuna is more widely used and better maintained than Orion.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
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
        params: Trial parameters from HPO.
        base_config: Optional base config to override.

    Returns:
        Training configuration with trial parameters.
    """
    config = base_config or TrainingConfig()

    # NOTE: div_factor=100 is critical for paper parity
    # FROM PAPER: "starts at 1/100th of the max learning rate"
    return TrainingConfig(
        # From HPO trial
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        background_weight=params["bg_weight"],
        pct_start=params["warmup_pct"],
        max_lr=params["learning_rate"],  # max_lr matches lr for OneCycleLR
        div_factor=100.0,  # FROM PAPER: "starts at 1/100th of max LR"
        # Preserve base config settings
        lesion_weight=config.lesion_weight,
        label_smoothing=config.label_smoothing,
        batch_size=config.batch_size,
        use_fp16=config.use_fp16,
        checkpoint_dir=config.checkpoint_dir,
        random_seed=config.random_seed,
    )


# ===========================================================================
# Optuna-based HPO (preferred, more widely used)
# ===========================================================================


def create_study(
    study_name: str = "meshnet_hpo",
    pruning: bool = True,
    storage: str | None = None,
    direction: str = "maximize",
) -> Any:
    """Create Optuna study for HPO.

    FROM PAPER: Uses ASHA for efficient HPO.
    Optuna's MedianPruner provides similar functionality.

    Args:
        study_name: Name for the study.
        pruning: Enable ASHA-style pruning.
        storage: Optional storage URL (default: in-memory).
        direction: Optimization direction ("maximize" for DICE).

    Returns:
        Optuna Study object.
    """
    try:
        import optuna
    except ImportError as e:
        raise ImportError("Optuna is required for HPO. Install with: pip install optuna") from e

    # Set up pruner (ASHA-style early stopping)
    # Use BasePruner type to allow both MedianPruner and NopPruner
    pruner: optuna.pruners.BasePruner
    if pruning:
        # MedianPruner prunes trials that fall below median at intermediate points
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,  # Don't prune until 5 complete trials
            n_warmup_steps=10,  # Don't prune first 10 epochs
            interval_steps=5,  # Check every 5 epochs
        )
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    return study


def run_hpo_trial(
    trial: Any,  # optuna.Trial
    data_dir: Path | str,
    outer_fold: int = 0,
    epochs: int = 50,
    channels: int = 26,
) -> float:
    """Run single HPO trial with Optuna.

    FROM PAPER Section 2:
    "HPO was conducted on the inner folds of the first outer fold"

    This trains on inner folds of specified outer fold and returns
    the mean validation DICE. Uses epoch-level training with ASHA-style
    pruning at each epoch for efficient HPO.

    Args:
        trial: Optuna Trial object.
        data_dir: Path to dataset info.
        outer_fold: Outer fold for HPO (paper uses 0).
        epochs: Number of epochs per trial.
        channels: Number of channels for MeshNet.

    Returns:
        Mean validation DICE (to maximize).
    """
    import torch

    from arc_meshchop.data import (
        ARCDataset,
        create_dataloaders,
        create_stratification_labels,
        generate_nested_cv_splits,
        get_lesion_quintile,
    )
    from arc_meshchop.models import MeshNet
    from arc_meshchop.training import Trainer, TrainingConfig
    from arc_meshchop.utils.device import get_device

    # Suggest hyperparameters (FROM PAPER search space)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 4e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 4e-2, log=True)
    bg_weight = trial.suggest_float("bg_weight", 0.0, 1.0)
    warmup_pct = trial.suggest_categorical("warmup_pct", [0.02, 0.1, 0.2])

    logger.info(
        "Trial %d: lr=%.6f, wd=%.6f, bg=%.3f, warmup=%.2f",
        trial.number,
        learning_rate,
        weight_decay,
        bg_weight,
        warmup_pct,
    )

    # Load dataset
    data_dir = Path(data_dir)
    info_path = data_dir / "dataset_info.json"
    with info_path.open() as f:
        dataset_info = json.load(f)

    image_paths = [Path(p) for p in dataset_info["image_paths"]]
    mask_paths = [Path(p) for p in dataset_info["mask_paths"]]
    lesion_volumes = dataset_info["lesion_volumes"]
    acquisition_types = dataset_info["acquisition_types"]

    # Generate CV splits
    quintiles = [get_lesion_quintile(v) for v in lesion_volumes]
    strat_labels = create_stratification_labels(quintiles, acquisition_types)

    splits = generate_nested_cv_splits(
        n_samples=len(image_paths),
        stratification_labels=strat_labels,
        num_outer_folds=3,
        num_inner_folds=3,
        random_seed=42,
    )

    # Train on each inner fold with epoch-level pruning
    inner_fold_dices = []
    device = get_device()

    for inner_fold in range(3):
        split = splits.get_split(outer_fold, inner_fold)

        train_dataset = ARCDataset(
            image_paths=[image_paths[i] for i in split.train_indices],
            mask_paths=[mask_paths[i] for i in split.train_indices],
        )
        val_dataset = ARCDataset(
            image_paths=[image_paths[i] for i in split.val_indices],
            mask_paths=[mask_paths[i] for i in split.val_indices],
        )

        train_loader, val_loader = create_dataloaders(
            train_dataset,
            val_dataset,
            batch_size=1,
            num_workers=4,
        )

        # Create model
        model = MeshNet(channels=channels)

        # Training config
        # NOTE: div_factor=100 is critical for paper parity
        # FROM PAPER: "starts at 1/100th of the max learning rate"
        config = TrainingConfig(
            epochs=epochs,
            learning_rate=learning_rate,
            max_lr=learning_rate,
            weight_decay=weight_decay,
            background_weight=bg_weight,
            pct_start=warmup_pct,
            div_factor=100.0,  # FROM PAPER: "starts at 1/100th of max LR"
            checkpoint_dir=Path(f"/tmp/hpo_trial_{trial.number}_fold_{inner_fold}"),
        )

        trainer = Trainer(model, config, device=device)

        # Set up scheduler for epoch-level training
        trainer.setup_scheduler(train_loader)

        # Epoch-level training with pruning (ASHA-style)
        # This allows pruning at epoch granularity, not just after full training
        best_dice = 0.0
        for epoch in range(epochs):
            # Train one epoch
            trainer.train_epoch(train_loader)

            # Validate and get DICE
            val_metrics = trainer.validate(val_loader)
            val_dice = val_metrics.get("dice", 0.0)

            if val_dice > best_dice:
                best_dice = val_dice

            # Report intermediate value for ASHA-style pruning
            # Step is (inner_fold * epochs + epoch) to give unique step across folds
            step = inner_fold * epochs + epoch
            trial.report(best_dice, step)

            # Check for pruning at each epoch
            if trial.should_prune():
                try:
                    import optuna
                except ImportError:
                    pass
                else:
                    raise optuna.TrialPruned()

        inner_fold_dices.append(best_dice)

        # Clear CUDA cache between folds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_dice = sum(inner_fold_dices) / len(inner_fold_dices)
    logger.info("Trial %d: Mean validation DICE = %.4f", trial.number, mean_dice)

    return mean_dice


# ===========================================================================
# Orion-based HPO (original paper method, kept for compatibility)
# ===========================================================================


def run_hpo(
    hpo_config: HPOConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    experiment_name: str = "meshnet_hpo",
    max_trials: int = 100,
) -> dict[str, Any]:
    """Run hyperparameter optimization using Orion.

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

            # Map Orion param names to our expected format
            params = {
                "learning_rate": trial.params["lr"],
                "weight_decay": trial.params["weight_decay"],
                "bg_weight": trial.params["bg_weight"],
                "warmup_pct": trial.params["warmup_pct"],
            }
            train_config = trial_params_to_training_config(params)
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
                best_params = params

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
