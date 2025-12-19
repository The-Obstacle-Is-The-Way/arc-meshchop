"""Hyperparameter optimization with Optuna/ASHA.

FROM PAPER Section 2:
"To optimize hyperparameters of MeshNet, we conducted a hyperparameter search
using Orion, an asynchronous framework for black-box function optimization.
We employed the Asynchronous Successive Halving Algorithm (ASHA)."

This implementation uses Optuna with SuccessiveHalvingPruner for the same effect.
Optuna is more widely used and better maintained than Orion.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

from arc_meshchop.training.config import HPOConfig, TrainingConfig

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


class FoldContext(TypedDict):
    """Context for a single fold during HPO."""

    trainer: Any  # Avoid circular import issues in some environments
    train_loader: Any
    val_loader: Any
    best_dice: float
    history: list[float]


logger = logging.getLogger(__name__)


@dataclass
class HPOTrial:
    """Single hyperparameter optimization trial."""

    trial_id: str
    params: dict[str, Any]
    dice_score: float | None = None
    status: str = "pending"


def get_search_space_strings(config: HPOConfig) -> dict[str, str]:
    """Get search space as string definitions (legacy Orion format).

    DEPRECATED: This function is kept for compatibility with old Orion code.
    For new code, use Optuna's trial.suggest_* methods directly in run_hpo_trial().

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
        String-based search space dictionary.
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
    seed: int | None = 42,
) -> Any:
    """Create Optuna study for HPO.

    FROM PAPER: Uses ASHA for efficient HPO.
    Optuna's SuccessiveHalvingPruner provides ASHA functionality.

    Args:
        study_name: Name for the study.
        pruning: Enable ASHA-style pruning.
        storage: Optional storage URL (default: in-memory).
        direction: Optimization direction ("maximize" for DICE).
        seed: Optional sampler seed for reproducible HPO.

    Returns:
        Optuna Study object.
    """
    try:
        import optuna
    except ImportError as e:
        raise ImportError("Optuna is required for HPO. Install with: pip install optuna") from e

    # Set up pruner (ASHA-style early stopping)
    # Use BasePruner type to allow both SuccessiveHalvingPruner and NopPruner
    pruner: optuna.pruners.BasePruner
    if pruning:
        # ASHA via SuccessiveHalvingPruner
        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=1,  # Minimum epochs before pruning
            reduction_factor=3,  # Keep top 1/3 at each rung
            min_early_stopping_rate=0,
        )
    else:
        pruner = optuna.pruners.NopPruner()

    sampler: optuna.samplers.BaseSampler | None = None
    if seed is not None:
        sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        pruner=pruner,
        sampler=sampler,
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
    seed: int = 42,
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
        seed: Random seed for reproducibility.

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
    from arc_meshchop.utils.seeding import seed_everything

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

    # Set seed for reproducibility (BUG-012)
    # Use a fixed seed so all trials see the same data splits/initialization order
    seed_everything(seed)

    # Load dataset
    data_dir = Path(data_dir)
    info_path = data_dir / "dataset_info.json"
    with info_path.open() as f:
        dataset_info = json.load(f)

    from arc_meshchop.data.huggingface_loader import parse_dataset_info, validate_masks_present
    from arc_meshchop.utils.paths import resolve_dataset_path

    (
        image_paths_raw,
        mask_paths_raw,
        lesion_volumes,
        acquisition_types,
        _subject_ids,
    ) = parse_dataset_info(
        dataset_info,
        context="Hyperparameter optimization",
    )

    # Resolve paths relative to data_dir (BUG-006 fix)
    data_dir_resolved = data_dir.resolve()
    image_paths = [cast(Path, resolve_dataset_path(data_dir_resolved, p)) for p in image_paths_raw]
    mask_paths_validated = validate_masks_present(
        mask_paths_raw, context="Hyperparameter optimization"
    )
    mask_paths = [
        cast(Path, resolve_dataset_path(data_dir_resolved, p)) for p in mask_paths_validated
    ]

    # Generate CV splits
    quintiles = [get_lesion_quintile(v) for v in lesion_volumes]
    strat_labels = create_stratification_labels(quintiles, acquisition_types)

    splits = generate_nested_cv_splits(
        n_samples=len(image_paths),
        stratification_labels=strat_labels,
        num_outer_folds=3,
        num_inner_folds=3,
        random_seed=seed,
    )

    # Prepare trainers and loaders for all inner folds (BUG-011)
    # We must interleave training to report aggregated metrics per epoch
    fold_contexts: list[FoldContext] = []
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
            seed=seed,  # Deterministic loading
        )

        # Create model
        model = MeshNet(channels=channels)

        # Training config
        config = TrainingConfig(
            epochs=epochs,
            learning_rate=learning_rate,
            max_lr=learning_rate,
            weight_decay=weight_decay,
            background_weight=bg_weight,
            pct_start=warmup_pct,
            div_factor=100.0,
            checkpoint_dir=Path(
                tempfile.mkdtemp(prefix=f"hpo_trial_{trial.number}_fold_{inner_fold}_")
            ),
            random_seed=seed,
        )

        trainer = Trainer(model, config, device=device)
        trainer.setup_scheduler(train_loader)

        fold_contexts.append(
            {
                "trainer": trainer,
                "train_loader": train_loader,
                "val_loader": val_loader,
                "best_dice": 0.0,
                "history": [],
            }
        )

    # Epoch-level training with aggregation (BUG-011)
    mean_dice_history: list[float] = []

    def record_history() -> None:
        if not hasattr(trial, "set_user_attr"):
            return
        fold_history = {str(i): ctx["history"] for i, ctx in enumerate(fold_contexts)}
        trial.set_user_attr("fold_dice_history", fold_history)
        trial.set_user_attr("mean_dice_history", mean_dice_history)

    for epoch in range(epochs):
        fold_dices = []

        # Train one epoch for each fold
        for ctx in fold_contexts:
            trainer = ctx["trainer"]
            train_loader = ctx["train_loader"]
            val_loader = ctx["val_loader"]

            trainer.train_epoch(train_loader)
            val_metrics = trainer.validate(val_loader)
            val_dice = val_metrics.get("dice", 0.0)

            # Track per-fold best (metrics can be noisy)
            if val_dice > ctx["best_dice"]:
                ctx["best_dice"] = val_dice

            ctx["history"].append(val_dice)
            fold_dices.append(val_dice)

        # Aggregate across folds
        mean_dice = sum(fold_dices) / len(fold_dices)
        mean_dice_history.append(mean_dice)

        # Report to Optuna (maximize mean_dice)
        trial.report(mean_dice, epoch)

        # Check for pruning
        if trial.should_prune():
            record_history()
            try:
                import optuna
            except ImportError:
                pass
            else:
                raise optuna.TrialPruned()

        # Optional: Clear cache periodically?
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    record_history()

    mean_best_dice = sum(ctx["best_dice"] for ctx in fold_contexts) / len(fold_contexts)
    logger.info("Trial %d: Mean validation DICE = %.4f", trial.number, mean_best_dice)
    return mean_best_dice


# ===========================================================================
# DEPRECATED: Orion-based HPO (original paper method)
# ===========================================================================
# NOTE: Orion is broken on Python 3.12 (uses deprecated configparser.SafeConfigParser)
# Use the Optuna functions above (create_study, run_hpo_trial) instead.
# This code is kept only for historical reference and will be removed in a future version.


def run_hpo_orion(
    hpo_config: HPOConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    experiment_name: str = "meshnet_hpo",
    max_trials: int = 100,
) -> dict[str, Any]:
    """DEPRECATED: Run hyperparameter optimization using Orion.

    WARNING: Orion is broken on Python 3.12. Use create_study() + run_hpo_trial() instead.

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
        space=get_search_space_strings(hpo_config),
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


# ===========================================================================
# Backward-compatible aliases (DEPRECATED)
# ===========================================================================
# These aliases are kept for backward compatibility with existing code/tests.
# New code should use the renamed functions directly.

get_orion_search_space = get_search_space_strings
"""DEPRECATED: Use get_search_space_strings instead."""

run_hpo = run_hpo_orion
"""DEPRECATED: Use run_hpo_orion (for Orion) or create_study + run_hpo_trial (for Optuna)."""
