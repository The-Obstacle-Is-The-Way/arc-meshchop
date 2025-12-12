"""Training infrastructure for MeshNet stroke lesion segmentation."""

from arc_meshchop.training.config import HPOConfig, TrainingConfig
from arc_meshchop.training.hpo import (
    create_study,
    get_orion_search_space,  # Deprecated: use get_search_space_strings
    get_search_space_strings,
    run_hpo,  # Deprecated: use run_hpo_orion or create_study + run_hpo_trial
    run_hpo_orion,
    run_hpo_trial,
    trial_params_to_training_config,
)
from arc_meshchop.training.loss import (
    WeightedCrossEntropyLoss,
    create_loss_function,
)
from arc_meshchop.training.optimizer import create_optimizer, create_scheduler
from arc_meshchop.training.trainer import Trainer, TrainingState

__all__ = [
    "HPOConfig",
    "Trainer",
    "TrainingConfig",
    "TrainingState",
    "WeightedCrossEntropyLoss",
    "create_loss_function",
    "create_optimizer",
    "create_scheduler",
    "create_study",
    "get_orion_search_space",  # Deprecated alias
    "get_search_space_strings",
    "run_hpo",  # Deprecated alias
    "run_hpo_orion",
    "run_hpo_trial",
    "trial_params_to_training_config",
]
