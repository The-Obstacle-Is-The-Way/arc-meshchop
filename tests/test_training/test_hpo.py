"""Tests for HPO helpers.

Tests cover:
- get_orion_search_space: Verifies search space matches paper
- trial_params_to_training_config: Verifies config conversion
- HPOTrial dataclass
"""

from typing import Any

import pytest

from arc_meshchop.training.config import HPOConfig, TrainingConfig
from arc_meshchop.training.hpo import (
    HPOTrial,
    get_orion_search_space,
    trial_params_to_training_config,
)


class TestGetOrionSearchSpace:
    """Tests for get_orion_search_space."""

    def test_returns_dict(self) -> None:
        """Verify returns dictionary."""
        config = HPOConfig()
        space = get_orion_search_space(config)

        assert isinstance(space, dict)

    def test_contains_all_paper_params(self) -> None:
        """Verify all paper parameters are in search space.

        FROM PAPER Section 2:
        - channels: uniform(5, 21)
        - lr: loguniform(1e-4, 4e-2)
        - weight_decay: loguniform(1e-4, 4e-2)
        - bg_weight: uniform(0, 1)
        - warmup_pct: choices([0.02, 0.1, 0.2])
        - epochs: fidelity([15, 50])
        """
        config = HPOConfig()
        space = get_orion_search_space(config)

        assert "channels" in space
        assert "lr" in space
        assert "weight_decay" in space
        assert "bg_weight" in space
        assert "warmup_pct" in space
        assert "epochs" in space  # Fidelity dimension

    def test_channels_uniform_discrete(self) -> None:
        """Verify channels uses uniform discrete distribution."""
        config = HPOConfig()
        space = get_orion_search_space(config)

        assert "uniform" in space["channels"]
        assert "5" in space["channels"]
        assert "21" in space["channels"]
        assert "discrete=True" in space["channels"]

    def test_lr_loguniform(self) -> None:
        """Verify lr uses log-uniform distribution."""
        config = HPOConfig()
        space = get_orion_search_space(config)

        assert "loguniform" in space["lr"]
        assert "0.0001" in space["lr"] or "1e-4" in space["lr"].lower()
        assert "0.04" in space["lr"] or "4e-2" in space["lr"].lower()

    def test_weight_decay_loguniform(self) -> None:
        """Verify weight_decay uses log-uniform distribution."""
        config = HPOConfig()
        space = get_orion_search_space(config)

        assert "loguniform" in space["weight_decay"]

    def test_bg_weight_uniform(self) -> None:
        """Verify bg_weight uses uniform distribution."""
        config = HPOConfig()
        space = get_orion_search_space(config)

        assert "uniform" in space["bg_weight"]
        assert "0" in space["bg_weight"]
        assert "1" in space["bg_weight"]

    def test_warmup_choices(self) -> None:
        """Verify warmup uses categorical choices."""
        config = HPOConfig()
        space = get_orion_search_space(config)

        assert "choices" in space["warmup_pct"]
        assert "0.02" in space["warmup_pct"]
        assert "0.1" in space["warmup_pct"]
        assert "0.2" in space["warmup_pct"]

    def test_epochs_fidelity(self) -> None:
        """Verify epochs uses fidelity distribution.

        FROM PAPER Section 2:
        "Epochs (fidelity) | Fidelity | [15, 50]"
        """
        config = HPOConfig()
        space = get_orion_search_space(config)

        assert "fidelity" in space["epochs"]
        assert "15" in space["epochs"]
        assert "50" in space["epochs"]

    def test_respects_config_values(self) -> None:
        """Verify search space respects custom config."""
        config = HPOConfig(
            channels_min=10,
            channels_max=30,
            lr_min=1e-5,
            lr_max=1e-1,
        )
        space = get_orion_search_space(config)

        assert "10" in space["channels"]
        assert "30" in space["channels"]
        assert "1e-05" in space["lr"] or "1e-5" in space["lr"].lower()


class TestTrialParamsToTrainingConfig:
    """Tests for trial_params_to_training_config."""

    @pytest.fixture
    def sample_trial_params(self) -> dict[str, Any]:
        """Sample trial parameters from HPO."""
        return {
            "channels": 16,
            "lr": 0.005,
            "weight_decay": 1e-4,
            "bg_weight": 0.3,
            "warmup_pct": 0.1,
            "epochs": 30,
        }

    def test_returns_training_config(
        self,
        sample_trial_params: dict[str, Any],
    ) -> None:
        """Verify returns TrainingConfig instance."""
        config = trial_params_to_training_config(sample_trial_params)

        assert isinstance(config, TrainingConfig)

    def test_sets_learning_rate(
        self,
        sample_trial_params: dict[str, Any],
    ) -> None:
        """Verify learning rate is set from trial params."""
        config = trial_params_to_training_config(sample_trial_params)

        assert config.learning_rate == pytest.approx(0.005)

    def test_sets_max_lr_same_as_lr(
        self,
        sample_trial_params: dict[str, Any],
    ) -> None:
        """Verify max_lr matches lr for OneCycleLR."""
        config = trial_params_to_training_config(sample_trial_params)

        assert config.max_lr == pytest.approx(0.005)
        assert config.max_lr == config.learning_rate

    def test_sets_weight_decay(
        self,
        sample_trial_params: dict[str, Any],
    ) -> None:
        """Verify weight decay is set from trial params."""
        config = trial_params_to_training_config(sample_trial_params)

        assert config.weight_decay == pytest.approx(1e-4)

    def test_sets_background_weight(
        self,
        sample_trial_params: dict[str, Any],
    ) -> None:
        """Verify background weight is set from trial params."""
        config = trial_params_to_training_config(sample_trial_params)

        assert config.background_weight == pytest.approx(0.3)

    def test_sets_pct_start(
        self,
        sample_trial_params: dict[str, Any],
    ) -> None:
        """Verify warmup percentage is set from trial params."""
        config = trial_params_to_training_config(sample_trial_params)

        assert config.pct_start == pytest.approx(0.1)

    def test_preserves_base_config_lesion_weight(
        self,
        sample_trial_params: dict[str, Any],
    ) -> None:
        """Verify lesion weight is preserved from base config."""
        base = TrainingConfig(lesion_weight=2.0)
        config = trial_params_to_training_config(sample_trial_params, base)

        assert config.lesion_weight == pytest.approx(2.0)

    def test_preserves_base_config_label_smoothing(
        self,
        sample_trial_params: dict[str, Any],
    ) -> None:
        """Verify label smoothing is preserved from base config."""
        base = TrainingConfig(label_smoothing=0.05)
        config = trial_params_to_training_config(sample_trial_params, base)

        assert config.label_smoothing == pytest.approx(0.05)

    def test_uses_defaults_without_base_config(
        self,
        sample_trial_params: dict[str, Any],
    ) -> None:
        """Verify uses TrainingConfig defaults without base config."""
        config = trial_params_to_training_config(sample_trial_params)

        # These should be paper defaults
        assert config.lesion_weight == pytest.approx(1.0)
        assert config.label_smoothing == pytest.approx(0.01)


class TestHPOTrial:
    """Tests for HPOTrial dataclass."""

    def test_creates_with_required_fields(self) -> None:
        """Verify can create with required fields."""
        trial = HPOTrial(
            trial_id="trial_001",
            params={"lr": 0.001},
        )

        assert trial.trial_id == "trial_001"
        assert trial.params == {"lr": 0.001}

    def test_default_dice_score_is_none(self) -> None:
        """Verify default dice score is None."""
        trial = HPOTrial(
            trial_id="trial_001",
            params={},
        )

        assert trial.dice_score is None

    def test_default_status_is_pending(self) -> None:
        """Verify default status is pending."""
        trial = HPOTrial(
            trial_id="trial_001",
            params={},
        )

        assert trial.status == "pending"

    def test_can_set_dice_score(self) -> None:
        """Verify can set dice score."""
        trial = HPOTrial(
            trial_id="trial_001",
            params={},
            dice_score=0.876,
        )

        assert trial.dice_score == pytest.approx(0.876)

    def test_can_set_status(self) -> None:
        """Verify can set status."""
        trial = HPOTrial(
            trial_id="trial_001",
            params={},
            status="completed",
        )

        assert trial.status == "completed"
