"""Tests for evaluation pipeline."""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from arc_meshchop.evaluation.evaluator import EvaluationResult, Evaluator
from arc_meshchop.evaluation.metrics import MetricResult
from arc_meshchop.models import meshnet_5


@pytest.fixture
def tiny_eval_loader() -> DataLoader:
    """Create tiny evaluation data loader for tests."""
    # 4 samples, 8³ volumes
    images = torch.randn(4, 1, 8, 8, 8)
    masks = torch.randint(0, 2, (4, 8, 8, 8))
    dataset = TensorDataset(images, masks)
    return DataLoader(dataset, batch_size=2, shuffle=False)


@pytest.fixture
def sample_metric_result() -> MetricResult:
    """Create sample MetricResult for tests."""
    return MetricResult(
        name="DICE",
        mean=0.876,
        std=0.016,
        values=[0.85, 0.87, 0.89, 0.88],
    )


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_to_dict(self, sample_metric_result: MetricResult) -> None:
        """Verify to_dict serialization."""
        result = EvaluationResult(
            model_name="MeshNet-26",
            dice=sample_metric_result,
            avd=sample_metric_result,
            mcc=sample_metric_result,
            num_samples=4,
        )

        d = result.to_dict()

        assert d["model_name"] == "MeshNet-26"
        assert d["num_samples"] == 4
        assert "dice" in d
        dice_dict = d["dice"]
        assert isinstance(dice_dict, dict)
        assert dice_dict["mean"] == pytest.approx(0.876)

    def test_save_and_load(
        self,
        sample_metric_result: MetricResult,
        tmp_path: Path,
    ) -> None:
        """Verify save and load round-trip."""
        result = EvaluationResult(
            model_name="MeshNet-26",
            dice=sample_metric_result,
            avd=sample_metric_result,
            mcc=sample_metric_result,
            num_samples=4,
        )

        path = tmp_path / "results.json"
        result.save(path)

        loaded = EvaluationResult.load(path)

        assert loaded.model_name == result.model_name
        assert loaded.num_samples == result.num_samples
        assert loaded.dice.mean == pytest.approx(result.dice.mean)
        assert loaded.dice.values == result.dice.values

    def test_save_creates_file(
        self,
        sample_metric_result: MetricResult,
        tmp_path: Path,
    ) -> None:
        """Verify save creates JSON file."""
        result = EvaluationResult(
            model_name="MeshNet-26",
            dice=sample_metric_result,
            avd=sample_metric_result,
            mcc=sample_metric_result,
            num_samples=4,
        )

        path = tmp_path / "results.json"
        result.save(path)

        assert path.exists()
        assert path.read_text().startswith("{")


class TestEvaluator:
    """Tests for Evaluator class."""

    def test_initializes(self) -> None:
        """Verify evaluator initializes correctly."""
        evaluator = Evaluator()

        assert evaluator.device is not None
        assert evaluator.metrics_calculator is not None

    def test_initializes_with_device(self) -> None:
        """Verify evaluator accepts custom device."""
        evaluator = Evaluator(device=torch.device("cpu"))

        assert evaluator.device.type == "cpu"

    def test_evaluate_returns_result(
        self,
        tiny_eval_loader: DataLoader,
    ) -> None:
        """Verify evaluate returns EvaluationResult."""
        model = meshnet_5()
        evaluator = Evaluator(device=torch.device("cpu"))

        result = evaluator.evaluate(
            model,
            tiny_eval_loader,
            model_name="test_model",
            use_fp16=False,
        )

        assert isinstance(result, EvaluationResult)
        assert result.model_name == "test_model"

    def test_evaluate_computes_metrics(
        self,
        tiny_eval_loader: DataLoader,
    ) -> None:
        """Verify evaluate computes all three metrics."""
        model = meshnet_5()
        evaluator = Evaluator(device=torch.device("cpu"))

        result = evaluator.evaluate(
            model,
            tiny_eval_loader,
            model_name="test_model",
            use_fp16=False,
        )

        assert result.dice is not None
        assert result.avd is not None
        assert result.mcc is not None

    def test_evaluate_counts_samples(
        self,
        tiny_eval_loader: DataLoader,
    ) -> None:
        """Verify evaluate counts samples correctly."""
        model = meshnet_5()
        evaluator = Evaluator(device=torch.device("cpu"))

        result = evaluator.evaluate(
            model,
            tiny_eval_loader,
            model_name="test_model",
            use_fp16=False,
        )

        # tiny_eval_loader has 4 samples
        assert result.num_samples == 4

    def test_evaluate_metrics_in_valid_range(
        self,
        tiny_eval_loader: DataLoader,
    ) -> None:
        """Verify metrics are in valid ranges."""
        model = meshnet_5()
        evaluator = Evaluator(device=torch.device("cpu"))

        result = evaluator.evaluate(
            model,
            tiny_eval_loader,
            model_name="test_model",
            use_fp16=False,
        )

        # DICE should be in [0, 1]
        assert 0 <= result.dice.mean <= 1
        # AVD should be non-negative
        assert result.avd.mean >= 0
        # MCC should be in [-1, 1]
        assert -1 <= result.mcc.mean <= 1

    def test_evaluate_per_sample_values(
        self,
        tiny_eval_loader: DataLoader,
    ) -> None:
        """Verify per-sample values are recorded."""
        model = meshnet_5()
        evaluator = Evaluator(device=torch.device("cpu"))

        result = evaluator.evaluate(
            model,
            tiny_eval_loader,
            model_name="test_model",
            use_fp16=False,
        )

        # Should have 4 per-sample values
        assert len(result.dice.values) == 4
        assert len(result.avd.values) == 4
        assert len(result.mcc.values) == 4


class TestEvaluatorComparison:
    """Tests for model comparison functionality."""

    def test_invalid_metric_raises(
        self,
        sample_metric_result: MetricResult,
    ) -> None:
        """Verify invalid metric raises ValueError."""
        evaluator = Evaluator(device=torch.device("cpu"))
        result = EvaluationResult(
            model_name="test",
            dice=sample_metric_result,
            avd=sample_metric_result,
            mcc=sample_metric_result,
            num_samples=4,
        )

        with pytest.raises(ValueError, match="Invalid metric"):
            evaluator.compare_to_reference(result, [result], metric="invalid")

    def test_compare_to_reference(
        self,
        tiny_eval_loader: DataLoader,
    ) -> None:
        """Verify compare_to_reference returns results."""
        model = meshnet_5()
        evaluator = Evaluator(device=torch.device("cpu"))

        # Evaluate same model twice (simulates reference vs comparison)
        ref_result = evaluator.evaluate(
            model,
            tiny_eval_loader,
            model_name="reference",
            use_fp16=False,
        )
        other_result = evaluator.evaluate(
            model,
            tiny_eval_loader,
            model_name="other",
            use_fp16=False,
        )

        comparisons = evaluator.compare_to_reference(
            ref_result,
            [other_result],
            metric="dice",
        )

        assert len(comparisons) == 1
        assert comparisons[0].model_name == "other"
        # Same model on same data yields identical scores -> p=1.0
        assert comparisons[0].p_value == 1.0
        assert not comparisons[0].is_significant

    def test_compare_multiple_models(
        self,
        tiny_eval_loader: DataLoader,
    ) -> None:
        """Verify can compare multiple models."""
        model = meshnet_5()
        evaluator = Evaluator(device=torch.device("cpu"))

        ref_result = evaluator.evaluate(
            model,
            tiny_eval_loader,
            model_name="reference",
            use_fp16=False,
        )
        other1 = evaluator.evaluate(
            model,
            tiny_eval_loader,
            model_name="model_a",
            use_fp16=False,
        )
        other2 = evaluator.evaluate(
            model,
            tiny_eval_loader,
            model_name="model_b",
            use_fp16=False,
        )

        comparisons = evaluator.compare_to_reference(
            ref_result,
            [other1, other2],
            metric="dice",
        )

        assert len(comparisons) == 2
        model_names = [c.model_name for c in comparisons]
        assert "model_a" in model_names
        assert "model_b" in model_names
