"""Tests for statistical testing."""

import numpy as np
import pytest

from arc_meshchop.evaluation.statistics import (
    ComparisonResult,
    compare_models_to_reference,
    format_results_table,
    holm_bonferroni_correction,
    wilcoxon_test,
)


class TestWilcoxonTest:
    """Tests for Wilcoxon signed-rank test."""

    def test_identical_samples(self) -> None:
        """Verify p = 1 for identical samples."""
        scores1 = np.array([0.8, 0.85, 0.9, 0.82, 0.88])
        scores2 = np.array([0.8, 0.85, 0.9, 0.82, 0.88])

        p = wilcoxon_test(scores1, scores2)
        assert p == 1.0

    def test_different_samples(self) -> None:
        """Verify p < 1 for different samples."""
        np.random.seed(42)
        scores1 = np.random.normal(0.8, 0.05, 50)
        scores2 = np.random.normal(0.7, 0.05, 50)  # Different mean

        p = wilcoxon_test(scores1, scores2)
        assert p < 0.05  # Should be significant

    def test_similar_samples(self) -> None:
        """Verify p > 0.05 for similar samples."""
        np.random.seed(42)
        scores1 = np.random.normal(0.8, 0.05, 50)
        scores2 = np.random.normal(0.8, 0.05, 50)  # Same mean

        p = wilcoxon_test(scores1, scores2)
        # With same mean distributions and fixed seed, should not be significant
        assert p > 0.05

    def test_returns_float(self) -> None:
        """Verify returns float p-value."""
        scores1 = np.array([0.8, 0.85, 0.9])
        scores2 = np.array([0.7, 0.75, 0.8])

        p = wilcoxon_test(scores1, scores2)
        assert isinstance(p, float)


class TestHolmBonferroniCorrection:
    """Tests for Holm-Bonferroni correction."""

    def test_single_p_value(self) -> None:
        """Test with single p-value."""
        corrected, significant = holm_bonferroni_correction([0.03])

        assert len(corrected) == 1
        assert corrected[0] == pytest.approx(0.03)
        assert significant[0]  # True (avoid is for numpy bool)

    def test_multiple_p_values(self) -> None:
        """Test with multiple p-values."""
        p_values = [0.01, 0.03, 0.05, 0.1]
        corrected, significant = holm_bonferroni_correction(p_values)

        assert len(corrected) == 4
        assert len(significant) == 4

        # First (smallest) should be multiplied by 4
        assert corrected[0] == pytest.approx(0.04)

    def test_no_significant(self) -> None:
        """Test when no p-values are significant after correction."""
        p_values = [0.3, 0.4, 0.5]
        _, significant = holm_bonferroni_correction(p_values)

        assert not any(significant)

    def test_all_significant(self) -> None:
        """Test when all p-values are significant."""
        p_values = [0.001, 0.002, 0.003]
        _, significant = holm_bonferroni_correction(p_values)

        assert all(significant)

    def test_corrected_values_capped_at_one(self) -> None:
        """Verify corrected p-values don't exceed 1.0."""
        p_values = [0.5, 0.6, 0.7]
        corrected, _ = holm_bonferroni_correction(p_values)

        for p in corrected:
            assert p <= 1.0


class TestComparisonResult:
    """Tests for ComparisonResult dataclass."""

    def test_creates_with_all_fields(self) -> None:
        """Verify can create with all fields."""
        result = ComparisonResult(
            model_name="ModelA",
            p_value=0.03,
            corrected_p_value=0.06,
            is_significant=False,
            effect_size=0.5,
        )

        assert result.model_name == "ModelA"
        assert result.p_value == 0.03
        assert result.corrected_p_value == 0.06
        assert result.is_significant is False
        assert result.effect_size == 0.5

    def test_effect_size_optional(self) -> None:
        """Verify effect_size is optional."""
        result = ComparisonResult(
            model_name="ModelA",
            p_value=0.03,
            corrected_p_value=0.06,
            is_significant=True,
        )

        assert result.effect_size is None


class TestCompareModelsToReference:
    """Tests for model comparison pipeline."""

    def test_returns_results_for_all_models(self) -> None:
        """Verify returns results for all comparison models."""
        reference = np.array([0.8, 0.85, 0.9, 0.82, 0.88])
        models = {
            "model_a": np.array([0.7, 0.75, 0.8, 0.72, 0.78]),
            "model_b": np.array([0.79, 0.84, 0.89, 0.81, 0.87]),
        }

        results = compare_models_to_reference(reference, models)

        assert len(results) == 2
        assert results[0].model_name in ["model_a", "model_b"]
        assert results[1].model_name in ["model_a", "model_b"]

    def test_identifies_significant_differences(self) -> None:
        """Verify identifies significantly different models."""
        np.random.seed(42)
        reference = np.random.normal(0.85, 0.02, 50)
        models = {
            "much_worse": np.random.normal(0.70, 0.02, 50),  # Clearly worse
            "similar": np.random.normal(0.84, 0.02, 50),  # Similar
        }

        results = compare_models_to_reference(reference, models)

        results_dict = {r.model_name: r for r in results}
        assert results_dict["much_worse"].is_significant

    def test_returns_comparison_result_objects(self) -> None:
        """Verify returns ComparisonResult objects."""
        reference = np.array([0.8, 0.85, 0.9])
        models = {"model_a": np.array([0.7, 0.75, 0.8])}

        results = compare_models_to_reference(reference, models)

        assert all(isinstance(r, ComparisonResult) for r in results)

    def test_custom_alpha(self) -> None:
        """Verify custom alpha level is respected."""
        np.random.seed(42)
        reference = np.random.normal(0.85, 0.05, 20)
        models = {"model_a": np.random.normal(0.80, 0.05, 20)}

        # More strict alpha
        results_strict = compare_models_to_reference(reference, models, alpha=0.01)

        # Less strict alpha
        results_lenient = compare_models_to_reference(reference, models, alpha=0.10)

        # With stricter alpha, fewer things should be significant
        # (or same, but never more)
        assert sum(r.is_significant for r in results_strict) <= sum(
            r.is_significant for r in results_lenient
        )


class TestFormatResultsTable:
    """Tests for results table formatting."""

    def test_basic_formatting(self) -> None:
        """Test basic table formatting."""
        results = {
            "MeshNet-26": {"dice": 0.876, "avd": 0.245, "mcc": 0.760},
            "MeshNet-16": {"dice": 0.850, "avd": 0.300, "mcc": 0.720},
        }

        table = format_results_table(results)

        assert "MeshNet-26" in table
        assert "MeshNet-16" in table
        assert "0.876" in table
        assert "|" in table

    def test_with_comparisons(self) -> None:
        """Test formatting with statistical comparisons."""
        results = {
            "MeshNet-26": {"dice": 0.876, "avd": 0.245, "mcc": 0.760},
            "MeshNet-16": {"dice": 0.850, "avd": 0.300, "mcc": 0.720},
        }
        comparisons = [
            ComparisonResult(
                model_name="MeshNet-16",
                p_value=0.01,
                corrected_p_value=0.01,
                is_significant=True,
            ),
        ]

        table = format_results_table(results, comparisons)

        assert "*" in table  # Significance marker

    def test_markdown_header(self) -> None:
        """Verify includes markdown table header."""
        results = {"Model": {"dice": 0.8, "avd": 0.3, "mcc": 0.7}}

        table = format_results_table(results)

        assert "| Model |" in table
        assert "|-------|" in table
