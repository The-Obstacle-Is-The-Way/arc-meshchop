"""Tests for paper parity validation."""

from __future__ import annotations

import pytest


class TestValidateParity:
    """Tests for validate_parity function."""

    def test_strict_parity(self) -> None:
        """Verify strict parity detection."""
        from arc_meshchop.validation.parity import validate_parity

        results = {
            "summary": {
                "mean_dice": 0.876,
                "std_dice": 0.016,
            },
            "test_results": [
                {"test_dice": 0.876, "test_avd": 0.245, "test_mcc": 0.760},
            ],
        }

        parity = validate_parity(results, "meshnet_26")

        assert parity.parity_level == "strict"

    def test_acceptable_parity(self) -> None:
        """Verify acceptable parity detection."""
        from arc_meshchop.validation.parity import validate_parity

        results = {
            "summary": {
                "mean_dice": 0.865,  # Within acceptable range
                "std_dice": 0.020,
            },
            "test_results": [
                {"test_dice": 0.865, "test_avd": 0.28, "test_mcc": 0.75},
            ],
        }

        parity = validate_parity(results, "meshnet_26")

        assert parity.parity_level == "acceptable"

    def test_minimum_parity(self) -> None:
        """Verify minimum parity detection."""
        from arc_meshchop.validation.parity import validate_parity

        results = {
            "summary": {
                "mean_dice": 0.852,  # Above 0.85 but below 0.86
                "std_dice": 0.025,
            },
            "test_results": [
                {"test_dice": 0.852, "test_avd": 0.35, "test_mcc": 0.70},
            ],
        }

        parity = validate_parity(results, "meshnet_26")

        assert parity.parity_level == "minimum"

    def test_failed_parity(self) -> None:
        """Verify failed parity detection."""
        from arc_meshchop.validation.parity import validate_parity

        results = {
            "summary": {
                "mean_dice": 0.80,  # Below minimum
                "std_dice": 0.05,
            },
            "test_results": [
                {"test_dice": 0.80, "test_avd": 0.50, "test_mcc": 0.60},
            ],
        }

        parity = validate_parity(results, "meshnet_26")

        assert parity.parity_level == "failed"


class TestBenchmarkTable:
    """Tests for benchmark table generation."""

    def test_generates_markdown(self) -> None:
        """Verify markdown table generation."""
        from arc_meshchop.validation.parity import generate_benchmark_table

        results = [{
            "summary": {"mean_dice": 0.87, "std_dice": 0.02},
            "test_results": [
                {"test_dice": 0.87, "test_avd": 0.25, "test_mcc": 0.76},
            ],
        }]

        table = generate_benchmark_table(results)

        assert "| Model |" in table
        assert "DICE" in table
        assert "paper" in table.lower()


class TestStatisticalComparison:
    """Tests for statistical comparison function."""

    def test_wilcoxon_test_equivalent(self) -> None:
        """Verify Wilcoxon test for equivalent results."""
        from arc_meshchop.validation.parity import run_statistical_comparison

        # Results close to paper mean
        our_results = [0.87, 0.88, 0.875, 0.878, 0.872]
        paper_mean = 0.876
        paper_std = 0.016

        result = run_statistical_comparison(our_results, paper_mean, paper_std)

        assert result["test_type"] == "Wilcoxon signed-rank"
        assert "p_value" in result
        assert "cohens_d" in result
        assert "conclusion" in result

    def test_wilcoxon_test_different(self) -> None:
        """Verify Wilcoxon test detects significant difference."""
        from arc_meshchop.validation.parity import run_statistical_comparison

        # Results significantly different from paper mean
        our_results = [0.70, 0.72, 0.71, 0.69, 0.73, 0.70, 0.71, 0.72, 0.69, 0.70]
        paper_mean = 0.876
        paper_std = 0.016

        result = run_statistical_comparison(our_results, paper_mean, paper_std)

        assert result["significant"] is True
        assert "DIFFERENT" in result["conclusion"]
