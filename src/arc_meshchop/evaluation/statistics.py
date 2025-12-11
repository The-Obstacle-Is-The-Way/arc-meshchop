"""Statistical testing for model comparison.

Implements the statistical testing from the paper:
- Wilcoxon signed-rank test (paired, non-parametric)
- Holm-Bonferroni correction for multiple comparisons

FROM PAPER:
"(*) indicate models statistically significantly different from MeshNet-26
(p < 0.05, Holm-corrected Wilcoxon test)"
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class ComparisonResult:
    """Result of statistical comparison between two models."""

    model_name: str
    p_value: float
    corrected_p_value: float
    is_significant: bool
    effect_size: float | None = None


def wilcoxon_test(
    reference_scores: npt.NDArray[np.float64],
    model_scores: npt.NDArray[np.float64],
) -> float:
    """Perform Wilcoxon signed-rank test.

    Non-parametric paired test for comparing two related samples.

    Args:
        reference_scores: Scores from reference model (e.g., MeshNet-26).
        model_scores: Scores from comparison model.

    Returns:
        Two-sided p-value.
    """
    from scipy.stats import wilcoxon

    # Handle identical arrays (would cause warning)
    if np.allclose(reference_scores, model_scores):
        return 1.0

    _, p_value = wilcoxon(reference_scores, model_scores, alternative="two-sided")
    return float(p_value)


def holm_bonferroni_correction(
    p_values: list[float],
    alpha: float = 0.05,
) -> tuple[list[float], list[bool]]:
    """Apply Holm-Bonferroni correction for multiple comparisons.

    Less conservative than Bonferroni, controls family-wise error rate.

    Args:
        p_values: List of uncorrected p-values.
        alpha: Significance level (default 0.05).

    Returns:
        Tuple of (corrected_p_values, is_significant_list).
    """
    # Use statsmodels if available (more complete implementation)
    try:
        from statsmodels.stats.multitest import multipletests  # type: ignore[import-untyped]

        reject, corrected_p, _, _ = multipletests(
            p_values,
            alpha=alpha,
            method="holm",
        )
        return list(corrected_p), list(reject)
    except ImportError:
        pass

    # Fallback: manual Holm-Bonferroni implementation
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    corrected_p = np.zeros(n)
    is_significant = np.zeros(n, dtype=bool)

    for rank, idx in enumerate(sorted_indices):
        # Holm correction: multiply by (n - rank)
        corrected = sorted_p[rank] * (n - rank)
        corrected_p[idx] = min(corrected, 1.0)

        # Check significance with sequential testing
        if rank == 0:
            is_significant[idx] = corrected_p[idx] < alpha
        else:
            # Only significant if previous was also significant
            prev_idx = sorted_indices[rank - 1]
            is_significant[idx] = is_significant[prev_idx] and (corrected_p[idx] < alpha)

    return list(corrected_p), list(is_significant)


def compare_models_to_reference(
    reference_scores: npt.NDArray[np.float64],
    model_scores_dict: dict[str, npt.NDArray[np.float64]],
    alpha: float = 0.05,
) -> list[ComparisonResult]:
    """Compare multiple models to a reference using Wilcoxon + Holm.

    Implements the paper's statistical testing methodology:
    1. Wilcoxon signed-rank test (paired, non-parametric)
    2. Holm-Bonferroni correction for multiple comparisons
    3. Significance at p < 0.05 after correction

    Args:
        reference_scores: Scores from reference model (MeshNet-26).
        model_scores_dict: Dictionary mapping model names to score arrays.
        alpha: Significance level (default 0.05).

    Returns:
        List of ComparisonResult objects.
    """
    model_names = list(model_scores_dict.keys())
    p_values: list[float] = []

    for name in model_names:
        scores = model_scores_dict[name]
        p = wilcoxon_test(reference_scores, scores)
        p_values.append(p)

    # Apply Holm-Bonferroni correction
    corrected_p, is_significant = holm_bonferroni_correction(p_values, alpha)

    results: list[ComparisonResult] = []
    for name, p, cp, sig in zip(model_names, p_values, corrected_p, is_significant, strict=True):
        results.append(
            ComparisonResult(
                model_name=name,
                p_value=p,
                corrected_p_value=cp,
                is_significant=sig,
            )
        )

    return results


def format_results_table(
    results: dict[str, dict[str, float]],
    comparisons: list[ComparisonResult] | None = None,
) -> str:
    """Format results as a markdown table.

    Args:
        results: Dictionary mapping model names to metric dictionaries.
        comparisons: Optional list of statistical comparison results.

    Returns:
        Markdown-formatted table string.
    """
    # Build comparison lookup
    sig_lookup: dict[str, bool] = {}
    if comparisons:
        for comp in comparisons:
            sig_lookup[comp.model_name] = comp.is_significant

    lines = [
        "| Model | DICE (↑) | AVD (↓) | MCC (↑) | Sig? |",
        "|-------|----------|---------|---------|------|",
    ]

    for model_name, metrics in results.items():
        dice = metrics.get("dice", 0.0)
        avd = metrics.get("avd", 0.0)
        mcc = metrics.get("mcc", 0.0)

        sig = "*" if sig_lookup.get(model_name, False) else ""

        lines.append(f"| {model_name} | {dice:.3f} | {avd:.3f} | {mcc:.3f} | {sig} |")

    return "\n".join(lines)
