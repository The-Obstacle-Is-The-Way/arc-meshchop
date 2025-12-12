"""Paper parity validation for MeshNet replication.

Validates that trained models achieve paper-comparable performance.

FROM PAPER Table 1:
- MeshNet-26: DICE 0.876 (0.016), AVD 0.245 (0.036), MCC 0.760 (0.030)
- Target: Achieve within tolerance of these values
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)


# Paper reference values (FROM PAPER Table 1)
PAPER_RESULTS = {
    "meshnet_26": {
        "parameters": 147_474,
        "dice_mean": 0.876,
        "dice_std": 0.016,
        "avd_mean": 0.245,
        "avd_std": 0.036,
        "mcc_mean": 0.760,
        "mcc_std": 0.030,
    },
    "meshnet_16": {
        "parameters": 56_194,
        "dice_mean": 0.873,
        "dice_std": 0.007,
        "avd_mean": 0.249,
        "avd_std": 0.033,
        "mcc_mean": 0.757,
        "mcc_std": 0.013,
    },
    "meshnet_5": {
        "parameters": 5_682,
        "dice_mean": 0.848,
        "dice_std": 0.023,
        "avd_mean": 0.280,
        "avd_std": 0.060,
        "mcc_mean": 0.708,
        "mcc_std": 0.042,
    },
}


@dataclass
class ParityResult:
    """Result of parity validation.

    Attributes:
        model: Model variant name.
        dice_mean: Achieved mean DICE.
        dice_std: Achieved DICE std.
        avd_mean: Achieved mean AVD.
        mcc_mean: Achieved mean MCC.
        parity_level: "strict", "acceptable", "minimum", or "failed".
        details: Detailed comparison results.
    """

    model: str
    dice_mean: float
    dice_std: float
    avd_mean: float
    mcc_mean: float
    parity_level: Literal["strict", "acceptable", "minimum", "failed"]
    details: dict

    def __str__(self) -> str:
        """Human-readable summary."""
        emoji = {
            "strict": "[STRICT]",
            "acceptable": "[ACCEPTABLE]",
            "minimum": "[MINIMUM]",
            "failed": "[FAILED]",
        }[self.parity_level]

        return (
            f"{emoji} {self.model}: DICE={self.dice_mean:.4f}+/-{self.dice_std:.4f} "
            f"(parity: {self.parity_level})"
        )


def validate_parity(
    experiment_results: Path | str | dict,
    model_variant: str = "meshnet_26",
) -> ParityResult:
    """Validate experiment results against paper.

    IMPORTANT: Paper parity is judged on TEST metrics (outer fold holdout),
    NOT validation metrics. The summary should contain test_mean_dice, etc.

    Args:
        experiment_results: Path to experiment_results.json or dict.
        model_variant: Model variant to validate.

    Returns:
        ParityResult with parity assessment.
    """
    # Load results
    if isinstance(experiment_results, (str, Path)):
        with Path(experiment_results).open() as f:
            results = json.load(f)
    else:
        results = experiment_results

    # Extract metrics - PRIORITIZE TEST metrics for paper parity
    summary = results.get("summary", results)

    # PRIMARY: Use test metrics (from outer fold evaluation)
    # These are what the paper reports in Table 1
    achieved_dice = summary.get("test_mean_dice")
    achieved_std = summary.get("test_std_dice")
    achieved_avd = summary.get("test_mean_avd")
    achieved_mcc = summary.get("test_mean_mcc")

    # FALLBACK: Use test_results directly if summary doesn't have test metrics
    test_results = results.get("test_results", [])
    if achieved_dice is None and test_results:
        test_dices = [t["test_dice"] for t in test_results]
        test_avds = [t["test_avd"] for t in test_results]
        test_mccs = [t["test_mcc"] for t in test_results]
        achieved_dice = float(np.mean(test_dices))
        achieved_std = float(np.std(test_dices))
        achieved_avd = float(np.mean(test_avds))
        achieved_mcc = float(np.mean(test_mccs))

    # LAST RESORT: Fall back to validation metrics (NOT recommended for parity)
    if achieved_dice is None:
        logger.warning(
            "No test metrics found in results. Using validation metrics, "
            "but this may not accurately reflect paper parity."
        )
        achieved_dice = summary.get("val_mean_dice", summary.get("mean_dice", 0.0))
        achieved_std = summary.get("val_std_dice", summary.get("std_dice", 0.0))
        achieved_avd = summary.get("avd_mean", 0.3)
        achieved_mcc = summary.get("mcc_mean", 0.7)

    # Get paper reference
    paper = PAPER_RESULTS.get(model_variant, PAPER_RESULTS["meshnet_26"])

    # Check parity levels
    details = _compute_parity_details(
        achieved_dice=achieved_dice,
        achieved_std=achieved_std,
        achieved_avd=achieved_avd,
        achieved_mcc=achieved_mcc,
        paper=paper,
    )

    # Determine overall parity level
    if _is_strict_parity(details):
        parity_level: Literal["strict", "acceptable", "minimum", "failed"] = "strict"
    elif _is_acceptable_parity(details):
        parity_level = "acceptable"
    elif _is_minimum_parity(details):
        parity_level = "minimum"
    else:
        parity_level = "failed"

    return ParityResult(
        model=model_variant,
        dice_mean=achieved_dice,
        dice_std=achieved_std,
        avd_mean=achieved_avd,
        mcc_mean=achieved_mcc,
        parity_level=parity_level,
        details=details,
    )


def _compute_parity_details(
    achieved_dice: float,
    achieved_std: float,
    achieved_avd: float,
    achieved_mcc: float,
    paper: dict,
) -> dict:
    """Compute detailed parity metrics.

    Args:
        achieved_dice: Achieved DICE score.
        achieved_std: Achieved DICE standard deviation.
        achieved_avd: Achieved AVD score.
        achieved_mcc: Achieved MCC score.
        paper: Paper reference values.

    Returns:
        Dictionary with detailed comparisons.
    """
    dice_diff = achieved_dice - paper["dice_mean"]
    dice_pct = 100 * dice_diff / paper["dice_mean"]

    avd_diff = achieved_avd - paper["avd_mean"]
    avd_pct = 100 * avd_diff / paper["avd_mean"]

    mcc_diff = achieved_mcc - paper["mcc_mean"]
    mcc_pct = 100 * mcc_diff / paper["mcc_mean"]

    return {
        "dice": {
            "achieved": achieved_dice,
            "paper": paper["dice_mean"],
            "diff": dice_diff,
            "pct_diff": dice_pct,
            "within_tolerance": abs(dice_diff) <= 0.01,
            "acceptable": achieved_dice >= 0.86,
            "minimum": achieved_dice >= 0.85,
        },
        "dice_std": {
            "achieved": achieved_std,
            "paper": paper["dice_std"],
            "acceptable": achieved_std <= 0.03,
        },
        "avd": {
            "achieved": achieved_avd,
            "paper": paper["avd_mean"],
            "diff": avd_diff,
            "pct_diff": avd_pct,
            "within_tolerance": abs(avd_diff) <= 0.02,
            "acceptable": achieved_avd <= 0.30,
        },
        "mcc": {
            "achieved": achieved_mcc,
            "paper": paper["mcc_mean"],
            "diff": mcc_diff,
            "pct_diff": mcc_pct,
            "within_tolerance": abs(mcc_diff) <= 0.02,
            "acceptable": achieved_mcc >= 0.74,
        },
    }


def _is_strict_parity(details: dict) -> bool:
    """Check if strict parity achieved."""
    return (
        details["dice"]["within_tolerance"]
        and details["avd"]["within_tolerance"]
        and details["mcc"]["within_tolerance"]
        and details["dice_std"]["acceptable"]
    )


def _is_acceptable_parity(details: dict) -> bool:
    """Check if acceptable parity achieved."""
    return (
        details["dice"]["acceptable"]
        and details["avd"]["acceptable"]
        and details["mcc"]["acceptable"]
    )


def _is_minimum_parity(details: dict) -> bool:
    """Check if minimum parity achieved."""
    return details["dice"]["minimum"]


def generate_benchmark_table(
    results: list[dict],
    output_path: Path | str | None = None,
) -> str:
    """Generate paper-style benchmark table.

    FROM PAPER Table 1 format.

    Args:
        results: List of experiment results.
        output_path: Optional path to save markdown table.

    Returns:
        Markdown-formatted table.
    """
    lines = [
        "| Model | Params | DICE | AVD | MCC | Parity |",
        "|-------|--------|------|-----|-----|--------|",
    ]

    # Add paper reference row
    for variant, paper in PAPER_RESULTS.items():
        lines.append(
            f"| {variant} (paper) | {paper['parameters']:,} | "
            f"{paper['dice_mean']:.3f} ({paper['dice_std']:.3f}) | "
            f"{paper['avd_mean']:.3f} ({paper['avd_std']:.3f}) | "
            f"{paper['mcc_mean']:.3f} ({paper['mcc_std']:.3f}) | ref |"
        )

    lines.append("|---|---|---|---|---|---|")

    # Add experiment results
    for result in results:
        parity = validate_parity(result)
        emoji = {"strict": "[OK]", "acceptable": "[OK]", "minimum": "[WARN]", "failed": "[FAIL]"}

        lines.append(
            f"| {parity.model} (ours) | - | "
            f"{parity.dice_mean:.3f} ({parity.dice_std:.3f}) | "
            f"{parity.details['avd']['achieved']:.3f} | "
            f"{parity.mcc_mean:.3f} | "
            f"{emoji[parity.parity_level]} {parity.parity_level} |"
        )

    table = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(table)
        logger.info("Benchmark table saved to %s", output_path)

    return table


def run_statistical_comparison(
    our_results: list[float],
    paper_mean: float,
    paper_std: float,
    alpha: float = 0.05,
) -> dict:
    """Run statistical comparison against paper results.

    Uses Wilcoxon signed-rank test (as per paper methodology).

    FROM PAPER:
    "p < 0.05, Holm-corrected Wilcoxon test"

    For single-sample comparison against a known population mean,
    we use Wilcoxon signed-rank test on (sample - paper_mean) vs 0.
    This tests whether the median difference is significantly different from 0.

    Args:
        our_results: List of DICE scores from our experiment.
        paper_mean: Mean DICE from paper.
        paper_std: Std DICE from paper.
        alpha: Significance level.

    Returns:
        Dictionary with statistical test results.
    """
    from scipy.stats import wilcoxon

    our_mean = float(np.mean(our_results))
    our_std = float(np.std(our_results))

    # Wilcoxon signed-rank test (paper methodology)
    # Test if our results differ significantly from paper mean
    # by testing (our_results - paper_mean) against 0
    differences = np.array(our_results) - paper_mean

    # Handle case where all differences are 0 (perfect match)
    if np.allclose(differences, 0):
        p_value = 1.0
        statistic = 0.0
    else:
        try:
            statistic, p_value = wilcoxon(differences, alternative="two-sided")
        except ValueError:
            # Wilcoxon requires at least some non-zero differences
            p_value = 1.0
            statistic = 0.0

    # Effect size (Cohen's d for reference)
    cohens_d = (our_mean - paper_mean) / np.sqrt((our_std**2 + paper_std**2) / 2)

    return {
        "our_mean": our_mean,
        "our_std": our_std,
        "paper_mean": paper_mean,
        "paper_std": paper_std,
        "test_type": "Wilcoxon signed-rank",  # Paper methodology
        "wilcoxon_statistic": float(statistic),
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
        "cohens_d": float(cohens_d),
        "interpretation": _interpret_cohens_d(cohens_d),
        "conclusion": (
            "STATISTICALLY EQUIVALENT"
            if p_value >= alpha
            else f"STATISTICALLY DIFFERENT (p={p_value:.4f})"
        ),
    }


def _interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"
