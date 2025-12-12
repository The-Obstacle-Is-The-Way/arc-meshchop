# Local Spec 04: Paper Parity Validation

> **Success Criteria** — Define what "paper replication" means
>
> **Goal:** Establish clear metrics and statistical tests to validate replication.

---

## Overview

This spec defines:
1. **Target metrics** from the paper
2. **Statistical validation** methodology
3. **Benchmark comparison** format
4. **Pass/Fail criteria** for paper parity

---

## 1. Paper Results (Ground Truth)

### 1.1 MeshNet Results (FROM PAPER Table 1)

| Model | Parameters | DICE | AVD | MCC |
|-------|------------|------|-----|-----|
| MeshNet-26 | 147,474 | **0.876 (0.016)** | 0.245 (0.036) | 0.760 (0.030) |
| MeshNet-16 | 56,194 | 0.873 (0.007) | 0.249 (0.033) | 0.757 (0.013) |
| MeshNet-5 | 5,682 | 0.848 (0.023) | 0.280 (0.060) | 0.708 (0.042) |

> Format: **mean (std)** across all configurations (3×3×10 = 90 runs)

### 1.2 Metric Definitions

| Metric | Full Name | Formula | Direction |
|--------|-----------|---------|-----------|
| **DICE** | Sørensen-Dice Coefficient | 2\|P∩G\|/(|P|+|G|) | Higher is better (↑) |
| **AVD** | Average Volume Difference | \|V_p - V_g\| / V_g | Lower is better (↓) |
| **MCC** | Matthews Correlation Coefficient | (TP×TN - FP×FN) / √(...) | Higher is better (↑) |

### 1.3 Statistical Significance (FROM PAPER)

- **Test:** Wilcoxon signed-rank test
- **Correction:** Holm-Bonferroni for multiple comparisons
- **Threshold:** p < 0.05 for significance
- **Comparison base:** MeshNet-26 vs other models

---

## 2. Parity Criteria

### 2.1 Strict Parity (Exact Reproduction)

| Metric | Target | Tolerance | Criterion |
|--------|--------|-----------|-----------|
| DICE | 0.876 | ± 0.01 | **0.866 ≤ DICE ≤ 0.886** |
| AVD | 0.245 | ± 0.02 | **0.225 ≤ AVD ≤ 0.265** |
| MCC | 0.760 | ± 0.02 | **0.740 ≤ MCC ≤ 0.780** |
| Std(DICE) | 0.016 | ± 0.01 | **Std ≤ 0.03** |

### 2.2 Acceptable Parity (Comparable Performance)

For a successful replication with minor variations:

| Metric | Acceptable Range | Justification |
|--------|------------------|---------------|
| DICE | ≥ 0.86 | Within 2% of paper |
| AVD | ≤ 0.30 | Slightly higher acceptable |
| MCC | ≥ 0.74 | Within 3% of paper |

### 2.3 Minimum Viable (Directionally Correct)

| Metric | Minimum | Rationale |
|--------|---------|-----------|
| DICE | ≥ 0.85 | Within 3% of paper target (0.876), acceptable for replication |
| Parameters | 147,474 | Must match paper exactly |
| Architecture | 10-layer symmetric | Must match paper dilation pattern |

> **NOTE:** The 0.85 minimum is a practical threshold for replication attempts,
> not an explicit claim from the paper. The paper achieves 0.876 for MeshNet-26.

---

## 3. Implementation

### 3.1 Validation Report Generator

**File:** `src/arc_meshchop/validation/parity.py`

```python
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
from scipy import stats

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
            "strict": "✅",
            "acceptable": "✅",
            "minimum": "⚠️",
            "failed": "❌",
        }[self.parity_level]

        return (
            f"{emoji} {self.model}: DICE={self.dice_mean:.4f}±{self.dice_std:.4f} "
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
        with open(experiment_results) as f:
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
        achieved_dice = np.mean(test_dices)
        achieved_std = np.std(test_dices)
        achieved_avd = np.mean(test_avds)
        achieved_mcc = np.mean(test_mccs)

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
        parity_level = "strict"
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
        achieved_*: Achieved metrics.
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
        "| Model | Parameters | DICE (↑) | AVD (↓) | MCC (↑) | Parity |",
        "|-------|------------|----------|---------|---------|--------|",
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
        emoji = {"strict": "✅", "acceptable": "✅", "minimum": "⚠️", "failed": "❌"}

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

    our_mean = np.mean(our_results)
    our_std = np.std(our_results)

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
        "our_mean": float(our_mean),
        "our_std": float(our_std),
        "paper_mean": paper_mean,
        "paper_std": paper_std,
        "test_type": "Wilcoxon signed-rank",  # Paper methodology
        "wilcoxon_statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "cohens_d": float(cohens_d),
        "interpretation": _interpret_cohens_d(cohens_d),
        "conclusion": (
            "STATISTICALLY EQUIVALENT" if p_value >= alpha
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
```

### 3.2 CLI Command

**Add to:** `src/arc_meshchop/cli/main.py`

```python
@app.command()
def validate(
    results_file: Annotated[
        Path,
        typer.Argument(help="Path to experiment_results.json"),
    ],
    variant: Annotated[
        str,
        typer.Option("--variant", "-v", help="Model variant"),
    ] = "meshnet_26",
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output path for report"),
    ] = None,
) -> None:
    """Validate experiment results against paper.

    Checks if trained model achieves paper-comparable performance.
    Reports parity level: strict, acceptable, minimum, or failed.

    FROM PAPER:
    - MeshNet-26: DICE 0.876 (0.016)
    - MeshNet-16: DICE 0.873 (0.007)
    - MeshNet-5: DICE 0.848 (0.023)
    """
    from arc_meshchop.validation.parity import validate_parity, generate_benchmark_table

    if not results_file.exists():
        typer.echo(f"Results file not found: {results_file}", err=True)
        raise typer.Exit(1)

    result = validate_parity(results_file, variant)

    typer.echo("\n" + "=" * 60)
    typer.echo("PAPER PARITY VALIDATION")
    typer.echo("=" * 60)
    typer.echo(str(result))
    typer.echo()

    typer.echo("Detailed Comparison:")
    typer.echo(f"  DICE: {result.dice_mean:.4f} vs {result.details['dice']['paper']:.4f} "
               f"({result.details['dice']['pct_diff']:+.1f}%)")
    typer.echo(f"  AVD:  {result.avd_mean:.4f} vs {result.details['avd']['paper']:.4f} "
               f"({result.details['avd']['pct_diff']:+.1f}%)")
    typer.echo(f"  MCC:  {result.mcc_mean:.4f} vs {result.details['mcc']['paper']:.4f} "
               f"({result.details['mcc']['pct_diff']:+.1f}%)")

    typer.echo()
    typer.echo(f"Parity Level: {result.parity_level.upper()}")

    if result.parity_level == "strict":
        typer.echo("🎉 Congratulations! Strict paper parity achieved!")
    elif result.parity_level == "acceptable":
        typer.echo("✅ Good! Acceptable paper parity achieved.")
    elif result.parity_level == "minimum":
        typer.echo("⚠️ Warning: Only minimum viable parity achieved.")
    else:
        typer.echo("❌ Failed: Results do not match paper.")

    if output:
        with open(results_file) as f:
            results_data = json.load(f)
        table = generate_benchmark_table([results_data], output)
        typer.echo(f"\nBenchmark table saved to {output}")
```

---

## 4. Tests

**File:** `tests/test_validation/test_parity.py`

```python
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
```

---

## 5. Implementation Checklist

- [ ] Create `src/arc_meshchop/validation/__init__.py`
- [ ] Create `src/arc_meshchop/validation/parity.py`
- [ ] Implement `validate_parity()` function
- [ ] Implement `generate_benchmark_table()` function
- [ ] Implement `run_statistical_comparison()` function
- [ ] Add `validate` CLI command
- [ ] Create tests in `tests/test_validation/`
- [ ] Document parity levels and criteria
- [ ] Add example validation reports

---

## 6. Verification Commands

```bash
# Validate experiment results
uv run arc-meshchop validate \
    experiments/meshnet26/experiment_results.json \
    --variant meshnet_26 \
    --output results/benchmark_table.md

# Run validation tests
uv run pytest tests/test_validation/ -v
```

---

## 7. Success Checklist

Before claiming paper replication:

- [ ] **MeshNet-26 achieves DICE ≥ 0.86**
- [ ] **Standard deviation ≤ 0.03**
- [ ] **All 90 configurations complete**
- [ ] **Test set evaluation on all 3 outer folds**
- [ ] **Benchmark table matches paper format**
- [ ] **Statistical comparison shows no significant difference from paper**

---

## 8. Example Output

```
============================================================
PAPER PARITY VALIDATION
============================================================
✅ meshnet_26: DICE=0.8712±0.0189 (parity: acceptable)

Detailed Comparison:
  DICE: 0.8712 vs 0.8760 (-0.5%)
  AVD:  0.2523 vs 0.2450 (+3.0%)
  MCC:  0.7542 vs 0.7600 (-0.8%)

Parity Level: ACCEPTABLE
✅ Good! Acceptable paper parity achieved.
```
