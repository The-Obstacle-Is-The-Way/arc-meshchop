# Spec 05: Evaluation & Metrics

> **Phase 5 of 7** — Evaluation metrics and statistical testing
>
> **Goal:** Implement complete evaluation pipeline matching the paper's methodology.

---

## Overview

This spec covers:
- DICE coefficient implementation
- Average Volume Difference (AVD)
- Matthews Correlation Coefficient (MCC)
- Statistical testing (Wilcoxon + Holm-Bonferroni)
- Benchmark reporting

---

## 1. Metrics Specification (FROM PAPER)

### 1.1 DICE Coefficient

| Property | Value |
|----------|-------|
| Formula | `2 × |A ∩ B| / (|A| + |B|)` |
| Range | [0, 1] |
| Target | ≥ 0.85 (reliable segmentation threshold) |
| Paper best | 0.876 (MeshNet-26) |

### 1.2 Average Volume Difference (AVD)

| Property | Value |
|----------|-------|
| Formula | `|V_pred - V_true| / V_true` |
| Range | [0, ∞) |
| Target | Lower is better |
| Paper best | 0.245 (MeshNet-26) |

### 1.3 Matthews Correlation Coefficient (MCC)

| Property | Value |
|----------|-------|
| Formula | `(TP×TN - FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))` |
| Range | [-1, 1] |
| Target | Higher is better |
| Paper best | 0.760 (MeshNet-26) |

### 1.4 Statistical Testing

| Property | Value |
|----------|-------|
| Test | Wilcoxon signed-rank (paired, non-parametric) |
| Correction | Holm-Bonferroni |
| Significance | p < 0.05 after correction |
| Reference | MeshNet-26 |

---

## 2. Implementation

### 2.1 Core Metrics

**File:** `src/arc_meshchop/evaluation/metrics.py`

```python
"""Evaluation metrics for stroke lesion segmentation.

Implements the three metrics from the paper:
1. DICE coefficient (Sørensen–Dice index)
2. Average Volume Difference (AVD)
3. Matthews Correlation Coefficient (MCC)

FROM PAPER: All metrics computed on binary segmentation masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclass
class MetricResult:
    """Single metric result with statistics."""

    name: str
    mean: float
    std: float
    values: list[float]

    def __str__(self) -> str:
        """Format as 'mean (std)'."""
        return f"{self.mean:.3f} ({self.std:.3f})"


def dice_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
) -> float:
    """Compute DICE coefficient.

    DICE = 2 × |A ∩ B| / (|A| + |B|)

    Args:
        pred: Predicted binary mask (B, D, H, W) or (D, H, W).
        target: Ground truth binary mask (same shape as pred).
        smooth: Smoothing factor to avoid division by zero.

    Returns:
        DICE coefficient in [0, 1].
    """
    pred_flat = pred.flatten().float()
    target_flat = target.flatten().float()

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()


def average_volume_difference(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """Compute Average Volume Difference.

    AVD = |V_pred - V_true| / V_true

    Args:
        pred: Predicted binary mask.
        target: Ground truth binary mask.

    Returns:
        AVD (0 = perfect, higher = worse).
    """
    pred_volume = pred.sum().float().item()
    target_volume = target.sum().float().item()

    if target_volume == 0:
        return float("inf") if pred_volume > 0 else 0.0

    avd = abs(pred_volume - target_volume) / target_volume
    return avd


def matthews_correlation_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """Compute Matthews Correlation Coefficient.

    MCC = (TP×TN - FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))

    Args:
        pred: Predicted binary mask.
        target: Ground truth binary mask.

    Returns:
        MCC in [-1, 1] (1 = perfect, 0 = random, -1 = inverse).
    """
    pred_bool = pred.flatten().bool()
    target_bool = target.flatten().bool()

    tp = (pred_bool & target_bool).sum().float()
    tn = (~pred_bool & ~target_bool).sum().float()
    fp = (pred_bool & ~target_bool).sum().float()
    fn = (~pred_bool & target_bool).sum().float()

    numerator = tp * tn - fp * fn
    denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    if denominator == 0:
        return 0.0

    mcc = numerator / denominator
    return mcc.item()


def compute_confusion_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, int]:
    """Compute confusion matrix elements.

    Args:
        pred: Predicted binary mask.
        target: Ground truth binary mask.

    Returns:
        Dictionary with TP, TN, FP, FN counts.
    """
    pred_bool = pred.flatten().bool()
    target_bool = target.flatten().bool()

    return {
        "tp": int((pred_bool & target_bool).sum()),
        "tn": int((~pred_bool & ~target_bool).sum()),
        "fp": int((pred_bool & ~target_bool).sum()),
        "fn": int((~pred_bool & target_bool).sum()),
    }


class SegmentationMetrics:
    """Compute all segmentation metrics for batch evaluation.

    Usage:
        metrics = SegmentationMetrics()
        results = metrics.compute_batch(predictions, targets)
    """

    def __init__(self, smooth: float = 1e-6) -> None:
        """Initialize metrics calculator.

        Args:
            smooth: Smoothing factor for DICE computation.
        """
        self.smooth = smooth

    def compute_single(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, float]:
        """Compute all metrics for a single sample.

        Args:
            pred: Predicted binary mask (D, H, W) or (1, D, H, W).
            target: Ground truth binary mask (same shape).

        Returns:
            Dictionary with dice, avd, mcc values.
        """
        # Ensure 3D (remove batch dimension if present)
        if pred.dim() == 4:
            pred = pred.squeeze(0)
        if target.dim() == 4:
            target = target.squeeze(0)

        return {
            "dice": dice_coefficient(pred, target, self.smooth),
            "avd": average_volume_difference(pred, target),
            "mcc": matthews_correlation_coefficient(pred, target),
        }

    def compute_batch(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, float]:
        """Compute mean metrics for a batch.

        Args:
            preds: Predicted masks (B, D, H, W).
            targets: Ground truth masks (B, D, H, W).

        Returns:
            Dictionary with mean dice, avd, mcc values.
        """
        dice_scores = []
        avd_scores = []
        mcc_scores = []

        batch_size = preds.shape[0]
        for i in range(batch_size):
            metrics = self.compute_single(preds[i], targets[i])
            dice_scores.append(metrics["dice"])
            avd_scores.append(metrics["avd"])
            mcc_scores.append(metrics["mcc"])

        return {
            "dice": sum(dice_scores) / len(dice_scores),
            "avd": sum(avd_scores) / len(avd_scores),
            "mcc": sum(mcc_scores) / len(mcc_scores),
        }

    def compute_with_stats(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, MetricResult]:
        """Compute metrics with mean, std, and per-sample values.

        Args:
            preds: Predicted masks (B, D, H, W).
            targets: Ground truth masks (B, D, H, W).

        Returns:
            Dictionary with MetricResult for each metric.
        """
        dice_scores: list[float] = []
        avd_scores: list[float] = []
        mcc_scores: list[float] = []

        batch_size = preds.shape[0]
        for i in range(batch_size):
            metrics = self.compute_single(preds[i], targets[i])
            dice_scores.append(metrics["dice"])
            avd_scores.append(metrics["avd"])
            mcc_scores.append(metrics["mcc"])

        import numpy as np

        return {
            "dice": MetricResult(
                name="DICE",
                mean=float(np.mean(dice_scores)),
                std=float(np.std(dice_scores)),
                values=dice_scores,
            ),
            "avd": MetricResult(
                name="AVD",
                mean=float(np.mean(avd_scores)),
                std=float(np.std(avd_scores)),
                values=avd_scores,
            ),
            "mcc": MetricResult(
                name="MCC",
                mean=float(np.mean(mcc_scores)),
                std=float(np.std(mcc_scores)),
                values=mcc_scores,
            ),
        }
```

### 2.2 Statistical Testing

**File:** `src/arc_meshchop/evaluation/statistics.py`

```python
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
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
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
        from statsmodels.stats.multitest import multipletests

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
    p_values = []

    for name in model_names:
        scores = model_scores_dict[name]
        p = wilcoxon_test(reference_scores, scores)
        p_values.append(p)

    # Apply Holm-Bonferroni correction
    corrected_p, is_significant = holm_bonferroni_correction(p_values, alpha)

    results = []
    for name, p, cp, sig in zip(model_names, p_values, corrected_p, is_significant):
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
    sig_lookup = {}
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
```

### 2.3 Evaluation Pipeline

**File:** `src/arc_meshchop/evaluation/evaluator.py`

```python
"""Complete evaluation pipeline for model assessment."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from tqdm import tqdm

from arc_meshchop.evaluation.metrics import MetricResult, SegmentationMetrics
from arc_meshchop.evaluation.statistics import compare_models_to_reference

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Complete evaluation results for a model."""

    model_name: str
    dice: MetricResult
    avd: MetricResult
    mcc: MetricResult
    num_samples: int

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "dice": asdict(self.dice),
            "avd": asdict(self.avd),
            "mcc": asdict(self.mcc),
            "num_samples": self.num_samples,
        }

    def save(self, path: Path | str) -> None:
        """Save results to JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path | str) -> EvaluationResult:
        """Load results from JSON file."""
        data = json.loads(Path(path).read_text())
        return cls(
            model_name=data["model_name"],
            dice=MetricResult(**data["dice"]),
            avd=MetricResult(**data["avd"]),
            mcc=MetricResult(**data["mcc"]),
            num_samples=data["num_samples"],
        )


class Evaluator:
    """Evaluator for stroke lesion segmentation models.

    Implements the evaluation protocol from the paper:
    - DICE, AVD, MCC metrics
    - Per-sample and aggregate statistics
    - Statistical comparison between models
    """

    def __init__(self, device: torch.device | None = None) -> None:
        """Initialize evaluator.

        Args:
            device: Device for inference (default: auto-detect).
        """
        from arc_meshchop.utils.device import get_device

        # Cross-platform device selection (CUDA > MPS > CPU)
        self.device = device or get_device()
        self.metrics_calculator = SegmentationMetrics()

    @torch.no_grad()
    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        model_name: str = "model",
        use_fp16: bool = True,
    ) -> EvaluationResult:
        """Evaluate model on dataset.

        Args:
            model: Model to evaluate.
            dataloader: DataLoader for evaluation data.
            model_name: Name for logging and results.
            use_fp16: Whether to use FP16 inference.

        Returns:
            EvaluationResult with all metrics.
        """
        model = model.to(self.device)
        model.eval()

        all_preds = []
        all_targets = []

        logger.info("Evaluating %s on %d batches", model_name, len(dataloader))

        for images, masks in tqdm(dataloader, desc=f"Evaluating {model_name}"):
            images = images.to(self.device)

            # Inference with platform-aware mixed precision
            # Only use FP16 autocast on CUDA (MPS/CPU don't benefit)
            if use_fp16 and self.device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(images)
            else:
                outputs = model(images)

            # Get predictions
            preds = outputs.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_targets.append(masks)

        # Concatenate all batches
        all_preds_cat = torch.cat(all_preds, dim=0)
        all_targets_cat = torch.cat(all_targets, dim=0)

        # Compute metrics with statistics
        metrics = self.metrics_calculator.compute_with_stats(all_preds_cat, all_targets_cat)

        result = EvaluationResult(
            model_name=model_name,
            dice=metrics["dice"],
            avd=metrics["avd"],
            mcc=metrics["mcc"],
            num_samples=len(all_preds_cat),
        )

        logger.info(
            "%s: DICE=%s, AVD=%s, MCC=%s",
            model_name,
            result.dice,
            result.avd,
            result.mcc,
        )

        return result

    def compare_to_reference(
        self,
        reference_result: EvaluationResult,
        other_results: list[EvaluationResult],
        metric: str = "dice",
    ) -> list:
        """Compare models to reference using statistical tests.

        Args:
            reference_result: Results from reference model (e.g., MeshNet-26).
            other_results: Results from other models to compare.
            metric: Metric to use for comparison ("dice", "avd", "mcc").

        Returns:
            List of ComparisonResult objects.
        """
        import numpy as np

        # Get reference scores
        reference_scores = np.array(getattr(reference_result, metric).values)

        # Build comparison dict
        model_scores_dict = {}
        for result in other_results:
            scores = np.array(getattr(result, metric).values)
            model_scores_dict[result.model_name] = scores

        return compare_models_to_reference(reference_scores, model_scores_dict)
```

### 2.4 Module `__init__.py`

**File:** `src/arc_meshchop/evaluation/__init__.py`

```python
"""Evaluation metrics and statistical testing."""

from arc_meshchop.evaluation.evaluator import EvaluationResult, Evaluator
from arc_meshchop.evaluation.metrics import (
    MetricResult,
    SegmentationMetrics,
    average_volume_difference,
    compute_confusion_matrix,
    dice_coefficient,
    matthews_correlation_coefficient,
)
from arc_meshchop.evaluation.statistics import (
    ComparisonResult,
    compare_models_to_reference,
    format_results_table,
    holm_bonferroni_correction,
    wilcoxon_test,
)

__all__ = [
    "ComparisonResult",
    "EvaluationResult",
    "Evaluator",
    "MetricResult",
    "SegmentationMetrics",
    "average_volume_difference",
    "compare_models_to_reference",
    "compute_confusion_matrix",
    "dice_coefficient",
    "format_results_table",
    "holm_bonferroni_correction",
    "matthews_correlation_coefficient",
    "wilcoxon_test",
]
```

---

## 3. Tests

### 3.1 Metrics Tests

**File:** `tests/test_evaluation/test_metrics.py`

```python
"""Tests for evaluation metrics."""

import pytest
import torch

from arc_meshchop.evaluation.metrics import (
    average_volume_difference,
    dice_coefficient,
    matthews_correlation_coefficient,
    SegmentationMetrics,
)


class TestDiceCoefficient:
    """Tests for DICE coefficient."""

    def test_perfect_overlap(self) -> None:
        """Verify DICE = 1 for perfect overlap."""
        pred = torch.ones(10, 10, 10)
        target = torch.ones(10, 10, 10)

        dice = dice_coefficient(pred, target)
        assert dice == pytest.approx(1.0, abs=1e-5)

    def test_no_overlap(self) -> None:
        """Verify DICE ≈ 0 for no overlap."""
        pred = torch.zeros(10, 10, 10)
        pred[:5, :, :] = 1
        target = torch.zeros(10, 10, 10)
        target[5:, :, :] = 1

        dice = dice_coefficient(pred, target)
        assert dice < 0.01  # Should be very small (smoothing prevents exact 0)

    def test_partial_overlap(self) -> None:
        """Verify DICE is in (0, 1) for partial overlap."""
        pred = torch.zeros(10, 10, 10)
        pred[2:8, 2:8, 2:8] = 1  # 6×6×6 = 216 voxels

        target = torch.zeros(10, 10, 10)
        target[4:10, 4:10, 4:10] = 1  # 6×6×6 = 216 voxels

        # Overlap: 4×4×4 = 64 voxels
        # DICE = 2×64 / (216 + 216) = 128/432 ≈ 0.296

        dice = dice_coefficient(pred, target)
        assert 0 < dice < 1

    def test_empty_masks(self) -> None:
        """Verify DICE ≈ 1 for both empty (no lesion case)."""
        pred = torch.zeros(10, 10, 10)
        target = torch.zeros(10, 10, 10)

        dice = dice_coefficient(pred, target)
        # With smoothing, empty masks give ~1.0
        assert dice > 0.99

    def test_range(self) -> None:
        """Verify DICE is always in [0, 1]."""
        for _ in range(10):
            pred = (torch.rand(8, 8, 8) > 0.5).float()
            target = (torch.rand(8, 8, 8) > 0.5).float()

            dice = dice_coefficient(pred, target)
            assert 0 <= dice <= 1


class TestAverageVolumeDifference:
    """Tests for AVD."""

    def test_perfect_volume(self) -> None:
        """Verify AVD = 0 for identical volumes."""
        pred = torch.zeros(10, 10, 10)
        pred[3:7, 3:7, 3:7] = 1

        target = torch.zeros(10, 10, 10)
        target[3:7, 3:7, 3:7] = 1

        avd = average_volume_difference(pred, target)
        assert avd == pytest.approx(0.0)

    def test_double_volume(self) -> None:
        """Verify AVD = 1 for double volume."""
        pred = torch.ones(10, 10, 10)  # 1000 voxels

        target = torch.zeros(10, 10, 10)
        target[:5, :, :] = 1  # 500 voxels

        avd = average_volume_difference(pred, target)
        assert avd == pytest.approx(1.0)  # |1000-500|/500 = 1.0

    def test_half_volume(self) -> None:
        """Verify AVD = 0.5 for half volume."""
        pred = torch.zeros(10, 10, 10)
        pred[:5, :, :] = 1  # 500 voxels

        target = torch.ones(10, 10, 10)  # 1000 voxels

        avd = average_volume_difference(pred, target)
        assert avd == pytest.approx(0.5)  # |500-1000|/1000 = 0.5

    def test_empty_target(self) -> None:
        """Verify AVD = inf for empty target with non-empty pred."""
        pred = torch.ones(10, 10, 10)
        target = torch.zeros(10, 10, 10)

        avd = average_volume_difference(pred, target)
        assert avd == float("inf")


class TestMatthewsCorrelationCoefficient:
    """Tests for MCC."""

    def test_perfect_prediction(self) -> None:
        """Verify MCC = 1 for perfect prediction."""
        pred = torch.zeros(10, 10, 10)
        pred[3:7, 3:7, 3:7] = 1

        target = pred.clone()

        mcc = matthews_correlation_coefficient(pred, target)
        assert mcc == pytest.approx(1.0)

    def test_inverse_prediction(self) -> None:
        """Verify MCC = -1 for inverse prediction."""
        target = torch.zeros(10, 10, 10)
        target[3:7, 3:7, 3:7] = 1

        pred = 1 - target  # Inverted

        mcc = matthews_correlation_coefficient(pred, target)
        assert mcc == pytest.approx(-1.0)

    def test_random_prediction(self) -> None:
        """Verify MCC ≈ 0 for random prediction."""
        torch.manual_seed(42)
        pred = (torch.rand(100, 100, 100) > 0.5).float()
        target = (torch.rand(100, 100, 100) > 0.5).float()

        mcc = matthews_correlation_coefficient(pred, target)
        assert -0.1 < mcc < 0.1  # Should be close to 0

    def test_range(self) -> None:
        """Verify MCC is always in [-1, 1]."""
        for _ in range(10):
            pred = (torch.rand(8, 8, 8) > 0.5).float()
            target = (torch.rand(8, 8, 8) > 0.5).float()

            mcc = matthews_correlation_coefficient(pred, target)
            assert -1 <= mcc <= 1


class TestSegmentationMetrics:
    """Tests for SegmentationMetrics class."""

    def test_compute_single(self) -> None:
        """Test computing metrics for single sample."""
        metrics = SegmentationMetrics()

        pred = (torch.rand(8, 8, 8) > 0.5).float()
        target = (torch.rand(8, 8, 8) > 0.5).float()

        result = metrics.compute_single(pred, target)

        assert "dice" in result
        assert "avd" in result
        assert "mcc" in result

    def test_compute_batch(self) -> None:
        """Test computing metrics for batch."""
        metrics = SegmentationMetrics()

        preds = (torch.rand(4, 8, 8, 8) > 0.5).float()
        targets = (torch.rand(4, 8, 8, 8) > 0.5).float()

        result = metrics.compute_batch(preds, targets)

        assert "dice" in result
        assert "avd" in result
        assert "mcc" in result

    def test_compute_with_stats(self) -> None:
        """Test computing metrics with statistics."""
        metrics = SegmentationMetrics()

        preds = (torch.rand(4, 8, 8, 8) > 0.5).float()
        targets = (torch.rand(4, 8, 8, 8) > 0.5).float()

        result = metrics.compute_with_stats(preds, targets)

        assert result["dice"].mean is not None
        assert result["dice"].std is not None
        assert len(result["dice"].values) == 4
```

### 3.2 Statistics Tests

**File:** `tests/test_evaluation/test_statistics.py`

```python
"""Tests for statistical testing."""

import numpy as np
import pytest

from arc_meshchop.evaluation.statistics import (
    compare_models_to_reference,
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
        # May or may not be significant due to randomness
        assert 0 < p <= 1


class TestHolmBonferroniCorrection:
    """Tests for Holm-Bonferroni correction."""

    def test_single_p_value(self) -> None:
        """Test with single p-value."""
        corrected, significant = holm_bonferroni_correction([0.03])

        assert len(corrected) == 1
        assert corrected[0] == pytest.approx(0.03)
        assert significant[0] is True

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
```

---

## 4. Implementation Checklist

### Phase 5.1: Core Metrics

- [ ] Create `src/arc_meshchop/evaluation/metrics.py`
- [ ] Implement `dice_coefficient`
- [ ] Implement `average_volume_difference`
- [ ] Implement `matthews_correlation_coefficient`
- [ ] Implement `SegmentationMetrics` class

### Phase 5.2: Statistical Testing

- [ ] Create `src/arc_meshchop/evaluation/statistics.py`
- [ ] Implement `wilcoxon_test`
- [ ] Implement `holm_bonferroni_correction`
- [ ] Implement `compare_models_to_reference`

### Phase 5.3: Evaluation Pipeline

- [ ] Create `src/arc_meshchop/evaluation/evaluator.py`
- [ ] Implement `Evaluator` class
- [ ] Implement result serialization

### Phase 5.4: Tests

- [ ] Create metrics tests
- [ ] Create statistics tests
- [ ] All tests pass

---

## 5. Verification Commands

```bash
# Run evaluation tests
uv run pytest tests/test_evaluation/ -v

# Test metrics manually
uv run python -c "
import torch
from arc_meshchop.evaluation import dice_coefficient, average_volume_difference, matthews_correlation_coefficient

# Create sample masks
pred = torch.zeros(64, 64, 64)
pred[20:40, 20:40, 20:40] = 1

target = torch.zeros(64, 64, 64)
target[25:45, 25:45, 25:45] = 1

print(f'DICE: {dice_coefficient(pred, target):.4f}')
print(f'AVD: {average_volume_difference(pred, target):.4f}')
print(f'MCC: {matthews_correlation_coefficient(pred, target):.4f}')
"
```

---

## 6. Expected Results (FROM PAPER)

### Target Performance for MeshNet-26

| Metric | Target | Tolerance |
|--------|--------|-----------|
| DICE | 0.876 | ±0.016 |
| AVD | 0.245 | ±0.036 |
| MCC | 0.760 | ±0.030 |

If your implementation achieves these values on the ARC dataset with the same cross-validation setup, the replication is successful.

---

## 7. References

- Paper Section 3: Results and statistical testing
- Research docs: `docs/research/05-evaluation-metrics.md`
- DICE: Sørensen (1948), Dice (1945)
- MCC: Matthews (1975)
- Wilcoxon test: scipy.stats.wilcoxon
- Holm correction: statsmodels.stats.multitest.multipletests
