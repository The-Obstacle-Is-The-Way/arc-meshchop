"""Evaluation metrics for stroke lesion segmentation.

Implements the three metrics from the paper:
1. DICE coefficient (Sorensen-Dice index)
2. Average Volume Difference (AVD)
3. Matthews Correlation Coefficient (MCC)

FROM PAPER: All metrics computed on binary segmentation masks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


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

    DICE = 2 * |A & B| / (|A| + |B|)

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
    return float(dice.item())


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
    pred_volume = float(pred.sum().float().item())
    target_volume = float(target.sum().float().item())

    if target_volume == 0:
        return float("inf") if pred_volume > 0 else 0.0

    avd = abs(pred_volume - target_volume) / target_volume
    return avd


def matthews_correlation_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """Compute Matthews Correlation Coefficient.

    MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))

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
    return float(mcc.item())


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
        dice_scores: list[float] = []
        avd_scores: list[float] = []
        mcc_scores: list[float] = []

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
