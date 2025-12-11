"""Loss functions for stroke lesion segmentation.

FROM PAPER Section 2:
"The objective is cross-entropy loss with label smoothing of 0.01
and class weighting of 0.5 for the background and 1.0 for lesions."
"""

from __future__ import annotations

import torch
import torch.nn as nn


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted cross-entropy loss with label smoothing.

    Implements the loss function from the paper:
    - Class weights: [0.5, 1.0] for [background, lesion]
    - Label smoothing: 0.01

    This addresses class imbalance where lesions are typically
    1-10% of brain volume.
    """

    def __init__(
        self,
        background_weight: float = 0.5,
        lesion_weight: float = 1.0,
        label_smoothing: float = 0.01,
    ) -> None:
        """Initialize loss function.

        Args:
            background_weight: Weight for background class.
            lesion_weight: Weight for lesion class.
            label_smoothing: Label smoothing factor.
        """
        super().__init__()

        self.class_weights = torch.tensor([background_weight, lesion_weight])
        self.label_smoothing = label_smoothing

        self._loss_fn: nn.CrossEntropyLoss | None = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted cross-entropy loss.

        Args:
            logits: Model output of shape (B, C, D, H, W).
            targets: Ground truth of shape (B, D, H, W).

        Returns:
            Scalar loss value.
        """
        # Initialize loss function on first call (to get correct device)
        if self._loss_fn is None or self.class_weights.device != logits.device:
            self.class_weights = self.class_weights.to(logits.device)
            self._loss_fn = nn.CrossEntropyLoss(
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
            )

        result: torch.Tensor = self._loss_fn(logits, targets)
        return result


def create_loss_function(
    background_weight: float = 0.5,
    lesion_weight: float = 1.0,
    label_smoothing: float = 0.01,
) -> WeightedCrossEntropyLoss:
    """Create loss function with paper defaults.

    Args:
        background_weight: Weight for background class.
        lesion_weight: Weight for lesion class.
        label_smoothing: Label smoothing factor.

    Returns:
        Configured loss function.
    """
    return WeightedCrossEntropyLoss(
        background_weight=background_weight,
        lesion_weight=lesion_weight,
        label_smoothing=label_smoothing,
    )
