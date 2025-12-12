"""Optimizer and scheduler setup.

FROM PAPER Section 2:
- AdamW optimizer (lr=0.001, weight_decay=3e-5, eps=1e-4)
- OneCycleLR scheduler (max_lr=0.001, pct_start=0.01)
"""

from __future__ import annotations

import torch
import torch.optim as optim


def create_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 0.001,
    weight_decay: float = 3e-5,
    eps: float = 1e-4,
) -> optim.AdamW:
    """Create AdamW optimizer with paper defaults.

    FROM PAPER Section 2:
    "AdamW optimizer with a learning rate of 0.001,
    weight decay of 3e-5, and epsilon of 1e-4"

    Args:
        model: Model to optimize.
        learning_rate: Learning rate.
        weight_decay: Weight decay for regularization.
        eps: Epsilon for numerical stability (higher for FP16).

    Returns:
        Configured AdamW optimizer.
    """
    return optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=eps,
    )


def create_scheduler(
    optimizer: optim.Optimizer,
    max_lr: float,
    total_steps: int,
    pct_start: float = 0.01,
    div_factor: float = 100.0,
) -> optim.lr_scheduler.OneCycleLR:
    """Create OneCycleLR scheduler with paper defaults.

    FROM PAPER Section 2:
    "OneCycleLR learning rate scheduler. The scheduler starts at 1/100th
    of the max learning rate, quickly increases to the maximum, and
    gradually decreases. The maximum learning rate was set to 0.001,
    with the increase phase occurring in the first 1% of training time."

    Args:
        optimizer: Optimizer to schedule.
        max_lr: Maximum learning rate.
        total_steps: Total training steps (epochs * steps_per_epoch).
        pct_start: Fraction of training for warmup.
        div_factor: Divisor for initial LR (initial_lr = max_lr/div_factor).
            FROM PAPER: 100.0 ("starts at 1/100th of the max learning rate").

    Returns:
        Configured OneCycleLR scheduler.
    """
    return optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        div_factor=div_factor,  # FROM PAPER: "starts at 1/100th of max LR"
        # PyTorch defaults: anneal_strategy='cos', cycle_momentum=True
    )
