"""Seeding utilities for reproducibility.

Ensures deterministic behavior across random, numpy, and torch.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for reproducibility.

    Seeds:
    - random
    - numpy
    - torch
    - torch.cuda

    Args:
        seed: Random seed.
        deterministic: If True, sets cuDNN to deterministic mode.
            (May impact performance).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """Initialize worker with deterministic seed.

    Used for PyTorch DataLoader to ensure workers have different but
    deterministic seeds based on the global seed.
    """
    # Get the base seed from torch (set by seed_everything)
    # Use initial_seed() to respect the generator state if one is used,
    # or the global state.
    worker_seed = torch.initial_seed() % 2**32
    # Add worker_id to avoid all workers having same seed
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def get_generator(seed: int) -> torch.Generator:
    """Get a PyTorch generator with specific seed.

    Args:
        seed: Random seed.

    Returns:
        torch.Generator seeded with seed.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g
