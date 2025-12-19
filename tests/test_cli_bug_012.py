"""Tests for BUG-012: Deterministic Seeding.

Ensures that:
1. DataLoaders produce deterministic batches when seeded.
2. Training produces deterministic results when seeded.
3. Seeding affects random, numpy, and torch.
"""

import random
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from arc_meshchop.data.dataset import create_dataloaders
from arc_meshchop.utils.seeding import get_generator, seed_everything, worker_init_fn


def test_seed_everything() -> None:
    """Test that seed_everything seeds all RNGs."""
    seed = 12345
    seed_everything(seed)

    # Check random
    r1 = random.random()
    seed_everything(seed)
    r2 = random.random()
    assert r1 == r2

    # Check numpy
    n1 = np.random.rand()
    seed_everything(seed)
    n2 = np.random.rand()
    assert n1 == n2

    # Check torch
    t1 = torch.rand(1)
    seed_everything(seed)
    t2 = torch.rand(1)
    assert torch.allclose(t1, t2)


def test_dataloader_determinism() -> None:
    """Test that seeded DataLoaders produce deterministic batches."""
    # Create dummy dataset
    data = torch.randn(10, 3, 32, 32)
    labels = torch.randint(0, 2, (10,))
    dataset = TensorDataset(data, labels)

    seed = 42

    # Create two loaders with same seed
    loader1 = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        worker_init_fn=worker_init_fn,
        generator=get_generator(seed),
    )

    loader2 = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        worker_init_fn=worker_init_fn,
        generator=get_generator(seed),
    )

    # Collect batches
    batches1 = list(loader1)
    batches2 = list(loader2)

    # Verify exact match
    for (b1_x, b1_y), (b2_x, b2_y) in zip(batches1, batches2, strict=False):
        assert torch.allclose(b1_x, b2_x)
        assert torch.allclose(b1_y, b2_y)


# Mock ARCDataset behavior for simplicity
class MockDataset(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Simple mock dataset for testing dataloader determinism."""

    def __len__(self) -> int:
        """Return dataset length."""
        return 10

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a single sample."""
        return torch.tensor([float(idx)]), torch.tensor([0])


def test_create_dataloaders_determinism() -> None:
    """Test create_dataloaders wrapper with seeding."""
    train_ds = cast(Any, MockDataset())
    val_ds = cast(Any, MockDataset())

    seed = 999

    l1_train, l1_val = create_dataloaders(train_ds, val_ds, batch_size=2, num_workers=2, seed=seed)
    l2_train, l2_val = create_dataloaders(train_ds, val_ds, batch_size=2, num_workers=2, seed=seed)

    # Check train loader shuffle
    batches1 = [b[0] for b in l1_train]
    batches2 = [b[0] for b in l2_train]

    # Concatenate to see order
    order1 = torch.cat(batches1).flatten()
    order2 = torch.cat(batches2).flatten()

    assert torch.allclose(order1, order2)
