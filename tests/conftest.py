"""Shared pytest fixtures for arc-meshchop tests."""

import numpy as np
import pytest
import torch


@pytest.fixture
def device() -> torch.device:
    """Get available device (GPU if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def random_seed() -> int:
    """Fixed random seed for reproducibility."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture
def sample_volume() -> torch.Tensor:
    """Create a sample 256³ volume tensor."""
    return torch.randn(1, 1, 256, 256, 256)


@pytest.fixture
def small_volume() -> torch.Tensor:
    """Create a small 32³ volume for fast tests."""
    return torch.randn(1, 1, 32, 32, 32)


@pytest.fixture
def tiny_volume() -> torch.Tensor:
    """Create a tiny 8³ volume for unit tests."""
    return torch.randn(1, 1, 8, 8, 8)
