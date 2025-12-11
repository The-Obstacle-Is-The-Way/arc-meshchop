"""Shared pytest fixtures for arc-meshchop tests."""

import numpy as np
import pytest
import torch


def _mps_is_functional() -> bool:
    """Check if MPS backend actually works (some ops may not be supported)."""
    try:
        x = torch.ones(2, 2, device="mps")
        _ = (x + x).cpu()
        return True
    except Exception:
        return False


@pytest.fixture
def device() -> torch.device:
    """Get available device (CUDA, MPS, or CPU).

    Priority:
    1. CUDA (Linux, Windows/WSL2)
    2. MPS (Mac Apple Silicon, if functional)
    3. CPU (fallback)

    Note: Inline implementation to avoid import dependencies during test setup.
    Mirrors logic from arc_meshchop.utils.device.get_device().
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and _mps_is_functional():
        return torch.device("mps")
    return torch.device("cpu")


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
