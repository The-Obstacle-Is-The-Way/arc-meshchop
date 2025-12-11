"""Cross-platform device selection for PyTorch.

Supports:
- CUDA (Linux, Windows/WSL2)
- MPS (Mac Apple Silicon)
- CPU (fallback)

References:
- https://pytorch.org/docs/stable/notes/mps.html
- https://developer.apple.com/metal/pytorch/
"""

from __future__ import annotations

import logging
import os
from typing import Literal

import torch

logger = logging.getLogger(__name__)

DeviceType = Literal["cuda", "mps", "cpu"]


def get_device(preferred: DeviceType | None = None) -> torch.device:
    """Get the best available device for training/inference.

    Priority:
    1. User-specified preference (if available)
    2. CUDA (if available)
    3. MPS (if available, Mac Apple Silicon)
    4. CPU (fallback)

    Args:
        preferred: Preferred device type. If not available, falls back.

    Returns:
        torch.device for training/inference.
    """
    if preferred == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
        return device

    if preferred == "mps" and torch.backends.mps.is_available():
        if _mps_is_functional():
            device = torch.device("mps")
            logger.info("Using MPS device (Apple Silicon)")
            return device
        logger.warning("MPS requested but not functional, falling back")

    if preferred == "cpu":
        logger.info("Using CPU device (requested)")
        return torch.device("cpu")

    # Auto-detect best available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Auto-selected CUDA device: %s", torch.cuda.get_device_name(0))
        return device

    if torch.backends.mps.is_available():
        if _mps_is_functional():
            device = torch.device("mps")
            logger.info("Auto-selected MPS device (Apple Silicon)")
            return device
        logger.warning("MPS available but not functional, falling back to CPU")

    logger.info("Using CPU device (no GPU available)")
    return torch.device("cpu")


def _mps_is_functional() -> bool:
    """Check if MPS backend actually works.

    Some PyTorch operations are not implemented for MPS.
    This performs a quick sanity check.
    """
    try:
        x = torch.ones(2, 2, device="mps")
        y = x + x
        _ = y.cpu()  # Ensure we can move back to CPU
        return True
    except Exception:
        return False


def get_device_info() -> dict[str, str | bool | int | float]:
    """Get information about available devices.

    Returns:
        Dictionary with device availability and details.
    """
    info: dict[str, str | bool | int | float] = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "cpu_count": os.cpu_count() or 1,
    }

    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)

    if torch.backends.mps.is_available():
        info["mps_functional"] = _mps_is_functional()

    return info


def enable_mps_fallback() -> None:
    """Enable CPU fallback for unsupported MPS operations.

    Set PYTORCH_ENABLE_MPS_FALLBACK=1 environment variable.
    This allows training to continue when MPS doesn't support an op.
    """
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    logger.info("Enabled MPS fallback to CPU for unsupported operations")
