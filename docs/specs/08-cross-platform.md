# Spec 08: Cross-Platform Support (Mac, Linux, Windows/WSL2)

> **Supplementary Spec** — Device-agnostic training and deployment
>
> **Goal:** Enable full MeshNet implementation on Mac (MPS), Linux (CUDA), and Windows/WSL2 (CUDA) without code changes.

---

> **NOTE: Integration with Core Specs**
>
> The cross-platform patterns from this spec have been integrated into the core specs:
> - **Spec 01** - Device utility (`src/arc_meshchop/utils/device.py`) and test fixtures
> - **Spec 03** - Platform-aware `pin_memory` in DataLoaders
> - **Spec 04** - Platform-aware trainer (`torch.amp` instead of `torch.cuda.amp`)
> - **Spec 05** - Platform-aware evaluator
> - **Spec 06** - Mac/TFJS compatibility notes
>
> This spec serves as the reference documentation for cross-platform behavior.

---

## Overview

This spec ensures the arc-meshchop project runs seamlessly across:
- **Mac (Apple Silicon)**: M1/M2/M3/M4 with MPS backend
- **Linux**: NVIDIA CUDA (native)
- **Windows**: NVIDIA CUDA via WSL2

Key insight: **MeshNet is small enough to train on any platform.** The 147K parameters fit comfortably in 4-8GB VRAM (FP16), well within Mac unified memory limits.

---

## 1. Device Detection (2025 Best Practice)

### 1.1 Device Selection Utility

**File:** `src/arc_meshchop/utils/device.py`

```python
"""Cross-platform device selection for PyTorch.

Supports:
- CUDA (Linux, Windows/WSL2)
- MPS (Mac Apple Silicon)
- CPU (fallback)

References:
- https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/
- https://developer.apple.com/metal/pytorch/
- https://huggingface.co/docs/accelerate/en/usage_guides/mps
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
        # Check if MPS is actually usable (some ops may not be supported)
        if _mps_is_functional():
            device = torch.device("mps")
            logger.info("Auto-selected MPS device (Apple Silicon)")
            return device
        else:
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


def get_device_info() -> dict[str, str | bool | int]:
    """Get information about available devices.

    Returns:
        Dictionary with device availability and details.
    """
    info: dict[str, str | bool | int] = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "cpu_count": os.cpu_count() or 1,
    }

    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 2
        )

    if torch.backends.mps.is_available():
        info["mps_functional"] = _mps_is_functional()

    return info


def enable_mps_fallback() -> None:
    """Enable CPU fallback for unsupported MPS operations.

    Set PYTORCH_ENABLE_MPS_FALLBACK=1 environment variable.
    This allows training to continue when MPS doesn't support an op.

    Reference: https://huggingface.co/docs/transformers/v4.47.1/en/perf_train_special
    """
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    logger.info("Enabled MPS fallback to CPU for unsupported operations")
```

### 1.2 Module Export

**Add to:** `src/arc_meshchop/utils/__init__.py`

```python
"""Utility functions."""

from arc_meshchop.utils.device import (
    DeviceType,
    enable_mps_fallback,
    get_device,
    get_device_info,
)

__all__ = [
    "DeviceType",
    "enable_mps_fallback",
    "get_device",
    "get_device_info",
]
```

---

## 2. Training Infrastructure Updates

### 2.1 Platform-Aware Mixed Precision

The current spec uses `torch.cuda.amp` which is CUDA-specific. Update for cross-platform:

**Update:** `src/arc_meshchop/training/trainer.py`

```python
# BEFORE (CUDA-only):
from torch.cuda.amp import GradScaler, autocast

# AFTER (cross-platform):
from torch.amp import GradScaler, autocast

# In Trainer.__init__:
def __init__(self, model, config, device=None):
    from arc_meshchop.utils.device import get_device

    self.device = device or get_device()

    # Platform-specific precision handling
    if self.device.type == "cuda":
        self.scaler = GradScaler("cuda", enabled=config.use_fp16)
        self.amp_dtype = torch.float16
    elif self.device.type == "mps":
        # MPS has limited FP16 support - use bfloat16 or FP32
        self.scaler = GradScaler("cpu", enabled=False)  # No scaling on MPS
        self.amp_dtype = torch.float32  # MPS FP16 is experimental
        if config.use_fp16:
            logger.warning("FP16 not fully supported on MPS, using FP32")
    else:
        self.scaler = GradScaler("cpu", enabled=False)
        self.amp_dtype = torch.float32

# In training loop:
with autocast(device_type=self.device.type, dtype=self.amp_dtype):
    outputs = self.model(images)
    loss = self.loss_fn(outputs, masks)
```

### 2.2 MPS-Specific Considerations

| Issue | Workaround |
|-------|------------|
| FP16 not fully supported | Use FP32 on MPS (still fast due to unified memory) |
| Some ops not implemented | Set `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| No distributed training | Single-GPU only (fine for MeshNet) |
| Memory management | Use `torch.mps.empty_cache()` periodically |

---

## 3. Export Pipeline Updates

### 3.1 ONNX Export (Cross-Platform)

ONNX export works identically on all platforms. No changes needed.

### 3.2 TensorFlow.js Export Considerations

**Critical finding from research:**

> "Both models and filters currently use the **WebGL2 TFJS backend** that leverages the graphics card of the user's computer (and can be extended to the WebGPU backend if this matures to support 3D convolutions)."
> — [BrainChop: Providing an Edge Ecosystem](https://pmc.ncbi.nlm.nih.gov/articles/PMC11411854/)

**TensorFlow.js DOES support Conv3D** via `tf.conv3d()` and `tf.layers.conv3d()`.

**Export path:**
```
PyTorch → ONNX → TensorFlow SavedModel → TensorFlow.js
```

**Platform notes:**
- `onnx-tf` (ONNX to TensorFlow) works on all platforms
- `tensorflowjs_converter` works on all platforms
- **BUT:** TensorFlow itself doesn't have ARM Mac wheels (use `tensorflow-macos` or skip TFJS export on Mac)

**Recommended approach:**
1. Train on Mac (MPS or CPU)
2. Export to ONNX (works everywhere)
3. Convert ONNX → TFJS on Linux (where TensorFlow is well-supported)

---

## 4. Updated pyproject.toml

Add platform-specific notes:

```toml
# =============================================================================
# PLATFORM NOTES
# =============================================================================
#
# Mac (Apple Silicon):
#   - Use MPS backend for GPU acceleration
#   - FP16 training may fall back to FP32
#   - TFJS export requires Linux (no TF ARM wheels)
#   - Install: uv sync --all-extras
#
# Linux (NVIDIA CUDA):
#   - Full FP16 support
#   - Full TFJS export support
#   - Install: uv sync --all-extras
#
# Windows (WSL2 + CUDA):
#   - Use WSL2 Ubuntu for best compatibility
#   - CUDA is auto-detected from Windows driver
#   - Install: uv sync --all-extras
# =============================================================================
```

---

## 5. Tests

### 5.1 Device Detection Tests

**File:** `tests/test_utils/test_device.py`

```python
"""Tests for cross-platform device detection."""

import pytest
import torch

from arc_meshchop.utils.device import get_device, get_device_info


class TestGetDevice:
    """Tests for get_device function."""

    def test_returns_torch_device(self) -> None:
        """Verify returns a torch.device."""
        device = get_device()
        assert isinstance(device, torch.device)

    def test_cpu_fallback_always_works(self) -> None:
        """Verify CPU fallback always succeeds."""
        device = get_device(preferred="cpu")
        assert device.type == "cpu"

    def test_preferred_device_respected(self) -> None:
        """Verify preferred device is used when available."""
        device = get_device(preferred="cpu")
        assert device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_detection(self) -> None:
        """Verify CUDA is detected when available."""
        device = get_device(preferred="cuda")
        assert device.type == "cuda"

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS not available"
    )
    def test_mps_detection(self) -> None:
        """Verify MPS is detected on Mac."""
        device = get_device(preferred="mps")
        assert device.type == "mps"


class TestGetDeviceInfo:
    """Tests for get_device_info function."""

    def test_returns_dict(self) -> None:
        """Verify returns a dictionary."""
        info = get_device_info()
        assert isinstance(info, dict)

    def test_contains_cuda_availability(self) -> None:
        """Verify CUDA availability is reported."""
        info = get_device_info()
        assert "cuda_available" in info
        assert isinstance(info["cuda_available"], bool)

    def test_contains_mps_availability(self) -> None:
        """Verify MPS availability is reported."""
        info = get_device_info()
        assert "mps_available" in info
        assert isinstance(info["mps_available"], bool)


class TestModelOnDevice:
    """Tests for running MeshNet on detected device."""

    def test_meshnet_forward_on_device(self) -> None:
        """Verify MeshNet can run on detected device."""
        from arc_meshchop.models import meshnet_5

        device = get_device()
        model = meshnet_5().to(device)

        x = torch.randn(1, 1, 8, 8, 8, device=device)
        output = model(x)

        assert output.device.type == device.type
        assert output.shape == (1, 2, 8, 8, 8)
```

---

## 6. CLI Updates

### 6.1 Device Selection in CLI

**Update:** `src/arc_meshchop/cli.py`

```python
import typer
from arc_meshchop.utils.device import get_device, get_device_info

app = typer.Typer()


@app.command()
def info() -> None:
    """Show device and environment information."""
    import torch
    from rich.console import Console
    from rich.table import Table

    console = Console()
    info = get_device_info()

    table = Table(title="Device Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("PyTorch Version", torch.__version__)
    table.add_row("CUDA Available", str(info["cuda_available"]))
    table.add_row("MPS Available", str(info["mps_available"]))
    table.add_row("CPU Cores", str(info["cpu_count"]))

    if info["cuda_available"]:
        table.add_row("CUDA Device", str(info.get("cuda_device_name", "N/A")))
        table.add_row("CUDA Memory (GB)", str(info.get("cuda_memory_gb", "N/A")))

    if info["mps_available"]:
        table.add_row("MPS Functional", str(info.get("mps_functional", "N/A")))

    console.print(table)


@app.command()
def train(
    device: str = typer.Option("auto", help="Device: auto, cuda, mps, cpu"),
) -> None:
    """Train MeshNet model."""
    from arc_meshchop.utils.device import get_device

    if device == "auto":
        selected_device = get_device()
    else:
        selected_device = get_device(preferred=device)  # type: ignore

    typer.echo(f"Training on: {selected_device}")
    # ... training code
```

---

## 7. Quick Start by Platform

### Mac (Apple Silicon)

```bash
# Install
uv sync --all-extras

# Check device
uv run arc-meshchop info

# Train (auto-detects MPS)
uv run arc-meshchop train

# Export to ONNX
uv run arc-meshchop export --format onnx
```

### Linux (NVIDIA CUDA)

```bash
# Install
uv sync --all-extras

# Check device
uv run arc-meshchop info

# Train (auto-detects CUDA)
uv run arc-meshchop train

# Export to ONNX and TFJS
uv run arc-meshchop export --format onnx
uv run arc-meshchop export --format tfjs
```

### Windows (WSL2)

```bash
# In WSL2 Ubuntu terminal
uv sync --all-extras

# Check device (should detect CUDA from Windows driver)
uv run arc-meshchop info

# Train
uv run arc-meshchop train

# Export
uv run arc-meshchop export --format onnx
```

---

## 8. BrainChop Integration Validation

### 8.1 Critical Path

For Chris Rorden demo (BrainChop browser deployment):

1. **Train MeshNet** on Mac (MPS) - ✅ Supported
2. **Export to ONNX** on Mac - ✅ Supported
3. **Convert ONNX → TFJS** on Linux - ✅ Supported (do this step on Linux)
4. **Load in BrainChop** - ✅ Supported (TF.js has Conv3D)

### 8.2 BrainChop Confirmation

From the [BrainChop paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11411854/):

> "MeshNet is a feed-forward **3D convolutional neural network** with dilated kernels."

> "The current BrainChop models are all based on **MeshNet models** that are renowned for their modest computational requirements. These models were **converted to TensorFlow.JS (TFJS)**."

> "Both models and filters currently use the **WebGL2 TFJS backend**."

**Conclusion:** The export pipeline is proven to work. BrainChop already uses MeshNet with TF.js.

---

## 9. Implementation Checklist

### Phase 8.1: Device Utilities

- [ ] Create `src/arc_meshchop/utils/device.py`
- [ ] Implement `get_device()` with auto-detection
- [ ] Implement `get_device_info()`
- [ ] Implement `enable_mps_fallback()`

### Phase 8.2: Update Training

- [ ] Replace `torch.cuda.amp` with `torch.amp`
- [ ] Add device parameter to Trainer
- [ ] Handle MPS FP16 limitations
- [ ] Test on Mac MPS

### Phase 8.3: Update CLI

- [ ] Add `info` command
- [ ] Add `--device` option to `train`
- [ ] Display device in training output

### Phase 8.4: Tests

- [ ] Create `tests/test_utils/test_device.py`
- [ ] Test device detection on CI (CPU)
- [ ] Test model forward on detected device

---

## 10. References

### Official Documentation

- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)
- [Apple Metal PyTorch](https://developer.apple.com/metal/pytorch/)
- [HuggingFace MPS Guide](https://huggingface.co/docs/accelerate/en/usage_guides/mps)
- [TensorFlow.js Conv3D](https://js.tensorflow.org/api/1.0.0/#conv3d)

### BrainChop

- [BrainChop GitHub](https://github.com/neuroneural/brainchop)
- [BrainChop Paper (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11411854/)
- [BrainChop arXiv](https://ar5iv.labs.arxiv.org/html/2310.16162)

### 2025 Best Practices

- [PyTorch Install Guide 2025](https://huggingface.co/blog/daya-shankar/pytorch-install-guide)
- [WSL2 CUDA Setup](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl)
