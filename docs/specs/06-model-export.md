# Spec 06: Model Export (ONNX & TensorFlow.js)

> **Phase 6 of 7** — Model export for deployment
>
> **Goal:** Export trained MeshNet models to ONNX and TensorFlow.js formats for browser and edge deployment.

---

## Overview

This spec covers:
- PyTorch → ONNX export
- ONNX → TensorFlow.js conversion
- Model quantization (FP16, INT8)
- Export validation and testing
- BrainChop integration preparation

---

## 1. Export Targets

### 1.1 Deployment Scenarios

| Target | Format | Use Case |
|--------|--------|----------|
| Browser (BrainChop) | TensorFlow.js | Client-side inference |
| Python backends | ONNX | FastAPI, Flask servers |
| Edge devices | ONNX | Mobile, embedded |
| Cloud inference | ONNX + TensorRT | High-throughput servers |

### 1.2 Memory Constraints

| Variant | Model Size (FP32) | Model Size (FP16) | Browser Limit |
|---------|-------------------|-------------------|---------------|
| MeshNet-5 | ~23 KB | ~11 KB | ✅ |
| MeshNet-16 | ~225 KB | ~112 KB | ✅ |
| MeshNet-26 | ~590 KB | ~295 KB | ✅ |

All MeshNet variants fit comfortably within browser memory limits (~2GB WebGL).

---

## 2. Implementation

### 2.1 ONNX Export

**File:** `src/arc_meshchop/export/onnx_export.py`

```python
"""ONNX export for MeshNet models.

Exports PyTorch MeshNet models to ONNX format for:
- Cross-platform inference
- TensorRT optimization
- TensorFlow.js conversion

FROM BRAINCHOP REFERENCE:
The BrainChop project exports models via ONNX as an intermediate step
before converting to TensorFlow.js.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.onnx

if TYPE_CHECKING:
    from arc_meshchop.models import MeshNet

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: MeshNet,
    output_path: Path | str,
    input_shape: tuple[int, int, int, int, int] = (1, 1, 256, 256, 256),
    opset_version: int = 17,
    dynamic_axes: dict | None = None,
    simplify: bool = True,
) -> Path:
    """Export MeshNet model to ONNX format.

    Args:
        model: Trained MeshNet model.
        output_path: Path for output ONNX file.
        input_shape: Input tensor shape (B, C, D, H, W).
        opset_version: ONNX opset version.
        dynamic_axes: Optional dynamic axes for variable batch size.
        simplify: Whether to simplify the ONNX graph.

    Returns:
        Path to exported ONNX file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Set model to eval mode
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(*input_shape)

    # Default dynamic axes for batch size
    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    logger.info("Exporting model to ONNX: %s", output_path)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    logger.info("ONNX export complete: %s", output_path)

    # Simplify if requested
    if simplify:
        try:
            import onnxsim

            logger.info("Simplifying ONNX model...")
            simplified_model, check = onnxsim.simplify(str(output_path))
            if check:
                import onnx

                onnx.save(simplified_model, str(output_path))
                logger.info("ONNX simplification complete")
            else:
                logger.warning("ONNX simplification check failed, keeping original")
        except ImportError:
            logger.warning("onnxsim not installed, skipping simplification")

    return output_path


def validate_onnx_export(
    onnx_path: Path | str,
    pytorch_model: MeshNet,
    input_shape: tuple[int, int, int, int, int] = (1, 1, 32, 32, 32),
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> bool:
    """Validate ONNX export matches PyTorch model.

    Args:
        onnx_path: Path to ONNX model.
        pytorch_model: Original PyTorch model.
        input_shape: Input shape for validation (smaller for speed).
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.

    Returns:
        True if outputs match within tolerance.
    """
    import numpy as np
    import onnxruntime as ort

    # Create test input
    test_input = torch.randn(*input_shape)

    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input).numpy()

    # ONNX inference
    session = ort.InferenceSession(str(onnx_path))
    onnx_output = session.run(None, {"input": test_input.numpy()})[0]

    # Compare outputs
    matches = np.allclose(pytorch_output, onnx_output, rtol=rtol, atol=atol)

    if matches:
        logger.info("ONNX validation passed: outputs match within tolerance")
    else:
        max_diff = np.max(np.abs(pytorch_output - onnx_output))
        logger.error("ONNX validation failed: max difference = %f", max_diff)

    return matches


def quantize_onnx(
    input_path: Path | str,
    output_path: Path | str,
    quantization_type: str = "fp16",
) -> Path:
    """Quantize ONNX model for smaller size and faster inference.

    Args:
        input_path: Path to input ONNX model.
        output_path: Path for quantized model.
        quantization_type: "fp16" or "int8".

    Returns:
        Path to quantized model.
    """
    import onnx
    from onnxconverter_common import float16

    output_path = Path(output_path)
    model = onnx.load(str(input_path))

    if quantization_type == "fp16":
        logger.info("Quantizing to FP16...")
        model_fp16 = float16.convert_float_to_float16(model)
        onnx.save(model_fp16, str(output_path))

    elif quantization_type == "int8":
        logger.info("Quantizing to INT8...")
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantize_dynamic(
            str(input_path),
            str(output_path),
            weight_type=QuantType.QUInt8,
        )

    else:
        raise ValueError(f"Unknown quantization type: {quantization_type}")

    logger.info("Quantization complete: %s", output_path)
    return output_path
```

### 2.2 TensorFlow.js Export

**File:** `src/arc_meshchop/export/tfjs_export.py`

```python
"""TensorFlow.js export for browser deployment.

Converts PyTorch models to TensorFlow.js format via ONNX intermediate.
This enables BrainChop integration for client-side MRI segmentation.

FROM BRAINCHOP REFERENCE:
BrainChop uses TensorFlow.js for browser-based inference with WebGL
acceleration. Models are stored as JSON + binary weight files.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arc_meshchop.models import MeshNet

logger = logging.getLogger(__name__)


def export_to_tfjs(
    onnx_path: Path | str,
    output_dir: Path | str,
    quantization: str | None = None,
) -> Path:
    """Convert ONNX model to TensorFlow.js format.

    Uses tensorflowjs_converter to convert ONNX → TF SavedModel → TFJS.

    Args:
        onnx_path: Path to ONNX model.
        output_dir: Directory for TFJS output files.
        quantization: Optional quantization ("float16", "uint8", "uint16").

    Returns:
        Path to output directory containing model.json and weight files.
    """
    onnx_path = Path(onnx_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Intermediate SavedModel path
    saved_model_dir = output_dir / "saved_model_temp"

    logger.info("Converting ONNX to TensorFlow SavedModel...")

    # Step 1: ONNX → TensorFlow SavedModel
    try:
        import onnx
        from onnx_tf.backend import prepare

        onnx_model = onnx.load(str(onnx_path))
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(str(saved_model_dir))

    except ImportError:
        # Fallback to command-line tool
        logger.info("Using onnx-tf CLI for conversion...")
        subprocess.run(
            [
                "onnx-tf",
                "convert",
                "-i", str(onnx_path),
                "-o", str(saved_model_dir),
            ],
            check=True,
        )

    logger.info("Converting TensorFlow SavedModel to TFJS...")

    # Step 2: TensorFlow SavedModel → TensorFlow.js
    tfjs_cmd = [
        "tensorflowjs_converter",
        "--input_format=tf_saved_model",
        "--output_format=tfjs_graph_model",
    ]

    if quantization:
        tfjs_cmd.extend(["--quantization_dtype", quantization])

    tfjs_cmd.extend([str(saved_model_dir), str(output_dir)])

    subprocess.run(tfjs_cmd, check=True)

    # Cleanup intermediate SavedModel
    shutil.rmtree(saved_model_dir)

    logger.info("TFJS export complete: %s", output_dir)
    return output_dir


def export_pytorch_to_tfjs(
    model: MeshNet,
    output_dir: Path | str,
    model_name: str = "meshnet",
    input_shape: tuple[int, int, int, int, int] = (1, 1, 256, 256, 256),
    quantization: str | None = "float16",
) -> Path:
    """Export PyTorch model directly to TensorFlow.js.

    Convenience function that handles ONNX intermediate step.

    Args:
        model: PyTorch MeshNet model.
        output_dir: Directory for output files.
        model_name: Name for the model files.
        input_shape: Input tensor shape.
        quantization: Optional quantization for TFJS.

    Returns:
        Path to TFJS model directory.
    """
    from arc_meshchop.export.onnx_export import export_to_onnx

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Export to ONNX
    onnx_path = output_dir / f"{model_name}.onnx"
    export_to_onnx(model, onnx_path, input_shape=input_shape)

    # Step 2: Convert to TFJS
    tfjs_dir = output_dir / f"{model_name}_tfjs"
    export_to_tfjs(onnx_path, tfjs_dir, quantization=quantization)

    return tfjs_dir


def create_brainchop_model_info(
    model_dir: Path | str,
    model_name: str,
    channels: int,
    input_shape: tuple[int, int, int] = (256, 256, 256),
    labels: list[str] | None = None,
) -> Path:
    """Create BrainChop-compatible model info JSON.

    BrainChop expects a specific JSON format for model metadata.

    Args:
        model_dir: Directory containing TFJS model.
        model_name: Display name for the model.
        channels: Number of channels in the model.
        input_shape: Expected input shape (D, H, W).
        labels: List of class labels.

    Returns:
        Path to created model_info.json.
    """
    if labels is None:
        labels = ["Background", "Lesion"]

    model_info = {
        "name": model_name,
        "type": "segmentation",
        "architecture": "meshnet",
        "channels": channels,
        "inputShape": list(input_shape),
        "outputLabels": labels,
        "numClasses": len(labels),
        "preprocessor": {
            "normalize": True,
            "normalizationMethod": "minmax",
            "targetRange": [0, 1],
        },
        "postprocessor": {
            "argmax": True,
        },
        "source": "arc-meshchop paper replication",
        "reference": "Fedorov et al. 2024",
    }

    output_path = Path(model_dir) / "model_info.json"
    output_path.write_text(json.dumps(model_info, indent=2))

    logger.info("Created BrainChop model info: %s", output_path)
    return output_path
```

### 2.3 Export Pipeline

**File:** `src/arc_meshchop/export/pipeline.py`

```python
"""Complete export pipeline for MeshNet models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arc_meshchop.models import MeshNet

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for model export."""

    output_dir: Path
    model_name: str = "meshnet"
    input_shape: tuple[int, int, int, int, int] = (1, 1, 256, 256, 256)

    # ONNX options
    export_onnx: bool = True
    onnx_opset: int = 17
    onnx_simplify: bool = True

    # TFJS options
    export_tfjs: bool = True
    tfjs_quantization: str | None = "float16"

    # Quantization options
    export_quantized: bool = True
    quantization_types: tuple[str, ...] = ("fp16",)

    # Validation
    validate_exports: bool = True

    def __post_init__(self) -> None:
        """Convert paths to Path objects."""
        self.output_dir = Path(self.output_dir)


def export_model(
    model: MeshNet,
    config: ExportConfig,
) -> dict[str, Path]:
    """Export model to all configured formats.

    Args:
        model: Trained MeshNet model.
        config: Export configuration.

    Returns:
        Dictionary mapping format names to output paths.
    """
    from arc_meshchop.export.onnx_export import (
        export_to_onnx,
        quantize_onnx,
        validate_onnx_export,
    )
    from arc_meshchop.export.tfjs_export import (
        create_brainchop_model_info,
        export_to_tfjs,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    # Export to ONNX
    if config.export_onnx:
        onnx_path = config.output_dir / f"{config.model_name}.onnx"
        export_to_onnx(
            model,
            onnx_path,
            input_shape=config.input_shape,
            opset_version=config.onnx_opset,
            simplify=config.onnx_simplify,
        )
        outputs["onnx"] = onnx_path

        # Validate
        if config.validate_exports:
            valid = validate_onnx_export(
                onnx_path,
                model,
                input_shape=(1, 1, 32, 32, 32),  # Smaller for speed
            )
            if not valid:
                logger.warning("ONNX validation failed for %s", onnx_path)

        # Quantized versions
        if config.export_quantized:
            for qtype in config.quantization_types:
                q_path = config.output_dir / f"{config.model_name}_{qtype}.onnx"
                quantize_onnx(onnx_path, q_path, quantization_type=qtype)
                outputs[f"onnx_{qtype}"] = q_path

    # Export to TensorFlow.js
    if config.export_tfjs and "onnx" in outputs:
        tfjs_dir = config.output_dir / f"{config.model_name}_tfjs"
        export_to_tfjs(
            outputs["onnx"],
            tfjs_dir,
            quantization=config.tfjs_quantization,
        )
        outputs["tfjs"] = tfjs_dir

        # Create BrainChop model info
        create_brainchop_model_info(
            tfjs_dir,
            model_name=f"MeshNet-{model.channels}",
            channels=model.channels,
            input_shape=config.input_shape[2:],
        )

    logger.info("Export complete. Outputs: %s", list(outputs.keys()))
    return outputs


def load_exported_model(
    path: Path | str,
    format: str = "onnx",
):
    """Load an exported model for inference.

    Args:
        path: Path to exported model.
        format: Export format ("onnx", "tfjs").

    Returns:
        Loaded model ready for inference.
    """
    path = Path(path)

    if format == "onnx":
        import onnxruntime as ort

        return ort.InferenceSession(str(path))

    elif format == "tfjs":
        raise NotImplementedError(
            "TFJS models must be loaded in JavaScript. "
            "Use tf.loadGraphModel() in browser."
        )

    else:
        raise ValueError(f"Unknown format: {format}")
```

### 2.4 Module `__init__.py`

**File:** `src/arc_meshchop/export/__init__.py`

```python
"""Model export utilities for deployment."""

from arc_meshchop.export.onnx_export import (
    export_to_onnx,
    quantize_onnx,
    validate_onnx_export,
)
from arc_meshchop.export.pipeline import (
    ExportConfig,
    export_model,
    load_exported_model,
)
from arc_meshchop.export.tfjs_export import (
    create_brainchop_model_info,
    export_pytorch_to_tfjs,
    export_to_tfjs,
)

__all__ = [
    "ExportConfig",
    "create_brainchop_model_info",
    "export_model",
    "export_pytorch_to_tfjs",
    "export_to_onnx",
    "export_to_tfjs",
    "load_exported_model",
    "quantize_onnx",
    "validate_onnx_export",
]
```

---

## 3. Tests

### 3.1 ONNX Export Tests

**File:** `tests/test_export/test_onnx.py`

```python
"""Tests for ONNX export."""

import pytest
import torch
from pathlib import Path

from arc_meshchop.models import meshnet_5
from arc_meshchop.export.onnx_export import export_to_onnx, validate_onnx_export


class TestONNXExport:
    """Tests for ONNX export functionality."""

    def test_export_creates_file(self, tmp_path: Path) -> None:
        """Verify ONNX export creates file."""
        model = meshnet_5()
        output_path = tmp_path / "model.onnx"

        export_to_onnx(
            model,
            output_path,
            input_shape=(1, 1, 32, 32, 32),
            simplify=False,
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_export_validates(self, tmp_path: Path) -> None:
        """Verify exported model matches PyTorch output."""
        model = meshnet_5()
        output_path = tmp_path / "model.onnx"

        export_to_onnx(
            model,
            output_path,
            input_shape=(1, 1, 32, 32, 32),
            simplify=False,
        )

        is_valid = validate_onnx_export(
            output_path,
            model,
            input_shape=(1, 1, 16, 16, 16),
        )

        assert is_valid

    def test_export_with_different_shapes(self, tmp_path: Path) -> None:
        """Test export with various input shapes."""
        model = meshnet_5()

        for size in [16, 32, 64]:
            output_path = tmp_path / f"model_{size}.onnx"
            export_to_onnx(
                model,
                output_path,
                input_shape=(1, 1, size, size, size),
                simplify=False,
            )
            assert output_path.exists()


@pytest.mark.slow
class TestONNXInference:
    """Tests for ONNX inference (slower, uses ONNX Runtime)."""

    def test_onnx_inference_matches_pytorch(self, tmp_path: Path) -> None:
        """Verify ONNX inference produces same results as PyTorch."""
        import numpy as np
        import onnxruntime as ort

        model = meshnet_5()
        model.eval()

        output_path = tmp_path / "model.onnx"
        export_to_onnx(
            model,
            output_path,
            input_shape=(1, 1, 32, 32, 32),
            simplify=False,
        )

        # Create test input
        test_input = torch.randn(1, 1, 32, 32, 32)

        # PyTorch inference
        with torch.no_grad():
            pytorch_output = model(test_input).numpy()

        # ONNX inference
        session = ort.InferenceSession(str(output_path))
        onnx_output = session.run(None, {"input": test_input.numpy()})[0]

        # Compare
        np.testing.assert_allclose(
            pytorch_output,
            onnx_output,
            rtol=1e-3,
            atol=1e-5,
        )
```

### 3.2 Export Pipeline Tests

**File:** `tests/test_export/test_pipeline.py`

```python
"""Tests for export pipeline."""

import pytest
from pathlib import Path

from arc_meshchop.models import meshnet_5
from arc_meshchop.export import ExportConfig, export_model


class TestExportPipeline:
    """Tests for complete export pipeline."""

    def test_export_onnx_only(self, tmp_path: Path) -> None:
        """Test ONNX-only export."""
        model = meshnet_5()
        config = ExportConfig(
            output_dir=tmp_path,
            model_name="test_model",
            input_shape=(1, 1, 32, 32, 32),
            export_onnx=True,
            export_tfjs=False,
            export_quantized=False,
            validate_exports=True,
        )

        outputs = export_model(model, config)

        assert "onnx" in outputs
        assert outputs["onnx"].exists()

    def test_export_with_quantization(self, tmp_path: Path) -> None:
        """Test export with FP16 quantization."""
        model = meshnet_5()
        config = ExportConfig(
            output_dir=tmp_path,
            model_name="test_model",
            input_shape=(1, 1, 32, 32, 32),
            export_onnx=True,
            export_tfjs=False,
            export_quantized=True,
            quantization_types=("fp16",),
            validate_exports=False,
        )

        outputs = export_model(model, config)

        assert "onnx_fp16" in outputs
        assert outputs["onnx_fp16"].exists()

        # FP16 should be smaller
        original_size = outputs["onnx"].stat().st_size
        fp16_size = outputs["onnx_fp16"].stat().st_size
        assert fp16_size < original_size
```

---

## 4. Implementation Checklist

### Phase 6.1: ONNX Export

- [ ] Create `src/arc_meshchop/export/onnx_export.py`
- [ ] Implement `export_to_onnx`
- [ ] Implement `validate_onnx_export`
- [ ] Implement `quantize_onnx`

### Phase 6.2: TensorFlow.js Export

- [ ] Create `src/arc_meshchop/export/tfjs_export.py`
- [ ] Implement `export_to_tfjs`
- [ ] Implement `create_brainchop_model_info`

### Phase 6.3: Export Pipeline

- [ ] Create `src/arc_meshchop/export/pipeline.py`
- [ ] Implement `ExportConfig`
- [ ] Implement `export_model`

### Phase 6.4: Tests

- [ ] Create ONNX export tests
- [ ] Create pipeline tests
- [ ] All tests pass

---

## 5. Dependencies for Export

Add to `pyproject.toml` under `[project.optional-dependencies]`:

```toml
export = [
    "onnx>=1.14.0",
    "onnxruntime>=1.16.0",
    "onnxsim>=0.4.33",           # ONNX simplification
    "onnxconverter-common>=1.13.0",  # FP16 conversion
    "onnx-tf>=1.10.0",           # ONNX to TensorFlow (for TFJS path)
    "tensorflowjs>=4.0.0",       # TensorFlow.js converter
]
```

> **NOTE: TFJS Export Compatibility**
> The ONNX → TensorFlow → TensorFlow.js conversion path requires validation for 3D convolution ops.
> - `onnx-tf` must support Conv3D with dilation (verify with installed version)
> - TensorFlow.js must support the 3D ops in WebGL backend
> - Test the full export pipeline before relying on TFJS deployment
>
> If TFJS conversion fails, the ONNX export path remains available for Python/ONNX Runtime inference.
> The paper does NOT specify a deployment format - ONNX is sufficient for most use cases.

> **PLATFORM NOTE: Mac Apple Silicon (M1/M2/M3/M4)**
>
> TensorFlow and related packages (`onnx-tf`, `tensorflowjs`) do not have official ARM Mac wheels
> with full 3D convolution support. This affects the TFJS export path on Mac.
>
> **Recommended workflow for Mac development:**
> 1. **Train on Mac (MPS)** - Works with PyTorch MPS backend ✅
> 2. **Export to ONNX on Mac** - Works everywhere ✅
> 3. **Convert ONNX → TFJS on Linux** - Use Docker, GitHub Actions CI, or a Linux server
>
> **Alternatives:**
> - Use ONNX Runtime Web (instead of TFJS) for browser deployment - no TensorFlow required
> - Use `tensorflow-macos` + `tensorflow-metal` for Mac TensorFlow, but verify Conv3D support
> - Run TFJS conversion in a Linux Docker container on Mac
>
> The ONNX export and ONNX Runtime inference work flawlessly on all platforms including Mac.

---

## 6. Verification Commands

```bash
# Install export dependencies
uv sync --extra export

# Run export tests
uv run pytest tests/test_export/ -v

# Export a model manually
uv run python -c "
from arc_meshchop.models import meshnet_16
from arc_meshchop.export import ExportConfig, export_model
from pathlib import Path

model = meshnet_16()
config = ExportConfig(
    output_dir=Path('exports'),
    model_name='meshnet_16',
    input_shape=(1, 1, 64, 64, 64),  # Smaller for testing
    export_onnx=True,
    export_tfjs=False,  # Requires TF dependencies
    export_quantized=True,
)

outputs = export_model(model, config)
print('Exported:', outputs)
"

# Validate ONNX export
uv run python -c "
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('exports/meshnet_16.onnx')
test_input = np.random.randn(1, 1, 64, 64, 64).astype(np.float32)
output = session.run(None, {'input': test_input})[0]
print('Output shape:', output.shape)
"
```

---

## 7. BrainChop Integration Notes

### Model Directory Structure

BrainChop expects this structure for custom models:

```
models/
└── meshnet_16/
    ├── model.json           # TFJS graph model
    ├── group1-shard1of1.bin # Weight files
    └── model_info.json      # Custom metadata
```

### Loading in BrainChop

```javascript
// In BrainChop JavaScript
const model = await tf.loadGraphModel('models/meshnet_16/model.json');
const modelInfo = await fetch('models/meshnet_16/model_info.json').then(r => r.json());

// Inference
const input = tf.tensor5d(mriData, [1, 1, 256, 256, 256]);
const output = model.predict(input);
const segmentation = output.argMax(-1);
```

---

## 8. References

- BrainChop: [github.com/neuroneural/brainchop](https://github.com/neuroneural/brainchop) (JS-based, local read-only copy in `_references/brainchop/`)
- ONNX documentation: https://onnx.ai/
- TensorFlow.js converter: https://www.tensorflow.org/js/guide/conversion
- ONNX Runtime: https://onnxruntime.ai/
