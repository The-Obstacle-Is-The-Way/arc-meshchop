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
    dynamic_axes: dict[str, dict[int, str]] | None = None,
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
        (dummy_input,),  # Wrap in tuple for type safety
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
            import onnx
            import onnxsim

            logger.info("Simplifying ONNX model...")
            model_onnx = onnx.load(str(output_path))
            simplified_model, check = onnxsim.simplify(model_onnx)
            if check:
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
    matches = bool(np.allclose(pytorch_output, onnx_output, rtol=rtol, atol=atol))

    if matches:
        logger.info("ONNX validation passed: outputs match within tolerance")
    else:
        max_diff = float(np.max(np.abs(pytorch_output - onnx_output)))
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

    Raises:
        ValueError: If quantization_type is not "fp16" or "int8".
    """
    import onnx

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if quantization_type == "fp16":
        logger.info("Quantizing to FP16...")
        try:
            from onnxconverter_common import float16

            model = onnx.load(str(input_path))
            model_fp16 = float16.convert_float_to_float16(model)
            onnx.save(model_fp16, str(output_path))
        except ImportError:
            # Fallback: use onnxruntime's fp16 conversion
            logger.warning(
                "onnxconverter-common not installed, using onnxruntime for FP16 conversion"
            )
            from onnxruntime.transformers import float16

            model = onnx.load(str(input_path))
            model_fp16 = float16.convert_float_to_float16(model)
            onnx.save(model_fp16, str(output_path))

    elif quantization_type == "int8":
        logger.info("Quantizing to INT8...")
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(
            str(input_path),
            str(output_path),
            weight_type=QuantType.QUInt8,
        )

    else:
        raise ValueError(f"Unknown quantization type: {quantization_type}")

    logger.info("Quantization complete: %s", output_path)
    return output_path
