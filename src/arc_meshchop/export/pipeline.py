"""Complete export pipeline for MeshNet models.

Orchestrates export to multiple formats (ONNX, TFJS) with optional
quantization and validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import onnxruntime as ort

if TYPE_CHECKING:
    from arc_meshchop.models import MeshNet

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for model export.

    Attributes:
        output_dir: Directory for exported models.
        model_name: Base name for exported files.
        input_shape: Input tensor shape (B, C, D, H, W).
        export_onnx: Whether to export ONNX format.
        onnx_opset: ONNX opset version.
        onnx_simplify: Whether to simplify ONNX graph.
        export_tfjs: Whether to export TensorFlow.js format.
        tfjs_quantization: Quantization for TFJS ("float16", "uint8", etc).
        export_quantized: Whether to export quantized ONNX versions.
        quantization_types: Types of quantization ("fp16", "int8").
        validate_exports: Whether to validate exports match PyTorch.
    """

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
    quantization_types: tuple[str, ...] = field(default_factory=lambda: ("fp16",))

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
        try:
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
        except ImportError as e:
            logger.warning("TFJS export skipped: %s", e)

    logger.info("Export complete. Outputs: %s", list(outputs.keys()))
    return outputs


def load_exported_model(
    path: Path | str,
    output_format: str = "onnx",
) -> ort.InferenceSession:
    """Load an exported model for inference.

    Args:
        path: Path to exported model.
        output_format: Export format ("onnx", "tfjs").

    Returns:
        ONNX Runtime InferenceSession ready for inference.

    Raises:
        ValueError: If output_format is unknown.
        NotImplementedError: If output_format is "tfjs" (must load in JavaScript).
        FileNotFoundError: If the model file does not exist (for ONNX format).
    """
    path = Path(path)

    # Validate format first before checking path existence
    if output_format == "tfjs":
        raise NotImplementedError(
            "TFJS models must be loaded in JavaScript. Use tf.loadGraphModel() in browser."
        )

    if output_format != "onnx":
        raise ValueError(f"Unknown format: {output_format}")

    # For ONNX, check file existence and load
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    return ort.InferenceSession(str(path))
