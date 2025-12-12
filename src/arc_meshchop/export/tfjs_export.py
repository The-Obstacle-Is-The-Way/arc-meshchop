"""TensorFlow.js export for browser deployment.

Converts PyTorch models to TensorFlow.js format via ONNX intermediate.
This enables BrainChop integration for client-side MRI segmentation.

FROM BRAINCHOP REFERENCE:
BrainChop uses TensorFlow.js for browser-based inference with WebGL
acceleration. Models are stored as JSON + binary weight files.

PLATFORM NOTE:
TensorFlow and related packages (onnx-tf, tensorflowjs) do not have
official ARM Mac wheels. TFJS export only works on Linux.
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

    Uses tensorflowjs_converter to convert ONNX -> TF SavedModel -> TFJS.

    Args:
        onnx_path: Path to ONNX model.
        output_dir: Directory for TFJS output files.
        quantization: Optional quantization ("float16", "uint8", "uint16").

    Returns:
        Path to output directory containing model.json and weight files.

    Raises:
        ImportError: If onnx-tf is not installed.
        subprocess.CalledProcessError: If tensorflowjs_converter fails.
    """
    onnx_path = Path(onnx_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Intermediate SavedModel path
    saved_model_dir = output_dir / "saved_model_temp"

    logger.info("Converting ONNX to TensorFlow SavedModel...")

    # Step 1: ONNX -> TensorFlow SavedModel
    try:
        import onnx
        from onnx_tf.backend import prepare

        onnx_model = onnx.load(str(onnx_path))
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(str(saved_model_dir))

    except ImportError as e:
        # Fallback to command-line tool
        logger.info("onnx-tf not installed as Python package, trying CLI...")
        try:
            subprocess.run(
                [
                    "onnx-tf",
                    "convert",
                    "-i",
                    str(onnx_path),
                    "-o",
                    str(saved_model_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            raise ImportError(
                "onnx-tf is not installed. Install with: pip install onnx-tf\n"
                "Note: This requires TensorFlow which may not work on ARM Mac."
            ) from e

    logger.info("Converting TensorFlow SavedModel to TFJS...")

    # Step 2: TensorFlow SavedModel -> TensorFlow.js
    tfjs_cmd = [
        "tensorflowjs_converter",
        "--input_format=tf_saved_model",
        "--output_format=tfjs_graph_model",
    ]

    if quantization:
        tfjs_cmd.extend(["--quantization_dtype", quantization])

    tfjs_cmd.extend([str(saved_model_dir), str(output_dir)])

    try:
        subprocess.run(tfjs_cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise ImportError(
            "tensorflowjs_converter is not installed. Install with: pip install tensorflowjs\n"
            "Note: This requires TensorFlow which may not work on ARM Mac."
        ) from e

    # Cleanup intermediate SavedModel
    if saved_model_dir.exists():
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
    export_to_onnx(model, onnx_path, input_shape=input_shape, simplify=False)

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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(model_info, indent=2))

    logger.info("Created BrainChop model info: %s", output_path)
    return output_path
