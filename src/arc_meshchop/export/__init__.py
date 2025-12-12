"""Model export utilities for deployment.

Provides functions for exporting MeshNet models to:
- ONNX format for cross-platform inference
- TensorFlow.js format for browser deployment (Linux only)
- Quantized variants (FP16, INT8) for smaller models

Example:
    >>> from arc_meshchop.models import meshnet_26
    >>> from arc_meshchop.export import ExportConfig, export_model
    >>> model = meshnet_26()
    >>> config = ExportConfig(
    ...     output_dir="exports",
    ...     model_name="meshnet_26",
    ...     export_onnx=True,
    ...     export_tfjs=False,  # Requires TF dependencies
    ... )
    >>> outputs = export_model(model, config)
"""

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
