"""Tests for export pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from arc_meshchop.models import meshnet_5, meshnet_16


class TestExportConfig:
    """Tests for ExportConfig dataclass."""

    def test_default_values(self, tmp_path: Path) -> None:
        """Test ExportConfig default values."""
        from arc_meshchop.export.pipeline import ExportConfig

        config = ExportConfig(output_dir=tmp_path)

        assert config.output_dir == tmp_path
        assert config.model_name == "meshnet"
        assert config.input_shape == (1, 1, 256, 256, 256)
        assert config.export_onnx is True
        assert config.onnx_opset == 17
        assert config.onnx_simplify is True
        assert config.export_tfjs is True
        assert config.tfjs_quantization == "float16"
        assert config.export_quantized is True
        assert config.quantization_types == ("fp16",)
        assert config.validate_exports is True

    def test_path_conversion(self, tmp_path: Path) -> None:
        """Test that string paths are converted to Path objects."""
        from arc_meshchop.export.pipeline import ExportConfig

        # String is converted to Path via __post_init__
        config = ExportConfig(output_dir=str(tmp_path))  # type: ignore[arg-type]

        assert isinstance(config.output_dir, Path)
        assert config.output_dir == tmp_path

    def test_custom_values(self, tmp_path: Path) -> None:
        """Test ExportConfig with custom values."""
        from arc_meshchop.export.pipeline import ExportConfig

        config = ExportConfig(
            output_dir=tmp_path,
            model_name="custom_model",
            input_shape=(1, 1, 128, 128, 128),
            export_onnx=True,
            onnx_opset=15,
            onnx_simplify=False,
            export_tfjs=False,
            export_quantized=True,
            quantization_types=("fp16", "int8"),
            validate_exports=False,
        )

        assert config.model_name == "custom_model"
        assert config.input_shape == (1, 1, 128, 128, 128)
        assert config.onnx_opset == 15
        assert config.onnx_simplify is False
        assert config.export_tfjs is False
        assert config.quantization_types == ("fp16", "int8")
        assert config.validate_exports is False


class TestExportPipeline:
    """Tests for complete export pipeline."""

    def test_export_onnx_only(self, tmp_path: Path) -> None:
        """Test ONNX-only export."""
        from arc_meshchop.export import ExportConfig, export_model

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

    def test_export_with_validation_disabled(self, tmp_path: Path) -> None:
        """Test export with validation disabled."""
        from arc_meshchop.export import ExportConfig, export_model

        model = meshnet_5()
        config = ExportConfig(
            output_dir=tmp_path,
            model_name="test_model",
            input_shape=(1, 1, 32, 32, 32),
            export_onnx=True,
            export_tfjs=False,
            export_quantized=False,
            validate_exports=False,
        )

        outputs = export_model(model, config)

        assert "onnx" in outputs
        assert outputs["onnx"].exists()

    def test_export_with_quantization(self, tmp_path: Path) -> None:
        """Test export with FP16 quantization."""
        from arc_meshchop.export import ExportConfig, export_model

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

        # FP16 model should be created
        assert "onnx_fp16" in outputs
        assert outputs["onnx_fp16"].exists()
        assert outputs["onnx_fp16"].stat().st_size > 0

    def test_export_with_multiple_quantizations(self, tmp_path: Path) -> None:
        """Test export with multiple quantization types."""
        from arc_meshchop.export import ExportConfig, export_model

        model = meshnet_5()
        config = ExportConfig(
            output_dir=tmp_path,
            model_name="test_model",
            input_shape=(1, 1, 32, 32, 32),
            export_onnx=True,
            export_tfjs=False,
            export_quantized=True,
            quantization_types=("fp16", "int8"),
            validate_exports=False,
        )

        outputs = export_model(model, config)

        assert "onnx_fp16" in outputs
        assert "onnx_int8" in outputs
        assert outputs["onnx_fp16"].exists()
        assert outputs["onnx_int8"].exists()

    def test_export_creates_output_directory(self, tmp_path: Path) -> None:
        """Test that export creates output directory if it doesn't exist."""
        from arc_meshchop.export import ExportConfig, export_model

        model = meshnet_5()
        output_dir = tmp_path / "nested" / "output"

        config = ExportConfig(
            output_dir=output_dir,
            model_name="test_model",
            input_shape=(1, 1, 32, 32, 32),
            export_onnx=True,
            export_tfjs=False,
            export_quantized=False,
            validate_exports=False,
        )

        outputs = export_model(model, config)

        assert output_dir.exists()
        assert "onnx" in outputs

    def test_export_with_different_model_name(self, tmp_path: Path) -> None:
        """Test export with custom model name."""
        from arc_meshchop.export import ExportConfig, export_model

        model = meshnet_16()
        config = ExportConfig(
            output_dir=tmp_path,
            model_name="meshnet_16_custom",
            input_shape=(1, 1, 32, 32, 32),
            export_onnx=True,
            export_tfjs=False,
            export_quantized=False,
            validate_exports=False,
        )

        outputs = export_model(model, config)

        assert outputs["onnx"].name == "meshnet_16_custom.onnx"


class TestLoadExportedModel:
    """Tests for loading exported models."""

    def test_load_onnx_model(self, tmp_path: Path) -> None:
        """Test loading ONNX model for inference."""
        from arc_meshchop.export import ExportConfig, export_model, load_exported_model

        model = meshnet_5()
        config = ExportConfig(
            output_dir=tmp_path,
            model_name="test_model",
            input_shape=(1, 1, 32, 32, 32),
            export_onnx=True,
            export_tfjs=False,
            export_quantized=False,
            validate_exports=False,
        )

        outputs = export_model(model, config)

        # Load and run inference
        session = load_exported_model(outputs["onnx"], format="onnx")
        test_input = np.random.randn(1, 1, 32, 32, 32).astype(np.float32)
        result = session.run(None, {"input": test_input})[0]

        assert result.shape == (1, 2, 32, 32, 32)

    def test_load_tfjs_raises_not_implemented(self, tmp_path: Path) -> None:
        """Test that loading TFJS model raises NotImplementedError."""
        from arc_meshchop.export import load_exported_model

        with pytest.raises(NotImplementedError, match="TFJS models must be loaded in JavaScript"):
            load_exported_model(tmp_path / "model_tfjs", format="tfjs")

    def test_load_unknown_format_raises_error(self, tmp_path: Path) -> None:
        """Test that loading unknown format raises ValueError."""
        from arc_meshchop.export import load_exported_model

        with pytest.raises(ValueError, match="Unknown format"):
            load_exported_model(tmp_path / "model", format="unknown")


class TestBrainChopModelInfo:
    """Tests for BrainChop model info JSON."""

    def test_create_brainchop_model_info(self, tmp_path: Path) -> None:
        """Test creating BrainChop-compatible model info."""
        from arc_meshchop.export.tfjs_export import create_brainchop_model_info

        result = create_brainchop_model_info(
            model_dir=tmp_path,
            model_name="MeshNet-5",
            channels=5,
            input_shape=(256, 256, 256),
        )

        assert result.exists()
        assert result.name == "model_info.json"

        # Verify JSON content
        with result.open() as f:
            info = json.load(f)

        assert info["name"] == "MeshNet-5"
        assert info["type"] == "segmentation"
        assert info["architecture"] == "meshnet"
        assert info["channels"] == 5
        assert info["inputShape"] == [256, 256, 256]
        assert info["outputLabels"] == ["Background", "Lesion"]
        assert info["numClasses"] == 2

    def test_create_brainchop_model_info_custom_labels(self, tmp_path: Path) -> None:
        """Test creating BrainChop model info with custom labels."""
        from arc_meshchop.export.tfjs_export import create_brainchop_model_info

        custom_labels = ["Class0", "Class1", "Class2"]
        result = create_brainchop_model_info(
            model_dir=tmp_path,
            model_name="TestModel",
            channels=16,
            input_shape=(128, 128, 128),
            labels=custom_labels,
        )

        with result.open() as f:
            info = json.load(f)

        assert info["outputLabels"] == custom_labels
        assert info["numClasses"] == 3

    def test_model_info_preprocessor_settings(self, tmp_path: Path) -> None:
        """Test that model info includes correct preprocessor settings."""
        from arc_meshchop.export.tfjs_export import create_brainchop_model_info

        result = create_brainchop_model_info(
            model_dir=tmp_path,
            model_name="MeshNet-26",
            channels=26,
        )

        with result.open() as f:
            info = json.load(f)

        assert "preprocessor" in info
        assert info["preprocessor"]["normalize"] is True
        assert info["preprocessor"]["normalizationMethod"] == "minmax"
        assert info["preprocessor"]["targetRange"] == [0, 1]

        assert "postprocessor" in info
        assert info["postprocessor"]["argmax"] is True


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="TFJS export requires TensorFlow which has no ARM Mac wheels",
)
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="TFJS export requires TensorFlow which may have issues on Windows",
)
class TestTFJSExport:
    """Tests for TensorFlow.js export (Linux only)."""

    @pytest.mark.slow
    def test_export_pytorch_to_tfjs(self, tmp_path: Path) -> None:
        """Test full PyTorch to TFJS export pipeline."""
        # This test is skipped on Mac/Windows and requires TF dependencies
        pytest.skip("TFJS export requires TensorFlow dependencies not in default install")

    @pytest.mark.slow
    def test_export_to_tfjs(self, tmp_path: Path) -> None:
        """Test ONNX to TFJS conversion."""
        # This test is skipped on Mac/Windows and requires TF dependencies
        pytest.skip("TFJS export requires TensorFlow dependencies not in default install")
