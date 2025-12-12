"""Tests for ONNX export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from arc_meshchop.models import meshnet_5


class TestONNXExport:
    """Tests for ONNX export functionality."""

    def test_export_creates_file(self, tmp_path: Path) -> None:
        """Verify ONNX export creates file."""
        from arc_meshchop.export.onnx_export import export_to_onnx

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
        from arc_meshchop.export.onnx_export import export_to_onnx, validate_onnx_export

        model = meshnet_5()
        output_path = tmp_path / "model.onnx"

        # Use same shape for export and validation
        # (dynamic axes not fully supported with new torch.onnx.export)
        export_to_onnx(
            model,
            output_path,
            input_shape=(1, 1, 32, 32, 32),
            simplify=False,
        )

        is_valid = validate_onnx_export(
            output_path,
            model,
            input_shape=(1, 1, 32, 32, 32),
        )

        assert is_valid

    def test_export_with_different_shapes(self, tmp_path: Path) -> None:
        """Test export with various input shapes."""
        from arc_meshchop.export.onnx_export import export_to_onnx

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

    def test_export_with_dynamic_axes(self, tmp_path: Path) -> None:
        """Test export with custom dynamic axes."""
        from arc_meshchop.export.onnx_export import export_to_onnx

        model = meshnet_5()
        output_path = tmp_path / "model_dynamic.onnx"

        custom_dynamic_axes = {
            "input": {0: "batch_size", 2: "depth", 3: "height", 4: "width"},
            "output": {0: "batch_size", 2: "depth", 3: "height", 4: "width"},
        }

        export_to_onnx(
            model,
            output_path,
            input_shape=(1, 1, 32, 32, 32),
            dynamic_axes=custom_dynamic_axes,
            simplify=False,
        )

        assert output_path.exists()

    def test_export_opset_version(self, tmp_path: Path) -> None:
        """Test export with different opset versions."""
        from arc_meshchop.export.onnx_export import export_to_onnx

        model = meshnet_5()

        for opset in [14, 15, 17]:
            output_path = tmp_path / f"model_opset{opset}.onnx"
            export_to_onnx(
                model,
                output_path,
                input_shape=(1, 1, 32, 32, 32),
                opset_version=opset,
                simplify=False,
            )
            assert output_path.exists()

    def test_export_creates_parent_directory(self, tmp_path: Path) -> None:
        """Test that export creates parent directories if needed."""
        from arc_meshchop.export.onnx_export import export_to_onnx

        model = meshnet_5()
        output_path = tmp_path / "nested" / "dir" / "model.onnx"

        export_to_onnx(
            model,
            output_path,
            input_shape=(1, 1, 32, 32, 32),
            simplify=False,
        )

        assert output_path.exists()


@pytest.mark.slow
class TestONNXInference:
    """Tests for ONNX inference (slower, uses ONNX Runtime)."""

    def test_onnx_inference_matches_pytorch(self, tmp_path: Path) -> None:
        """Verify ONNX inference produces same results as PyTorch."""
        import onnxruntime as ort

        from arc_meshchop.export.onnx_export import export_to_onnx

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

    def test_onnx_inference_output_shape(self, tmp_path: Path) -> None:
        """Verify ONNX model produces correct output shape."""
        import onnxruntime as ort

        from arc_meshchop.export.onnx_export import export_to_onnx

        model = meshnet_5()
        output_path = tmp_path / "model.onnx"

        export_to_onnx(
            model,
            output_path,
            input_shape=(1, 1, 32, 32, 32),
            simplify=False,
        )

        # ONNX inference
        session = ort.InferenceSession(str(output_path))
        test_input = np.random.randn(1, 1, 32, 32, 32).astype(np.float32)
        onnx_output = session.run(None, {"input": test_input})[0]

        # MeshNet outputs 2 channels (background, lesion)
        assert onnx_output.shape == (1, 2, 32, 32, 32)


class TestONNXQuantization:
    """Tests for ONNX quantization."""

    def test_fp16_quantization(self, tmp_path: Path) -> None:
        """Test FP16 quantization creates valid model."""
        from arc_meshchop.export.onnx_export import export_to_onnx, quantize_onnx

        model = meshnet_5()
        onnx_path = tmp_path / "model.onnx"
        fp16_path = tmp_path / "model_fp16.onnx"

        export_to_onnx(
            model,
            onnx_path,
            input_shape=(1, 1, 32, 32, 32),
            simplify=False,
        )

        quantize_onnx(onnx_path, fp16_path, quantization_type="fp16")

        # FP16 model should be created and runnable
        assert fp16_path.exists()
        assert fp16_path.stat().st_size > 0

    def test_int8_quantization(self, tmp_path: Path) -> None:
        """Test INT8 quantization creates valid model."""
        from arc_meshchop.export.onnx_export import export_to_onnx, quantize_onnx

        model = meshnet_5()
        onnx_path = tmp_path / "model.onnx"
        int8_path = tmp_path / "model_int8.onnx"

        export_to_onnx(
            model,
            onnx_path,
            input_shape=(1, 1, 32, 32, 32),
            simplify=False,
        )

        quantize_onnx(onnx_path, int8_path, quantization_type="int8")

        assert int8_path.exists()
        assert int8_path.stat().st_size > 0

    def test_invalid_quantization_type(self, tmp_path: Path) -> None:
        """Test that invalid quantization type raises error."""
        from arc_meshchop.export.onnx_export import export_to_onnx, quantize_onnx

        model = meshnet_5()
        onnx_path = tmp_path / "model.onnx"
        invalid_path = tmp_path / "model_invalid.onnx"

        export_to_onnx(
            model,
            onnx_path,
            input_shape=(1, 1, 32, 32, 32),
            simplify=False,
        )

        with pytest.raises(ValueError, match="Unknown quantization type"):
            quantize_onnx(onnx_path, invalid_path, quantization_type="invalid")


class TestValidateONNXExport:
    """Tests for ONNX export validation."""

    def test_validation_with_matching_outputs(self, tmp_path: Path) -> None:
        """Test validation returns True for matching outputs."""
        from arc_meshchop.export.onnx_export import export_to_onnx, validate_onnx_export

        model = meshnet_5()
        output_path = tmp_path / "model.onnx"

        # Use same shape for export and validation
        # (dynamic axes not fully supported with new torch.onnx.export)
        export_to_onnx(
            model,
            output_path,
            input_shape=(1, 1, 32, 32, 32),
            simplify=False,
        )

        result = validate_onnx_export(
            output_path,
            model,
            input_shape=(1, 1, 32, 32, 32),
            rtol=1e-3,
            atol=1e-5,
        )

        assert result is True

    def test_validation_with_custom_tolerances(self, tmp_path: Path) -> None:
        """Test validation with custom tolerances."""
        from arc_meshchop.export.onnx_export import export_to_onnx, validate_onnx_export

        model = meshnet_5()
        output_path = tmp_path / "model.onnx"

        export_to_onnx(
            model,
            output_path,
            input_shape=(1, 1, 32, 32, 32),
            simplify=False,
        )

        # Validate with looser tolerances
        result = validate_onnx_export(
            output_path,
            model,
            input_shape=(1, 1, 32, 32, 32),
            rtol=1e-2,
            atol=1e-4,
        )

        assert result is True
