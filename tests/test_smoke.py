"""Smoke tests to verify basic project setup."""

import pytest


def test_import_package() -> None:
    """Test that the main package can be imported."""
    import arc_meshchop

    assert arc_meshchop.__version__ == "0.1.0"


def test_import_submodules() -> None:
    """Test that all submodules can be imported."""
    import arc_meshchop.data
    import arc_meshchop.evaluation
    import arc_meshchop.export
    import arc_meshchop.models
    import arc_meshchop.training

    # Verify they're real modules
    assert arc_meshchop.data is not None
    assert arc_meshchop.evaluation is not None
    assert arc_meshchop.export is not None
    assert arc_meshchop.models is not None
    assert arc_meshchop.training is not None


def test_cli_import() -> None:
    """Test that CLI can be imported."""
    from arc_meshchop.cli import app

    assert app is not None


def test_torch_available() -> None:
    """Test that PyTorch is available."""
    import torch

    assert torch.__version__ is not None


def test_cuda_detection() -> None:
    """Test CUDA detection (should not fail even without GPU)."""
    import torch

    # Just verify the check doesn't error
    _ = torch.cuda.is_available()


@pytest.mark.slow
def test_monai_available() -> None:
    """Test that MONAI is available."""
    import monai

    assert monai.__version__ is not None
