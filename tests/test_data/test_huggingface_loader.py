"""Tests for HuggingFace loader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestLoadARCFromHuggingFace:
    """Tests for load_arc_from_huggingface function."""

    def test_imports_datasets_library(self) -> None:
        """Verify datasets import error is informative."""
        with patch.dict("sys.modules", {"datasets": None}):
            # Need to reimport to trigger the import check
            import importlib

            import arc_meshchop.data.huggingface_loader as loader

            importlib.reload(loader)

            # Should raise ImportError with helpful message
            with pytest.raises(ImportError, match="datasets"):
                loader.load_arc_from_huggingface()

    def test_returns_arc_dataset_info(self) -> None:
        """Verify return type is ARCDatasetInfo."""
        # Mock the HuggingFace dataset
        mock_dataset = _create_mock_dataset(n_samples=10)
        mock_return = {"train": mock_dataset}

        with patch(
            "arc_meshchop.data.huggingface_loader.load_dataset",
            return_value=mock_return,
        ):
            from arc_meshchop.data.huggingface_loader import load_arc_from_huggingface

            info = load_arc_from_huggingface(verify_counts=False)

            assert hasattr(info, "samples")
            assert hasattr(info, "image_paths")
            assert hasattr(info, "mask_paths")

    def test_filters_by_acquisition_type(self) -> None:
        """Verify acquisition type filtering works."""
        mock_dataset = _create_mock_dataset(
            n_samples=10,
            acquisition_types=["space_2x"] * 5 + ["turbo_spin_echo"] * 5,
        )
        mock_return = {"train": mock_dataset}

        with patch(
            "arc_meshchop.data.huggingface_loader.load_dataset",
            return_value=mock_return,
        ):
            from arc_meshchop.data.huggingface_loader import load_arc_from_huggingface

            info = load_arc_from_huggingface(
                exclude_turbo_spin_echo=True,
                verify_counts=False,
            )

            # Should exclude turbo-spin echo
            assert all(s.acquisition_type != "turbo_spin_echo" for s in info.samples)

    def test_requires_lesion_mask_by_default(self) -> None:
        """Verify samples without masks are excluded by default."""
        mock_dataset = _create_mock_dataset(
            n_samples=10,
            has_mask=[True] * 5 + [False] * 5,
        )
        mock_return = {"train": mock_dataset}

        with patch(
            "arc_meshchop.data.huggingface_loader.load_dataset",
            return_value=mock_return,
        ):
            from arc_meshchop.data.huggingface_loader import load_arc_from_huggingface

            info = load_arc_from_huggingface(
                require_lesion_mask=True,
                verify_counts=False,
            )

            # Should only have samples with masks
            assert len(info) == 5


class TestARCDatasetInfo:
    """Tests for ARCDatasetInfo dataclass."""

    def test_property_accessors(self) -> None:
        """Verify property accessors work correctly."""
        from arc_meshchop.data.huggingface_loader import ARCDatasetInfo, ARCSample

        samples = [
            ARCSample(
                subject_id="sub-001",
                session_id="ses-1",
                image_path=Path("/path/image.nii.gz"),
                mask_path=Path("/path/mask.nii.gz"),
                lesion_volume=1000,
                acquisition_type="space_2x",
            ),
        ]

        info = ARCDatasetInfo(samples=samples)

        assert len(info) == 1
        assert info.image_paths == [Path("/path/image.nii.gz")]
        assert info.mask_paths == [Path("/path/mask.nii.gz")]
        assert info.lesion_volumes == [1000]
        assert info.acquisition_types == ["space_2x"]


class TestDetermineAcquisitionType:
    """Tests for acquisition type detection."""

    def test_detects_space_2x_from_filename(self) -> None:
        """Verify SPACE 2x detection from BIDS filename."""
        from arc_meshchop.data.huggingface_loader import _determine_acquisition_type

        acq_type = _determine_acquisition_type({}, "/cache/sub-001_ses-1_acq-space2x_T2w.nii.gz")
        assert acq_type == "space_2x"

    def test_detects_space_no_accel_from_filename(self) -> None:
        """Verify SPACE no accel detection from BIDS filename."""
        from arc_meshchop.data.huggingface_loader import _determine_acquisition_type

        acq_type = _determine_acquisition_type({}, "/cache/sub-001_ses-1_acq-space_T2w.nii.gz")
        assert acq_type == "space_no_accel"

    def test_detects_tse_from_filename(self) -> None:
        """Verify TSE detection from BIDS filename."""
        from arc_meshchop.data.huggingface_loader import _determine_acquisition_type

        acq_type = _determine_acquisition_type({}, "/cache/sub-001_ses-1_acq-tse_T2w.nii.gz")
        assert acq_type == "turbo_spin_echo"

    def test_fallback_to_metadata(self) -> None:
        """Verify fallback to row metadata when filename doesn't have acq."""
        from arc_meshchop.data.huggingface_loader import _determine_acquisition_type

        acq_type = _determine_acquisition_type(
            {"acquisition": "SPACE2x"}, "/cache/sub-001_ses-1_T2w.nii.gz"
        )
        assert acq_type == "space_2x"

    def test_returns_unknown_when_undetectable(self) -> None:
        """Verify unknown returned when acquisition cannot be determined."""
        from arc_meshchop.data.huggingface_loader import _determine_acquisition_type

        acq_type = _determine_acquisition_type({}, "/cache/sub-001_ses-1_T2w.nii.gz")
        assert acq_type == "unknown"


class TestVerifySampleCounts:
    """Tests for sample count verification."""

    def test_passes_with_correct_counts(self) -> None:
        """Verify no error with correct sample counts."""
        from arc_meshchop.data.huggingface_loader import ARCSample, verify_sample_counts

        space_2x = [
            ARCSample("sub", "ses", Path("i"), Path("m"), 100, "space_2x") for _ in range(115)
        ]
        space_no = [
            ARCSample("sub", "ses", Path("i"), Path("m"), 100, "space_no_accel") for _ in range(109)
        ]
        samples = space_2x + space_no

        # Should not raise
        verify_sample_counts(samples)

    def test_raises_with_wrong_total(self) -> None:
        """Verify error raised with wrong total count."""
        from arc_meshchop.data.huggingface_loader import ARCSample, verify_sample_counts

        samples = [
            ARCSample("sub", "ses", Path("i"), Path("m"), 100, "space_2x") for _ in range(100)
        ]

        with pytest.raises(ValueError, match="Sample count verification failed"):
            verify_sample_counts(samples)


def _create_mock_dataset(
    n_samples: int = 10,
    acquisition_types: list[str] | None = None,
    has_mask: list[bool] | None = None,
) -> MagicMock:
    """Create mock HuggingFace dataset for testing."""
    if acquisition_types is None:
        acquisition_types = ["space_2x"] * n_samples
    if has_mask is None:
        has_mask = [True] * n_samples

    mock = MagicMock()
    mock.__len__ = MagicMock(return_value=n_samples)

    def getitem(idx: int) -> dict:
        acq = acquisition_types[idx] if idx < len(acquisition_types) else "space_2x"
        if acq == "space_2x":
            acq_str = "space2x"
        elif acq == "turbo_spin_echo":
            acq_str = "tse"
        else:
            acq_str = "space"
        return {
            "subject_id": f"sub-{idx:04d}",
            "session_id": "ses-1",
            "t2w": f"/path/sub-{idx:04d}_ses-1_acq-{acq_str}_T2w.nii.gz",
            "lesion": f"/path/mask_{idx}.nii.gz" if has_mask[idx] else None,
            "acquisition": acquisition_types[idx],
        }

    mock.__getitem__ = getitem
    return mock
