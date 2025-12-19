"""Data pipeline for ARC stroke lesion segmentation.

This module provides the complete data pipeline:
- Configuration (DataConfig, lesion quintiles)
- Preprocessing (resampling, normalization)
- Cross-validation splits (nested stratified)
- PyTorch Dataset and DataLoader
"""

from typing import Any

from arc_meshchop.data.config import (
    LESION_QUINTILES,
    AcquisitionType,
    DataConfig,
)
from arc_meshchop.data.dataset import (
    ARCDataset,
    create_dataloaders,
)

try:
    from arc_meshchop.data.huggingface_loader import (
        ARCDatasetInfo,
        ARCSample,
        download_arc_to_local,
        load_arc_from_huggingface,
        load_arc_samples,
        verify_sample_counts,
    )
except ImportError as exc:  # pragma: no cover - depends on optional deps
    _HF_IMPORT_ERROR = exc

    def _missing_hf(*_args: object, **_kwargs: object) -> Any:
        raise ImportError(
            "HuggingFace dataset utilities require optional dependency "
            "`neuroimaging-go-brrrr` (bids_hub). "
            "Install with: pip install 'arc-meshchop[huggingface]'"
        ) from _HF_IMPORT_ERROR

    class ARCDatasetInfo:  # type: ignore[no-redef]
        """Stub for ARCDatasetInfo when optional dependencies are missing."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            _missing_hf()

    class ARCSample:  # type: ignore[no-redef]
        """Stub for ARCSample when optional dependencies are missing."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            _missing_hf()

    download_arc_to_local = _missing_hf  # type: ignore[assignment]
    load_arc_from_huggingface = _missing_hf  # type: ignore[assignment]
    load_arc_samples = _missing_hf  # type: ignore[assignment]
    verify_sample_counts = _missing_hf  # type: ignore[assignment]
from arc_meshchop.data.preprocessing import (
    compute_lesion_volume,
    crop_or_pad,
    get_lesion_quintile,
    get_spacing,
    load_nifti,
    normalize_intensity,
    preprocess_volume,
    resample_volume,
    save_nifti,
)
from arc_meshchop.data.splits import (
    CVSplit,
    InnerFold,
    NestedCVSplits,
    OuterFold,
    create_stratification_labels,
    generate_nested_cv_splits,
)

__all__ = [
    "LESION_QUINTILES",
    "ARCDataset",
    "ARCDatasetInfo",
    "ARCSample",
    "AcquisitionType",
    "CVSplit",
    "DataConfig",
    "InnerFold",
    "NestedCVSplits",
    "OuterFold",
    "compute_lesion_volume",
    "create_dataloaders",
    "create_stratification_labels",
    "crop_or_pad",
    "download_arc_to_local",
    "generate_nested_cv_splits",
    "get_lesion_quintile",
    "get_spacing",
    "load_arc_from_huggingface",
    "load_arc_samples",
    "load_nifti",
    "normalize_intensity",
    "preprocess_volume",
    "resample_volume",
    "save_nifti",
    "verify_sample_counts",
]
