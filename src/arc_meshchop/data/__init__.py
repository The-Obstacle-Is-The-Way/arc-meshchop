"""Data pipeline for ARC stroke lesion segmentation.

This module provides the complete data pipeline:
- Configuration (DataConfig, lesion quintiles)
- Preprocessing (resampling, normalization)
- Cross-validation splits (nested stratified)
- PyTorch Dataset and DataLoader
"""

from arc_meshchop.data.config import (
    LESION_QUINTILES,
    AcquisitionType,
    DataConfig,
)
from arc_meshchop.data.dataset import (
    ARCDataset,
    create_dataloaders,
)
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
    "generate_nested_cv_splits",
    "get_lesion_quintile",
    "get_spacing",
    "load_nifti",
    "normalize_intensity",
    "preprocess_volume",
    "resample_volume",
    "save_nifti",
]
