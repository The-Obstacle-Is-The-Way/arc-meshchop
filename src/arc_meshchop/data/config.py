"""Data pipeline configuration.

Configuration for the ARC stroke lesion segmentation data pipeline.

FROM PAPER (Section 2):
- 230 subjects total, 224 SPACE scans used (excluding 5 turbo-spin-echo)
- Resampled to 256x256x256 @ 1mm isotropic
- 0-1 min-max normalization
- Nested 3x3 cross-validation with stratification
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class DataConfig:
    """Configuration for data pipeline.

    Attributes:
        data_root: Root directory for data storage.
        cache_dir: Directory for caching preprocessed data.
        hf_repo: HuggingFace Hub repository ID.
        target_shape: Target volume shape after resampling.
        target_spacing: Target voxel spacing in mm.
        num_outer_folds: Number of outer CV folds.
        num_inner_folds: Number of inner CV folds per outer.
        random_seed: Random seed for reproducibility.
        include_space_2x: Include SPACE with 2x acceleration.
        include_space_no_accel: Include SPACE without acceleration.
        exclude_turbo_spin_echo: Exclude turbo-spin-echo sequences.
    """

    data_root: Path = field(default_factory=lambda: Path("data"))
    cache_dir: Path = field(default_factory=lambda: Path("data/processed"))
    hf_repo: str = "hugging-science/arc-aphasia-bids"  # ARC dataset on HuggingFace Hub

    # Preprocessing targets (FROM PAPER)
    target_shape: tuple[int, int, int] = (256, 256, 256)
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Cross-validation structure (FROM PAPER)
    num_outer_folds: int = 3
    num_inner_folds: int = 3
    random_seed: int = 42

    # Acquisition types to include (FROM PAPER)
    include_space_2x: bool = True
    include_space_no_accel: bool = True
    exclude_turbo_spin_echo: bool = True

    def __post_init__(self) -> None:
        """Convert paths to Path objects."""
        self.data_root = Path(self.data_root)
        self.cache_dir = Path(self.cache_dir)


# Lesion size quintile boundaries (FROM PAPER Section 2)
# Paper uses interval notation with exclusive lower bounds: "(203, 33619]"
# means > 203 and ≤ 33,619. We use inclusive lower bounds for simplicity.
LESION_QUINTILES: dict[str, tuple[int, int]] = {
    "Q1": (203, 33_619),
    "Q2": (33_619, 67_891),
    "Q3": (67_891, 128_314),
    "Q4": (128_314, 363_885),
}

# Acquisition type identifiers
AcquisitionType = Literal["space_2x", "space_no_accel", "turbo_spin_echo"]
