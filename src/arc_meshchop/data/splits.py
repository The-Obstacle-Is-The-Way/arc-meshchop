"""Cross-validation split generation with stratification.

Implements nested 3-fold cross-validation from the paper:
- 3 outer folds (train/test)
- 3 inner folds per outer (train/val)
- Stratified by lesion size quintile and acquisition type

FROM PAPER (Section 2):
"We employed nested cross-validation for model selection and evaluation, with
3 outer folds and 3 inner folds, stratified by lesion size quintiles and
acquisition type."
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold


@dataclass
class CVSplit:
    """Single cross-validation split.

    Attributes:
        train_indices: Indices for training set.
        val_indices: Indices for validation set.
        test_indices: Indices for test set (outer fold only).
    """

    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int] = field(default_factory=list)


@dataclass
class InnerFold:
    """Inner fold data structure.

    Attributes:
        train_indices: Indices for training within this inner fold.
        val_indices: Indices for validation within this inner fold.
    """

    train_indices: list[int]
    val_indices: list[int]


@dataclass
class OuterFold:
    """Outer fold data structure.

    Attributes:
        test_indices: Indices for test set (held out for outer fold).
        inner_folds: List of inner folds for train/val splits.
    """

    test_indices: list[int]
    inner_folds: list[InnerFold]


@dataclass
class NestedCVSplits:
    """Complete nested cross-validation structure.

    Attributes:
        outer_folds: List of outer fold structures.
        random_seed: Random seed used for reproducibility.
        num_outer_folds: Number of outer folds.
        num_inner_folds: Number of inner folds per outer.
        stratification_labels: Combined stratification labels.
    """

    outer_folds: list[OuterFold]
    random_seed: int
    num_outer_folds: int
    num_inner_folds: int
    stratification_labels: list[str]

    def get_split(
        self,
        outer_fold: int,
        inner_fold: int | None = None,
    ) -> CVSplit:
        """Get a specific split configuration.

        Args:
            outer_fold: Outer fold index (0-2).
            inner_fold: Inner fold index (0-2) or None for outer test only.

        Returns:
            CVSplit with train/val/test indices.
        """
        outer = self.outer_folds[outer_fold]
        test_indices = outer.test_indices

        if inner_fold is None:
            # Return outer fold test set only
            return CVSplit(
                train_indices=[],
                val_indices=[],
                test_indices=test_indices.copy(),
            )

        inner = outer.inner_folds[inner_fold]
        return CVSplit(
            train_indices=inner.train_indices.copy(),
            val_indices=inner.val_indices.copy(),
            test_indices=test_indices.copy(),
        )

    def save(self, path: Path | str) -> None:
        """Save splits to JSON file.

        Args:
            path: Output file path.
        """
        # Convert dataclasses to dicts for JSON serialization
        outer_folds_data = [
            {
                "test_indices": outer.test_indices,
                "inner_folds": [asdict(inner) for inner in outer.inner_folds],
            }
            for outer in self.outer_folds
        ]

        data = {
            "random_seed": self.random_seed,
            "num_outer_folds": self.num_outer_folds,
            "num_inner_folds": self.num_inner_folds,
            "stratification_labels": self.stratification_labels,
            "outer_folds": outer_folds_data,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path | str) -> NestedCVSplits:
        """Load splits from JSON file.

        Args:
            path: Input file path.

        Returns:
            NestedCVSplits object.
        """
        data = json.loads(Path(path).read_text())

        # Reconstruct dataclasses from dicts
        outer_folds = [
            OuterFold(
                test_indices=outer["test_indices"],
                inner_folds=[
                    InnerFold(
                        train_indices=inner["train_indices"],
                        val_indices=inner["val_indices"],
                    )
                    for inner in outer["inner_folds"]
                ],
            )
            for outer in data["outer_folds"]
        ]

        return cls(
            outer_folds=outer_folds,
            random_seed=data["random_seed"],
            num_outer_folds=data["num_outer_folds"],
            num_inner_folds=data["num_inner_folds"],
            stratification_labels=data["stratification_labels"],
        )


def create_stratification_labels(
    lesion_quintiles: list[str],
    acquisition_types: list[str],
) -> list[str]:
    """Create combined stratification labels.

    Combines lesion size quintile and acquisition type for stratification.

    Args:
        lesion_quintiles: List of quintile labels (Q1-Q4) per sample.
        acquisition_types: List of acquisition types per sample.

    Returns:
        List of combined labels for stratification.
    """
    return [f"{q}_{a}" for q, a in zip(lesion_quintiles, acquisition_types, strict=True)]


def generate_nested_cv_splits(
    n_samples: int,
    stratification_labels: list[str],
    num_outer_folds: int = 3,
    num_inner_folds: int = 3,
    random_seed: int = 42,
) -> NestedCVSplits:
    """Generate nested cross-validation splits with stratification.

    Implements the nested CV structure from the paper:
    - 3 outer folds for train/test
    - 3 inner folds per outer for train/val
    - Stratified by combined lesion size + acquisition type

    Args:
        n_samples: Total number of samples.
        stratification_labels: Combined stratification labels (must match n_samples).
        num_outer_folds: Number of outer folds.
        num_inner_folds: Number of inner folds per outer.
        random_seed: Random seed for reproducibility.

    Returns:
        NestedCVSplits object with complete split structure.

    Raises:
        ValueError: If stratification_labels length doesn't match n_samples.
    """
    if len(stratification_labels) != n_samples:
        raise ValueError(
            f"stratification_labels length ({len(stratification_labels)}) "
            f"must match n_samples ({n_samples})"
        )

    indices = np.arange(n_samples)
    labels = np.array(stratification_labels)

    outer_splitter = StratifiedKFold(
        n_splits=num_outer_folds,
        shuffle=True,
        random_state=random_seed,
    )

    outer_folds: list[OuterFold] = []

    for outer_train_idx, test_idx in outer_splitter.split(indices, labels):
        # Inner splits on outer training data
        inner_indices = indices[outer_train_idx]
        inner_labels = labels[outer_train_idx]

        inner_splitter = StratifiedKFold(
            n_splits=num_inner_folds,
            shuffle=True,
            random_state=random_seed,
        )

        inner_folds: list[InnerFold] = []
        for train_idx, val_idx in inner_splitter.split(inner_indices, inner_labels):
            inner_folds.append(
                InnerFold(
                    train_indices=inner_indices[train_idx].tolist(),
                    val_indices=inner_indices[val_idx].tolist(),
                )
            )

        outer_folds.append(
            OuterFold(
                test_indices=indices[test_idx].tolist(),
                inner_folds=inner_folds,
            )
        )

    return NestedCVSplits(
        outer_folds=outer_folds,
        random_seed=random_seed,
        num_outer_folds=num_outer_folds,
        num_inner_folds=num_inner_folds,
        stratification_labels=stratification_labels,
    )
