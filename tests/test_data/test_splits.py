"""Tests for cross-validation split generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from arc_meshchop.data.splits import (
    CVSplit,
    InnerFold,
    NestedCVSplits,
    OuterFold,
    create_stratification_labels,
    generate_nested_cv_splits,
)


class TestStratificationLabels:
    """Tests for stratification label creation."""

    def test_creates_combined_labels(self) -> None:
        """Test combined label format."""
        quintiles = ["Q1", "Q2", "Q3"]
        acq_types = ["space_2x", "space_no_accel", "space_2x"]

        labels = create_stratification_labels(quintiles, acq_types)

        assert labels == ["Q1_space_2x", "Q2_space_no_accel", "Q3_space_2x"]

    def test_handles_empty_lists(self) -> None:
        """Test with empty input lists."""
        labels = create_stratification_labels([], [])
        assert labels == []

    def test_mismatched_lengths_raises(self) -> None:
        """Test that mismatched lengths raise error."""
        with pytest.raises(ValueError):
            create_stratification_labels(["Q1", "Q2"], ["space_2x"])

    def test_preserves_order(self) -> None:
        """Test that order is preserved."""
        quintiles = ["Q4", "Q1", "Q2", "Q3"]
        acq_types = ["a", "b", "c", "d"]

        labels = create_stratification_labels(quintiles, acq_types)

        assert labels[0] == "Q4_a"
        assert labels[1] == "Q1_b"
        assert labels[2] == "Q2_c"
        assert labels[3] == "Q3_d"


class TestNestedCVSplits:
    """Tests for nested cross-validation split generation."""

    def test_outer_fold_count(self) -> None:
        """Verify correct number of outer folds."""
        labels = ["Q1_a", "Q2_a", "Q1_b", "Q2_b"] * 10  # 40 samples

        splits = generate_nested_cv_splits(
            n_samples=len(labels),
            stratification_labels=labels,
            num_outer_folds=3,
            num_inner_folds=3,
        )

        assert len(splits.outer_folds) == 3

    def test_inner_fold_count(self) -> None:
        """Verify correct number of inner folds per outer."""
        labels = ["Q1_a", "Q2_a", "Q1_b", "Q2_b"] * 10

        splits = generate_nested_cv_splits(
            n_samples=len(labels),
            stratification_labels=labels,
            num_outer_folds=3,
            num_inner_folds=3,
        )

        for outer in splits.outer_folds:
            assert isinstance(outer, OuterFold)
            assert len(outer.inner_folds) == 3

    def test_no_data_leakage(self) -> None:
        """Verify test indices don't appear in train/val."""
        labels = ["Q1_a", "Q2_a", "Q1_b", "Q2_b"] * 10

        splits = generate_nested_cv_splits(
            n_samples=len(labels),
            stratification_labels=labels,
            num_outer_folds=3,
            num_inner_folds=3,
        )

        for outer in splits.outer_folds:
            test_set = set(outer.test_indices)

            for inner in outer.inner_folds:
                assert isinstance(inner, InnerFold)
                train_set = set(inner.train_indices)
                val_set = set(inner.val_indices)

                # No overlap between test and train/val
                assert len(test_set & train_set) == 0, "Test-train overlap detected"
                assert len(test_set & val_set) == 0, "Test-val overlap detected"

                # No overlap between train and val
                assert len(train_set & val_set) == 0, "Train-val overlap detected"

    def test_all_indices_covered(self) -> None:
        """Verify all indices appear exactly once in outer test folds."""
        n_samples = 40
        labels = ["Q1_a", "Q2_a", "Q1_b", "Q2_b"] * 10

        splits = generate_nested_cv_splits(
            n_samples=n_samples,
            stratification_labels=labels,
            num_outer_folds=3,
            num_inner_folds=3,
        )

        all_test_indices: list[int] = []
        for outer in splits.outer_folds:
            all_test_indices.extend(outer.test_indices)

        # Each index appears exactly once across test folds
        assert sorted(all_test_indices) == list(range(n_samples))

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Test saving and loading splits."""
        labels = ["Q1_a", "Q2_a", "Q1_b", "Q2_b"] * 10

        splits = generate_nested_cv_splits(
            n_samples=len(labels),
            stratification_labels=labels,
            num_outer_folds=3,
            num_inner_folds=3,
            random_seed=42,
        )

        # Save
        save_path = tmp_path / "splits.json"
        splits.save(save_path)

        # Load
        loaded = NestedCVSplits.load(save_path)

        assert loaded.random_seed == splits.random_seed
        assert loaded.num_outer_folds == splits.num_outer_folds
        assert loaded.num_inner_folds == splits.num_inner_folds
        assert loaded.outer_folds == splits.outer_folds
        assert loaded.stratification_labels == splits.stratification_labels

    def test_reproducibility(self) -> None:
        """Test that same seed produces same splits."""
        labels = ["Q1_a", "Q2_a", "Q1_b", "Q2_b"] * 10

        splits1 = generate_nested_cv_splits(
            n_samples=len(labels),
            stratification_labels=labels,
            random_seed=42,
        )

        splits2 = generate_nested_cv_splits(
            n_samples=len(labels),
            stratification_labels=labels,
            random_seed=42,
        )

        assert splits1.outer_folds == splits2.outer_folds

    def test_different_seeds_produce_different_splits(self) -> None:
        """Test that different seeds produce different splits."""
        labels = ["Q1_a", "Q2_a", "Q1_b", "Q2_b"] * 10

        splits1 = generate_nested_cv_splits(
            n_samples=len(labels),
            stratification_labels=labels,
            random_seed=42,
        )

        splits2 = generate_nested_cv_splits(
            n_samples=len(labels),
            stratification_labels=labels,
            random_seed=123,
        )

        assert splits1.outer_folds != splits2.outer_folds

    def test_get_split_method(self) -> None:
        """Test get_split method returns correct indices."""
        labels = ["Q1_a", "Q2_a", "Q1_b", "Q2_b"] * 10

        splits = generate_nested_cv_splits(
            n_samples=len(labels),
            stratification_labels=labels,
            num_outer_folds=3,
            num_inner_folds=3,
        )

        # Get specific split
        split = splits.get_split(outer_fold=0, inner_fold=0)

        assert isinstance(split, CVSplit)
        assert len(split.train_indices) > 0
        assert len(split.val_indices) > 0
        assert len(split.test_indices) > 0

    def test_get_split_outer_only(self) -> None:
        """Test get_split with only outer fold (no inner)."""
        labels = ["Q1_a", "Q2_a", "Q1_b", "Q2_b"] * 10

        splits = generate_nested_cv_splits(
            n_samples=len(labels),
            stratification_labels=labels,
        )

        # Get outer fold only
        split = splits.get_split(outer_fold=0, inner_fold=None)

        assert isinstance(split, CVSplit)
        assert len(split.train_indices) == 0
        assert len(split.val_indices) == 0
        assert len(split.test_indices) > 0


class TestCVSplit:
    """Tests for CVSplit dataclass."""

    def test_default_test_indices(self) -> None:
        """Test that test_indices defaults to empty list."""
        split = CVSplit(
            train_indices=[0, 1, 2],
            val_indices=[3, 4],
        )

        assert split.test_indices == []

    def test_all_indices_accessible(self) -> None:
        """Test that all indices are accessible."""
        split = CVSplit(
            train_indices=[0, 1, 2],
            val_indices=[3, 4],
            test_indices=[5, 6, 7],
        )

        assert split.train_indices == [0, 1, 2]
        assert split.val_indices == [3, 4]
        assert split.test_indices == [5, 6, 7]


class TestPaperConfiguration:
    """Tests verifying paper's CV configuration (3x3 nested CV)."""

    def test_paper_configuration_224_samples(self) -> None:
        """Test with paper's dataset size (224 samples)."""
        # Create labels simulating paper's stratification
        # 4 quintiles x 2 acquisition types = 8 strata
        labels = []
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            for acq in ["space_2x", "space_no_accel"]:
                # ~28 samples per stratum (224 / 8)
                labels.extend([f"{q}_{acq}"] * 28)

        assert len(labels) == 224

        splits = generate_nested_cv_splits(
            n_samples=224,
            stratification_labels=labels,
            num_outer_folds=3,
            num_inner_folds=3,
            random_seed=42,
        )

        # Paper configuration: 3 outer x 3 inner = 9 training configs
        assert splits.num_outer_folds == 3
        assert splits.num_inner_folds == 3

        # Total training configurations
        total_configs = 0
        for outer in splits.outer_folds:
            total_configs += len(outer.inner_folds)

        assert total_configs == 9

    def test_approximate_split_sizes(self) -> None:
        """Test that split sizes are approximately correct."""
        labels = ["Q1_a", "Q2_a", "Q1_b", "Q2_b"] * 56  # 224 samples

        splits = generate_nested_cv_splits(
            n_samples=224,
            stratification_labels=labels,
            num_outer_folds=3,
            num_inner_folds=3,
        )

        for outer in splits.outer_folds:
            # Test set should be ~1/3 of data (~74-75 samples)
            assert 70 <= len(outer.test_indices) <= 80

            for inner in outer.inner_folds:
                # Train should be ~2/3 of outer train (~100 samples)
                assert 90 <= len(inner.train_indices) <= 110

                # Val should be ~1/3 of outer train (~50 samples)
                assert 40 <= len(inner.val_indices) <= 60
