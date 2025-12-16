from pathlib import Path

from arc_meshchop.data import ARCDataset


def test_cache_key_depends_on_mask_path(tmp_path: Path) -> None:
    """Test that changing the mask path changes the cache key (BUG-005)."""
    image_path = tmp_path / "image.nii.gz"
    mask_v1 = tmp_path / "mask_v1.nii.gz"
    mask_v2 = tmp_path / "mask_v2.nii.gz"

    # Create dummy files
    image_path.touch()
    mask_v1.touch()
    mask_v2.touch()

    # Create dataset with mask v1
    ds1 = ARCDataset(
        image_paths=[image_path],
        mask_paths=[mask_v1],
        preprocess=False,
    )
    key1 = ds1._cache_key(0)

    # Create dataset with mask v2
    ds2 = ARCDataset(
        image_paths=[image_path],
        mask_paths=[mask_v2],
        preprocess=False,
    )
    key2 = ds2._cache_key(0)

    # Assert keys are different
    assert key1 != key2, "Cache key should change when mask path changes"
