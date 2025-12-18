from pathlib import Path

from arc_meshchop.utils.paths import resolve_dataset_path


def test_resolve_dataset_path_handles_repo_root_relative_paths(tmp_path: Path) -> None:
    """Test that repo-root relative paths resolve correctly (BUG-008)."""
    # Setup:
    # /tmp/project/
    #   data/arc/cache/nifti_cache/image.nii
    project_dir = tmp_path / "project"
    data_dir = project_dir / "data" / "arc"

    image_path = data_dir / "cache" / "nifti_cache" / "image.nii"
    image_path.parent.mkdir(parents=True)
    image_path.touch()

    # Some older dataset_info.json files stored paths like this (repo-root relative):
    path_str = "data/arc/cache/nifti_cache/image.nii"

    resolved = resolve_dataset_path(data_dir.resolve(), path_str)

    assert resolved is not None
    assert resolved.exists()
    assert resolved == image_path.resolve()
