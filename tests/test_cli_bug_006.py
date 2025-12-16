import os
from pathlib import Path

from arc_meshchop.utils.paths import resolve_dataset_path


def test_resolve_dataset_paths_works_with_cwd_change(tmp_path: Path) -> None:
    """Test that resolving paths works even when CWD is different (BUG-006 fix)."""
    # Setup:
    # /tmp/project/
    #   data/
    #     dataset_info.json
    #     image.nii

    project_dir = tmp_path / "project"
    data_dir = project_dir / "data"
    data_dir.mkdir(parents=True)

    image_path = data_dir / "image.nii"
    image_path.touch()

    # Change CWD to project_dir
    cwd = Path.cwd()
    os.chdir(project_dir)

    try:
        # Simulate loading from dataset_info.json where path is "image.nii" (relative)
        # We pass "data" as data_dir (relative to CWD)
        passed_data_dir = Path("data")
        relative_path_str = "image.nii"

        # OLD BEHAVIOR (would fail): Path(relative_path_str).exists() -> False

        # NEW BEHAVIOR:
        # resolve_dataset_path should combine data_dir (resolved) and relative_path
        resolved_path = resolve_dataset_path(passed_data_dir.resolve(), relative_path_str)

        assert resolved_path is not None
        assert resolved_path.exists(), f"Resolved path {resolved_path} should exist"
        assert resolved_path.resolve() == image_path.resolve()

    finally:
        os.chdir(cwd)
