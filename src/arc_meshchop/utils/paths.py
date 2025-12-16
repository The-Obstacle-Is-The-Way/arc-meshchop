from pathlib import Path


def resolve_dataset_path(data_dir: Path, path: str | None) -> Path | None:
    """Resolve a dataset path relative to data_dir if it's not absolute.

    Args:
        data_dir: The directory containing the dataset_info.json
        path: The path string from dataset_info.json (or None)

    Returns:
        Resolved absolute Path object, or None if input was None.
    """
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute():
        return p
    return (data_dir / p).resolve()
