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
    data_dir = data_dir.resolve()
    p = Path(path)

    if p.is_absolute():
        return p

    # Preferred: paths in dataset_info.json are relative to `data_dir`.
    candidate = (data_dir / p).resolve()
    if candidate.exists():
        return candidate

    # Backwards compatibility: some dataset_info.json files store repo-root relative
    # paths like "data/arc/cache/...". In that case, the correct base is a parent
    # of `data_dir` (e.g., the project root).
    for parent in data_dir.parents:
        candidate = (parent / p).resolve()
        if candidate.exists():
            return candidate

    # Last resort: interpret as relative to current working directory.
    candidate = p.resolve()
    if candidate.exists():
        return candidate

    # Nothing matched; return the intended (data_dir-relative) path for clearer errors.
    return (data_dir / p).resolve()
