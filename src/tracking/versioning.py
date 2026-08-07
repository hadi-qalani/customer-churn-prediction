from pathlib import Path


def create_version_dir(base_dir: str) -> Path:
    """Create and return a new versioned directory.

    Args:
        base_dir (str): Base directory for storing artifacts.

    Returns:
        Path: Newly created version directory.
    """

    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    versions = []

    for item in base_path.iterdir():
        if item.is_dir() and item.name.startswith("v"):
            try:
                versions.append(int(item.name[1:]))
            except ValueError:
                continue

    next_version = max(versions, default=0) + 1

    version_dir = base_path / f"v{next_version:03d}"
    version_dir.mkdir()

    return version_dir