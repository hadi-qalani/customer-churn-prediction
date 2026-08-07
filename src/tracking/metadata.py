import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def save_metadata(metadata: dict[str, Any], file_path: str | Path) -> None:
    """Save artifact metadata as a JSON file.

    Args:
        metadata (dict[str, Any]): Metadata information.
        file_path (str | Path): Path where metadata JSON will be saved.
    """

    file_path = Path(file_path)

    metadata["created_at"] = datetime.now(timezone.utc).isoformat()

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
