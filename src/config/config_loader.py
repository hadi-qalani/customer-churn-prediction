from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def load_schema() -> dict:
    with open(CONFIG_DIR / "schema.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)
