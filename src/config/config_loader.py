from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


CONFIG_FILES = {
    "preprocessing": "preprocessing.yaml",
    "model": "model.yaml",
    "schema": "schema.yaml",
    "dataset": "dataset.yaml",
    "evaluation": "evaluation.yaml",
}


def load_configs():
    configs = {}

    for key, filename in CONFIG_FILES.items():
        with open(CONFIG_DIR / filename, encoding="utf-8") as f:
            configs[key] = yaml.safe_load(f)

    return configs
