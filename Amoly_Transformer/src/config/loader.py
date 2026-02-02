"""Load and merge YAML config files into a single nested dict."""

import yaml
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(__file__).parent

_DEFAULT_CONFIGS = ["model.yaml", "training.yaml", "data.yaml", "inference.yaml"]


def load_config(config_names: list[str] | None = None) -> dict[str, Any]:
    """
    Load YAML config files from src/config/ and merge them.

    Args:
        config_names: List of filenames. If None, loads all default configs.

    Returns:
        Merged dict with top-level keys: model, training, data, inference.
    """
    if config_names is None:
        config_names = _DEFAULT_CONFIGS

    merged: dict[str, Any] = {}
    for name in config_names:
        path = CONFIG_DIR / name
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if data:
            merged.update(data)
    return merged
