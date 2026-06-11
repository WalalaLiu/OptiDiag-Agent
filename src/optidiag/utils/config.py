"""Small configuration loader with a YAML fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def load_config(path: str = "configs/default.yaml") -> Dict[str, Any]:
    """Load a YAML config file if PyYAML is installed."""
    config_path = Path(path)
    if not config_path.exists():
        return {}

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install pyyaml to load YAML configuration files.") from exc

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}

