"""Configuration loading utilities.

All paths, hyperparameters and thresholds are stored in ``config/config.yaml``
(PRD Section 13: no hard-coded configuration values in scripts).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


class Config:
    """Thin wrapper around the YAML configuration tree.

    Attributes are accessed with dotted paths via :meth:`get`, e.g.
    ``cfg.get("models.xgboost.max_depth")``.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def raw(self) -> dict[str, Any]:
        return self._data


def load_config(path: str | Path | None = None) -> Config:
    """Load the YAML configuration file.

    Args:
        path: Optional explicit path to a config file. Defaults to
            ``config/config.yaml`` relative to the repository root.

    Returns:
        A :class:`Config` instance wrapping the parsed YAML tree.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Config(data)


def project_root() -> Path:
    """Return the absolute repository root directory."""
    return Path(__file__).resolve().parents[2]


def resolve_path(relative_path: str | Path) -> Path:
    """Resolve a config-relative path against the repository root."""
    return project_root() / relative_path
