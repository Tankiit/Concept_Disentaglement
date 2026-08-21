"""Configuration loading for reproducible experiments."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import json


def _deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_mapping(path: str | Path) -> dict[str, Any]:
    """Load a JSON or YAML mapping."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            value = json.load(handle)
        elif path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError("YAML configs require `pip install pyyaml`.") from exc
            value = yaml.safe_load(handle)
        else:
            raise ValueError(f"unsupported config format: {path.suffix}")
    if not isinstance(value, dict):
        raise ValueError(f"config must contain a mapping: {path}")
    return value


def load_experiment_config(path: str | Path, _seen=None) -> dict[str, Any]:
    """Load a config and recursively merge its ``extends`` parent."""
    path = Path(path).resolve()
    seen = set() if _seen is None else _seen
    if path in seen:
        raise ValueError(f"cyclic experiment config inheritance at {path}")
    seen.add(path)
    child = load_mapping(path)
    parent_name = child.pop("extends", None)
    if parent_name is None:
        return child
    parent = load_experiment_config(path.parent / parent_name, seen)
    return _deep_merge(parent, child)


def packaged_config_dir() -> Path:
    return Path(__file__).resolve().parent / "configs"


def load_dataset_config(name: str, config_dir: str | Path | None = None) -> dict[str, Any]:
    """Load dataset metadata by name, accepting either JSON or YAML."""
    root = Path(config_dir) if config_dir else packaged_config_dir() / "datasets"
    for suffix in (".json", ".yaml", ".yml"):
        candidate = root / f"{name}{suffix}"
        if candidate.is_file():
            return load_mapping(candidate)
    raise FileNotFoundError(f"no dataset config for {name!r} under {root}")
