from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a recursive merge without mutating either input."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def load_config(*paths: str | Path) -> dict[str, Any]:
    if not paths:
        raise ValueError("At least one config path is required.")
    cfg: dict[str, Any] = {}
    for path in paths:
        cfg = deep_merge(cfg, load_yaml(path))
    return cfg


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def set_by_dotted_key(cfg: dict[str, Any], dotted_key: str, value: Any) -> dict[str, Any]:
    updated = copy.deepcopy(cfg)
    node = updated
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = node.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"Cannot set {dotted_key}: {part} is not a mapping.")
        node = child
    node[parts[-1]] = value
    return updated


def parse_scalar(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"none", "null"}:
        return None
    try:
        if "." not in value:
            return int(value)
        return float(value)
    except ValueError:
        return value


def parse_float_or_fraction(value: str | float | int) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = value.strip()
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return float(numerator) / float(denominator)
    return float(text)
