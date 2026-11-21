"""YAML configuration with layered overrides.

Order of precedence (last wins): configs/default.yaml -> --config file -> key=value overrides.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _parse_value(raw: str) -> Any:
    return yaml.safe_load(raw)


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply dotted overrides such as ``retrieval.mode=dense``."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        node = cfg
        parts = dotted.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_value(raw)
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    if path:
        with open(path) as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    cfg = apply_overrides(cfg, overrides)
    index_dir = Path(cfg["index_dir"])
    if not index_dir.is_absolute():
        cfg["index_dir"] = str(ROOT / index_dir)
    return cfg
