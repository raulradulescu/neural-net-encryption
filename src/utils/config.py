"""Config loading and CLI override helpers."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import yaml

from .errors import ConfigError, require


ConfigDict = dict[str, Any]


def load_config(path: str | Path) -> ConfigDict:
    """Load a YAML or JSON config file into a dictionary."""

    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()
    try:
        if suffix in {".yaml", ".yml"}:
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        elif suffix == ".json":
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
        else:
            raise ConfigError(f"Unsupported config format: {config_path.suffix}")
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"Failed to load config {config_path}: {exc}") from exc

    if loaded is None:
        return {}
    require(isinstance(loaded, dict), f"Config root must be a mapping: {config_path}", ConfigError)
    return loaded


def deep_merge(base: ConfigDict, updates: ConfigDict) -> ConfigDict:
    """Return a deep merged copy of ``base`` updated with ``updates``."""

    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def assign_path(mapping: ConfigDict, dotted_path: str, value: Any) -> None:
    """Assign ``value`` into ``mapping`` using dot notation."""

    require(dotted_path.strip() != "", "Override path cannot be empty", ConfigError)
    current = mapping
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        require(isinstance(current[part], dict), f"Cannot override through non-mapping key: {part}", ConfigError)
        current = current[part]
    current[parts[-1]] = value


def parse_overrides(overrides: Iterable[str]) -> ConfigDict:
    """Parse ``key=value`` overrides with dot-notation keys."""

    parsed: ConfigDict = {}
    for override in overrides:
        key, separator, raw_value = override.partition("=")
        if not separator:
            raise ConfigError(f"Override must use key=value syntax: {override}")
        value = yaml.safe_load(raw_value)
        assign_path(parsed, key.strip(), value)
    return parsed


def apply_overrides(config: ConfigDict, overrides: Iterable[str] | ConfigDict | None = None) -> ConfigDict:
    """Apply CLI-style overrides to a config without mutating the input."""

    if overrides is None:
        return deepcopy(config)
    if isinstance(overrides, dict):
        updates = overrides
    else:
        updates = parse_overrides(overrides)
    return deep_merge(config, updates)


def load_config_with_overrides(path: str | Path, overrides: Iterable[str] | ConfigDict | None = None) -> ConfigDict:
    """Load a config file and optionally apply overrides."""

    return apply_overrides(load_config(path), overrides)


def build_config_parser(description: str | None = None) -> argparse.ArgumentParser:
    """Build a small CLI parser for config-based entrypoints."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to a YAML or JSON config file")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values using dot notation",
    )
    return parser
