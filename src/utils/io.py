"""Run directory creation and artifact writers."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

import yaml

from .errors import ArtifactError


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def create_run_dir(base_dir: str | Path, *, prefix: str = "run", run_name: str | None = None) -> Path:
    """Create a unique run directory under ``base_dir``."""

    root = Path(base_dir)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid4().hex[:8]
    pieces = [prefix, timestamp]
    if run_name:
        safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in run_name).strip("-")
        if safe_name:
            pieces.append(safe_name)
    pieces.append(suffix)
    run_dir = root / "_".join(pieces)
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


def write_json(path: str | Path, data: Any, *, indent: int = 2, sort_keys: bool = True) -> Path:
    """Write JSON data to disk."""

    output = Path(path)
    _ensure_parent(output)
    try:
        output.write_text(json.dumps(data, indent=indent, sort_keys=sort_keys, ensure_ascii=False) + "\n", encoding="utf-8")
    except OSError as exc:
        raise ArtifactError(f"Failed to write JSON artifact to {output}: {exc}") from exc
    return output


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> Path:
    """Write newline-delimited JSON records."""

    output = Path(path)
    _ensure_parent(output)
    try:
        with output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")
    except OSError as exc:
        raise ArtifactError(f"Failed to write JSONL artifact to {output}: {exc}") from exc
    return output


def write_yaml(path: str | Path, data: Any, *, sort_keys: bool = False) -> Path:
    """Write YAML data to disk."""

    output = Path(path)
    _ensure_parent(output)
    try:
        with output.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=sort_keys)
    except (OSError, yaml.YAMLError) as exc:
        raise ArtifactError(f"Failed to write YAML artifact to {output}: {exc}") from exc
    return output


def write_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> Path:
    """Write a CSV file from an iterable of mapping rows."""

    output = Path(path)
    _ensure_parent(output)
    rows_list = [dict(row) for row in rows]
    if fieldnames is None:
        fieldnames = list(rows_list[0].keys()) if rows_list else []
    try:
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
            writer.writeheader()
            writer.writerows(rows_list)
    except OSError as exc:
        raise ArtifactError(f"Failed to write CSV artifact to {output}: {exc}") from exc
    return output
