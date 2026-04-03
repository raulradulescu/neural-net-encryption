from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .simulator import run_tpm_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the TPM synchronization simulator")
    parser.add_argument("--config", required=True, help="Path to a YAML TPM config")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_tpm_from_config(Path(args.config))
    print(result["run_dir"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

