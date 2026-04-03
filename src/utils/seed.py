"""Seed helpers for reproducible runs."""

from __future__ import annotations

import os
import random
from secrets import randbits
from typing import Final

import numpy as np

DEFAULT_SEED_BITS: Final[int] = 32


def set_seed(seed: int | None = None, *, deterministic: bool = True) -> int:
    """Seed Python and NumPy RNGs; seed optional torch RNGs if present."""

    resolved = randbits(DEFAULT_SEED_BITS) if seed is None else int(seed)
    os.environ["PYTHONHASHSEED"] = str(resolved)
    random.seed(resolved)
    np.random.seed(resolved)

    try:
        import torch
    except ModuleNotFoundError:
        torch = None

    if torch is not None:
        torch.manual_seed(resolved)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(resolved)
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass

    return resolved
