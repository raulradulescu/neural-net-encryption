"""Canonical bit utilities.

The canonical in-memory representation is ``{0, 1}`` with dtype ``int8`` when a
NumPy array is used. Conversion helpers expose the ``{-1, 1}`` view used by many
neural cryptography experiments.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from src.utils.errors import ValidationError

BitArray = np.ndarray

_CANONICAL_BITS = np.array([0, 1], dtype=np.int8)
_CANONICAL_PM1 = np.array([-1, 1], dtype=np.int8)


def _as_int8_array(values: Any, *, copy: bool = False) -> BitArray:
    array = np.asarray(values, dtype=np.int8)
    if copy:
        array = array.copy()
    return array


def _validate_membership(array: BitArray, allowed: Iterable[int], *, name: str) -> None:
    allowed_array = np.fromiter((int(item) for item in allowed), dtype=np.int8)
    if array.size == 0:
        return
    if not np.isin(array, allowed_array).all():
        raise ValidationError(f"{name} must contain only {tuple(int(x) for x in allowed_array.tolist())}")


def ensure_bits01(values: Any, *, copy: bool = False) -> BitArray:
    """Return a NumPy array containing only canonical ``{0, 1}`` values."""

    array = _as_int8_array(values, copy=copy)
    _validate_membership(array, _CANONICAL_BITS, name="bits")
    return array


def bits01_to_pm1(values: Any, *, copy: bool = False) -> BitArray:
    """Convert canonical bits from ``{0, 1}`` to ``{-1, 1}``."""

    bits = ensure_bits01(values, copy=copy)
    return bits * 2 - 1


def pm1_to_bits01(values: Any, *, copy: bool = False) -> BitArray:
    """Convert values from ``{-1, 1}`` back to canonical bits."""

    array = _as_int8_array(values, copy=copy)
    _validate_membership(array, _CANONICAL_PM1, name="pm1 values")
    return ((array + 1) // 2).astype(np.int8, copy=False)


def random_bits(shape: int | tuple[int, ...], *, seed: int | None = None, rng: np.random.Generator | None = None) -> BitArray:
    """Generate random canonical bits with the requested shape."""

    if rng is None:
        rng = np.random.default_rng(seed)
    actual_shape = (shape,) if isinstance(shape, int) else shape
    return rng.integers(0, 2, size=actual_shape, dtype=np.int8)


def flip_single_bit(values: Any, index: int | tuple[int, ...]) -> BitArray:
    """Flip exactly one canonical bit and return a copy."""

    array = ensure_bits01(values, copy=True)
    if isinstance(index, tuple):
        array[index] = np.int8(1 - int(array[index]))
    else:
        flat = array.reshape(-1)
        flat[index] = np.int8(1 - int(flat[index]))
    return array
