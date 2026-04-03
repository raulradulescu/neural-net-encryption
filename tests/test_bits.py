from __future__ import annotations

import numpy as np

from src.data.bits import bits01_to_pm1, flip_single_bit, pm1_to_bits01, random_bits


def test_bits_round_trip_between_canonical_and_pm1() -> None:
    bits = np.array([[0, 1, 1], [1, 0, 0]], dtype=np.int8)

    converted = bits01_to_pm1(bits)
    restored = pm1_to_bits01(converted)

    assert np.array_equal(restored, bits)
    assert set(np.unique(converted).tolist()) <= {-1, 1}


def test_random_bits_shape_and_values() -> None:
    bits = random_bits((4, 5), seed=123)

    assert bits.shape == (4, 5)
    assert bits.dtype == np.int8
    assert set(np.unique(bits).tolist()) <= {0, 1}


def test_flip_single_bit_flips_exactly_one_position() -> None:
    bits = np.array([[0, 1], [1, 0]], dtype=np.int8)

    flipped = flip_single_bit(bits, (0, 1))

    assert np.array_equal(bits, np.array([[0, 1], [1, 0]], dtype=np.int8))
    assert np.array_equal(flipped, np.array([[0, 0], [1, 0]], dtype=np.int8))
    assert int(np.sum(bits != flipped)) == 1
