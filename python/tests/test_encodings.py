# Copyright 2026 Quantum Motion Technologies Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the binary word encodings of rotation angles."""

import numpy as np
import pytest
from numpy.typing import ArrayLike

from quiche.core._encodings import angles_to_integer_representation, pack_into_word

NUM_BITS = 4
RESOLUTION = 4 * np.pi / 2**NUM_BITS  # The angle of the least significant bit.


class TestAnglesToIntegerRepresentation:
    """Tests for converting angles to integers."""

    @pytest.mark.parametrize(
        ("angles", "expected"),
        [
            ([0.0, RESOLUTION, 2 * RESOLUTION, 4 * RESOLUTION], [0, 1, 2, 4]),
            ([-2 * RESOLUTION, 8 * RESOLUTION, 9 * RESOLUTION], [6, 0, 1]),
            ([0.4 * RESOLUTION, 0.6 * RESOLUTION], [0, 1]),
            ([7.9 * RESOLUTION], [0]),
        ],
        ids=[
            "exact_multiples",
            "negative_and_beyond_2pi",
            "rounds_to_nearest",
            "rounds_up_and_wraps_to_zero",
        ],
    )
    def test_values(self, angles: list[float], expected: list[int]):
        np.testing.assert_array_equal(
            angles_to_integer_representation(angles, NUM_BITS), expected
        )

    @pytest.mark.parametrize("num_bits", [6, 12, 15])
    def test_accuracy(self, num_bits: int):
        rng = np.random.default_rng(0)
        angles = rng.uniform(-4 * np.pi, 4 * np.pi, size=(5, 4, 7))

        integers = angles_to_integer_representation(angles, num_bits)

        assert integers.shape == angles.shape
        assert integers.min() >= 0
        assert integers.max() < 2 ** (num_bits - 1)
        # Decoding gives back the angle up to half the angle of the least significant
        # bit, compared as points on the circle since rotations are periodic in 2 pi.
        decoded = integers * 4 * np.pi / 2**num_bits
        np.testing.assert_allclose(
            np.exp(1j * decoded),
            np.exp(1j * angles),
            rtol=0,
            atol=2 * np.pi / 2**num_bits,
        )

    @pytest.mark.parametrize("num_bits", [1, 0, -3])
    def test_too_few_bits(self, num_bits: int):
        with pytest.raises(ValueError, match="must be at least 2"):
            angles_to_integer_representation([0.5], num_bits)

    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_non_finite_angles(self, value: float):
        with pytest.raises(ValueError, match="must be finite"):
            angles_to_integer_representation([0.5, value], NUM_BITS)


class TestPackIntoWord:
    """Tests for packing integers into a single word."""

    def test_order(self):
        words = pack_into_word([[1, 2, 3], [4, 5, 6]], num_bits=4)

        assert words.shape == (2,)
        # The first integer takes the most significant bits.
        np.testing.assert_array_equal(words, [0x123, 0x456])

    def test_large_words(self):
        # FeMoco-54 sizes, where each word exceeds 64 bits.
        num_bits, num_angles = 15, 53
        rng = np.random.default_rng(1)
        integers = rng.integers(0, 2 ** (num_bits - 1), size=(3, 2, num_angles))

        words = pack_into_word(integers, num_bits)

        assert words.shape == (3, 2)
        mask = 2**num_bits - 1
        for idx in np.ndindex(words.shape):
            word = int(words[idx])
            unpacked = [
                (word >> (num_bits * (num_angles - 1 - k))) & mask
                for k in range(num_angles)
            ]
            np.testing.assert_array_equal(unpacked, integers[idx])

    @pytest.mark.parametrize(
        "integers",
        [[[16]], [[-1]], np.array([[16]]), np.array([[-1]])],
        ids=["list_too_large", "list_negative", "array_too_large", "array_negative"],
    )
    def test_entries_must_fit(self, integers: ArrayLike):
        with pytest.raises(ValueError, match="to fit in the word"):
            pack_into_word(integers, num_bits=4)
