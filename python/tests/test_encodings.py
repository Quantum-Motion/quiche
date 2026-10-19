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

from quiche.core.encodings import angles_to_integer_representation, pack_into_word

NUM_BITS = 4  # The least significant bit is 4 * pi / 2**4 = pi / 4.
LSB = np.pi / 4


class TestAnglesToIntegerRepresentation:
    """Tests for converting angles to integers."""

    @pytest.mark.parametrize(
        ("angles", "expected"),
        [
            ([0.0, LSB, 2 * LSB, 4 * LSB], [0, 1, 2, 4]),  # Exact multiples.
            ([-2 * LSB, 8 * LSB, 9 * LSB], [6, 0, 1]),  # Negative and beyond 2 * pi.
            ([0.4 * LSB, 0.6 * LSB], [0, 1]),  # Rounds to nearest.
            ([8 * LSB - 0.1 * LSB], [0]),  # Rounds up to 2**(n - 1), which wraps to 0.
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

        q = angles_to_integer_representation(angles, num_bits)

        assert q.shape == angles.shape
        assert q.min() >= 0
        assert q.max() < 2 ** (num_bits - 1)

        # Decoding gives back the angle up to half a least significant bit (mod 2 pi).
        decoded = q * 4 * np.pi / 2**num_bits
        error = np.abs(np.angle(np.exp(1j * (decoded - angles))))
        assert error.max() <= 2 * np.pi / 2**num_bits + 1e-12


class TestPackIntoWord:
    """Tests for packing integers into a single word."""

    def test_order(self):
        words = pack_into_word([[1, 2, 3], [4, 5, 6]], num_bits=4)

        assert words.shape == (2,)
        assert list(words) == [0x123, 0x456]  # The first entry is the most significant.

    def test_large_words(self):
        # FeMoco-54 sizes, where each word exceeds 64 bits.
        num_bits, num_angles = 15, 53
        rng = np.random.default_rng(1)
        q = rng.integers(0, 2 ** (num_bits - 1), size=(3, 2, num_angles))

        words = pack_into_word(q, num_bits)

        assert words.shape == (3, 2)
        mask = 2**num_bits - 1
        for idx in np.ndindex(words.shape):
            word = int(words[idx])
            assert word < 2 ** (num_bits * num_angles)
            unpacked = [
                (word >> (num_bits * (num_angles - 1 - k))) & mask
                for k in range(num_angles)
            ]
            assert unpacked == q[idx].tolist()
