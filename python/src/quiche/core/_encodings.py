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

"""Binary word encodings of rotation angles for QROM data loading."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def angles_to_integer_representation(
    angles: ArrayLike, num_bits: int
) -> NDArray[np.int64]:
    """
    Angles (radians) -> integers q in [0, 2**(num_bits - 1)).

    x = (theta mod 2*pi) / (4*pi)    in [0, 1/2)
    q = x * 2**num_bits              (rounded to an integer)
    q = q mod 2**(num_bits - 1)      (rounding can land exactly on 2**(num_bits - 1))

    Parameters
    ----------
    angles : ArrayLike
        An array-like object containing radian angles, as floats.
    num_bits : int
        Number of bits used to encode each angle.

    Returns
    -------
    An array of N-1 angles, encoded as integers.

    Raises
    ------
    ValueError
        If `num_bits` < 2.
    ValueError
        If angles are NaN or infinite.

    """
    if num_bits < 2:
        error_msg = f"`num_bits` must be at least 2, got {num_bits}."
        raise ValueError(error_msg)
    angles = np.asarray(angles, dtype=float)
    if not np.all(np.isfinite(angles)):
        error_msg = "`angles` must be finite, got NaN or infinite values."
        raise ValueError(error_msg)
    radian_angle_values = np.mod(np.asarray(angles, dtype=float), 2 * np.pi) / (
        4 * np.pi
    )
    integer_angles = radian_angle_values * 2**num_bits
    integer_angles = np.rint(integer_angles)
    return np.mod(integer_angles.astype(np.int64), 2 ** (num_bits - 1))


def pack_into_word(integer_angles: ArrayLike, num_bits: int) -> NDArray[np.object_]:
    """
    Pack integers along the last axis.

    W = sum_k q_k * 2**(num_bits * (K - 1 - k))

    Parameters
    ----------
    integer_angles : ArrayLike
        An array-like object containing angles, encoded as integers.
    num_bits : int
        Number of bits used to encode each angle.

    Returns
    -------
    An array of binary words, each encoding a series of angles.

    Raises
    ------
    ValueError
        If entries in the array of angles don't match `num_bits`.

    """
    integer_angles = np.asarray(integer_angles)
    if np.any((integer_angles < 0) | (integer_angles >= 2**num_bits)):
        error_msg = f"Entries must be in [0, 2**{num_bits}) to fit in the word."
        raise ValueError(error_msg)
    num_angles = integer_angles.shape[-1]
    words = np.empty(
        integer_angles.shape[:-1], dtype=object
    )  # `object` as words can exceed 64 bits.
    for idx in np.ndindex(words.shape):
        words[idx] = sum(
            int(integer_angles[(*idx, k)]) << (num_bits * (num_angles - 1 - k))
            for k in range(num_angles)
        )
    return words
