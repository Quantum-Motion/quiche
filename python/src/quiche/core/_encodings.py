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
    radian_angles: ArrayLike, num_bits: int, *, rng: np.random.Generator | None = None
) -> NDArray[np.int64]:
    """
    Encode the radian angles as integers.

    Each angle ``theta`` in ``radian_angles`` is reduced modulo ``2*pi``,
    expressed as a fraction ``x`` of ``4*pi``, and scaled onto a grid of
    ``2**num_bits`` points:

        x = (theta mod 2*pi) / (4*pi)    in [0, 1/2)
        q = round(x * 2**num_bits)       (see ``rng`` for the rounding mode)
        n = q mod 2**(num_bits - 1)      (rounding can land exactly on 2**(num_bits - 1))

    The result ``n`` is the encoded integer, so each entry occupies at most
    ``num_bits - 1`` bits. Period ``4*pi`` follows the [Qualtran]_ phase gradient convention.

    Parameters
    ----------
    radian_angles : ArrayLike
        An array-like object containing radian angles, as floats.
    num_bits : int
        Number of bits used to encode each angle.
    rng : np.random.Generator, optional
        Random number generator for unbiased rounding ([Low2025]_, appendix D).
        If None (default), angles are rounded to the nearest integer.

    Returns
    -------
    NDArray[np.int64]
        Integer encodings of the angles, with the same shape as ``radian_angles``.
        Each entry lies in ``[0, 2**(num_bits - 1))``.

    Raises
    ------
    ValueError
        If `num_bits` < 2.
    ValueError
        If angles are NaN or infinite.

    References
    ----------
    .. [Low2025] G. H. Low et al., "Fast Quantum Simulation of Electronic Structure
        by Spectral Amplification", Phys. Rev. X, vol. 15, no. 4, p. 041016,
        Oct. 2025.
    .. [Qualtran] https://qualtran.readthedocs.io/en/latest/bloqs/rotations/
                    rz_via_phase_gradient.html

    """
    if num_bits < 2:
        error_msg = f"`num_bits` must be at least 2, got {num_bits}."
        raise ValueError(error_msg)

    radian_angles = np.asarray(radian_angles, dtype=float)
    if not np.all(np.isfinite(radian_angles)):
        error_msg = "`angles` must be finite, got NaN or infinite values."
        raise ValueError(error_msg)
        
    fractional_angle_values = np.mod(radian_angles, 2 * np.pi) / (4 * np.pi)
    scaled_angles = fractional_angle_values * 2**num_bits
    if rng is None:
        rounded_angles = np.rint(scaled_angles)
    else:
        # Round up with probability equal to the fractional part, so that the
        # expected value of the rounded integer is the unrounded value.
        dither = rng.random(scaled_angles.shape)
        rounded_angles = np.floor(scaled_angles + dither)
    return np.mod(rounded_angles.astype(np.int64), 2 ** (num_bits - 1))


def pack_into_word(integer_angles: ArrayLike, num_bits: int) -> NDArray[np.object_]:
    """
    Pack integers along the last axis.

    W = sum_k q_k * 2**(num_bits * (K - 1 - k))

    Parameters
    ----------
    integer_angles : ArrayLike
        An array-like object containing angles, encoded as .
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
    )  # `object` because words can exceed 64 bits.
    for idx in np.ndindex(words.shape):
        words[idx] = sum(
            int(integer_angles[(*idx, k)]) << (num_bits * (num_angles - 1 - k))
            for k in range(num_angles)
        )
    return words
