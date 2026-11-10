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

from collections.abc import Iterable
from operator import index
from typing import SupportsIndex

import numpy as np
from numpy.typing import ArrayLike, NDArray

from quiche.core.electronic import DFTHCHamiltonian

type RotationData = tuple[tuple[int, ...], ...]
"""Packed rotation words loaded by ``RotationQROM``, indexed ``[rank][inner_index]``."""


def as_rotation_data(
    value: Iterable[Iterable[SupportsIndex]] | None,
) -> RotationData | None:
    """
    Coerce packed rotation words to nested tuples of Python integers.

    Used as the attrs converter for every ``rotation_data`` field, so that lists and
    NumPy arrays are accepted. Shape and bit width are validated by ``RotationQROM``.

    Parameters
    ----------
    value : Iterable[Iterable[SupportsIndex]] | None
        Nested iterable of integer words, indexed ``[rank][inner_index]``, or None.

    Returns
    -------
    RotationData | None
        The words as a tuple of tuples of ``int``, or None if ``value`` is None.

    Raises
    ------
    TypeError
        If any word is not an integer (e.g. a float).

    """
    if value is None:
        return None
    return tuple(tuple(index(word) for word in row) for row in value)


def givens_angles(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the Givens rotation angles that encode each vector along the last axis.

    Each vector of length :math:`N` is encoded as the :math:`N - 1` angles of a
    ladder of nearest-neighbour Givens rotations ([Low2025]_, appendix 5B). Angle
    :math:`k` is ``arctan2(norm(v[k + 1:]), v[k])``, except that the last angle keeps
    the sign of the final component. The angles are unchanged by positive rescaling,
    so the vectors do not need to be normalised.

    Parameters
    ----------
    vectors : NDArray[np.float64]
        Real array of shape ``(..., N)``. Each vector along the last axis is encoded
        independently.

    Returns
    -------
    NDArray[np.float64]
        An array of rotation (Givens) angles in radians of shape ``(R, B, N-1)``.

    References
    ----------
    .. [Low2025] G. H. Low et al., "Fast Quantum Simulation of Electronic Structure
        by Spectral Amplification", Phys. Rev. X, vol. 15, no. 4, p. 041016,
        Oct. 2025.

    """
    current = vectors[..., :-1]
    next_ = vectors[..., 1:]

    suffix_sum_squares = np.flip(
        np.cumsum(np.flip(next_**2, axis=-1), axis=-1), axis=-1
    )
    suffix_norm = np.sqrt(suffix_sum_squares)

    # The last remainder is the final coordinate itself; it keeps its sign,
    # otherwise flipping it would give the same angles.
    suffix_norm[..., -1:] = next_[..., -1:]

    return np.arctan2(suffix_norm, current)


def rotation_qrom_angles(dfthc_hamiltonian: DFTHCHamiltonian) -> NDArray[np.float64]:
    """
    Compute the Givens rotation angles loaded by ``RotationQROM``.

    Applies :func:`givens_angles` to the ``unit_vectors`` of a DFTHC Hamiltonian,
    giving the basis rotation angles for every unit vector :math:`u_b^{(r)}`
    ([Low2025]_, appendix 5B).

    Parameters
    ----------
    dfthc_hamiltonian : DFTHCHamiltonian
        A QUICHE DFTHCHamiltonian object containing the factors that
        need to be encoded into angles

    Returns
    -------
    NDArray[np.float64]
        An array of rotation (Givens) angles in radians of shape ``(R, B, N-1)``.

    References
    ----------
    .. [Low2025] G. H. Low et al., "Fast Quantum Simulation of Electronic Structure
        by Spectral Amplification", Phys. Rev. X, vol. 15, no. 4, p. 041016,
        Oct. 2025.

    """
    return givens_angles(dfthc_hamiltonian.unit_vectors)


def quantise_angles(
    radian_angles: ArrayLike, num_bits: int, *, rng: np.random.Generator | None = None
) -> NDArray[np.int64]:
    r"""
    Encode the radian angles as integers.

    Each angle ``theta`` in ``radian_angles`` is reduced modulo ``2*pi``,
    expressed as a fraction ``x`` of ``4*pi``, and scaled onto a grid of
    ``2**num_bits`` points:

    .. math::

        \begin{aligned}
            x &= \frac{\theta \bmod 2\pi}{4\pi} \in \left[0, \tfrac{1}{2}\right), \\
            q &= \mathrm{round}\left(x \, 2^{b}\right), \\
            n &= q \bmod 2^{b-1}.
        \end{aligned}

    The result ``n`` is the encoded integer, so each entry occupies at most
    ``num_bits - 1`` bits.
    Period ``4*pi`` follows the [Qualtran]_ phase gradient convention.

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
        error_msg = "`radian_angles` must be finite, got NaN or infinite values."
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


def pack_integers(integer_angles: ArrayLike, num_bits: int) -> int:
    r"""
    Pack an integer array into a single binary word.

    The ``num_bits``-bit integers ``q_0, ..., q_{K-1}`` are concatenated into
    one ``K * num_bits``-bit integer ``W``, with ``q_0`` in the most significant bits:

     .. math::

        W = \sum_{k=0}^{K-1} q_k * 2^{b * (K - 1 - k)},

    where :math:`b` is ``num_bits``.

    This is the bit string loaded into a register holding ``K`` consecutive
    ``num_bits``-bit angles.

    Parameters
    ----------
    integer_angles : ArrayLike
        An array-like object containing angles, encoded as integers in
        ``[0, 2**num_bits)``.
    num_bits : int
        Number of bits used to encode each angle.

    Returns
    -------
    int
        Python ``int`` holding the ``K`` integers packed into one
        word, with the first in the most significant bits.

    Raises
    ------
    ValueError
        If entries in the array of angles don't match `num_bits`.

    """
    values = np.asarray(integer_angles).tolist()
    if any(not 0 <= value < 2**num_bits for value in values):
        error_msg = f"Entries must be in [0, 2**{num_bits}) to fit in the word."
        raise ValueError(error_msg)
    word = 0
    for value in values:
        word = (word << num_bits) | value
    return word


def encode_rotation_data(
    radian_angles: ArrayLike, num_bits: int, *, rng: np.random.Generator | None = None
) -> RotationData:
    """
    Encode Givens rotation angles as the words loaded by the rotation QROM.

    Takes an array-like object containing radian angles as floats and quantises the
    angles to integers, before subsequently packing the last axis into binary words.
    The words are returned as nested tuples of Python integers, as expected by
    ``RotationQROM``.

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
    tuple[tuple[int, ...], ...]
        Packed rotation words, indexed ``[rank][inner_index]``. Each word is a Python
        ``int`` of ``(N - 1) * num_bits`` bits, with the first angle in the most
        significant bits.

    """
    integers = quantise_angles(radian_angles, num_bits, rng=rng)
    return tuple(
        tuple(pack_integers(angles, num_bits) for angles in row) for row in integers
    )
