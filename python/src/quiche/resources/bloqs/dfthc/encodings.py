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

"""Binary word encodings for QROM data loading."""

from collections.abc import Iterable
from operator import index
from typing import NewType, SupportsIndex

import numpy as np
from numpy.typing import ArrayLike, NDArray
from qualtran.linalg.lcu_util import preprocess_probabilities_for_reversible_sampling

from quiche.core.electronic import DFTHCHamiltonian

InnerData = NewType("InnerData", tuple[tuple[int, ...], ...])
"""Packed alias words for ``InnerQROM``, indexed ``[outer_index][inner_index]``."""

RotationData = NewType("RotationData", tuple[tuple[int, ...], ...])
"""Packed rotation words loaded by ``RotationQROM``, indexed ``[rank][inner_index]``."""

TailData = NewType("TailData", tuple[tuple[int, int], ...])
"""``(rotations, G_1)`` pairs for ``InnerQROMTail``, indexed ``[outer_index - RC]``."""


def _as_word_table(
    value: Iterable[Iterable[SupportsIndex]],
) -> tuple[tuple[int, ...], ...]:
    """
    Coerce a nested iterable of integer words to nested tuples of Python integers.

    Parameters
    ----------
    value : Iterable[Iterable[SupportsIndex]]
        Nested iterable of integer words, indexed ``[row][column]``.

    Returns
    -------
    tuple[tuple[int, ...], ...]
        The words as a tuple of tuples of ``int``.

    Raises
    ------
    TypeError
        If any word is not an integer (e.g. a float).
    """  # ruff: ignore[docstring-extraneous-exception]
    return tuple(tuple(index(word) for word in row) for row in value)


def as_inner_data(
    value: Iterable[Iterable[SupportsIndex]] | None,
) -> InnerData | None:
    """
    Coerce packed alias sampling words to nested tuples of Python integers.

    Used as the attrs converter for every ``inner_data`` field, so that lists and
    NumPy arrays are accepted. Shape and bit width are validated by ``InnerQROM``.

    Parameters
    ----------
    value : Iterable[Iterable[SupportsIndex]] | None
        Nested iterable of integer words, indexed ``[outer_index][inner_index]``,
        or None.

    Returns
    -------
    InnerData | None
        The words as a tuple of tuples of ``int``, or None if ``value`` is None.

    Raises
    ------
    TypeError
        If any word is not an integer (e.g. a float).
    """  # ruff: ignore[docstring-extraneous-exception]
    if value is None:
        return None
    return InnerData(_as_word_table(value))


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
    """  # ruff: ignore[docstring-extraneous-exception]
    if value is None:
        return None
    return RotationData(_as_word_table(value))


def as_tail_data(
    value: Iterable[tuple[SupportsIndex, SupportsIndex]] | None,
) -> TailData | None:
    """
    Coerce tail words and `G_1` bits to tuples of Python integers.

    Used as the attrs converter for every ``tail_data`` field, so that lists and
    NumPy arrays are accepted. Length and bit widths are validated by
    ``InnerQROMTail``.

    Parameters
    ----------
    value : Iterable[tuple[SupportsIndex, SupportsIndex]] | None
        Iterable of ``(rotations, G_1)`` pairs, indexed ``[outer_index - RC]``,
        or None.

    Returns
    -------
    TailData | None
        The pairs as a tuple of tuples of ``int``, or None if ``value`` is None.

    Raises
    ------
    TypeError
        If any entry is not an integer (e.g. a float).
    """  # ruff: ignore[docstring-extraneous-exception]
    if value is None:
        return None
    return TailData(tuple((index(word), index(g_1)) for word, g_1 in value))


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
        need to be encoded into angles.

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


def tail_qrom_angles(
    dfthc_hamiltonian: DFTHCHamiltonian,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """
    Compute the Givens rotation angles and `G_1` bits loaded by ``InnerQROMTail``.

    Applies :func:`givens_angles` to the eigenvectors of the sum-of-squares one body
    matrix (eq. (26) of [Low2025]_). Non-negative eigenvalues give the :math:`D_1`
    terms and negative ones the :math:`Q_1` terms, which are flagged by
    `G_1` = 1 ([Low2025]_, appendix B). As in eq. (B4), the :math:`D_1` terms
    come first.

    Parameters
    ----------
    dfthc_hamiltonian : DFTHCHamiltonian
        A QUICHE DFTHCHamiltonian object containing the one body term that
        needs to be encoded into angles.

    Returns
    -------
    radian_angles : NDArray[np.float64]
        An array of rotation (Givens) angles in radians of shape ``(N, N-1)``.
    negative : NDArray[np.bool_]
        The `G_1` bit of each term, of shape ``(N,)``.

    References
    ----------
    .. [Low2025] G. H. Low et al., "Fast Quantum Simulation of Electronic Structure
        by Spectral Amplification", Phys. Rev. X, vol. 15, no. 4, p. 041016,
        Oct. 2025.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(dfthc_hamiltonian.sos_one_body())

    # `np.eigh` sorts in ascending order, so reversing puts the D_1 terms
    # first (eq. (B4)).
    eigenvalues = eigenvalues[::-1]

    # Reverse eigenvectors to match. Transpose because `givens_angles` takes
    # one vector per row.
    eigenvectors = eigenvectors[:, ::-1].T

    # `G_1` = 1 flags Q_1 terms, which have negative eigenvalues.
    return givens_angles(eigenvectors), eigenvalues < 0


def inner_qrom_coefficients(dfthc_hamiltonian: DFTHCHamiltonian) -> NDArray[np.float64]:
    r"""
    Compute the signed SF coefficients that ``InnerQROM`` alias samples over.

    Each squared factor of the sum-of-squares representation (eq. (23) of
    [Low2025]_) is a linear combination of :math:`Z` operators on the rotated
    spin-orbitals and the identity, as applied by ``SelectDFTHC``. The coefficients
    are calculated as follows.

    1. Write each number operator as :math:`n_{b\sigma} = (I - Z_{b\sigma})/2`, so
       that the squared factor :math:`(r, c)` becomes

       .. math::

           \alpha_{rc} I + \sum_b w_b^{(rc)} \sum_\sigma n_{b\sigma}
           = w_B^{(rc)} I - \frac{1}{2} \sum_b w_b^{(rc)} \sum_\sigma Z_{b\sigma}.

    2. Collect the identity coefficient (below eq. (24)):

       .. math::

           w_B^{(rc)} = \alpha_{rc} + \sum_b w_b^{(rc)}.

    3. Order the coefficients of each factor as :math:`Z` terms first, then the
       identity term at :math:`b = B`:

       .. math::

           \left(-w_0^{(rc)}, \ldots, -w_{B-1}^{(rc)}, w_B^{(rc)}\right).

    4. Label factor :math:`(r, c)` by the outer index :math:`x_o = rC + c`, giving
        coefficients :math:`c_{x_o, b}` with :math:`x_o \in [0, RC)` and
        :math:`b \in [0, B]`.

        The spin sum and the factor :math:`1/2` are handled by the spin qubit, not the
        alias sampling. Here :math:`w_b^{(rc)}` are the `weight_vectors` and
        :math:`\alpha_{rc}` the `identity_weights`.

    Parameters
    ----------
    dfthc_hamiltonian : DFTHCHamiltonian
        A QUICHE DFTHCHamiltonian object containing the weights that need to be
        encoded into alias sampling words.

    Returns
    -------
    NDArray[np.float64]
        An array of signed coefficients of shape ``(R * C, B + 1)``, indexed
        ``[outer_index][inner_index]``, with the identity term last.

    References
    ----------
    .. [Low2025] G. H. Low et al., "Fast Quantum Simulation of Electronic Structure
        by Spectral Amplification", Phys. Rev. X, vol. 15, no. 4, p. 041016,
        Oct. 2025.
    """
    num_factors = dfthc_hamiltonian.num_ranks * dfthc_hamiltonian.num_copies
    num_inner = dfthc_hamiltonian.num_bases + 1

    # (R, B, C) -> (R, C, B), so that each row of the result is one factor (r, c).
    weights = dfthc_hamiltonian.weight_vectors.transpose(0, 2, 1)

    alphas = dfthc_hamiltonian.identity_weights[:, :, np.newaxis]
    identity = alphas + weights.sum(axis=2, keepdims=True)

    coefficients = np.concatenate([-weights, identity], axis=2)
    return coefficients.reshape(num_factors, num_inner)


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
    .. [Qualtran] https://qualtran.readthedocs.io/en/latest/bloqs/rotations/rz_via_phase_gradient.html
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


def encode_inner_data(
    coefficients: ArrayLike,
    num_bits_keep: int,
    *,
    rng: np.random.Generator | None = None,
) -> InnerData:
    r"""
    Encode signed coefficients as the alias sampling words loaded by the inner QROM.

    Each row of ``coefficients`` holds the :math:`L = B + 1` coefficients
    :math:`c_i` of one SF term. They are alias sampled over their magnitudes
    :math:`|c_i|` ([Low2025]_, appendix B, step 4), with a keep register of
    :math:`b_k` = ``num_bits_keep`` bits.

    1. Spread :math:`M = L 2^{b_k}` bins over the entries. With the cumulative
       fractions :math:`P_i = \sum_{j < i} |c_j| / \sum_j |c_j|` and an offset
       :math:`u` shared by the whole row, entry :math:`i` gets

       .. math::

           n_i = \lfloor M P_{i+1} + u \rfloor - \lfloor M P_i + u \rfloor

       bins. Every :math:`n_i \geq 0`, :math:`\sum_i n_i = M`, and each
       :math:`n_i` is within one bin of :math:`M |c_i| / \sum_j |c_j|`. With
       :math:`u = 1/2` the cumulative counts are rounded to the nearest bin. With
       :math:`u` uniform in :math:`[0, 1)` the counts are unbiased,
       :math:`\mathbb{E}[n_i] = M |c_i| / \sum_j |c_j|` ([Low2025]_, appendix D).

    2. Build the alias table, :math:`\mathrm{alt}_i` and :math:`\mathrm{keep}_i`,
       from the bin counts with ``preprocess_probabilities_for_reversible_sampling``
       ([Qualtran]_). The counts are whole numbers adding up to :math:`M`, so
       entry :math:`i` is sampled with probability exactly :math:`n_i / M`.

    3. Add two sign bits, :math:`s_i = [c_i < 0]` and
       :math:`d_i = s_i \oplus s_{\mathrm{alt}_i}`. With :math:`f = 1` if the alias
       is swapped in and :math:`f = 0` otherwise, the phase
       :math:`(-1)^{s_i \oplus f d_i}` is the sign of the sampled coefficient.
       :math: `f` is the ``compare`` register of ``InnerPrepareDFTHC``.

    4. Pack each entry into one word, with :math:`\mathrm{alt}_i` in the most
       significant bits: :math:`(\mathrm{alt}_i, \mathrm{keep}_i, s_i, d_i)`.

    Parameters
    ----------
    coefficients : ArrayLike
        An array-like object of shape ``(R * C, B + 1)`` containing the signed
        coefficients, as floats.
    num_bits_keep : int
        Number of bits for the keep probability, :math:`b_{k2}` in [Low2025]_.
    rng : np.random.Generator, optional
        Random number generator for unbiased rounding ([Low2025]_, appendix D).
        If None (default), the cumulative bin counts are rounded to the nearest
        integer.

    Returns
    -------
    InnerData
        Packed alias sampling words, indexed ``[outer_index][inner_index]``. Each
        word is a Python ``int`` of :math:`\lceil \log(B + 1) \rceil + b_{k2} + 2`
        bits.

    Raises
    ------
    ValueError
        If `num_bits_keep` < 1.
    ValueError
        If coefficients are NaN or infinite.
    ValueError
        If any row of coefficients is entirely zero.

    References
    ----------
    .. [Low2025] G. H. Low et al., "Fast Quantum Simulation of Electronic Structure
        by Spectral Amplification", Phys. Rev. X, vol. 15, no. 4, p. 041016,
        Oct. 2025.
    .. [Qualtran] https://qualtran.readthedocs.io/en/latest/bloqs/state_preparation/state_preparation_alias_sampling.html
    """
    if num_bits_keep < 1:
        error_msg = f"`num_bits_keep` must be at least 1, got {num_bits_keep}."
        raise ValueError(error_msg)

    coefficients = np.asarray(coefficients, dtype=float)
    if not np.all(np.isfinite(coefficients)):
        error_msg = "`coefficients` must be finite, got NaN or infinite values."
        raise ValueError(error_msg)

    if not np.all(np.any(coefficients, axis=-1)):
        error_msg = "Each row of `coefficients` must have a non-zero entry."
        raise ValueError(error_msg)

    num_rows, num_inner = coefficients.shape
    num_bins = num_inner * 2**num_bits_keep

    # Step 1, cumulative distribution of shape (R * C, B + 2).
    cumulative = np.cumsum(np.abs(coefficients), axis=1)
    cumulative = np.concatenate(
        [np.zeros((num_rows, 1)), cumulative / cumulative[:, -1:]], axis=1
    )
    offsets = np.full((num_rows, 1), 0.5) if rng is None else rng.random((num_rows, 1))
    bin_counts = np.diff(np.floor(cumulative * num_bins + offsets), axis=1)

    words = []
    for row, row_counts in zip(coefficients, bin_counts, strict=True):
        # Step 2. The counts are whole bins, so qualtran's own rounding keeps them.
        alternates, keep_numers, _ = preprocess_probabilities_for_reversible_sampling(
            row_counts.tolist(), sub_bit_precision=num_bits_keep
        )
        alt = np.array(alternates, dtype=np.int64)
        keep = np.array(keep_numers, dtype=np.int64)

        # Step 3.
        sign = (row < 0).astype(np.int64)
        swap_sign = sign ^ sign[alt]

        # Step 4.
        row_words = (alt << (num_bits_keep + 2)) | (keep << 2) | (sign << 1) | swap_sign
        words.append(tuple(row_words.tolist()))

    return InnerData(tuple(words))


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
    RotationData
        Packed rotation words, indexed ``[rank][inner_index]``. Each word is a Python
        ``int`` of ``(N - 1) * num_bits`` bits, with the first angle in the most
        significant bits.
    """
    integers = quantise_angles(radian_angles, num_bits, rng=rng)
    return RotationData(
        tuple(
            tuple(pack_integers(angles, num_bits) for angles in row) for row in integers
        )
    )


def encode_tail_data(
    radian_angles: ArrayLike,
    negative: ArrayLike,
    num_bits: int,
    *,
    rng: np.random.Generator | None = None,
) -> TailData:
    """
    Encode the Givens rotation angles and `G_1` bits as words loaded by the tail QROM.

    Quantises and packs the angles of each :math:`D_1/Q_1` term as
    :func:`encode_rotation_data` does, and pairs each word with its `G_1` bit,
    as expected by ``InnerQROMTail``.

    Parameters
    ----------
    radian_angles : ArrayLike
        An array-like object of shape ``(N, N-1)`` containing radian angles, as floats.
    negative : ArrayLike
        An array-like object of shape ``(N,)`` containing the `G_1` bits.
    num_bits : int
        Number of bits used to encode each angle.
    rng : np.random.Generator, optional
        Random number generator for unbiased rounding ([Low2025]_, appendix D).
        If None (default), angles are rounded to the nearest integer.

    Returns
    -------
    TailData
        ``(rotations, G_1)`` pairs, indexed ``[outer_index - RC]``. Each rotation word
        is a Python ``int`` of ``(N - 1) * num_bits`` bits, with the first angle in
        the most significant bits.
    """
    integers = quantise_angles(radian_angles, num_bits, rng=rng)
    return TailData(
        tuple(
            (pack_integers(angles, num_bits), int(flag))
            for angles, flag in zip(integers, np.asarray(negative), strict=True)
        )
    )
