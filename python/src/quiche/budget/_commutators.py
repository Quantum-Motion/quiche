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

"""
Trotter error constants from exact nested commutators of Pauli terms.

For Pauli words `P` and `Q`, `[Q, P]` is zero if they commute and `2 QP` otherwise, so
its norm is exactly `2` or `0`. `R` anticommutes with the product `QP` if and only if
it anticommutes with exactly one of `P` and `Q`, so every nested commutator norm follows
from the pairwise anticommutation matrix. The triangle inequality over the terms of
[Childs2021, Prop. 10] is then the only loosening.

Both constants assume the formula applies the terms in `PauliSum` order, first term
first, as every backend does: `S_1(dt) = e^{-i dt H_G} ... e^{-i dt H_1}` and
`S_2(dt) = S_1(dt / 2)^reversed S_1(dt / 2)`.
"""

import numpy as np
from numpy.typing import NDArray

from quiche.core import Pauli, PauliSum

# Rows of the anticommutation matrix built per block, bounding memory to a few
# `_BLOCK_ROWS * n_terms` arrays rather than `n_terms^2`.
_BLOCK_ROWS = 256


def _symplectic(paulis: PauliSum) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Get the X and Z bits of each term, as `(n_terms, n_qubits)` 0/1 arrays."""
    x = np.zeros((paulis.n_terms, paulis.n_qubits))
    z = np.zeros((paulis.n_terms, paulis.n_qubits))
    for row, word in enumerate(paulis.terms):
        for pauli, qubit in zip(word.terms, word.qubits, strict=True):
            x[row, qubit] = pauli in (Pauli.X, Pauli.Y)
            z[row, qubit] = pauli in (Pauli.Z, Pauli.Y)
    return x, z


def _anticommutation_rows(
    x: NDArray[np.float64], z: NDArray[np.float64], rows: slice
) -> NDArray[np.float64]:
    """Get rows of `A[i, j]`, 1 if terms `i` and `j` anticommute and 0 otherwise."""
    # Symplectic form: the number of qubits where the two words differ and neither is
    # the identity, mod 2.
    return (x[rows] @ z.T + z[rows] @ x.T) % 2


def _blocks(n: int) -> list[slice]:
    """Split `range(n)` into slices of at most `_BLOCK_ROWS` rows."""
    return [slice(i, min(i + _BLOCK_ROWS, n)) for i in range(0, n, _BLOCK_ROWS)]


def anticommutation_matrix(paulis: PauliSum) -> NDArray[np.bool_]:
    """Get `A[i, j]`, whether terms `i` and `j` of a PauliSum anticommute."""
    x, z = _symplectic(paulis)
    return _anticommutation_rows(x, z, slice(None)).astype(bool)


def first_order_constant(paulis: PauliSum) -> float:
    """
    Get the first-order constant `C_1 = sum_{i<j} |h_i h_j| [P_i, P_j anticommute]`.

    [Childs2021, Prop. 9]:
    `||S_1(dt) - U(dt)|| <= dt^2 / 2 sum_i ||[sum_{j>i} H_j, H_i]||`.
    """
    x, z = _symplectic(paulis)
    w = np.abs(paulis.coefficients)
    n = paulis.n_terms
    total = 0.0
    for rows in _blocks(n):
        later = np.arange(n)[None, :] > np.arange(n)[rows, None]  # [j > i]
        total += float(w[rows] @ (later * _anticommutation_rows(x, z, rows)) @ w)
    return total


def second_order_constant(paulis: PauliSum) -> float:
    """
    Get the second-order constant from [Childs2021, Prop. 10].

    `||S_2(dt) - U(dt)|| <= dt^3 (sum_i ||[sum_{k>i} H_k, [sum_{j>i} H_j, H_i]]|| / 12
    + sum_i ||[H_i, [sum_{j>i} H_j, H_i]]|| / 24)`. With `w = |h|`, each nonzero double
    commutator has norm `4 w_i w_j w_k`, so this is
    `sum_{i<j, A_ij} w_i w_j (sum_{k>i} w_k (A_ki XOR A_kj) / 3 + w_i / 6)`, and the
    XOR sum is `v_i + U_ij - 2 T_ij` with `v_i = sum_{k>i} w_k A_ki`,
    `U_ij = sum_{k>i} w_k A_kj` and `T_ij = sum_{k>i} w_k A_ki A_kj`.
    """
    x, z = _symplectic(paulis)
    a = _anticommutation_rows(x, z, slice(None))
    w = np.abs(paulis.coefficients)
    n = paulis.n_terms
    total = 0.0
    for rows in _blocks(n):
        later = np.arange(n)[None, :] > np.arange(n)[rows, None]  # [k > i]
        weights = later * w  # w_k [k > i], rows i, columns k
        u = weights @ a  # U_ij
        t = (weights * a[rows]) @ a  # T_ij
        v = (weights * a[rows]).sum(axis=1)  # v_i
        pairs = weights * a[rows]  # w_j [j > i] A_ij
        xor = v[:, None] + u - 2 * t
        total += float(w[rows] @ (pairs * (xor / 3 + w[rows, None] / 6)).sum(axis=1))
    return total
