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

"""Tests for models module."""

import numpy as np
import pytest
from numpy.typing import NDArray

from quiche.core import Pauli, PauliSum, PauliWord

I_MATRIX = np.identity(2)
X_MATRIX = np.array([[0, 1], [1, 0]])
Y_MATRIX = np.array([[0, -1j], [1j, 0]])
Z_MATRIX = np.array([[1, 0], [0, -1]])


class TestPauli:
    """Test Pauli class."""

    def test_invalid_pauli(self):
        error_msg = "'L' is not a valid Pauli"
        with pytest.raises(ValueError, match=error_msg):
            Pauli("L")

    @pytest.mark.parametrize(
        ("pauli", "expected"),
        [
            (Pauli.X, X_MATRIX),
            (Pauli.Y, Y_MATRIX),
            (Pauli.Z, Z_MATRIX),
        ],
    )
    def test_to_matrix(self, pauli: Pauli, expected: NDArray):
        """Test that Paulis correctly convert to their matrix representations."""
        np.testing.assert_equal(pauli.to_matrix(), expected)


class TestPauliWord:
    """Test PauliWord class."""

    def test_string_paulis_coerced(self):
        actual = PauliWord(paulis=("X", "Y", "Z"), qubits=(0, 1, 2))
        expected = PauliWord(paulis=(Pauli.X, Pauli.Y, Pauli.Z), qubits=(0, 1, 2))
        assert actual == expected
        # Ensure actually coerced since StrEnums compare equal to strings
        assert all(type(p) is Pauli for p in actual.paulis)

    def test_length_mismatch(self):
        error_msg = "The paulis and qubits of the PauliWord must be the same length"
        with pytest.raises(ValueError, match=error_msg):
            PauliWord(paulis=("X", "Y"), qubits=(1,))

    def test_duplicate_qubits(self):
        error_msg = "The target qubits of the PauliWord must be unique."
        with pytest.raises(ValueError, match=error_msg):
            PauliWord(paulis=("X", "Y", "Z"), qubits=(0, 1, 0))

    @pytest.mark.parametrize(
        ("qubit", "error_msg"),
        [
            (-1, "greater than or equal to 0"),
            (1.5, "valid integer"),
            ("c", "valid integer"),
        ],
    )
    def test_invalid_qubit(self, qubit: object, error_msg: str):
        with pytest.raises(ValueError, match=error_msg):
            PauliWord(paulis=("X",), qubits=(qubit,))

    def test_invalid_pauli(self):
        error_msg = "Input should be 'X', 'Y' or 'Z'"
        with pytest.raises(ValueError, match=error_msg):
            PauliWord(paulis=("L",), qubits=(1,))

    @pytest.mark.parametrize(
        ("paulis", "qubits", "expected_paulis", "expected_qubits"),
        [
            (("X", "Z"), (0, 2), ("X", "Z"), (0, 2)),
            (("Z", "X"), (2, 0), ("X", "Z"), (0, 2)),
            (
                ("Y", "X", "Z"),
                (5, 1, 3),
                ("X", "Z", "Y"),
                (1, 3, 5),
            ),
            (("X", "Y"), (10, 2), ("Y", "X"), (2, 10)),
            (("Y",), (4,), ("Y",), (4,)),
        ],
    )
    def test_sorted_on_construction(
        self,
        paulis: tuple[str, ...],
        qubits: tuple[int, ...],
        expected_paulis: tuple[str, ...],
        expected_qubits: tuple[int, ...],
    ):
        word = PauliWord(paulis=paulis, qubits=qubits)
        assert word.paulis == expected_paulis
        assert word.qubits == expected_qubits

    def test_to_matrix(self):
        word = PauliWord(paulis=("X", "Y", "Z"), qubits=(0, 1, 2))
        actual = word.to_matrix(ignore_idle_qubits=True)
        expected = np.kron(np.kron(X_MATRIX, Y_MATRIX), Z_MATRIX)
        np.testing.assert_equal(actual, expected)

    def test_to_matrix_include_idle_qubits(self):
        word = PauliWord(paulis=("X", "Z"), qubits=(0, 2))
        actual = word.to_matrix(ignore_idle_qubits=False)
        expected = np.kron(np.kron(X_MATRIX, I_MATRIX), Z_MATRIX)
        np.testing.assert_equal(actual, expected)

    def test_to_matrix_length(self):
        word = PauliWord(paulis=("X", "X", "Y"), qubits=(0, 1, 3))
        actual = word.to_matrix(length=5, ignore_idle_qubits=False)
        expected = np.kron(
            np.kron(np.kron(np.kron(X_MATRIX, X_MATRIX), I_MATRIX), Y_MATRIX), I_MATRIX
        )
        np.testing.assert_equal(actual, expected)


class TestPauliSum:
    """Test PauliSum class."""

    def test_no_words(self):
        error_msg = "The number of words of the PauliSum must be nonzero."
        with pytest.raises(ValueError, match=error_msg):
            PauliSum(coefficients=(), words=(), identity_coefficient=10.0)

    def test_length_mismatch(self):
        word1 = PauliWord(paulis=("X", "Y", "Z"), qubits=(0, 2, 3))
        word2 = PauliWord(paulis=("Y", "Z", "X"), qubits=(1, 2, 3))
        coeffs = (5.0,)

        error_msg = "The coefficients and words of the PauliSum must be the same length"
        with pytest.raises(ValueError, match=error_msg):
            PauliSum(coefficients=coeffs, words=(word1, word2), identity_coefficient=0)

    def test_without_identity(self, h2: PauliSum):
        filtered = h2.without_identity()
        assert filtered.identity_coefficient == 0.0
        assert filtered.words == h2.words
        assert filtered.coefficients == h2.coefficients
        assert filtered.num_words_with_identity == filtered.num_words
        assert filtered.lam == pytest.approx(h2.lam - abs(h2.identity_coefficient))

    def test_to_matrix(self):
        word1 = PauliWord(paulis=("X", "Z"), qubits=(0, 2))
        word2 = PauliWord(paulis=("Z", "Y"), qubits=(0, 1))
        coeffs = (5.0, 2.0)
        id_coeff = 10.0

        pauli_sum = PauliSum(
            coefficients=(5.0, 2.0),
            words=(word1, word2),
            identity_coefficient=id_coeff,
        )

        actual = pauli_sum.to_matrix()
        expected = (
            id_coeff * np.identity(2**pauli_sum.num_qubits)
            + coeffs[0] * np.kron(np.kron(X_MATRIX, I_MATRIX), Z_MATRIX)
            + coeffs[1] * np.kron(np.kron(Z_MATRIX, Y_MATRIX), I_MATRIX)
        )

        np.testing.assert_equal(actual, expected)
