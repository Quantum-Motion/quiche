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

ID = np.identity(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])


class TestPauli:
    """Test Pauli class."""

    def test_invalid_pauli(self):
        error_msg = "'L' is not a valid Pauli"
        with pytest.raises(ValueError, match=error_msg):
            Pauli("L")

    @pytest.mark.parametrize(
        ("term", "expected"),
        [
            (Pauli.X, X),
            (Pauli.Y, Y),
            (Pauli.Z, Z),
        ],
    )
    def test_to_matrix(self, term: Pauli, expected: NDArray):
        """Test that Pauli terms correctly convert to their matrix representations."""
        np.testing.assert_equal(term._to_matrix(), expected)


class TestPauliWord:
    """Test PauliWord class."""

    def test_length_mismatch(self):
        error_msg = "The terms and qubits of the PauliWord must be the same length"
        with pytest.raises(ValueError, match=error_msg):
            PauliWord(terms=(Pauli.X, Pauli.Y), qubits=(1,))

    def test_invalid_qubit(self):
        err_msg = "Input should be a valid integer, got a number with a fractional part"
        with pytest.raises(ValueError, match=err_msg):
            PauliWord(terms=(Pauli.X,), qubits=(1.5,))

    def test_invalid_pauli(self):
        error_msg = "Input should be 'X', 'Y' or 'Z'"
        with pytest.raises(ValueError, match=error_msg):
            PauliWord(terms=("L",), qubits=(1,))

    def test_to_matrix(self):
        word = PauliWord(terms=(Pauli.X, Pauli.Y, Pauli.Z), qubits=(0, 1, 2))
        actual = word._to_matrix(ignore_idle_qubits=True)
        expected = np.kron(np.kron(X, Y), Z)
        np.testing.assert_equal(actual, expected)

    def test_to_matrix_include_idle_qubits(self):
        word = PauliWord(terms=(Pauli.X, Pauli.Z), qubits=(0, 2))
        actual = word._to_matrix(ignore_idle_qubits=False)
        expected = np.kron(np.kron(X, ID), Z)
        np.testing.assert_equal(actual, expected)

    def test_to_matrix_length(self):
        word = PauliWord(terms=(Pauli.X, Pauli.X, Pauli.Y), qubits=(0, 1, 3))
        actual = word._to_matrix(length=5, ignore_idle_qubits=False)
        expected = np.kron(np.kron(np.kron(np.kron(X, X), ID), Y), ID)
        np.testing.assert_equal(actual, expected)


class TestPauliSum:
    """Test PauliSum class."""

    def test_length_mismatch(self):
        word1 = PauliWord(terms=(Pauli.X, Pauli.Y, Pauli.Z), qubits=(0, 2, 3))
        word2 = PauliWord(terms=(Pauli.Y, Pauli.Z, Pauli.X), qubits=(1, 2, 3))
        coeffs = (5.0,)

        error_msg = "The coefficients and terms of the PauliSum must be the same length"
        with pytest.raises(ValueError, match=error_msg):
            PauliSum(coefficients=coeffs, terms=(word1, word2), identity_coefficient=0)

    def test_to_matrix(self):
        word1 = PauliWord(terms=(Pauli.X, Pauli.Z), qubits=(0, 2))
        word2 = PauliWord(terms=(Pauli.Z, Pauli.Y), qubits=(0, 1))
        coeffs = (5.0, 2.0)
        id_coeff = 10.0

        psum = PauliSum(
            coefficients=(5.0, 2.0),
            terms=(word1, word2),
            identity_coefficient=id_coeff,
        )

        actual = psum._to_matrix()
        expected = (
            id_coeff * np.identity(2**psum.n_qubits)
            + coeffs[0] * np.kron(np.kron(X, ID), Z)
            + coeffs[1] * np.kron(np.kron(Z, Y), ID)
        )

        np.testing.assert_equal(actual, expected)
