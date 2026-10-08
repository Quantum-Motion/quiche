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

    @pytest.mark.parametrize(
        ("string", "big_endian", "expected"),
        [
            ("X", True, PauliWord(paulis=("X",), qubits=(0,))),
            ("XIZ", True, PauliWord(paulis=("X", "Z"), qubits=(0, 2))),
            ("XIZ", False, PauliWord(paulis=("Z", "X"), qubits=(0, 2))),
            ("XYZ", False, PauliWord(paulis=("Z", "Y", "X"), qubits=(0, 1, 2))),
            ("IIY", True, PauliWord(paulis=("Y",), qubits=(2,))),
            ("IIY", False, PauliWord(paulis=("Y",), qubits=(0,))),
            ("  XZ  ", True, PauliWord(paulis=("X", "Z"), qubits=(0, 1))),
        ],
    )
    def test_from_dense_string(
        self, string: str, *, big_endian: bool, expected: PauliWord
    ):
        assert PauliWord.from_dense_string(string, big_endian=big_endian) == expected

    @pytest.mark.parametrize(
        ("string", "error_msg"),
        [
            ("XAZ", "Invalid Pauli 'A'"),
            ("xz", "Invalid Pauli 'x'"),
            ("X Z", "Invalid Pauli ' '"),
            ("X(0)", r"Invalid Pauli '\('"),
            ("", "no non-identity Paulis"),
            ("III", "no non-identity Paulis"),
        ],
    )
    def test_from_dense_string_invalid(self, string: str, error_msg: str):
        with pytest.raises(ValueError, match=error_msg):
            PauliWord.from_dense_string(string, big_endian=True)

    @pytest.mark.parametrize(
        ("string", "expected"),
        [
            ("X(0)", PauliWord(paulis=("X",), qubits=(0,))),
            ("X(0) Z(2)", PauliWord(paulis=("X", "Z"), qubits=(0, 2))),
            ("Z(2) X(0)", PauliWord(paulis=("X", "Z"), qubits=(0, 2))),
            ("Y(12)", PauliWord(paulis=("Y",), qubits=(12,))),
            ("X(007)", PauliWord(paulis=("X",), qubits=(7,))),
            ("  X(0)   Y(3) ", PauliWord(paulis=("X", "Y"), qubits=(0, 3))),
        ],
    )
    def test_from_sparse_string(self, string: str, expected: PauliWord):
        assert PauliWord.from_sparse_string(string) == expected

    @pytest.mark.parametrize(
        ("string", "error_msg"),
        [
            ("", "no non-identity Paulis"),
            ("I", "Invalid sparse Pauli"),
            ("I(0)", "Invalid sparse Pauli"),
            ("XIZ", "Invalid sparse Pauli"),
            ("X0", "Invalid sparse Pauli"),
            ("x(0)", "Invalid sparse Pauli"),
            ("A(0)", "Invalid sparse Pauli"),
            ("X(-1)", "Invalid sparse Pauli"),
            ("X(a)", "Invalid sparse Pauli"),
            ("X(0", "Invalid sparse Pauli"),
            ("X(0)Y(1)", "Invalid sparse Pauli"),
            ("X(0) Y(0)", "Qubit 0 appears more than once"),
            ("X(0) Y(00)", "Qubit 0 appears more than once"),
        ],
    )
    def test_from_sparse_string_invalid(self, string: str, error_msg: str):
        with pytest.raises(ValueError, match=error_msg):
            PauliWord.from_sparse_string(string)

    @pytest.mark.parametrize(
        ("word", "expected"),
        [
            (PauliWord(paulis=("X",), qubits=(0,)), "X(0)"),
            (
                PauliWord(paulis=("X", "Y", "Z"), qubits=(0, 1, 2)),
                "X(0) Y(1) Z(2)",
            ),
            (PauliWord(paulis=("X", "Z"), qubits=(0, 2)), "X(0) Z(2)"),
            (PauliWord(paulis=("Y", "X"), qubits=(3, 12)), "Y(3) X(12)"),
        ],
    )
    def test_to_sparse_string(self, word: PauliWord, expected: str):
        assert word.to_sparse_string() == expected

    @pytest.mark.parametrize(
        ("word", "length", "big_endian", "expected"),
        [
            (PauliWord(paulis=("Y",), qubits=(0,)), None, True, "Y"),
            (PauliWord(paulis=("Y",), qubits=(0,)), None, False, "Y"),
            (PauliWord(paulis=("X", "Z"), qubits=(0, 2)), None, True, "XIZ"),
            (PauliWord(paulis=("X", "Z"), qubits=(0, 2)), None, False, "ZIX"),
            (PauliWord(paulis=("Z",), qubits=(3,)), None, False, "ZIII"),
            (PauliWord(paulis=("X",), qubits=(0,)), 4, True, "XIII"),
        ],
    )
    def test_to_dense_string(
        self, word: PauliWord, length: int | None, *, big_endian: bool, expected: str
    ):
        assert word.to_dense_string(length, big_endian=big_endian) == expected

    @pytest.mark.parametrize("length", [-2, 0, 4])
    def test_to_dense_string_invalid_length(self, length: int):
        word = PauliWord(paulis=("X", "Y", "Z"), qubits=(1, 3, 4))
        error_msg = "Length must be greater than the maximum target qubit"
        with pytest.raises(ValueError, match=error_msg):
            word.to_dense_string(length, big_endian=True)

    @pytest.mark.parametrize("big_endian", [True, False])
    @pytest.mark.parametrize(
        "word",
        [
            PauliWord(paulis=("X",), qubits=(0,)),
            PauliWord(paulis=("X", "Y", "Z"), qubits=(1, 2, 5)),
        ],
    )
    def test_dense_round_trip(self, word: PauliWord, *, big_endian: bool):
        string = word.to_dense_string(big_endian=big_endian)
        assert PauliWord.from_dense_string(string, big_endian=big_endian) == word

    @pytest.mark.parametrize(
        "word",
        [
            PauliWord(paulis=("X",), qubits=(0,)),
            PauliWord(paulis=("X", "Y", "Z"), qubits=(1, 2, 5)),
        ],
    )
    def test_sparse_round_trip(self, word: PauliWord):
        string = word.to_sparse_string()
        assert PauliWord.from_sparse_string(string) == word


class TestPauliSum:
    """Test PauliSum class."""

    @pytest.fixture
    def example_sum(self) -> PauliSum:
        return PauliSum(
            coefficients=(5.0, -2.0),
            words=(
                PauliWord(paulis=("X", "Z"), qubits=(0, 2)),
                PauliWord(paulis=("Z", "Y"), qubits=(0, 1)),
            ),
            identity_coefficient=10.0,
        )

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

    def test_to_sparse_string(self, example_sum: PauliSum):
        expected = "+10.0\n+5.0 X(0) Z(2)\n-2.0 Z(0) Y(1)"
        actual = example_sum.to_sparse_string()
        assert actual == expected

    def test_to_sparse_string_zero_identity(self):
        pauli_sum = PauliSum(
            coefficients=(1.5,), words=(PauliWord(paulis=("X",), qubits=(0,)),)
        )
        expected = "+0.0\n+1.5 X(0)"
        actual = pauli_sum.to_sparse_string()
        assert actual == expected

    @pytest.mark.parametrize(
        ("length", "big_endian", "expected"),
        [
            (None, True, "+10.0 III\n+5.0 XIZ\n-2.0 ZYI"),
            (None, False, "+10.0 III\n+5.0 ZIX\n-2.0 IYZ"),
            (4, True, "+10.0 IIII\n+5.0 XIZI\n-2.0 ZYII"),
            (4, False, "+10.0 IIII\n+5.0 IZIX\n-2.0 IIYZ"),
        ],
    )
    def test_to_dense_string(
        self,
        example_sum: PauliSum,
        length: int | None,
        expected: str,
        *,
        big_endian: bool,
    ):
        assert example_sum.to_dense_string(length, big_endian=big_endian) == expected

    @pytest.mark.parametrize("length", [-1, 0, 2])
    def test_to_dense_string_invalid_length(self, example_sum: PauliSum, length: int):
        with pytest.raises(ValueError, match="Length must be greater"):
            example_sum.to_dense_string(length, big_endian=True)

    @pytest.mark.parametrize(
        "string",
        [
            "+10.0 \n+5.0 X(0) Z(2)\n-2.0 Z(0) Y(1)",
            "10.0 \n5.0 X(0) Z(2)\n-2e0 Z(0) Y(1)",
            "4.0 \n6.0 \n5.0 X(0) Z(2)\n-2.0 Z(0) Y(1)",
            "\n  10.0   \n\n5.0 Z(2) X(0)\n-2.0 Y(1) Z(0)\n",
        ],
    )
    def test_from_sparse_string(self, example_sum: PauliSum, string: str):
        assert PauliSum.from_sparse_string(string) == example_sum

    def test_from_sparse_string_no_identity_line(self):
        pauli_sum = PauliSum.from_sparse_string("1.5 X(0)")
        assert pauli_sum.identity_coefficient == 0.0

    @pytest.mark.parametrize(
        ("string", "error_msg"),
        [
            ("X(0)", "Invalid PauliSum line"),
            ("abc X(0)", "Invalid PauliSum line"),
            ("1.0 A(0)", "Invalid sparse Pauli"),
            ("1.0 X0", "Invalid sparse Pauli"),
            ("1.0 X(0) Y(0)", "appears more than once"),
            ("0 X(0)", "zero-valued coefficients"),
            ("", "must be nonzero"),
            ("1.0 ", "must be nonzero"),
            ("1.0 \n 2.0", "must be nonzero"),
        ],
    )
    def test_from_sparse_string_invalid(self, string: str, error_msg: str):
        with pytest.raises(ValueError, match=error_msg):
            PauliSum.from_sparse_string(string)

    @pytest.mark.parametrize(
        ("string", "big_endian"),
        [
            ("10 III\n5 XIZ\n-2 ZYI", True),
            ("10 III\n5 ZIX\n-2 IYZ", False),
        ],
    )
    def test_from_dense_string(
        self, example_sum: PauliSum, string: str, *, big_endian: bool
    ):
        assert PauliSum.from_dense_string(string, big_endian=big_endian) == example_sum

    @pytest.mark.parametrize(
        ("string", "error_msg"),
        [
            ("XZ", "Invalid PauliSum line"),
            ("abc XZ", "Invalid PauliSum line"),
            ("1.0 XAZ", "Invalid Pauli 'A'"),
            ("1.0 xz", "Invalid Pauli 'x'"),
            ("1.0 X Z", "Invalid Pauli ' '"),
            ("0 XZ", "zero-valued coefficients"),
            ("", "must be nonzero"),
            ("1.0 III", "must be nonzero"),
            ("10 I\n5 XIZ\n-2 ZY", "must have the same length"),
            ("10.0\n5.0 XIZ", "must have the same length"),
        ],
    )
    def test_from_dense_string_invalid(self, string: str, error_msg: str):
        with pytest.raises(ValueError, match=error_msg):
            PauliSum.from_dense_string(string, big_endian=True)

    @pytest.mark.parametrize("big_endian", [True, False])
    def test_dense_round_trip(self, example_sum: PauliSum, *, big_endian: bool):
        string = example_sum.to_dense_string(big_endian=big_endian)
        assert PauliSum.from_dense_string(string, big_endian=big_endian) == example_sum

    def test_sparse_round_trip(self, example_sum: PauliSum):
        string = example_sum.to_sparse_string()
        assert PauliSum.from_sparse_string(string) == example_sum
