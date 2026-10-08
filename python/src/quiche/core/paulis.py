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

"""Quantum primitives and data structures."""

from __future__ import annotations

import re
from enum import StrEnum
from functools import cached_property, reduce
from math import isclose
from typing import TYPE_CHECKING, Self

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    computed_field,
    model_validator,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from cirq import DensePauliString
    from numpy.typing import NDArray

from quiche.bindings.quest_bindings import PauliStr, PauliStrSum


class Pauli(StrEnum):
    """Non-identity single-qubit Pauli operators."""

    X = "X"
    Y = "Y"
    Z = "Z"

    def to_matrix(self) -> NDArray:
        """Get the matrix representation of the Pauli."""
        match self:
            case Pauli.X:
                return np.array([[0, 1], [1, 0]], dtype=complex)
            case Pauli.Y:
                return np.array([[0, -1j], [1j, 0]], dtype=complex)
            case Pauli.Z:
                return np.array([[1, 0], [0, -1]], dtype=complex)


class PauliWord(BaseModel):
    """Multi-qubit tensor products of Pauli operators."""

    model_config = ConfigDict(frozen=True)

    paulis: tuple[Pauli, ...]
    qubits: tuple[NonNegativeInt, ...]

    @model_validator(mode="after")
    def check_nonzero_lengths(self) -> Self:
        """Validate number of paulis and number of qubits is nonzero."""
        if len(self.paulis) == 0:
            error_msg = "The number of paulis of the PauliWord must be nonzero."
            raise ValueError(error_msg)

        if len(self.qubits) == 0:
            error_msg = "The number of qubits of the PauliWord must be nonzero."
            raise ValueError(error_msg)
        return self

    @model_validator(mode="after")
    def check_lengths_match(self) -> Self:
        """Validate number of paulis and number of qubits."""
        if len(self.paulis) != len(self.qubits):
            error_msg = "The paulis and qubits of the PauliWord must be the same length"
            raise ValueError(error_msg)
        return self

    @model_validator(mode="after")
    def check_qubits_unique(self) -> Self:
        """Validate each target qubit appears at most once."""
        if len(set(self.qubits)) != len(self.qubits):
            error_msg = "The target qubits of the PauliWord must be unique."
            raise ValueError(error_msg)
        return self

    @model_validator(mode="after")
    def _sort(self) -> Self:
        """Sort paulis and qubits by increasing qubit index."""
        pairs = sorted(zip(self.qubits, self.paulis, strict=True))
        qubits, paulis = zip(*pairs, strict=True)
        object.__setattr__(self, "qubits", qubits)
        object.__setattr__(self, "paulis", paulis)
        return self

    @staticmethod
    def _split_dense_string(string: str, *, big_endian: bool) -> list[tuple[str, int]]:
        chars = string.strip()

        if not big_endian:
            chars = chars[::-1]

        return [(char, qubit) for qubit, char in enumerate(chars)]

    @staticmethod
    def _split_sparse_string(string: str) -> list[tuple[str, int]]:
        pairs = []

        for token in string.split():
            match = re.fullmatch(r"([XYZ])\(([0-9]+)\)", token)

            if match is None:
                error_msg = f"Invalid sparse Pauli {token!r}."
                raise ValueError(error_msg)

            char, qubit = match.groups()
            pairs.append((char, int(qubit)))

        return pairs

    @staticmethod
    def _validate_pairs(pairs: list[tuple[str, int]]) -> list[tuple[Pauli, int]]:
        validated = []
        qubits_seen = set()

        for char, qubit in pairs:
            if char not in {"I", "X", "Y", "Z"}:
                error_msg = f"Invalid Pauli {char!r}."
                raise ValueError(error_msg)

            if qubit in qubits_seen:
                error_msg = f"Qubit {qubit} appears more than once."
                raise ValueError(error_msg)

            qubits_seen.add(qubit)

            if char != "I":
                validated.append((Pauli(char), qubit))

        if not validated:
            error_msg = "String contains no non-identity Paulis."
            raise ValueError(error_msg)

        return validated

    @classmethod
    def from_dense_string(cls, dense_string: str, *, big_endian: bool) -> Self:
        """
        Parse a dense string into a PauliWord.

        Each character in the string should correspond to the (possibly identity) Pauli
        acting on the corresponding qubit, e.g.::

            XIZY

        At least one Pauli must be non-identity.

        Parameters
        ----------
        dense_string : str
            The dense string to parse.
        big_endian : bool
            If True, qubit 0 is the leftmost character; if False, it is the rightmost.

        Returns
        -------
        PauliWord
            The parsed PauliWord.

        """
        pairs = cls._split_dense_string(dense_string, big_endian=big_endian)
        validated = cls._validate_pairs(pairs)
        paulis, qubits = zip(*validated, strict=True)
        return cls(paulis=paulis, qubits=qubits)

    @classmethod
    def from_sparse_string(cls, sparse_string: str) -> Self:
        """
        Parse a sparse string into a PauliWord.

        Each Pauli is written as ``P(q)`` for a (non-identity) Pauli ``P`` acting on
        (zero-indexed) qubit ``q``, separated by spaces, e.g.::

            X(0) Z(2)

        Each qubit index may appear at most once. Untargeted qubits are
        implicitly treated as identities and must not be explicitly written.

        Parameters
        ----------
        sparse_string : str
            The sparse string to parse.

        Returns
        -------
        PauliWord
            The parsed PauliWord.

        """
        pairs = cls._split_sparse_string(sparse_string)
        validated = cls._validate_pairs(pairs)
        paulis, qubits = zip(*validated, strict=True)
        return cls(paulis=paulis, qubits=qubits)

    @computed_field
    @cached_property
    def greatest_qubit(self) -> int:
        """Identify highest index qubit included in targets."""
        return max(self.qubits)

    def __str__(self) -> str:
        """Get the string representation of a PauliWord."""
        return self.to_sparse_string()

    def to_sparse_string(self) -> str:
        """Get the sparse string representation of a PauliWord."""
        return " ".join(
            f"{pauli}({qubit})"
            for pauli, qubit in zip(self.paulis, self.qubits, strict=True)
        )

    def _resolve_length(self, length: int | None) -> int:
        if length is None:
            return self.greatest_qubit + 1
        if length <= self.greatest_qubit:
            error_msg = "Length must be greater than the maximum target qubit."
            raise ValueError(error_msg)
        return length

    def to_dense_string(self, length: int | None = None, *, big_endian: bool) -> str:
        """
        Get the dense string representation of a PauliWord.

        Parameters
        ----------
        length : int | None
            Length to pad the dense string to. Must be greater than the maximum target
            qubit.
        big_endian : bool
            If True, qubit 0 is the leftmost character; if False, it is the rightmost.

        Returns
        -------
        str
            The dense string representation of the PauliWord.

        """
        length = self._resolve_length(length)
        ops = ["I"] * length

        for qubit, pauli in zip(self.qubits, self.paulis, strict=True):
            ops[qubit] = str(pauli)

        if not big_endian:
            ops.reverse()

        return "".join(ops)

    def to_cirq(self, length: int) -> DensePauliString:
        """Get dense cirq representation of a PauliWord."""
        # Lazily import cirq only used here
        from cirq import DensePauliString  # noqa: PLC0415

        string = self.to_dense_string(length, big_endian=True)  # validates length
        return DensePauliString(string)

    def to_quest(self, length: int) -> PauliStr:
        """Get the QuEST representation of a PauliWord."""
        string = self.to_dense_string(length, big_endian=False)  # validates length
        return PauliStr(string)

    def to_matrix(
        self, length: int | None = None, *, ignore_idle_qubits: bool
    ) -> NDArray:
        """Get the matrix representation of the PauliWord."""
        length = self._resolve_length(length)
        identity = np.identity(1) if ignore_idle_qubits else np.identity(2)
        matrices = [identity] * length

        for qubit, pauli in zip(self.qubits, self.paulis, strict=True):
            matrices[qubit] = pauli.to_matrix()
        return reduce(np.kron, matrices)


class PauliSum(BaseModel):
    """Linear combinations of Pauli words."""

    model_config = ConfigDict(frozen=True)

    coefficients: tuple[float, ...]
    words: tuple[PauliWord, ...]
    identity_coefficient: float = 0.0

    @model_validator(mode="after")
    def check_nonzero_lengths(self) -> Self:
        """Validate number of words is nonzero."""
        if len(self.words) == 0:
            error_msg = "The number of words of the PauliSum must be nonzero."
            raise ValueError(error_msg)

        return self

    @model_validator(mode="after")
    def check_lengths_match(self) -> Self:
        """Validate number of words and coefficients match."""
        if len(self.coefficients) != len(self.words):
            error_msg = (
                "The coefficients and words of the PauliSum must be the same length"
            )
            raise ValueError(error_msg)

        return self

    @model_validator(mode="after")
    def check_for_zero_coefficients(self) -> Self:
        """Validate linear combination has no zero coefficient words."""
        if any(isclose(c, 0) for c in self.coefficients):
            error_msg = "PauliSum should not contain zero-valued coefficients"
            raise ValueError(error_msg)
        return self

    @staticmethod
    def _split_lines(string: str) -> list[tuple[float, str]]:
        pairs = []

        for line in string.splitlines():
            if not line.strip():
                continue

            coeff_string, *rest = line.split(maxsplit=1)
            rest_string = rest[0].strip() if rest else ""

            try:
                coeff = float(coeff_string)
            except ValueError:
                error_msg = f"Invalid PauliSum line: {line!r}."
                raise ValueError(error_msg) from None

            pairs.append((coeff, rest_string))

        return pairs

    @staticmethod
    def _split_identity(
        lines: list[tuple[float, str]],
        is_identity_predicate: Callable[[str], bool],
    ) -> tuple[float, list[float], list[str]]:
        id_coeff = 0.0
        coeffs = []
        strings = []

        for coeff, string in lines:
            if is_identity_predicate(string):
                id_coeff += coeff
            else:
                coeffs.append(coeff)
                strings.append(string)

        return id_coeff, coeffs, strings

    @staticmethod
    def _validate_length_dense_strings(strings: list[str]) -> None:
        lengths_set = {len(s) for s in strings}
        if len(lengths_set) > 1:
            error_msg = "All non-identity dense strings must have the same length."
            raise ValueError(error_msg)

    @classmethod
    def from_dense_string(cls, dense_string: str, *, big_endian: bool) -> Self:
        """
        Parse a dense string into a PauliSum.

        Expects a multiline string where each line is a coefficient followed by a dense
        PauliWord string (see `PauliWord.from_dense_string`), e.g.::

            +10.0 III
            +5.0 XIZ
            -2.0 ZYI

        A line of only ``I`` represents an identity term. Multiple identity lines are
        summed into the identity coefficient. Blank lines are ignored. All non-identity
        lines must have the same length.

        Parameters
        ----------
        dense_string : str
            The dense string to parse.
        big_endian : bool
            If True, qubit 0 is the leftmost character; if False, it is the rightmost.

        Returns
        -------
        PauliSum
            The parsed PauliSum.

        """
        lines = cls._split_lines(dense_string)
        id_coeff, coeffs, strings = cls._split_identity(
            lines, lambda s: set(s) == {"I"}
        )
        cls._validate_length_dense_strings(strings)
        words = [PauliWord.from_dense_string(s, big_endian=big_endian) for s in strings]
        return cls(coefficients=coeffs, words=words, identity_coefficient=id_coeff)

    @classmethod
    def from_sparse_string(cls, sparse_string: str) -> Self:
        """
        Parse a sparse string into a PauliSum.

        Expects a multiline string where each line is a coefficient followed by a sparse
        PauliWord string (see `PauliWord.from_sparse_string`), e.g.::

            +10.0
            +5.0 X(0) Z(2)
            -2.0 Z(0) Y(1)

        A line containing only a coefficient represents an identity term. Multiple
        identity lines are summed into the identity coefficient. Blank lines are
        ignored.

        Parameters
        ----------
        sparse_string : str
            The sparse string to parse.

        Returns
        -------
        PauliSum
            The parsed PauliSum.

        """
        lines = cls._split_lines(sparse_string)
        id_coeff, coeffs, strings = cls._split_identity(lines, lambda s: not s)
        words = [PauliWord.from_sparse_string(s) for s in strings]
        return cls(coefficients=coeffs, words=words, identity_coefficient=id_coeff)

    @computed_field
    @cached_property
    def num_qubits(self) -> int:
        """Get the number of qubits targeted by all the operators."""
        return max(word.greatest_qubit for word in self.words) + 1

    @computed_field
    @cached_property
    def has_identity(self) -> bool:
        """Whether the identity term contributes to the operator."""
        return bool(self.identity_coefficient)

    @computed_field
    @cached_property
    def num_words(self) -> int:
        """Get number of words in linear combination."""
        return len(self.words)

    @computed_field
    @cached_property
    def num_words_with_identity(self) -> int:
        """Get the number of words including the identity if non-zero."""
        return self.num_words + (1 if self.has_identity else 0)

    @computed_field
    @cached_property
    def lam(self) -> float:
        """Get the 1-norm of the operator."""
        return sum(map(abs, self.coefficients)) + abs(self.identity_coefficient)

    def to_sparse_string(self) -> str:
        """Get the sparse string representation of a PauliSum."""
        lines = [f"{self.identity_coefficient:+}"] + [
            f"{coeff:+} {word.to_sparse_string()}"
            for coeff, word in zip(self.coefficients, self.words, strict=True)
        ]
        return "\n".join(lines)

    def _resolve_length(self, length: int | None) -> int:
        if length is None:
            return self.num_qubits
        if length < self.num_qubits:
            error_msg = "Length must be greater than the maximum target qubit."
            raise ValueError(error_msg)
        return length

    def to_dense_string(self, length: int | None = None, *, big_endian: bool) -> str:
        """
        Get the dense string representation of a PauliSum.

        Parameters
        ----------
        length : int | None
            Length to pad each word to. Must be greater than the maximum target qubit.
        big_endian : bool
            If True, qubit 0 is the leftmost character; if False, it is the rightmost.

        Returns
        -------
        str
            The dense string representation of the PauliSum.

        """
        length = self._resolve_length(length)
        lines = [f"{self.identity_coefficient:+} {'I' * length}"] + [
            f"{coeff:+} {word.to_dense_string(length, big_endian=big_endian)}"
            for coeff, word in zip(self.coefficients, self.words, strict=True)
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        """Get the string representation of a PauliSum."""
        return self.to_sparse_string()

    def without_identity(self) -> PauliSum:
        """Get a copy of the PauliSum with the identity coefficient zeroed."""
        return PauliSum(
            coefficients=self.coefficients,
            words=self.words,
            identity_coefficient=0.0,
        )

    def split_identity(self) -> tuple[float, PauliSum]:
        """Get the identity coefficient and the remaining words separately."""
        return (self.identity_coefficient, self.without_identity())

    def to_quest(self) -> PauliStrSum:
        """
        Get the QuEST representation of a PauliSum.

        Will raise if called outside of a QuESTEnv.
        """
        strings = [word.to_quest(self.num_qubits) for word in self.words]
        coefficients = list(self.coefficients)

        if self.has_identity:
            strings.append(PauliStr("I" * self.num_qubits))
            coefficients.append(self.identity_coefficient)

        return PauliStrSum(strings, coefficients)

    def to_matrix(self) -> NDArray:
        """Get the matrix representation of the PauliSum."""
        id_mat = np.identity(2**self.num_qubits, dtype=complex)
        total = self.identity_coefficient * id_mat
        for word, coeff in zip(self.words, self.coefficients, strict=True):
            mat = word.to_matrix(length=self.num_qubits, ignore_idle_qubits=False)
            total += coeff * mat
        return total
