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

from enum import StrEnum
from functools import cached_property, reduce
from math import isclose
from typing import TYPE_CHECKING, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, computed_field, model_validator

if TYPE_CHECKING:
    from cirq import DensePauliString
    from numpy.typing import NDArray

from quiche.bindings.quest_bindings import PauliStr, PauliStrSum


class Pauli(StrEnum):
    """Single-qubit Pauli operators."""

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
    qubits: tuple[int, ...]

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

    @computed_field
    @cached_property
    def greatest_qubit(self) -> int:
        """Identify highest index qubit included in targets."""
        return max(self.qubits)

    def __str__(self) -> str:
        """Get the string representation of a PauliWord, in big endian ordering."""
        length = self.greatest_qubit + 1
        return self.to_str(length, big_endian=True)

    def _resolve_length(self, length: int | None) -> int:
        if length is None:
            return self.greatest_qubit + 1
        if length <= self.greatest_qubit:
            error_msg = "Length must be greater than the maximum target qubit."
            raise ValueError(error_msg)
        return length

    def to_str(self, length: int | None = None, *, big_endian: bool) -> str:
        """
        Get the string representation of a PauliWord for a given register size.

        Endianness must be explicitly set with the ``big_endian`` arg::

            Big endian:    |psi> = |q_0, q_1, q_2, ..., q_n>
            Little endian: |psi> = |q_n, ..., q_2, q_1, q_0>
        """
        if length is None:
            length = self.greatest_qubit + 1
        elif length <= self.greatest_qubit:
            error_msg = "String length must be greater than the maximum target qubit."
            raise ValueError(error_msg)

        ops = ["I"] * length

        for i, pauli in zip(self.qubits, self.paulis, strict=True):
            ops[i] = str(pauli)

        if not big_endian:
            ops.reverse()

        return "".join(ops)

    def to_cirq(self, length: int) -> DensePauliString:
        """Get dense cirq representation of a PauliWord."""
        # Lazily import cirq only used here
        from cirq import DensePauliString  # noqa: PLC0415

        string = self.to_str(length, big_endian=True)  # validates length
        return DensePauliString(string)

    def to_quest(self, length: int) -> PauliStr:
        """Get the QuEST representation of a PauliWord."""
        string = self.to_str(length, big_endian=False)  # validates length
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
    """Linear combinations of multi-qubit Pauli operators."""

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
        """Get the number of terms including the identity if non-zero."""
        return self.num_words + (1 if self.has_identity else 0)

    @computed_field
    @cached_property
    def lam(self) -> float:
        """Get the 1-norm of the operator."""
        return sum(map(abs, self.coefficients)) + abs(self.identity_coefficient)

    def __str__(self) -> str:
        """Define printing for PauliSum class."""
        msg = str(self.identity_coefficient) + " * I\n"
        for ii in range(self.num_words):
            msg += f"+ {self.coefficients[ii]:f} * "
            for op, qubit in zip(
                self.words[ii].paulis, self.words[ii].qubits, strict=True
            ):
                if op is Pauli.X:
                    msg += "X"
                if op is Pauli.Y:
                    msg += "Y"
                if op is Pauli.Z:
                    msg += "Z"
                msg += f"({qubit})"
            msg += "\n"
        return msg

    def without_identity(self) -> PauliSum:
        """Get a copy of the PauliSum with the identity coefficient zeroed."""
        return PauliSum(
            coefficients=self.coefficients,
            words=self.words,
            identity_coefficient=0.0,
        )

    def split_identity(self) -> tuple[float, PauliSum]:
        """Get the identity coefficient and the remaining terms separately."""
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
