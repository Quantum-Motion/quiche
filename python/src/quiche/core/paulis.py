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

from enum import StrEnum
from functools import cached_property
from math import isclose
from typing import Self

import numpy as np
from cirq import DensePauliString
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, computed_field, model_validator

from quiche.bindings.quest_bindings import PauliStr, PauliStrSum


class Pauli(StrEnum):
    """Single-qubit Pauli operators."""

    X = "X"
    Y = "Y"
    Z = "Z"

    def _to_matrix(self) -> NDArray:
        if self is Pauli.X:
            return np.array([[0, 1], [1, 0]])
        if self is Pauli.Y:
            return np.array([[0, -1j], [1j, 0]])
        if self is Pauli.Z:
            return np.array([[1, 0], [0, -1]])
        return None


class PauliWord(BaseModel):
    """Multi-qubit tensor products of Pauli operators."""

    model_config = ConfigDict(frozen=True)

    terms: tuple[Pauli, ...]
    qubits: tuple[int, ...]

    @model_validator(mode="after")
    def check_nonzero_lengths(self) -> Self:
        """Validate number of terms and number of qubits is nonzero."""
        if len(self.terms) == 0:
            error_msg = "The number of terms of the PauliWord must be nonzero."
            raise ValueError(error_msg)

        if len(self.qubits) == 0:
            error_msg = "The number of qubits of the PauliWord must be nonzero."
            raise ValueError(error_msg)
        return self

    @model_validator(mode="after")
    def check_lengths_match(self) -> Self:
        """Validate number of terms and number of qubits."""
        if len(self.terms) != len(self.qubits):
            error_msg = "The terms and qubits of the PauliWord must be the same length"
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

        for i, pauli in zip(self.qubits, self.terms, strict=True):
            ops[i] = str(pauli)

        if not big_endian:
            ops.reverse()

        return "".join(ops)

    def to_cirq(self, length: int) -> DensePauliString:
        """Get dense cirq representation of a PauliWord."""
        string = self.to_str(length, big_endian=True)  # validates length
        return DensePauliString(string)

    def to_quest(self, length: int) -> PauliStr:
        """Get the QuEST representation of a PauliWord."""
        string = self.to_str(length, big_endian=False)  # validates length
        return PauliStr(string)

    def _to_matrix(
        self, length: int | None = None, *, ignore_idle_qubits: bool
    ) -> NDArray:
        """Transform PauliWord to matrix."""
        identity = 1 if ignore_idle_qubits else np.identity(2)

        if length is None:
            length = self.greatest_qubit + 1
        if length <= self.greatest_qubit:
            error_msg = "String length must be greater than the maximum target qubit."
            raise ValueError(error_msg)

        result = 1.0
        for ii in range(length):
            if ii in self.qubits:
                idx = self.qubits.index(ii)
                curr = self.terms[idx]._to_matrix()  # noqa: SLF001
            else:
                curr = identity
            result = np.kron(result, curr)
        return result


class PauliSum(BaseModel):
    """Linear combinations of multi-qubit Pauli operators."""

    model_config = ConfigDict(frozen=True)

    coefficients: tuple[float, ...]
    terms: tuple[PauliWord, ...]
    identity_coefficient: float

    @model_validator(mode="after")
    def check_lengths_match(self) -> Self:
        """Validate number of terms and coefficients match."""
        if len(self.coefficients) != len(self.terms):
            error_msg = (
                "The coefficients and terms of the PauliSum must be the same length"
            )
            raise ValueError(error_msg)

        return self

    @model_validator(mode="after")
    def check_for_zero_coefficients(self) -> Self:
        """Validate linear combination has no zero coefficient terms."""
        if any(isclose(c, 0) for c in self.coefficients):
            error_msg = "PauliSum should not contain zero-valued coefficients"
            raise ValueError(error_msg)
        return self

    @computed_field
    @cached_property
    def n_qubits(self) -> int:
        """Get the number of qubits targeted by all the operators."""
        return max(term.greatest_qubit for term in self.terms) + 1

    @computed_field
    @cached_property
    def n_terms(self) -> int:
        """Get number of terms in linear combination."""
        return len(self.terms)

    @computed_field
    @cached_property
    def lam(self) -> float:
        """Get the 1-norm of the coefficients."""
        return sum(map(abs, self.coefficients))

    def __str__(self) -> str:
        """Define printing for PauliSum class."""
        msg = str(self.identity_coefficient) + " * I\n"
        for ii in range(self.n_terms):
            msg += f"+ {self.coefficients[ii]:f} * "
            for op, qubit in zip(
                self.terms[ii].terms, self.terms[ii].qubits, strict=True
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

    def to_quest(self) -> PauliStrSum:
        """
        Get the QuEST representation of a PauliSum.

        Will raise if called outside of a QuESTEnv.
        """
        strings = [word.to_quest(self.n_qubits) for word in self.terms]
        return PauliStrSum(strings, self.coefficients)

    def _to_matrix(self) -> NDArray:
        total = self.identity_coefficient * np.identity(2**self.n_qubits, dtype=complex)
        for word, coeff in zip(self.terms, self.coefficients, strict=True):
            total += coeff * word._to_matrix(  # noqa: SLF001
                length=self.n_qubits, ignore_idle_qubits=False
            )
        return total
