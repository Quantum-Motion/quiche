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

"""State preparation algorithms."""

from typing import Self

from pydantic import PositiveInt, validate_call
from pydantic.dataclasses import dataclass

from quiche.chemistry import BinaryTuple, Mapping, get_hf_state, get_qubit_state

type StateprepAlgorithm = IdentityStateprep | BitstringStateprep


@dataclass(frozen=True)
class IdentityStateprep:
    """
    Identity (zero-state) state preparation.

    Attributes
    ----------
    num_qubits : int
        Number of qubits to initialise. Must be positive.
    """

    num_qubits: PositiveInt


@dataclass(frozen=True)
class BitstringStateprep:
    """
    Arbitrary computational basis state preparation.

    Attributes
    ----------
    bitstring : tuple[{0, 1}, ...]
        Bitstring specifying the basis state to prepare.
    """

    bitstring: BinaryTuple

    @classmethod
    @validate_call
    def occupation(cls, occupation: BinaryTuple, mapping: Mapping) -> Self:
        """Initialise from a given fermionic occupation state."""
        qubit_state = get_qubit_state(occupation, mapping)
        return cls(qubit_state)

    @classmethod
    def hartree_fock(
        cls,
        num_electrons: int,
        num_spin_orbitals: int,
        mapping: Mapping,
    ) -> Self:
        """Initialise the Hartree-Fock state for a closed-shell system."""
        if num_electrons % 2 != 0:
            error_msg = "Closed shell system must have even number of electrons."
            raise ValueError(error_msg)

        # get_hf_state validates arguments
        occupation = get_hf_state(num_electrons, num_spin_orbitals)
        return cls.occupation(occupation, mapping)
