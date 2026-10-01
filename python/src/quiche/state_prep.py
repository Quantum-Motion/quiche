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

"""State-preparation methods implementing the common `Spec` interface."""

from functools import partial
from typing import Self

import numpy as np
from numpy.typing import NDArray
from pydantic.dataclasses import dataclass
from qualtran import Bloq

from quiche.bindings.quiche_bindings import initClassicalState
from quiche.chemistry import (
    get_bk_state,
    get_hf_state,
    get_jw_state,
    get_parity_state,
)
from quiche.core import Mapping
from quiche.cudaq import CudaqKernel, bitstring_kernel
from quiche.dispatch.spec import Spec
from quiche.qualtran.bloqs import BitstringStatePrep
from quiche.quest import QuestRoutine


@dataclass(frozen=True)
class HartreeFock(Spec):
    """
    State preparation for a Hartree-Fock reference state.

    `occupation` gives the occupation (0 or 1) of each spin orbital, which `mapping`
    transforms to the qubit basis. `closed_shell` builds the occupation from electron
    and spin orbital counts.

    `qubits` optionally keeps only those qubits of the mapped state, in order. Pass the
    indices returned by `PauliSum.compact()` so the state lines up with the compacted
    Hamiltonian. Selecting after the mapping is correct for every mapping, because the
    mapped state is a computational basis state and the dropped qubits are idle.
    """

    occupation: tuple[int, ...]
    mapping: Mapping
    qubits: tuple[int, ...] | None = None

    @classmethod
    def closed_shell(
        cls,
        electrons: int,
        spin_orbitals: int,
        *,
        mapping: Mapping,
        qubits: tuple[int, ...] | None = None,
    ) -> Self:
        """Build the Hartree-Fock state of a closed-shell system."""
        if electrons % 2 != 0:
            err_msg = "Closed shell system must have even number of electrons."
            raise ValueError(err_msg)
        occupation = tuple(int(i) for i in get_hf_state(spin_orbitals, electrons))
        return cls(occupation=occupation, mapping=mapping, qubits=qubits)

    def __post_init__(self) -> None:
        """Validate the occupation and the kept qubits."""
        if not all(i in {0, 1} for i in self.occupation):
            err_msg = "Spin orbital occupation must contain binary entries."
            raise ValueError(err_msg)

        if self.qubits is not None and not all(
            0 <= i < self.num_spin_orbitals for i in self.qubits
        ):
            msg = (
                f"Kept qubits {self.qubits} must lie in [0, {self.num_spin_orbitals})."
            )
            raise ValueError(msg)

    @property
    def num_electrons(self) -> int:
        """Get the number of electrons."""
        return sum(self.occupation)

    @property
    def num_spin_orbitals(self) -> int:
        """Get the number of spin orbitals."""
        return len(self.occupation)

    @property
    def num_qubits(self) -> int:
        """Get the number of qubits the state is prepared on."""
        return len(self._bitstring())

    def to_qualtran(self) -> Bloq:
        """Build the Qualtran Bloq preparing the Hartree-Fock state."""
        return BitstringStatePrep(tuple(self._bitstring()))

    def to_quest(self) -> QuestRoutine:
        """Build the QuEST routine preparing the Hartree-Fock state."""
        routine = QuestRoutine()
        routine.append(partial(initClassicalState, state=self._bitstring()))
        return routine

    def to_cudaq(self) -> CudaqKernel:
        """Build the CUDA-Q kernel preparing the Hartree-Fock state."""
        return bitstring_kernel(self._bitstring())

    def _bitstring(self) -> NDArray[np.int_]:
        """Transform the occupation basis state to the qubit basis via `mapping`."""
        match self.mapping:
            case Mapping.JordanWigner:
                bitstring = get_jw_state(self.occupation)
            case Mapping.BravyiKitaev:
                bitstring = get_bk_state(self.occupation)
            case Mapping.Parity:
                bitstring = get_parity_state(self.occupation)
        if self.qubits is None:
            return bitstring
        return np.asarray(bitstring)[list(self.qubits)]
