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

"""Structures to define and dispatch phase estimation calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic.dataclasses import Field, dataclass

from quiche.chemistry import (
    HartreeFockState,
)
from quiche.core import (
    ElectronicHamiltonian,
    Errors,
    PhaseEstimation,
    Simulation,
)
from quiche.dispatch.budget.estimation import (
    get_kitaev_qpe_rounds,
    get_textbook_qpe_ancillas,
)
from quiche.dispatch.budget.simulation import (
    get_qdrift_params,
    get_qubitisation_ancillas,
    get_trotter_params,
)

if TYPE_CHECKING:
    from quiche.dispatch.backends import QPESpecBackend


# TODO(Vasco): refactor the algorithm -> parameter calculation out of here
@dataclass()
class QPESpec:
    """QPE calculation specification for simulation and resource estimates."""

    hamiltonian: ElectronicHamiltonian
    state_prep: HartreeFockState | None
    algorithm: PhaseEstimation
    simulation: Simulation
    error_budget: Errors
    seed: int | None = None
    extras: dict = Field(default_factory=dict)

    def __post_init__(self) -> None:
        """Calculate the circuit properties for the given QPE routine."""
        if isinstance(self.state_prep, HartreeFockState) and (
            (self.state_prep.num_spin_orbitals != self.hamiltonian.paulis.num_qubits)
            or (self.state_prep.num_electrons != self.hamiltonian.num_electrons)
        ):
            error_msg = (
                "Provided HartreeFockState is inconsistent with ElectronicHamiltonian."
            )
            raise ValueError(error_msg)

        self.num_data = self.hamiltonian.paulis.num_qubits
        self.num_phase_ancillas: int = 0
        self.num_qpe_ancillas: int = 0
        self.num_index_ancillas: int = 0
        self.num_rounds: int | None = None
        self.time: float | None = None
        self.order: int | None = None
        self.reps: int | None = None

        match self.algorithm:
            case PhaseEstimation.Textbook:
                self.num_qpe_ancillas = get_textbook_qpe_ancillas(self.error_budget)

            case (
                PhaseEstimation.Iterative
                | PhaseEstimation.Kitaev
                | PhaseEstimation.Naive
            ):
                self.num_qpe_ancillas = 1
                self.num_rounds = get_kitaev_qpe_rounds(self.error_budget)

        match self.simulation:
            case Simulation.QDRIFT:
                if self.seed is None:
                    error_msg = "QDRIFT requires a seed."
                    raise ValueError(error_msg)

                self.time, self.reps = get_qdrift_params(
                    self.hamiltonian.paulis,
                    self.error_budget,
                )

            case Simulation.Qubitised:
                self.num_index_ancillas, self.num_phase_ancillas = (
                    get_qubitisation_ancillas(
                        self.hamiltonian.paulis, self.error_budget
                    )
                )

            case Simulation.Trotter:
                self.time, self.order, self.reps = get_trotter_params(
                    self.hamiltonian.paulis,
                    self.error_budget,
                )

        self.num_simulation_ancillas = self.num_index_ancillas + self.num_phase_ancillas
        self.num_qubits = (
            self.num_data + self.num_qpe_ancillas + self.num_simulation_ancillas
        )

    def generate[T](self, backend: QPESpecBackend[T]) -> T:
        """Generate the implementation of this QPESpec for the given backend."""
        return backend.generate(self)
