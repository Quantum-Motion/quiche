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

from collections.abc import Callable
from functools import partial

from pydantic.dataclasses import Field, dataclass
from qualtran import Bloq
from qualtran.bloqs.qubitization.qubitization_walk_operator import (
    QubitizationWalkOperator,
)

from quiche.core import (
    Errors,
    PauliSum,
    PhaseEstimation,
    Simulation,
)
from quiche.cudaq import CudaqKernel, qubitised_qpe_kernel, textbook_qpe_kernel
from quiche.dispatch.budget.estimation import (
    get_kitaev_qpe_rounds,
    get_textbook_qpe_ancillas,
)
from quiche.dispatch.budget.simulation import (
    get_qdrift_params,
    get_qubitisation_ancillas,
    get_trotter_params,
)
from quiche.dispatch.spec import Spec
from quiche.qualtran.bloqs import (
    QDRIFT,
    LCUBlockEncodingWrapper,
    Trotterisation,
)
from quiche.qualtran.bloqs.estimation import (
    QubitisationLadder,
    TextbookQPE,
    TrotterLadder,
)
from quiche.quest import QuestRoutine
from quiche.quest.estimation import (
    getPhaseKitaevQDRIFT,
    getPhaseKitaevTrotter,
    getPhaseTextbookQDRIFT,
    getPhaseTextbookQubitised,
    getPhaseTextbookTrotter,
)


@dataclass()
class QPESpec(Spec):
    """Specification for the QPE algorithm circuit, not including state preparation."""

    hamiltonian: PauliSum
    n_qubits: int
    algorithm: PhaseEstimation
    simulation: Simulation
    error_budget: Errors
    extras: dict = Field(default_factory=dict)

    def __post_init__(self) -> None:
        """Calculate the circuit properties for the given QPE routine."""
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

        self.num_index_ancillas = 0
        self.num_phase_ancillas = 0

        match self.simulation:
            case Simulation.QDRIFT:
                self.time, self.reps = get_qdrift_params(
                    self.hamiltonian,
                    self.error_budget,
                )

            case Simulation.Qubitised:
                self.num_index_ancillas, self.num_phase_ancillas = (
                    get_qubitisation_ancillas(self.hamiltonian, self.error_budget)
                )

            case Simulation.Trotter:
                self.time, self.order, self.reps = get_trotter_params(
                    self.hamiltonian,
                    self.error_budget,
                )

        # Returns None if missing
        self.seed = self.extras.get("seed")

        self.num_data = self.n_qubits
        self.num_simulation_ancillas = self.num_index_ancillas + self.num_phase_ancillas
        self.num_qubits = (
            self.num_data + self.num_qpe_ancillas + self.num_simulation_ancillas
        )

    def _get_simulation_qualtran_factory(self) -> Callable[[int], Bloq]:
        """Pattern matching and construction for simulation factories."""
        match self.simulation:
            case Simulation.QDRIFT:
                bloq = QDRIFT(
                    h=self.hamiltonian,
                    t=self.time,
                    n_terms=self.reps,
                    seed=self.seed,
                )

                simulation_bloq_factory = partial(
                    TrotterLadder,
                    simulation=bloq,
                    num_data=self.num_data,
                    num_qpe_ancillas=self.num_qpe_ancillas,
                )

            case Simulation.Qubitised:
                blockencoding = LCUBlockEncodingWrapper.from_hamiltonian(
                    self.hamiltonian, self.num_phase_ancillas
                )
                walk_op = QubitizationWalkOperator(blockencoding)

                simulation_bloq_factory = partial(
                    QubitisationLadder,
                    walk=walk_op,
                    num_data=self.num_data,
                    num_qpe_ancillas=self.num_qpe_ancillas,
                    num_selection_ancillas=self.num_simulation_ancillas,
                )

            case Simulation.Trotter:
                bloq = Trotterisation(
                    h=self.hamiltonian,
                    t=self.time,
                    n_steps=self.reps,
                    order=self.order,
                )

                simulation_bloq_factory = partial(
                    TrotterLadder,
                    simulation=bloq,
                    num_data=self.num_data,
                    num_qpe_ancillas=self.num_qpe_ancillas,
                )

        return simulation_bloq_factory

    def _get_estimation_qualtran(self) -> Bloq:
        """Pattern matching and construction for QPE Bloq."""
        match self.algorithm:
            case PhaseEstimation.Iterative:
                msg = "Iterative QPE Bloq not yet implemented."
                raise NotImplementedError(msg)

            case PhaseEstimation.Kitaev:
                msg = "Kitaev QPE Bloq not yet implemented."
                raise NotImplementedError(msg)

            case PhaseEstimation.Naive:
                msg = "Naive QPE Bloq not yet implemented."
                raise NotImplementedError(msg)

            case PhaseEstimation.Textbook:
                estimation_bloq = TextbookQPE(
                    simulation_factory=self._get_simulation_qualtran_factory(),
                    num_data=self.num_data,
                    num_qpe_ancillas=self.num_qpe_ancillas,
                    num_other_ancillas=self.num_simulation_ancillas,
                )

        return estimation_bloq

    def to_qualtran(self) -> Bloq:
        """
        Build the Qualtran Bloq implementing the QPE algorithm.

        The returned Bloq expects an externally-prepared state on its `data` register;
        compose it with a state-preparation Bloq to get a runnable circuit.
        """
        return self._get_estimation_qualtran()

    def _get_estimation_quest(self) -> Callable:
        # Data register: [0, num_data]
        # Ancilla register: [num_data, num_data + num_qpe_ancillas]
        qpe_ancillas = list(range(self.num_data, self.num_data + self.num_qpe_ancillas))

        # Simulation currently doesn't use ancillas for rotations
        # so only include index ancillas
        index_ancillas = list(
            range(
                self.num_data + self.num_qpe_ancillas,
                self.num_data + self.num_qpe_ancillas + self.num_index_ancillas,
            )
        )

        match self.algorithm:
            case PhaseEstimation.Iterative:
                msg = "Iterative QPE not yet implemented in simulation backend."
                raise NotImplementedError(msg)

            case PhaseEstimation.Kitaev:
                match self.simulation:
                    case Simulation.QDRIFT:
                        sim = partial(
                            getPhaseKitaevQDRIFT,
                            hamiltonian=self.hamiltonian,
                            ancilla_index=qpe_ancillas[0],
                            reps=self.reps,
                            time=self.time,
                            num_bits=self.num_rounds,
                            seed=self.seed,
                        )

                    case Simulation.Qubitised:
                        msg = "Qubitisation not yet implemented in simulation backend."
                        raise NotImplementedError(msg)

                    case Simulation.Trotter:
                        sim = partial(
                            getPhaseKitaevTrotter,
                            hamiltonian=self.hamiltonian,
                            ancilla_index=qpe_ancillas[0],
                            order=self.order,
                            reps=self.reps,
                            time=self.time,
                            num_bits=self.num_rounds,
                        )

            case PhaseEstimation.Naive:
                msg = "Naive QPE not yet implemented in simulation backend."
                raise NotImplementedError(msg)

            case PhaseEstimation.Textbook:
                match self.simulation:
                    case Simulation.QDRIFT:
                        sim = partial(
                            getPhaseTextbookQDRIFT,
                            hamiltonian=self.hamiltonian,
                            ancillas=qpe_ancillas,
                            reps=self.reps,
                            time=self.time,
                            seed=self.seed,
                        )

                    case Simulation.Qubitised:
                        sim = partial(
                            getPhaseTextbookQubitised,
                            hamiltonian=self.hamiltonian,
                            qpe_ancillas=qpe_ancillas,
                            qubitisation_ancillas=index_ancillas,
                        )

                    case Simulation.Trotter:
                        sim = partial(
                            getPhaseTextbookTrotter,
                            hamiltonian=self.hamiltonian,
                            ancillas=qpe_ancillas,
                            order=self.order,
                            reps=self.reps,
                            time=self.time,
                        )

        return sim

    def to_quest(self) -> QuestRoutine:
        """
        Generate the QuEST routine implementing the QPE algorithm.

        The returned routine expects the Qureg to already be in an externally-prepared
        initial state; compose it (e.g. via `QuestRoutine.extend`) with a state
        preparation routine to get a runnable simulation.
        """
        routine = QuestRoutine()
        routine.append(self._get_estimation_quest())
        return routine

    def to_cudaq(self) -> CudaqKernel:
        """
        Build the CUDA-Q kernel implementing the QPE algorithm.

        The returned kernel has signature `(data: cudaq.qview) -> float`: it
        operates in place on an already-allocated data register (the same
        contract as `HartreeFockSpec.to_cudaq()`'s kernel) and returns one
        shot's own energy estimate, decoded from the measured ancillas as
        classical kernel-mode arithmetic - no separate post-processing step.
        Compose the two by allocating the register once and calling both
        kernels on it inside one top-level kernel, e.g.:

        ```python
        prep = state_spec.to_cudaq()
        qpe = qpe_spec.to_cudaq()

        @cudaq.kernel
        def run() -> float:
            data = cudaq.qvector(qpe_spec.num_data)
            prep(data)
            return qpe(data)

        energy = run()                               # a single shot
        energies = cudaq.run(run, shots_count=100)    # or many, as a list[float]
        ```

        See `quiche.cudaq.estimation.textbook_qpe_kernel`/`qubitised_qpe_kernel`
        for the exact contract and why the energy recovery lives in-kernel here
        (unlike `to_qualtran`/`to_quest`).

        `PhaseEstimation.Textbook` is implemented for all three `Simulation`
        variants; the other `PhaseEstimation` algorithms still raise
        `NotImplementedError`.
        """
        if self.algorithm is not PhaseEstimation.Textbook:
            msg = f"{self.algorithm} is not yet implemented in the CUDA-Q backend."
            raise NotImplementedError(msg)

        match self.simulation:
            case Simulation.Trotter:
                return textbook_qpe_kernel(
                    self.hamiltonian,
                    self.simulation,
                    self.num_qpe_ancillas,
                    self.time,
                    self.reps,
                    order=self.order,
                    seed=self.seed,
                    n_qubits=self.n_qubits,
                )

            case Simulation.QDRIFT:
                return textbook_qpe_kernel(
                    self.hamiltonian,
                    self.simulation,
                    self.num_qpe_ancillas,
                    self.time,
                    self.reps,
                    seed=self.seed,
                    n_qubits=self.n_qubits,
                )

            case Simulation.Qubitised:
                return qubitised_qpe_kernel(
                    self.hamiltonian,
                    self.num_qpe_ancillas,
                    n_qubits=self.n_qubits,
                )
