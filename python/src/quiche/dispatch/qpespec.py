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

# TODO(Vasco): refactor
# ruff: noqa: C901

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Literal

from pydantic.dataclasses import Field, dataclass
from qualtran import Bloq, BloqBuilder, CompositeBloq
from qualtran.bloqs.qubitization.qubitization_walk_operator import (
    QubitizationWalkOperator,
)

from quiche.bindings.quiche_bindings import initClassicalState
from quiche.chemistry import (
    HartreeFockState,
    get_bk_state,
    get_jw_state,
    get_parity_state,
)
from quiche.core import (
    ElectronicHamiltonian,
    Errors,
    Mapping,
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
from quiche.resources.bloqs import (
    QDRIFT,
    BitstringStatePrep,
    IdentityStatePrep,
    LCUBlockEncodingWrapper,
    Trotterisation,
)
from quiche.resources.bloqs.estimation import (
    QubitisationLadder,
    TextbookQPE,
    TrotterLadder,
)
from quiche.simulation import SimulationRoutine
from quiche.simulation.estimation import (
    getPhaseKitaevQDRIFT,
    getPhaseKitaevTrotter,
    getPhaseTextbookQDRIFT,
    getPhaseTextbookQubitised,
    getPhaseTextbookTrotter,
)

if TYPE_CHECKING:
    from quiche.cudaq import CudaqKernel


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
        if self.state_prep is not None and not isinstance(
            self.state_prep, HartreeFockState
        ):
            error_msg = f"Invalid state preparation {self.state_prep}"
            raise ValueError(error_msg)

        if isinstance(self.state_prep, HartreeFockState) and (
            (self.state_prep.num_spin_orbitals != self.hamiltonian.paulis.num_qubits)
            or (self.state_prep.num_electrons != self.hamiltonian.num_electrons)
        ):
            error_msg = (
                "Provided HartreeFockState is inconsistent with ElectronicHamiltonian."
            )
            raise ValueError(error_msg)

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

        self.num_data = self.hamiltonian.paulis.num_qubits
        self.num_simulation_ancillas = self.num_index_ancillas + self.num_phase_ancillas
        self.num_qubits = (
            self.num_data + self.num_qpe_ancillas + self.num_simulation_ancillas
        )

    def _get_stateprep_bloq(self) -> Bloq:
        """Pattern matching and construction for state preparation Bloq."""
        match self.state_prep:
            case None:
                state_prep_bloq = IdentityStatePrep(self.num_data)
            case HartreeFockState():
                mapping = self.hamiltonian.mapping
                occupation = self.state_prep.occupation
                match mapping:
                    case Mapping.JordanWigner:
                        bitstring = get_jw_state(occupation)
                    case Mapping.BravyiKitaev:
                        bitstring = get_bk_state(occupation)
                    case Mapping.Parity:
                        bitstring = get_parity_state(occupation)
                state_prep_bloq = BitstringStatePrep(tuple(bitstring))

        return state_prep_bloq

    def _get_simulation_bloq_factory(self) -> Callable[[int], Bloq]:
        """Pattern matching and construction for simulation factories."""
        match self.simulation:
            case Simulation.QDRIFT:
                bloq = QDRIFT(
                    sum=self.hamiltonian.paulis,
                    time=self.time,
                    num_samples=self.reps,
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
                    self.hamiltonian.paulis, self.num_phase_ancillas
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
                    sum=self.hamiltonian.paulis,
                    time=self.time,
                    num_steps=self.reps,
                    order=self.order,
                )

                simulation_bloq_factory = partial(
                    TrotterLadder,
                    simulation=bloq,
                    num_data=self.num_data,
                    num_qpe_ancillas=self.num_qpe_ancillas,
                )

        return simulation_bloq_factory

    def _get_estimation_bloq(self) -> Bloq:
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
                    simulation_factory=self._get_simulation_bloq_factory(),
                    num_data=self.num_data,
                    num_qpe_ancillas=self.num_qpe_ancillas,
                    num_other_ancillas=self.num_simulation_ancillas,
                )

        return estimation_bloq

    def get_composite_bloq(self) -> CompositeBloq:
        """Estimate resources for the calculation using the qualtran backend."""
        bb = BloqBuilder()
        data = bb.add(self._get_stateprep_bloq())
        data = bb.add(self._get_estimation_bloq(), data=data)
        bb.free(data)
        return bb.finalize()

    def _get_stateprep_quest(self) -> Callable:
        match self.state_prep:
            case None:

                def noop(_: object) -> None:
                    pass

                init = noop

            case HartreeFockState():
                occupation = self.state_prep.occupation
                mapping = self.hamiltonian.mapping

                match mapping:
                    case Mapping.JordanWigner:
                        bitstring = get_jw_state(occupation)
                    case Mapping.BravyiKitaev:
                        bitstring = get_bk_state(occupation)
                    case Mapping.Parity:
                        bitstring = get_parity_state(occupation)

                init = partial(initClassicalState, state=bitstring)
        return init

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
                            hamiltonian=self.hamiltonian.paulis,
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
                            hamiltonian=self.hamiltonian.paulis,
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
                            hamiltonian=self.hamiltonian.paulis,
                            ancillas=qpe_ancillas,
                            reps=self.reps,
                            time=self.time,
                            seed=self.seed,
                        )

                    case Simulation.Qubitised:
                        sim = partial(
                            getPhaseTextbookQubitised,
                            hamiltonian=self.hamiltonian.paulis,
                            qpe_ancillas=qpe_ancillas,
                            qubitisation_ancillas=index_ancillas,
                        )

                    case Simulation.Trotter:
                        sim = partial(
                            getPhaseTextbookTrotter,
                            hamiltonian=self.hamiltonian.paulis,
                            ancillas=qpe_ancillas,
                            order=self.order,
                            reps=self.reps,
                            time=self.time,
                        )

        return sim

    def to_quest(self) -> SimulationRoutine:
        """Generate a QuEST simulation implementing the specified calculation."""
        routine = SimulationRoutine()
        routine.append(self._get_stateprep_quest())
        routine.append(self._get_estimation_quest())
        return routine

    def _get_stateprep_bitstring(self) -> tuple[int, ...]:
        """Get the mapped computational basis state prepared on the data register."""
        match self.state_prep:
            case None:
                return (0,) * self.num_data
            case HartreeFockState():
                occupation = self.state_prep.occupation
                match self.hamiltonian.mapping:
                    case Mapping.JordanWigner:
                        bitstring = get_jw_state(occupation)
                    case Mapping.BravyiKitaev:
                        bitstring = get_bk_state(occupation)
                    case Mapping.Parity:
                        bitstring = get_parity_state(occupation)
                return tuple(int(bit) for bit in bitstring)

    def _get_estimation_cudaq(self) -> "tuple[CudaqKernel, int]":
        """Build the `(data, work) -> float` QPE core kernel and its work size."""
        from quiche.cudaq import (  # noqa: PLC0415
            iterative_qpe_core,
            qubitised_qpe_core,
            textbook_qpe_core,
        )

        hamiltonian = self.hamiltonian.paulis
        # Only Trotterisation has an order; QDRIFT is first order by construction.
        order = self.order if self.simulation == Simulation.Trotter else 1

        match (self.algorithm, self.simulation):
            case (PhaseEstimation.Textbook, Simulation.Qubitised):
                return qubitised_qpe_core(
                    hamiltonian, self.num_qpe_ancillas, n_qubits=self.num_data
                )

            case (PhaseEstimation.Textbook, _):
                return textbook_qpe_core(
                    hamiltonian,
                    self.simulation,
                    self.num_qpe_ancillas,
                    self.time,
                    self.reps,
                    order,
                    seed=self.seed,
                    n_qubits=self.num_data,
                )

            case (PhaseEstimation.Iterative, Simulation.Qubitised):
                msg = (
                    "Qubitised simulation is not yet implemented for Iterative QPE "
                    "in the CUDA-Q backend."
                )
                raise NotImplementedError(msg)

            case (PhaseEstimation.Iterative, _):
                return iterative_qpe_core(
                    hamiltonian,
                    self.simulation,
                    self.num_rounds,
                    self.time,
                    self.reps,
                    order,
                    seed=self.seed,
                    n_qubits=self.num_data,
                )

            case (PhaseEstimation.Kitaev, _):
                msg = (
                    "Kitaev QPE needs many-shot statistics at several powers combined "
                    "on the host, so it has no single CUDA-Q kernel; build it from "
                    "quiche.cudaq.naive_qpe_kernel instead."
                )
                raise NotImplementedError(msg)

            case (PhaseEstimation.Naive, _):
                msg = (
                    "Naive QPE returns single-shot +-1 outcomes, not energies; use "
                    "quiche.cudaq.naive_qpe_kernel directly."
                )
                raise NotImplementedError(msg)

    def to_cudaq(
        self, kernel: Literal["all", "circuit", "post"] = "all"
    ) -> "CudaqKernel | Callable":
        """
        Generate a CUDA-Q kernel implementing the specified calculation.

        Requires the optional `cudaq` and `cudaq_algorithms` packages.

        - `kernel="all"` (default): a `() -> float` kernel that prepares the state,
          runs one QPE and returns the decoded energy (the phase is converted
          in-kernel using the simulation time, or the block-encoding normalisation
          for qubitisation).
        - `kernel="circuit"`: the measured Textbook QPE circuit, including state
          preparation, to sample with `cudaq.sample`.
        - `kernel="post"`: the host-side decoder for the `cudaq.sample` result of the
          `"circuit"` kernel, returning one energy per shot.
        """
        from quiche.cudaq import (  # noqa: PLC0415
            bitstring_kernel,
            measured_circuit_kernel,
            qubitised_qpe_circuit,
            repeated_minimum_kernel,
            sample_post_processor,
            textbook_qpe_circuit,
        )

        if kernel not in {"all", "circuit", "post"}:
            msg = f"kernel must be 'all', 'circuit' or 'post', got {kernel!r}."
            raise ValueError(msg)

        if kernel == "all":
            core, work_qubits = self._get_estimation_cudaq()
            state_prep = bitstring_kernel(self._get_stateprep_bitstring())
            return repeated_minimum_kernel(
                core, work_qubits, state_prep, self.num_data, 1
            )

        if self.algorithm == PhaseEstimation.Iterative:
            msg = (
                "Iterative QPE needs mid-circuit measurement feedback, so it has no "
                "circuit-only form; use kernel='all'."
            )
            raise ValueError(msg)

        if self.algorithm != PhaseEstimation.Textbook:
            msg = (
                f"kernel={kernel!r} is only implemented for Textbook QPE in the "
                "CUDA-Q backend."
            )
            raise NotImplementedError(msg)

        hamiltonian = self.hamiltonian.paulis
        if self.simulation == Simulation.Qubitised:
            circuit, work_qubits, decode = qubitised_qpe_circuit(
                hamiltonian, self.num_qpe_ancillas, n_qubits=self.num_data
            )
        else:
            order = self.order if self.simulation == Simulation.Trotter else 1
            circuit, work_qubits, decode = textbook_qpe_circuit(
                hamiltonian,
                self.simulation,
                self.num_qpe_ancillas,
                self.time,
                self.reps,
                order,
                seed=self.seed,
                n_qubits=self.num_data,
            )

        if kernel == "post":
            return sample_post_processor(decode, 1)

        return measured_circuit_kernel(
            circuit,
            work_qubits,
            self.num_qpe_ancillas,
            state_prep=bitstring_kernel(self._get_stateprep_bitstring()),
            num_data=self.num_data,
        )
