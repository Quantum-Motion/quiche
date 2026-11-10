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

"""Methods to generate the QuEST implementation of a QPESpec object."""

from collections.abc import Callable
from functools import partial
from typing import assert_never

from quiche import PauliSum, algorithms
from quiche.algorithms import (
    EstimationAlgorithm,
    SimulationAlgorithm,
    StateprepAlgorithm,
)
from quiche.bindings.quest_bindings import Qureg
from quiche.bindings.quiche_bindings import (
    getPhaseIterativeQDRIFT,
    getPhaseIterativeTrotter,
    getPhaseKitaevQDRIFT,
    getPhaseKitaevTrotter,
    getPhaseNaiveQDRIFT,
    getPhaseNaiveTrotter,
    getPhaseTextbookQDRIFT,
    getPhaseTextbookQubitised,
    getPhaseTextbookTrotter,
    initClassicalState,
)
from quiche.simulation import SimulationRoutine
from quiche.simulation.qubit_mapping import QubitMapping
from quiche.simulation.routine import QuestOp


def _with_quest_hamiltonian(
    binding: Callable[..., float],
    hamiltonian: PauliSum,
    **kwargs: object,
) -> QuestOp:
    """Defer PauliSum -> PauliStrSum conversion until run inside a QuESTEnv."""

    def op(qureg: Qureg) -> float:
        return binding(qureg, hamiltonian.to_quest(), **kwargs)

    return op


def get_stateprep_questop(stateprep: StateprepAlgorithm) -> QuestOp:
    """Pattern matching and construction for QuEST state preparation."""
    match stateprep:
        case algorithms.IdentityStateprep():
            return lambda _qureg: None
        case algorithms.BitstringStateprep(bitstring=bitstring):
            return partial(initClassicalState, state=bitstring)
        case _:
            assert_never(stateprep)


def get_estimation_questop(
    estimation: EstimationAlgorithm,
    simulation: SimulationAlgorithm,
    *,
    mapping: QubitMapping,
) -> QuestOp:
    """Pattern matching and construction for QuEST phase estimation."""
    match estimation, simulation:
        case algorithms.IterativeQPE(), algorithms.QDRIFT():
            return _with_quest_hamiltonian(
                getPhaseIterativeQDRIFT,
                simulation.hamiltonian,
                reps=simulation.num_samples,
                time=simulation.time,
                seed=simulation.seed,
                num_bits=estimation.num_rounds,
                ancilla_index=mapping["estimation"][0],
            )

        case algorithms.IterativeQPE(), algorithms.Trotter():
            return _with_quest_hamiltonian(
                getPhaseIterativeTrotter,
                simulation.hamiltonian,
                order=simulation.order,
                reps=simulation.num_steps,
                time=simulation.time,
                num_bits=estimation.num_rounds,
                ancilla_index=mapping["estimation"][0],
            )

        case algorithms.KitaevQPE(), algorithms.QDRIFT():
            return _with_quest_hamiltonian(
                getPhaseKitaevQDRIFT,
                simulation.hamiltonian,
                reps=simulation.num_samples,
                time=simulation.time,
                seed=simulation.seed,
                num_bits=estimation.num_rounds,
                ancilla_index=mapping["estimation"][0],
            )

        case algorithms.KitaevQPE(), algorithms.Trotter():
            return _with_quest_hamiltonian(
                getPhaseKitaevTrotter,
                simulation.hamiltonian,
                order=simulation.order,
                reps=simulation.num_steps,
                time=simulation.time,
                num_bits=estimation.num_rounds,
                ancilla_index=mapping["estimation"][0],
            )

        case algorithms.NaiveQPE(), algorithms.QDRIFT():
            return _with_quest_hamiltonian(
                getPhaseNaiveQDRIFT,
                simulation.hamiltonian,
                reps=simulation.num_samples,
                time=simulation.time,
                seed=simulation.seed,
                ancilla_index=mapping["estimation"][0],
            )

        case algorithms.NaiveQPE(), algorithms.Trotter():
            return _with_quest_hamiltonian(
                getPhaseNaiveTrotter,
                simulation.hamiltonian,
                order=simulation.order,
                reps=simulation.num_steps,
                time=simulation.time,
                ancilla_index=mapping["estimation"][0],
            )

        case algorithms.TextbookQPE(), algorithms.QDRIFT():
            return _with_quest_hamiltonian(
                getPhaseTextbookQDRIFT,
                simulation.hamiltonian,
                reps=simulation.num_samples,
                time=simulation.time,
                seed=simulation.seed,
                ancillas=mapping["estimation"],
            )

        case algorithms.TextbookQPE(), algorithms.Qubitisation():
            return _with_quest_hamiltonian(
                getPhaseTextbookQubitised,
                simulation.hamiltonian,
                qpe_ancillas=mapping["estimation"],
                qubitisation_ancillas=mapping["simulation"],
            )

        case algorithms.TextbookQPE(), algorithms.Trotter():
            return _with_quest_hamiltonian(
                getPhaseTextbookTrotter,
                simulation.hamiltonian,
                order=simulation.order,
                reps=simulation.num_steps,
                time=simulation.time,
                ancillas=mapping["estimation"],
            )

        case _:
            msg = f"{estimation} QPE with {simulation} not available in QuEST backend."
            raise NotImplementedError(msg)


def _num_simulation_ancillas(simulation: SimulationAlgorithm) -> int:
    """Ancillas the QuEST implementation of `simulation` needs."""
    match simulation:
        case algorithms.Trotter() | algorithms.QDRIFT():
            return 0
        case algorithms.Qubitisation():
            # QuEST prepares the index state with direct rotations, so its PREP
            # needs no QROM or phase gradient register, whatever `prep` says.
            return simulation.num_index_ancillas
        case _:
            assert_never(simulation)


class QuESTBackend:
    """QuEST backend generating a SimulationRoutine for statevector simulation."""

    def generate(
        self,
        stateprep: StateprepAlgorithm,
        estimation: EstimationAlgorithm,
        simulation: SimulationAlgorithm,
    ) -> SimulationRoutine:
        """Build the QuEST SimulationRoutine implementing the given QPESpec."""
        mapping = QubitMapping.contiguous(
            data=simulation.hamiltonian.num_qubits,
            estimation=estimation.num_ancillas,
            simulation=_num_simulation_ancillas(simulation),
        )

        return SimulationRoutine(
            mapping=mapping,
            ops=(
                get_stateprep_questop(stateprep),
                get_estimation_questop(estimation, simulation, mapping=mapping),
            ),
        )
