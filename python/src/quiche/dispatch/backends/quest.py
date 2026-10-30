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

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from quiche.bindings.quiche_bindings import initClassicalState
from quiche.chemistry import (
    HartreeFockState,
    get_qubit_state,
)
from quiche.core import (
    PhaseEstimation,
    Simulation,
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
    from collections.abc import Callable

    from quiche import QPESpec


def _get_stateprep(spec: QPESpec) -> Callable:
    """Pattern matching and construction for QuEST state preparation."""
    match spec.state_prep:
        case None:
            return lambda _qureg: None

        case HartreeFockState():
            occupation = spec.state_prep.occupation
            mapping = spec.hamiltonian.mapping
            bitstring = get_qubit_state(occupation, mapping)
            return partial(initClassicalState, state=bitstring)


def _get_estimation(spec: QPESpec) -> Callable:
    """Pattern matching and construction for QuEST phase estimation."""
    # Register layout: [data | qpe ancillas | index ancillas]
    qpe_ancillas_start = spec.num_data

    # Simulation doesn't use rotation ancillas, so only index ancillas are included
    index_ancillas_start = spec.num_data + spec.num_qpe_ancillas
    index_ancillas_end = index_ancillas_start + spec.num_index_ancillas

    qpe_ancillas = list(range(qpe_ancillas_start, index_ancillas_start))
    index_ancillas = list(range(index_ancillas_start, index_ancillas_end))

    match spec.algorithm, spec.simulation:
        case PhaseEstimation.Kitaev, Simulation.QDRIFT:
            return partial(
                getPhaseKitaevQDRIFT,
                hamiltonian=spec.hamiltonian.paulis,
                ancilla_index=qpe_ancillas[0],
                reps=spec.reps,
                time=spec.time,
                num_bits=spec.num_rounds,
                seed=spec.seed,
            )

        case PhaseEstimation.Kitaev, Simulation.Trotter:
            return partial(
                getPhaseKitaevTrotter,
                hamiltonian=spec.hamiltonian.paulis,
                ancilla_index=qpe_ancillas[0],
                order=spec.order,
                reps=spec.reps,
                time=spec.time,
                num_bits=spec.num_rounds,
            )

        case PhaseEstimation.Textbook, Simulation.QDRIFT:
            return partial(
                getPhaseTextbookQDRIFT,
                hamiltonian=spec.hamiltonian.paulis,
                ancillas=qpe_ancillas,
                reps=spec.reps,
                time=spec.time,
                seed=spec.seed,
            )

        case PhaseEstimation.Textbook, Simulation.Qubitised:
            return partial(
                getPhaseTextbookQubitised,
                hamiltonian=spec.hamiltonian.paulis,
                qpe_ancillas=qpe_ancillas,
                qubitisation_ancillas=index_ancillas,
            )

        case PhaseEstimation.Textbook, Simulation.Trotter:
            return partial(
                getPhaseTextbookTrotter,
                hamiltonian=spec.hamiltonian.paulis,
                ancillas=qpe_ancillas,
                order=spec.order,
                reps=spec.reps,
                time=spec.time,
            )

        case algorithm, simulation:
            msg = f"{algorithm} QPE with {simulation} not available in QuEST backend."
            raise NotImplementedError(msg)


class QuESTBackend:
    """QuEST backend generating a SimulationRoutine for statevector simulation."""

    def generate(self, spec: QPESpec) -> SimulationRoutine:
        """Build the QuEST SimulationRoutine implementing the given QPESpec."""
        routine = SimulationRoutine()
        routine.append(_get_stateprep(spec))
        routine.append(_get_estimation(spec))
        return routine
