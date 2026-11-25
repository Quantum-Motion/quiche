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
from typing import TYPE_CHECKING, assert_never

from quiche.bindings.quiche_bindings import (
    getPhaseKitaevQDRIFT,
    getPhaseKitaevTrotter,
    getPhaseTextbookQDRIFT,
    getPhaseTextbookQubitised,
    getPhaseTextbookTrotter,
    initClassicalState,
)
from quiche.chemistry import (
    HartreeFockState,
    get_qubit_state,
)
from quiche.core import (
    PhaseEstimation,
    Simulation,
)
from quiche.simulation import QuESTRoutine, QuregLayout, QuregOp
from quiche.simulation.utils import _with_quest_hamiltonian

if TYPE_CHECKING:
    from quiche import QPESpec


def get_stateprep_op(spec: QPESpec) -> QuregOp:
    """Pattern matching and construction for QuEST state preparation."""
    match spec.state_prep:
        case None:
            return lambda _qureg: None
        case HartreeFockState():
            occupation = spec.state_prep.occupation
            mapping = spec.hamiltonian.mapping
            bitstring = get_qubit_state(occupation, mapping)
            return partial(initClassicalState, state=bitstring)
        case _:
            assert_never(spec.state_prep)


def get_estimation_op(spec: QPESpec, layout: QuregLayout) -> QuregOp:
    """Pattern matching and construction for QuEST phase estimation."""
    match spec.algorithm, spec.simulation:
        case PhaseEstimation.Kitaev, Simulation.QDRIFT:
            return _with_quest_hamiltonian(
                getPhaseKitaevQDRIFT,
                hamiltonian=spec.hamiltonian.paulis,
                reps=spec.reps,
                time=spec.time,
                num_bits=spec.num_rounds,
                seed=spec.seed,
                ancilla_index=layout.qpe[0],
            )

        case PhaseEstimation.Kitaev, Simulation.Trotter:
            return _with_quest_hamiltonian(
                getPhaseKitaevTrotter,
                hamiltonian=spec.hamiltonian.paulis,
                order=spec.order,
                reps=spec.reps,
                time=spec.time,
                num_bits=spec.num_rounds,
                ancilla_index=layout.qpe[0],
            )

        case PhaseEstimation.Textbook, Simulation.QDRIFT:
            return _with_quest_hamiltonian(
                getPhaseTextbookQDRIFT,
                hamiltonian=spec.hamiltonian.paulis,
                reps=spec.reps,
                time=spec.time,
                seed=spec.seed,
                ancillas=layout.qpe,
            )

        case PhaseEstimation.Textbook, Simulation.Qubitised:
            return _with_quest_hamiltonian(
                getPhaseTextbookQubitised,
                hamiltonian=spec.hamiltonian.paulis,
                qpe_ancillas=layout.qpe,
                qubitisation_ancillas=layout.index,
            )

        case PhaseEstimation.Textbook, Simulation.Trotter:
            return _with_quest_hamiltonian(
                getPhaseTextbookTrotter,
                hamiltonian=spec.hamiltonian.paulis,
                order=spec.order,
                reps=spec.reps,
                time=spec.time,
                ancillas=layout.qpe,
            )

        case algorithm, simulation:
            error_msg = (
                f"{algorithm.name} QPE with {simulation.name} not available in QuEST"
                " backend."
            )
            raise NotImplementedError(error_msg)


class QuESTBackend:
    """QuEST backend generating a statevector simulation."""

    def generate(self, spec: QPESpec) -> QuESTRoutine:
        """Build the QuEST routine implementing the given QPESpec."""
        layout = QuregLayout(
            num_data=spec.num_data,
            num_qpe=spec.num_qpe_ancillas,
            num_index=spec.num_index_ancillas,
        )

        return QuESTRoutine(
            ops=(get_stateprep_op(spec), get_estimation_op(spec, layout)),
            layout=layout,
        )
