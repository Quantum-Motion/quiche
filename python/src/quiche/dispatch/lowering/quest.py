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

"""Lowering of phase estimation algorithms to QuEST routines."""

from collections.abc import Callable
from functools import partial

from quiche.estimation import EstimationMethod, Iterative, Kitaev, Naive, Textbook
from quiche.quest import QuestRoutine
from quiche.quest.estimation import (
    getPhaseKitaevQDRIFT,
    getPhaseKitaevTrotter,
    getPhaseTextbookQDRIFT,
    getPhaseTextbookQubitised,
    getPhaseTextbookTrotter,
)
from quiche.simulation import QDRIFT, Qubitised, Trotter


def _estimation_op(qpe: EstimationMethod) -> Callable:
    """Pattern matching and construction for the QuEST phase estimation call."""
    # Data register: [0, num_data]
    # Ancilla register: [num_data, num_data + num_qpe_ancillas]
    qpe_ancillas = list(range(qpe.num_data, qpe.num_data + qpe.num_qpe_ancillas))

    match qpe:
        case Iterative():
            msg = "Iterative QPE not yet implemented in simulation backend."
            raise NotImplementedError(msg)

        case Kitaev(simulation=QDRIFT() as sim):
            return partial(
                getPhaseKitaevQDRIFT,
                hamiltonian=sim.hamiltonian,
                ancilla_index=qpe_ancillas[0],
                reps=sim.reps,
                time=sim.time,
                num_bits=qpe.num_rounds,
                seed=sim.seed,
            )

        case Kitaev(simulation=Qubitised()):
            msg = "Qubitisation not yet implemented in simulation backend."
            raise NotImplementedError(msg)

        case Kitaev(simulation=Trotter() as sim):
            return partial(
                getPhaseKitaevTrotter,
                hamiltonian=sim.hamiltonian,
                ancilla_index=qpe_ancillas[0],
                order=sim.order,
                reps=sim.reps,
                time=sim.time,
                num_bits=qpe.num_rounds,
            )

        case Naive():
            msg = "Naive QPE not yet implemented in simulation backend."
            raise NotImplementedError(msg)

        case Textbook(simulation=QDRIFT() as sim):
            return partial(
                getPhaseTextbookQDRIFT,
                hamiltonian=sim.hamiltonian,
                ancillas=qpe_ancillas,
                reps=sim.reps,
                time=sim.time,
                seed=sim.seed,
            )

        case Textbook(simulation=Qubitised() as sim):
            # Simulation currently doesn't use ancillas for rotations
            # so only include index ancillas
            start = qpe.num_data + qpe.num_qpe_ancillas
            index_ancillas = list(range(start, start + sim.num_index_ancillas))
            return partial(
                getPhaseTextbookQubitised,
                hamiltonian=sim.hamiltonian,
                qpe_ancillas=qpe_ancillas,
                qubitisation_ancillas=index_ancillas,
            )

        case Textbook(simulation=Trotter() as sim):
            return partial(
                getPhaseTextbookTrotter,
                hamiltonian=sim.hamiltonian,
                ancillas=qpe_ancillas,
                order=sim.order,
                reps=sim.reps,
                time=sim.time,
            )


def estimation(qpe: EstimationMethod) -> QuestRoutine:
    """Build the QPE routine, expecting an externally-prepared initial state."""
    routine = QuestRoutine()
    routine.append(_estimation_op(qpe))
    return routine
