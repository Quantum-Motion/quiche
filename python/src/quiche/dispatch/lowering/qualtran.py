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

"""Lowering of phase estimation algorithms to Qualtran Bloqs."""

from collections.abc import Callable
from functools import partial

from qualtran import Bloq
from qualtran.bloqs.qubitization.qubitization_walk_operator import (
    QubitizationWalkOperator,
)

from quiche.estimation import EstimationMethod, Iterative, Kitaev, Naive, Textbook
from quiche.qualtran import bloqs
from quiche.qualtran.bloqs.estimation import (
    QubitisationLadder,
    TextbookQPE,
    TrotterLadder,
)
from quiche.simulation import QDRIFT, Qubitised, Trotter


def _simulation_factory(qpe: EstimationMethod) -> Callable[[int], Bloq]:
    """Pattern matching and construction for simulation factories."""
    match qpe.simulation:
        case QDRIFT() as sim:
            bloq = bloqs.QDRIFT(
                h=sim.hamiltonian,
                t=sim.time,
                n_terms=sim.reps,
                seed=sim.seed,
            )
            return partial(
                TrotterLadder,
                simulation=bloq,
                num_data=qpe.num_data,
                num_qpe_ancillas=qpe.num_qpe_ancillas,
            )

        case Qubitised() as sim:
            blockencoding = bloqs.LCUBlockEncodingWrapper.from_hamiltonian(
                sim.hamiltonian, sim.num_phase_ancillas
            )
            return partial(
                QubitisationLadder,
                walk=QubitizationWalkOperator(blockencoding),
                num_data=qpe.num_data,
                num_qpe_ancillas=qpe.num_qpe_ancillas,
                num_selection_ancillas=qpe.num_simulation_ancillas,
            )

        case Trotter() as sim:
            bloq = bloqs.Trotterisation(
                h=sim.hamiltonian,
                t=sim.time,
                n_steps=sim.reps,
                order=sim.order,
            )
            return partial(
                TrotterLadder,
                simulation=bloq,
                num_data=qpe.num_data,
                num_qpe_ancillas=qpe.num_qpe_ancillas,
            )


def estimation(qpe: EstimationMethod) -> Bloq:
    """Build the QPE Bloq, expecting an externally-prepared `data` register."""
    match qpe:
        case Iterative():
            msg = "Iterative QPE Bloq not yet implemented."
            raise NotImplementedError(msg)

        case Kitaev():
            msg = "Kitaev QPE Bloq not yet implemented."
            raise NotImplementedError(msg)

        case Naive():
            msg = "Naive QPE Bloq not yet implemented."
            raise NotImplementedError(msg)

        case Textbook():
            return TextbookQPE(
                simulation_factory=_simulation_factory(qpe),
                num_data=qpe.num_data,
                num_qpe_ancillas=qpe.num_qpe_ancillas,
                num_other_ancillas=qpe.num_simulation_ancillas,
            )
