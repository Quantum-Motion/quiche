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

"""Methods to generate the Qualtran implementation of a QPESpec object."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from qualtran import Bloq, BloqBuilder, CompositeBloq
from qualtran.bloqs.qubitization.qubitization_walk_operator import (
    QubitizationWalkOperator,
)

from quiche.chemistry import (
    HartreeFockState,
    get_qubit_state,
)
from quiche.core import (
    PhaseEstimation,
    Simulation,
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

if TYPE_CHECKING:
    from collections.abc import Callable

    from quiche import QPESpec


def _get_stateprep_bloq(spec: QPESpec) -> Bloq:
    """Pattern matching and construction for state preparation Bloq."""
    match spec.state_prep:
        case None:
            state_prep_bloq = IdentityStatePrep(spec.num_data)
        case HartreeFockState():
            mapping = spec.hamiltonian.mapping
            occupation = spec.state_prep.occupation
            bitstring = get_qubit_state(occupation, mapping)
            state_prep_bloq = BitstringStatePrep(tuple(bitstring))

    return state_prep_bloq


def _get_simulation_bloq_factory(spec: QPESpec) -> Callable[[int], Bloq]:
    """Pattern matching and construction for simulation factories."""
    match spec.simulation:
        case Simulation.QDRIFT:
            bloq = QDRIFT(
                sum=spec.hamiltonian.paulis,
                time=spec.time,
                num_samples=spec.reps,
                seed=spec.seed,
            )

            simulation_bloq_factory = partial(
                TrotterLadder,
                simulation=bloq,
                num_data=spec.num_data,
                num_qpe_ancillas=spec.num_qpe_ancillas,
            )

        case Simulation.Qubitised:
            blockencoding = LCUBlockEncodingWrapper.from_hamiltonian(
                spec.hamiltonian.paulis, spec.num_phase_ancillas
            )
            walk_op = QubitizationWalkOperator(blockencoding)

            simulation_bloq_factory = partial(
                QubitisationLadder,
                walk=walk_op,
                num_data=spec.num_data,
                num_qpe_ancillas=spec.num_qpe_ancillas,
                num_selection_ancillas=spec.num_simulation_ancillas,
            )

        case Simulation.Trotter:
            bloq = Trotterisation(
                sum=spec.hamiltonian.paulis,
                time=spec.time,
                num_steps=spec.reps,
                order=spec.order,
            )

            simulation_bloq_factory = partial(
                TrotterLadder,
                simulation=bloq,
                num_data=spec.num_data,
                num_qpe_ancillas=spec.num_qpe_ancillas,
            )

    return simulation_bloq_factory


def _get_estimation_bloq(spec: QPESpec) -> Bloq:
    """Pattern matching and construction for QPE Bloq."""
    match spec.algorithm:
        case PhaseEstimation.Textbook:
            estimation_bloq = TextbookQPE(
                simulation_factory=_get_simulation_bloq_factory(spec),
                num_data=spec.num_data,
                num_qpe_ancillas=spec.num_qpe_ancillas,
                num_other_ancillas=spec.num_simulation_ancillas,
            )

        case PhaseEstimation.Iterative | PhaseEstimation.Kitaev | PhaseEstimation.Naive:
            msg = f"{spec.algorithm} QPE not available in Qualtran backend."
            raise NotImplementedError(msg)

    return estimation_bloq


class QualtranBackend:
    """Qualtran backend generating a CompositeBloq for resource estimation."""

    def generate(self, spec: QPESpec) -> CompositeBloq:
        """Build the Qualtran CompositeBloq implementing the given QPESpec."""
        bb = BloqBuilder()
        data = bb.add(_get_stateprep_bloq(spec))
        data = bb.add(_get_estimation_bloq(spec), data=data)
        bb.free(data)
        return bb.finalize()
