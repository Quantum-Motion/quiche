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

from collections.abc import Callable
from functools import partial
from typing import assert_never

from qualtran import Bloq, BloqBuilder, CompositeBloq
from qualtran.bloqs.qubitization.qubitization_walk_operator import (
    QubitizationWalkOperator,
)

from quiche import algorithms
from quiche.algorithms import (
    EstimationAlgorithm,
    RotationTree,
    SimulationAlgorithm,
    StateprepAlgorithm,
)
from quiche.resources import bloqs


def get_stateprep_bloq(stateprep: StateprepAlgorithm) -> Bloq:
    """Pattern matching and construction for state preparation Bloq."""
    match stateprep:
        case algorithms.IdentityStateprep(num_qubits=num_qubits):
            return bloqs.IdentityStatePrep(num_qubits)
        case algorithms.BitstringStateprep(bitstring=bitstring):
            return bloqs.BitstringStatePrep(bitstring)
        case _:
            assert_never(stateprep)


def get_simulation_bloq_factory(
    estimation: EstimationAlgorithm,
    simulation: SimulationAlgorithm,
) -> Callable[[int], Bloq]:
    """Pattern matching and construction for simulation factories."""
    match simulation:
        case algorithms.QDRIFT():
            bloq = bloqs.QDRIFT(
                sum=simulation.hamiltonian,
                time=simulation.time,
                num_samples=simulation.num_samples,
                seed=simulation.seed,
            )

            return partial(
                bloqs.TrotterLadder,
                simulation=bloq,
                num_data=simulation.hamiltonian.num_qubits,
                num_qpe_ancillas=estimation.num_ancillas,
            )

        case algorithms.Qubitisation():
            match simulation.prep:
                case RotationTree():
                    blockencoding = bloqs.LCUBlockEncodingWrapper.from_hamiltonian(
                        simulation.hamiltonian,
                        simulation.prep.angle_bits,
                    )
                case _:
                    msg = "Stateprep algorithm not available in Qualtran backend."
                    raise NotImplementedError(msg)

            walk_op = QubitizationWalkOperator(blockencoding)

            return partial(
                bloqs.QubitisationLadder,
                walk=walk_op,
                num_data=simulation.hamiltonian.num_qubits,
                num_qpe_ancillas=estimation.num_ancillas,
                num_selection_ancillas=simulation.num_ancillas,
            )

        case algorithms.Trotter():
            bloq = bloqs.Trotterisation(
                sum=simulation.hamiltonian,
                time=simulation.time,
                num_steps=simulation.num_steps,
                order=simulation.order,
            )

            return partial(
                bloqs.TrotterLadder,
                simulation=bloq,
                num_data=simulation.hamiltonian.num_qubits,
                num_qpe_ancillas=estimation.num_ancillas,
            )

        case _:
            assert_never(simulation)


def get_estimation_bloq(
    estimation: EstimationAlgorithm,
    simulation: SimulationAlgorithm,
) -> Bloq:
    """Pattern matching and construction for QPE Bloq."""
    match estimation:
        case algorithms.TextbookQPE():
            return bloqs.TextbookQPE(
                simulation_factory=get_simulation_bloq_factory(estimation, simulation),
                num_data=simulation.hamiltonian.num_qubits,
                num_qpe_ancillas=estimation.num_bits,
                num_other_ancillas=simulation.num_ancillas,
            )

        case algorithms.IterativeQPE() | algorithms.KitaevQPE() | algorithms.NaiveQPE():
            msg = f"{estimation} QPE not available in Qualtran backend."
            raise NotImplementedError(msg)

        case _:
            assert_never(estimation)


class QualtranBackend:
    """Qualtran backend generating a CompositeBloq for resource estimation."""

    def generate(
        self,
        stateprep: StateprepAlgorithm,
        estimation: EstimationAlgorithm,
        simulation: SimulationAlgorithm,
    ) -> CompositeBloq:
        """Build the Qualtran CompositeBloq implementing the given QPESpec."""
        bb = BloqBuilder()
        data = bb.add(get_stateprep_bloq(stateprep))
        data = bb.add(get_estimation_bloq(estimation, simulation), data=data)
        bb.free(data)
        return bb.finalize()
