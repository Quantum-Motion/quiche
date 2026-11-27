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
from typing import TYPE_CHECKING, assert_never

from qualtran import Bloq, BloqBuilder, CompositeBloq
from qualtran.bloqs.qubitization import QubitizationWalkOperator

from quiche import algorithms
from quiche.dispatch.backends import UnsupportedAlgorithmError
from quiche.resources import bloqs

if TYPE_CHECKING:
    from collections.abc import Callable

    from quiche import QPESpec


def get_stateprep_bloq(spec: QPESpec) -> Bloq:
    """Pattern matching and construction for state preparation Bloq."""
    match spec.stateprep:
        case algorithms.IdentityStateprep(num_qubits=num_qubits):
            return bloqs.IdentityStatePrep(num_qubits)
        case algorithms.BitstringStateprep(bitstring=bitstring):
            return bloqs.BitstringStatePrep(bitstring)
        case _:
            assert_never(spec.stateprep)


def get_simulation_bloq_factory(spec: QPESpec) -> Callable[[int], Bloq]:
    """Pattern matching and construction for simulation factories."""
    match spec.simulation:
        case algorithms.QDRIFT(
            hamiltonian=hamiltonian,
            time=time,
            num_samples=num_samples,
            seed=seed,
        ):
            bloq = bloqs.QDRIFT(
                sum=hamiltonian,
                time=time,
                num_samples=num_samples,
                seed=seed,
            )

            factory = partial(
                bloqs.TrotterLadder,
                simulation=bloq,
                num_data=spec.simulation.hamiltonian.num_qubits,
                num_qpe_ancillas=spec.estimation.num_ancillas,
            )

        case algorithms.Qubitisation(hamiltonian=hamiltonian, prep=prep):
            match prep:
                case algorithms.QROMRotationTree(angle_bits=angle_bits):
                    blockencoding = bloqs.LCUBlockEncodingWrapper.from_hamiltonian(
                        hamiltonian,
                        angle_bits,
                    )
                case algorithms.AliasSampling() | algorithms.ExactRotationTree():
                    raise UnsupportedAlgorithmError(spec, QualtranBackend)
                case _:
                    assert_never(prep)

            factory = partial(
                bloqs.QubitisationLadder,
                walk=QubitizationWalkOperator(blockencoding),
                num_data=hamiltonian.num_qubits,
                num_qpe_ancillas=spec.estimation.num_ancillas,
                num_selection_ancillas=spec.simulation.num_ancillas,
            )

        case algorithms.Trotter(
            hamiltonian=hamiltonian,
            time=time,
            order=order,
            num_steps=num_steps,
        ):
            bloq = bloqs.Trotterisation(
                sum=hamiltonian,
                time=time,
                order=order,
                num_steps=num_steps,
            )

            factory = partial(
                bloqs.TrotterLadder,
                simulation=bloq,
                num_data=spec.simulation.hamiltonian.num_qubits,
                num_qpe_ancillas=spec.estimation.num_ancillas,
            )

        case _:
            assert_never(spec.simulation)

    return factory


def get_estimation_bloq(spec: QPESpec) -> Bloq:
    """Pattern matching and construction for QPE Bloq."""
    match spec.estimation:
        case algorithms.TextbookQPE():
            return bloqs.TextbookQPE(
                simulation_factory=get_simulation_bloq_factory(spec),
                num_data=spec.simulation.hamiltonian.num_qubits,
                num_qpe_ancillas=spec.estimation.num_ancillas,
                num_other_ancillas=spec.simulation.num_ancillas,
            )

        # TODO(Vasco): wire in the other single ancilla QPE methods

        case _:
            raise UnsupportedAlgorithmError(spec, QualtranBackend)


class QualtranBackend:
    """Qualtran backend generating a CompositeBloq for resource estimation."""

    def generate(self, spec: QPESpec) -> CompositeBloq:
        """Build the Qualtran CompositeBloq implementing the given QPESpec."""
        bb = BloqBuilder()
        data = bb.add(get_stateprep_bloq(spec))
        data = bb.add(get_estimation_bloq(spec), data=data)
        bb.free(data)
        return bb.finalize()
