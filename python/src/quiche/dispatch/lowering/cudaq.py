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

"""Lowering of phase estimation algorithms to CUDA-Q kernels."""

from quiche.cudaq import (
    CudaqKernel,
    iterative_qpe_kernel,
    naive_qpe_kernel,
    qubitised_naive_qpe_kernel,
    qubitised_qpe_kernel,
    textbook_qpe_kernel,
)
from quiche.estimation import EstimationMethod, Iterative, Kitaev, Naive, Textbook
from quiche.simulation import Qubitised


def estimation(qpe: EstimationMethod) -> CudaqKernel:
    """Build the `(data: cudaq.qview) -> float` QPE kernel; see `to_cudaq`."""
    match qpe:
        case Textbook(simulation=Qubitised() as sim):
            return qubitised_qpe_kernel(
                sim.hamiltonian, qpe.num_qpe_ancillas, n_qubits=qpe.num_data
            )

        case Textbook(simulation=sim):
            return textbook_qpe_kernel(sim, qpe.num_qpe_ancillas, n_qubits=qpe.num_data)

        case Naive(simulation=Qubitised() as sim):
            return qubitised_naive_qpe_kernel(sim.hamiltonian, n_qubits=qpe.num_data)

        case Naive(simulation=sim):
            return naive_qpe_kernel(sim, n_qubits=qpe.num_data)

        case Iterative(simulation=Qubitised()):
            msg = (
                "Qubitised simulation is not yet implemented for "
                "Iterative QPE in the CUDA-Q backend."
            )
            raise NotImplementedError(msg)

        case Iterative(simulation=sim):
            return iterative_qpe_kernel(sim, qpe.num_rounds, n_qubits=qpe.num_data)

        case Kitaev():
            msg = (
                "Kitaev QPE (in this repo's sense - per-round "
                "expectation-value estimation via repeated measurement, "
                "no feedback rotation) has no single-kernel decode: it "
                "needs many-shot statistics at each of several `power` "
                "values combined classically across rounds. Build it "
                "from naive_qpe_kernel at varying power plus host-side "
                "aggregation instead."
            )
            raise NotImplementedError(msg)
