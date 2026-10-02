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
    iterative_qpe_core,
    naive_qpe_kernel,
    qubitised_naive_qpe_kernel,
    qubitised_qpe_core,
    repeated_minimum_kernel,
    single_run_kernel,
    textbook_qpe_core,
)
from quiche.estimation import EstimationMethod, Iterative, Kitaev, Naive, Textbook
from quiche.simulation import Qubitised


def estimation(
    qpe: EstimationMethod, state_prep: CudaqKernel | None = None
) -> CudaqKernel:
    """Build the QPE kernel, repeated with `state_prep` if given; see `to_cudaq`."""
    if isinstance(qpe, Naive):
        if state_prep is not None:
            msg = (
                "Naive QPE returns single-shot +-1 outcomes, not energies, so there "
                "is no minimum to take over repetitions; call to_cudaq() without "
                "state_prep."
            )
            raise ValueError(msg)
        return _naive(qpe)

    core, work_qubits = _core(qpe)
    if state_prep is None:
        return single_run_kernel(core, work_qubits)
    return repeated_minimum_kernel(
        core, work_qubits, state_prep, qpe.num_data, qpe.repetitions
    )


def _naive(qpe: Naive) -> CudaqKernel:
    """Build the `(data: cudaq.qview) -> float` Hadamard-test kernel."""
    match qpe.simulation:
        case Qubitised() as sim:
            return qubitised_naive_qpe_kernel(sim.hamiltonian, n_qubits=qpe.num_data)
        case sim:
            return naive_qpe_kernel(sim, n_qubits=qpe.num_data)


def _core(qpe: Textbook | Kitaev | Iterative) -> tuple[CudaqKernel, int]:
    """Build the `(data, work) -> float` core kernel and its work register size."""
    match qpe:
        case Textbook(simulation=Qubitised() as sim):
            return qubitised_qpe_core(
                sim.hamiltonian, qpe.num_qpe_ancillas, n_qubits=qpe.num_data
            )

        case Textbook(simulation=sim):
            return textbook_qpe_core(sim, qpe.num_qpe_ancillas, n_qubits=qpe.num_data)

        case Iterative(simulation=Qubitised()):
            msg = (
                "Qubitised simulation is not yet implemented for "
                "Iterative QPE in the CUDA-Q backend."
            )
            raise NotImplementedError(msg)

        case Iterative(simulation=sim):
            return iterative_qpe_core(sim, qpe.num_rounds, n_qubits=qpe.num_data)

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
