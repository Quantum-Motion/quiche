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

"""
CUDA-Q backend for QUICHE algorithm specs.

Built on NVIDIA's `cudaq_algorithms` library. Both `cudaq` and `cudaq_algorithms`
are soft, optional dependencies: nothing in this package imports them until a
kernel-building function is actually called (`quiche.cudaq._runtime.load_cudaq`
is the single chokepoint).

When `cudaq` *is* installed, import order matters: its MLIR bindings must be the
first native extension loaded in the process, so `import cudaq` before
`import quiche` (or anything that transitively imports quiche's compiled QuEST
extension). `load_cudaq` warns if this order is violated rather than failing
opaquely inside MLIR initialisation.
"""

from quiche.cudaq._runtime import CudaqKernel
from quiche.cudaq.estimation import (
    inverse_qft_kernel,
    iterative_qpe_kernel,
    naive_qpe_kernel,
    qubitised_naive_qpe_kernel,
    qubitised_qpe_kernel,
    textbook_qpe_kernel,
)
from quiche.cudaq.simulation import (
    qdrift_kernel,
    qubitised_controlled_kernel,
    qubitised_kernel,
    simulation_kernel,
    trotter_kernel,
)
from quiche.cudaq.state_prep import bitstring_kernel

__all__ = [
    "CudaqKernel",
    "bitstring_kernel",
    "inverse_qft_kernel",
    "iterative_qpe_kernel",
    "naive_qpe_kernel",
    "qdrift_kernel",
    "qubitised_controlled_kernel",
    "qubitised_kernel",
    "qubitised_naive_qpe_kernel",
    "qubitised_qpe_kernel",
    "simulation_kernel",
    "textbook_qpe_kernel",
    "trotter_kernel",
]
