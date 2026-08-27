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

"""State-preparation kernels for the CUDA-Q backend."""

from collections.abc import Sequence

from quiche.cudaq._runtime import CudaqKernel, load_cudaq


def bitstring_kernel(bitstring: Sequence[int]) -> CudaqKernel:
    """
    Build a CUDA-Q kernel preparing a computational basis state.

    `bitstring[i]` is the target value of qubit `i`, matching
    `quiche.qualtran.bloqs.BitstringStatePrep` and QuEST's `initClassicalState`.
    The returned kernel has signature `(qubits: cudaq.qview)` and is injectable
    directly as the `state_prep=` argument of the CUDA-Q simulation kernel
    factories in `quiche.cudaq.simulation`; the register it is handed must be at
    least `len(bitstring)` wide and start in |0...0>.
    """
    cudaq, _ = load_cudaq()
    # Bound to a bare name: the CUDA-Q kernel AST compiler resolves calls inside a
    # `@cudaq.kernel` body by name, not by attribute access on a captured module.
    from cudaq_algorithms.stateprep import hartree_fock_occupation  # noqa: PLC0415

    occupied = [int(index) for index, bit in enumerate(bitstring) if bit]

    if not occupied:
        # An empty list capture cannot cross the CUDA-Q kernel boundary
        # (cuda-quantum#4847), so the all-zero bitstring needs its own kernel.
        @cudaq.kernel
        def prepare_zero(qubits: cudaq.qview) -> None:
            """Leave the register in |0...0>."""

        return prepare_zero

    @cudaq.kernel
    def prepare_bitstring(qubits: cudaq.qview) -> None:
        """Flip each occupied qubit to prepare the target bitstring."""
        hartree_fock_occupation(qubits, occupied)

    return prepare_bitstring
