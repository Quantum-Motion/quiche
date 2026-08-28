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

"""Quantum phase estimation kernels for the CUDA-Q backend."""

# CUDA-Q's intrinsic gates (h, r1, swap, mz, ...) are not real Python names -
# they're recognised by the @cudaq.kernel AST bridge by bare-name pattern
# matching (confirmed: `hasattr(cudaq, "h")` is False), so ruff sees every
# in-kernel gate call as undefined. This file is otherwise plain kernel
# construction, so a file-level suppression is low-risk. The kernel factories
# below also take many named, optional simulation parameters, matching the
# equally wide `getPhaseTextbook*` signatures in quiche.quest.estimation.
# ruff: noqa: F821, PLR0913, PLR0917

from math import pi

from quiche.core import PauliSum, Seed, Simulation
from quiche.cudaq._runtime import CudaqKernel, load_cudaq
from quiche.cudaq.simulation import qdrift_evolution, trotter_evolution


def inverse_qft_kernel() -> CudaqKernel:
    """
    Build a `(qubits: cudaq.qview)` kernel applying the inverse QFT.

    CUDA-Q Algorithms has no QFT primitive; this is a from-scratch
    implementation (bit-reversal swaps, then the standard controlled-rotation
    ladder). Validated directly as a unitary matrix - built from
    `cudaq.get_state` on every computational basis input - against the
    closed-form `(1/sqrt(N)) exp(-2*pi*i*x*y/N)`: an exact match already in
    CUDA-Q's own qubit convention (qubit `i` = bit `i`), with no bit-reversal
    of the *output* needed. `textbook_qpe_kernel`'s controlled-power ladder
    assigns ancilla `k` to `U^(2^k)` specifically because of this convention -
    the more common "qubit 0 = most significant bit" textbook-diagram
    assignment is wrong for this particular circuit.
    """
    cudaq, _ = load_cudaq()

    @cudaq.kernel
    def inverse_qft(qubits: cudaq.qview) -> None:
        """Apply the inverse QFT: bit-reversal swaps, then rotations and Hadamards."""
        n = qubits.size()
        for i in range(n // 2):
            swap(qubits[i], qubits[n - 1 - i])
        for j in range(n):
            for k in range(j):
                angle = -pi / (2.0 ** (j - k))
                r1.ctrl(angle, qubits[j], qubits[k])
            h(qubits[j])

    return inverse_qft


def textbook_qpe_kernel(
    hamiltonian: PauliSum,
    simulation: Simulation,
    num_qpe_ancillas: int,
    time: float,
    reps: int,
    order: int = 2,
    *,
    seed: Seed = None,
    n_qubits: int | None = None,
) -> CudaqKernel:
    """
    Build the Textbook QPE kernel for `Simulation.Trotter`/`Simulation.QDRIFT`.

    The returned kernel has signature `(state: cudaq.State)`: it expects an
    externally-prepared state on the data register (mirroring `to_qualtran`'s
    "compose with a state-preparation Bloq" and `to_quest`'s "compose with a
    state preparation routine" contracts - `state` must have dimension
    `2**n_qubits`, the caller's responsibility, same as
    `cudaq_algorithms.trotter.Trotter.state_kernel`), and ends with `mz` on the
    QPE ancilla register, ready for `cudaq.sample`.

    Ladder rung `k` (`k = 0, ..., num_qpe_ancillas - 1`) applies
    controlled-U^(2^k) by repeating the base `(time, reps, order)`-parametrized
    controlled block `2**k` times - mathematically identical to scaling `time`
    and `reps` by `2**k` in one call (same total step count, same `dt`), but
    avoids needing a fresh, longer QDRIFT sample per rung. Cost grows as `2^k`
    per rung (`2^n - 1` total blocks) - inherent to the textbook algorithm, not
    specific to this implementation (QuEST's C++ backend has the same growth);
    it's the reason iterative/Kitaev QPE (single ancilla, sequential rounds)
    exists as a cheaper alternative for large ancilla counts.

    `identity_coefficient` (dropped from the circuit by `trotter_evolution`/
    `qdrift_evolution` - see their docstrings) is restored per rung with a
    single `r1` phase gate on that rung's ancilla, since `r1(t) = diag(1,
    e^(i*t))` naturally applies only when the ancilla is `|1>`.

    Energy recovery from the sampled ancilla bitstring (interpreted as an
    unsigned integer `y` with ancilla `k` weighted `2**k`, matching this
    kernel's own convention) is the caller's responsibility, exactly as for
    `to_qualtran`/`to_quest`: `phase = y / 2**num_qpe_ancillas`, wrapped to
    `[-0.5, 0.5)`, then
    `energy = phase * (2*pi/time) + hamiltonian.identity_coefficient`.

    `Simulation.Qubitised` raises `NotImplementedError`: `cudaq.control()`
    cannot wrap the qubitisation walk step (it calls other kernels
    internally - confirmed to silently produce wrong results, or fail to
    compile at all, depending on the kernel), so it needs a different
    controlled-power technique than this ladder uses.
    """
    cudaq, _ = load_cudaq()

    match simulation:
        case Simulation.Trotter:
            evolution = trotter_evolution(hamiltonian, n_qubits=n_qubits)
            base_steps, base_order = reps, order
        case Simulation.QDRIFT:
            evolution = qdrift_evolution(
                hamiltonian, reps, seed=seed, n_qubits=n_qubits
            )
            base_steps, base_order = 1, 1
        case Simulation.Qubitised:
            msg = (
                "Simulation.Qubitised is not yet implemented in the CUDA-Q "
                "Textbook QPE backend."
            )
            raise NotImplementedError(msg)

    coefficients = evolution.coefficients
    words = [cudaq.pauli_word(word) for word in evolution.words]
    identity_coefficient = evolution.identity_coefficient

    from cudaq_algorithms.trotter import apply_trotter  # noqa: PLC0415

    inverse_qft = inverse_qft_kernel()

    @cudaq.kernel
    def qpe(state: cudaq.State) -> None:
        """Ladder controlled-U^(2^k), then inverse QFT, then measure the ancillas."""
        data = cudaq.qvector(state)
        ancilla = cudaq.qvector(num_qpe_ancillas)
        h(ancilla)
        for k in range(num_qpe_ancillas):
            power = 1 << k
            for _ in range(power):
                cudaq.control(
                    apply_trotter,
                    ancilla[k],
                    coefficients,
                    words,
                    time,
                    base_steps,
                    base_order,
                    data,
                )
            r1(-identity_coefficient * time * power, ancilla[k])
        inverse_qft(ancilla)
        mz(ancilla)

    return qpe
