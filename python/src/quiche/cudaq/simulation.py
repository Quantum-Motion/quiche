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

"""Hamiltonian-simulation kernels for the CUDA-Q backend."""

# The kernel factories below take many named, optional simulation parameters,
# matching the equally wide `getPhaseTextbook*` signatures in quiche.quest.estimation.
# ruff: noqa: PLR0913

from math import copysign
from typing import TYPE_CHECKING

from quiche.core import PauliSum, Seed, Simulation
from quiche.core.qdrift import sample_qdrift_indices
from quiche.cudaq._runtime import CudaqKernel, load_cudaq

if TYPE_CHECKING:
    # cudaq_algorithms is a soft, optional dependency; only imported for real
    # inside `load_cudaq`, never at module import time.
    from cudaq_algorithms.trotter import Trotter as CudaqTrotter

# CUDA-Q's Suzuki-Trotter primitive only implements these product-formula orders.
_SUPPORTED_TROTTER_ORDERS = (1, 2, 4)


def _pauli_pairs(hamiltonian: PauliSum, n_qubits: int) -> list[tuple[float, str]]:
    """
    Render a PauliSum as `(coefficient, word)` pairs in CUDA-Q's word convention.

    A list of pairs, never a `{word: coefficient}` mapping: `cudaq_algorithms`
    collapses duplicate mapping keys, and QDRIFT's sampled sequence intentionally
    repeats words.

    The trailing all-identity pair carries `hamiltonian.identity_coefficient`
    onto the returned `Trotter.identity_coefficient` (see `trotter_evolution`)
    and pins the register width at `n_qubits` even when no term reaches the top
    qubit. A zero identity coefficient is dropped again once the width is fixed,
    so the pad is always safe to append.
    """
    pairs = [
        (coefficient, term.to_str(n_qubits, big_endian=True))
        for coefficient, term in zip(
            hamiltonian.coefficients, hamiltonian.terms, strict=True
        )
    ]
    pairs.append((hamiltonian.identity_coefficient, "I" * n_qubits))
    return pairs


def trotter_evolution(
    hamiltonian: PauliSum, *, n_qubits: int | None = None
) -> "CudaqTrotter":
    """
    Build the `cudaq_algorithms.trotter.Trotter` object for a PauliSum.

    `n_qubits` fixes the register width; it defaults to the Hamiltonian's own
    width but must be given explicitly when the caller's register is wider (e.g.
    a `QPESpec` whose `n_qubits` exceeds every term's extent).

    The resulting kernels realise `exp(-i (H - identity_coefficient) t)`: CUDA-Q
    cannot represent a global phase in a circuit, so `identity_coefficient` is
    carried on the returned object (`Trotter.identity_coefficient`) rather than
    applied. `cudaq_algorithms.sim_utils.evolve` restores it for a full
    statevector; a controlled use inside QPE will need its own phase gate.
    """
    _, algorithms = load_cudaq()
    width = hamiltonian.n_qubits if n_qubits is None else n_qubits
    pairs = _pauli_pairs(hamiltonian, width)
    return algorithms.trotter.Trotter(
        pairs,
        algorithms.trotter.TrotterOrdering.PRESERVE_INPUT,
        coefficient_tolerance=0.0,
    )


def qdrift_evolution(
    hamiltonian: PauliSum,
    reps: int,
    *,
    seed: Seed = None,
    n_qubits: int | None = None,
) -> "CudaqTrotter":
    """
    Build a `Trotter` object whose term list is one QDRIFT sample sequence.

    Draws `reps` terms with `sample_qdrift_indices` (term `j` with probability
    `|h_j| / lam`) and gives each sampled term magnitude `lam / reps`, keeping
    only the sign of its original coefficient - the frequency of sampling
    already accounts for its magnitude. `trotter.apply_trotter` at
    `steps=1, order=1` applies `exp(-i * time * coefficient_k * P_k)` per list
    entry in order, so this reproduces the QDRIFT channel
    `prod_k exp(-i * sign(h_jk) * (lam * time / reps) * P_jk)` exactly once
    `time` is supplied at kernel-build time (see `qdrift_kernel`).

    Register width and the identity-coefficient handling are as in
    `trotter_evolution`.
    """
    if reps < 1:
        msg = "QDRIFT requires a positive number of repetitions."
        raise ValueError(msg)

    _, algorithms = load_cudaq()
    width = hamiltonian.n_qubits if n_qubits is None else n_qubits
    words = [term.to_str(width, big_endian=True) for term in hamiltonian.terms]
    magnitude = hamiltonian.lam / reps

    pairs = [
        (copysign(magnitude, hamiltonian.coefficients[index]), words[index])
        for index in sample_qdrift_indices(hamiltonian, reps, seed)
    ]
    pairs.append((hamiltonian.identity_coefficient, "I" * width))

    return algorithms.trotter.Trotter(
        pairs,
        algorithms.trotter.TrotterOrdering.PRESERVE_INPUT,
        coefficient_tolerance=0.0,
    )


def trotter_kernel(
    hamiltonian: PauliSum,
    time: float,
    reps: int,
    order: int = 2,
    *,
    n_qubits: int | None = None,
    state_prep: CudaqKernel | None = None,
) -> CudaqKernel:
    """
    Build the CUDA-Q kernel for `reps` steps of the `order`-order Trotter formula.

    Evolves for `time`, optionally preceded by `state_prep`. See
    `trotter_evolution` for the register-width and identity-phase contract.
    """
    if order not in _SUPPORTED_TROTTER_ORDERS:
        msg = (
            f"CUDA-Q Trotter supports order in {_SUPPORTED_TROTTER_ORDERS}, "
            f"got {order}."
        )
        raise ValueError(msg)

    evolution = trotter_evolution(hamiltonian, n_qubits=n_qubits)
    return evolution.kernel(time=time, steps=reps, order=order, state_prep=state_prep)


def qdrift_kernel(
    hamiltonian: PauliSum,
    time: float,
    reps: int,
    *,
    seed: Seed = None,
    n_qubits: int | None = None,
    state_prep: CudaqKernel | None = None,
) -> CudaqKernel:
    """
    Build the CUDA-Q kernel for one QDRIFT sample sequence of `reps` terms.

    Evolves for `time`, optionally preceded by `state_prep`. See
    `qdrift_evolution` for the sampling contract and `trotter_evolution` for the
    register-width and identity-phase contract.
    """
    evolution = qdrift_evolution(hamiltonian, reps, seed=seed, n_qubits=n_qubits)
    return evolution.kernel(time=time, steps=1, order=1, state_prep=state_prep)


def simulation_kernel(
    simulation: Simulation,
    hamiltonian: PauliSum,
    time: float,
    reps: int,
    *,
    order: int = 2,
    seed: Seed = None,
    n_qubits: int | None = None,
    state_prep: CudaqKernel | None = None,
) -> CudaqKernel:
    """
    Dispatch to the CUDA-Q Hamiltonian-simulation kernel for `simulation`.

    `order` is ignored for `Simulation.QDRIFT` (its product formula is
    first-order by construction) and `seed` is ignored for `Simulation.Trotter`
    (it is deterministic). `Simulation.Qubitised` is not yet implemented in the
    CUDA-Q backend and is checked before any CUDA-Q import is attempted.
    """
    match simulation:
        case Simulation.Trotter:
            return trotter_kernel(
                hamiltonian, time, reps, order, n_qubits=n_qubits, state_prep=state_prep
            )

        case Simulation.QDRIFT:
            return qdrift_kernel(
                hamiltonian,
                time,
                reps,
                seed=seed,
                n_qubits=n_qubits,
                state_prep=state_prep,
            )

        case Simulation.Qubitised:
            msg = "Qubitised simulation is not yet implemented in the CUDA-Q backend."
            raise NotImplementedError(msg)
