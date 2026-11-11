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

"""Tests for the CUDA-Q backend."""

import statistics
import subprocess
import sys
from math import copysign
from types import ModuleType

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.linalg import expm

from quiche.chemistry import HartreeFockState, get_jw_state
from quiche.core import (
    ElectronicHamiltonian,
    Errors,
    Mapping,
    PauliSum,
    PauliWord,
    PhaseEstimation,
    Simulation,
)
from quiche.core.qdrift import sample_qdrift_indices
from quiche.cudaq import CudaqKernel
from quiche.cudaq.estimation import (
    inverse_qft_kernel,
    iterative_qpe_kernel,
    measured_circuit_kernel,
    naive_qpe_kernel,
    qubitised_naive_qpe_kernel,
    qubitised_qpe_circuit,
    qubitised_qpe_kernel,
    repeated_minimum_kernel,
    sample_post_processor,
    textbook_qpe_circuit,
    textbook_qpe_core,
    textbook_qpe_kernel,
)
from quiche.cudaq.simulation import (
    qdrift_evolution,
    qdrift_kernel,
    qubitised_controlled_kernel,
    qubitised_encoding,
    qubitised_kernel,
    qubitised_walk,
    simulation_kernel,
    trotter_evolution,
    trotter_kernel,
)
from quiche.cudaq.state_prep import bitstring_kernel
from quiche.dispatch import QPESpec
from quiche.resources.bloqs import QDRIFT

TIME = 0.7
ZERO_KET = np.array([1, 0, 0, 0], dtype=complex)  # |00>, big-endian, 2 qubits


def _pauli_sum(*terms: tuple[float, str], identity: float = 0.0) -> PauliSum:
    """Build a PauliSum from (coeff, dense word) pairs; word index == qubit index."""
    return PauliSum(
        coefficients=tuple(c for c, _ in terms),
        words=tuple(PauliWord.from_dense_string(w, big_endian=True) for _, w in terms),
        identity_coefficient=identity,
    )


# All-Z, so every Trotter order is exact regardless of step count.
COMMUTING = _pauli_sum((0.5, "ZI"), (0.4, "IZ"), (0.3, "ZZ"), identity=0.2)
# Non-commuting, for genuine Trotter/QDRIFT convergence checks.
GENERAL = _pauli_sum((0.7, "XI"), (-0.4, "IZ"), (0.31, "XZ"), identity=-0.2)
# 1-norm of the non-identity terms, the QDRIFT sampling normalisation.
GENERAL_LAM = GENERAL.without_identity().lam
SINGLE = _pauli_sum((-0.6, "XZ"))
# One qubit narrower than its register, to exercise explicit `n_qubits` padding.
NARROW = _pauli_sum((1.0, "X"))


def _big_endian(state: NDArray) -> NDArray[np.complex128]:
    """
    Reindex a CUDA-Q statevector into quiche's qubit-0-most-significant order.

    CUDA-Q indexes amplitudes with qubit `b` as bit `b` of the index (qubit 0
    least significant); `PauliSum.to_matrix` krons qubit 0 first (qubit 0 most
    significant). Splitting the flat index into one axis per qubit and reversing
    the axis order is exactly that relabelling, and is its own inverse.
    """
    amplitudes = np.asarray(state, dtype=complex)
    n_qubits = amplitudes.size.bit_length() - 1
    return amplitudes.reshape([2] * n_qubits).transpose().reshape(-1)


def _exact(hamiltonian: PauliSum, time: float, ket: NDArray) -> NDArray[np.complex128]:
    """Compute exp(-i H t)|ket>, in quiche's big-endian basis."""
    return expm(-1j * time * hamiltonian.to_matrix()) @ ket


def _identity_phase(hamiltonian: PauliSum, time: float) -> complex:
    """Get exp(-i c_I t), the global phase tracked by `identity_coefficient`."""
    return complex(np.exp(-1j * hamiltonian.identity_coefficient * time))


def _reference_pairs(hamiltonian: PauliSum, n_qubits: int) -> list[tuple[float, str]]:
    """Get (coefficient, word) pairs in CUDA-Q's convention, for `_dense_matrix`."""
    pairs = [
        (c, word.to_dense_string(n_qubits, big_endian=True))
        for c, word in zip(hamiltonian.coefficients, hamiltonian.words, strict=True)
    ]
    pairs.append((hamiltonian.identity_coefficient, "I" * n_qubits))
    return pairs


def _dense_matrix(
    terms: list[tuple[float, str]], n_qubits: int
) -> NDArray[np.complex128]:
    """
    Get the dense Pauli-sum matrix in CUDA-Q's own qubit order (qubit 0 = LSB).

    A from-scratch port of `cudaq_algorithms`'s own `dense_references.dense_matrix`
    test helper - not `PauliSum.to_matrix`, which uses the opposite convention.
    Kept independent of quiche's production code as a genuine cross-check, used
    only for the Qubitised tests below (block encoding and Walk operate directly
    in CUDA-Q's basis, unlike the Trotter/QDRIFT kernels above which are bridged
    back to quiche's basis via `_big_endian`).
    """
    dimension = 1 << n_qubits
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for coefficient, word in terms:
        for column in range(dimension):
            row = column
            phase = complex(coefficient)
            for qubit, label in enumerate(word):
                bit = (column >> qubit) & 1
                if label == "X":
                    row ^= 1 << qubit
                elif label == "Y":
                    row ^= 1 << qubit
                    phase *= 1.0j if bit == 0 else -1.0j
                elif label == "Z":
                    phase *= 1.0 if bit == 0 else -1.0
            matrix[row, column] += phase
    return matrix


@pytest.fixture(scope="module")
def cudaq() -> ModuleType:
    """Get CUDA-Q pinned to the fp64 CPU simulator; skip where cudaq is absent."""
    module = pytest.importorskip("cudaq")
    module.set_target("qpp-cpu")
    yield module
    module.reset_target()


def _run_qpe_on_bitstring(
    cudaq: ModuleType,
    qpe: CudaqKernel,
    bitstring: tuple[int, ...],
    shots_count: int,
) -> list[float]:
    """
    Compose `bitstring_kernel` with a `(data: qview) -> float` QPE kernel.

    The canonical composition of state preparation and QPE: allocate the
    data register once, prep it, then run QPE on it - which decodes the energy
    for that shot in-kernel and returns it directly. `cudaq.run` (not
    `cudaq.sample`) collects `shots_count` independent per-shot energies.
    """
    prep = bitstring_kernel(bitstring)
    n_qubits = len(bitstring)

    @cudaq.kernel
    def run() -> float:
        data = cudaq.qvector(n_qubits)
        prep(data)
        return qpe(data)

    return cudaq.run(run, shots_count=shots_count)


def cudaq_sample(kernel: CudaqKernel, shots: int) -> object:
    """Sample a no-argument kernel with CUDA-Q."""
    import cudaq  # noqa: PLC0415

    return cudaq.sample(kernel, shots_count=shots)


def _run(cudaq: ModuleType, kernel: CudaqKernel) -> NDArray[np.complex128]:
    """Execute a no-argument kernel and return its statevector, big-endian."""
    return _big_endian(np.asarray(cudaq.get_state(kernel)))


def _prepared(
    cudaq: ModuleType, n_qubits: int, state_prep: CudaqKernel
) -> NDArray[np.complex128]:
    """Execute a `(qubits: qview)` state-prep kernel on a fresh all-zero register."""

    @cudaq.kernel
    def entry() -> None:
        """Allocate a fresh register and run the injected preparation."""
        qubits = cudaq.qvector(n_qubits)
        state_prep(qubits)

    return _run(cudaq, entry)


class TestBitstringKernel:
    """Tests for quiche.cudaq.state_prep.bitstring_kernel."""

    @pytest.mark.parametrize(
        ("bitstring", "index"),
        [((0, 0), 0), ((1, 0), 2), ((0, 1), 1), ((1, 1), 3), ((1, 0, 1), 5)],
    )
    def test_prepares_basis_state(
        self, cudaq: ModuleType, bitstring: tuple[int, ...], index: int
    ):
        # bitstring[i] targets qubit i, so (1, 0) lands at big-endian index 2 -
        # this pins both the kernel and the _big_endian bridge together.
        state = _prepared(cudaq, len(bitstring), bitstring_kernel(bitstring))
        np.testing.assert_allclose(state, np.eye(len(state))[index], atol=1e-12)

    def test_hartree_fock_state(self, cudaq: ModuleType, h2: PauliSum, budget: Errors):
        hf = HartreeFockState.closed_shell(num_electrons=2, num_spin_orbitals=4)
        spec = QPESpec(
            hamiltonian=ElectronicHamiltonian(
                num_electrons=2, mapping=Mapping.JordanWigner, paulis=h2
            ),
            state_prep=hf,
            algorithm=PhaseEstimation.Textbook,
            simulation=Simulation.Trotter,
            error_budget=budget,
        )

        state = _prepared(cudaq, 4, bitstring_kernel(spec._get_stateprep_bitstring()))

        bitstring = get_jw_state(hf.occupation)
        index = int("".join(str(int(bit)) for bit in bitstring), 2)
        assert np.argmax(np.abs(state)) == index


@pytest.mark.usefixtures("cudaq")
class TestEvolutionTerms:
    """
    Host-side term construction for the CUDA-Q Trotter/QDRIFT kernels.

    No kernel is compiled here, but the `cudaq` fixture is still required class-wide:
    `cudaq_algorithms.trotter.Trotter` itself needs `cudaq` importable, and the fixture
    is how these tests skip cleanly where it isn't installed.
    """

    def test_trotter_uses_cudaq_word_order(self):
        assert trotter_evolution(GENERAL).words == ["XI", "IZ", "XZ"]

    def test_register_width_is_pinned(self):
        assert trotter_evolution(NARROW, n_qubits=3).num_qubits == 3

    @pytest.mark.parametrize("hamiltonian", [COMMUTING, GENERAL])
    def test_identity_coefficient_is_carried(self, hamiltonian: PauliSum):
        for evolution in (
            trotter_evolution(hamiltonian),
            qdrift_evolution(hamiltonian, reps=5, seed=7),
        ):
            assert evolution.identity_coefficient == pytest.approx(
                hamiltonian.identity_coefficient
            )

    def test_qdrift_coefficients_are_uniform_magnitude(self):
        reps = 12
        evolution = qdrift_evolution(GENERAL, reps, seed=7)
        assert len(evolution.words) == reps
        assert all(
            abs(c) == pytest.approx(GENERAL_LAM / reps) for c in evolution.coefficients
        )

    def test_qdrift_preserves_duplicate_words(self):
        # A one-term Hamiltonian is sampled `reps` times: this depends on
        # PRESERVE_INPUT keeping repeats, which a mapping input would not.
        assert qdrift_evolution(SINGLE, reps=4, seed=1).words == ["XZ"] * 4

    def test_qdrift_seed_matches_the_qualtran_backend(self):
        bloq = QDRIFT(sum=GENERAL, time=TIME, num_samples=6, seed=7)
        expected = [
            GENERAL.words[i].to_dense_string(2, big_endian=True)
            for i in bloq.sampled_indices
        ]
        assert qdrift_evolution(GENERAL, reps=6, seed=7).words == expected

    def test_qdrift_requires_positive_reps(self):
        with pytest.raises(ValueError, match="positive number of repetitions"):
            qdrift_evolution(GENERAL, reps=0)

    def test_trotter_rejects_unsupported_order(self):
        with pytest.raises(ValueError, match="order"):
            trotter_kernel(GENERAL, TIME, reps=1, order=3)


@pytest.mark.usefixtures("cudaq")
class TestQubitisedEncoding:
    """Host-side PauliLCU construction for the CUDA-Q backend."""

    def test_alpha_includes_identity(self):
        # Unlike Trotter, PauliLCU folds the identity coefficient into alpha
        # rather than dropping it - there is no unrepresentable global phase
        # here, so `qubitised_kernel` (unlike `trotter_kernel`/`qdrift_kernel`)
        # needs no separate identity-phase correction.
        encoding = qubitised_encoding(GENERAL)
        assert encoding.alpha == pytest.approx(GENERAL.lam)

    def test_num_ancilla(self):
        # GENERAL has 3 non-identity terms plus a nonzero identity coefficient,
        # so PauliLCU retains 4 terms: (4 - 1).bit_length() = 2.
        #
        # (This matches get_qubitisation_ancillas's
        # ceil(log2(num_words_with_identity)), which counts the identity too.)
        assert qubitised_encoding(GENERAL).num_ancilla == 2

    def test_register_width_is_pinned(self):
        assert qubitised_encoding(NARROW, n_qubits=3).num_system == 3


class TestSimulationKernels:
    """Real CUDA-Q output on qpp-cpu, checked against scipy expm."""

    @pytest.mark.parametrize("order", [1, 2, 4])
    def test_trotter_exact_for_commuting_hamiltonian(
        self, cudaq: ModuleType, order: int
    ):
        # Commuting terms make every Trotter order exact, so this pins the
        # (order, steps, exp_pauli-sign) wiring without a convergence tolerance.
        kernel = trotter_kernel(COMMUTING, TIME, reps=3, order=order)
        state = _run(cudaq, kernel) * _identity_phase(COMMUTING, TIME)
        expected = _exact(COMMUTING, TIME, ZERO_KET)
        np.testing.assert_allclose(state, expected, atol=1e-9)

    def test_trotter_converges_to_exact(self, cudaq: ModuleType):
        kernel = trotter_kernel(GENERAL, TIME, reps=50, order=2)
        state = _run(cudaq, kernel) * _identity_phase(GENERAL, TIME)
        expected = _exact(GENERAL, TIME, ZERO_KET)
        np.testing.assert_allclose(state, expected, atol=1e-3)

    def test_qdrift_exact_for_single_term_hamiltonian(self, cudaq: ModuleType):
        # `reps` rotations of magnitude lam * t / reps compose to |h_0| t, so a
        # one-term QDRIFT is exact - this pins the lam / reps normalisation
        # end to end, deterministically and with no statistical tolerance.
        kernel = qdrift_kernel(SINGLE, TIME, reps=5, seed=3)
        state = _run(cudaq, kernel)
        expected = _exact(SINGLE, TIME, ZERO_KET)
        np.testing.assert_allclose(state, expected, atol=1e-9)

    def test_qdrift_matches_its_sampled_product_formula(self, cudaq: ModuleType):
        # Not a statistical check: the seed fixes the sequence, so the circuit
        # must reproduce the host-side product exactly. No identity phase here -
        # unlike the exact/converges tests above, this reference is the sampled
        # rotations only, which is exactly what the circuit itself applies.
        reps, seed = 8, 11
        state = _run(cudaq, qdrift_kernel(GENERAL, TIME, reps, seed=seed))

        expected = ZERO_KET
        for index in sample_qdrift_indices(GENERAL, reps, seed):
            angle = copysign(GENERAL_LAM * TIME / reps, GENERAL.coefficients[index])
            pauli = GENERAL.words[index].to_matrix(2, ignore_idle_qubits=False)
            expected = expm(-1j * angle * pauli) @ expected

        np.testing.assert_allclose(state, expected, atol=1e-9)

    def test_state_prep_is_injected_uncontrolled(self, cudaq: ModuleType):
        ket = np.array([0, 0, 0, 1], dtype=complex)  # |11>, big-endian
        kernel = trotter_kernel(
            COMMUTING, TIME, reps=1, order=2, state_prep=bitstring_kernel((1, 1))
        )
        state = _run(cudaq, kernel) * _identity_phase(COMMUTING, TIME)
        np.testing.assert_allclose(state, _exact(COMMUTING, TIME, ket), atol=1e-9)

    def test_qubitised_not_implemented(self):
        # Dispatched before any cudaq import, so this runs without cudaq installed.
        with pytest.raises(NotImplementedError, match="Qubitised"):
            simulation_kernel(Simulation.Qubitised, GENERAL, TIME, 1)


class TestQubitisedKernels:
    """
    Real CUDA-Q output on qpp-cpu, checked against dense references.

    Every check here is exact (no approximation, no convergence tolerance to
    reason about) - block encoding and qubitisation are exact constructions,
    unlike Trotter/QDRIFT above. Works directly in CUDA-Q's own qubit order via
    `_dense_matrix`/`_reference_pairs` and `cudaq_algorithms.sim_utils`, rather
    than bridging through `_big_endian` - the system+ancilla register layout
    makes that bridge more trouble than it's worth for these checks (see
    `_dense_matrix`'s docstring).
    """

    @pytest.mark.usefixtures("cudaq")
    def test_action_matches_dense_hamiltonian(self):
        from cudaq_algorithms import sim_utils  # noqa: PLC0415

        ket = np.array([0.6, 0.0, 0.0, 0.8], dtype=complex)
        encoding = qubitised_encoding(GENERAL)

        action = sim_utils.action(encoding, ket)

        dense = _dense_matrix(_reference_pairs(GENERAL, 2), 2)
        expected = (dense @ ket) / encoding.alpha
        np.testing.assert_allclose(action, expected, atol=1e-9)

    @pytest.mark.usefixtures("cudaq")
    def test_moments_match_dense_chebyshev(self):
        ket = np.array([0.6, 0.0, 0.0, 0.8], dtype=complex)
        encoding = qubitised_encoding(GENERAL)
        walk = qubitised_walk(GENERAL)

        count = 5
        measured = walk.moments(ket, count)

        scaled = _dense_matrix(_reference_pairs(GENERAL, 2), 2) / encoding.alpha
        chebyshev = [np.eye(4, dtype=complex), scaled]
        while len(chebyshev) < count:
            chebyshev.append(2.0 * scaled @ chebyshev[-1] - chebyshev[-2])
        expected = [
            float(np.real(ket.conj() @ chebyshev[k] @ ket)) for k in range(count)
        ]
        np.testing.assert_allclose(measured, expected, atol=1e-9)

    @pytest.mark.parametrize("power", [1, 2, 3])
    def test_roundtrip_is_identity(self, cudaq: ModuleType, power: int):
        # PREPARE, W^power, (W^power)^-1, UNPREPARE == identity on the whole
        # register (system back to `ket`, ancillas back to |0...0>).
        from cudaq_algorithms import sim_utils  # noqa: PLC0415

        ket = np.array([0.6, 0.0, 0.0, 0.8], dtype=complex)
        encoding = qubitised_encoding(GENERAL)
        walk = qubitised_walk(GENERAL)

        kernel = walk.roundtrip_kernel(power=power)
        state = np.asarray(cudaq.get_state(kernel, sim_utils.state_from(ket)))

        n_register = encoding.num_system + encoding.num_ancilla
        expected = np.zeros(1 << n_register, dtype=complex)
        expected[: len(ket)] = ket
        np.testing.assert_allclose(state, expected, atol=1e-9)

    def test_controlled_off_is_identity(self, cudaq: ModuleType):
        # control_state=0 must reduce the controlled walk to the identity -
        # documented directly on Walk.controlled_kernel.
        from cudaq_algorithms import sim_utils  # noqa: PLC0415

        ket = np.array([0.6, 0.0, 0.0, 0.8], dtype=complex)
        encoding = qubitised_encoding(GENERAL)

        kernel = qubitised_controlled_kernel(GENERAL, power=2, control_state=0)
        state = np.asarray(cudaq.get_state(kernel, sim_utils.state_from(ket)))

        n_control_and_ancilla = 1 + encoding.num_ancilla
        n_register = encoding.num_system + n_control_and_ancilla
        expected = np.zeros(1 << n_register, dtype=complex)
        expected[: len(ket)] = ket
        np.testing.assert_allclose(state, expected, atol=1e-9)

    def test_state_prep_is_injected(self, cudaq: ModuleType):
        from cudaq_algorithms import sim_utils  # noqa: PLC0415

        ket = np.array([0, 0, 0, 1], dtype=complex)  # |11>, CUDA-Q basis
        prep = bitstring_kernel((1, 1))

        kernel = qubitised_kernel(GENERAL, power=2, state_prep=prep)
        injected = np.asarray(cudaq.get_state(kernel))

        kernel = qubitised_kernel(GENERAL, power=2)
        direct = np.asarray(cudaq.get_state(kernel, sim_utils.state_from(ket)))

        np.testing.assert_allclose(injected, direct, atol=1e-9)


class TestInverseQFT:
    """Tests for quiche.cudaq.estimation.inverse_qft_kernel."""

    def test_matches_standard_iqft_matrix(self, cudaq: ModuleType):
        # Built as its own unitary (one basis input per column) and compared to
        # the closed-form (1/sqrt(N)) exp(-2*pi*i*x*y/N) directly in CUDA-Q's own
        # qubit order (qubit i = bit i) - no bit-reversal of the output needed,
        # unlike some textbook-diagram conventions.
        n = 3
        dim = 1 << n
        inverse_qft = inverse_qft_kernel(n)

        @cudaq.kernel
        def prep_and_iqft(target: int) -> None:
            """Prepare basis state |target> then apply the inverse QFT."""
            qubits = cudaq.qvector(n)
            for i in range(n):
                if ((target >> i) & 1) == 1:
                    x(qubits[i])  # noqa: F821 -- CUDA-Q intrinsic, not a real Python name
            inverse_qft(qubits)

        matrix = np.zeros((dim, dim), dtype=complex)
        for target in range(dim):
            matrix[:, target] = np.asarray(cudaq.get_state(prep_and_iqft, target))

        row, col = np.meshgrid(np.arange(dim), np.arange(dim), indexing="ij")
        expected = np.exp(-2j * np.pi * col * row / dim) / np.sqrt(dim)
        np.testing.assert_allclose(matrix, expected, atol=1e-9)

    def test_angles_are_not_computed_in_kernel(self, cudaq: ModuleType):
        # CUDA-Q Logical's Quake import rejects in-kernel powers such as
        # `math.fpowi`, so the rotation angles must arrive precomputed.
        inverse_qft = inverse_qft_kernel(4)

        @cudaq.kernel
        def iqft() -> None:
            """Apply the inverse QFT to a fresh register."""
            inverse_qft(cudaq.qvector(4))

        quake = str(iqft) + str(inverse_qft)
        assert "fpowi" not in quake
        assert "powf" not in quake


class TestTextbookQPEKernel:
    """Real CUDA-Q output on qpp-cpu: Textbook QPE for Trotter and QDRIFT."""

    @pytest.mark.parametrize("method", [Simulation.Trotter, Simulation.QDRIFT])
    def test_exact_phase_for_single_term_eigenstate(
        self, cudaq: ModuleType, method: Simulation
    ):
        # H = Z, eigenstate |1>: true eigenvalue -1. `time`/`target` are chosen
        # so the true phase lands exactly on a representable bin, making the
        # in-kernel decode exactly recover the eigenvalue, not just close to it.
        h = _pauli_sum((1.0, "Z"))
        num_qpe_ancillas = 6
        target = 5
        time = 2 * np.pi * target / (1 << num_qpe_ancillas)

        qpe = textbook_qpe_kernel(h, method, num_qpe_ancillas, time, reps=1)
        energies = _run_qpe_on_bitstring(cudaq, qpe, (1,), shots_count=20)

        np.testing.assert_allclose(energies, -1.0, atol=1e-9)

    @pytest.mark.parametrize("method", [Simulation.Trotter, Simulation.QDRIFT])
    def test_identity_coefficient_shifts_phase(
        self, cudaq: ModuleType, method: Simulation
    ):
        # H = Z + 0.3*I, eigenstate |1>: true eigenvalue -1 + 0.3 = -0.7. This
        # is the test that would have caught the identity term being
        # double-counted (once via the in-circuit r1 correction, once via a
        # separate addition after decoding) - comparing against the true
        # eigenvalue directly catches that; comparing only the raw measured
        # bitstring against a self-referential target (as an earlier version of
        # this test did) would not have.
        h = _pauli_sum((1.0, "Z"), identity=0.3)
        num_qpe_ancillas = 6
        target = 11
        time = 2 * np.pi * target / ((1 << num_qpe_ancillas) * (1.0 - 0.3))

        qpe = textbook_qpe_kernel(h, method, num_qpe_ancillas, time, reps=1)
        energies = _run_qpe_on_bitstring(cudaq, qpe, (1,), shots_count=20)

        np.testing.assert_allclose(energies, -0.7, atol=1e-9)

    def test_converges_for_noncommuting_hamiltonian(self, cudaq: ModuleType):
        from cudaq_algorithms import sim_utils  # noqa: PLC0415

        # Non-exact case: pins genuine Trotter approximation behaviour (hence a
        # tolerance, not an exact equality) rather than the wiring above. The
        # eigenvector isn't a computational basis state, so (unlike the two
        # exact tests above) it needs a `cudaq.State`-based entry kernel rather
        # than `bitstring_kernel`.
        h = _pauli_sum((0.6, "X"), (0.4, "Z"))
        eigvals, eigvecs = np.linalg.eigh(h.to_matrix())
        eigenvalue, eigenvector = eigvals[1], eigvecs[:, 1]

        num_qpe_ancillas = 8
        time = 0.5

        qpe = textbook_qpe_kernel(
            h, Simulation.Trotter, num_qpe_ancillas, time, reps=4, order=2
        )

        @cudaq.kernel
        def run(state: cudaq.State) -> float:
            """Load an externally-computed state, then run QPE on it."""
            data = cudaq.qvector(state)
            return qpe(data)

        initial_state = sim_utils.state_from(eigenvector.astype(complex))
        energies = cudaq.run(run, initial_state, shots_count=200)

        # One QPE bin, converted from phase-cycles to energy units.
        bin_width = (2 * np.pi / time) / (1 << num_qpe_ancillas)
        assert abs(statistics.mode(energies) - eigenvalue) < bin_width

    def test_qubitised_not_implemented(self):
        # Dispatched before any cudaq import, so this runs without cudaq installed.
        with pytest.raises(NotImplementedError, match="Qubitised"):
            textbook_qpe_kernel(GENERAL, Simulation.Qubitised, 4, TIME, reps=1)


class TestQubitisedQPEKernel:
    """
    Real CUDA-Q output on qpp-cpu: Textbook QPE via the qubitisation walk.

    Exact-comparison style throughout (like `TestQubitisedKernels`), not
    Trotter/QDRIFT's convergence-tolerance style - block encoding is exact, no
    Trotter-order approximation to reason about. All cases use `num_qpe_ancillas`
    in the 3-5 range rather than commit 3's 6: `controlled_walk_step_kernel` is
    heavier per application than a single `apply_trotter` call, and the ladder
    still costs `2**n - 1` total steps.
    """

    def test_exact_phase_no_identity(self, cudaq: ModuleType):
        # H = Z(0) - Z(1), eigenstate |11>: both terms diagonal/commuting, so
        # eigenvalue is exactly 0. alpha=2, cos(theta)=-0/2=0, theta=pi/2,
        # phase=0.25 - exact for any num_qpe_ancillas >= 2.
        h = _pauli_sum((1.0, "ZI"), (-1.0, "IZ"))
        qpe = qubitised_qpe_kernel(h, num_qpe_ancillas=3)
        energies = _run_qpe_on_bitstring(cudaq, qpe, (1, 1), shots_count=20)
        np.testing.assert_allclose(energies, 0.0, atol=1e-9)

    def test_exact_phase_with_identity(self, cudaq: ModuleType):
        # H = Z + I, eigenstate |1>: eigenvalue -1+1=0, independently known
        # (not self-referential). alpha=2 (test_alpha_includes_identity's own
        # formula), same theta=pi/2 exact-bin construction as above - this is
        # the test shaped to catch an identity double-count, the way commit
        # 3's analogous Trotter bug was only caught by comparing to a real
        # eigenvalue rather than a self-referential target.
        h = _pauli_sum((1.0, "Z"), identity=1.0)
        qpe = qubitised_qpe_kernel(h, num_qpe_ancillas=3)
        energies = _run_qpe_on_bitstring(cudaq, qpe, (1,), shots_count=20)
        np.testing.assert_allclose(energies, 0.0, atol=1e-9)

    def test_exact_phase_pi(self, cudaq: ModuleType):
        # H = Z, eigenstate |0>: eigenvalue +1, alpha=1, cos(theta)=-1,
        # theta=pi, phase=0.5 - exact for any num_qpe_ancillas >= 1, and
        # exercises the 2**k ladder's parity: only rung 0 contributes a
        # nontrivial kickback at theta=pi (rungs k>=1 apply W an even number
        # of times, W^2=I here) - a broken "apply once per rung" ladder would
        # get this wrong for k>=1, not rung 0.
        h = _pauli_sum((1.0, "Z"))
        qpe = qubitised_qpe_kernel(h, num_qpe_ancillas=3)
        energies = _run_qpe_on_bitstring(cudaq, qpe, (0,), shots_count=20)
        np.testing.assert_allclose(energies, 1.0, atol=1e-9)

    def test_converges_for_noncommuting_hamiltonian(self, cudaq: ModuleType):
        from cudaq_algorithms import sim_utils  # noqa: PLC0415

        h = _pauli_sum((0.6, "X"), (0.4, "Z"))
        eigvals, eigvecs = np.linalg.eigh(h.to_matrix())
        eigenvalue, eigenvector = eigvals[1], eigvecs[:, 1]

        num_qpe_ancillas = 5
        qpe = qubitised_qpe_kernel(h, num_qpe_ancillas)

        @cudaq.kernel
        def run(state: cudaq.State) -> float:
            """Load an externally-computed state, then run QPE on it."""
            data = cudaq.qvector(state)
            return qpe(data)

        initial_state = sim_utils.state_from(eigenvector.astype(complex))
        energies = cudaq.run(run, initial_state, shots_count=200)

        # Max slope of d(-alpha*cos(theta))/dtheta, converted to a bin width
        # in energy units - unlike textbook_qpe_kernel's linear decode, the
        # bin width here depends on where on the cosine curve the true phase
        # lands, so this bound is a representative upper bound, not exact.
        alpha = qubitised_encoding(h).alpha
        max_slope = alpha * (2 * np.pi / (1 << num_qpe_ancillas))
        assert abs(statistics.mode(energies) - eigenvalue) < max_slope


# A fixed non-eigenstate ket, RY(THETA)|0> = 0.8|0> + 0.6|1>, shared by the
# Naive QPE tests below - both re/im parts of <psi|U|psi> are nontrivial here,
# unlike an eigenstate (which would give a trivial re/im split).
THETA = 2 * np.arccos(0.8)


def _prep_theta_kernel(cudaq: ModuleType) -> CudaqKernel:
    """Build a `(qubits: qview)` kernel preparing RY(THETA)|0> on one qubit."""

    @cudaq.kernel
    def prep(qubits: cudaq.qview) -> None:
        ry(THETA, qubits[0])  # noqa: F821 -- CUDA-Q intrinsic, not a real Python name

    return prep


class TestNaiveQPEKernel:
    """
    Real CUDA-Q output on qpp-cpu: Naive (Hadamard-test) QPE for Trotter/QDRIFT.

    Unlike Textbook/Qubitised/Iterative, a single shot carries no phase
    information - the caller must average many shots to estimate
    Re/Im(<psi|U|psi>), so this is the one place in the whole CUDA-Q backend
    where a *statistical* tolerance is unavoidable, not a design choice.
    """

    @pytest.mark.parametrize("method", [Simulation.Trotter, Simulation.QDRIFT])
    @pytest.mark.parametrize("mode", ["re", "im"])
    def test_matches_exact_expectation_value(
        self, cudaq: ModuleType, method: Simulation, mode: str
    ):
        h = _pauli_sum((0.6, "X"), (0.4, "Z"), identity=0.2)
        time = 0.5
        shots_count = 4000

        reps = 30 if method is Simulation.QDRIFT else 5
        qpe = naive_qpe_kernel(h, method, time, reps, mode=mode, seed=7)

        prep = _prep_theta_kernel(cudaq)

        @cudaq.kernel
        def run() -> float:
            data = cudaq.qvector(1)
            prep(data)
            return qpe(data)

        measured = statistics.mean(cudaq.run(run, shots_count=shots_count))

        ket = np.array([np.cos(THETA / 2), np.sin(THETA / 2)], dtype=complex)
        expectation = ket.conj() @ expm(-1j * time * h.to_matrix()) @ ket
        expected = expectation.real if mode == "re" else expectation.imag

        # Shot-noise tolerance for a +-1-valued estimator: std <= 1, so SEM =
        # 1/sqrt(n); a 6-sigma margin keeps this from being flaky.
        tolerance = 6.0 / np.sqrt(shots_count)
        assert abs(measured - expected) < tolerance

    def test_qubitised_not_implemented(self):
        # Dispatched before any cudaq import, so this runs without cudaq installed.
        with pytest.raises(NotImplementedError, match="Qubitised"):
            naive_qpe_kernel(GENERAL, Simulation.Qubitised, TIME, reps=1)


class TestQubitisedNaiveQPEKernel:
    """Real CUDA-Q output on qpp-cpu: Naive QPE via the qubitisation walk."""

    @pytest.mark.parametrize("mode", ["re", "im"])
    def test_matches_walk_expectation_value(self, cudaq: ModuleType, mode: str):
        from cudaq_algorithms import sim_utils  # noqa: PLC0415

        h = _pauli_sum((0.6, "X"), (0.4, "Z"))
        shots_count = 4000
        ket = np.array([np.cos(THETA / 2), np.sin(THETA / 2)], dtype=complex)

        prep = _prep_theta_kernel(cudaq)
        qpe = qubitised_naive_qpe_kernel(h, mode=mode)

        @cudaq.kernel
        def run() -> float:
            data = cudaq.qvector(1)
            prep(data)
            return qpe(data)

        measured = statistics.mean(cudaq.run(run, shots_count=shots_count))

        # Reference: <ket|W|ket>, via the walk's own uncontrolled kernel,
        # projected onto the ancilla=|0...0> block (the leading len(ket)
        # amplitudes) - the same construction validated during this
        # commit's own research, not a from-scratch dense-matrix rebuild.
        walk = qubitised_walk(h)
        out_state = np.asarray(
            cudaq.get_state(walk.kernel(power=1), sim_utils.state_from(ket))
        )
        expectation = np.vdot(ket, out_state[: len(ket)])
        expected = expectation.real if mode == "re" else expectation.imag

        tolerance = 6.0 / np.sqrt(shots_count)
        assert abs(measured - expected) < tolerance


class TestIterativeQPEKernel:
    """
    Real CUDA-Q output on qpp-cpu: Iterative QPE for Trotter and QDRIFT.

    Same exact-comparison style as `TestTextbookQPEKernel` - IPEA extracts
    bits deterministically given an eigenstate, so these should be exact,
    not tolerance-based.
    """

    @pytest.mark.parametrize("method", [Simulation.Trotter, Simulation.QDRIFT])
    def test_exact_phase_for_single_term_eigenstate(
        self, cudaq: ModuleType, method: Simulation
    ):
        # H = Z, eigenstate |1>: true eigenvalue -1. num_rounds/target chosen
        # so the true phase lands exactly on a representable bin.
        h = _pauli_sum((1.0, "Z"))
        num_rounds = 6
        target = 5
        time = 2 * np.pi * target / (1 << num_rounds)

        qpe = iterative_qpe_kernel(h, method, num_rounds, time, reps=1)
        energies = _run_qpe_on_bitstring(cudaq, qpe, (1,), shots_count=20)

        np.testing.assert_allclose(energies, -1.0, atol=1e-9)

    @pytest.mark.parametrize("method", [Simulation.Trotter, Simulation.QDRIFT])
    def test_identity_coefficient_shifts_phase(
        self, cudaq: ModuleType, method: Simulation
    ):
        # H = Z + 0.3*I, eigenstate |1>: true eigenvalue -1+0.3=-0.7 - the test
        # shaped to catch an identity double-count (the combined
        # feedback+identity r1 correction), the way the analogous
        # textbook_qpe_kernel bug was only caught by comparing to a true
        # eigenvalue rather than a self-referential target.
        h = _pauli_sum((1.0, "Z"), identity=0.3)
        num_rounds = 6
        target = 11
        time = 2 * np.pi * target / ((1 << num_rounds) * (1.0 - 0.3))

        qpe = iterative_qpe_kernel(h, method, num_rounds, time, reps=1)
        energies = _run_qpe_on_bitstring(cudaq, qpe, (1,), shots_count=20)

        np.testing.assert_allclose(energies, -0.7, atol=1e-9)

    def test_converges_for_noncommuting_hamiltonian(self, cudaq: ModuleType):
        from cudaq_algorithms import sim_utils  # noqa: PLC0415

        h = _pauli_sum((0.6, "X"), (0.4, "Z"))
        eigvals, eigvecs = np.linalg.eigh(h.to_matrix())
        eigenvalue, eigenvector = eigvals[1], eigvecs[:, 1]

        num_rounds = 8
        time = 0.5
        qpe = iterative_qpe_kernel(
            h, Simulation.Trotter, num_rounds, time, reps=4, order=2
        )

        @cudaq.kernel
        def run(state: cudaq.State) -> float:
            """Load an externally-computed state, then run QPE on it."""
            data = cudaq.qvector(state)
            return qpe(data)

        initial_state = sim_utils.state_from(eigenvector.astype(complex))
        energies = cudaq.run(run, initial_state, shots_count=200)

        bin_width = (2 * np.pi / time) / (1 << num_rounds)
        assert abs(statistics.mode(energies) - eigenvalue) < bin_width

    def test_qubitised_not_implemented(self):
        # Dispatched before any cudaq import, so this runs without cudaq installed.
        with pytest.raises(NotImplementedError, match="Qubitised"):
            iterative_qpe_kernel(GENERAL, Simulation.Qubitised, 4, TIME, reps=1)


# H = Z with time and ancillas placing both eigenvalues exactly on a bin.
EXACT_NUM_QPE_ANCILLAS = 6
EXACT_TIME = 2 * np.pi * 5 / (1 << EXACT_NUM_QPE_ANCILLAS)


def _plus_kernel(cudaq: ModuleType) -> CudaqKernel:
    """Build a `(qubits: qview)` kernel preparing |+> on one qubit."""

    @cudaq.kernel
    def plus(qubits: cudaq.qview) -> None:
        h(qubits[0])  # noqa: F821 -- CUDA-Q intrinsic, not a real Python name

    return plus


class TestRepeatedMinimum:
    """`repeated_minimum_kernel` runs the repeat-and-take-minimum protocol in-kernel."""

    @staticmethod
    def _repeated(state_prep: CudaqKernel, repetitions: int) -> CudaqKernel:
        core, work_qubits = textbook_qpe_core(
            _pauli_sum((1.0, "Z")),
            Simulation.Trotter,
            EXACT_NUM_QPE_ANCILLAS,
            EXACT_TIME,
            reps=1,
        )
        return repeated_minimum_kernel(core, work_qubits, state_prep, 1, repetitions)

    def test_minimum_finds_ground_state(self, cudaq: ModuleType):
        # |+> has overlap 1/2 with the ground state |1> (E = -1) and 1/2 with |0>
        # (E = +1), so a single run reads +1 half the time; the minimum of 20 runs
        # reads +1 only with probability 2^-20.
        repeated = self._repeated(_plus_kernel(cudaq), repetitions=20)
        energies = cudaq.run(repeated, shots_count=10)

        np.testing.assert_allclose(energies, -1.0, atol=1e-9)

    def test_single_repetition_matches_single_run(self, cudaq: ModuleType):
        repeated = self._repeated(bitstring_kernel((0,)), repetitions=1)
        energies = cudaq.run(repeated, shots_count=5)

        np.testing.assert_allclose(energies, 1.0, atol=1e-9)


class TestCircuitAndPost:
    """`measured_circuit_kernel` sampled once, decoded by `sample_post_processor`."""

    @staticmethod
    def _sample(  # noqa: PLR0913
        circuit: tuple[CudaqKernel, int, object],
        num_qpe_ancillas: int,
        state_prep: CudaqKernel,
        shots: int,
        *,
        num_data: int = 1,
        repetitions: int = 1,
    ) -> list[float]:
        kernel, work_qubits, decode = circuit
        measured = measured_circuit_kernel(
            kernel,
            work_qubits,
            num_qpe_ancillas,
            state_prep=state_prep,
            num_data=num_data,
        )
        post = sample_post_processor(decode, repetitions)
        return post(cudaq_sample(measured, shots * repetitions))

    @staticmethod
    def _plus_circuit() -> tuple[CudaqKernel, int, object]:
        return textbook_qpe_circuit(
            _pauli_sum((1.0, "Z")),
            Simulation.Trotter,
            EXACT_NUM_QPE_ANCILLAS,
            EXACT_TIME,
            reps=1,
        )

    @pytest.mark.parametrize("method", [Simulation.Trotter, Simulation.QDRIFT])
    @pytest.mark.parametrize(
        ("identity", "target", "expected"), [(0.0, 5, -1.0), (0.3, 11, -0.7)]
    )
    def test_exact_eigenstate(
        self,
        cudaq: ModuleType,  # noqa: ARG002 -- skips without cudaq
        method: Simulation,
        identity: float,
        target: int,
        expected: float,
    ):
        # Same exact-bin construction as TestTextbookQPEKernel: the readout `target`
        # is not a palindrome in 6 bits, so this also pins the bit order.
        h = _pauli_sum((1.0, "Z"), identity=identity)
        time = 2 * np.pi * target / ((1 << 6) * (1.0 - identity))
        circuit = textbook_qpe_circuit(h, method, 6, time, reps=1)
        energies = self._sample(circuit, 6, bitstring_kernel((1,)), shots=20)
        np.testing.assert_allclose(energies, expected, atol=1e-9)

    @pytest.mark.parametrize(
        ("terms", "identity", "bits", "expected"),
        [
            (((1.0, "ZI"), (-1.0, "IZ")), 0.0, (1, 1), 0.0),
            (((1.0, "Z"),), 1.0, (1,), 0.0),
            (((1.0, "Z"),), 0.0, (0,), 1.0),
        ],
    )
    def test_qubitised_exact_eigenstate(
        self,
        cudaq: ModuleType,  # noqa: ARG002 -- skips without cudaq
        terms: tuple[tuple[float, str], ...],
        identity: float,
        bits: tuple[int, ...],
        expected: float,
    ):
        # The cases of TestQubitisedQPEKernel, via the sampled circuit.
        h = _pauli_sum(*terms, identity=identity)
        circuit = qubitised_qpe_circuit(h, 3)
        energies = self._sample(
            circuit, 3, bitstring_kernel(bits), shots=20, num_data=len(bits)
        )
        np.testing.assert_allclose(energies, expected, atol=1e-9)

    def test_statistics_match_overlap(self, cudaq: ModuleType):
        # |+> reads the ground state (E = -1) or |0> (E = +1), each with probability ½.
        shots = 400
        energies = np.array(
            self._sample(
                self._plus_circuit(),
                EXACT_NUM_QPE_ANCILLAS,
                _plus_kernel(cudaq),
                shots,
            )
        )
        assert set(np.round(energies, 9)) <= {-1.0, 1.0}
        assert abs(np.mean(energies == -1.0) - 0.5) < 6 * 0.5 / np.sqrt(shots)

    def test_minimum_over_repetitions(self, cudaq: ModuleType):
        energies = self._sample(
            self._plus_circuit(),
            EXACT_NUM_QPE_ANCILLAS,
            _plus_kernel(cudaq),
            shots=10,
            repetitions=20,
        )
        assert len(energies) == 10
        np.testing.assert_allclose(energies, -1.0, atol=1e-9)

    def test_post_rejects_partial_block(self):
        _, _, decode = self._plus_circuit()
        post = sample_post_processor(decode, 3)
        with pytest.raises(ValueError, match="multiple of repetitions=3"):
            post(["000000"] * 4)

    def test_post_decodes_bitstrings(self):
        # Character k is ancilla k, weighted 2^k: "101000" reads y = 5.
        _, _, decode = self._plus_circuit()
        post = sample_post_processor(decode, 1)
        np.testing.assert_allclose(post(["101000"]), -1.0, atol=1e-9)


# Commuting, so Trotterisation is exact. In the |q0 = 0, q1 = 1> eigenstate the energy
# is 0.8 - 0.3 + 0.7 = 1.2, above the non-identity 1-norm 1.1: with t = pi / 1.1 its
# phase would wrap, whereas QPESpec's t = pi / lam (lam includes the identity, 1.8)
# keeps it in range.
WRAP_HAMILTONIAN = _pauli_sum((0.8, "ZI"), (0.3, "IZ"), identity=0.7)
WRAP_ENERGY = 1.2


def _qpe_spec(
    hamiltonian: PauliSum,
    algorithm: PhaseEstimation,
    simulation: Simulation,
    state_prep: HartreeFockState | None = None,
    num_electrons: int = 0,
) -> QPESpec:
    """Build a QPESpec with a loose error budget, to keep the circuits small."""
    return QPESpec(
        hamiltonian=ElectronicHamiltonian(
            num_electrons=num_electrons,
            mapping=Mapping.JordanWigner,
            paulis=hamiltonian,
        ),
        state_prep=state_prep,
        algorithm=algorithm,
        simulation=simulation,
        error_budget=Errors(
            estimation=0.5, simulation=0.5, rotations=0.5, state_prep=0.5, overlap=1
        ),
        seed=7,
    )


class TestQPESpecToCudaq:
    """`QPESpec.to_cudaq()` wiring across implemented (algorithm, simulation) pairs."""

    @pytest.mark.parametrize(
        ("algorithm", "simulation"),
        [
            (PhaseEstimation.Textbook, Simulation.Trotter),
            (PhaseEstimation.Textbook, Simulation.QDRIFT),
            (PhaseEstimation.Textbook, Simulation.Qubitised),
            (PhaseEstimation.Iterative, Simulation.Trotter),
            (PhaseEstimation.Iterative, Simulation.QDRIFT),
        ],
    )
    def test_runs_and_returns_floats(
        self,
        cudaq: ModuleType,
        algorithm: PhaseEstimation,
        simulation: Simulation,
    ):
        spec = _qpe_spec(_pauli_sum((0.6, "X"), (0.4, "Z")), algorithm, simulation)

        energies = cudaq.run(spec.to_cudaq(), shots_count=20)

        assert len(energies) == 20
        assert all(isinstance(energy, float) for energy in energies)

    @pytest.mark.parametrize(
        "algorithm", [PhaseEstimation.Textbook, PhaseEstimation.Iterative]
    )
    def test_default_time_does_not_wrap(
        self, cudaq: ModuleType, algorithm: PhaseEstimation
    ):
        assert WRAP_HAMILTONIAN.without_identity().lam < WRAP_ENERGY
        spec = _qpe_spec(
            WRAP_HAMILTONIAN,
            algorithm,
            Simulation.Trotter,
            state_prep=HartreeFockState(occupation=(0, 1)),
            num_electrons=1,
        )

        energies = cudaq.run(spec.to_cudaq(), shots_count=50)

        num_bits = (
            spec.num_qpe_ancillas
            if algorithm == PhaseEstimation.Textbook
            else spec.num_rounds
        )
        bin_width = (2 * np.pi / spec.time) / (1 << num_bits)
        assert abs(statistics.mode(energies) - WRAP_ENERGY) < bin_width

    def test_circuit_and_post(self, cudaq: ModuleType):  # noqa: ARG002
        spec = _qpe_spec(
            WRAP_HAMILTONIAN,
            PhaseEstimation.Textbook,
            Simulation.Trotter,
            state_prep=HartreeFockState(occupation=(0, 1)),
            num_electrons=1,
        )

        post = spec.to_cudaq(kernel="post")
        energies = post(cudaq_sample(spec.to_cudaq(kernel="circuit"), 50))

        bin_width = (2 * np.pi / spec.time) / (1 << spec.num_qpe_ancillas)
        assert abs(statistics.mode(energies) - WRAP_ENERGY) < bin_width

    def test_unknown_kernel(self):
        spec = _qpe_spec(GENERAL, PhaseEstimation.Textbook, Simulation.Trotter)
        with pytest.raises(ValueError, match="kernel must be"):
            spec.to_cudaq(kernel="everything")  # type: ignore[arg-type]

    @pytest.mark.parametrize("kernel", ["circuit", "post"])
    def test_iterative_has_no_circuit(self, kernel: str):
        spec = _qpe_spec(GENERAL, PhaseEstimation.Iterative, Simulation.Trotter)
        with pytest.raises(ValueError, match="mid-circuit"):
            spec.to_cudaq(kernel=kernel)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("algorithm", "simulation", "match"),
        [
            (PhaseEstimation.Kitaev, Simulation.Trotter, "Kitaev"),
            (PhaseEstimation.Naive, Simulation.Trotter, "Naive"),
            (PhaseEstimation.Iterative, Simulation.Qubitised, "Qubitised"),
        ],
    )
    def test_not_implemented(
        self, algorithm: PhaseEstimation, simulation: Simulation, match: str
    ):
        # Dispatched before any cudaq import, so this runs without cudaq installed.
        spec = _qpe_spec(GENERAL, algorithm, simulation)
        with pytest.raises(NotImplementedError, match=match):
            spec.to_cudaq()


class TestPauliSumFromCudaq:
    """Tests for PauliSum.from_cudaq against the real cudaq.SpinOperator."""

    def test_from_spin_operator(self, cudaq: ModuleType):
        spin = cudaq.spin
        op = 0.5 * spin.z(0) - 0.3 * spin.x(0) * spin.x(1)

        result = PauliSum.from_cudaq(op)

        assert result.num_qubits == 2
        assert result.num_words == 2
        assert result.identity_coefficient == 0.0

    def test_from_spin_operator_with_identity(self, cudaq: ModuleType):
        spin = cudaq.spin
        op = 1.2 * spin.i(0) + 0.5 * spin.z(0)

        result = PauliSum.from_cudaq(op)

        assert result.identity_coefficient == pytest.approx(1.2)
        assert result.num_words == 1

    def test_from_spin_operator_term(self, cudaq: ModuleType):
        term = cudaq.spin.y(1)

        result = PauliSum.from_cudaq(term)

        assert result.num_words == 1
        assert result.words[0].to_dense_string(2, big_endian=True) == "IY"


class TestSoftDependency:
    """cudaq must stay optional; these run whether or not it is installed."""

    def test_import_quiche_does_not_import_cudaq(self):
        # A subprocess, because tests/conftest.py deliberately pre-imports cudaq.
        code = (
            "import quiche, quiche.cudaq, quiche.dispatch, sys; "
            "assert 'cudaq' not in sys.modules; "
            "assert 'cudaq_algorithms' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-c", code], check=False)  # noqa: S603
        assert result.returncode == 0
