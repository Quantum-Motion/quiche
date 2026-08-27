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

import subprocess
import sys
from math import copysign
from types import ModuleType

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.linalg import expm

from quiche.chemistry import HartreeFockState, get_jw_state
from quiche.core import Mapping, PauliSum, PauliWord, Simulation
from quiche.core.qdrift import sample_qdrift_indices
from quiche.cudaq import CudaqKernel
from quiche.cudaq.simulation import (
    qdrift_evolution,
    qdrift_kernel,
    simulation_kernel,
    trotter_evolution,
    trotter_kernel,
)
from quiche.cudaq.state_prep import bitstring_kernel
from quiche.dispatch import HartreeFockSpec, Spec
from quiche.qualtran.bloqs import QDRIFT

TIME = 0.7
ZERO_KET = np.array([1, 0, 0, 0], dtype=complex)  # |00>, big-endian, 2 qubits


def _pauli_sum(*terms: tuple[float, str], identity: float = 0.0) -> PauliSum:
    """Build a PauliSum from (coeff, dense word) pairs; word index == qubit index."""
    return PauliSum(
        coefficients=tuple(c for c, _ in terms),
        terms=tuple(PauliWord.from_str(w, big_endian=True) for _, w in terms),
        identity_coefficient=identity,
    )


# All-Z, so every Trotter order is exact regardless of step count.
COMMUTING = _pauli_sum((0.5, "ZI"), (0.4, "IZ"), (0.3, "ZZ"), identity=0.2)
# Non-commuting, for genuine Trotter/QDRIFT convergence checks.
GENERAL = _pauli_sum((0.7, "XI"), (-0.4, "IZ"), (0.31, "XZ"), identity=-0.2)
SINGLE = _pauli_sum((-0.6, "XZ"))
# One qubit narrower than its register, to exercise explicit `n_qubits` padding.
NARROW = _pauli_sum((1.0, "X"))


def _big_endian(state: NDArray) -> NDArray[np.complex128]:
    """
    Reindex a CUDA-Q statevector into quiche's qubit-0-most-significant order.

    CUDA-Q indexes amplitudes with qubit `b` as bit `b` of the index (qubit 0
    least significant); `PauliSum._to_matrix` krons qubit 0 first (qubit 0 most
    significant). Splitting the flat index into one axis per qubit and reversing
    the axis order is exactly that relabelling, and is its own inverse.
    """
    amplitudes = np.asarray(state, dtype=complex)
    n_qubits = amplitudes.size.bit_length() - 1
    return amplitudes.reshape([2] * n_qubits).transpose().reshape(-1)


def _exact(hamiltonian: PauliSum, time: float, ket: NDArray) -> NDArray[np.complex128]:
    """Compute exp(-i H t)|ket>, in quiche's big-endian basis."""
    return expm(-1j * time * hamiltonian._to_matrix()) @ ket


def _identity_phase(hamiltonian: PauliSum, time: float) -> complex:
    """Get exp(-i c_I t), the global phase tracked by `identity_coefficient`."""
    return complex(np.exp(-1j * hamiltonian.identity_coefficient * time))


@pytest.fixture(scope="module")
def cudaq() -> ModuleType:
    """Get CUDA-Q pinned to the fp64 CPU simulator; skip where cudaq is absent."""
    module = pytest.importorskip("cudaq")
    module.set_target("qpp-cpu")
    yield module
    module.reset_target()


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

    def test_hartree_fock_spec_to_cudaq(self, cudaq: ModuleType):
        hf_state = HartreeFockState.closed_shell(electrons=2, spin_orbitals=4)
        spec = HartreeFockSpec(state=hf_state, mapping=Mapping.JordanWigner)

        state = _prepared(cudaq, 4, spec.to_cudaq())

        bitstring = get_jw_state(hf_state.occupation)
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
            abs(c) == pytest.approx(GENERAL.lam / reps) for c in evolution.coefficients
        )

    def test_qdrift_preserves_duplicate_words(self):
        # A one-term Hamiltonian is sampled `reps` times: this depends on
        # PRESERVE_INPUT keeping repeats, which a mapping input would not.
        assert qdrift_evolution(SINGLE, reps=4, seed=1).words == ["XZ"] * 4

    def test_qdrift_seed_matches_the_qualtran_backend(self):
        bloq = QDRIFT(h=GENERAL, t=TIME, n_terms=6, seed=7)
        expected = [
            GENERAL.terms[i].to_str(2, big_endian=True)
            for i in bloq.sample_term_indices()
        ]
        assert qdrift_evolution(GENERAL, reps=6, seed=7).words == expected

    def test_qdrift_requires_positive_reps(self):
        with pytest.raises(ValueError, match="positive number of repetitions"):
            qdrift_evolution(GENERAL, reps=0)

    def test_trotter_rejects_unsupported_order(self):
        with pytest.raises(ValueError, match="order"):
            trotter_kernel(GENERAL, TIME, reps=1, order=3)


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
            angle = copysign(GENERAL.lam * TIME / reps, GENERAL.coefficients[index])
            pauli = GENERAL.terms[index]._to_matrix(2, ignore_idle_qubits=False)
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
            simulation_kernel(Simulation.Qubitised, GENERAL, TIME, reps=1)


class TestSoftDependency:
    """cudaq must stay optional; these run whether or not it is installed."""

    def test_import_quiche_does_not_import_cudaq(self):
        # A subprocess, because tests/conftest.py deliberately pre-imports cudaq.
        code = (
            "import quiche, sys; "
            "assert 'cudaq' not in sys.modules; "
            "assert 'cudaq_algorithms' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-c", code], check=False)  # noqa: S603
        assert result.returncode == 0

    def test_spec_requires_to_cudaq(self):
        class Incomplete(Spec):
            def to_qualtran(self) -> None: ...
            def to_quest(self) -> None: ...

        with pytest.raises(TypeError, match="to_cudaq"):
            Incomplete()
