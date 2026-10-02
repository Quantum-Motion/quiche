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


"""Numerical checks that the error bounds in `quiche.budget` really are bounds."""

from math import pi

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.linalg import expm

from quiche import estimation as est
from quiche import simulation as sim
from quiche.core import PauliSum, PauliWord

# Fourth-order splitting weights: Suzuki's recursion (QuEST, Qualtran) and the
# Forest-Ruth formula (CUDA-Q Algorithms).
SUZUKI_P = 1 / (4 - 4 ** (1 / 3))
FOREST_RUTH_W1 = 1 / (2 - 2 ** (1 / 3))
FOREST_RUTH_W0 = 1 - 2 * FOREST_RUTH_W1


def _random_hamiltonian(seed: int, n_qubits: int = 2, n_terms: int = 5) -> PauliSum:
    """Build a random non-commuting PauliSum with an identity term."""
    rng = np.random.default_rng(seed)
    words = set()
    while len(words) < n_terms:
        word = "".join(rng.choice(list("IXYZ"), n_qubits))
        if word != "I" * n_qubits:
            words.add(word)
    return PauliSum(
        coefficients=tuple(rng.uniform(-1, 1, n_terms)),
        terms=tuple(PauliWord.from_str(w, big_endian=True) for w in sorted(words)),
        identity_coefficient=float(rng.uniform(-1, 1)),
    )


def _term_matrices(h: PauliSum) -> list[NDArray[np.complex128]]:
    """Get each non-identity term `h_j P_j` as a dense matrix."""
    return [
        c * term._to_matrix(h.n_qubits, ignore_idle_qubits=False)
        for c, term in zip(h.coefficients, h.terms, strict=True)
    ]


def _first_order(terms: list[NDArray], dt: float) -> NDArray:
    step = np.eye(len(terms[0]), dtype=complex)
    for term in terms:
        step = expm(-1j * dt * term) @ step
    return step


def _second_order(terms: list[NDArray], dt: float) -> NDArray:
    return _first_order(terms[::-1], dt / 2) @ _first_order(terms, dt / 2)


def _suzuki(terms: list[NDArray], dt: float) -> NDArray:
    outer = _second_order(terms, SUZUKI_P * dt)
    middle = _second_order(terms, (1 - 4 * SUZUKI_P) * dt)
    return outer @ outer @ middle @ outer @ outer


def _forest_ruth(terms: list[NDArray], dt: float) -> NDArray:
    outer = _second_order(terms, FOREST_RUTH_W1 * dt)
    return outer @ _second_order(terms, FOREST_RUTH_W0 * dt) @ outer


FORMULAS = [
    (1, _first_order),
    (2, _second_order),
    (4, _suzuki),
    (4, _forest_ruth),
]


def _eigenphase_gap(u: NDArray, v: NDArray) -> float:
    """Get the largest distance from an eigenphase of either unitary to the other's."""
    a = np.angle(np.linalg.eigvals(u))
    b = np.angle(np.linalg.eigvals(v))
    diff = np.abs(a[:, None] - b[None, :])
    circular = np.minimum(diff, 2 * pi - diff)
    return max(circular.min(axis=1).max(), circular.min(axis=0).max())


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize(
    ("order", "formula"), FORMULAS, ids=["lie", "strang", "suzuki4", "forest_ruth4"]
)
@pytest.mark.parametrize("reps", [1, 4, 16])
class TestTrotterBound:
    """The Trotter bounds hold for dense product formulas of every backend."""

    def test_unitary_error(self, seed: int, order: int, formula: object, reps: int):
        h = _random_hamiltonian(seed)
        trotter = sim.Trotter(hamiltonian=h, order=order, reps=reps)
        terms = _term_matrices(h)
        approx = np.linalg.matrix_power(formula(terms, trotter.time / reps), reps)
        exact = expm(-1j * trotter.time * sum(terms))

        assert np.linalg.norm(approx - exact, 2) <= trotter.unitary_error

    def test_energy_error(self, seed: int, order: int, formula: object, reps: int):
        h = _random_hamiltonian(seed)
        trotter = sim.Trotter(hamiltonian=h, order=order, reps=reps)
        terms = _term_matrices(h)
        approx = np.linalg.matrix_power(formula(terms, trotter.time / reps), reps)
        exact = expm(-1j * trotter.time * sum(terms))

        # Each eigenvalue of the simulated Hamiltonian is within `error` of one of H.
        gap = _eigenphase_gap(approx, exact) / trotter.time
        assert gap <= trotter.error


class TestQubitisedBound:
    """The Prepare error bound holds for Qualtran's rotation-based state preparation."""

    @pytest.mark.parametrize("num_phase_ancillas", [2, 3, 4])
    def test_prepare_error(self, num_phase_ancillas: int):
        from qualtran.bloqs.rotations.phase_gradient import (  # noqa: PLC0415
            PhaseGradientState,
        )

        from quiche.qualtran.bloqs import LCUBlockEncodingWrapper  # noqa: PLC0415

        h = PauliSum(
            coefficients=(0.7, -0.4, 0.31),
            terms=tuple(
                PauliWord.from_str(w, big_endian=True) for w in ("XI", "IZ", "XZ")
            ),
            identity_coefficient=-0.2,
        )
        weights = np.abs([*h.coefficients, h.identity_coefficient])
        alpha = weights.sum()

        block_encoding = LCUBlockEncodingWrapper.from_hamiltonian(h, num_phase_ancillas)
        state_prep = block_encoding.prepare.stateprep
        num_index = block_encoding.prepare.select_nqubits
        gradient = PhaseGradientState(num_phase_ancillas).tensor_contract()
        initial = np.kron(np.eye(2**num_index)[0], gradient)
        prepared = (state_prep.tensor_contract() @ initial).reshape(2**num_index, -1)
        amplitudes = prepared @ gradient.conj()

        # The block-encoded term weights are the prepared probabilities, times alpha.
        probabilities = np.abs(amplitudes) ** 2
        weight_error = (
            alpha * np.abs(probabilities[: len(weights)] - weights / alpha).sum()
        )
        weight_error += alpha * probabilities[len(weights) :].sum()

        qubitised = sim.Qubitised(hamiltonian=h, num_phase_ancillas=num_phase_ancillas)
        assert weight_error <= qubitised.error


def _qpe_distribution(phase: float, num_bits: int) -> NDArray[np.float64]:
    """Get the textbook QPE outcome distribution for an eigenphase, in cycles."""
    dimension = 2**num_bits
    delta = phase - np.arange(dimension) / dimension
    amplitude = np.exp(2j * pi * np.outer(np.arange(dimension), delta)).sum(axis=0)
    return np.abs(amplitude / dimension) ** 2


@pytest.mark.parametrize("overlap", [1.0, 0.7, 0.3, 0.1])
@pytest.mark.parametrize("ground_phase", [0.1234, 0.0567])
class TestEstimationBound:
    """`success_probability` lower-bounds the exact repeat-and-minimum protocol."""

    def _textbook(self, h2: PauliSum, overlap: float) -> est.Textbook:
        # A near-exact simulation isolates the estimation bookkeeping.
        trotter = sim.Trotter(hamiltonian=h2, reps=10**6)
        return est.Textbook(
            simulation=trotter, overlap=overlap, error=trotter.energy_scale / 16
        )

    def test_tail_probability(self, h2: PauliSum, overlap: float, ground_phase: float):
        textbook = self._textbook(h2, overlap)
        dimension = 2**textbook.num_ancillas
        readings = np.arange(dimension) / dimension
        distance = np.abs((readings - ground_phase + 0.5) % 1 - 0.5)
        window = 2.0**-textbook.precision_bits

        probabilities = _qpe_distribution(ground_phase, textbook.num_ancillas)
        assert probabilities[distance > window].sum() <= textbook.tail_probability

    def test_success_probability(
        self, h2: PauliSum, overlap: float, ground_phase: float
    ):
        textbook = self._textbook(h2, overlap)
        scale = textbook.simulation.energy_scale
        ground, excited = ground_phase * scale, (ground_phase + 0.2) * scale
        dimension = 2**textbook.num_ancillas
        readings = np.arange(dimension) / dimension
        energies = np.where(readings >= 0.5, readings - 1, readings) * scale

        # Each run reads from the ground state with probability `overlap`.
        probabilities = overlap * _qpe_distribution(
            ground / scale, textbook.num_ancillas
        )
        probabilities += (1 - overlap) * _qpe_distribution(
            excited / scale, textbook.num_ancillas
        )
        above_low = probabilities[energies >= ground - textbook.total_error].sum()
        above_high = probabilities[energies > ground + textbook.total_error].sum()
        # The minimum of R runs lies in the window iff all lie above its lower edge
        # and not all lie above its upper edge.
        exact = above_low**textbook.repetitions - above_high**textbook.repetitions

        assert exact >= textbook.success_probability
