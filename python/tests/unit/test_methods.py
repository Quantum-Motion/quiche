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


"""Tests for the simulation and estimation method objects."""

from math import ceil, inf, isclose, log2, pi

import pytest
from pydantic import ValidationError

from quiche import estimation as est
from quiche import simulation as sim
from quiche.core import PauliSum


class TestTrotter:
    """Tests for simulation.Trotter."""

    def test_default_time(self, h2: PauliSum):
        trotter = sim.Trotter(hamiltonian=h2, reps=10)
        assert isclose(trotter.time, pi / h2.lam)

    @pytest.mark.parametrize("order", [1, 2, 4])
    def test_error_target_is_met(self, h2: PauliSum, order: int):
        trotter = sim.Trotter(hamiltonian=h2, order=order, error=1e-3)
        assert trotter.error <= 1e-3
        # One fewer step would miss the target.
        fewer = sim.Trotter(hamiltonian=h2, order=order, reps=trotter.reps - 1)
        assert fewer.error > 1e-3

    def test_error_from_reps(self, h2: PauliSum):
        trotter = sim.Trotter(hamiltonian=h2, order=4, time=0.5, reps=7)
        assert isclose(trotter.error, (h2.lam * 0.5) ** 5 / 7**4)

    def test_matches_old_budget_formula(self, h2: PauliSum):
        # The formula previously used by QPESpec, with order fixed at 2.
        time = pi / h2.lam
        reps = ceil((h2.lam**3 * time**3 / 1e-2) ** 0.5)
        assert sim.Trotter(hamiltonian=h2, error=1e-2).reps == reps

    def test_no_ancillas(self, h2: PauliSum):
        assert sim.Trotter(hamiltonian=h2, reps=1).num_ancillas == 0

    @pytest.mark.parametrize("kwargs", [{}, {"reps": 3, "error": 0.1}])
    def test_exactly_one_of_reps_or_error(self, h2: PauliSum, kwargs: dict):
        with pytest.raises(ValidationError, match="Exactly one of `reps` or `error`"):
            sim.Trotter(hamiltonian=h2, **kwargs)

    @pytest.mark.parametrize("kwargs", [{"reps": 0}, {"error": -0.1}, {"order": 0}])
    def test_invalid_values(self, h2: PauliSum, kwargs: dict):
        with pytest.raises(ValidationError):
            sim.Trotter(hamiltonian=h2, **({"reps": 1} | kwargs))

    def test_frozen(self, h2: PauliSum):
        trotter = sim.Trotter(hamiltonian=h2, reps=1)
        with pytest.raises(AttributeError):
            trotter.reps = 2  # type: ignore[misc]


class TestQDRIFT:
    """Tests for simulation.QDRIFT."""

    def test_error_target_is_met(self, h2: PauliSum):
        qdrift = sim.QDRIFT(hamiltonian=h2, error=0.1)
        assert qdrift.error <= 0.1
        assert sim.QDRIFT(hamiltonian=h2, reps=qdrift.reps - 1).error > 0.1

    def test_error_from_reps(self, h2: PauliSum):
        qdrift = sim.QDRIFT(hamiltonian=h2, time=0.5, reps=40)
        assert isclose(qdrift.error, 2 * h2.lam**2 * 0.5**2 / 40)

    def test_seed_is_kept(self, h2: PauliSum):
        assert sim.QDRIFT(hamiltonian=h2, reps=1, seed=7).seed == 7

    def test_exactly_one_of_reps_or_error(self, h2: PauliSum):
        with pytest.raises(ValidationError, match="Exactly one"):
            sim.QDRIFT(hamiltonian=h2)


class TestQubitised:
    """Tests for simulation.Qubitised."""

    def test_ancillas(self, h2: PauliSum):
        qubitised = sim.Qubitised(hamiltonian=h2, num_phase_ancillas=5)
        assert qubitised.num_index_ancillas == ceil(log2(h2.n_terms))
        assert qubitised.num_ancillas == qubitised.num_index_ancillas + 5

    def test_error_target_is_met(self, h2: PauliSum):
        qubitised = sim.Qubitised(hamiltonian=h2, prepare_error=0.01)
        assert qubitised.error == qubitised.prepare_error
        assert qubitised.error <= 0.01
        fewer = sim.Qubitised(
            hamiltonian=h2, num_phase_ancillas=qubitised.num_phase_ancillas - 1
        )
        assert fewer.error > 0.01

    def test_matches_old_budget_formula(self, h2: PauliSum):
        num_index = ceil(log2(h2.n_terms))
        num_phase = max(ceil(log2(2.0 * num_index / 0.05)), 2)
        qubitised = sim.Qubitised(hamiltonian=h2, prepare_error=0.05)
        assert qubitised.num_phase_ancillas == num_phase

    def test_exactly_one_of_ancillas_or_error(self, h2: PauliSum):
        with pytest.raises(ValidationError, match="Exactly one"):
            sim.Qubitised(hamiltonian=h2, num_phase_ancillas=3, prepare_error=0.1)


class TestTextbook:
    """Tests for estimation.Textbook."""

    @pytest.fixture
    def trotter(self, h2: PauliSum) -> sim.Trotter:
        return sim.Trotter(hamiltonian=h2, error=1e-3)

    def test_matches_old_budget_formula(self, trotter: sim.Trotter):
        textbook = est.Textbook(simulation=trotter, overlap=0.3, error=0.01)
        expected = ceil(log2(1 / 0.01)) + ceil(log2(1 / 0.3)) + 4
        assert textbook.num_ancillas == expected

    @pytest.mark.parametrize("overlap", [1, 0.5, 0.3])
    def test_round_trip(self, trotter: sim.Trotter, overlap: float):
        from_error = est.Textbook(simulation=trotter, overlap=overlap, error=0.01)
        assert from_error.error <= 0.01
        from_ancillas = est.Textbook(
            simulation=trotter, overlap=overlap, num_ancillas=from_error.num_ancillas
        )
        assert from_ancillas.error == from_error.error

    def test_too_few_ancillas(self, trotter: sim.Trotter):
        with pytest.raises(ValidationError, match="at least 5 ancillas"):
            est.Textbook(simulation=trotter, overlap=0.5, num_ancillas=4)

    def test_qubit_counts(self, h2: PauliSum):
        qubitised = sim.Qubitised(hamiltonian=h2, num_phase_ancillas=3)
        textbook = est.Textbook(simulation=qubitised, overlap=1, num_ancillas=6)
        assert textbook.num_data == h2.n_qubits
        assert textbook.num_qpe_ancillas == 6
        assert textbook.num_simulation_ancillas == qubitised.num_ancillas
        assert textbook.num_qubits == h2.n_qubits + 6 + qubitised.num_ancillas

    def test_total_error(self, trotter: sim.Trotter):
        textbook = est.Textbook(simulation=trotter, overlap=1, num_ancillas=8)
        assert textbook.total_error == textbook.error + trotter.error

    def test_keeps_simulation_instance(self, trotter: sim.Trotter):
        assert est.Textbook(simulation=trotter, overlap=1, error=0.1).simulation is (
            trotter
        )

    @pytest.mark.parametrize("overlap", [0, -0.5, 1.5])
    def test_invalid_overlap(self, trotter: sim.Trotter, overlap: float):
        with pytest.raises(ValidationError):
            est.Textbook(simulation=trotter, overlap=overlap, error=0.1)

    def test_exactly_one_of_ancillas_or_error(self, trotter: sim.Trotter):
        with pytest.raises(ValidationError, match="Exactly one"):
            est.Textbook(simulation=trotter, overlap=1)


@pytest.mark.parametrize("algorithm", [est.Kitaev, est.Iterative])
class TestSingleAncilla:
    """Tests for estimation.Kitaev and estimation.Iterative."""

    def test_rounds_match_textbook_ancillas(
        self, h2: PauliSum, algorithm: type[est.Kitaev | est.Iterative]
    ):
        trotter = sim.Trotter(hamiltonian=h2, reps=1)
        qpe = algorithm(simulation=trotter, overlap=0.5, error=0.01)
        textbook = est.Textbook(simulation=trotter, overlap=0.5, error=0.01)
        assert qpe.num_rounds == textbook.num_ancillas
        assert qpe.error == textbook.error

    def test_single_ancilla(
        self, h2: PauliSum, algorithm: type[est.Kitaev | est.Iterative]
    ):
        trotter = sim.Trotter(hamiltonian=h2, reps=1)
        qpe = algorithm(simulation=trotter, overlap=1, num_rounds=6)
        assert qpe.num_qpe_ancillas == 1
        assert qpe.num_qubits == h2.n_qubits + 1

    def test_exactly_one_of_rounds_or_error(
        self, h2: PauliSum, algorithm: type[est.Kitaev | est.Iterative]
    ):
        trotter = sim.Trotter(hamiltonian=h2, reps=1)
        with pytest.raises(ValidationError, match="Exactly one"):
            algorithm(simulation=trotter, overlap=1, num_rounds=6, error=0.1)


class TestNaive:
    """Tests for estimation.Naive."""

    def test_unbounded_error(self, h2: PauliSum):
        naive = est.Naive(simulation=sim.Trotter(hamiltonian=h2, reps=1))
        assert naive.error == inf
        assert naive.total_error == inf
        assert naive.num_qpe_ancillas == 1
