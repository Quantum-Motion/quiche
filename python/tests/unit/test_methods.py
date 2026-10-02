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

from math import ceil, inf, isclose, isnan, log2, pi

import pytest
from pydantic import ValidationError

from quiche import estimation as est
from quiche import simulation as sim
from quiche.budget.estimation import get_tail_probability
from quiche.core import PauliSum


def _alpha(h: PauliSum) -> float:
    return h.lam + abs(h.identity_coefficient)


class TestTrotter:
    """Tests for simulation.Trotter."""

    def test_default_time_includes_identity(self, h2: PauliSum):
        trotter = sim.Trotter(hamiltonian=h2, reps=10)
        assert isclose(trotter.time, pi / _alpha(h2))

    @pytest.mark.parametrize("order", [1, 2, 4])
    def test_error_target_is_met(self, h2: PauliSum, order: int):
        trotter = sim.Trotter(hamiltonian=h2, order=order, error=1e-3)
        assert trotter.error <= 1e-3
        # One fewer step would miss the target.
        fewer = sim.Trotter(hamiltonian=h2, order=order, reps=trotter.reps - 1)
        assert fewer.error > 1e-3

    @pytest.mark.parametrize("order", [1, 2, 4])
    def test_round_trip(self, h2: PauliSum, order: int):
        from_reps = sim.Trotter(hamiltonian=h2, order=order, reps=30)
        from_error = sim.Trotter(hamiltonian=h2, order=order, error=from_reps.error)
        assert from_error.reps == 30

    def test_energy_error_from_unitary_error(self, h2: PauliSum):
        trotter = sim.Trotter(hamiltonian=h2, reps=7)
        assert trotter.unitary_error > 0
        # 2 arcsin(eps / 2) / t is just above eps / t for small eps.
        assert trotter.error >= trotter.unitary_error / trotter.time
        assert isclose(
            trotter.error, trotter.unitary_error / trotter.time, rel_tol=1e-3
        )

    def test_deterministic(self, h2: PauliSum):
        trotter = sim.Trotter(hamiltonian=h2, reps=1)
        assert trotter.channel_error == 0
        assert trotter.num_ancillas == 0
        assert isclose(trotter.energy_scale, 2 * pi / trotter.time)

    @pytest.mark.parametrize("order", [3, 5])
    def test_odd_order_rejected(self, h2: PauliSum, order: int):
        with pytest.raises(ValidationError, match="1 or even"):
            sim.Trotter(hamiltonian=h2, order=order, reps=1)

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

    def test_no_energy_error(self, h2: PauliSum):
        assert sim.QDRIFT(hamiltonian=h2, reps=100).error == 0

    def test_channel_error(self, h2: PauliSum):
        qdrift = sim.QDRIFT(hamiltonian=h2, time=0.5, reps=40)
        expected = 2 * h2.lam**2 * 0.5**2 / 40
        # Campbell's bound is the leading term times exp(2 lam t / N) >= 1.
        assert qdrift.channel_error >= expected
        assert isclose(qdrift.channel_error, expected, rel_tol=0.05)

    def test_more_reps_is_more_accurate(self, h2: PauliSum):
        few = sim.QDRIFT(hamiltonian=h2, reps=10)
        many = sim.QDRIFT(hamiltonian=h2, reps=100)
        assert many.channel_error < few.channel_error

    def test_reps_required(self, h2: PauliSum):
        with pytest.raises(ValidationError):
            sim.QDRIFT(hamiltonian=h2)  # type: ignore[call-arg]

    def test_seed_is_kept(self, h2: PauliSum):
        assert sim.QDRIFT(hamiltonian=h2, reps=1, seed=7).seed == 7


class TestQubitised:
    """Tests for simulation.Qubitised."""

    def test_ancillas(self, h2: PauliSum):
        qubitised = sim.Qubitised(hamiltonian=h2, num_phase_ancillas=5)
        assert qubitised.num_index_ancillas == ceil(log2(h2.n_terms))
        assert qubitised.num_ancillas == qubitised.num_index_ancillas + 5

    def test_error_target_is_met(self, h2: PauliSum):
        qubitised = sim.Qubitised(hamiltonian=h2, error=1e-3)
        assert qubitised.error <= 1e-3
        fewer = sim.Qubitised(
            hamiltonian=h2, num_phase_ancillas=qubitised.num_phase_ancillas - 1
        )
        assert fewer.error > 1e-3

    def test_energy_scale(self, h2: PauliSum):
        qubitised = sim.Qubitised(hamiltonian=h2, num_phase_ancillas=5)
        assert isclose(qubitised.energy_scale, 2 * pi * _alpha(h2))
        assert qubitised.channel_error == 0

    def test_exactly_one_of_ancillas_or_error(self, h2: PauliSum):
        with pytest.raises(ValidationError, match="Exactly one"):
            sim.Qubitised(hamiltonian=h2, num_phase_ancillas=3, error=0.1)


class TestTextbook:
    """Tests for estimation.Textbook."""

    @pytest.fixture
    def trotter(self, h2: PauliSum) -> sim.Trotter:
        return sim.Trotter(hamiltonian=h2, error=1e-3)

    def test_error_target_is_met(self, trotter: sim.Trotter):
        textbook = est.Textbook(simulation=trotter, overlap=0.5, error=1e-3)
        assert textbook.error <= 1e-3
        assert textbook.error == trotter.energy_scale / 2**textbook.precision_bits
        # One fewer precision bit would miss the target.
        assert trotter.energy_scale / 2 ** (textbook.precision_bits - 1) > 1e-3

    @pytest.mark.parametrize(("overlap", "overlap_bits"), [(1, 0), (0.5, 1), (0.3, 2)])
    def test_ancillas(self, trotter: sim.Trotter, overlap: float, overlap_bits: int):
        textbook = est.Textbook(simulation=trotter, overlap=overlap, error=1e-3)
        expected = textbook.precision_bits + overlap_bits + 4
        assert textbook.num_ancillas == expected
        assert textbook.tail_probability == get_tail_probability(overlap_bits + 4)

    def test_round_trip(self, trotter: sim.Trotter):
        from_error = est.Textbook(simulation=trotter, overlap=0.3, error=1e-2)
        from_ancillas = est.Textbook(
            simulation=trotter, overlap=0.3, num_ancillas=from_error.num_ancillas
        )
        assert from_ancillas.error == from_error.error

    def test_extra_ancillas(self, trotter: sim.Trotter):
        default = est.Textbook(simulation=trotter, overlap=1, error=1e-2)
        more = est.Textbook(simulation=trotter, overlap=1, error=1e-2, extra_ancillas=6)
        assert more.num_ancillas == default.num_ancillas + 2
        assert more.tail_probability < default.tail_probability
        with pytest.raises(ValidationError):
            est.Textbook(simulation=trotter, overlap=1, error=1e-2, extra_ancillas=1)

    def test_too_few_ancillas(self, trotter: sim.Trotter):
        with pytest.raises(ValidationError, match="at least 6 bits"):
            est.Textbook(simulation=trotter, overlap=0.5, num_ancillas=5)

    def test_qubitised_energy_scale(self, h2: PauliSum):
        qubitised = sim.Qubitised(hamiltonian=h2, num_phase_ancillas=3)
        textbook = est.Textbook(simulation=qubitised, overlap=1, num_ancillas=10)
        assert isclose(textbook.error, 2 * pi * _alpha(h2) / 2**6)

    def test_qubit_counts(self, h2: PauliSum):
        qubitised = sim.Qubitised(hamiltonian=h2, num_phase_ancillas=3)
        textbook = est.Textbook(simulation=qubitised, overlap=1, num_ancillas=6)
        assert textbook.num_data == h2.n_qubits
        assert textbook.num_qpe_ancillas == 6
        assert textbook.num_simulation_ancillas == qubitised.num_ancillas
        assert textbook.num_qubits == h2.n_qubits + 6 + qubitised.num_ancillas
        assert textbook.applications == 2**6 - 1

    def test_total_error(self, trotter: sim.Trotter):
        textbook = est.Textbook(simulation=trotter, overlap=1, num_ancillas=8)
        assert textbook.total_error == textbook.error + trotter.error

    @pytest.mark.parametrize("overlap", [1, 0.9, 0.5, 0.1])
    def test_default_repetitions(self, trotter: sim.Trotter, overlap: float):
        textbook = est.Textbook(simulation=trotter, overlap=overlap, error=1e-2)
        hit = overlap * (1 - textbook.tail_probability)
        assert (1 - hit) ** textbook.repetitions <= 0.05
        assert (1 - hit) ** (
            textbook.repetitions - 1
        ) > 0.05 or textbook.repetitions == 1

    def test_success_probability(self, trotter: sim.Trotter):
        def success(**kwargs: float) -> float:
            return est.Textbook(
                simulation=trotter, error=1e-2, **kwargs
            ).success_probability

        assert 0 < success(overlap=0.5) < 1
        assert success(overlap=0.5, repetitions=10) > success(
            overlap=0.5, repetitions=2
        )
        assert success(overlap=0.5, extra_ancillas=8) > success(overlap=0.5)

    def test_qdrift_lowers_success_probability(self, h2: PauliSum):
        def success(reps: int) -> float:
            qdrift = sim.QDRIFT(hamiltonian=h2, reps=reps)
            return est.Textbook(
                simulation=qdrift, overlap=1, num_ancillas=8
            ).success_probability

        assert success(10**9) > success(10**7) > success(10**6)
        assert success(1) == 0

    def test_keeps_simulation_instance(self, trotter: sim.Trotter):
        textbook = est.Textbook(simulation=trotter, overlap=1, error=0.1)
        assert textbook.simulation is trotter

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
        assert qpe.success_probability == textbook.success_probability

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

    def test_unbounded(self, h2: PauliSum):
        naive = est.Naive(simulation=sim.Trotter(hamiltonian=h2, reps=1))
        assert naive.error == inf
        assert naive.total_error == inf
        assert isnan(naive.success_probability)
        assert naive.num_qpe_ancillas == 1
