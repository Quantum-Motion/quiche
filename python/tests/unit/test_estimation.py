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

"""Tests for estimation module."""

# TODO(Vasco): refactor these tests.

from math import ceil, log2, pi

import pytest
from qualtran import Bloq
from qualtran.bloqs.qubitization.qubitization_walk_operator import (
    QubitizationWalkOperator,
)
from qualtran.resource_counting.generalizers import ignore_split_join
from qualtran.testing import (
    assert_equivalent_bloq_counts,
)

from quiche.core import Errors, PauliSum
from quiche.hamlib import get_dataset, parse_hamiltonian
from quiche.resources import logical_qubit_resources
from quiche.resources.bloqs import (
    QDRIFT,
    IterativeQPE,
    KitaevQPE,
    LCUBlockEncodingWrapper,
    NaiveQPE,
    QubitisationLadder,
    TextbookQPE,
    Trotterisation,
    TrotterLadder,
)


def _load_h2() -> PauliSum:
    """Load the H2 Hamiltonian from tests/data/H2.hdf5."""
    filename = "tests/data/H2.hdf5"  # assumes this is called from the top level
    dataset = "ham_JW-4"  # choose smallest size to keep tests fast
    raw_data = get_dataset(filename, dataset)
    return parse_hamiltonian(raw_data)


def _default_budget() -> Errors:
    tot_error = 0.16
    return Errors(
        estimation=tot_error / 3.0,
        simulation=1.0,
        rotations=tot_error / 3.0,
        state_prep=tot_error / 3.0,
        overlap=1,
    )


def _make_qdrift(h: PauliSum, budget: Errors) -> QDRIFT:
    t = 2 * pi / h.lam
    seed = 20148
    eps_sim = budget.simulation
    n_terms = ceil(2 * h.lam**2 * t**2 / eps_sim)
    return QDRIFT(h, t, n_terms, seed)


def _make_trotter(h: PauliSum, _budget: Errors, order: int) -> Trotterisation:
    t = 2 * pi / h.lam
    n_steps = 100
    return Trotterisation(h, t, n_steps, order)


def _make_qubitisation_walk(
    h: PauliSum, budget: Errors
) -> tuple[QubitizationWalkOperator, int, int]:
    e_prep = budget.simulation
    select_nqubits = ceil(log2(h.n_terms))
    phase_bitsize = max(ceil(log2(2.0 * select_nqubits / e_prep)), 2)
    blockencoding = LCUBlockEncodingWrapper.from_hamiltonian(h, phase_bitsize)
    return QubitizationWalkOperator(blockencoding), select_nqubits, phase_bitsize


class TestNaiveQPE:
    """Test NaiveQPE class."""

    h = _load_h2()
    budget = _default_budget()
    qdrift = _make_qdrift(h, budget)
    trotter_order2 = _make_trotter(h, budget, 2)
    trotter_order4 = _make_trotter(h, budget, 4)

    @pytest.mark.parametrize("simulation", [qdrift, trotter_order2, trotter_order4])
    @pytest.mark.parametrize("mode", ["re", "im"])
    def test_bloq_counts(self, simulation: Bloq, mode: str):
        bloq = NaiveQPE(simulation, mode)
        assert_equivalent_bloq_counts(bloq, generalizer=[ignore_split_join])

    @pytest.mark.parametrize("simulation", [qdrift, trotter_order2, trotter_order4])
    @pytest.mark.parametrize("mode", ["re", "im"])
    def test_qubit_counts(self, simulation: Bloq, mode: str):
        bloq = NaiveQPE(simulation, mode)
        manual_counts = logical_qubit_resources(bloq)
        decomp_counts = logical_qubit_resources(bloq.decompose_bloq())
        assert manual_counts == decomp_counts


class TestKitaevQPE:
    """Test KitaevQPE class."""

    h = _load_h2()
    budget = _default_budget()
    qdrift = _make_qdrift(h, budget)
    trotter_order2 = _make_trotter(h, budget, 2)
    trotter_order4 = _make_trotter(h, budget, 4)

    @pytest.mark.parametrize("k", list(range(4)))
    @pytest.mark.parametrize("simulation", [qdrift, trotter_order2, trotter_order4])
    @pytest.mark.parametrize("mode", ["re", "im"])
    def test_bloq_counts(self, simulation: Bloq, k: int, mode: str):
        bloq = KitaevQPE(simulation, k, mode)
        assert_equivalent_bloq_counts(bloq, generalizer=[ignore_split_join])

    @pytest.mark.parametrize("k", list(range(4)))
    @pytest.mark.parametrize("simulation", [qdrift, trotter_order2, trotter_order4])
    @pytest.mark.parametrize("mode", ["re", "im"])
    def test_qubit_counts(self, simulation: Bloq, k: int, mode: str):
        bloq = KitaevQPE(simulation, k, mode)
        manual_counts = logical_qubit_resources(bloq)
        decomp_counts = logical_qubit_resources(bloq.decompose_bloq())
        assert manual_counts == decomp_counts


class TestIterativeQPE:
    """Test IterativeQPE class."""

    h = _load_h2()
    budget = _default_budget()
    qdrift = _make_qdrift(h, budget)
    trotter_order2 = _make_trotter(h, budget, 2)
    trotter_order4 = _make_trotter(h, budget, 4)

    @pytest.mark.parametrize(
        ("simulation", "k", "mode", "err_msg"),
        [
            (qdrift, -1, "re", "Exponent k must be positive"),
            (qdrift, 3, "a", "Measurement mode must be either 're' or 'im'"),
        ],
    )

    def test_invalid_inputs(self, simulation: Bloq, k: int, mode: str, err_msg: str):
        with pytest.raises(ValueError, match=err_msg):
            IterativeQPE(simulation, k, mode)

    @pytest.mark.parametrize("k", list(range(4)))
    @pytest.mark.parametrize("simulation", [qdrift, trotter_order2, trotter_order4])
    @pytest.mark.parametrize("mode", ["re", "im"])
    def test_bloq_counts(self, simulation: Bloq, k: int, mode: str):
        bloq = IterativeQPE(simulation, k, mode)
        assert_equivalent_bloq_counts(bloq, generalizer=[ignore_split_join])

    @pytest.mark.parametrize("k", list(range(4)))
    @pytest.mark.parametrize("simulation", [qdrift, trotter_order2, trotter_order4])
    @pytest.mark.parametrize("mode", ["re", "im"])
    def test_qubit_counts(self, simulation: Bloq, k: int, mode: str):
        bloq = IterativeQPE(simulation, k, mode)
        manual_counts = logical_qubit_resources(bloq)
        decomp_counts = logical_qubit_resources(bloq.decompose_bloq())
        assert manual_counts == decomp_counts


class TestTextbookQPE:
    """Test TextbookQPE class."""

    h = _load_h2()
    budget = _default_budget()
    n_estimation_qubits = (
        ceil(log2(1 / budget.estimation)) + ceil(log2(1 / budget.overlap)) + 4
    )
    qdrift = _make_qdrift(h, budget)
    trotter_order2 = _make_trotter(h, budget, 2)
    trotter_order4 = _make_trotter(h, budget, 4)

    @pytest.mark.parametrize(("simulation"), [qdrift, trotter_order2, trotter_order4])
    def test_bloq_counts_trotter(self, simulation: Trotterisation | QDRIFT):

        def ladder(index: int) -> TrotterLadder:
            return TrotterLadder(
                index, simulation, self.h.n_qubits, self.n_estimation_qubits
            )

        bloq = TextbookQPE(ladder, self.h.n_qubits, self.n_estimation_qubits, 0)
        assert_equivalent_bloq_counts(bloq, generalizer=[ignore_split_join])

    def test_bloq_count_qubitisation(self):
        walk, select_nqubits, phase_bitsize = _make_qubitisation_walk(
            self.h, self.budget
        )

        def qubitisationladder(index: int) -> QubitisationLadder:
            return QubitisationLadder(
                index,
                walk,
                self.h.n_qubits,
                self.n_estimation_qubits,
                select_nqubits + phase_bitsize,
            )

        bloq = TextbookQPE(
            qubitisationladder,
            self.h.n_qubits,
            self.n_estimation_qubits,
            select_nqubits + phase_bitsize,
        )

        assert_equivalent_bloq_counts(bloq, generalizer=[ignore_split_join])

    @pytest.mark.parametrize(("simulation"), [qdrift, trotter_order2, trotter_order4])
    def test_qubit_counts_trotter(self, simulation: Trotterisation | QDRIFT):

        def ladder(index: int) -> TrotterLadder:
            return TrotterLadder(
                index, simulation, self.h.n_qubits, self.n_estimation_qubits
            )

        bloq = TextbookQPE(ladder, self.h.n_qubits, self.n_estimation_qubits, 0)
        manual_counts = logical_qubit_resources(bloq)
        decomp_counts = logical_qubit_resources(bloq.decompose_bloq())
        assert manual_counts == decomp_counts

    def test_qubit_counts_qubitisation(self):
        walk, select_nqubits, phase_bitsize = _make_qubitisation_walk(
            self.h, self.budget
        )

        def qubitisationladder(index: int) -> QubitisationLadder:
            return QubitisationLadder(
                index,
                walk,
                self.h.n_qubits,
                self.n_estimation_qubits,
                select_nqubits + phase_bitsize,
            )

        bloq = TextbookQPE(
            qubitisationladder,
            self.h.n_qubits,
            self.n_estimation_qubits,
            select_nqubits + phase_bitsize,
        )

        manual_counts = logical_qubit_resources(bloq)
        decomp_counts = logical_qubit_resources(bloq.decompose_bloq())
        assert manual_counts == decomp_counts
