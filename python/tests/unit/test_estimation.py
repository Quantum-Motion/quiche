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

from math import ceil, log2, pi

import pytest
from qualtran.bloqs.qubitization.qubitization_walk_operator import (
    QubitizationWalkOperator,
)
from qualtran.resource_counting.generalizers import ignore_split_join
from qualtran.testing import (
    assert_equivalent_bloq_counts,
)

from quiche.core import Errors, PauliSum
from quiche.qualtran import logical_qubit_resources
from quiche.qualtran.bloqs import (
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


def _make_qdrift(h: PauliSum, budget: Errors, seed: int = 20148) -> QDRIFT:
    t = 2 * pi / h.lam
    n_terms = ceil(2 * h.lam**2 * t**2 / budget.simulation)
    return QDRIFT(h, t, n_terms, seed)


def _make_trotter(h: PauliSum, order: int, n_steps: int = 100) -> Trotterisation:
    return Trotterisation(h, 2 * pi / h.lam, n_steps, order)


def _make_qubitisation_walk(
    h: PauliSum, budget: Errors
) -> tuple[QubitizationWalkOperator, int, int]:
    select_nqubits = ceil(log2(h.n_terms))
    phase_bitsize = max(ceil(log2(2.0 * select_nqubits / budget.simulation)), 2)
    blockencoding = LCUBlockEncodingWrapper.from_hamiltonian(h, phase_bitsize)
    return QubitizationWalkOperator(blockencoding), select_nqubits, phase_bitsize


def _make_textbookqpe_trotter(
    simulation: QDRIFT | Trotterisation,
    data_qubits: int,
    estimation_qubits: int,
) -> TextbookQPE:
    def ladder(index: int) -> TrotterLadder:
        return TrotterLadder(index, simulation, data_qubits, estimation_qubits)

    return TextbookQPE(ladder, data_qubits, estimation_qubits, 0)


def _make_textbookqpe_qubitised(
    walk: QubitizationWalkOperator,
    data_qubits: int,
    estimation_qubits: int,
    selection_ancillas: int,
) -> TextbookQPE:

    def ladder(index: int) -> QubitisationLadder:
        return QubitisationLadder(
            index, walk, data_qubits, estimation_qubits, selection_ancillas
        )

    return TextbookQPE(ladder, data_qubits, estimation_qubits, selection_ancillas)


def _get_num_estimation_qubits(budget: Errors) -> int:
    return ceil(log2(1 / budget.estimation)) + ceil(log2(1 / budget.overlap)) + 4


@pytest.fixture(
    params=[
        _make_qdrift,
        lambda h, _budget: _make_trotter(h, 2),
        lambda h, _budget: _make_trotter(h, 4),
    ],
    ids=["qdrift", "trotter_order2", "trotter_order4"],
)
def simulation(
    request: pytest.FixtureRequest, h2: PauliSum, budget: Errors
) -> QDRIFT | Trotterisation:
    """Test fixture generating Trotter and QDRIFT simulations."""
    return request.param(h2, budget)


class TestNaiveQPE:
    """Test NaiveQPE class."""

    @pytest.mark.parametrize("mode", ["re", "im"])
    def test_bloq_counts(self, simulation: QDRIFT | Trotterisation, mode: str):
        bloq = NaiveQPE(simulation, mode)
        assert_equivalent_bloq_counts(bloq, generalizer=[ignore_split_join])

    @pytest.mark.parametrize("mode", ["re", "im"])
    def test_qubit_counts(self, simulation: QDRIFT | Trotterisation, mode: str):
        bloq = NaiveQPE(simulation, mode)
        manual_counts = logical_qubit_resources(bloq)
        decomp_counts = logical_qubit_resources(bloq.decompose_bloq())
        assert manual_counts == decomp_counts


class TestKitaevQPE:
    """Test KitaevQPE class."""

    @pytest.mark.parametrize("k", range(4))
    @pytest.mark.parametrize("mode", ["re", "im"])
    def test_bloq_counts(self, simulation: QDRIFT | Trotterisation, k: int, mode: str):
        bloq = KitaevQPE(simulation, k, mode)
        assert_equivalent_bloq_counts(bloq, generalizer=[ignore_split_join])

    @pytest.mark.parametrize("k", range(4))
    @pytest.mark.parametrize("mode", ["re", "im"])
    def test_qubit_counts(self, simulation: QDRIFT | Trotterisation, k: int, mode: str):
        bloq = KitaevQPE(simulation, k, mode)
        manual_counts = logical_qubit_resources(bloq)
        decomp_counts = logical_qubit_resources(bloq.decompose_bloq())
        assert manual_counts == decomp_counts


class TestIterativeQPE:
    """Test IterativeQPE class."""

    @pytest.mark.parametrize(
        ("k", "mode", "err_msg"),
        [
            (-1, "re", "Exponent k must be positive"),
            (3, "a", "Measurement mode must be either 're' or 'im'"),
        ],
    )
    def test_invalid_inputs(
        self, h2: PauliSum, budget: Errors, k: int, mode: str, err_msg: str
    ):
        simulation = _make_qdrift(h2, budget)
        with pytest.raises(ValueError, match=err_msg):
            IterativeQPE(simulation, k, mode)

    @pytest.mark.parametrize("k", range(4))
    @pytest.mark.parametrize("mode", ["re", "im"])
    def test_bloq_counts(self, simulation: QDRIFT | Trotterisation, k: int, mode: str):
        bloq = IterativeQPE(simulation, k, mode)
        assert_equivalent_bloq_counts(bloq, generalizer=[ignore_split_join])

    @pytest.mark.parametrize("k", range(4))
    @pytest.mark.parametrize("mode", ["re", "im"])
    def test_qubit_counts(self, simulation: QDRIFT | Trotterisation, k: int, mode: str):
        bloq = IterativeQPE(simulation, k, mode)
        manual_counts = logical_qubit_resources(bloq)
        decomp_counts = logical_qubit_resources(bloq.decompose_bloq())
        assert manual_counts == decomp_counts


class TestTextbookQPE:
    """Test TextbookQPE class."""

    def test_bloq_counts_trotter_ladder(
        self, simulation: QDRIFT | Trotterisation, h2: PauliSum, budget: Errors
    ):
        num_data = h2.n_qubits
        num_estimation = _get_num_estimation_qubits(budget)

        bloq = _make_textbookqpe_trotter(simulation, num_data, num_estimation)
        assert_equivalent_bloq_counts(bloq, generalizer=[ignore_split_join])

    def test_bloq_count_qubitisation_ladder(self, h2: PauliSum, budget: Errors):
        walk, select_nqubits, phase_bitsize = _make_qubitisation_walk(h2, budget)
        num_data = h2.n_qubits
        num_estimation = _get_num_estimation_qubits(budget)
        num_ancillas = select_nqubits + phase_bitsize

        bloq = _make_textbookqpe_qubitised(walk, num_data, num_estimation, num_ancillas)
        assert_equivalent_bloq_counts(bloq, generalizer=[ignore_split_join])

    def test_qubit_counts_trotter_ladder(
        self, simulation: QDRIFT | Trotterisation, h2: PauliSum, budget: Errors
    ):
        num_data = h2.n_qubits
        num_estimation = _get_num_estimation_qubits(budget)
        bloq = _make_textbookqpe_trotter(simulation, num_data, num_estimation)
        manual_counts = logical_qubit_resources(bloq)
        decomp_counts = logical_qubit_resources(bloq.decompose_bloq())
        assert manual_counts == decomp_counts

    def test_qubit_counts_qubitisation_ladder(self, h2: PauliSum, budget: Errors):
        walk, select_nqubits, phase_bitsize = _make_qubitisation_walk(h2, budget)
        num_data = h2.n_qubits
        num_estimation = _get_num_estimation_qubits(budget)
        num_ancillas = select_nqubits + phase_bitsize

        bloq = _make_textbookqpe_qubitised(walk, num_data, num_estimation, num_ancillas)
        manual_counts = logical_qubit_resources(bloq)
        decomp_counts = logical_qubit_resources(bloq.decompose_bloq())
        assert manual_counts == decomp_counts
