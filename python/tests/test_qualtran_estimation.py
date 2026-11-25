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

"""Tests for the Qualtran implementation of phase estimation algorithms."""

from math import ceil, log2, pi
from typing import Literal

import pytest
from qualtran.bloqs.qubitization.qubitization_walk_operator import (
    QubitizationWalkOperator,
)
from qualtran.resource_counting.generalizers import ignore_split_join
from qualtran.testing import (
    assert_equivalent_bloq_counts,
)

from quiche.core import PauliSum
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

PHASE_BITSIZE = 5


@pytest.fixture
def qdrift(h2: PauliSum) -> QDRIFT:
    """QDRIFT product formula for H2."""
    return QDRIFT(h2, time=2 * pi / h2.lam, num_samples=100, seed=6498)


@pytest.fixture
def trotter2(h2: PauliSum) -> Trotterisation:
    """2nd order Trotterisation of H2."""
    return Trotterisation(h2, time=2 * pi / h2.lam, num_steps=100, order=2)


@pytest.fixture
def trotter4(h2: PauliSum) -> Trotterisation:
    """4th order Trotterisation of H2."""
    return Trotterisation(h2, time=2 * pi / h2.lam, num_steps=100, order=4)


@pytest.fixture(params=["qdrift", "trotter2", "trotter4"])
def product_formula(request: pytest.FixtureRequest) -> QDRIFT | Trotterisation:
    """QDRIFT and Trotter product formulas for H2."""
    return request.getfixturevalue(request.param)


@pytest.fixture
def walk(h2: PauliSum) -> QubitizationWalkOperator:
    """Qubitisation walk operator for H2."""
    blockencoding = LCUBlockEncodingWrapper.from_hamiltonian(
        h2, phase_bitsize=PHASE_BITSIZE
    )
    return QubitizationWalkOperator(blockencoding)


@pytest.fixture
def textbook_product_formula(
    product_formula: QDRIFT | Trotterisation,
    h2: PauliSum,
) -> TextbookQPE:
    """TextbookQPE over a QDRIFT or Trotter ladder for H2."""
    num_data = h2.num_qubits
    num_estimation_qubits = 5

    def ladder(index: int) -> TrotterLadder:
        return TrotterLadder(index, product_formula, num_data, num_estimation_qubits)

    return TextbookQPE(ladder, num_data, num_estimation_qubits, 0)


@pytest.fixture
def textbook_qubitised(walk: QubitizationWalkOperator, h2: PauliSum) -> TextbookQPE:
    """TextbookQPE over a qubitisation ladder for H2."""
    num_data = h2.num_qubits
    num_select_qubits = ceil(log2(h2.num_words_with_identity))
    num_ancillas = num_select_qubits + PHASE_BITSIZE
    num_estimation_qubits = 5

    def ladder(index: int) -> QubitisationLadder:
        return QubitisationLadder(
            index,
            walk,
            num_data,
            num_estimation_qubits,
            num_ancillas,
        )

    return TextbookQPE(ladder, num_data, num_estimation_qubits, num_ancillas)


class TestNaiveQPE:
    """Test NaiveQPE class."""

    @pytest.mark.parametrize("mode", ["re", "im"])
    def test_bloq_counts(
        self,
        product_formula: QDRIFT | Trotterisation,
        mode: Literal["re", "im"],
    ):
        bloq = NaiveQPE(product_formula, mode)
        assert_equivalent_bloq_counts(bloq, generalizer=[ignore_split_join])

    @pytest.mark.parametrize("mode", ["re", "im"])
    def test_qubit_counts(
        self,
        product_formula: QDRIFT | Trotterisation,
        mode: Literal["re", "im"],
    ):
        bloq = NaiveQPE(product_formula, mode)
        actual = logical_qubit_resources(bloq)
        expected = logical_qubit_resources(bloq.decompose_bloq())
        assert actual == expected


class TestKitaevQPE:
    """Test KitaevQPE class."""

    @pytest.mark.parametrize("k", range(4))
    @pytest.mark.parametrize("mode", ["re", "im"])
    def test_bloq_counts(
        self,
        product_formula: QDRIFT | Trotterisation,
        k: int,
        mode: Literal["re", "im"],
    ):
        bloq = KitaevQPE(product_formula, k=k, mode=mode)
        assert_equivalent_bloq_counts(bloq, generalizer=[ignore_split_join])

    @pytest.mark.parametrize("k", range(4))
    @pytest.mark.parametrize("mode", ["re", "im"])
    def test_qubit_counts(
        self,
        product_formula: QDRIFT | Trotterisation,
        k: int,
        mode: Literal["re", "im"],
    ):
        bloq = KitaevQPE(product_formula, k=k, mode=mode)
        actual = logical_qubit_resources(bloq)
        expected = logical_qubit_resources(bloq.decompose_bloq())
        assert actual == expected


class TestIterativeQPE:
    """Test IterativeQPE class."""

    @pytest.mark.parametrize(
        ("k", "mode", "error_msg"),
        [
            (-1, "re", "Exponent must be positive"),
            (3, "a", "Measurement mode must be either 're' or 'im'"),
        ],
    )
    def test_invalid_inputs(
        self,
        qdrift: QDRIFT,
        k: int,
        mode: Literal["re", "im"],
        error_msg: str,
    ):
        with pytest.raises(ValueError, match=error_msg):
            IterativeQPE(qdrift, k=k, mode=mode)

    @pytest.mark.parametrize("k", range(4))
    @pytest.mark.parametrize("mode", ["re", "im"])
    def test_bloq_counts(
        self,
        product_formula: QDRIFT | Trotterisation,
        k: int,
        mode: Literal["re", "im"],
    ):
        bloq = IterativeQPE(product_formula, k=k, mode=mode)
        assert_equivalent_bloq_counts(bloq, generalizer=[ignore_split_join])

    @pytest.mark.parametrize("k", range(4))
    @pytest.mark.parametrize("mode", ["re", "im"])
    def test_qubit_counts(
        self,
        product_formula: QDRIFT | Trotterisation,
        k: int,
        mode: Literal["re", "im"],
    ):
        bloq = IterativeQPE(product_formula, k=k, mode=mode)
        actual = logical_qubit_resources(bloq)
        expected = logical_qubit_resources(bloq.decompose_bloq())
        assert actual == expected


class TestTextbookQPE:
    """Test TextbookQPE class."""

    def test_bloq_counts_product_formula(self, textbook_product_formula: TextbookQPE):
        assert_equivalent_bloq_counts(
            textbook_product_formula,
            generalizer=[ignore_split_join],
        )

    def test_bloq_counts_qubitised(self, textbook_qubitised: TextbookQPE):
        assert_equivalent_bloq_counts(
            textbook_qubitised,
            generalizer=[ignore_split_join],
        )

    def test_qubit_counts_product_formula(self, textbook_product_formula: TextbookQPE):
        actual = logical_qubit_resources(textbook_product_formula)
        expected = logical_qubit_resources(textbook_product_formula.decompose_bloq())
        assert actual == expected

    def test_qubit_counts_qubitised(self, textbook_qubitised: TextbookQPE):
        actual = logical_qubit_resources(textbook_qubitised)
        expected = logical_qubit_resources(textbook_qubitised.decompose_bloq())
        assert actual == expected
