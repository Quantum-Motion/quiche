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

"""Tests for circuits module."""

import pytest
from qualtran.resource_counting import QubitCount
from qualtran.resource_counting.generalizers import ignore_split_join
from qualtran.testing import (
    assert_equivalent_bloq_counts,
)

from quiche.resources.bloqs import (
    BitstringStatePrep,
    IdentityStatePrep,
)


class TestBitstringStatePrep:
    """Tests for computational basis state bitstring statepreparation."""

    bitstring = (1, 1, 0, 1, 0)
    bloq = BitstringStatePrep(bitstring)

    @pytest.mark.parametrize("bitstring", [(0, 1, 0, 1, 2), (-1, 1, 1), ("b", 0, 1)])
    def test_invalid_bitstring(self, bitstring: tuple):
        with pytest.raises(ValueError, match="Invalid bitstring"):
            BitstringStatePrep(bitstring)

    def test_num_qubits(self):
        actual = self.bloq.my_static_costs(QubitCount())
        expected = len(self.bitstring)
        assert actual == expected

    def test_bloq_counts(self):
        assert_equivalent_bloq_counts(self.bloq, generalizer=[ignore_split_join])


class TestIdentityStatePrep:
    """Tests for identity state prep."""

    n_qubits = 5
    bloq = IdentityStatePrep(n_qubits)

    def test_num_qubits(self):
        actual = self.bloq.my_static_costs(QubitCount())
        expected = self.n_qubits
        assert actual == expected

    def test_bloq_counts(self):
        assert_equivalent_bloq_counts(self.bloq, generalizer=[ignore_split_join])
