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

"""Tests for statespec module."""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from quiche.chemistry import (
    HartreeFockState,
    get_bk_state,
    get_jw_state,
    get_parity_state,
)
from quiche.core import Mapping
from quiche.dispatch import HartreeFockSpec, Spec
from quiche.qualtran.bloqs import BitstringStatePrep
from quiche.quest import QuestRoutine


class TestHartreeFockSpec:
    """Tests for HartreeFockSpec."""

    state = HartreeFockState.closed_shell(electrons=2, spin_orbitals=4)

    def test_is_spec(self):
        spec = HartreeFockSpec(state=self.state, mapping=Mapping.JordanWigner)
        assert isinstance(spec, Spec)

    @pytest.mark.parametrize(
        ("mapping", "expected_fn"),
        [
            (Mapping.JordanWigner, get_jw_state),
            (Mapping.BravyiKitaev, get_bk_state),
            (Mapping.Parity, get_parity_state),
        ],
    )
    def test_to_qualtran_matches_mapping(
        self,
        mapping: Mapping,
        expected_fn: Callable[[NDArray[np.int_]], NDArray[np.int_]],
    ):
        """Validate the Bloq's bitstring matches the corresponding chemistry mapping."""
        spec = HartreeFockSpec(state=self.state, mapping=mapping)
        bloq = spec.to_qualtran()
        expected = tuple(expected_fn(self.state.occupation))

        assert isinstance(bloq, BitstringStatePrep)
        assert bloq.bitstring == expected

    def test_to_quest_routine(self):
        spec = HartreeFockSpec(state=self.state, mapping=Mapping.JordanWigner)
        routine = spec.to_quest()

        assert isinstance(routine, QuestRoutine)
        assert len(routine.ops) == 1
