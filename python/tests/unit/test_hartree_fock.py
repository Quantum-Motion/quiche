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

"""Tests for the state_prep module."""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from quiche.chemistry import (
    get_bk_state,
    get_jw_state,
    get_parity_state,
)
from quiche.core import Mapping
from quiche.dispatch import Spec
from quiche.qualtran.bloqs import BitstringStatePrep
from quiche.quest import QuestRoutine
from quiche.state_prep import HartreeFock


class TestHartreeFock:
    """Tests for HartreeFock."""

    occupation = (1, 1, 0, 0)

    def test_is_spec(self):
        spec = HartreeFock(occupation=self.occupation, mapping=Mapping.JordanWigner)
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
        spec = HartreeFock(occupation=self.occupation, mapping=mapping)
        bloq = spec.to_qualtran()
        expected = tuple(expected_fn(self.occupation))

        assert isinstance(bloq, BitstringStatePrep)
        assert bloq.bitstring == expected

    def test_to_quest_routine(self):
        spec = HartreeFock(occupation=self.occupation, mapping=Mapping.JordanWigner)
        routine = spec.to_quest()

        assert isinstance(routine, QuestRoutine)
        assert len(routine.ops) == 1

    def test_num_qubits(self):
        spec = HartreeFock(occupation=self.occupation, mapping=Mapping.JordanWigner)
        assert spec.num_qubits == 4

    @pytest.mark.parametrize(
        ("mapping", "expected_fn"),
        [
            (Mapping.JordanWigner, get_jw_state),
            (Mapping.BravyiKitaev, get_bk_state),
            (Mapping.Parity, get_parity_state),
        ],
    )
    def test_qubits_select_from_mapped_state(
        self,
        mapping: Mapping,
        expected_fn: Callable[[NDArray[np.int_]], NDArray[np.int_]],
    ):
        """Kept qubits index the mapped bitstring, not the occupations."""
        kept = (0, 2, 3)
        spec = HartreeFock(occupation=self.occupation, mapping=mapping, qubits=kept)
        expected = tuple(np.asarray(expected_fn(self.occupation))[list(kept)])

        assert spec.num_qubits == len(kept)
        assert spec.to_qualtran().bitstring == expected

    def test_qubits_out_of_range(self):
        with pytest.raises(ValueError, match="Kept qubits"):
            HartreeFock(
                occupation=self.occupation, mapping=Mapping.JordanWigner, qubits=(4,)
            )

    def test_counts(self):
        spec = HartreeFock(occupation=(1, 0, 1, 0, 0), mapping=Mapping.Parity)
        assert spec.num_electrons == 2
        assert spec.num_spin_orbitals == 5

    def test_closed_shell(self):
        spec = HartreeFock.closed_shell(
            electrons=2, spin_orbitals=4, mapping=Mapping.BravyiKitaev, qubits=(0, 1)
        )
        assert spec.occupation == self.occupation
        assert spec.mapping == Mapping.BravyiKitaev
        assert spec.qubits == (0, 1)

    @pytest.mark.parametrize(
        ("num_electrons", "num_spin_orbitals", "err_msg"),
        [
            (2, 1, "electrons must not exceed number of spin orbitals"),
            (3, 5, "must have even number of electrons"),
            (-2, 3, "Number of electrons must be non-negative"),
            (2, 0, "Number of spin orbitals must be positive"),
        ],
    )
    def test_invalid_closed_shell(
        self,
        num_electrons: int,
        num_spin_orbitals: int,
        err_msg: str,
    ):
        with pytest.raises(ValueError, match=err_msg):
            HartreeFock.closed_shell(
                num_electrons, num_spin_orbitals, mapping=Mapping.JordanWigner
            )

    @pytest.mark.parametrize("occupation", [(1, 2, 1, 1), (1, -1, 0)])
    def test_invalid_occupation(self, occupation: tuple):
        with pytest.raises(ValueError, match="occupation must contain binary entries"):
            HartreeFock(occupation=occupation, mapping=Mapping.JordanWigner)

    def test_non_integer_occupation(self):
        with pytest.raises(ValueError, match="valid integer"):
            HartreeFock(occupation=(1, "a", 0), mapping=Mapping.JordanWigner)

    def test_mapping_is_required(self):
        with pytest.raises(ValidationError, match="mapping"):
            HartreeFock(occupation=self.occupation)  # type: ignore[call-arg]
