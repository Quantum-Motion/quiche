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

"""Tests for stateprep algorithms."""

import pytest

from quiche.algorithms.stateprep import BitstringStateprep, IdentityStateprep
from quiche.chemistry import Mapping, get_hf_state, get_qubit_state


class TestIdentityStateprep:
    """Test IdentityStateprep class."""

    @pytest.mark.parametrize(
        ("num_qubits", "error_msg"),
        [
            (-3, "should be greater than 0"),
            (0, "should be greater than 0"),
            (2.5, "should be a valid integer"),
        ],
    )
    def test_invalid(self, num_qubits: float, error_msg: str):
        with pytest.raises(ValueError, match=error_msg):
            IdentityStateprep(num_qubits)  # ty: ignore[invalid-argument-type]


class TestBitstringStateprep:
    """Test BitstringStateprep class."""

    @pytest.mark.parametrize(
        "mapping",
        [Mapping.BravyiKitaev, Mapping.JordanWigner, Mapping.Parity],
    )
    def test_valid_occupation(self, mapping: Mapping):
        occupation = (0, 1, 1, 0)
        state = BitstringStateprep.occupation(occupation, mapping)
        assert state.bitstring == get_qubit_state(occupation, mapping)

    @pytest.mark.parametrize(
        ("num_electrons", "num_spin_orbitals"),
        [(2, 4), (4, 10), (0, 3)],
    )
    @pytest.mark.parametrize(
        "mapping",
        [Mapping.BravyiKitaev, Mapping.Parity, Mapping.JordanWigner],
    )
    def test_valid_hf(
        self,
        num_electrons: int,
        num_spin_orbitals: int,
        mapping: Mapping,
    ):
        state = BitstringStateprep.hartree_fock(
            num_electrons,
            num_spin_orbitals,
            mapping,
        )

        hf = get_hf_state(num_electrons, num_spin_orbitals)
        expected = get_qubit_state(hf, mapping)

        assert state.bitstring == expected

    @pytest.mark.parametrize(
        ("num_electrons", "num_spin_orbitals", "error_msg"),
        [
            (2, 1, "electrons must not exceed number of spin orbitals"),
            (3, 5, "must have even number of electrons"),
            (-2, 3, "Number of electrons must be non-negative"),
            (2, 0, "Number of spin orbitals must be positive"),
        ],
    )
    def test_invalid_hf(
        self,
        num_electrons: int,
        num_spin_orbitals: int,
        error_msg: str,
    ):
        with pytest.raises(ValueError, match=error_msg):
            BitstringStateprep.hartree_fock(
                num_electrons,
                num_spin_orbitals,
                Mapping.BravyiKitaev,
            )

    @pytest.mark.parametrize("occupation", [(1, 2, 1, 1), (1, -1, 0), (1, "a", 0)])
    def test_invalid(self, occupation: tuple):
        with pytest.raises(ValueError, match="should be 0 or 1"):
            BitstringStateprep.occupation(occupation, Mapping.Parity)
