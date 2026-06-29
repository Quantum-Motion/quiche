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

"""Tests for chemistry module."""

import numpy as np
import pytest
from numpy.typing import NDArray

from quiche.chemistry import (
    HartreeFockState,
    get_bk_state,
    get_hf_state,
    get_jw_state,
    get_parity_state,
)


@pytest.mark.parametrize(
    ("num_spin_orbitals", "num_electrons", "err_msg"),
    [
        (1, -1, "electrons must be non-negative"),
        (0, 0, "spin orbitals must be positive"),
        (3, 4, "electrons must not exceed number of spin orbitals"),
    ],
)
def test_get_hf_state_invalid_cases(
    num_spin_orbitals: int,
    num_electrons: int,
    err_msg: str,
):
    """Test input validation for Hartree-Fock basis state."""
    with pytest.raises(ValueError, match=err_msg):
        get_hf_state(num_spin_orbitals, num_electrons)


@pytest.mark.parametrize(
    ("num_spin_orbitals", "num_electrons", "expected"),
    [
        (1, 1, np.array([1])),
        (5, 2, np.array([1, 1, 0, 0, 0])),
        (3, 0, np.array([0, 0, 0])),
    ],
)
def test_get_hf_state_state(
    num_spin_orbitals: int,
    num_electrons: int,
    expected: NDArray[np.int_],
):
    """Validate Hartree-Fock basis state."""
    actual = get_hf_state(num_spin_orbitals, num_electrons)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("occupation", "expected"),
    [
        (np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])),
        (np.array([1, 0, 1]), np.array([1, 0, 1])),
        (np.array([1, 1, 1, 1, 1]), np.array([1, 1, 1, 1, 1])),
        (np.array([1, 1, 0, 1, 1, 0, 0, 1, 0]), np.array([1, 1, 0, 1, 1, 0, 0, 1, 0])),
    ],
)
def test_get_jw_state(occupation: NDArray[np.int_], expected: NDArray[np.int_]):
    """Validate Jordan-Wigner basis state mapping."""
    actual = get_jw_state(occupation)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("occupation", "expected"),
    [
        (np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])),
        (np.array([1, 0, 1]), np.array([1, 1, 0])),
        (np.array([1, 1, 1, 1, 1]), np.array([1, 0, 1, 0, 1])),
        (np.array([1, 1, 0, 1, 1, 0, 0, 1, 0]), np.array([1, 0, 0, 1, 0, 0, 0, 1, 1])),
    ],
)
def test_get_parity_state(occupation: NDArray[np.int_], expected: NDArray[np.int_]):
    """Validate Parity basis state mapping."""
    actual = get_parity_state(occupation)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("occupation", "expected"),
    [
        (np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])),
        (np.array([1, 0, 1]), np.array([1, 1, 1])),
        (np.array([1, 1, 1, 1, 1]), np.array([1, 0, 1, 0, 1])),
        (np.array([1, 1, 0, 1, 1, 0, 0, 1, 0]), np.array([1, 0, 0, 1, 1, 1, 0, 1, 0])),
    ],
)
def test_get_bk_state(occupation: NDArray[np.int_], expected: NDArray[np.int_]):
    """Validate Bravyi-Kitaev basis state mapping."""
    actual = get_bk_state(occupation)
    np.testing.assert_array_equal(actual, expected)


class TestHartreeFockState:
    """Test HartreeFockState class."""

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
            HartreeFockState.closed_shell(num_electrons, num_spin_orbitals)

    @pytest.mark.parametrize("occupation", [(1, 2, 1, 1), (1, -1, 0), (1, "a", 0)])
    def test_invalid(self, occupation: tuple):
        with pytest.raises(ValueError, match="occupation must contain binary entries"):
            HartreeFockState(occupation)
