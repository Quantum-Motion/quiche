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

import pytest

from quiche.chemistry import (
    Mapping,
    get_bk_state,
    get_hf_state,
    get_jw_state,
    get_parity_state,
    get_qubit_state,
)


@pytest.mark.parametrize(
    ("num_electrons", "num_spin_orbitals", "error_msg"),
    [
        (-1, 1, "electrons must be non-negative"),
        (0, 0, "spin orbitals must be positive"),
        (4, 3, "electrons must not exceed number of spin orbitals"),
    ],
)
def test_get_hf_state_invalid_cases(
    num_electrons: int,
    num_spin_orbitals: int,
    error_msg: str,
):
    """Test input validation for Hartree-Fock basis state."""
    with pytest.raises(ValueError, match=error_msg):
        get_hf_state(num_electrons, num_spin_orbitals)


@pytest.mark.parametrize(
    ("num_electrons", "num_spin_orbitals", "expected"),
    [
        (1, 1, (1,)),
        (2, 5, (1, 1, 0, 0, 0)),
        (0, 3, (0, 0, 0)),
    ],
)
def test_get_hf_state_state(
    num_electrons: int, num_spin_orbitals: int, expected: tuple
):
    """Validate Hartree-Fock basis state."""
    actual = get_hf_state(num_electrons, num_spin_orbitals)
    assert actual == expected


@pytest.mark.parametrize(
    ("occupation", "expected"),
    [
        ((0, 0, 0, 0), (0, 0, 0, 0)),
        ((1, 0, 1), (1, 0, 1)),
        ((1, 1, 1, 1, 1), (1, 1, 1, 1, 1)),
        ((1, 1, 0, 1, 1, 0, 0, 1, 0), (1, 1, 0, 1, 1, 0, 0, 1, 0)),
    ],
)
def test_get_jw_state(occupation: tuple, expected: tuple):
    """Validate Jordan-Wigner basis state mapping."""
    actual = get_jw_state(occupation)
    assert actual == expected


@pytest.mark.parametrize(
    ("occupation", "expected"),
    [
        ((0, 0, 0, 0), (0, 0, 0, 0)),
        ((1, 0, 1), (1, 1, 0)),
        ((1, 1, 1, 1, 1), (1, 0, 1, 0, 1)),
        ((1, 1, 0, 1, 1, 0, 0, 1, 0), (1, 0, 0, 1, 0, 0, 0, 1, 1)),
    ],
)
def test_get_parity_state(occupation: tuple, expected: tuple):
    """Validate Parity basis state mapping."""
    actual = get_parity_state(occupation)
    assert actual == expected


@pytest.mark.parametrize(
    ("occupation", "expected"),
    [
        ((0, 0, 0, 0), (0, 0, 0, 0)),
        ((1, 0, 1), (1, 1, 1)),
        ((1, 1, 1, 1, 1), (1, 0, 1, 0, 1)),
        ((1, 1, 0, 1, 1, 0, 0, 1, 0), (1, 0, 0, 1, 1, 1, 0, 1, 0)),
    ],
)
def test_get_bk_state(occupation: tuple, expected: tuple):
    """Validate Bravyi-Kitaev basis state mapping."""
    actual = get_bk_state(occupation)
    assert actual == expected


@pytest.mark.parametrize(
    ("mapping", "expected"),
    [
        (Mapping.BravyiKitaev, (1, 1, 1)),
        (Mapping.JordanWigner, (1, 0, 1)),
        (Mapping.Parity, (1, 1, 0)),
    ],
)
def test_get_qubit_state(
    mapping: Mapping,
    expected: tuple,
):
    """Validate the generic fermion to qubit basis state mapping."""
    actual = get_qubit_state((1, 0, 1), mapping)
    assert actual == expected


def test_get_qubit_state_invalid():
    """Validate invalid inputs for the generic fermion to qubit basis state mapping."""
    with pytest.raises(AssertionError):
        get_qubit_state([0, 1, 0], "not_a_mapping")  # ty: ignore[invalid-argument-type]
