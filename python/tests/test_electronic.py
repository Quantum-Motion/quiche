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

"""Tests for electronic module."""

import numpy as np
import pytest
from numpy.typing import NDArray

from quiche.core import SecondQuantisedHamiltonian


class TestSecondQuantisedHamiltonian:
    """Test SecondQuantisedHamiltonian class."""

    @pytest.mark.parametrize(
        ("one_body", "two_body", "error_msg"),
        [
            (
                np.empty(2),
                np.empty((2,) * 4),
                "one body integrals must be a 2-dimensional tensor",
            ),
            (
                np.empty((2, 2)),
                np.empty((2, 2)),
                "two body integrals must be a 4-dimensional tensor",
            ),
        ],
    )
    def test_invalid_dims(
        self,
        one_body: NDArray[np.float64],
        two_body: NDArray[np.float64],
        error_msg: str,
    ):
        with pytest.raises(ValueError, match=error_msg):
            SecondQuantisedHamiltonian(
                one_body=one_body,
                two_body=two_body,
                electrons=0,
            )

    @pytest.mark.parametrize(
        ("shape", "error_msg"),
        [
            ((2, 3), "one body integrals must be square"),
            ((0, 0), "number of spatial orbitals must be nonzero"),
        ],
    )
    def test_invalid_shapes(self, shape: tuple[int, int], error_msg: str):
        with pytest.raises(ValueError, match=error_msg):
            SecondQuantisedHamiltonian(
                one_body=np.empty(shape),
                two_body=np.empty((shape[0],) * 4),
                electrons=0,
            )

    def test_mismatched_shapes(self):
        error_msg = "two body integrals must have the same orbital dimension"
        with pytest.raises(ValueError, match=error_msg):
            SecondQuantisedHamiltonian(
                one_body=np.empty((2, 2)),
                two_body=np.empty((3,) * 4),
                electrons=0,
            )

    @pytest.mark.parametrize(
        ("electrons", "error_msg"),
        [
            (-1, "number of electrons must be non-negative"),
            (5, "number of electrons must not exceed the number of spin orbitals"),
        ],
    )
    def test_invalid_electrons(self, electrons: int, error_msg: str):
        one_body = np.empty((2, 2))
        two_body = np.empty((2,) * 4)
        with pytest.raises(ValueError, match=error_msg):
            SecondQuantisedHamiltonian(
                one_body=one_body,
                two_body=two_body,
                electrons=electrons,
            )

    @pytest.mark.parametrize("norb", [1, 2, 5])
    def test_orbital_counts(self, norb: int):
        one_body = np.empty((norb, norb))
        two_body = np.empty((norb,) * 4)
        hamiltonian = SecondQuantisedHamiltonian(
            one_body=one_body,
            two_body=two_body,
            electrons=2,
        )

        assert hamiltonian.num_spatial_orbitals == norb
        assert hamiltonian.num_spin_orbitals == 2 * norb
