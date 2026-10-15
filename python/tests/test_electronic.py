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

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from quiche.core import SecondQuantisedHamiltonian
from quiche.core.electronic import DFTHCHamiltonian
from quiche.io import dfthc_db


@pytest.fixture
def factors() -> dict:
    """Return a consistent set of DFTHC factors (ones, so no zero vectors)."""
    norb, ranks, bases, copies = 3, 2, 4, 2
    return {
        "num_orbitals": norb,
        "num_ranks": ranks,
        "num_bases": bases,
        "num_copies": copies,
        "unit_vectors": np.ones((ranks, bases, norb)),
        "weight_vectors": np.ones((ranks, bases, copies)),
        "bliss_matrix": np.ones((norb, norb)),
        "factorised_one_body": np.ones((norb, norb)),
        "const": 0.5,
        "num_electrons": 2,
    }


@pytest.fixture
def path_to_dfthc_input_data() -> str:
    """Path to the DFTHC database test file."""
    return str(Path(__file__).parent / "dfthc_data" / "sq_results.db")


@pytest.fixture
def path_to_dfthc_one_body_reference() -> str:
    """Return the path to the reference reconstructed one-body integrals."""
    return str(Path(__file__).parent / "dfthc_data" / "reconstructed_one_body.npy")


@pytest.fixture
def path_to_dfthc_two_body_reference() -> str:
    """Return the path to the reference reconstructed two-body integrals."""
    return str(Path(__file__).parent / "dfthc_data" / "reconstructed_two_body.npy")


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
                num_electrons=0,
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
                num_electrons=0,
            )

    def test_mismatched_shapes(self):
        error_msg = "two body integrals must have the same orbital dimension"
        with pytest.raises(ValueError, match=error_msg):
            SecondQuantisedHamiltonian(
                one_body=np.empty((2, 2)),
                two_body=np.empty((3,) * 4),
                num_electrons=0,
            )

    @pytest.mark.parametrize(
        ("num_electrons", "error_msg"),
        [
            (-1, "number of electrons must be non-negative"),
            (5, "number of electrons must not exceed the number of spin orbitals"),
        ],
    )
    def test_invalid_num_electrons(self, num_electrons: int, error_msg: str):
        one_body = np.empty((2, 2))
        two_body = np.empty((2,) * 4)
        with pytest.raises(ValueError, match=error_msg):
            SecondQuantisedHamiltonian(
                one_body=one_body,
                two_body=two_body,
                num_electrons=num_electrons,
            )

    @pytest.mark.parametrize("norb", [1, 2, 5])
    def test_orbital_counts(self, norb: int):
        one_body = np.empty((norb, norb))
        two_body = np.empty((norb,) * 4)
        hamiltonian = SecondQuantisedHamiltonian(
            one_body=one_body,
            two_body=two_body,
            num_electrons=2,
        )

        assert hamiltonian.num_spatial_orbitals == norb
        assert hamiltonian.num_spin_orbitals == 2 * norb


class TestDFTHCHamiltonian:
    """Test DFTHCHamiltonian class."""

    @pytest.mark.parametrize(
        ("name", "shape"),
        [
            ("unit_vectors", (2, 4, 5)),
            ("weight_vectors", (2, 5, 2)),
            ("bliss_matrix", (3, 4)),
            ("factorised_one_body", (2, 2)),
        ],
    )
    def test_invalid_shapes(self, factors: dict, name: str, shape: tuple[int, ...]):
        factors[name] = np.ones(shape)

        with pytest.raises(ValueError, match=f"{name} has shape"):
            DFTHCHamiltonian(**factors)

    def test_zero_unit_vector(self, factors: dict):
        factors["unit_vectors"][0, 0, :] = 0.0

        with pytest.raises(ValueError, match="contains a zero vector"):
            DFTHCHamiltonian(**factors)

    def test_reconstruct(
        self,
        path_to_dfthc_input_data: str,
        path_to_dfthc_one_body_reference: str,
        path_to_dfthc_two_body_reference: str,
    ):
        hamiltonian = dfthc_db.load(path_to_dfthc_input_data, 1)
        reconstructed = hamiltonian.reconstruct()

        assert isinstance(reconstructed, SecondQuantisedHamiltonian)
        assert reconstructed.num_spatial_orbitals == 15
        assert reconstructed.num_electrons == hamiltonian.num_electrons
        assert reconstructed.core_energy == hamiltonian.const
        assert reconstructed.one_body.shape == (15, 15)
        assert reconstructed.two_body.shape == (15,) * 4
        assert np.testing.assert_allclose(
            reconstructed.one_body, np.load(path_to_dfthc_one_body_reference), atol=1e-6
        )
        assert np.testing.assert_allclose(
            reconstructed.two_body, np.load(path_to_dfthc_two_body_reference), atol=1e-6
        )
