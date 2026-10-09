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

"""Tests for the DFTHC database parsing."""

from pathlib import Path

import numpy as np
import pytest

from quiche.io import dfthc_db


@pytest.fixture(scope="session")
def db_input_file() -> str:
    """Path to the DFTHC database test file."""
    return str(Path(__file__).parent / "input" / "sq_results.db")


def test_dfthc_db_parsing(db_input_file: str):
    """Test the parsing of the database file."""
    factorised_hamiltonian = dfthc_db.parse(db_input_file, 1)

    # The data is calculated using external partner's code.
    # The system is a H_3 chain (3 electrons),
    # in a double zeta basis set (5 AO's per H).
    # RBC values are taken from the input of their code.
    # Integral value references are taken from their factorisation.
    assert factorised_hamiltonian.num_orbitals == 15
    assert factorised_hamiltonian.num_ranks == 1
    assert factorised_hamiltonian.num_bases == 40
    assert factorised_hamiltonian.num_copies == 40
    assert factorised_hamiltonian.num_electrons == 3
    assert np.isclose(factorised_hamiltonian.const, 1.250005222563424, 1e-8)
    assert np.allclose(factorised_hamiltonian.unit_vectors.shape, (1, 40, 15), 1e-16)
    assert np.allclose(factorised_hamiltonian.weight_vectors.shape, (1, 40, 40), 1e-16)
