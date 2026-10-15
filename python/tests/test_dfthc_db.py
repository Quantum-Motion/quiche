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

import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from quiche.io import dfthc_db


@pytest.fixture
def path_to_dfthc_input_data() -> str:
    """Path to the DFTHC database test file."""
    return str(Path(__file__).parent / "dfthc_data" / "sq_results.db")


def test_missing_file(tmp_path: Path):
    """Test behaviour when input file is missing."""
    path = tmp_path / "missing.db"

    with pytest.raises(FileNotFoundError, match="Database file not found"):
        dfthc_db.read(path, 1)

    assert not path.exists()  # Opening must not create the file.


def test_unknown_job_id(path_to_dfthc_input_data: str):
    """Test behaviour when wrong job ID is given."""
    with pytest.raises(ValueError, match="No job with id 999"):
        dfthc_db.read(path_to_dfthc_input_data, 999)


def test_job_not_completed(path_to_dfthc_input_data: str, tmp_path: Path):
    """Test behaviour when tryiong to read from an incomplete job."""
    path = tmp_path / "failed.db"
    shutil.copy(path_to_dfthc_input_data, path)
    with closing(sqlite3.connect(path)) as con, con:
        con.execute("UPDATE experiments SET status='failed' WHERE id=1")

    with pytest.raises(ValueError, match="not 'completed'"):
        dfthc_db.read(path, 1)


def test_dfthc_db_parsing(path_to_dfthc_input_data: str):
    """Test the parsing of the database file."""
    dfthc_hamiltonian = dfthc_db.read(path_to_dfthc_input_data, 1)

    # The data is calculated using external partner's code.
    # The system is a H_3 chain (3 electrons),
    # in a double zeta basis set (5 AO's per H).
    # RBC values are taken from the input of their code.
    # Integral value references are taken from their factorisation.
    assert dfthc_hamiltonian.num_orbitals == 15
    assert dfthc_hamiltonian.num_ranks == 1
    assert dfthc_hamiltonian.num_bases == 40
    assert dfthc_hamiltonian.num_copies == 40
    assert dfthc_hamiltonian.num_electrons == 3
    np.testing.assert_allclose(dfthc_hamiltonian.const, 1.250005222563424, rtol=1e-8)
    np.testing.assert_array_equal(dfthc_hamiltonian.unit_vectors.shape, (1, 40, 15))
    np.testing.assert_array_equal(dfthc_hamiltonian.weight_vectors.shape, (1, 40, 40))
