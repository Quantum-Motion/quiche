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

"""Common pytest fixtures and setup for test suite."""

from pathlib import Path

import pytest

from quiche.core import Errors, PauliSum
from quiche.hamlib import get_dataset, parse_hamiltonian

HAMLIB_DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def h2() -> PauliSum:
    """H2 Hamiltonian (smallest JW dataset, chosen to keep tests fast)."""
    raw = get_dataset(str(HAMLIB_DATA_DIR / "H2.hdf5"), "ham_JW-4")
    return parse_hamiltonian(raw)


@pytest.fixture
def budget() -> Errors:
    """Default error budget for test cases."""
    tot_error = 0.16
    return Errors(
        estimation=tot_error / 3,
        simulation=1.0,
        rotations=tot_error / 3,
        state_prep=tot_error / 3,
        overlap=1,
    )
