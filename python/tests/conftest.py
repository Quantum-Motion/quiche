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
