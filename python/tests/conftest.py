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

import pytest

from quiche.core import Errors, PauliSum, PauliWord


@pytest.fixture(scope="session")
def h2() -> PauliSum:
    """Minimal basis H2 Hamiltonian in Jordan-Wigner mapping."""
    terms = (
        PauliWord(terms=("X", "X", "Y", "Y"), qubits=(0, 1, 2, 3)),
        PauliWord(terms=("X", "Y", "Y", "X"), qubits=(0, 1, 2, 3)),
        PauliWord(terms=("Y", "X", "X", "Y"), qubits=(0, 1, 2, 3)),
        PauliWord(terms=("Y", "Y", "X", "X"), qubits=(0, 1, 2, 3)),
        PauliWord(terms=("Z", "Z"), qubits=(0, 1)),
        PauliWord(terms=("Z", "Z"), qubits=(0, 2)),
        PauliWord(terms=("Z", "Z"), qubits=(0, 3)),
        PauliWord(terms=("Z", "Z"), qubits=(1, 2)),
        PauliWord(terms=("Z", "Z"), qubits=(1, 3)),
        PauliWord(terms=("Z", "Z"), qubits=(2, 3)),
        PauliWord(terms=("Z",), qubits=(0,)),
        PauliWord(terms=("Z",), qubits=(1,)),
        PauliWord(terms=("Z",), qubits=(2,)),
        PauliWord(terms=("Z",), qubits=(3,)),
    )

    coeffs = (
        -0.045322,
        +0.045322,
        +0.045322,
        -0.045322,
        +0.168622,
        +0.120545,
        +0.165867,
        +0.165867,
        +0.120545,
        +0.174348,
        +0.171198,
        +0.171198,
        -0.222786,
        -0.222786,
    )

    id_coeff = -0.098864

    return PauliSum(coefficients=coeffs, terms=terms, identity_coefficient=id_coeff)


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
