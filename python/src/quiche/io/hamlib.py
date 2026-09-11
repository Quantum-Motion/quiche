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

"""Helpers to read and parse Hamlib Hamiltonians."""

from pathlib import Path

import h5py
from openfermion import QubitOperator

from quiche.core import PauliSum

from ._openfermion import _qubit_operator_to_pauli_sum


def read_dataset(path: str | Path, key: str) -> str:
    """Read and decode a dataset from a Hamlib HDF5 file."""
    with h5py.File(path, "r") as file:
        data = file[key][()]

    return data.decode("utf-8")


def parse(text: str) -> PauliSum:
    """Parse a Hamlib Hamiltonian string into a PauliSum."""
    operator = QubitOperator(text)
    return _qubit_operator_to_pauli_sum(operator)
