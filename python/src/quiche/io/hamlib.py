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

"""Helpers to read and parse hamlib Hamiltonians."""

import re

import h5py

from quiche.core.paulis import Pauli, PauliSum, PauliWord


def get_dataset(filename: str, dataset: str) -> str:
    """Find and decode a Hamlib dataset."""
    with h5py.File(filename, "r") as f:
        data = f[dataset][()]

    return data.decode("utf-8")


def parse_hamiltonian(data_string: str) -> PauliSum:
    """
    Quick and dirty parser for hamlib Hamiltonians.

    NB: THIS IS A NAIVE, NON-DEFENSIVE IMPLEMENTATION. NOT FOR PRODUCTION USE.
    """
    coeffs_list = []
    paulis_list = []
    id_coeff = 0

    main_pattern = r"\(?(.*?)\)?\s+\[(.*?)\]"
    pauli_pattern = r"([XYZ])(\d+)"

    for line in data_string.strip().splitlines():
        main_match = re.match(main_pattern, line)

        if main_match:
            coeff_string = main_match.group(1).strip()
            pauli_string = main_match.group(2).strip()

            coeff = complex(coeff_string).real

            # Empty operator string `[]` corresponds to identity
            if not pauli_string:
                id_coeff = coeff
            else:
                coeffs_list.append(coeff)

                matches = re.findall(pauli_pattern, pauli_string)
                terms = tuple(Pauli(m[0]) for m in matches)
                qubits = tuple(int(m[1]) for m in matches)

                pw = PauliWord(terms=terms, qubits=qubits)

                paulis_list.append(pw)
        else:
            error_msg = "String parsing error: failed to match data."
            raise RuntimeError(error_msg)

    return PauliSum(
        coefficients=tuple(coeffs_list),
        terms=tuple(paulis_list),
        identity_coefficient=id_coeff,
    )
