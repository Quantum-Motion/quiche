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

"""Conversions between quiche and openfermion representations."""

import numpy as np
from openfermion import InteractionOperator, QubitOperator
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.transforms import (
    binary_code_transform,
    bravyi_kitaev,
    get_fermion_operator,
    jordan_wigner,
    parity_code,
)

from quiche.core import (
    ElectronicHamiltonian,
    Mapping,
    Pauli,
    PauliSum,
    PauliWord,
    SecondQuantisedHamiltonian,
)


def _qubit_operator_to_pauli_sum(operator: QubitOperator) -> PauliSum:
    """Convert an openfermion QubitOperator into a PauliSum."""
    identity_coefficient = operator.constant
    if identity_coefficient.imag:
        error_msg = f"Complex identity coefficient: {identity_coefficient}."
        raise ValueError(error_msg)

    coefficients = []
    pauli_words = []

    for term, coefficient in operator.terms.items():
        # Skip identity term
        if not term:
            continue

        if coefficient.imag:
            error_msg = f"Complex coefficient: {coefficient}."
            raise ValueError(error_msg)

        if coefficient.real == 0.0:
            continue

        qubits, paulis = zip(*term, strict=True)

        coefficients.append(coefficient.real)
        pauli_words.append(
            PauliWord(terms=tuple(Pauli(pauli) for pauli in paulis), qubits=qubits)
        )

    return PauliSum(
        coefficients=tuple(coefficients),
        terms=tuple(pauli_words),
        identity_coefficient=identity_coefficient.real,
    )


def _second_quantised_to_interaction_operator(
    hamiltonian: SecondQuantisedHamiltonian,
) -> InteractionOperator:
    """Convert a SecondQuantisedHamiltonian into an openfermion InteractionOperator."""
    # Map from chemist to openfermion ordering
    reordered_two_body = np.einsum("pqrs->prsq", hamiltonian.two_body)

    one_body, two_body = spinorb_from_spatial(hamiltonian.one_body, reordered_two_body)
    return InteractionOperator(hamiltonian.core_energy, one_body, two_body / 2)


def _interaction_to_qubit_operator(
    operator: InteractionOperator,
    mapping: Mapping,
    num_spin_orbitals: int,
) -> QubitOperator:
    """Apply a fermion-to-qubit mapping to an openfermion InteractionOperator."""
    match mapping:
        case Mapping.JordanWigner:
            return jordan_wigner(operator)
        case Mapping.BravyiKitaev:
            return bravyi_kitaev(operator, n_qubits=num_spin_orbitals)
        case Mapping.Parity:
            return binary_code_transform(
                get_fermion_operator(operator), parity_code(num_spin_orbitals)
            )


def _second_quantised_to_electronic_hamiltonian(
    hamiltonian: SecondQuantisedHamiltonian,
    mapping: Mapping,
) -> ElectronicHamiltonian:
    """Convert a SecondQuantisedHamiltonian into an ElectronicHamiltonian."""
    interaction_operator = _second_quantised_to_interaction_operator(hamiltonian)
    qubit_operator = _interaction_to_qubit_operator(
        interaction_operator,
        mapping,
        hamiltonian.num_spin_orbitals,
    )
    paulis = _qubit_operator_to_pauli_sum(qubit_operator)

    return ElectronicHamiltonian(
        electrons=hamiltonian.electrons,
        mapping=mapping,
        paulis=paulis,
    )
