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


from openfermion import QubitOperator

from quiche.core.paulis import Pauli, PauliSum, PauliWord


def _qubit_operator_to_pauli_sum(operator: QubitOperator) -> PauliSum:
    """Convert an openfermion `QubitOperator` into a `PauliSum`."""
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
