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

"""Wrappers for phase estimation methods in the QuEST backend."""

# ruff: noqa: PLR0913, N802

from collections.abc import Sequence

from quiche.bindings import quiche_bindings as qb
from quiche.bindings.quest_bindings import Qureg
from quiche.core import PauliSum


def getPhaseKitaevQDRIFT(
    qureg: Qureg,
    hamiltonian: PauliSum,
    ancilla_index: int,
    reps: int,
    time: float,
    num_bits: int,
    seed: int | None,
) -> float:
    """Execute Kitaev phase estimation with QDRIFT."""
    quest_sum = hamiltonian.to_quest()
    return qb.getPhaseKitaevQDRIFT(
        qureg, quest_sum, ancilla_index, reps, time, num_bits, seed
    )


def getPhaseKitaevTrotter(
    qureg: Qureg,
    hamiltonian: PauliSum,
    ancilla_index: int,
    order: int,
    reps: int,
    time: float,
    num_bits: int,
) -> float:
    """Execute Kitaev phase estimation with Trotterisation."""
    quest_sum = hamiltonian.to_quest()
    return qb.getPhaseKitaevTrotter(
        qureg, quest_sum, ancilla_index, order, reps, time, num_bits
    )


def getPhaseTextbookQDRIFT(
    qureg: Qureg,
    hamiltonian: PauliSum,
    ancillas: Sequence[int],
    reps: int,
    time: float,
    seed: int | None,
) -> float:
    """Execute Textbook phase estimation with QDRIFT."""
    quest_sum = hamiltonian.to_quest()
    return qb.getPhaseTextbookQDRIFT(qureg, quest_sum, ancillas, reps, time, seed)


def getPhaseTextbookTrotter(
    qureg: Qureg,
    hamiltonian: PauliSum,
    ancillas: Sequence[int],
    order: int,
    reps: int,
    time: float,
) -> float:
    """Execute Textbook phase estimation with Trotterisation."""
    quest_sum = hamiltonian.to_quest()
    return qb.getPhaseTextbookTrotter(qureg, quest_sum, ancillas, order, reps, time)


def getPhaseTextbookQubitised(
    qureg: Qureg,
    hamiltonian: PauliSum,
    qpe_ancillas: Sequence[int],
    qubitisation_ancillas: Sequence[int],
) -> float:
    """Execute Textbook phase estimation with Qubitisation."""
    quest_sum = hamiltonian.to_quest()
    return qb.getPhaseTextbookQubitised(
        qureg, quest_sum, qpe_ancillas, qubitisation_ancillas
    )
