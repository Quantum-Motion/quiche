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

"""State preparation methods for the simulation backend."""

from quiche.bindings.quest_bindings import Qureg
from quiche.bindings.quiche_bindings import (
    getHartreeFockStateBK,
    getHartreeFockStateJW,
    getHartreeFockStateParity,
)

# ruff: noqa: N802


def initHartreeFockStateBK(
    qureg: Qureg,
    num_electrons: int,
    num_qubits: int,
) -> None:
    """Initialise a Qureg to the HF state in the Bravyi-Kitaev mapping."""
    hf = getHartreeFockStateBK(num_electrons, num_qubits)
    qureg.initClassicalState(hf)


def initHartreeFockStateJW(
    qureg: Qureg,
    num_electrons: int,
) -> None:
    """Initialise a Qureg to the HF state in the Jordan-Wigner mapping."""
    hf = getHartreeFockStateJW(num_electrons)
    qureg.initClassicalState(hf)


def initHartreeFockStateParity(
    qureg: Qureg,
    num_electrons: int,
    num_qubits: int,
) -> None:
    """Initialise a Qureg to the HF state in the parity mapping."""
    hf = getHartreeFockStateParity(num_electrons, num_qubits)
    qureg.initClassicalState(hf)
