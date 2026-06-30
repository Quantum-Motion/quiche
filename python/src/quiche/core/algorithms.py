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

"""Structures for specifying quantum algorithms and routines."""

from enum import StrEnum, auto


class PhaseEstimation(StrEnum):
    """Quantum algorithm to perform phase estimation."""

    # Naive QPE: single-ancilla, using C-U gate.
    Naive = auto()

    # Kitaev QPE: single-ancilla, using C-U^(2k) gates.
    Kitaev = auto()

    # Iterative QPE: single-ancilla, using C-U^(2k) gates and Rz for feedback.
    Iterative = auto()

    # Textbook QPE: multi-ancilla, using C-U^(2k) gates and inverse QFT.
    Textbook = auto()


class Simulation(StrEnum):
    """Hamiltonian simulation algorithm for phase estimation algorithms."""

    QDRIFT = auto()
    Qubitised = auto()
    Trotter = auto()


class Mapping(StrEnum):
    """Fermion-to-Qubit mappings."""

    BravyiKitaev = auto()
    JordanWigner = auto()
    Parity = auto()
