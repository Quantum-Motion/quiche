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

"""Structures to define and dispatch initial state preparation."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial

import numpy as np
from numpy.typing import NDArray
from pydantic.dataclasses import dataclass
from qualtran import Bloq

from quiche.bindings.quiche_bindings import initClassicalState
from quiche.chemistry import (
    HartreeFockState,
    get_bk_state,
    get_jw_state,
    get_parity_state,
)
from quiche.core import Mapping
from quiche.qualtran.bloqs import BitstringStatePrep


class StateSpec(ABC):
    """
    Interface for objects describing how to prepare a circuit's initial state.

    Concrete implementations must expose `num_qubits` as an `@property` (not a plain
    dataclass field): a dataclass field without a default only becomes an instance
    attribute set in `__init__`, which does not satisfy an abstract property and would
    leave the class un-instantiable.
    """

    @property
    @abstractmethod
    def num_qubits(self) -> int:
        """Number of qubits this spec prepares state for."""

    @abstractmethod
    def to_qualtran(self) -> Bloq:
        """Build the Qualtran Bloq preparing this state."""

    @abstractmethod
    def to_quest(self) -> Callable:
        """Build the QuEST routine preparing this state."""


@dataclass(frozen=True)
class HartreeFockSpec(StateSpec):
    """State-preparation spec for a Hartree-Fock reference state."""

    state: HartreeFockState
    mapping: Mapping

    @property
    def num_qubits(self) -> int:
        """Get number of qubits prepared by this spec."""
        return self.state.num_spin_orbitals

    def to_qualtran(self) -> Bloq:
        """Build the Qualtran Bloq preparing the Hartree-Fock state."""
        return BitstringStatePrep(tuple(self._bitstring()))

    def to_quest(self) -> Callable:
        """Build the QuEST routine preparing the Hartree-Fock state."""
        return partial(initClassicalState, state=self._bitstring())

    def _bitstring(self) -> NDArray[np.int_]:
        """Transform the occupation basis state to the qubit basis via `mapping`."""
        match self.mapping:
            case Mapping.JordanWigner:
                return get_jw_state(self.state.occupation)
            case Mapping.BravyiKitaev:
                return get_bk_state(self.state.occupation)
            case Mapping.Parity:
                return get_parity_state(self.state.occupation)
