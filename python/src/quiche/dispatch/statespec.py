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

"""State-preparation specs implementing the common `Spec` interface."""

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
from quiche.cudaq import CudaqKernel, bitstring_kernel
from quiche.dispatch.spec import Spec
from quiche.qualtran.bloqs import BitstringStatePrep
from quiche.quest import QuestRoutine


@dataclass(frozen=True)
class HartreeFockSpec(Spec):
    """State-preparation spec for a Hartree-Fock reference state."""

    state: HartreeFockState
    mapping: Mapping

    def to_qualtran(self) -> Bloq:
        """Build the Qualtran Bloq preparing the Hartree-Fock state."""
        return BitstringStatePrep(tuple(self._bitstring()))

    def to_quest(self) -> QuestRoutine:
        """Build the QuEST routine preparing the Hartree-Fock state."""
        routine = QuestRoutine()
        routine.append(partial(initClassicalState, state=self._bitstring()))
        return routine

    def to_cudaq(self) -> CudaqKernel:
        """Build the CUDA-Q kernel preparing the Hartree-Fock state."""
        return bitstring_kernel(self._bitstring())

    def _bitstring(self) -> NDArray[np.int_]:
        """Transform the occupation basis state to the qubit basis via `mapping`."""
        match self.mapping:
            case Mapping.JordanWigner:
                return get_jw_state(self.state.occupation)
            case Mapping.BravyiKitaev:
                return get_bk_state(self.state.occupation)
            case Mapping.Parity:
                return get_parity_state(self.state.occupation)
