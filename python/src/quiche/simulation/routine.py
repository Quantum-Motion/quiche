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

"""Objects to construct and lazily evaluate QuEST simulations."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from quiche.bindings.quest_bindings import Qureg
from quiche.simulation.qubit_mapping import QubitMapping

if TYPE_CHECKING:
    from collections.abc import Callable


type QuestOp = Callable[[Qureg], object]


@dataclass(frozen=True)
class SimulationRoutine:
    """Sequence of QuEST ops, along with the register layout they assume."""

    mapping: QubitMapping
    ops: tuple[QuestOp, ...]

    @property
    def num_qubits(self) -> int:
        """Number of qubits needed to run the routine."""
        return self.mapping.num_qubits

    def run(self, qureg: Qureg | None = None) -> object:
        """Run on a fresh Qureg (or a given one) and return the estimated phase."""
        if qureg is None:
            qureg = Qureg(self.num_qubits)
        result = None
        for op in self.ops:
            result = op(qureg)
        return result
