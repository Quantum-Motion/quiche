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

from collections.abc import Callable
from dataclasses import dataclass

from quiche.bindings.quest_bindings import Qureg
from quiche.simulation.layout import QuregLayout

type QuESTOp = Callable[[Qureg], object]


@dataclass(frozen=True)
class QuESTRoutine:
    """Sequence of QuEST ops, along with the register layout they assume."""

    ops: tuple[QuESTOp, ...]
    layout: QuregLayout

    @property
    def num_qubits(self) -> int:
        """Number of qubits needed to run the routine."""
        return self.layout.num_qubits

    def run(self) -> object:
        """Run on a clean Qureg and return the estimated phase."""
        qureg = Qureg(self.num_qubits)

        result = None
        try:
            for op in self.ops:
                result = op(qureg)
        finally:
            qureg.destroy()

        return result
