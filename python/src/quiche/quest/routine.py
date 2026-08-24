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

from typing import TYPE_CHECKING

from quiche.bindings.quest_bindings import Qureg

if TYPE_CHECKING:
    from collections.abc import Callable


class QuestRoutine:
    """Sequence of callables acting on a Qureg for QuEST simulations."""

    def __init__(self) -> None:
        """QuestRoutine constructor."""
        self.ops: list[Callable] = []

    def evaluate(self, qureg: Qureg) -> list:
        """Execute the simulation."""
        return [op(qureg) for op in self.ops]

    def append(self, other: object) -> None:
        """Add another simulation function to the routine."""
        if callable(other):
            self.ops.append(other)
        else:
            msg = "Expected callable object, got non-callable."
            raise TypeError(msg)

    def extend(self, other: object) -> None:
        """Add another simulation routine to this routine."""
        if isinstance(other, QuestRoutine):
            self.ops.extend(other.ops)
        else:
            msg = f"Expected 'QuestRoutine' object, got '{type(other)}'."
            raise TypeError(msg)
