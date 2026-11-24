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

"""Helpers for building QuEST simulation operations."""

from collections.abc import Callable
from typing import Concatenate

from quiche.bindings.quest_bindings import PauliStrSum, Qureg
from quiche.core import PauliSum

from .routine import QuregOp

type QuESTBinding = Callable[Concatenate[Qureg, PauliStrSum, ...], float | None]


def _with_quest_hamiltonian(
    binding: QuESTBinding,
    hamiltonian: PauliSum,
    **kwargs: object,
) -> QuregOp:
    """Defer PauliSum to PauliStrSum conversion until run inside a QuESTEnv."""

    def op(qureg: Qureg) -> float | None:
        quest_sum = hamiltonian.to_quest()

        try:
            result = binding(qureg, quest_sum, **kwargs)
        finally:
            quest_sum.destroy()

        return result

    return op
