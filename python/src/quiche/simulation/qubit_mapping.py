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

"""Layout of named qubit registers for QuEST simulations."""

from dataclasses import dataclass
from typing import Self


@dataclass(frozen=True)
class QubitMapping:
    """Mapping of named registers to contiguous qubit indices for QuEST simulations."""

    registers: dict[str, range]

    @classmethod
    def contiguous(cls, **sizes: int) -> Self:
        """Arrange registers contiguously, in argument order."""
        registers = {}
        start = 0
        for name, size in sizes.items():
            registers[name] = range(start, start + size)
            start += size
        return cls(registers)

    def __getitem__(self, name: str) -> range:
        """Get the qubit indices of the given register."""
        return self.registers[name]

    @property
    def num_qubits(self) -> int:
        """Overall width of the Qureg needed for all registers."""
        return max(
            (max(_range) + 1 for _range in self.registers.values() if _range),
            default=0,
        )
