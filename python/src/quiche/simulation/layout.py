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

from pydantic import NonNegativeInt, PositiveInt
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class QuregLayout:
    """Contiguous set of qubit registers QuEST simulations of QPE calculations."""

    num_data: PositiveInt
    num_qpe: PositiveInt
    num_index: NonNegativeInt = 0

    @property
    def data(self) -> range:
        """Qubit indices of the data register."""
        return range(self.num_data)

    @property
    def qpe(self) -> range:
        """Qubit indices of the QPE ancilla register."""
        return range(self.num_data, self.num_data + self.num_qpe)

    @property
    def index(self) -> range:
        """Qubit indices of the index ancilla register."""
        start = self.num_data + self.num_qpe
        return range(start, start + self.num_index)

    @property
    def num_qubits(self) -> int:
        """Total width of the Qureg."""
        return self.num_data + self.num_qpe + self.num_index
