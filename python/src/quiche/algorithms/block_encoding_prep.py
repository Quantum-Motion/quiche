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

"""Block encoding PREP algorithms."""

from abc import ABC, abstractmethod
from typing import override

from pydantic import PositiveInt
from pydantic.dataclasses import dataclass


type BlockEncodingPrep = AliasSampling | ExactRotationTree | QROMRotationTree


class _BlockEncodingPrepBase(ABC):
    """Interface shared by block-encoding PREP algorithms."""

    @abstractmethod
    def num_ancillas(self, num_index_qubits: int) -> int: ...


@dataclass(frozen=True)
class AliasSampling(_BlockEncodingPrepBase):
    """
    Coherent alias sampling state preparation.

    Parameters
    ----------
    keep_bits : int
        Number of bits to use for the keep register. Must be positive.
    """

    keep_bits: PositiveInt

    @override
    def num_ancillas(self, num_index_qubits: int) -> int:
        """Get the number of ancillas used by the state preparation."""
        # sigma_mu (mu) + keep (mu) + alt (n) + less_than_equal flag (1)
        return 2 * self.keep_bits + num_index_qubits + 1


@dataclass(frozen=True)
class ExactRotationTree(_BlockEncodingPrepBase):
    """Rotation tree applying exact angles via multi-controlled rotations."""

    @override
    def num_ancillas(self, num_index_qubits: int) -> int:
        """Get the number of ancillas used by the state preparation."""
        return 0


@dataclass(frozen=True)
class QROMRotationTree(_BlockEncodingPrepBase):
    """
    Rotation tree loading finite-precision angles via QROM and phase-gradient register.

    Parameters
    ----------
    angle_bits : int
        Number of bits used to represent the angle. Must be positive.

    References
    ----------
    .. [Low2024] G. H. Low, V. Kliuchnikov, and L. Schaeffer, 'Trading T gates for dirty
       qubits in state preparation and unitary synthesis', Quantum, vol. 8, p. 1375,
       Jun. 2024, doi: 10.22331/q-2024-06-17-1375.
    """

    angle_bits: PositiveInt

    @override
    def num_ancillas(self, num_index_qubits: int) -> int:
        """Get the number of ancillas used by the state preparation."""
        return self.angle_bits
