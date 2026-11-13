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

"""State prepration algorithms."""

from typing import Literal

from pydantic import PositiveInt
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class IdentityStateprep:
    """
    Identity (zero-state) state preparation.

    Attributes
    ----------
    num_qubits : int
        Number of qubits to initialise. Must be positive.

    """

    num_qubits: PositiveInt


@dataclass(frozen=True)
class BitstringStateprep:
    """
    Arbitrary computational basis state preparation.

    Attributes
    ----------
    bitstring : tuple[{0, 1}, ...]
        Bitstring specifiying the basis state to prepare.

    """

    bitstring: tuple[Literal[0, 1], ...]


StateprepAlgorithm = IdentityStateprep | BitstringStateprep
