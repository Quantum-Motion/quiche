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

"""Hamiltonian simulation algorithms."""

from abc import ABC, abstractmethod
from math import ceil, cos, log2, pi
from typing import ClassVar, Literal, override

from pydantic import PositiveFloat, PositiveInt, field_validator
from pydantic.dataclasses import dataclass

from quiche.algorithms.block_encoding_prep import BlockEncodingPrep
from quiche.core.paulis import PauliSum

type SimulationAlgorithm = Trotter | QDRIFT | Qubitisation


class _SimulationBase(ABC):
    """Interface shared by Hamiltonian simulation algorithms."""

    @property
    @abstractmethod
    def num_measurement_bases(self) -> Literal[1, 2]:
        """
        Number of Hadamard-test measurement bases needed to recover the phase.

        Two for sign and magnitude, one for magnitude only.
        """
        ...

    @property
    @abstractmethod
    def num_ancillas(self) -> int:
        """Number of ancillas used by the Hamiltonian simulation."""
        ...

    @abstractmethod
    def phase_to_energy(self, phase: float) -> float:
        """Convert a phase from this propagator into an energy."""
        ...


@dataclass(frozen=True)
class Trotter(_SimulationBase):
    """
    Trotterised Hamiltonian simulation.

    Parameters
    ----------
    hamiltonian : PauliSum
        Hamiltonian to simulate.
    time : float
        Total evolution time. Must be positive.
    order : int
        Order of the Trotter product formula. Must be 1 or a positive even integer.
    num_steps : int
        Number of Trotter steps the evolution is split into. Must be positive.

    Attributes
    ----------
    num_measurement_bases : Literal[1, 2]
        Trotter energies are linear in phase, so both sign and magnitude are required.
    """

    hamiltonian: PauliSum
    time: PositiveFloat
    order: PositiveInt
    num_steps: PositiveInt

    num_measurement_bases: ClassVar[Literal[1, 2]] = 2

    @field_validator("order")
    @classmethod
    def _check_order(cls, order: int) -> int:
        """Trotter product formulas only exist for order 1 and even orders."""
        if order > 1 and order % 2:
            error_msg = "Trotter order must be 1 or even."
            raise ValueError(error_msg)
        return order

    @property
    @override
    def num_ancillas(self) -> int:
        """Number of ancillas used by the Hamiltonian simulation."""
        return 0

    @override
    def phase_to_energy(self, phase: float) -> float:
        """Convert a phase from this propagator into an energy."""
        return phase * (2 * pi / self.time)


@dataclass(frozen=True)
class QDRIFT(_SimulationBase):
    """
    QDRIFT Hamiltonian simulation.

    Parameters
    ----------
    hamiltonian : PauliSum
        Hamiltonian to simulate.
    time : float
        Total evolution time. Must be positive.
    num_samples : int
        Number of randomised samples the evolution is split into. Must be positive.
    seed : int
        Seed used for random sampling.

    Attributes
    ----------
    num_measurement_bases : Literal[1, 2]
        QDRIFT energies are linear in phase, so both sign and magnitude are required.
    """

    hamiltonian: PauliSum
    time: PositiveFloat
    num_samples: PositiveInt
    seed: PositiveInt

    num_measurement_bases: ClassVar[Literal[1, 2]] = 2

    @property
    @override
    def num_ancillas(self) -> int:
        """Number of ancillas used by the Hamiltonian simulation."""
        return 0

    @override
    def phase_to_energy(self, phase: float) -> float:
        """Convert a phase from this propagator into an energy."""
        return phase * (2 * pi / self.time)


@dataclass(frozen=True)
class Qubitisation(_SimulationBase):
    """
    Qubitised Hamiltonian propagator.

    Parameters
    ----------
    hamiltonian : PauliSum
        Hamiltonian to simulate.
    prep : BlockEncodingPrep
        State preparation algorithm to use for block encoding.

    Attributes
    ----------
    num_measurement_bases : Literal[1, 2]
        Qubitisation energies require only the magnitude of the phase.
    """

    hamiltonian: PauliSum
    prep: BlockEncodingPrep

    num_measurement_bases: ClassVar[Literal[1, 2]] = 1

    @property
    def num_index_ancillas(self) -> int:
        """Number of ancillas used to specify each term index in the sum."""
        return max(1, ceil(log2(self.hamiltonian.num_words_with_identity)))

    @property
    def num_prep_ancillas(self) -> int:
        """Number of ancillas used to implement PREP oracle."""
        return self.prep.num_ancillas(self.num_index_ancillas)

    @property
    @override
    def num_ancillas(self) -> int:
        """Number of ancillas used by the block encoding."""
        return self.num_index_ancillas + self.num_prep_ancillas

    @override
    def phase_to_energy(self, phase: float) -> float:
        """Convert a phase from this propagator into an energy."""
        return cos(phase * 2 * pi) * self.hamiltonian.lam
