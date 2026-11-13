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

from collections.abc import Sequence
from enum import Enum, auto
from math import ceil, cos, log2, pi
from typing import ClassVar, Literal, Self, assert_never

from pydantic import PositiveFloat, PositiveInt, field_validator
from pydantic.dataclasses import dataclass

from quiche.core.paulis import PauliSum

# TODO: detailed docstrings with error types


class PhaseInformation(Enum):
    """
    Information about the phase recovered from a Hadamard test.

    Attributes
    ----------
    SIGN_AND_MAGNITUDE : PhaseInformation
        Both the sign and magnitude of the phase, from the real and imaginary parts
        of the expectation value.
    MAGNITUDE_ONLY : PhaseInformation
        The magnitude of the phase only, from the real part of the expectation
        value. A phase cannot be distinguished from its negation.

    """

    SIGN_AND_MAGNITUDE = auto()
    MAGNITUDE_ONLY = auto()

    @property
    def num_measurement_bases(self) -> Literal[1, 2]:
        """Get the number of measurement bases needed for each Hadamard-test."""
        match self:
            case PhaseInformation.SIGN_AND_MAGNITUDE:
                return 2
            case PhaseInformation.MAGNITUDE_ONLY:
                return 1
            case _:
                assert_never(self)


@dataclass(frozen=True)
class Trotter:
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
    phase_information : PhaseInformation
        Phase information the propagator must resolve. Trotter energies are linear in
        the phase, so both sign and magnitude are required.

    """

    hamiltonian: PauliSum
    time: PositiveFloat
    order: PositiveInt
    num_steps: PositiveInt

    phase_information: ClassVar[PhaseInformation] = PhaseInformation.SIGN_AND_MAGNITUDE

    @field_validator("order")
    @classmethod
    def _check_order(cls, order: int) -> int:
        """Trotter product formulas only exist for order 1 and even orders."""
        if order > 1 and order % 2:
            error_msg = "Trotter order must be 1 or even."
            raise ValueError(error_msg)
        return order

    @property
    def num_ancillas(self) -> int:
        """Number of ancillas used by the Hamiltonian simulation."""
        return 0

    @classmethod
    def from_error(
        cls,
        hamiltonian: PauliSum,
        time: PositiveFloat,
        order: PositiveInt,
        error: PositiveFloat,
    ) -> Self:
        """Construct a Trotter instance given a target error."""
        return cls(
            hamiltonian=hamiltonian,
            time=time,
            order=order,
            # Corollary 2, Childs 2021
            num_steps=ceil(
                ((hamiltonian.lam * time) ** (1 + 1 / order)) / ((error) ** (1 / order))
            ),
        )

    def phase_to_energy(self, phase: float) -> float:
        """Convert a phase from this propagator into an energy."""
        return phase * (2 * pi / self.time)


@dataclass(frozen=True)
class QDRIFT:
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
    phase_information : PhaseInformation
        Phase information the propagator must resolve. QDRIFT energies are linear in
        the phase, so both sign and magnitude are required.

    """

    hamiltonian: PauliSum
    time: PositiveFloat
    num_samples: PositiveInt
    seed: PositiveInt

    phase_information: ClassVar[PhaseInformation] = PhaseInformation.SIGN_AND_MAGNITUDE

    @property
    def num_ancillas(self) -> int:
        """Number of ancillas used by the Hamiltonian simulation."""
        return 0

    @classmethod
    def from_failure_prob(
        cls,
        hamiltonian: PauliSum,
        time: PositiveFloat,
        seed: PositiveInt,
        failure_prob: PositiveFloat,
        powers: Sequence[PositiveInt],
    ) -> Self:
        """Construct a QDRIFT instance given a target maximum failure probability."""
        return cls(
            hamiltonian=hamiltonian,
            time=time,
            seed=seed,
            num_samples=cls.samples_from_failure_prob(
                hamiltonian, time, failure_prob, powers
            ),
        )

    @staticmethod
    def samples_from_failure_prob(
        hamiltonian: PauliSum,
        time: PositiveFloat,
        failure_prob: PositiveFloat,
        powers: Sequence[PositiveInt],
    ) -> int:
        # Fold diamond norm error into failure probability
        total_error = failure_prob

        total_time = sum(powers) * time
        error_fraction = total_error * time / total_time

        # Time and samples are for the lowest power, number of samples needs to be
        # multiplied by the same factor that the time is multiplied for higher powers
        lambda_no_identity = hamiltonian.without_identity().lam
        return ceil(2 * lambda_no_identity**2 * time**2 / error_fraction)

    def phase_to_energy(self, phase: float) -> float:
        """Convert a phase from this propagator into an energy."""
        return phase * (2 * pi / self.time)


@dataclass(frozen=True)
class AliasSampling:
    """
    Coherent alias sampling state preparation.

    Parameters
    ----------
    keep_bits : int
        Number of bits to use for the keep register. Must be positive.

    """

    keep_bits: PositiveInt

    def num_ancillas(self, num_levels: int) -> int:
        """Get the number of ancillas used by the state preparation."""
        # sigma_mu (mu) + keep (mu) + alt (n) + less_than_equal flag (1)
        return 2 * self.keep_bits + num_levels + 1

    @classmethod
    def from_error(cls, hamiltonian: PauliSum, hamiltonian_error: float) -> Self:
        """Construct an AliasSampling instance given a target error."""
        prob_l1_error = hamiltonian_error / hamiltonian.lam
        keep_bits = max(1, ceil(log2(1 / prob_l1_error)))
        return cls(keep_bits=keep_bits)


@dataclass(frozen=True)
class RotationTree:
    """
    State preparation using rotation tree with angles.

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

    def num_ancillas(self, num_levels: int) -> int:
        """Get the number of ancillas used by the state preparation."""
        return self.angle_bits

    @classmethod
    def from_error(
        cls, hamiltonian: PauliSum, hamiltonian_error: float, *, real: bool
    ) -> Self:
        """Construct a RotationTree instance given a target error."""
        prep_op_error = hamiltonian_error / (2 * hamiltonian.lam)
        num_rotations_per_level = 1 if real else 2
        num_levels = max(1, ceil(log2(hamiltonian.num_words_with_identity)))
        num_rotations = num_levels * num_rotations_per_level
        angle_bits = max(1, ceil(log2(2 * pi * num_rotations / prep_op_error)))
        return cls(angle_bits=angle_bits)


PrepAlgorithm = AliasSampling | RotationTree


@dataclass(frozen=True)
class Qubitisation:
    """
    Qubitised Hamiltonian propagator.

    Parameters
    ----------
    hamiltonian : PauliSum
        Hamiltonian to simulate.
    prep : PrepAlgorithm
        State preparation algorithm to use for block encoding.

    Attributes
    ----------
    phase_information : PhaseInformation
        Phase information the propagator must resolve. Qubitisation energies require
        only the magnitude of the phase.

    """

    hamiltonian: PauliSum
    prep: PrepAlgorithm

    phase_information: ClassVar[PhaseInformation] = PhaseInformation.MAGNITUDE_ONLY

    @property
    def num_index_ancillas(self) -> int:
        """Number of ancillas used to specify each term index in the sum."""
        return max(1, ceil(log2(self.hamiltonian.num_words_with_identity)))

    @property
    def num_ancillas(self) -> int:
        """Number of ancillas used by the state preparation."""
        return self.num_index_ancillas + self.prep.num_ancillas(self.num_index_ancillas)

    def phase_to_energy(self, phase: float) -> float:
        """Convert a phase from this propagator into an energy."""
        return cos(phase * 2 * pi) * self.hamiltonian.lam


SimulationAlgorithm = Trotter | QDRIFT | Qubitisation
