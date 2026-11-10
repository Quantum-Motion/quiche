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

"""Quantum phase estimation algorithms."""

from abc import ABC, abstractmethod
from math import ceil, log, log2, pi, sqrt
from typing import Literal, Self

from pydantic import PositiveInt
from pydantic.dataclasses import dataclass

from quiche.algorithms.simulation import SimulationAlgorithm


def _num_bits_exact(phase_error: float) -> int:
    """Get the number of bits required for phases which are exactly representable."""
    return ceil(log2(1 / phase_error))


def _num_extra_bits_inexact(failure_prob: float) -> int:
    """
    Get the number of extra bits required for phases which aren't exactly representable.

    Additional bits are used to ensure the likelihood of an incorrectly rounded result
    is bound by a given probability.

    Parameters
    ----------
    failure_prob : float
        Maximum probability of the phase being resolved incorrectly.

    Returns
    -------
    int
        Required number of additional bits.

    References
    ----------
    .. [NielsenChuang2010] M. A. Nielsen and I. L. Chuang, Quantum computation and
       quantum information, 10th anniversary edition. Cambridge: Cambridge university
       press, 2010.

    """
    return ceil(log2(2 + 1 / (2 * failure_prob)))


def _num_bits_from_accuracy(phase_error: float, failure_prob: float) -> int:
    """
    Get the total number of bits required to resolve an arbitrary phase.

    Parameters
    ----------
    phase_error : float
        Precision to which the phase should be resolved to.
    failure_prob : float
        Maximum probability of the phase being resolved incorrectly.

    Returns
    -------
    int
        Number of bits required to resolve the phase.

    """
    return _num_bits_exact(phase_error) + _num_extra_bits_inexact(failure_prob)


def _hoeffding_bound(half_width: float, failure_prob: float) -> int:
    """
    Get the number of samples needed to estimate an expectation value.

    Applies Hoeffding's inequality to measurement outcomes in ``[-1, 1]`` (range
    sets the constant factor in the bound).

    Parameters
    ----------
    half_width : float
        Half-width of the confidence interval: the estimate lies within
        ``result±half_width`` of the true value. Must be positive.
    failure_prob : float
        Permitted probability that the estimate falls outside ``±half_width``. Must
        lie in ``(0, 1)``.

    Returns
    -------
    int
        Number of samples required, at least 1.

    """
    return max(1, ceil((2 / half_width**2) * log(2 / failure_prob)))


class EstimationAlgorithm(ABC):
    """Abstract class for phase estimation algorithm interface."""

    @property
    @abstractmethod
    def num_ancillas(self) -> int:
        """Number of ancillas used by the phase estimation."""
        ...

    @property
    @abstractmethod
    def powers(self) -> tuple[int, ...]:
        """Powers of the Hamiltonian simulation needed for phase estimation."""
        ...

    @classmethod
    @abstractmethod
    def from_accuracy(
        cls,
        *,
        simulation: type[SimulationAlgorithm] | SimulationAlgorithm,
        phase_error: float,
        failure_prob: float,
    ) -> Self:
        """
        Construct a phase estimation algorithm instance matching a target accuracy.

        Parameters
        ----------
        simulation : type[SimulationAlgorithm] | SimulationAlgorithm
            Hamiltonian simulation algorithm, or its class, used with this estimation.
        phase_error : float
            Target error when resolving the estimated phase, as a dimensionless fraction
            of a full period.
        failure_prob : float
            Maximum allowable failure probability, i.e. that the resolved phase is more
            than ``±phase_error`` away from the correct phase. Must lie in ``(0, 1)``.

        Returns
        -------
        Self
            Instance sized to meet the target accuracy.

        """
        ...


@dataclass(frozen=True)
class TextbookQPE(EstimationAlgorithm):
    """
    Textbook, multi-ancilla phase estimation with QFT.

    Parameters
    ----------
    num_rounds: int
        Number of estimation rounds, each contributing one estimation ancilla and one
        controlled propagator power. Must be positive.

    """

    num_bits: PositiveInt

    @property
    def num_ancillas(self) -> int:
        """Number of ancillas used by the phase estimation."""
        return self.num_bits

    @property
    def powers(self) -> tuple[int, ...]:
        """Powers of the Hamiltonian simulation needed for phase estimation."""
        return tuple(2**j for j in range(self.num_bits))

    @classmethod
    def from_accuracy(
        cls,
        *,
        simulation: type[SimulationAlgorithm] | SimulationAlgorithm,
        phase_error: float,
        failure_prob: float,
    ) -> Self:
        """Construct a QPE instance given a target error and failure probability."""
        return cls(num_bits=_num_bits_from_accuracy(phase_error, failure_prob))


@dataclass(frozen=True)
class IterativeQPE(EstimationAlgorithm):
    """
    Iterative single-ancilla phase estimation with semiclassical QFT.

    Parameters
    ----------
    num_rounds: int
        Number of estimation rounds, each a Hadamard test on the single estimation
        ancilla, with a feedback rotation angle based on the preceding rounds'
        measurement outcomes. Must be positive.

    """

    num_rounds: PositiveInt

    @property
    def num_ancillas(self) -> int:
        """Number of ancillas used by the phase estimation."""
        return 1

    @property
    def powers(self) -> tuple[int, ...]:
        """Powers of the Hamiltonian simulation needed for phase estimation."""
        return tuple(2**j for j in range(self.num_rounds))

    @classmethod
    def from_accuracy(
        cls,
        *,
        simulation: type[SimulationAlgorithm] | SimulationAlgorithm,
        phase_error: float,
        failure_prob: float,
    ) -> Self:
        """Construct a QPE instance given a target error and failure probability."""
        return cls(num_rounds=_num_bits_from_accuracy(phase_error, failure_prob))


@dataclass(frozen=True)
class KitaevQPE(EstimationAlgorithm):
    """
    Kitaev's single-ancilla phase estimation.

    Parameters
    ----------
    num_rounds: int
        Number of estimation rounds, each a Hadamard test at a round-dependent
        propagator power. Must be positive.
    num_shots : int
        Number of individual samples needed per measurement basis. Must be positive.
    num_bases : {1, 2}
        Number of bases that must be measured to determine the expectation value.

    References
    ----------
    .. [Kitaev1995] A. Y. Kitaev, 'Quantum measurements and the Abelian Stabilizer
       Problem', Nov. 20, 1995, arXiv: arXiv:quant-ph/9511026. doi:
        10.48550/arXiv.quant-ph/9511026.
    .. [Kitaev2002] A. Y. Kitaev, A. Shen, and M. N. Vyalyi, Classical and quantum
       computation. in Graduate studies in mathematics, no. v. 47. Providence, R.I:
       American Mathematical Society, 2002.

    """

    num_rounds: PositiveInt
    num_shots: PositiveInt
    num_bases: Literal[1, 2]

    @property
    def num_ancillas(self) -> int:
        """Number of ancillas used by the phase estimation."""
        return 1

    @property
    def powers(self) -> tuple[int, ...]:
        """Powers of the Hamiltonian simulation needed for phase estimation."""
        return tuple(2**j for j in range(self.num_rounds))

    @classmethod
    def from_accuracy(
        cls,
        *,
        simulation: type[SimulationAlgorithm] | SimulationAlgorithm,
        phase_error: float,
        failure_prob: float,
    ) -> Self:
        """Construct a QPE instance given a target error and failure probability."""
        num_bases = simulation.phase_information.num_measurement_bases
        num_rounds = cls._num_rounds(phase_error)
        num_shots = cls._num_shots_per_measurement(
            failure_prob,
            num_rounds,
            num_bases,
        )
        return cls(
            num_rounds=num_rounds,
            num_shots=num_shots,
            num_bases=num_bases,
        )

    @staticmethod
    def _num_rounds(phase_error: float) -> int:
        """Calculate number of Kitaev QPE rounds required for target phase error."""
        return max(1, ceil(log2(1 / phase_error)) - 2)

    @staticmethod
    def _num_shots_per_measurement(
        failure_prob: float,
        num_rounds: int,
        num_bases: int,
    ) -> int:
        """Calculate number of samples per basis for a total failure probability."""
        # by union bound
        failure_prob_per_measurement = failure_prob / (num_rounds * num_bases)
        angle_precision = pi / 8  # 1/16 error per angle
        return _hoeffding_bound(angle_precision, failure_prob_per_measurement)


@dataclass(frozen=True)
class NaiveQPE(EstimationAlgorithm):
    """
    Naive, single-ancilla phase estimation (Hadamard test).

    Parameters
    ----------
    num_shots : int
        Number of individual samples needed per measurement basis. Must be positive.
    num_bases : {1, 2}
        Number of bases that must be measured to determine the expectation value.

    """

    num_shots: PositiveInt
    num_bases: Literal[1, 2]

    @property
    def num_ancillas(self) -> int:
        """Number of ancillas used by the phase estimation."""
        return 1

    @property
    def powers(self) -> tuple[int, ...]:
        """Powers of the Hamiltonian simulation needed for phase estimation."""
        return (1,)

    @classmethod
    def from_accuracy(
        cls,
        *,
        simulation: type[SimulationAlgorithm] | SimulationAlgorithm,
        phase_error: float,
        failure_prob: float,
    ) -> Self:
        """Construct a QPE instance given a target error and failure probability."""
        num_bases = simulation.phase_information.num_measurement_bases
        failure_prob_per_basis = failure_prob / num_bases  # by union bound
        # so that error =|z - \tilde z | = sqrt(error_re**2 + error_im**2)
        error_per_basis = phase_error / sqrt(num_bases)
        num_shots = _hoeffding_bound(error_per_basis, failure_prob_per_basis)
        return cls(num_shots=num_shots, num_bases=num_bases)
