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
from typing import Literal

from pydantic import PositiveInt
from pydantic.dataclasses import dataclass


type EstimationAlgorithm = TextbookQPE | IterativeQPE | KitaevQPE | NaiveQPE


class _EstimationBase(ABC):
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


@dataclass(frozen=True)
class TextbookQPE(_EstimationBase):
    """
    Textbook, multi-ancilla phase estimation with QFT.

    Parameters
    ----------
    num_bits : int
        Number of estimation bits, each contributing one estimation ancilla and one
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


@dataclass(frozen=True)
class IterativeQPE(_EstimationBase):
    """
    Iterative single-ancilla phase estimation with semiclassical QFT.

    Parameters
    ----------
    num_rounds : int
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


@dataclass(frozen=True)
class KitaevQPE(_EstimationBase):
    """
    Kitaev's single-ancilla phase estimation.

    Parameters
    ----------
    num_rounds : int
        Number of estimation rounds, each a Hadamard test at a round-dependent
        propagator power. Must be positive.
    num_shots : int
        Number of individual samples needed per measurement basis. Must be positive.
    num_bases : {1, 2}
        Number of bases that must be measured to determine the expectation value.

    References
    ----------
    .. [Kitaev1995] A. Y. Kitaev, 'Quantum measurements and the Abelian Stabilizer
       Problem', Nov. 20, 1995, arXiv:quant-ph/9511026. doi:
       10.48550/arXiv.quant-ph/9511026.
    .. [Kitaev2002] A. Y. Kitaev, A. Shen, and M. N. Vyalyi, Classical and Quantum
       Computation. Graduate Studies in Mathematics, vol. 47. Providence, R.I.:
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


@dataclass(frozen=True)
class NaiveQPE(_EstimationBase):
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
