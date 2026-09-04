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

"""Quantum chemistry related methods and transformations."""

from dataclasses import dataclass
from math import ceil, log2
from typing import Self

import numpy as np
from numpy.typing import NDArray


def get_hf_state(num_spin_orbitals: int, num_electrons: int) -> NDArray[np.int_]:
    """
    Get the occupation basis state for a given number of electrons and spin orbitals.

    Parameters
    ----------
    num_spin_orbitals : int
        Number of spin orbitals (and hence length) of the final state.
    num_electrons : int
        Number of electrons, and thus number of orbitals occupied in the final state.

    Returns
    -------
    ``NDArray[np.int_]``
        The HF state with the lowest ``num_electron`` number of orbitals occupied.

    """
    if num_electrons < 0:
        err_msg = "Number of electrons must be non-negative."
        raise ValueError(err_msg)

    if num_spin_orbitals < 1:
        err_msg = "Number of spin orbitals must be positive."
        raise ValueError(err_msg)

    if num_electrons > num_spin_orbitals:
        err_msg = "Number of electrons must not exceed number of spin orbitals."
        raise ValueError(err_msg)

    occupation = np.zeros(num_spin_orbitals, dtype=int)
    occupation[:num_electrons] = 1

    return occupation


def get_jw_state(occupation: NDArray[np.int_]) -> NDArray[np.int_]:
    """
    Transform an occupation state to qubit basis with the Jordan-Wigner mapping.

    Parameters
    ----------
    occupation : ``NDArray[np.int_]``
        Occupation basis state to convert.

    Returns
    -------
    ``NDArray[np.int_]``
        The qubit basis state resulting from the fermion-to-qubit mapping.

    """
    return occupation


def get_parity_state(occupation: NDArray[np.int_]) -> NDArray[np.int_]:
    """
    Transform an occupation state to qubit basis with the Parity mapping.

    Parameters
    ----------
    occupation : ``NDArray[np.int_]``
        Occupation basis state to convert.

    Returns
    -------
    ``NDArray[np.int_]``
        The qubit basis state resulting from the fermion-to-qubit mapping.

    """
    return np.cumsum(occupation) % 2


def _bk_transformation_matrix(size: int) -> NDArray[np.int_]:
    """
    Construct the matrix for occupation basis to qubit basis Bravyi-Kitaev mapping.

    Parameters
    ----------
    size : int
        Size (number of orbitals/ qubits) for the transformation matrix.

    Returns
    -------
    NDArray
        Matrix transforming from occupation basis to qubit basis.

    """
    mat = np.array([1], dtype=int)

    id2 = np.eye(2, dtype=int)
    for i in range(ceil(log2(size))):
        mat = np.kron(id2, mat)
        mat[-1, : 2**i] = 1

    return mat[:size, :size]


def get_bk_state(occupation: NDArray[np.int_]) -> NDArray[np.int_]:
    """
    Transform an occupation state to qubit basis with the Bravyi-Kitaev mapping.

    Parameters
    ----------
    occupation : ``NDArray[np.int_]``
        Occupation basis state to convert.

    Returns
    -------
    ``NDArray[np.int_]``
        The qubit basis state resulting from the fermion-to-qubit mapping.

    """
    mat = _bk_transformation_matrix(len(occupation))
    return (mat @ occupation) % 2


@dataclass(frozen=True)
class HartreeFockState:
    """Dataclass representing a single Hartree-Fock state."""

    occupation: tuple[int, ...]

    @classmethod
    def closed_shell(cls, electrons: int, spin_orbitals: int) -> Self:
        """Initialise the Hartree-Fock state for a closed-shell system."""
        if electrons % 2 != 0:
            err_msg = "Closed shell system must have even number of electrons."
            raise ValueError(err_msg)
        return cls(occupation=tuple(get_hf_state(spin_orbitals, electrons)))

    def __post_init__(self) -> None:
        """Input validation for constructors."""
        if not all(i in {0, 1} for i in self.occupation):
            err_msg = "Spin orbital occupation must contain binary entries."
            raise ValueError(err_msg)

    @property
    def num_electrons(self) -> int:
        """Get number of electrons of Hartree-Fock state."""
        return sum(self.occupation)

    @property
    def num_spin_orbitals(self) -> int:
        """Get number of spin orbitals of Hartree-Fock state."""
        return len(self.occupation)
