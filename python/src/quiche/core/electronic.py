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

"""Structures for specifying electronic systems."""

from functools import cached_property
from typing import Self

import numpy as np
from numpy.typing import NDArray
from pydantic import (
    BaseModel,
    ConfigDict,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic.dataclasses import dataclass

from .algorithms import Mapping
from .paulis import PauliSum


@dataclass(frozen=True)
class ElectronicHamiltonian:
    """Class encompassing a physical electronic Hamiltonian."""

    electrons: int
    mapping: Mapping
    paulis: PauliSum


class SecondQuantisedHamiltonian(BaseModel):
    """
    Class encompassing a second quantisation electronic Hamiltonian.

    One body integrals are indexed `h[p, q]` and two body integrals are in chemist
    notation, `h[p, q, r, s] = (pq|rs)`, over spatial orbitals.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    one_body: NDArray[np.float64]
    two_body: NDArray[np.float64]
    electrons: int
    core_energy: float = 0.0

    @field_validator("one_body", "two_body", mode="before")
    @classmethod
    def as_immutable_array(cls, value: object) -> NDArray[np.float64]:
        """Coerce the integrals to an immutable real float array."""
        array = np.asarray(value)

        if np.issubdtype(array.dtype, np.complexfloating):
            error_msg = "The integrals must be real, got complex values."
            raise ValueError(error_msg)

        array = np.array(array, dtype=float)
        array.flags.writeable = False
        return array

    @model_validator(mode="after")
    def check_dims(self) -> Self:
        """Validate the integral tensors have correct number of dimensions."""
        if self.one_body.ndim != 2:
            error_msg = "The one body integrals must be a 2-dimensional tensor."
            raise ValueError(error_msg)

        if self.two_body.ndim != 4:
            error_msg = "The two body integrals must be a 4-dimensional tensor."
            raise ValueError(error_msg)

        return self

    @model_validator(mode="after")
    def check_shapes(self) -> Self:
        """Validate the integral tensors are square and mutually consistent."""
        norb = self.one_body.shape[0]

        if norb == 0:
            error_msg = "The number of spatial orbitals must be nonzero."
            raise ValueError(error_msg)

        if self.one_body.shape != (norb, norb):
            error_msg = "The one body integrals must be square."
            raise ValueError(error_msg)

        if self.two_body.shape != (norb,) * 4:
            error_msg = (
                "The two body integrals must have the same orbital dimension as "
                f"the one body integrals, expected {(norb,) * 4} but got "
                f"{self.two_body.shape}."
            )
            raise ValueError(error_msg)

        return self

    @model_validator(mode="after")
    def check_electrons(self) -> Self:
        """Validate the electron count fits in the given orbital space."""
        if self.electrons < 0:
            error_msg = "The number of electrons must be non-negative."
            raise ValueError(error_msg)

        if self.electrons > self.num_spin_orbitals:
            error_msg = (
                "The number of electrons must not exceed the number of spin orbitals."
            )
            raise ValueError(error_msg)

        return self

    @computed_field
    @cached_property
    def num_spatial_orbitals(self) -> int:
        """Get the number of spatial molecular orbitals."""
        return self.one_body.shape[0]

    @computed_field
    @cached_property
    def num_spin_orbitals(self) -> int:
        """Get the number of spin orbitals."""
        return 2 * self.num_spatial_orbitals

    def __eq__(self, other: object) -> bool:
        """Compare two SecondQuantisedHamiltonians."""
        if not isinstance(other, SecondQuantisedHamiltonian):
            return NotImplemented

        return (
            self.electrons == other.electrons
            and self.core_energy == other.core_energy
            and np.array_equal(self.one_body, other.one_body)
            and np.array_equal(self.two_body, other.two_body)
        )

    def __hash__(self) -> int:
        """Hash a SecondQuantisedHamiltonian."""
        return hash(
            (
                self.electrons,
                self.core_energy,
                self.one_body.shape,
                self.one_body.tobytes(),
                self.two_body.shape,
                self.two_body.tobytes(),
            )
        )

    def to_qubit_hamiltonian(self, mapping: Mapping) -> ElectronicHamiltonian:
        """Get the qubit Hamiltonian after applying fermion-to-qubit mapping."""
        # Lazily import openfermion wrappers, only used here
        from quiche.io._openfermion import (  # noqa: PLC0415
            _second_quantised_to_electronic_hamiltonian,
        )

        return _second_quantised_to_electronic_hamiltonian(self, mapping)


class FactorisedHamiltonian(BaseModel):
    """
    Class encompassing a Factorized Hamiltonian, generated from DFTHC output.

    Contains number of orbitals, ranks, bases and copies,
    as well as the factors themselves.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    N: int  # number of spatial orbitals
    R: int  # outer rank
    B: int  # inner rank
    C: int  # copies
    U: NDArray[np.float64]  # (R, B, N) unit vectors
    W: NDArray[np.float64]  # (R, B, C) weights
    bliss_matrix: NDArray[np.float64]  # (N, N) symmetric BLISS matrix
    h1: NDArray[np.float64]  # (N, N) one-body matrix, without the const/N shift
    const: float  # constant energy
    electrons: int  # number of electrons
    job_id: int | None = None  # job ID in the original db file

    @field_validator("U", "W", "bliss_matrix", "h1", mode="before")
    @classmethod
    def as_immutable_array(cls, value: object) -> NDArray[np.float64]:
        """Coerce the factors to an immutable real float array."""
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.complexfloating):
            error_msg = "The factors must be real, got complex values."
            raise ValueError(error_msg)
        array = np.array(array, dtype=float)
        array.flags.writeable = False
        return array

    @model_validator(mode="after")
    def check_shapes(self) -> Self:
        """Validate the factor shapes are consistent with N, R, B and C."""
        expected = {
            "U": (self.R, self.B, self.N),
            "W": (self.R, self.B, self.C),
            "bliss_matrix": (self.N, self.N),
            "h1": (self.N, self.N),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                error_msg = (
                    f"{name} has shape {getattr(self, name).shape}, expected {shape}."
                )
                raise ValueError(error_msg)

        # Reconstruct divides by U, so a zero column would result in NaNs.
        if np.any(np.linalg.norm(self.U, axis=2) == 0.0):
            error_msg = "U contains a zero vector, which cannot be normalised."
            raise ValueError(error_msg)

        return self

    def reconstruct(self) -> SecondQuantisedHamiltonian:
        """Rebuild the integrals from the factors."""
        u_normalized = self.U / np.linalg.norm(self.U, axis=2, keepdims=True)

        # W_rc[p,q] = sum_b W[r,b,c] * u[r,b,p] * u[r,b,q]
        w_rc = np.einsum("rbp,rbq,rbc->rcpq", u_normalized, u_normalized, self.W)

        # g_approx[p,q,t,s] = sum_{r,c} W_rc[p,q] * W_rc[t,s]
        g_approx = np.einsum("rcpq,rcts->pqts", w_rc, w_rc)

        # Remove the BLISS symmetry shift: g = g_approx - 1/2 (B x I + I x B)
        identity = np.eye(self.N)
        shift = np.einsum("pq,rs->pqrs", self.bliss_matrix, identity) + np.einsum(
            "pq,rs->pqrs", identity, self.bliss_matrix
        )
        two_body = g_approx - 0.5 * shift

        # E-form -> bare one-body integrals: t1 = h1 + 1/2 sum_r (pr|rq)
        one_body = self.h1 + 0.5 * np.einsum("prrq->pq", two_body)

        return SecondQuantisedHamiltonian(
            one_body=one_body,
            two_body=two_body,
            electrons=self.electrons,
            core_energy=self.const,
        )
