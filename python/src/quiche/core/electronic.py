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
    PositiveInt,
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

    num_electrons: int
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
    num_electrons: int
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
        if self.num_electrons < 0:
            error_msg = "The number of electrons must be non-negative."
            raise ValueError(error_msg)

        if self.num_electrons > self.num_spin_orbitals:
            error_msg = (
                "The number of electrons must not exceed the number of spin orbitals."
            )
            raise ValueError(error_msg)

        return self

    @computed_field
    @cached_property
    def num_spatial_orbitals(self) -> int:
        """Number of spatial molecular orbitals."""
        return self.one_body.shape[0]

    @computed_field
    @cached_property
    def num_spin_orbitals(self) -> int:
        """Number of spin orbitals."""
        return 2 * self.num_spatial_orbitals

    def __eq__(self, other: object) -> bool:
        """Compare two SecondQuantisedHamiltonians."""
        if not isinstance(other, SecondQuantisedHamiltonian):
            return NotImplemented

        return (
            self.num_electrons == other.num_electrons
            and self.core_energy == other.core_energy
            and np.array_equal(self.one_body, other.one_body)
            and np.array_equal(self.two_body, other.two_body)
        )

    def __hash__(self) -> int:
        """Hash a SecondQuantisedHamiltonian."""
        return hash(
            (
                self.num_electrons,
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


class DFTHCHamiltonian(BaseModel):
    r"""
    Class encompassing an electronic Hamiltonian in DFTHC factorised form.

    Parameters
    ----------
    num_orbitals : int
        Number of spatial orbitals, :math:`N`. Must be positive.
    num_ranks : int
        Number of ranks in the factorisation, :math:`R`. Must be positive.
    num_bases : int
        Number of basis vectors per rank, :math:`B`. Must be positive.
    num_copies : int
        Number of copies (weight channels) per rank, :math:`C`. Must be positive.
    unit_vectors : NDArray[np.float64]
        Real array of shape ``(R, B, N)``. Each vector along the last axis is
        normalised on reconstruction, so must be non-zero.
    weight_vectors : NDArray[np.float64]
        Real array of shape ``(R, B, C)`` containing the weights that
        combine the unit vectors into the factors :math:`W_{rc}`.
    identity_weights : NDArray[np.float64]
        Real array of shape ``(R, C)`` containing the coefficient of the identity
        inside each squared factor, :math:`\alpha_{rc}` (:math:`W^{(rc)}` in eq. (23)
        of [Low2025]_), one scalar per rank and copy. Only affects the
        sum-of-squares representation.
    bliss_matrix : NDArray[np.float64]
        Real array of shape ``(N, N)`` containing the BLISS symmetry shift
        matrix, which is removed from the factorised two body integrals on
        reconstruction.
    bliss_constant : float
        Coefficient of the identity in the one body BLISS shift, :math:`\beta_1` in
        eq. (40) of [Low2025]_. Only affects the sum-of-squares representation.
    factorised_one_body : NDArray[np.float64]
        Real array of shape ``(N, N)`` containing the one body integrals in the
        factorised ("E-form") convention, which absorb a contribution from
        the two body integrals. The bare one body integrals are recovered on
        reconstruction.
    const : float
        Constant energy offset, used as the core energy of the reconstructed
        Hamiltonian.
    num_electrons : int
        Number of electrons in the system. Must be positive (to allow sum-of-squares
        construction) and at most :math:`2N`.

    References
    ----------
    .. [Low2025] G. H. Low et al., "Fast Quantum Simulation of Electronic Structure
       by Spectral Amplification", Phys. Rev. X, vol. 15, no. 4, p. 041016,
       Oct. 2025.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    num_orbitals: int
    num_ranks: int
    num_bases: int
    num_copies: int
    unit_vectors: NDArray[np.float64]
    weight_vectors: NDArray[np.float64]
    identity_weights: NDArray[np.float64]
    bliss_matrix: NDArray[np.float64]
    bliss_constant: float
    factorised_one_body: NDArray[np.float64]
    const: float
    num_electrons: PositiveInt

    @field_validator(
        "unit_vectors",
        "weight_vectors",
        "identity_weights",
        "bliss_matrix",
        "factorised_one_body",
        mode="before",
    )
    @classmethod
    def as_immutable_array(cls, value: object) -> NDArray[np.float64]:
        """Coerce the factors to an immutable real float array."""
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.complexfloating):
            error_msg = "The factors must be real, got complex values."
            raise ValueError(error_msg)
        array = np.array(array, dtype=float)
        if not np.all(np.isfinite(array)):
            error_msg = "The factors must be finite, got NaN or infinite values."
            raise ValueError(error_msg)
        array.flags.writeable = False
        return array

    @model_validator(mode="after")
    def check_dimension_positive(self) -> Self:
        """Validate that the dimensions N, R, B and C are strictly positive."""
        dimensions = {
            "num_orbitals": self.num_orbitals,
            "num_ranks": self.num_ranks,
            "num_bases": self.num_bases,
            "num_copies": self.num_copies,
        }
        if any(value <= 0 for value in dimensions.values()):
            error_msg = f"All dimensions must be positive, got {dimensions}."
            raise ValueError(error_msg)
        return self

    @model_validator(mode="after")
    def check_shapes(self) -> Self:
        """Validate the factor shapes are consistent with N, R, B and C."""
        expected = {
            "unit_vectors": (self.num_ranks, self.num_bases, self.num_orbitals),
            "weight_vectors": (self.num_ranks, self.num_bases, self.num_copies),
            "identity_weights": (self.num_ranks, self.num_copies),
            "bliss_matrix": (self.num_orbitals, self.num_orbitals),
            "factorised_one_body": (self.num_orbitals, self.num_orbitals),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                error_msg = (
                    f"{name} has shape {getattr(self, name).shape}, expected {shape}."
                )
                raise ValueError(error_msg)

        # Reconstruct divides by U, so a zero vector would result in NaNs.
        if np.any(np.linalg.norm(self.unit_vectors, axis=2) == 0.0):
            error_msg = (
                "`unit_vectors` contains a zero vector, which cannot be normalised."
            )
            raise ValueError(error_msg)

        return self

    @model_validator(mode="after")
    def check_electrons(self) -> Self:
        """Validate the electron count fits in the given orbital space."""
        if self.num_electrons > 2 * self.num_orbitals:
            error_msg = (
                "The number of electrons must not exceed the number of spin orbitals."
            )
            raise ValueError(error_msg)

        return self

    def reconstruct(self) -> SecondQuantisedHamiltonian:
        r"""
        Rebuild the one and two body integrals from the factors.

        The approximated two body tensor is built from the factor matrices, and the
        BLISS shift is removed from it:

        .. math::

            h^{(2)}_{pqts} = \sum_{rc} W_{rc}[p, q] \, W_{rc}[t, s]
                - \frac{1}{2} \left( B_{pq} \delta_{ts} + \delta_{pq} B_{ts} \right).

        The one body integrals are converted from the factorised ("E-form") convention
        to the bare convention, :math:`h_{pq} = h^{(1)}_{pq} + \frac{1}{2} \sum_r
        h^{(2)}_{prrq}`, and `const` is used as the core energy.

        Returns
        -------
        SecondQuantisedHamiltonian
            The reconstructed Hamiltonian, with the same number of electrons.
        """
        w_rc = self._factor_matrices()

        # g_approx[p,q,t,s] = sum_{r,c} W_rc[p,q] * W_rc[t,s]
        g_approx = np.einsum("rcpq,rcts->pqts", w_rc, w_rc)

        # Remove the BLISS symmetry shift: g = g_approx - 1/2 (B x I + I x B)
        identity = np.eye(self.num_orbitals)
        shift = np.einsum("pq,rs->pqrs", self.bliss_matrix, identity) + np.einsum(
            "pq,rs->pqrs", identity, self.bliss_matrix
        )
        two_body = g_approx - 0.5 * shift

        # E-form -> bare one-body integrals = factorised_one_body + 1/2 sum_r (pr|rq)
        one_body = self.factorised_one_body + 0.5 * np.einsum("prrq->pq", two_body)

        return SecondQuantisedHamiltonian(
            one_body=one_body,
            two_body=two_body,
            num_electrons=self.num_electrons,
            core_energy=self.const,
        )

    def sos_one_body(self) -> NDArray[np.float64]:
        r"""
        Get the one body matrix of the sum-of-squares representation.

        This is :math:`h^{(1)'}` of eq. (26) in [Low2025]_, whose eigendecomposition
        gives the :math:`D_1/Q_1` terms of the block encoding:

        .. math::

            h^{(1)'} = h^{(1)} + \frac{c}{\eta} I - \frac{\eta}{2} B + \beta_1 I
                - \sum_{rc} \alpha_{rc} \sum_b w_b^{(rc)} u_b^{(r)} u_b^{(r)T},

        where :math:`h^{(1)}` is `factorised_one_body`, :math:`c` is `const`,
        :math:`\eta` is `num_electrons`, :math:`B` is `bliss_matrix`,
        :math:`\beta_1` is `bliss_constant` and :math:`\alpha_{rc}` are the
        `identity_weights`. The :math:`(c / \eta) I` term matches the factorisation
        code, which folds the constant into the one body term while optimising.

        Returns
        -------
        NDArray[np.float64]
            Real symmetric array of shape ``(N, N)``.
        """
        w_rc = self._factor_matrices()

        identity = np.eye(self.num_orbitals)
        return (
            self.factorised_one_body
            + (self.const / self.num_electrons) * identity
            - 0.5 * self.num_electrons * self.bliss_matrix
            + self.bliss_constant * identity
            - np.einsum("rc,rcpq->pq", self.identity_weights, w_rc)
        )

    def _factor_matrices(self) -> NDArray[np.float64]:
        r"""
        Combine the unit and weight vectors into the factor matrices :math:`W_{rc}`.

        Each factor is the weighted sum of outer products of the unit vectors of its
        rank, normalised along the orbital axis:

        .. math::

            W_{rc}[p, q] = \sum_{b} w_b^{(rc)} u_{b,p}^{(r)} u_{b,q}^{(r)},

        so that the approximated two body tensor is
        :math:`\sum_{rc} W_{rc}[p, q] \, W_{rc}[t, s]` (eq. (24) of [Low2025]_).

        Returns
        -------
        NDArray[np.float64]
            Real array of shape ``(R, C, N, N)``, where each ``(N, N)`` matrix is
            symmetric.
        """
        u_normalised = self.unit_vectors / np.linalg.norm(
            self.unit_vectors, axis=2, keepdims=True
        )
        # W_rc[p,q] = sum_b w[r,b,c] * u[r,b,p] * u[r,b,q]
        return np.einsum(
            "rbc,rbp,rbq->rcpq",
            self.weight_vectors,
            u_normalised,
            u_normalised,
        )
