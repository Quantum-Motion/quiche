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

"""Random second quantised Hamiltonians for the example notebooks."""

import numpy as np

from quiche.core import SecondQuantisedHamiltonian


def generate_random_one_body_integrals(
    n_orb: int, seed: int | None = None
) -> np.ndarray:
    """
    Generate random one body integrals for real-valued spatial orbitals.

    Args:
        n_orb (int): The number of spatial orbitals
        seed (int, optional): Random seed. Defaults to None.

    Returns:
        np.ndarray: A valid n_orb X n_orb symmetric matrix.

    """
    rng = np.random.default_rng(seed)
    h_core = np.zeros([n_orb] * 2)

    for p in range(n_orb):
        for q in range(p, n_orb):
            # Diagonal elements (on-site energy) are usually negative (bound states).
            # Off-diagonal elements are hopping terms.
            val = rng.uniform(-3.0, -1.0) if p == q else rng.uniform(-0.5, 0.5)

            h_core[p, q] = val
            h_core[q, p] = val  # Symmetric

    return h_core


def generate_random_two_body_integrals(
    n_orb: int, seed: int | None = None
) -> np.ndarray:
    """
    Generate random two body integrals for real-valued spatial orbitals.

    Args:
        n_orb (int): The number of spatial orbitals
        seed (int, optional): Random seed. Defaults to None.

    Returns:
        np.ndarray: A valid n_orb X n_orb X n_orb X n_orb tensor.

    """
    rng = np.random.default_rng(seed)
    eri = np.zeros([n_orb] * 4)

    for p in range(n_orb):
        for q in range(n_orb):
            for r in range(n_orb):
                for s in range(n_orb):
                    if eri[p, q, r, s] == 0:
                        if p == q and q == r and r == s:
                            # onsite Coulomb repulsion > 0.
                            val = rng.uniform(0.5, 2.0)
                        elif p == q and r == s:
                            # Inter-site Coulomb repulsion > 0.
                            val = rng.uniform(0.2, 1.5)
                        else:
                            # off-diagonal transition elements.
                            val = rng.uniform(-0.5, 0.5)

                        # eight-fold symmetry (real-valued spatial orbitals)
                        eri[p, q, r, s] = val
                        eri[q, p, r, s] = val
                        eri[p, q, s, r] = val
                        eri[q, p, s, r] = val
                        eri[r, s, p, q] = val
                        eri[s, r, p, q] = val
                        eri[r, s, q, p] = val
                        eri[s, r, q, p] = val

    return eri


def generate_random_sq_hamiltonian(
    n_orb: int, n_elec: int, seed: int | None = None
) -> SecondQuantisedHamiltonian:
    """
    Generate a random second quantised Hamiltonian for real-valued spatial orbitals.

    Args:
        n_orb (int): The number of spatial orbitals
        n_elec (int): The number of electrons
        seed (int, optional): Random seed. Defaults to None.

    Returns:
        SecondQuantisedHamiltonian: Hamiltonian with random integrals and core energy.

    """
    # Define the chemical system.
    rng = np.random.default_rng(seed)
    one_body = generate_random_one_body_integrals(n_orb, seed)
    two_body = generate_random_two_body_integrals(n_orb, seed)
    core_energy = rng.random()

    # Define the second quantisation electronic Hamiltonian.
    return SecondQuantisedHamiltonian(
        one_body=one_body,
        two_body=two_body,
        num_electrons=n_elec,
        core_energy=core_energy,
    )
