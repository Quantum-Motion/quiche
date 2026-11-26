# %% tags=["remove-cell"]
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

# %% [markdown]
# # 3. Fermion-to-qubit mappings
#
# To perform chemical calculations on a quantum computer, one needs to map the fermions
# to qubits. In QUICHE, one can use the `quiche.chemistry` module. In this tutorial, we
# cover three fermion-to-qubit mappings: the Jordan-Wigner, Bravyi-Kitaev and Parity
# mappings.
#
# Furthermore, we explore how to use the `.to_qubit_hamiltonian` method of
# `SecondQuantisedHamiltonian` to apply a fermion-to-qubit mapping and construct the
# qubit Hamiltonian from the second-quantised Hamiltonian.

# %% [markdown]
# Below are the short descriptions of the fermion-to-qubit mappings:
#
# - **Jordan-Wigner** (`Mapping.JordanWigner`): Each qubit stores whether one spin
# orbital is occupied, so the qubit basis state matches the fermionic occupation vector
# directly. The fermionic sign is tracked by a string of Pauli-$Z$ operators on all
# lower-indexed qubits, so operators act on up to $\mathcal{O}(N)$ qubits.
#
# - **Parity** (`Mapping.Parity`): Each qubit stores the parity (sum mod 2) of the
# occupations of all orbitals up to and including its own index, so the sign comes from
# a single qubit. Occupation information is now spread across qubits, though, so
# creation and annihilation operators still act on $\mathcal{O}(N)$ qubits, through a
# string of Pauli-$X$ operators.
#
# - **Bravyi-Kitaev** (`Mapping.BravyiKitaev`): A hybrid of the two, where each qubit
# stores the parity of a subset of orbitals chosen by a binary-tree (Fenwick tree)
# structure. Both occupation and parity information are stored non-locally, so each
# fermionic operator acts on only $\mathcal{O}(\log N)$ qubits.

# %% [markdown]
# ## 3.1. `quiche.chemistry`: Fermion-to-qubit mappings
#
# The `quiche.chemistry` module contains various functions to obtain the occupation
# basis state and map into other bases.

# %%
from quiche.chemistry import (
    HartreeFockState,
    get_bk_state,
    get_hf_state,
    get_jw_state,
    get_parity_state,
)

# %% [markdown]
# First, define the parameters of the chemical system.

# %%
# Define the parameters of the chemical system.
n_orb = 2  # two spatial orbitals
num_electrons = 2
num_spin_orbitals = n_orb * 2

# %%
# Fermion-to-qubit mapping.
# The HF state is the occupation basis state.
# `get_hf_state` gives you the HF ground state, in which the lowest energy
# spin orbitals are occupied.
hf_state = get_hf_state(num_spin_orbitals, num_electrons)
bk_state = get_bk_state(hf_state)
jw_state = get_jw_state(hf_state)
parity_state = get_parity_state(hf_state)
print("HF state:", hf_state)
print("BK state:", bk_state)
print("JW state:", jw_state)
print("Parity state:", parity_state)

# %% [markdown]
# One can define the Hartree-Fock state from the `HartreeFockState` class.

# %%
# Define the HF state from the HartreeFockState class.
hf_state = HartreeFockState.closed_shell(
    num_electrons=num_electrons,
    num_spin_orbitals=num_spin_orbitals,
)
print("HF state:", hf_state)

# %% [markdown]
# Furthermore, one can define the occupation basis state directly from the occupation
# vector.

# %%
hf_state = HartreeFockState(occupation=(0, 1, 1, 0))
print("HF state:", hf_state)

# %% [markdown]
# ## 3.2. Mapping the second-quantised Hamiltonian into a qubit Hamiltonian
#
# One can map the second-quantised Hamiltonian into a qubit Hamiltonian with the
# built-in `.to_qubit_hamiltonian` method. One can specify the fermion-to-qubit mapping
# as an argument.
#
# For the purpose of demonstration, we implemented methods to generate random
# second-quantised Hamiltonians for real-valued spatial orbitals in the
# `random_hamiltonian` module

# %%
import numpy as np
from random_hamiltonian import (
    generate_random_one_body_integrals,
    generate_random_two_body_integrals,
)

from quiche.core import Mapping, SecondQuantisedHamiltonian

# %%
# Define the chemical system.
rng = np.random.default_rng(seed=1234)
n_orb = 1
n_elec = 2

# Define the second-quantised electronic Hamiltonian.
sq_ham = SecondQuantisedHamiltonian(
    one_body=generate_random_one_body_integrals(n_orb, rng),
    two_body=generate_random_two_body_integrals(n_orb, rng),
    core_energy=rng.random(),
    num_electrons=n_elec,
)

# %%
# Convert the second-quantised Hamiltonian to a qubit Hamiltonian (PauliSum).
ham_bk = sq_ham.to_qubit_hamiltonian(mapping=Mapping.BravyiKitaev)
ham_jw = sq_ham.to_qubit_hamiltonian(mapping=Mapping.JordanWigner)
ham_p = sq_ham.to_qubit_hamiltonian(mapping=Mapping.Parity)

print("Bravyi-Kitaev:")
print(ham_bk.paulis)
print("Jordan-Wigner:")
print(ham_jw.paulis)
print("Parity:")
print(ham_p.paulis)

# %% [markdown]
# You can see that different fermion-to-qubit mappings result in different qubit
# Hamiltonians for the same second-quantised Hamiltonian.
