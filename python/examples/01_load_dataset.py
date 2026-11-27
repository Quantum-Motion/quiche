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
# # 1. Loading the molecular data files into the Hamiltonian
#
# The data to construct the Hamiltonian for the chemical system of interest are stored
# in two file formats:
#
# - `.fcidump`: This is a simple, widely used ASCII text format designed specifically to
# store quantum chemistry Hamiltonians and molecular integrals (one- and two-electron
# integrals).
#
# - `.hdf5`: This is a high-performance binary data format used for the efficient,
# portable storage of massive scientific datasets.
#
# In this tutorial, we will cover how to construct the relevant Hamiltonians from the
# given data stored in the above two file formats.

# %% [markdown]
# ## 1.1. Loading `.fcidump` into the second-quantised Hamiltonian
#
# `.fcidump` files contain the constant energy offset, the one-body and two-body
# molecular integrals and other quantities needed to construct the second-quantised
# Hamiltonian.
#
# Below is an example FCIDUMP for the $\mathrm{H}_2$ molecule, generated using
# [ORCA](https://www.faccts.de/orca/) with the STO-3G basis set. The data is small, so
# we store it directly as a string below.
#
# Below, we load the data into `quiche.core.SecondQuantisedHamiltonian` by using the
# `quiche.io` module. `fcidump.parse` reads an FCIDUMP string, and `fcidump.read` reads
# an FCIDUMP file.

# %%
from quiche.io import fcidump

# %%
FCIDUMP = """
&FCI NORB= 2,NELEC= 2,MS2= 0,
 ORBSYM=1,1,
 ISYM= 1,
&END
    0.714139282456140   0 0 0 0
   -1.252705257753715   1 1 0 0
   -0.475697705661824   2 2 0 0
    0.674565096014104   1 1 1 1
    0.181266416982487   2 1 2 1
    0.663537594129644   2 2 1 1
    0.697467384335182   2 2 2 2
"""

sq_ham = fcidump.parse(FCIDUMP)

print("Number of electrons:", sq_ham.num_electrons)
print("Core energy:", sq_ham.core_energy)
print("One-body integrals:\n", sq_ham.one_body)
print("Two-body integrals:\n", sq_ham.two_body)

# %% [markdown]
# The FCIDUMP data is loaded into `SecondQuantisedHamiltonian`, and we printed the
# number of electrons, core energy, one-body and two-body integrals successfully.

# %% [markdown]
# ## 1.2. Loading `.hdf5` into the qubit Hamiltonian
#
# The `.hdf5` file contains the qubit Hamiltonians in the byte string format for
# different fermion-to-qubit mappings with different numbers of qubits (corresponding to
# different numbers of spin orbitals).
#
# We use the $\mathrm{H}_2$ dataset from
# [HamLib](https://portal.nersc.gov/cfs/m888/dcamps/hamlib/) [[Sawaya et al.
# (2024)](https://arxiv.org/abs/2306.13126)], which we download below. Then we load the
# qubit Hamiltonian data by using the `quiche.io` module.

# %%
import urllib.request
import zipfile
from pathlib import Path

URL = "https://portal.nersc.gov/cfs/m888/dcamps/hamlib/chemistry/electronic/standard/H2.zip"
if not Path("H2.hdf5").exists():
    urllib.request.urlretrieve(URL, "H2.zip")
    with zipfile.ZipFile("H2.zip") as zf:
        zf.extract("H2.hdf5")

# %%
from quiche.io import hamlib

# %%
# Jordan-Wigner mapping with 4 spin orbitals (1s for each hydrogen; 2 spatial orbitals).
raw_data = hamlib.read_dataset("H2.hdf5", "ham_JW-4")
paulis = hamlib.parse(raw_data)

print("Number of qubits:", paulis.num_qubits)
print("Number of terms in the qubit Hamiltonian:", paulis.num_words)
print("1-norm of the qubit Hamiltonian:", paulis.lam)

# %% [markdown]
# The `.hdf5` file is loaded, and we printed the number of qubits, the number of Pauli
# terms and the 1-norm of the qubit Hamiltonian successfully.
#
# Note that `hamlib.parse` returns a `quiche.core.PauliSum` object, which is generally
# used to define the sum of Pauli operators (see
# [02_pauli_operators](./02_pauli_operators)).
#
# Later, we show that a `PauliSum` object can be used to define
# `quiche.core.ElectronicHamiltonian` object with the relevant [fermion-to-qubit
# mappings](./03_fermion_to_qubit).
