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
# # Quickstart
#
# This page walks through a complete QUICHE calculation from loading a Hamiltonian, to
# setting up the phase estimation calculation, and finally dispatching to both backends.

# %% [markdown]
# This example uses the `H2.hdf5` dataset from [Hamlib](https://arxiv.org/abs/2306.13126),
# which is downloaded below if it isn't already present.

# %%
import urllib.request
import zipfile
from pathlib import Path

URL = "https://portal.nersc.gov/cfs/m888/dcamps/hamlib/chemistry/electronic/standard/H2.zip"
if not Path("H2.hdf5").exists():
    urllib.request.urlretrieve(URL, "H2.zip")
    with zipfile.ZipFile("H2.zip") as file:
        file.extract("H2.hdf5")

# %% [markdown]
# ## 1. Load a Hamiltonian
#
# QUICHE can open Hamiltonians from the [Hamlib](https://portal.nersc.gov/cfs/m888/dcamps/hamlib/)
# library and parse them into a {py:class}`~quiche.core.paulis.PauliSum`.


# %%
from quiche.io import hamlib

raw_data = hamlib.read_dataset("H2.hdf5", "ham_JW-4")
paulis = hamlib.parse(raw_data)

print(paulis.num_qubits, paulis.num_words, paulis.lam)

# %% [markdown]
# Pairing the sum with the electron count and the relevant fermion-to-qubit mapping one
# can define an {py:class}`~quiche.core.electronic.ElectronicHamiltonian`.

# %%
from quiche.core import ElectronicHamiltonian, Mapping

ham = ElectronicHamiltonian(
    num_electrons=2,
    paulis=paulis,
    mapping=Mapping.JordanWigner,
)

# %% [markdown]
# ## 2. Set an error budget
#
# Every approximate method in the calculation draws on an {py:class}`~quiche.Errors`
# budget. Using it, QUICHE derives all relevant parameters, such as ancilla counts,
# number of Trotter steps, QDRIFT repetitions, and so on.

# %%
from quiche.core import Errors

error = Errors(
    estimation=1e-2,
    simulation=1e-3,
    rotations=1e-3,
    state_prep=1e-3,
    overlap=0.9,
)

# %% [markdown]
# ## 3. Specify the calculation
#
# An initial state for the phase estimation calculation, such as a Hartree-Fock state,
# can be set:
# %%
from quiche.chemistry import HartreeFockState

hf = HartreeFockState.closed_shell(num_electrons=2, num_spin_orbitals=4)

# %% [markdown]
# and the specific algorithms for Hamiltonian simulation and phase estimation can be
# picked:
# %%
from quiche.core import PhaseEstimation, Simulation

phase_estimation = PhaseEstimation.Textbook
ham_simulation = Simulation.Qubitised

# %% [markdown]
# {py:class}`~quiche.dispatch.qpespec.QPESpec` brings everything together defining the
# complete phase estimation calculation.
# %%
from quiche.dispatch import QPESpec

spec = QPESpec(
    hamiltonian=ham,
    state_prep=hf,
    algorithm=phase_estimation,
    simulation=ham_simulation,
    error_budget=error,
)

# %% [markdown]
# ## 4. Estimate resources
#
# `.get_composite_bloq()` compiles the algorithm into a Qualtran
# [`CompositeBloq`](https://qualtran.readthedocs.io/en/latest/reference/qualtran/CompositeBloq.html)
# which can be costed:

# %%
from quiche.resources.logical import logical_gate_resources, logical_qubit_resources

bloq = spec.get_composite_bloq()

print("Qubit count:", logical_qubit_resources(bloq))
print("Logical resources:", logical_gate_resources(bloq))

# %% [markdown]
# Rotations can be converted into T gates at a synthesis cost implied by the budget:

# %%
from quiche.resources.logical import logical_rotations_to_tgates

gates = logical_gate_resources(bloq)
print(logical_rotations_to_tgates(gates, error, rotation_synthesis="direct"))

# %% [markdown]
# The algorithm can also be visualised by drawing the `bloq`. Note that this requires
# [Graphviz](https://graphviz.org/download/) to be installed.

# %%
from qualtran.drawing import show_bloq

show_bloq(bloq)

# %% [markdown]
# Each [bloq](https://qualtran.readthedocs.io/en/latest/_infra/Bloqs-Tutorial.html)
# decomposes into smaller ones; `flatten_once` expands them one level:

# %%
show_bloq(bloq.flatten_once())

# %% [markdown]
# ## 5. Simulate
#
# Similarly, a QPE specification can be used to generate a
# {py:class}`~quiche.simulation.routine.SimulationRoutine` and executed by QuEST.

# %%
from math import pi

from quiche.bindings.quest_bindings import QuESTEnv, Qureg

spec = QPESpec(
    hamiltonian=ham,
    state_prep=hf,
    algorithm=PhaseEstimation.Kitaev,
    simulation=Simulation.Trotter,
    error_budget=error,
)

routine = spec.to_quest()

with QuESTEnv():
    qureg = Qureg(spec.num_qubits)
    results = routine.evaluate(qureg)

phase = results[-1]
energy = phase * (2 * pi / spec.time)

print(f"Phase: {phase:.5f}")
print(f"Energy: {energy:.5f} Ha")

# %% [markdown]
# The routine returns one result per appended operation; the phase is the last one.
# Because the propagator is simulated for a time `spec.time`, the phase is rescaled by
# `2 * pi / spec.time` to recover an energy.

# %% [markdown]
# ## Next steps
#
# The [tutorials](../tutorials) go through each step in more depth, starting with
# [loading datasets](01_load_dataset).
