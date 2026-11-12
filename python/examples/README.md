# QUICHE tutorials

Jupyter notebooks that walk through QUICHE, from loading a molecular Hamiltonian to
estimating the fault-tolerant cost of quantum phase estimation (QPE) and simulating it
with QuEST. The numbered notebooks build on each other, so we recommend reading them in
order.

## Notebooks

| Notebook | Topic | What it covers |
| --- | --- | --- |
| [01_load_dataset.ipynb](01_load_dataset.ipynb) | Loading data | Reading a `.fcidump` file into a `SecondQuantisedHamiltonian` with `quiche.io.fcidump`, and a Hamlib `.hdf5` dataset into a `PauliSum` with `quiche.io.hamlib`. |
| [02_pauli_operators.ipynb](02_pauli_operators.ipynb) | Pauli operators | The `Pauli`, `PauliWord` and `PauliSum` classes: building operators, converting them to and from dense and sparse strings, converting them to matrices, the 1-norm, and splitting off the identity term. |
| [03_fermion_to_qubit.ipynb](03_fermion_to_qubit.ipynb) | Fermion-to-qubit mappings | The Jordan–Wigner, Bravyi–Kitaev and Parity mappings in `quiche.chemistry`, Hartree–Fock states, and mapping a second-quantised Hamiltonian to a qubit Hamiltonian with `to_qubit_hamiltonian`. |
| [04_resource_estimation.ipynb](04_resource_estimation.ipynb) | Resource estimation | The full `QPESpec` → Qualtran pipeline. Compares Trotterisation, QDRIFT and qubitisation on logical metrics (qubits, Toffoli and T counts) and physical surface code metrics, as the system size and the error budget change. |
| [05_simulation.ipynb](05_simulation.ipynb) | QuEST simulation | Runs the same `QPESpec` calculations as state-vector simulations in QuEST. Compares textbook and Kitaev QPE with each simulation method on H<sub>2</sub> for energy error, number of simulated qubits, memory and runtime. |

## Supporting files

- [random_hamiltonian.py](random_hamiltonian.py): helpers that generate random
  second-quantised Hamiltonians with real-valued spatial orbitals
  (`generate_random_one_body_integrals`, `generate_random_two_body_integrals`,
  `generate_random_sq_hamiltonian`). Notebooks 03–05 use them to build systems of any
  size from a NumPy random number generator (`np.random.Generator`). These Hamiltonians
  are not real molecules.

## Data sources

The repository doesn't ship any data files. The notebooks get their data as follows:

- **FCIDUMP for H<sub>2</sub>** (2 spatial orbitals, 2 electrons): generated using
  [ORCA](https://www.faccts.de/orca/) with the STO-3G basis set. Notebooks 01 and 05 contain it as a string.
- **Qubit Hamiltonians for H<sub>2</sub>** (`H2.hdf5`): from
  [HamLib](https://portal.nersc.gov/cfs/m888/dcamps/hamlib/)
  [[Sawaya et al. (2024)](https://arxiv.org/abs/2306.13126)], for several mappings and
  numbers of qubits (for example `ham_JW-4`).

## Running the notebooks

1. Install QUICHE (see the [top-level README](../../README.md#installation)).

2. Start Jupyter **from this directory**. The notebooks download data into
   `python/examples`, and import `random_hamiltonian` from it:

   ```bash
   cd python/examples
   jupyter lab
   ```

Notebook 05 runs full state-vector simulations and takes a few minutes.
