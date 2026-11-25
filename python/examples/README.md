# QUICHE tutorials

Tutorials that walk through QUICHE, from loading a molecular Hamiltonian to
estimating the fault-tolerant cost of quantum phase estimation (QPE) and simulating it
with QuEST. Sections build on each other, so we recommend reading them in order.

## Tutorials

| File | Topic | What it covers |
| --- | --- | --- |
| [01_load_dataset.py](01_load_dataset.py) | Loading data | Reading a `.fcidump` file into a `SecondQuantisedHamiltonian` with `quiche.io.fcidump`, and a Hamlib `.hdf5` dataset into a `PauliSum` with `quiche.io.hamlib`. |
| [02_pauli_operators.py](02_pauli_operators.py) | Pauli operators | The `Pauli`, `PauliWord` and `PauliSum` classes: building operators, converting them to and from dense and sparse strings, converting them to matrices, the 1-norm, and splitting off the identity term. |
| [03_fermion_to_qubit.py](03_fermion_to_qubit.py) | Fermion-to-qubit mappings | The Jordan-Wigner, Bravyi-Kitaev and Parity mappings in `quiche.chemistry`, Hartree-Fock states, and mapping a second-quantised Hamiltonian to a qubit Hamiltonian with `to_qubit_hamiltonian`. |
| [04_resource_estimation.py](04_resource_estimation.py) | Resource estimation | The full `QPESpec` → Qualtran pipeline. Compares Trotterisation, QDRIFT and qubitisation on logical metrics (qubits, Toffoli and T counts) and physical surface code metrics, as the system size and the error budget change. |
| [05_simulation.py](05_simulation.py) | QuEST simulation | Runs the same `QPESpec` calculations as state-vector simulations in QuEST. Compares textbook and Kitaev QPE with each simulation method on H<sub>2</sub> for energy error, number of simulated qubits, memory and runtime. |
| [06_resource_estimation_using_dfthc.py](06_resource_estimation_using_dfthc.py) | DFTHC resource estimation | The DFTHC block encoding with spectral amplification of [Low et al. (2025)](https://doi.org/10.1103/pb2g-j9cw) for FeMoco-54: building a `DFTHCWalkOperator` from a `DFTHCHamiltonian`, the Toffoli cost of each subroutine, tuning the QROAM block sizes, and the total Toffoli count and surface code cost of phase estimation. |

## Supporting files

- [random_hamiltonian.py](random_hamiltonian.py): helpers that generate random
  second-quantised Hamiltonians with real-valued spatial orbitals
  (`generate_random_one_body_integrals`, `generate_random_two_body_integrals`,
  `generate_random_sq_hamiltonian`). Tutorials 03–05 use them to build systems of any
  size from a NumPy random number generator (`np.random.Generator`). These Hamiltonians
  are not real molecules.

## Data sources

The repository doesn't ship any data files. The tutorials get their data as follows:

- **FCIDUMP for H<sub>2</sub>** (2 spatial orbitals, 2 electrons): generated using
  [ORCA](https://www.faccts.de/orca/) with the STO-3G basis set. Tutorials 01 and 05 contain it as a string.
- **Qubit Hamiltonians for H<sub>2</sub>** (`H2.hdf5`): from
  [HamLib](https://portal.nersc.gov/cfs/m888/dcamps/hamlib/)
  [[Sawaya et al. (2024)](https://arxiv.org/abs/2306.13126)], for several mappings and
  numbers of qubits (for example `ham_JW-4`).

## Running the tutorials

The tutorials are [jupytext](https://jupytext.readthedocs.io/) percent-format `.py` files,
so they can be opened as notebooks or run as plain scripts.

1. Install QUICHE (see the [top-level README](../../README.md#installation)).

2. Start Jupyter **from this directory**. The tutorials download data into
   `python/examples`, and import `random_hamiltonian` from it:

   ```bash
   cd python/examples
   jupyter lab
   ```

Tutorial 05 runs full state-vector simulations and takes a few minutes. Tutorial 06 takes
about two minutes, mostly for the QROAM block size scan.
