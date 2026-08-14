# Quickstart

This page walks through a complete QUICHE calculation on the hydrogen molecule: loading
a Hamiltonian, describing the calculation once, and then dispatching that description to
both backends.

It uses the `H2.hdf5` [Hamlib](https://arxiv.org/abs/2306.13126) file shipped with the repository in
[`python/examples/H2`](https://github.com/Quantum-Motion/quiche/tree/main/python/examples/H2),
so run it from that directory (or adjust the path).

## 1. Load a Hamiltonian

QUICHE can read a Hamiltonians from the [Hamlib](https://portal.nersc.gov/cfs/m888/dcamps/hamlib/)
HDF5 library and parses into a {py:class}`~quiche.core.paulis.PauliSum`:

```python
from quiche import hamlib

raw_data = hamlib.get_dataset("H2.hdf5", "ham_JW-4")
paulis = hamlib.parse_hamiltonian(raw_data)

print(paulis.n_qubits, paulis.n_terms, paulis.lam)
```

```text
4 14 1.2336349637769919
```

A `PauliSum` is a linear combination of Pauli words plus an identity coefficient. Its
`lam` property — the 1-norm of the coefficients — is what sets the simulation time and
the cost of qubitisation.

Pairing the operator with the electron count and the fermion-to-qubit mapping it was
generated with gives an {py:class}`~quiche.core.electronic.ElectronicHamiltonian`:

```python
from quiche.core import ElectronicHamiltonian, Mapping

ham = ElectronicHamiltonian(
    electrons=2,
    paulis=paulis,
    mapping=Mapping.JordanWigner,
)
```

## 2. Set an error budget

Every approximation in the calculation draws on an {py:class}`~quiche.core.errors.Errors`
budget, and QUICHE derives the routine parameters — ancilla counts, Trotter steps, QDRIFT
repetitions — from it. A deliberately loose budget keeps this example small enough to
simulate:

```python
from quiche.core import Errors

eps = 0.5
error = Errors(
    estimation=eps,
    simulation=eps,
    rotations=eps,
    state_prep=eps,
    overlap=eps,
)
```

## 3. Specify the calculation

The initial state is a Hartree-Fock state, which QUICHE maps into the qubit basis using
the mapping recorded on the Hamiltonian:

```python
from quiche.chemistry import HartreeFockState

hf = HartreeFockState.closed_shell(electrons=2, spin_orbitals=4)
```

{py:class}`~quiche.dispatch.qpespec.QPESpec` ties everything together. Constructing it
gives the recipe to generate the phase estimation circuit to estimate the
ground state energy:

```python
from quiche.core import PhaseEstimation, Simulation
from quiche.dispatch import QPESpec

spec = QPESpec(
    hamiltonian=ham,
    state_prep=hf,
    algorithm=PhaseEstimation.Textbook,
    simulation=Simulation.Qubitised,
    error_budget=error,
)

print("Total qubits:", spec.num_qubits)
print("QPE ancillas:", spec.num_qpe_ancillas)
```

```text
Total qubits: 18
QPE ancillas: 6
```

## 4. Estimate resources

`get_composite_bloq` compiles the specification into a Qualtran
[`CompositeBloq`](https://qualtran.readthedocs.io/en/latest/reference/qualtran.CompositeBloq.html),
which can be costed or drawn:

```python
from quiche.resources.logical import logical_gate_resources, logical_qubit_resources

bloq = spec.get_composite_bloq()

print("Qubit count:", logical_qubit_resources(bloq))
print("Logical resources:", logical_gate_resources(bloq))
```

```text
Qubit count: 27
Logical resources: t: 4, toffoli: 640, and_bloq: 3574, clifford: 16722, rotation: 6, measurement: 3574
```

Rotations can be converted into T gates at a synthesis cost implied by the budget:

```python
from quiche.resources.logical import logical_rotations_to_tgates

gates = logical_gate_resources(bloq)
print(logical_rotations_to_tgates(gates, error, rotation_synthesis="direct"))
```

```text
t: 64, toffoli: 640, and_bloq: 3574, clifford: 16722, measurement: 3574
```

To inspect the circuit, flatten the bloq one level at a time and draw it:

```python
from qualtran.drawing import show_bloq

show_bloq(bloq.flatten_once())
```

## 5. Simulate

The same specification can instead be turned into a
{py:class}`~quiche.simulation.routine.SimulationRoutine` and executed by QuEST. Not every
algorithm and simulation method is wired up in both backends yet, so this example
switches to Kitaev QPE with Trotterisation:

```python
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
energy = phase * (2 * pi / spec.time) + paulis.identity_coefficient

print(f"Phase: {phase:.5f}")
print(f"Energy: {energy:.5f} Ha")
```

```text
Phase: -0.27709
Energy: -1.13144 Ha
```

The routine returns one result per appended operation; the phase is the last one. Because
the propagator is simulated for a time `spec.time`, the phase is rescaled by
`2 * pi / spec.time` to recover an energy, and the identity coefficient — which QUICHE
does not simulate — is added back. Under qubitisation the walk operator's eigenphase
relates to the energy differently:

```python
from math import cos

energy = cos(phase * 2 * pi) * paulis.lam + paulis.identity_coefficient
```

## Next steps

- [Examples](examples/index.md) — the same workflow as a runnable notebook.
- [API reference](api/index.rst) — everything the package exposes.
