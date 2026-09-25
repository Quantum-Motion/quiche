# Quickstart

This page walks through a complete QUICHE calculation from loading a Hamiltonian, to setting
the phase estimation calculation, and finally dispatching to both backends.

> [!TIP]
> This example uses the `H2.hdf5` [Hamlib](https://arxiv.org/abs/2306.13126) dataset.
> To download and unzip the dataset run:
> ```bash
> curl -O https://portal.nersc.gov/cfs/m888/dcamps/hamlib/chemistry/electronic/standard/H2.zip && unzip -n -q H2.zip
> ```

## 1. Load a Hamiltonian

QUICHE can open Hamiltonians from the [Hamlib](https://portal.nersc.gov/cfs/m888/dcamps/hamlib/)
library and parse them into a {py:class}`~quiche.core.paulis.PauliSum`.

```python
from quiche.io import hamlib

raw_data = hamlib.read_dataset("H2.hdf5", "ham_JW-4")
paulis = hamlib.parse(raw_data)

print(paulis.n_qubits, paulis.n_terms, paulis.lam)
```

```text
4 14 1.6814325431165813
```

Pairing the sum with the electron count and the relevant fermion-to-qubit mapping one can define
an {py:class}`~quiche.core.electronic.ElectronicHamiltonian`.

```python
from quiche.core import ElectronicHamiltonian, Mapping

ham = ElectronicHamiltonian(
    electrons=2,
    paulis=paulis,
    mapping=Mapping.JordanWigner,
)
```

## 2. Set an error budget

Every approximate method in the calculation draws on an {py:class}`~quiche.core.errors.Errors`
 budget. Using it, QUICHE derives all relevant parameters, such as ancilla counts, number of Trotter steps, QDRIFT repetitions, and so on.

```python
from quiche.core import Errors

error = Errors(
    estimation=1e-2,
    simulation=1e-3,
    rotations=1e-3,
    state_prep=1e-3,
    overlap=0.9,
)
```

## 3. Specify the calculation

An initial state for the phase estimation calculation, such as a Hartree-Fock state, can be set
```python
from quiche.chemistry import HartreeFockState

hf = HartreeFockState.closed_shell(electrons=2, spin_orbitals=4)
```

and the specific algorithms for Hamiltonian simulation and phase estimation can be picked:
```python
from quiche.core import PhaseEstimation, Simulation

phase_estimation = PhaseEstimation.Textbook
ham_simulation = Simulation.Qubitised
```

{py:class}`~quiche.dispatch.qpespec.QPESpec` brings everything together defining the complete phase estimation calculation.
```python
from quiche.dispatch import QPESpec

spec = QPESpec(
    hamiltonian=ham,
    state_prep=hf,
    algorithm=phase_estimation,
    simulation=ham_simulation,
    error_budget=error,
)
```

## 4. Estimate resources

`.get_composite_bloq()` compiles the algorithm into a Qualtran
[`CompositeBloq`](https://qualtran.readthedocs.io/en/latest/reference/qualtran/CompositeBloq.html)  which can be costed:

```python
from quiche.resources.logical import logical_gate_resources, logical_qubit_resources

bloq = spec.get_composite_bloq()

print("Qubit count:", logical_qubit_resources(bloq))
print("Logical resources:", logical_gate_resources(bloq))
```

```text
Qubit count: 51
Logical resources: t: 10, toffoli: 225280, and_bloq: 166021, clifford: 1403494, rotation: 45, measurement: 166021
```

Rotations can be converted into T gates at a synthesis cost implied by the budget:

```python
from quiche.resources.logical import logical_rotations_to_tgates

gates = logical_gate_resources(bloq)
print(logical_rotations_to_tgates(gates, error, rotation_synthesis="direct"))
```

```text
t: 2080, toffoli: 225280, and_bloq: 166021, clifford: 1403494, measurement: 166021
```

To visualise the circuit, flatten the `bloq` one level at a time and draw it:

```python
from qualtran.drawing import show_bloq

show_bloq(bloq.flatten_once())
```

## 5. Simulate

Similarly, a QPE specification can be used to generate a
{py:class}`~quiche.simulation.routine.SimulationRoutine` and executed by QuEST.

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
energy = phase * (2 * pi / spec.time)

print(f"Phase: {phase:.5f}")
print(f"Energy: {energy:.5f} Ha")
```

```text
Phase: -0.33646
Energy: -1.13146 Ha
```

The routine returns one result per appended operation; the phase is the last one. Because
the propagator is simulated for a time `spec.time`, the phase is rescaled by
`2 * pi / spec.time` to recover an energy.
