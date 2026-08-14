# QUICHE

**QU**antum **I**ntegrated **CHE**mistry is a toolkit for studying quantum computing
algorithms for quantum chemistry, with a focus on ground state energy calculations
via quantum phase estimation (QPE).

The main entrypoint into QUICHE is a single specification object, ``QPESpec``,
which contains information to describe the target chemical system, such as
Hamiltonian, the input state, choice of QPE circuit and Hamiltonian simulation
method, etc. Once defined, ``QPESpec`` can output in either of two backends:

- a **resource estimation** backend built on [Qualtran](https://github.com/quantumlib/Qualtran),
  which compiles the calculation to a circuit and counts the logical qubits and gates it
  would need;
- a **simulation** backend built on [QuEST](https://github.com/QuEST-Kit/QuEST), which
  executes the calculation as a state-vector simulation and returns the estimated phase.

:::{warning}
QUICHE is in early active development and should not be considered production-ready.
Breaking changes may occur without notice.
:::

```{code-block} python
:caption: From a Hamiltonian to a logical resource estimate

from quiche import hamlib
from quiche.core import ElectronicHamiltonian, Errors, Mapping, PhaseEstimation, Simulation
from quiche.chemistry import HartreeFockState
from quiche.dispatch import QPESpec
from quiche.resources.logical import logical_gate_resources, logical_qubit_resources

paulis = hamlib.parse_hamiltonian(hamlib.get_dataset("H2.hdf5", "ham_JW-4"))

spec = QPESpec(
    hamiltonian=ElectronicHamiltonian(electrons=2, paulis=paulis, mapping=Mapping.JordanWigner),
    state_prep=HartreeFockState.closed_shell(electrons=2, spin_orbitals=4),
    algorithm=PhaseEstimation.Textbook,
    simulation=Simulation.Qubitised,
    error_budget=Errors(estimation=0.5, simulation=0.5, rotations=0.5, state_prep=0.5, overlap=0.5),
)

bloq = spec.get_composite_bloq()
print(logical_qubit_resources(bloq), logical_gate_resources(bloq))
```

## Getting started

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`download` Installation
:link: installation
:link-type: doc

How to install QUICHE, including the C++ simulator bindings.
:::

:::{grid-item-card} {octicon}`rocket` Quickstart
:link: quickstart
:link-type: doc

Go from a Hamiltonian to a resource estimate and a simulated phase in a few lines.
:::

:::{grid-item-card} {octicon}`code` API reference
:link: api/index
:link-type: doc

Every public module, class and function in the `quiche` package.
:::
::::

## Features

- A range of quantum phase estimation algorithms, including single- and multi-ancilla
  methods.
- A wide variety of Hamiltonian simulation techniques: Suzuki-Trotter, QDRIFT and
  qubitisation.
- Extensible, Python-based resource estimation tooling.
- High-performance state-vector simulation.

## Funding

QUICHE is a UK-Germany collaboration between [Quantum Motion](https://quantummotion.com),
[FACCTs](https://www.faccts.de) (developers of ORCA) and
[Riverlane](https://www.riverlane.com), supported by Innovate UK and Germany's ZIM via
the [QUantum-Integrated CHEmistry (QUICHE)](https://gtr.ukri.org/projects?ref=10150101)
project. See the [project announcement](https://quantummotion.com/quiche-project-uk-germany-collaboration-bringing-chemistry-software-into-the-quantum-computing-era/)
for background on its aims.

```{toctree}
:hidden:
:caption: User guide

installation
quickstart
examples/index
```

```{toctree}
:hidden:
:caption: Reference

api/index
contributing
changelog
License <https://github.com/Quantum-Motion/quiche/blob/main/LICENSE>
```
