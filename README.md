# QUICHE

<!-- readme-intro-start -->

**QU**antum **I**ntegrated **CHE**mistry is a toolkit for studying quantum computing
algorithms for quantum chemistry, with a focus on ground-state energy calculations
via quantum phase estimation (QPE).

The main entry point to QUICHE is a `QPESpec` object, which describes the target chemical system, phase estimation circuit and Hamiltonian simulation method. It then dispatches to either of two backends:

- a **resource estimation** backend built on [Qualtran](https://github.com/quantumlib/Qualtran), which decomposes the calculation into its subroutines and counts the logical qubits and gates it would need.
- a **simulation** backend built on [QuEST](https://github.com/QuEST-Kit/QuEST), which executes the calculation as a state-vector simulation and returns the estimated phase.

> [!WARNING]
> This project is in early active development and should not be considered production-ready.  
> Breaking changes may occur without notice before v1.0.

<!-- readme-intro-end -->
---

## Features

<!-- readme-features-start -->

- A range of quantum phase estimation algorithms including single- and multi-ancilla methods.
- A wide variety of Hamiltonian simulation techniques, such as Suzuki–Trotter, QDRIFT and qubitisation.
- Extensible, Python-based resource estimation tooling.
- High-performance simulation capabilities.

<!-- readme-features-end -->
---

## Usage

<!-- readme-usage-start -->

Going from a Hamiltonian to a logical resource estimate:
```python
from quiche.io import hamlib
from quiche.core import (
    ElectronicHamiltonian,
    Errors,
    Mapping,
    PhaseEstimation,
    Simulation,
)
from quiche.chemistry import HartreeFockState
from quiche.dispatch import QPESpec
from quiche.resources.logical import logical_gate_resources, logical_qubit_resources

paulis = hamlib.parse(hamlib.read_dataset("H2.hdf5", "ham_JW-4"))

spec = QPESpec(
    hamiltonian=ElectronicHamiltonian(
        electrons=2,
        paulis=paulis,
        mapping=Mapping.JordanWigner,
    ),
    state_prep=HartreeFockState.closed_shell(electrons=2, spin_orbitals=4),
    algorithm=PhaseEstimation.Textbook,
    simulation=Simulation.Qubitised,
    error_budget=Errors(
        estimation=1e-3,
        simulation=1e-3,
        rotations=1e-4,
        state_prep=1e-4,
        overlap=0.90,
    ),
)

bloq = spec.get_composite_bloq()
print(logical_qubit_resources(bloq), logical_gate_resources(bloq))
```

<!-- readme-usage-end -->
---

## Installation

<!-- readme-installation-start -->

QUICHE is available on PyPI as `pyquiche`. For a basic install, run:
```bash
pip install pyquiche
```

Then import the package from Python:
```python
import quiche
```

Prebuilt wheels are available for Linux (`x86_64`, `aarch64`), macOS 15+ (`arm64`) and Windows (`x64`), on Python 3.12 or later. They bundle a multithreaded, double-precision build of QuEST.

For other platforms, custom precision, GPU acceleration or MPI support, QUICHE must be built from source.

<!-- readme-installation-end -->

For more information, see the [install instructions](https://quantum-motion.github.io/quiche/installation.html).

---

## Contributing

For further information about how you can contribute to QUICHE, see the [contributing guide](https://github.com/Quantum-Motion/quiche/blob/main/CONTRIBUTING.md).

---

## Funding

<!-- readme-funding-start -->

QUICHE is a UK–Germany collaboration between [Quantum Motion](https://quantummotion.com),
[FACCTs](https://www.faccts.de) (developers of ORCA) and
[Riverlane](https://www.riverlane.com), supported by Innovate UK and Germany's ZIM via
the [QUantum-Integrated CHEmistry (QUICHE)](https://gtr.ukri.org/projects?ref=10150101)
project. See the [project announcement](https://quantummotion.com/quiche-project-uk-germany-collaboration-bringing-chemistry-software-into-the-quantum-computing-era/)
for background on its aims.

<!-- readme-funding-end -->
---

## License

Copyright 2026 Quantum Motion Technologies Ltd. Licensed under the Apache License, Version 2.0.
