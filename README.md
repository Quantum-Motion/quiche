# QUICHE

<!-- readme-intro-start -->

**QU**antum **I**ntegrated **CHE**mistry is a toolkit for studying quantum computing
algorithms for quantum chemistry, with a focus on ground-state energy calculations
via quantum phase estimation (QPE).

The main entry point to QUICHE is a `QPESpec` object, which describes the target chemical system, phase estimation circuit and Hamiltonian simulation method. `QPESpec` then dispatches to either of two backends:

- a **resource estimation** backend built on [Qualtran](https://github.com/quantumlib/Qualtran), which decomposes the calculation into its subroutines and counts the logical qubits and gates it would need
- a **simulation** backend built on [QuEST](https://github.com/QuEST-Kit/QuEST), which executes the calculation as a state-vector simulation and returns the estimated phase.

> [!WARNING]
> This project is in early active development and should not be considered production-ready.  
> Breaking changes may occur without notice before v1.0.

<!-- readme-intro-end -->
---
<!-- readme-features-start -->

## Features

- A range of quantum phase estimation algorithms including single- and multi-ancilla methods.
- A wide variety of Hamiltonian simulation techniques, such as Suzuki–Trotter, QDRIFT and qubitisation.
- Extensible, Python-based resource estimation tooling.
- High-performance simulation capabilities.

<!-- readme-features-end -->
---
<!-- readme-usage-start -->

## Usage

Going from a Hamiltonian to a logical resource estimate:
```python
from quiche.io import hamlib
from quiche.core import ElectronicHamiltonian, Errors, Mapping, PhaseEstimation, Simulation
from quiche.chemistry import HartreeFockState
from quiche.dispatch import QPESpec
from quiche.resources.logical import logical_gate_resources, logical_qubit_resources

paulis = hamlib.parse(hamlib.read_dataset("H2.hdf5", "ham_JW-4"))

spec = QPESpec(
    hamiltonian=ElectronicHamiltonian(electrons=2, paulis=paulis, mapping=Mapping.JordanWigner),
    state_prep=HartreeFockState.closed_shell(electrons=2, spin_orbitals=4),
    algorithm=PhaseEstimation.Textbook,
    simulation=Simulation.Qubitised,
    error_budget=Errors(estimation=1e-3, simulation=1e-3, rotations=1e-4, state_prep=1e-4, overlap=0.90),
)

bloq = spec.get_composite_bloq()
print(logical_qubit_resources(bloq), logical_gate_resources(bloq))
```

<!-- readme-usage-end -->
---
<!-- readme-installation-start -->

## Installation

### Basic install

QUICHE is available on PyPI as `pyquiche`. For a basic install, run:
```bash
pip install pyquiche
```

Then import the package from Python:
```python
import quiche
```

Prebuilt wheels are available for Linux (`x86_64`, `aarch64`), macOS 15+ (`arm64`) and Windows (`x64`), on Python 3.12 or later. They bundle a multithreaded, double-precision build of QuEST. On other platforms, pip falls back to [building from source](#custom-install). That is also needed for custom precision, GPU acceleration or MPI-enabled builds.

For an editable development install, see the [contributing guide](https://github.com/Quantum-Motion/quiche/blob/main/CONTRIBUTING.md).


### Custom install

Custom installs require building QUICHE from source, which needs CMake, a C++17 compiler and network access to fetch dependencies. MPI- and GPU-enabled builds additionally require the corresponding toolchains, which are not fetched automatically.

QUICHE's compiled extension depends on:
- [QuEST](https://github.com/QuEST-Kit/QuEST) for simulation. Always built from a pinned source archive, since QUICHE depends on internal headers that are not part of the installed interface.
- [nanobind](https://github.com/wjakob/nanobind) for Python bindings. Toggled with `QUICHE_BUILD_BINDINGS`.
- [Catch2](https://github.com/catchorg/Catch2) for testing. Toggled with `QUICHE_BUILD_TESTS`.

nanobind and Catch2 are located with `find_package` and downloaded via `FetchContent` if unavailable.

To build the Python package from the published `sdist`, passing configuration flags through to CMake:
```bash
pip install pyquiche --no-binary pyquiche -C cmake.define.ENABLE_DISTRIBUTION=ON
```

> [!TIP]
> `--no-binary` is required: without it pip installs the prebuilt wheel and the configuration flags are ignored.

To build from a local clone instead, run:
```bash
git clone https://github.com/Quantum-Motion/quiche.git
cd quiche
pip install .
```

### C++-only install

To build only the C++ simulator backend, run:
```bash
cd quiche
cmake -B build
cmake --build build
```

Other build options can be customised by passing CMake flags (`-D<OPTION>=<VALUE>`). Visit the [QuEST docs](https://quest-kit.github.io/QuEST/) for detailed information on the available simulator options.

Then install using:
```bash
cmake --install build --prefix </path/to/install>
```

<!-- readme-installation-end -->
---
<!-- readme-contributing-start -->

## Contributing

For further information about how you can contribute to QUICHE, see the [contributing guide](https://github.com/Quantum-Motion/quiche/blob/main/CONTRIBUTING.md).

<!-- readme-contributing-end -->
---
<!-- readme-funding-start -->

## Funding

QUICHE is a UK–Germany collaboration between [Quantum Motion](https://quantummotion.com),
[FACCTs](https://www.faccts.de) (developers of ORCA) and
[Riverlane](https://www.riverlane.com), supported by Innovate UK and Germany's ZIM via
the [QUantum-Integrated CHEmistry (QUICHE)](https://gtr.ukri.org/projects?ref=10150101)
project. See the [project announcement](https://quantummotion.com/quiche-project-uk-germany-collaboration-bringing-chemistry-software-into-the-quantum-computing-era/)
for background on its aims.

<!-- readme-funding-end -->
---
<!-- readme-license-start -->

## License

Copyright 2026 Quantum Motion Technologies Ltd. Licensed under the Apache License, Version 2.0.

<!-- readme-license-end -->
