# QUICHE

QUICHE (QUantum Integrated CHEmistry) is a toolkit for studying quantum computing algorithms for quantum chemistry, with a focus on quantum phase estimation (QPE). It integrates a resource estimation backend based on [Qualtran](https://github.com/quantumlib/Qualtran), with a [QuEST](https://github.com/QuEST-Kit/QuEST)-powered simulation backend. The two backends can be used in isolation or combined using the Python package capable of dispatching between the two.

> ⚠️ This project is in early active development and should not be considered production-ready.
> Breaking changes may occur without notice before v1.0.


## Features

- Range of quantum phase estimation algorithms including various single- and multi-ancilla methods.
- Wide variety of Hamiltonian simulation techniques, such as Suzuki-Trotter, QDRIFT and qubitisation.
- Extensible, Python-based resource estimation tooling.
- High-performance simulation capabilities.


## Installation

QUICHE is available on PyPI as `pyquiche`, so for a simple install just
```bash
pip install pyquiche
```

Then import the package from Python
```python
import quiche
```

Prebuilt wheels are available for Linux (`x86_64`, `aarch64`), macOS 15+ (`arm64`) and Windows (`x64`), on Python 3.12 or later. They bundle a multithreaded, double precision build of QuEST. On other platforms pip falls back to [building from source](#building-from-source). That is also needed for custom precision, GPU acceleration or MPI-enabled builds.


## Building from source

Building from source compiles the C++ simulation backend locally. This requires CMake, a C++17 compiler and network access to fetch dependencies. MPI- and GPU-enabled builds additionally require the corresponding toolchains, which are not fetched automatically.

The simulation backend depends on:
- [QuEST](https://github.com/QuEST-Kit/QuEST) for simulation. Always built from a pinned source archive, since QUICHE depends on internal headers that are not part of the installed interface.
- [nanobind](https://github.com/wjakob/nanobind) for Python bindings. Toggled with `QUICHE_BUILD_BINDINGS`.
- [Catch2](https://github.com/catchorg/Catch2) for testing. Toggled with `QUICHE_BUILD_TESTS`.

nanobind and Catch2 are located with `find_package` and downloaded via `FetchContent` if unavailable.

To build the Python package from the published sdist, passing configuration flags through to CMake:
```bash
pip install pyquiche --no-binary pyquiche -C cmake.define.ENABLE_DISTRIBUTION=ON
```

> Note: `--no-binary` is required: without it pip installs the prebuilt wheel and the configuration flags are ignored.

Or to build from a checkout
```bash
git clone https://github.com/Quantum-Motion/quiche.git
cd quiche
pip install .
```

See the [QuEST docs](https://quest-kit.github.io/QuEST/) for the available simulation flags.

### C++ backend only

To build just the C++ simulator backend, along with the examples, simply execute
```bash
cd quiche
cmake -B build -D QUICHE_BUILD_EXAMPLES=ON
cmake --build build
```

Then execute an example (e.g. the Textbook QPE example)
```bash
./build/cpp/examples/qpe-textbook
```

The other C++ configuration flags can be similarly toggled `ON` and `OFF`.


## Usage

For Python usage examples see the [`python/examples`](https://github.com/Quantum-Motion/quiche/tree/main/python/examples) directory.
For C++ simulator usage examples see the [`cpp/examples`](https://github.com/Quantum-Motion/quiche/tree/main/cpp/examples) directory.


## Contributing

For further information about how you can contribute to QUICHE see the [contributing guide](https://github.com/Quantum-Motion/quiche/blob/main/CONTRIBUTING.md).


## License

Copyright 2026 Quantum Motion Technologies Ltd. Licensed under the Apache License, Version 2.0.


## Funding

This software is supported by Innovate UK and Germany's ZIM via the [QUantum-Integrated CHEmistry (QUICHE)](https://gtr.ukri.org/projects?ref=10150101) project.
