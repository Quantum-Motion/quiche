# QUICHE

QUICHE (QUantum Integrated CHEmistry) is a toolkit for studying quantum computing algorithms for quantum chemistry, with a focus on quantum phase estimation (QPE). It integrates a resource estimation backend based on [Qualtran](https://github.com/quantumlib/Qualtran), with a [QuEST](https://github.com/QuEST-Kit/QuEST)-powered simulation backend. The two backends can be used in isolation or combined using the Python package capable of dispatching between the two.

> ⚠️ This project is in early active development and should not be considered production-ready.
> Breaking changes may occur without notice.


## Features

- Range of quantum phase estimation algorithms including various single- and multi-ancilla methods.
- Wide variety of Hamiltonian simulation techniques, such as Suzuki-Trotter, QDRIFT and qubitisation.
- Extensible, Python-based resource estimation tooling
- High-performance simulation capabilities


## Installation

> QUICHE's simulation backend relies on [QuEST](https://github.com/QuEST-Kit/QuEST), [nanobind](https://github.com/wjakob/nanobind) (for Python bindings) and [Catch2](https://github.com/catchorg/Catch2) (for testing). The latter two can be toggled using the `QUICHE_BUILD_BINDINGS` and `QUICHE_BUILD_TESTS` flags respectively (see below for usage). nanobind and Catch2 are located with `find_package` and downloaded via `FetchContent` if unavailable. QuEST is always built from a pinned source archive, since QUICHE depends on internal QuEST headers that are not part of its installed interface.

Begin by cloning and navigating to the QUICHE repo
```bash
git clone https://github.com/Quantum-Motion/quiche.git
```

Then install the Python package, including the simulator bindings
```bash
cd quiche
python -m venv .venv
source .venv/bin/activate
pip install .
```

Then just import the package from Python
```python3
import quiche
```

Alternatively to build only the C++ simulator backend along with the examples, for instance, simply execute
```bash
cd quiche
cmake -B build -D QUICHE_BUILD_EXAMPLES=ON
cmake --build build
```

Then execute an example (e.g. the Textbook QPE example)
```bash
./build/cpp/examples/qpe-textbook
```

The other C++ configuration flags, can be similarly toggled `ON` and `OFF`. For additional configuration flags available for the QuEST simulation see also the [QuEST docs](https://quest-kit.github.io/QuEST/).


## Usage

For Python usage examples see the [`python/examples`](https://github.com/Quantum-Motion/quiche/blob/main/python/examples) directory.
For C++ simulator usage examples see the [`cpp/examples`](https://github.com/Quantum-Motion/quiche/blob/main/cpp/examples) directory.


## Contributing

For further information about how you can contribute to QUICHE see the [contributing guide](https://github.com/Quantum-Motion/quiche/blob/main/CONTRIBUTING.md).


## License

Copyright 2026 Quantum Motion Technologies Ltd. Licensed under the Apache License, Version 2.0.

## Funding

This software is supported by Innovate UK and Germany's ZIM via the [QUantum-Integrated CHEmistry (QUICHE)](https://gtr.ukri.org/projects?ref=10150101) project.
