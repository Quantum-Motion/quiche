# Installation

## Basic install
```{include} ../README.md
:start-after: <!-- readme-installation-start -->
:end-before: <!-- readme-installation-end -->
```

## Custom install

Custom installs require building QUICHE from source, which needs CMake, a C++17 compiler and network access to fetch dependencies. MPI- and GPU-enabled builds additionally require the corresponding toolchains, which are not fetched automatically.

QUICHE's compiled extension depends on:
- [QuEST](https://github.com/QuEST-Kit/QuEST) for simulation. Always fetched from a pinned source archive, since QUICHE depends on internal headers that are not part of the installed interface.
- [nanobind](https://github.com/wjakob/nanobind) for Python bindings. Toggled with `QUICHE_BUILD_BINDINGS`.
- [Catch2](https://github.com/catchorg/Catch2) for testing. Toggled with `QUICHE_BUILD_TESTS`.

nanobind and Catch2 are located with `find_package` and downloaded via `FetchContent` if not found.

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

Build options are set with `-D<OPTION>=<VALUE>`. See the [QuEST docs](https://quest-kit.github.io/QuEST/) for the available simulator options.

Then install using:
```bash
cmake --install build --prefix </path/to/install>
```
