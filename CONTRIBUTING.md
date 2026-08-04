# Contributing

## Python

### Installing
For development, we use the [`uv`](https://docs.astral.sh/uv/) package manager. To install the Python package with development dependencies in editable mode
```bash
cd quiche
uv sync --group dev
```

If necessary, the C++ backend and bindings can also be rebuilt during development using
```
uv sync --reinstall-package=quiche
```

### Testing
To execute the Python test suite use [`pytest`](https://github.com/pytest-dev/pytest)
```bash
cd quiche
pytest
```

### Linting and formatting
Linting and formatting is handled using [`ruff`](https://github.com/astral-sh/ruff). For linting simply execute
```bash
ruff check
```
and for formatting use
```bash
ruff format
```

## C++

### Installing
For development, simply follow the usual installation steps for the C++ backend.

### Testing
The [`Catch2`](https://github.com/catchorg/Catch2) test suite can be enabled using the CMake build flag `QUICHE_BUILD_TESTS`.
```
cd QUICHE
cmake -B build -D QUICHE_BUILD_TESTS=ON
cmake --build build
```

Then execute the tests using
```bash
./build/cpp/tests/tests
```

### Linting and formatting
You can use the provided `.clang-tidy` and `.clang-format` as general references to guide code style and static analysis.
They are not strictly enforced, and deviations are acceptable where appropriate.

## Styleguide

### Commit messages
Aim to keep your PRs and commits self-contained and commit messages descriptive. Although not strictly enforced we recommend following the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) format.

### Changelog
Before a new release, the changelog file (`CHANGELOG.md`) should be updated, following the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.
