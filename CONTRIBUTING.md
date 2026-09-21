# Contributing

## Python

### Installing
For development, we use the [`uv`](https://docs.astral.sh/uv/) package manager.
To install the Python package with development dependencies in editable mode
```bash
cd quiche
uv sync --group dev
```

If necessary, the C++ backend and bindings can also be rebuilt during development using
```bash
uv sync --reinstall-package=pyquiche
```

### Testing
To execute the Python test suite use [`pytest`](https://github.com/pytest-dev/pytest)
```bash
cd quiche
pytest
```

### Linting and formatting
Linting and formatting are handled using [`ruff`](https://github.com/astral-sh/ruff).
For linting execute
```bash
ruff check
```
and for formatting use
```bash
ruff format
```

The enabled rules are configured in `pyproject.toml` under `[tool.ruff]`.

## C++

### Installing
For development, follow the usual installation steps for the C++ backend.

### Testing
The [`Catch2`](https://github.com/catchorg/Catch2) test suite can be enabled using the CMake build flag `QUICHE_BUILD_TESTS`.
```bash
cd quiche
cmake -B build -D QUICHE_BUILD_TESTS=ON
cmake --build build
```

Tests are registered with CTest via `catch_discover_tests`, so you can execute them with
```bash
ctest --test-dir build --output-on-failure
```
The Catch2 binary can also be run directly with `./build/cpp/tests/tests`.

### Linting and formatting
Formatting is handled using [`clang-format`](https://clang.llvm.org/docs/ClangFormat.html), with the style defined in `.clang-format`.
To format a file in place use `-i`
```bash
clang-format -i some_file.cpp
```
To check a file without writing use `--dry-run --Werror`
```bash
clang-format --dry-run --Werror some_file.cpp
```

`.clang-tidy` is provided as a general reference for static analysis; it isn't currently enforced.

## Pre-commit hooks

Basic hygiene and secret-scanning checks are handled using pre-commit hooks, configured in `.pre-commit-config.yaml`.
The checks can be run with [`pre-commit`](https://github.com/pre-commit/pre-commit) or [`prek`](https://github.com/j178/prek).

To install the hooks so they run automatically on `git commit`
```bash
prek install
```

To run the hooks on currently staged files
```bash
prek run
```

## Continuous integration

Every pull request and push to `main` runs:

- Python tests: `pytest`
- Python lint: `ruff check` and `ruff format --check`
- C++ tests: build with `QUICHE_BUILD_TESTS=ON`, then `ctest`
- C++ lint: `clang-format` in check mode

## Styleguide

### Commit messages
Aim to keep your PRs and commits self-contained and commit messages descriptive.
Although not strictly enforced we recommend following the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) format.

### Changelog
Changes to the API, behaviour, packaging or build requirements should be recorded in the `[Unreleased]` section of `CHANGELOG.md`, following the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

## Releasing

1. Open a PR to `main` that:
    - Renames the `[Unreleased]` section of `CHANGELOG.md` to `## [X.Y.Z] - YYYY-MM-DD`, and adds a new `[Unreleased]` above it with empty category headings.
    - Bumps `version` in `pyproject.toml`.
    - Bumps `VERSION` in `CMakeLists.txt`.
2. Once merged, tag the merge commit with the version from step 1 (prefixed with `v`) and push it:
    ```bash
    git tag vX.Y.Z
    git push origin vX.Y.Z
    ```
3. The `publish-wheels.yml` workflow will trigger on the tag, build the wheels and sdist, then pause for approval before uploading to PyPI.

> A published version is permanent. A release can be yanked (hidden from dependency resolution) but never replaced or re-uploaded, so fixes require a new version number.
