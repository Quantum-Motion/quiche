# Contributing

## Python

### Installing
For development, we use the [`uv`](https://docs.astral.sh/uv/) package manager. To install the Python package with development dependencies in editable mode
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
```bash
cd quiche
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
