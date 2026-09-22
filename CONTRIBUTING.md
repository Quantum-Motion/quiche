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

## Continuous integration

Every pull request and push to `main` runs:

- Python tests: `pytest`
- Python lint: `ruff check` and `ruff format --check`
- C++ tests: build with `QUICHE_BUILD_TESTS=ON`, then `ctest`
- C++ lint: `clang-format` in check mode

## Documentation

The documentation lives in `docs` and is built with [`sphinx`](https://www.sphinx-doc.org/) using [MyST](https://myst-parser.readthedocs.io/) markdown. To install the documentation dependencies and build the site
```bash
cd quiche
uv sync --group docs --no-install-project
source .venv/bin/activate
cd docs
make html
```

The `--no-install-project` flag skips compiling the C++ backend: the docs are built against the sources in `python/src` and the compiled bindings are mocked if they cannot be imported. The rendered site is written to `docs/_build/html`, and `make clean` removes it.

A few things worth knowing when editing the docs:
- The Python API pages under `docs/api` use `automodule`, so docstrings are the source of truth. Adding a new module means adding one `automodule` entry.
- Docstrings follow the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) format. The non-standard `Properties` and `Resources` sections used by the bloqs are mapped onto numpydoc sections in `docs/conf.py`.
- Notebooks in `python/examples` are copied into the docs at build time and rendered from their stored outputs — they are never executed by the docs build, so commit notebooks with the outputs you want published.
- The `docs` workflow builds the documentation with warnings treated as errors and deploys it to GitHub Pages on every push to `main`. Pull requests build the docs without deploying them.

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
