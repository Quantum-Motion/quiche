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
Aim to keep your PRs and commits self-contained and commit messages descriptive. Although not strictly enforced we recommend following the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) format.

### Changelog
Before a new release, the changelog file (`CHANGELOG.md`) should be updated, following the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.
