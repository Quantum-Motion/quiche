# Installation

The steps below are shared with the [repository README](https://github.com/Quantum-Motion/quiche#installation).

```{include} ../README.md
:start-after: "## Installation"
:end-before: "## C++ backend only"
```

## C++ backend only

```{include} ../README.md
:start-after: "## C++ backend only"
:end-before: "## Usage"
```

## Building the documentation

The docs are built with [Sphinx](https://www.sphinx-doc.org/) and document the package
from the source tree, so no compilation of the C++ backend is required:

```bash
cd quiche
uv sync --group docs --no-install-project
source .venv/bin/activate
cd docs
make html
```

The rendered site is written to `docs/_build/html`. If the compiled bindings happen to
be importable they are documented too; otherwise they are mocked and the build proceeds.
