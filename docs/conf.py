# Copyright 2026 Quantum Motion Technologies Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sphinx configuration for the QUICHE documentation."""

import sys
import tomllib
from pathlib import Path

DOCS_DIR = Path(__file__).parent.resolve()
REPO_ROOT = DOCS_DIR.parent
PACKAGE_SRC = REPO_ROOT / "python" / "src"

# Document the in-tree package, so that the docs can be built without installing it.
sys.path.insert(0, str(PACKAGE_SRC))

# -- Project information -----------------------------------------------------

project = "QUICHE"
author = "Quantum Motion Technologies Ltd."
copyright = "2026, Quantum Motion Technologies Ltd."  # noqa: A001

with (REPO_ROOT / "pyproject.toml").open("rb") as f:
    release = tomllib.load(f)["project"]["version"]

version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_design",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

# -- Autodoc -----------------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"

# The QuEST/QUICHE bindings are a compiled extension module built by CMake. When the
# docs are built against a source checkout that has not been compiled (the default in
# CI, which needs no C++ toolchain) the bindings are mocked so that the pure-Python
# modules can still be imported and documented.
try:
    import quiche.bindings.quest_bindings  # noqa: F401
except ImportError:
    autodoc_mock_imports = ["quiche.bindings"]
    print("NOTE: compiled bindings not importable, mocking 'quiche.bindings'.")

# -- Napoleon ----------------------------------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
# QUICHE docstrings use a couple of sections that are not part of numpydoc.
napoleon_custom_sections = [
    ("Properties", "params_style"),
    ("Resources", "notes_style"),
]

# -- MyST ----------------------------------------------------------------------

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]
myst_heading_anchors = 3

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "sympy": ("https://docs.sympy.org/latest", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = f"QUICHE {version}"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/Quantum-Motion/quiche/",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/Quantum-Motion/quiche",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" '
                'viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 '
                "8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 "
                "0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1"
                ".13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 "
                "2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 "
                "0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64"
                "-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 "
                "1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 "
                "3.95.28.24.53.71.53 1.43 0 1.03-.01 1.87-.01 2.13 0 .21.15.46.55.38A8"
                '.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>'
            ),
            "class": "",
        },
    ],
}
