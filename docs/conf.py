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

from pathlib import Path

DOCS_DIR = Path(__file__).parent.resolve()
REPO_ROOT = DOCS_DIR.parent

# -- Project information -----------------------------------------------------

project = "QUICHE"
author = "Quantum Motion Technologies Ltd."
copyright = "2026, Quantum Motion Technologies Ltd."  # noqa: A001

release = (REPO_ROOT / "VERSION").read_text().strip()
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_design",
]

exclude_patterns = [
    "_build",
    ".DS_Store",
    "conf.py",
    "tutorials/random_hamiltonian.py",
    "tutorials/README.md",
]

add_module_names = False
python_use_unqualified_type_names = True
toc_object_entries_show_parents = "hide"

# -- Autodoc -----------------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
}
autodoc_typehints = "signature"
autodoc_typehints_format = "short"

# -- Napoleon ----------------------------------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_custom_sections = [("Registers", "params_style")]

# -- MyST ----------------------------------------------------------------------

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "alert"]
myst_heading_anchors = 3

# -- MyST-NB -------------------------------------------------------------------

nb_custom_formats = {".py": ["jupytext.reads", {"fmt": "py:percent"}]}
nb_execution_mode = "cache"
nb_execution_timeout = 600  # TODO(Vasco): update tutorials to be smaller
nb_execution_show_tb = True

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "sympy": ("https://docs.sympy.org/latest", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "shibuya"
html_theme_options = {
    "nav_socials": [],
    "github_url": "https://github.com/Quantum-Motion/quiche",
    "show_ai_links": False,
}
html_baseurl = "https://quantum-motion.github.io/quiche/"
