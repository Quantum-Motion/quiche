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

"""Tests for lazy module imports and re-exports."""

import importlib
import pkgutil
from typing import Never

import pytest

import quiche


def _raise_import_error(name: str) -> Never:
    raise ImportError(name)


TOPLEVEL_MODULE_NAMES = tuple(m.name for m in pkgutil.iter_modules(quiche.__path__))
ALL_MODULE_NAMES = (
    "quiche",
    *(
        m.name
        for m in pkgutil.walk_packages(
            quiche.__path__,
            "quiche.",
            onerror=_raise_import_error,
        )
    ),
)


def test_top_level_all():
    """Check quiche.__all__ is the sorted re-exports and top-level submodules."""
    assert quiche.__all__ == sorted(quiche._REEXPORTS.keys() | TOPLEVEL_MODULE_NAMES)


@pytest.mark.parametrize("module_name", ALL_MODULE_NAMES)
def test_module_exports(module_name: str):
    """Check all modules and their exports."""
    module = importlib.import_module(module_name)
    exports = getattr(module, "__all__", ())

    assert len(exports) == len(set(exports)), f"{module.__name__}: duplicate in __all__"

    for name in exports:
        getattr(module, name)


def test_unknown_attribute_raises():
    """Check unknown attributes raise correctly."""
    with pytest.raises(AttributeError, match="has no attribute 'nonexistent'"):
        _ = quiche.nonexistent
