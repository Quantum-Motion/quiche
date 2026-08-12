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

"""QUICHE - A library for QUantum Integrated CHEmistry."""

import importlib
from types import ModuleType
from typing import TYPE_CHECKING

__all__ = [
    "bindings",
    "chemistry",
    "core",
    "dispatch",
    "hamlib",
    "resources",
    "simulation",
]

if TYPE_CHECKING:
    from . import (
        bindings,
        chemistry,
        core,
        dispatch,
        hamlib,
        resources,
        simulation,
    )


def __getattr__(name: str) -> ModuleType:
    """Import submodules on first access."""
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)

    err_msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(err_msg)


def __dir__() -> list[str]:
    return list(__all__)
