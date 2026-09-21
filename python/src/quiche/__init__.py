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
from typing import TYPE_CHECKING

__all__ = [
    "ElectronicHamiltonian",
    "Errors",
    "Mapping",
    "Pauli",
    "PauliSum",
    "PauliWord",
    "PhaseEstimation",
    "QPESpec",
    "SecondQuantisedHamiltonian",
    "Simulation",
    "bindings",
    "chemistry",
    "core",
    "dispatch",
    "io",
    "resources",
    "simulation",
]

_REEXPORTS = {
    "ElectronicHamiltonian": ".core.electronic",
    "Errors": ".core.errors",
    "Mapping": ".core.algorithms",
    "Pauli": ".core.paulis",
    "PauliSum": ".core.paulis",
    "PauliWord": ".core.paulis",
    "PhaseEstimation": ".core.algorithms",
    "QPESpec": ".dispatch.qpespec",
    "SecondQuantisedHamiltonian": ".core.electronic",
    "Simulation": ".core.algorithms",
}

if TYPE_CHECKING:
    from . import (
        bindings,
        chemistry,
        core,
        dispatch,
        io,
        resources,
        simulation,
    )
    from .core.algorithms import Mapping, PhaseEstimation, Simulation
    from .core.electronic import ElectronicHamiltonian, SecondQuantisedHamiltonian
    from .core.errors import Errors
    from .core.paulis import Pauli, PauliSum, PauliWord
    from .dispatch import QPESpec


def __getattr__(name: str) -> object:
    """Import submodules and re-exported names on first access."""
    if (module_path := _REEXPORTS.get(name)) is not None:
        module = importlib.import_module(module_path, __name__)
        return getattr(module, name)

    if name in __all__:
        return importlib.import_module(f".{name}", __name__)

    err_msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(err_msg)


def __dir__() -> list[str]:
    return list(__all__)
