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

from . import bindings, chemistry, core, cudaq, dispatch, hamlib, qualtran, quest
from .chemistry import HartreeFockState
from .core import (
    Errors,
    Mapping,
    Pauli,
    PauliSum,
    PauliWord,
    PhaseEstimation,
    Seed,
    Simulation,
)
from .cudaq import CudaqKernel
from .dispatch import HartreeFockSpec, QPESpec, Spec
from .quest import QuestRoutine

__all__ = [
    "CudaqKernel",
    "Errors",
    "HartreeFockSpec",
    "HartreeFockState",
    "Mapping",
    "Pauli",
    "PauliSum",
    "PauliWord",
    "PhaseEstimation",
    "QPESpec",
    "QuestRoutine",
    "Seed",
    "Simulation",
    "Spec",
    "bindings",
    "chemistry",
    "core",
    "cudaq",
    "dispatch",
    "hamlib",
    "qualtran",
    "quest",
]
