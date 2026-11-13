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

"""Data structures for defining quantum routines and algorithms."""

from .estimation import (
    EstimationAlgorithm,
    IterativeQPE,
    KitaevQPE,
    NaiveQPE,
    TextbookQPE,
)
from .simulation import (
    QDRIFT,
    AliasSampling,
    Qubitisation,
    RotationTree,
    SimulationAlgorithm,
    Trotter,
)
from .stateprep import BitstringStateprep, IdentityStateprep, StateprepAlgorithm

__all__ = [
    "QDRIFT",
    "AliasSampling",
    "BitstringStateprep",
    "EstimationAlgorithm",
    "IdentityStateprep",
    "IterativeQPE",
    "KitaevQPE",
    "NaiveQPE",
    "Qubitisation",
    "RotationTree",
    "SimulationAlgorithm",
    "StateprepAlgorithm",
    "TextbookQPE",
    "Trotter",
]
