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

"""Structures to define and dispatch phase estimation calculations."""

from pydantic.dataclasses import dataclass
from qualtran import CompositeBloq

from quiche.algorithms import (
    EstimationAlgorithm,
    SimulationAlgorithm,
    StateprepAlgorithm,
)
from quiche.dispatch.backends import QPESpecBackend
from quiche.simulation import QuESTRoutine


@dataclass()
class QPESpec:
    """QPE calculation specification for simulation and resource estimates."""

    stateprep: StateprepAlgorithm
    simulation: SimulationAlgorithm
    estimation: EstimationAlgorithm

    def generate[T](self, backend: QPESpecBackend[T]) -> T:
        """Generate the implementation of this QPESpec for the given backend."""
        return backend.generate(self)

    def to_qualtran(self) -> CompositeBloq:
        """
        Generate the Qualtran (resource-estimation backend) implementation of this spec.

        This is equivalent to calling ``spec.generate(quiche.QualtranBackend())``.

        Returns
        -------
        CompositeBloq
            A Qualtran representation of the QPE algorithm.

        Raises
        ------
        NotImplementedError
            If the specified algorithm is unsupported by the Qualtran backend.
        """  # ruff: ignore[docstring-extraneous-exception]
        from quiche.dispatch.backends.qualtran import QualtranBackend  # ruff: ignore[import-outside-top-level]

        return self.generate(QualtranBackend())

    def to_quest(self) -> QuESTRoutine:
        """
        Generate the QuEST (statevector simulation backend) implementation of this spec.

        This is equivalent to calling ``spec.generate(quiche.QuESTBackend())``.

        Returns
        -------
        QuESTRoutine
            A QuEST representation of the QPE algorithm.

        Raises
        ------
        NotImplementedError
            If the specified algorithm is unsupported by the QuEST backend.
        """  # ruff: ignore[docstring-extraneous-exception]
        from quiche.dispatch.backends.quest import QuESTBackend  # ruff: ignore[import-outside-top-level]

        return self.generate(QuESTBackend())
