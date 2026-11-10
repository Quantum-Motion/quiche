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

"""Interface for objects defining an implementation of a QPESpec."""

from typing import Protocol

from quiche.algorithms import (
    EstimationAlgorithm,
    SimulationAlgorithm,
    StateprepAlgorithm,
)


class QPESpecBackend[T](Protocol):
    """Backend capable of generating an implementation of a QPESpec."""

    def generate(
        self,
        stateprep: StateprepAlgorithm,
        estimation: EstimationAlgorithm,
        simulation: SimulationAlgorithm,
    ) -> T:
        """Generate the backend implementation of the given QPESpec."""
