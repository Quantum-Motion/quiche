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

"""Structures for specifying calculation error sources."""

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class Errors:
    """Error budget tracker for approximate quantum methods."""

    # The estimation error is the error in the energy incured by QPE algorithms.
    # TODO(Annina): Decide whether estimation error should be input in Ha. In that case
    # we will have to convert it to a dimensionless error in the phase.
    estimation: float

    # The simulation error is the error incured during one instance of the Hamiltonian
    # simulation. If Hamiltonian simulation is a Trotter method, then the simulation
    # error is related to the Trotter error. If the Hamiltonian simulation is a
    # qubitisation, then the simulation error is related to the Prepare operation in
    # the qubitisation walk operator.
    # TODO(Annina): If the simulation bloq is used multiple times (e.g. in QPE), the
    # simulation error should account for the total error, not just the error of a
    # single simulation as it is currently.
    simulation: float

    # The rotation error is the error made when synthesising a rotation from some
    # fundamental gate set (usually probably from T and Clifford gates). This should
    # provide the total error for all rotations necessary in the circuit.
    rotations: float

    # The state preparation error is the error incured during the preparation of the
    # initial state of the system to be studied for a QPE algorithm.
    state_prep: float

    # The overlap error quantifies the magnitude of the overlap of the loaded initial
    # state with the true target state.
    overlap: float
