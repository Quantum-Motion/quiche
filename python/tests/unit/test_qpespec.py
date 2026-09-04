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

"""Tests for qpespec module."""

import pytest
from qualtran import Bloq

from quiche.core import Errors, PauliSum, PhaseEstimation, Simulation
from quiche.dispatch import QPESpec, Spec
from quiche.quest import QuestRoutine


class TestQPESpec:
    """Tests for QPESpec."""

    def test_is_spec(self, h2: PauliSum, budget: Errors):
        spec = QPESpec(
            hamiltonian=h2,
            n_qubits=h2.n_qubits,
            algorithm=PhaseEstimation.Textbook,
            simulation=Simulation.Qubitised,
            error_budget=budget,
        )
        assert isinstance(spec, Spec)

    def test_to_qualtran_data_register(self, h2: PauliSum, budget: Errors):
        spec = QPESpec(
            hamiltonian=h2,
            n_qubits=h2.n_qubits,
            algorithm=PhaseEstimation.Textbook,
            simulation=Simulation.Qubitised,
            error_budget=budget,
        )
        bloq = spec.to_qualtran()

        assert isinstance(bloq, Bloq)
        assert bloq.signature[0].name == "data"
        assert bloq.signature[0].total_bits() == spec.num_data

    def test_to_quest_routine(self, h2: PauliSum, budget: Errors):
        spec = QPESpec(
            hamiltonian=h2,
            n_qubits=h2.n_qubits,
            algorithm=PhaseEstimation.Textbook,
            simulation=Simulation.Qubitised,
            error_budget=budget,
        )
        routine = spec.to_quest()

        assert isinstance(routine, QuestRoutine)
        assert len(routine.ops) == 1

    def test_unimplemented_combination_raises(self, h2: PauliSum, budget: Errors):
        # Kitaev QPE + Qubitised simulation is not yet implemented in the QuEST
        # backend; other (algorithm, simulation) pairs raise the same way.
        spec = QPESpec(
            hamiltonian=h2,
            n_qubits=h2.n_qubits,
            algorithm=PhaseEstimation.Kitaev,
            simulation=Simulation.Qubitised,
            error_budget=budget,
        )
        with pytest.raises(NotImplementedError, match="Qubitisation"):
            spec.to_quest()

    def test_to_cudaq_not_implemented_for_kitaev(self, h2: PauliSum, budget: Errors):
        # Kitaev has no single-kernel decode (see QPESpec.to_cudaq's docstring);
        # Naive and Iterative (Trotter/QDRIFT) are implemented and succeed.
        spec = QPESpec(
            hamiltonian=h2,
            n_qubits=h2.n_qubits,
            algorithm=PhaseEstimation.Kitaev,
            simulation=Simulation.Trotter,
            error_budget=budget,
        )
        with pytest.raises(NotImplementedError, match="Kitaev"):
            spec.to_cudaq()
