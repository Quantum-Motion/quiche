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

"""Tests for dispatch module."""

import pytest
from qualtran import CompositeBloq

from quiche.chemistry import HartreeFockState, Mapping
from quiche.core import (
    ElectronicHamiltonian,
    Errors,
    PauliSum,
    PhaseEstimation,
    Simulation,
)
from quiche.dispatch import QPESpec
from quiche.dispatch.backends.qualtran import QualtranBackend
from quiche.dispatch.backends.quest import QuESTBackend
from quiche.simulation import QuESTRoutine


@pytest.fixture
def spec(h2: PauliSum, budget: Errors) -> QPESpec:
    """Construct an example H2 QPESpec fixture."""
    ham = ElectronicHamiltonian(
        paulis=h2,
        mapping=Mapping.JordanWigner,
        num_electrons=2,
    )

    hf = HartreeFockState.closed_shell(
        num_electrons=ham.num_electrons,
        num_spin_orbitals=ham.paulis.num_qubits,
    )

    return QPESpec(
        hamiltonian=ham,
        state_prep=hf,
        algorithm=PhaseEstimation.Textbook,
        simulation=Simulation.Trotter,
        error_budget=budget,
        seed=3,
    )


class TestQualtranBackend:
    """Test QualtranBackend class."""

    def test_generate(self, spec: QPESpec):
        bloq = QualtranBackend().generate(spec)
        assert isinstance(bloq, CompositeBloq)

    def test_to_qualtran(self, spec: QPESpec):
        bloq = spec.to_qualtran()
        assert isinstance(bloq, CompositeBloq)


class TestQuESTBackend:
    """Test QuESTBackend class."""

    def test_generate(self, spec: QPESpec):
        routine = QuESTBackend().generate(spec)
        assert isinstance(routine, QuESTRoutine)

    def test_to_quest(self, spec: QPESpec):
        routine = spec.to_quest()
        assert isinstance(routine, QuESTRoutine)
