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


"""Tests for the backend lowering of the phase estimation algorithms."""

import pytest
from qualtran import Bloq

from quiche import estimation as est
from quiche import simulation as sim
from quiche.core import PauliSum
from quiche.dispatch import Spec
from quiche.quest import QuestRoutine


@pytest.fixture
def textbook(h2: PauliSum) -> est.Textbook:
    """Textbook QPE with qubitisation, sized as the old default error budget."""
    return est.Textbook(
        simulation=sim.Qubitised(hamiltonian=h2, prepare_error=0.16 / 3),
        overlap=1,
        error=0.16 / 3,
    )


class TestEstimationLowering:
    """Tests for the estimation objects' `to_qualtran`/`to_quest`/`to_cudaq`."""

    def test_is_spec(self, textbook: est.Textbook):
        assert isinstance(textbook, Spec)

    def test_to_qualtran_data_register(self, textbook: est.Textbook):
        bloq = textbook.to_qualtran()

        assert isinstance(bloq, Bloq)
        assert bloq.signature[0].name == "data"
        assert bloq.signature[0].total_bits() == textbook.num_data

    def test_to_quest_routine(self, textbook: est.Textbook):
        routine = textbook.to_quest()

        assert isinstance(routine, QuestRoutine)
        assert len(routine.ops) == 1

    @pytest.mark.parametrize("method", [sim.Trotter, sim.QDRIFT])
    def test_to_quest_kitaev(self, h2: PauliSum, method: type[sim.SimulationMethod]):
        qpe = est.Kitaev(
            simulation=method(hamiltonian=h2, reps=2), overlap=1, num_rounds=4
        )
        assert len(qpe.to_quest().ops) == 1

    def test_unimplemented_combination_raises(self, h2: PauliSum):
        # Kitaev QPE + Qubitised simulation is not yet implemented in the QuEST
        # backend; other (algorithm, simulation) pairs raise the same way.
        qpe = est.Kitaev(
            simulation=sim.Qubitised(hamiltonian=h2, num_phase_ancillas=2),
            overlap=1,
            num_rounds=4,
        )
        with pytest.raises(NotImplementedError, match="Qubitisation"):
            qpe.to_quest()

    @pytest.mark.parametrize(
        "algorithm", [est.Iterative, est.Kitaev], ids=["iterative", "kitaev"]
    )
    def test_to_qualtran_not_implemented(
        self, h2: PauliSum, algorithm: type[est.Iterative | est.Kitaev]
    ):
        qpe = algorithm(
            simulation=sim.Trotter(hamiltonian=h2, reps=1), overlap=1, num_rounds=4
        )
        with pytest.raises(NotImplementedError, match=algorithm.__name__):
            qpe.to_qualtran()

    def test_to_cudaq_not_implemented_for_kitaev(self, h2: PauliSum):
        # Kitaev has no single-kernel decode (see `to_cudaq`'s docstring);
        # Naive and Iterative (Trotter/QDRIFT) are implemented and succeed.
        qpe = est.Kitaev(
            simulation=sim.Trotter(hamiltonian=h2, reps=1), overlap=1, num_rounds=4
        )
        with pytest.raises(NotImplementedError, match="Kitaev"):
            qpe.to_cudaq()
