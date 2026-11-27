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

"""Tests for QuESTRoutine, QuregLayout and associated functionality."""

import pytest

from quiche.bindings.quest_bindings import QuESTEnv, QuESTError
from quiche.simulation import QuESTRoutine, QuregLayout


class TestQuregLayout:
    """Test QuregLayout class."""

    def test_contiguous_registers(self):
        layout = QuregLayout(num_data=4, num_qpe=3, num_simulation=2)
        assert layout.data == range(4)
        assert layout.qpe == range(4, 7)
        assert layout.simulation == range(7, 9)
        assert layout.num_qubits == 9

    def test_optional_registers(self):
        layout = QuregLayout(num_data=4, num_qpe=3)
        assert layout.simulation == range(7, 7)
        assert layout.num_qubits == 7

    @pytest.mark.parametrize(
        ("num_data", "num_qpe", "num_simulation", "error_msg"),
        [
            (0, 1, 0, "greater than 0"),
            (1, 0, 0, "greater than 0"),
            (1, 1, -1, "greater than or equal to 0"),
        ],
    )
    def test_invalid(
        self,
        num_data: int,
        num_qpe: int,
        num_simulation: int,
        error_msg: str,
    ):
        with pytest.raises(ValueError, match=error_msg):
            QuregLayout(
                num_data=num_data,
                num_qpe=num_qpe,
                num_simulation=num_simulation,
            )


class TestQuESTRoutine:
    """Test QuESTRoutine class."""

    def test_run(self, quest_env: QuESTEnv):  # ruff: ignore[unused-method-argument]
        routine = QuESTRoutine(
            ops=(
                lambda qureg: qureg.applyPauliX(2),
                lambda qureg: qureg.calcProbOfQubitOutcome(2, 1),
            ),
            layout=QuregLayout(num_data=2, num_qpe=1),
        )
        assert routine.run() == pytest.approx(1.0)

        # Check that each run starts on a fresh Qureg
        assert routine.run() == pytest.approx(1.0)

    def test_cleanup(self, quest_env: QuESTEnv):  # ruff: ignore[unused-method-argument]
        captured = []
        routine = QuESTRoutine(
            ops=(captured.append,),
            layout=QuregLayout(num_data=1, num_qpe=1),
        )

        routine.run()
        captured_qureg = captured[0]
        with pytest.raises(QuESTError, match="already been destroyed"):
            captured_qureg.applyPauliX(0)

    def test_cleanup_on_failure(self, quest_env: QuESTEnv):  # ruff: ignore[unused-method-argument]
        captured = []
        routine = QuESTRoutine(
            ops=(
                captured.append,
                lambda qureg: qureg.applyHadamard(3),
            ),
            layout=QuregLayout(num_data=1, num_qpe=1),
        )

        # Out of bounds op should raise
        with pytest.raises(QuESTError, match="Invalid target qubit"):
            routine.run()

        captured_qureg = captured[0]
        with pytest.raises(QuESTError, match="already been destroyed"):
            captured_qureg.applyPauliX(0)
