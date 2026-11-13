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

"""Tests for build, install and running of compiled C++ bindings."""

# ruff: noqa: PLC0415, F401

from __future__ import annotations

import re
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from quiche.bindings.quest_bindings import QuESTEnv


@pytest.fixture(scope="session")
def quest_env() -> QuESTEnv:
    """Initialise a QuESTEnv to reuse for QuEST tests."""
    from quiche.bindings.quest_bindings import QuESTEnv

    return QuESTEnv()


def test_bindings_import():
    """Test that the bindings module and submodules import cleanly."""
    import quiche.bindings
    import quiche.bindings.quest_bindings
    import quiche.bindings.quiche_bindings


def test_submodule_attributes():
    """Test submodules actually include some expected attributes."""
    import quiche.bindings.quest_bindings as quest
    import quiche.bindings.quiche_bindings as quiche

    # Check main classes from QuEST API
    for attr in (
        "QuESTEnv",
        "Qureg",
        "PauliStr",
        "PauliStrSum",
        "KrausMap",
        "SuperOp",
        "CompMatr1",
        "CompMatr2",
        "CompMatr",
        "DiagMatr1",
        "DiagMatr2",
        "DiagMatr",
        "FullStateDiagMatr",
    ):
        assert hasattr(quest, attr), attr

    # Check a few functions from the QUICHE interface
    for attr in (
        "getHartreeFockStateJW",
        "initClassicalState",
        "getPhaseTextbookQubitised",
    ):
        assert hasattr(quiche, attr), attr


def test_quiche_call():
    """Test self-contained QUICHE call."""
    from quiche.bindings.quiche_bindings import getHartreeFockStateJW

    assert getHartreeFockStateJW(2) == 0b11


def test_quiche_quest_call(quest_env: QuESTEnv):  # noqa: ARG001
    """Test combined QuEST and QUICHE call."""
    from quiche.bindings.quest_bindings import PauliStr, PauliStrSum, Qureg
    from quiche.bindings.quiche_bindings import initClassicalState

    qureg = Qureg(2)
    initClassicalState(qureg, [1, 0])

    pauli_sum = PauliStrSum([PauliStr("XZ"), PauliStr("II")], [1 + 0j, 0.5 + 0.5j])
    pauli_sum.report()


IMPORTS = "from quiche.bindings.quest_bindings import QuESTEnv, QuESTError, Qureg\n"

INIT_AFTER_FAILURE = """\
try:
    QuESTEnv(0, 1, 0)
except QuESTError:
    pass
QuESTEnv()
"""

INIT_AFTER_FINALISE = """\
with QuESTEnv():
    pass
QuESTEnv()
"""

ERROR_AFTER_INIT = """\
with QuESTEnv():
    Qureg(1).initRandomMixedState(5)
"""


@pytest.mark.parametrize(
    ("code", "expected_error"),
    [
        pytest.param(
            "Qureg(1)",
            r"QuESTError.*not initialised",
            id="not-init",
        ),
        pytest.param(
            INIT_AFTER_FAILURE,
            r"QuESTError.*previous initialisation failed",
            id="init-after-failure",
        ),
        pytest.param(
            INIT_AFTER_FINALISE,
            r"QuESTError.*finalise",
            id="init-after-finalise",
        ),
        pytest.param(
            ERROR_AFTER_INIT,
            r"QuESTError.*density matrix",
            id="error-after-init",
        ),
    ],
)
def test_quest_error_handler(code: str, expected_error: str):
    """Test QuEST errors raise rather than exit (in a subprocess)."""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", IMPORTS + code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert re.search(expected_error, result.stderr), f"{result.stdout}\n{result.stderr}"
