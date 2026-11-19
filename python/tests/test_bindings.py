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


INIT_AFTER_FAILURE = """\
try:
    QuESTEnv(0, 2, 0)
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
            r"QuESTError.*initialised again",
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
    imports = "from quiche.bindings.quest_bindings import QuESTEnv, QuESTError, Qureg\n"
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", imports + code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert re.search(expected_error, result.stderr), f"{result.stdout}\n{result.stderr}"


def test_resource_use_after_destroy(quest_env: QuESTEnv):  # noqa: ARG001
    """Test using a destroyed resource raises."""
    from quiche.bindings.quest_bindings import QuESTError, Qureg

    qureg = Qureg(1)
    qureg.destroy()
    qureg.destroy()  # a second destroy should be a no-op

    with pytest.raises(QuESTError, match="already been destroyed"):
        qureg.initZeroState()


def test_destroyed_resource_as_argument(quest_env: QuESTEnv):  # noqa: ARG001
    """Test a destroyed resource raises when passed as an argument."""
    from quiche.bindings.quest_bindings import QuESTError, Qureg
    from quiche.bindings.quiche_bindings import initClassicalState

    qureg = Qureg(1)

    other = Qureg(1)
    other.destroy()

    with pytest.raises(QuESTError, match="already been destroyed"):
        qureg.calcFidelity(other)

    with pytest.raises(QuESTError, match="already been destroyed"):
        initClassicalState(other, [0])


def test_none_in_resource_list(quest_env: QuESTEnv):  # noqa: ARG001
    """Test None inside a list of resources raises TypeError."""
    from quiche.bindings.quest_bindings import Qureg

    qureg = Qureg(1)
    other = Qureg(1)

    with pytest.raises(TypeError, match="None where a QuEST resource"):
        qureg.setQuregToWeightedSum([1, 0], [other, None])  # ty: ignore[invalid-argument-type]


def test_destroyed_resource_in_list(quest_env: QuESTEnv):  # noqa: ARG001
    """Test a destroyed resource inside a list raises when passed as an argument."""
    from quiche.bindings.quest_bindings import QuESTError, Qureg

    qureg = Qureg(1)
    other = Qureg(1)

    other.destroy()

    with pytest.raises(QuESTError, match="already been destroyed"):
        qureg.setQuregToWeightedSum([1], [other])


RESOURCE_AFTER_EXIT = """\
from quiche.bindings.quest_bindings import (
    CompMatr,
    DiagMatr,
    FullStateDiagMatr,
    KrausMap,
    PauliStrSum,
    QuESTEnv,
    Qureg,
    SuperOp,
)

with QuESTEnv():
    resource = {create}
resource.report()
"""


@pytest.mark.parametrize(
    "create",
    [
        pytest.param("Qureg(1)", id="qureg"),
        pytest.param('PauliStrSum("1 X")', id="pauli-str-sum"),
        pytest.param("KrausMap(1, 1)", id="kraus-map"),
        pytest.param("SuperOp(1)", id="super-op"),
        pytest.param("CompMatr(1)", id="comp-matr"),
        pytest.param("DiagMatr(1)", id="diag-matr"),
        pytest.param("FullStateDiagMatr(1)", id="full-state-diag-matr"),
    ],
)
def test_env_exit_destroys_live_resources(create: str):
    """
    Test resources alive at QuESTEnv exit are destroyed (in a subprocess).

    Relies on resource liveness being checked before the environment.
    """
    expected_error = r"QuESTError.*already been destroyed"
    code = RESOURCE_AFTER_EXIT.format(create=create)
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert re.search(expected_error, result.stderr), f"{result.stdout}\n{result.stderr}"
