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

"""Tests for the FCIDUMP module."""

import numpy as np
import pytest

from quiche.io import fcidump


@pytest.fixture
def h2_fcidump() -> str:
    """Minimal basis H2 FCIDUMP."""
    return """\
&FCI NORB= 2,NELEC= 2,MS2= 0,
 ORBSYM=1,1,
 ISYM= 1,
&END
    0.714139282456140   0 0 0 0
   -1.252705257753715   1 1 0 0
   -0.475697705661824   2 2 0 0
    0.674565096014104   1 1 1 1
    0.181266416982487   2 1 2 1
    0.663537594129644   2 2 1 1
    0.697467384335182   2 2 2 2
"""


def test_fcidump_parsing(h2_fcidump: str):
    """Test the header counts, core energy and one-body block of a full FCIDUMP."""
    expected_one_body = np.array(
        [
            [-1.252705257753715, 0.0],
            [0.0, -0.475697705661824],
        ]
    )

    hamiltonian = fcidump.parse(h2_fcidump)

    assert hamiltonian.electrons == 2
    assert hamiltonian.two_body.shape == (2, 2, 2, 2)
    assert hamiltonian.core_energy == pytest.approx(0.714139282456140)
    np.testing.assert_allclose(hamiltonian.one_body, expected_one_body)


@pytest.mark.parametrize(
    ("indices", "expected"),
    [
        ((0, 0, 0, 0), 0.674565096014104),
        ((1, 0, 1, 0), 0.181266416982487),
        ((0, 1, 0, 1), 0.181266416982487),
        ((1, 1, 0, 0), 0.663537594129644),
        ((0, 0, 1, 1), 0.663537594129644),
        ((1, 1, 1, 1), 0.697467384335182),
        ((0, 0, 0, 1), 0.0),
    ],
)
def test_two_body_integrals(h2_fcidump: str, indices: tuple[int, ...], expected: float):
    """Test the two-body tensor is filled with the integrals and their permutations."""
    hamiltonian = fcidump.parse(h2_fcidump)
    actual = hamiltonian.two_body[indices]
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    "text",
    [
        "&FCI NORB=1,NELEC=2,\n/\n 1.0 1 1 1 1\n",
        "&FCI NORB=1,NELEC=2, &END\n 1.0 1 1 1 1\n",
        "&FCI NORB=1,NELEC=2,\n&END\n\n 1.0 1 1 1 1\n\n",
        "&FCI NORB=4,\n ORBSYM=1,1,\n 1,1,\n NELEC=2,\n&END\n 1.0 1 1 1 1\n",
    ],
)
def test_header_variants(text: str):
    """Test accepted namelist formats all parse."""
    hamiltonian = fcidump.parse(text)
    assert hamiltonian.electrons == 2
    assert hamiltonian.two_body[0, 0, 0, 0] == pytest.approx(1.0)


@pytest.mark.parametrize("flag", ["IUHF=1", "UHF=.TRUE.", "UHF=1", "IUHF=T"])
def test_unrestricted(flag: str):
    """Test spin-unrestricted files are refused."""
    text = f"&FCI NORB=1,NELEC=2,{flag}\n&END\n"

    with pytest.raises(NotImplementedError, match="Spin-unrestricted"):
        fcidump.parse(text)


@pytest.mark.parametrize(
    ("text", "error_msg"),
    [
        ("", "must start with"),
        ("&FCIDUMP NORB=1,NELEC=2,\n&END\n", "must start with"),
        ("&FCI NORB=1,NELEC=2,\n 1.0 1 1 1 1\n", "not terminated"),
        ("&FCI NELEC=2,\n&END\n", "integer NORB"),
        ("&FCI NORB=two,NELEC=2,\n&END\n", "integer NORB"),
        ("&FCI NORB=2,\n&END\n", "integer NELEC"),
        ("&FCI NORB=0,NELEC=2,\n&END\n", "positive NORB"),
    ],
)
def test_invalid_header(text: str, error_msg: str):
    """Test a header that is absent, unterminated or incomplete is refused."""
    with pytest.raises(ValueError, match=error_msg):
        fcidump.parse(text)


@pytest.mark.parametrize(
    "integral",
    [
        "0.5 1 1 1",
        "0.5 1 1 1 1 1",
        "abc 1 1 1 1",
        "0.5 1 1 1 1.5",
        "0.5 1 1 1 3",
        "0.5 1 1 1 -1",
    ],
)
def test_malformed_integral(integral: str):
    """Test a malformed integral is refused."""
    text = f"&FCI NORB=2,NELEC=2,\n&END\n {integral}\n"

    with pytest.raises(ValueError, match="Malformed"):
        fcidump.parse(text)


@pytest.mark.parametrize("integral", ["0.5 1 0 1 0", "0.5 0 1 0 0"])
def test_unrecognised_integral(integral: str):
    """Test an integral with an unrecognised index pattern is refused."""
    text = f"&FCI NORB=2,NELEC=2,\n&END\n {integral}\n"

    with pytest.raises(ValueError, match="Unrecognised"):
        fcidump.parse(text)
