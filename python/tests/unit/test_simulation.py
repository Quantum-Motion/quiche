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

"""Tests for simulation module."""

from math import ceil, log2

import cirq
import numpy as np
import pytest
from qualtran import QAny, Side
from qualtran.bloqs.chemistry.trotter.trotterized_unitary import (
    TrotterizedUnitary,
)
from qualtran.resource_counting.generalizers import ignore_split_join
from qualtran.testing import (
    assert_equivalent_bloq_counts,
)
from scipy.linalg import expm

from quiche.core import Errors, Pauli, PauliSum, PauliWord
from quiche.resources import logical_qubit_resources
from quiche.resources.bloqs import (
    QDRIFT,
    LCUBlockEncodingWrapper,
    PauliWordRotation,
    SelectPauliLCUWrapper,
    Trotterisation,
)


def _flatten_trotterizedunitary(bloq_counts: dict) -> dict:
    """Unpack the TrotterizedUnitary entry of a bloq count dictionary."""
    flat_bloq_counts = {}
    for key, count in bloq_counts.items():
        if isinstance(key, TrotterizedUnitary):
            tmp = key.bloq_counts()
            tmp.update((k, tmp[k] * count) for k in tmp)
            flat_bloq_counts |= tmp
        else:
            flat_bloq_counts[key] = count
    return flat_bloq_counts


class TestSelectPauliLCUWrapper:
    """Tests for SelectPauliLCUWrapper."""

    @pytest.fixture
    def select(self, h2: PauliSum, budget: Errors) -> SelectPauliLCUWrapper:
        select_nqubits = ceil(log2(h2.n_terms))
        phase_bitsize = max(ceil(log2(2.0 * select_nqubits / budget.state_prep)), 2)
        terms = (term.to_cirq(h2.n_qubits) for term in h2.terms)

        return SelectPauliLCUWrapper(
            selection_bitsize=select_nqubits + phase_bitsize,
            target_bitsize=h2.n_qubits,
            select_unitaries=terms,
        )

    @pytest.mark.parametrize("controlled", [False, True])
    def test_bloq_counts(self, select: SelectPauliLCUWrapper, controlled: bool):
        bloq = select.controlled() if controlled else select
        assert_equivalent_bloq_counts(bloq, generalizer=[ignore_split_join])

    @pytest.mark.parametrize("controlled", [False, True])
    def test_qubit_counts(self, select: SelectPauliLCUWrapper, controlled: bool):
        bloq = select.controlled() if controlled else select
        manual_counts = logical_qubit_resources(bloq)
        decomp_counts = logical_qubit_resources(bloq.decompose_bloq())
        assert manual_counts == decomp_counts


class TestLCUBlockEncodingWrapper:
    """Tests for LCUBlockEncodingWrapper."""

    @pytest.fixture(autouse=True)
    def blockencoding(self, h2: PauliSum, budget: Errors) -> LCUBlockEncodingWrapper:
        select_nqubits = ceil(log2(h2.n_terms))
        phase_bitsize = max(ceil(log2(2.0 * select_nqubits / budget.state_prep)), 2)
        return LCUBlockEncodingWrapper.from_hamiltonian(h2, phase_bitsize)

    def test_signature(
        self, blockencoding: LCUBlockEncodingWrapper, h2: PauliSum, budget: Errors
    ):
        """Check bloq signature."""
        select_nqubits = ceil(log2(h2.n_terms))
        phase_bitsize = max(ceil(log2(2.0 * select_nqubits / budget.state_prep)), 2)
        sig = blockencoding.signature
        assert len(sig) == 2

        reg = sig[0]
        assert reg.name == "selection"
        assert reg.dtype == QAny(select_nqubits + phase_bitsize)
        assert reg.side == Side.THRU

        reg = sig[1]
        assert reg.name == "target"
        assert reg.dtype == QAny(h2.n_qubits)
        assert reg.side == Side.THRU

    def test_small_phase_bitsize(self, h2: PauliSum):
        phase_bitsize = 1
        error_msg = "Choose phase_bitsize at least 2"
        with pytest.raises(ValueError, match=error_msg):
            LCUBlockEncodingWrapper.from_hamiltonian(h2, phase_bitsize)

    def test_zerocoefficients(
        self, blockencoding: LCUBlockEncodingWrapper, h2: PauliSum
    ):
        prep_coeffs = blockencoding.prepare.stateprep.state_coefficients
        # The block encoding pads the coefficients to a power of two. All coefficients
        # beyond the ones needed for the Hamiltonian should be zero and have no effect.
        # There are a total of self.h.n_terms + 1 non-zero coefficients because the
        # identity coefficient is non-zero. Test that these are indeed non-zero and all
        # others are zero.
        np.testing.assert_equal(prep_coeffs[: h2.n_terms + 1] != 0, True)
        np.testing.assert_allclose(prep_coeffs[h2.n_terms + 1 :], 0.0)

    def test_selectunitaries(
        self, blockencoding: LCUBlockEncodingWrapper, h2: PauliSum
    ):
        # Checks that the right unitaries and coefficients will be applied in the select
        # unitary.
        true_unitaries = blockencoding.select.select_unitaries
        true_prep_coeffs = np.array(
            blockencoding.prepare.stateprep.state_coefficients, dtype=complex
        )
        # Truncate to the non-zero terms. All truncated coefficients are zero, which is
        # tested separately in test_zerocoefficients. Truncate after self.h.n_terms+1 in
        # order to count the identity contribution.
        true_unitaries = true_unitaries[: h2.n_terms + 1]
        true_prep_coeffs = true_prep_coeffs[: h2.n_terms + 1]

        # Set the target unitaries and coefficients
        target_unitaries = [u.to_cirq(h2.n_qubits) for u in h2.terms] + [
            cirq.DensePauliString.eye(h2.n_qubits)
        ]
        target_coefficients = [*h2.coefficients, h2.identity_coefficient]
        # Add the identity coefficient to the 1-norm
        lam = h2.lam + abs(h2.identity_coefficient)

        # Assert the unitaries
        for ii in range(len(target_unitaries)):
            err_msg = (
                f"Unitaries at index {ii} do not agree: "
                f"{true_unitaries[ii]} vs {target_unitaries[ii]}."
            )
            assert true_unitaries[ii] == target_unitaries[ii], err_msg

        # Assert the coefficients
        np.testing.assert_allclose(lam * true_prep_coeffs**2, target_coefficients)

    @pytest.mark.parametrize("controlled", [False, True])
    def test_bloq_counts(
        self, blockencoding: LCUBlockEncodingWrapper, controlled: bool
    ):
        bloq = blockencoding.controlled() if controlled else blockencoding
        assert_equivalent_bloq_counts(bloq, generalizer=[ignore_split_join])

    @pytest.mark.parametrize("controlled", [False, True])
    def test_qubit_counts(
        self, blockencoding: LCUBlockEncodingWrapper, controlled: bool
    ):
        bloq = blockencoding.controlled() if controlled else blockencoding
        manual_counts = logical_qubit_resources(bloq)
        decomp_counts = logical_qubit_resources(bloq.decompose_bloq())
        assert manual_counts == decomp_counts


class TestPauliWordRotation:
    @pytest.fixture
    def rotation(self) -> PauliWordRotation:
        qubits = (0, 2, 5)
        n_qubits = 7
        phase = 0.4
        word = PauliWord(terms=(Pauli.Y, Pauli.Z, Pauli.X), qubits=qubits)
        return PauliWordRotation(word, phase, n_qubits)

    def test_signature(self, rotation: PauliWordRotation):
        sig = rotation.signature
        assert len(sig) == 1

        reg = sig[0]
        assert reg.name == "system"
        assert reg.dtype == QAny(rotation.n_qubits)
        assert reg.side == Side.THRU

    def test_wrong_n_qubits(self):
        word = PauliWord(terms=(Pauli.X, Pauli.Y, Pauli.X), qubits=(0, 2, 7))
        rot = PauliWordRotation(word, 0.4, 7)

        with pytest.raises(ValueError, match="Target qubit 7 is out of range"):
            rot.decompose_bloq()

    def test_decomposition(self, rotation: PauliWordRotation):
        circ = rotation.decompose_bloq().to_cirq_circuit(
            cirq_quregs={"system": cirq.LineQubit.range(rotation.n_qubits)},
        )
        u = circ.unitary()

        h = rotation.word._to_matrix(ignore_idle_qubits=True)
        u_target = expm(-1j * h * rotation.angle / 2)

        np.testing.assert_allclose(u, u_target)

    @pytest.mark.parametrize("controlled", [False, True])
    def test_bloq_counts(self, rotation: PauliWordRotation, controlled: bool):
        bloq = rotation.controlled() if controlled else rotation
        assert_equivalent_bloq_counts(bloq, generalizer=[ignore_split_join])

    @pytest.mark.parametrize("controlled", [False, True])
    def test_qubit_counts(self, rotation: PauliWordRotation, controlled: bool):
        bloq = rotation.controlled() if controlled else rotation
        manual_counts = logical_qubit_resources(bloq)
        decomp_counts = logical_qubit_resources(bloq.decompose_bloq())
        assert manual_counts == decomp_counts


class TestQDRIFT:
    @pytest.fixture
    def qdrift(self, h2: PauliSum) -> QDRIFT:
        return QDRIFT(h2, t=5, n_terms=20, seed=1024)

    def test_invalid_negative_nsteps(self, h2: PauliSum):
        with pytest.raises(ValueError, match="Choose positive n_terms"):
            QDRIFT(h2, t=5, n_terms=-10)

    def test_invalid_negative_time(self, h2: PauliSum):
        with pytest.raises(ValueError, match="Choose positive evolution time"):
            QDRIFT(h2, t=-5, n_terms=4)

    @pytest.mark.parametrize("controlled", [False, True])
    def test_bloq_counts(self, qdrift: QDRIFT, controlled: bool):
        bloq = qdrift.controlled() if controlled else qdrift
        manual_counts = bloq.bloq_counts(generalizer=[ignore_split_join])
        decomp_counts = bloq.decompose_bloq().bloq_counts(
            generalizer=[ignore_split_join]
        )
        decomp_unpack = _flatten_trotterizedunitary(decomp_counts)
        assert manual_counts == decomp_unpack

    @pytest.mark.parametrize("controlled", [False, True])
    def test_qubit_counts(self, qdrift: QDRIFT, controlled: bool):
        bloq = qdrift.controlled() if controlled else qdrift
        manual_counts = logical_qubit_resources(bloq)
        decomp_counts = logical_qubit_resources(bloq.decompose_bloq())
        assert manual_counts == decomp_counts


class TestTrotterisation:
    @pytest.fixture
    def trotter(self, h2: PauliSum, request) -> Trotterisation:
        return Trotterisation(h2, t=5, n_steps=10, order=request.param)

    @pytest.mark.parametrize("n_steps", [-10, 0])
    def test_invalid_nsteps(self, h2: PauliSum, n_steps: int):
        with pytest.raises(ValueError, match="Choose positive n_steps"):
            Trotterisation(h2, t=1, n_steps=n_steps, order=2)

    @pytest.mark.parametrize(
        ("order", "err_msg"),
        [
            (-2, "positive Trotter order"),
            (0, "positive Trotter order"),
            (3, "order must be even"),
        ],
    )
    def test_invalid_trotter_order(self, h2: PauliSum, order: int, err_msg: str):
        with pytest.raises(ValueError, match=err_msg):
            Trotterisation(h2, t=0.2, n_steps=23, order=order)

    def test_coeffs_indices_lie_trotter(self):
        word1 = PauliWord(terms=(Pauli.X, Pauli.Y, Pauli.Z), qubits=(0, 2, 3))
        h = PauliSum(
            coefficients=(5.0, 19.0, 10.0, 15.0),
            terms=(word1, word1, word1, word1),
            identity_coefficient=0,
        )

        trotterisation = Trotterisation(h, t=42.0, n_steps=10, order=1)
        coeffs, indices = trotterisation.get_coeffs_indices()

        np.testing.assert_equal(indices, [0, 1, 2, 3])
        np.testing.assert_allclose(coeffs, (1.0, 1.0, 1.0, 1.0))

    def test_coeffs_indices_suzuki_2(self):
        word1 = PauliWord(terms=(Pauli.X, Pauli.Y, Pauli.Z), qubits=(0, 2, 3))
        h = PauliSum(
            coefficients=(5.0, 19.0, 10.0, 15.0),
            terms=(word1, word1, word1, word1),
            identity_coefficient=0,
        )

        trotterisation = Trotterisation(h, t=10, n_steps=100, order=2)
        coeffs, indices = trotterisation.get_coeffs_indices()

        np.testing.assert_equal(indices, [0, 1, 2, 3, 2, 1, 0])
        np.testing.assert_allclose(coeffs, [0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5])

    def test_coeffs_indices_suzuki_4(self):
        word1 = PauliWord(terms=(Pauli.X, Pauli.Y, Pauli.Z), qubits=(0, 2, 3))
        h = PauliSum(
            coefficients=(5.0, 19.0, 10.0),
            terms=(word1, word1, word1),
            identity_coefficient=0,
        )

        order = 4
        trotterisation = Trotterisation(h, t=5.0, n_steps=15, order=order)
        coeffs_actual, indices_actual = trotterisation.get_coeffs_indices()

        indices_expect = [0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0]
        uk = 1.0 / (4 - 4 ** (1 / (order - 1)))
        coeffs_expect = [
            0.5 * uk,
            0.5 * uk,
            uk,
            0.5 * uk,
            uk,
            0.5 * uk,
            uk,
            0.5 * uk,
            0.5 * (1 - 3 * uk),
            0.5 * (1 - 4 * uk),
            (1 - 4 * uk),
            0.5 * (1 - 4 * uk),
            0.5 * (1 - 3 * uk),
            0.5 * uk,
            uk,
            0.5 * uk,
            uk,
            0.5 * uk,
            uk,
            0.5 * uk,
            0.5 * uk,
        ]

        np.testing.assert_equal(indices_actual, indices_expect)
        np.testing.assert_allclose(coeffs_actual, coeffs_expect)

    def test_lt_singlequbit(self):
        word1 = PauliWord(terms=(Pauli.Z,), qubits=(0,))
        word2 = PauliWord(terms=(Pauli.Y,), qubits=(0,))
        word3 = PauliWord(terms=(Pauli.X,), qubits=(0,))

        coeffs = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        terms = (word1, word2, word3)
        h = PauliSum(coefficients=coeffs, terms=terms, identity_coefficient=0)

        trotterisation = Trotterisation(h, t=10, n_steps=20, order=1)

        h1 = coeffs[0] * word1._to_matrix(ignore_idle_qubits=False)
        h2 = coeffs[1] * word2._to_matrix(ignore_idle_qubits=False)
        h3 = coeffs[2] * word3._to_matrix(ignore_idle_qubits=False)

        u_target = np.identity(2**h.n_qubits)
        for _ in range(trotterisation.n_steps):
            u_target = np.dot(u_target, expm(-1j * h3 * trotterisation.dt))
            u_target = np.dot(u_target, expm(-1j * h2 * trotterisation.dt))
            u_target = np.dot(u_target, expm(-1j * h1 * trotterisation.dt))

        circ = trotterisation.decompose_bloq().to_cirq_circuit(
            cirq_quregs={"simulation": cirq.LineQubit.range(h.n_qubits)},
        )
        u = circ.unitary()
        np.testing.assert_allclose(u, u_target)

    def test_lt_fourqubits(self):
        word1 = PauliWord(terms=(Pauli.X, Pauli.Y, Pauli.Z), qubits=(0, 2, 3))
        word2 = PauliWord(terms=(Pauli.Y, Pauli.Z, Pauli.X), qubits=(1, 2, 3))
        h = PauliSum(
            coefficients=(5.0, 10.0), terms=(word1, word2), identity_coefficient=0
        )
        n_qubits = h.n_qubits
        trotterisation = Trotterisation(h, t=7.5, n_steps=10, order=1)

        h1 = h.coefficients[0] * word1._to_matrix(
            length=n_qubits, ignore_idle_qubits=False
        )
        h2 = h.coefficients[1] * word2._to_matrix(
            length=n_qubits, ignore_idle_qubits=False
        )

        u_target = np.identity(2**n_qubits)
        for _ in range(trotterisation.n_steps):
            u_target = np.dot(u_target, expm(-1j * h2 * trotterisation.dt))
            u_target = np.dot(u_target, expm(-1j * h1 * trotterisation.dt))

        circ = trotterisation.decompose_bloq().to_cirq_circuit(
            cirq_quregs={"simulation": cirq.LineQubit.range(n_qubits)},
        )
        u = circ.unitary()

        np.testing.assert_allclose(u, u_target, atol=1e-15)

    def test_st_singlequbit(self):
        word1 = PauliWord(terms=(Pauli.Z,), qubits=(0,))
        word2 = PauliWord(terms=(Pauli.Y,), qubits=(0,))
        word3 = PauliWord(terms=(Pauli.X,), qubits=(0,))

        coeffs = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        terms = (word1, word2, word3)
        h = PauliSum(coefficients=coeffs, terms=terms, identity_coefficient=0)
        n_qubits = h.n_qubits

        trotterisation = Trotterisation(h, t=10, n_steps=20, order=2)

        h1 = coeffs[0] * word1._to_matrix(ignore_idle_qubits=False)
        h2 = coeffs[1] * word2._to_matrix(ignore_idle_qubits=False)
        h3 = coeffs[2] * word3._to_matrix(ignore_idle_qubits=False)

        u_target = np.identity(2**n_qubits)
        for _ in range(trotterisation.n_steps):
            u_target = np.dot(u_target, expm(-0.5j * h1 * trotterisation.dt))
            u_target = np.dot(u_target, expm(-0.5j * h2 * trotterisation.dt))
            u_target = np.dot(u_target, expm(-1.0j * h3 * trotterisation.dt))
            u_target = np.dot(u_target, expm(-0.5j * h2 * trotterisation.dt))
            u_target = np.dot(u_target, expm(-0.5j * h1 * trotterisation.dt))

        circ = trotterisation.decompose_bloq().to_cirq_circuit(
            cirq_quregs={"simulation": cirq.LineQubit.range(n_qubits)},
        )
        u = circ.unitary()

        np.testing.assert_allclose(u, u_target)

    def test_st_fourqubits(self):
        word1 = PauliWord(terms=(Pauli.X, Pauli.Y, Pauli.Z), qubits=(0, 2, 3))
        word2 = PauliWord(terms=(Pauli.Y, Pauli.Z, Pauli.X), qubits=(1, 2, 3))
        word3 = PauliWord(terms=(Pauli.Z, Pauli.X), qubits=(1, 3))

        h = PauliSum(
            coefficients=(5.0, 3.0, 1.0),
            terms=(word1, word2, word3),
            identity_coefficient=0,
        )
        n_qubits = h.n_qubits

        trotterisation = Trotterisation(h, t=10, n_steps=10, order=2)

        h1 = h.coefficients[0] * word1._to_matrix(
            length=n_qubits, ignore_idle_qubits=False
        )
        h2 = h.coefficients[1] * word2._to_matrix(
            length=n_qubits, ignore_idle_qubits=False
        )
        h3 = h.coefficients[2] * word3._to_matrix(
            length=n_qubits, ignore_idle_qubits=False
        )

        u_target = np.identity(2**n_qubits)
        for _ in range(trotterisation.n_steps):
            u_target = np.dot(u_target, expm(-0.5j * h1 * trotterisation.dt))
            u_target = np.dot(u_target, expm(-0.5j * h2 * trotterisation.dt))
            u_target = np.dot(u_target, expm(-1.0j * h3 * trotterisation.dt))
            u_target = np.dot(u_target, expm(-0.5j * h2 * trotterisation.dt))
            u_target = np.dot(u_target, expm(-0.5j * h1 * trotterisation.dt))

        circ = trotterisation.decompose_bloq().to_cirq_circuit(
            cirq_quregs={"simulation": cirq.LineQubit.range(n_qubits)},
        )
        u = circ.unitary()

        np.testing.assert_allclose(u, u_target, atol=1e-15)

    def test_nonzero_identity_lt_fourqubits(self):
        word1 = PauliWord(terms=(Pauli.X, Pauli.Y, Pauli.Z), qubits=(0, 2, 3))
        word2 = PauliWord(terms=(Pauli.Y, Pauli.Z, Pauli.X), qubits=(1, 2, 3))
        h = PauliSum(
            coefficients=(0.5, 0.5), terms=(word1, word2), identity_coefficient=10
        )
        n_qubits = h.n_qubits
        t = 12
        trotterisation = Trotterisation(h, t=t, n_steps=20, order=1)

        h1 = h.coefficients[0] * word1._to_matrix(
            length=n_qubits, ignore_idle_qubits=False
        )
        h2 = h.coefficients[1] * word2._to_matrix(
            length=n_qubits, ignore_idle_qubits=False
        )

        u_target = expm(-1j * t * h.identity_coefficient * np.identity(2**n_qubits))
        for _ in range(trotterisation.n_steps):
            u_target = np.dot(u_target, expm(-1j * h2 * trotterisation.dt))
            u_target = np.dot(u_target, expm(-1j * h1 * trotterisation.dt))

        circ = trotterisation.decompose_bloq().to_cirq_circuit(
            cirq_quregs={"simulation": cirq.LineQubit.range(n_qubits)},
        )
        u = circ.unitary()

        np.testing.assert_allclose(u, u_target, atol=1e-15)

    @pytest.mark.parametrize("trotter", [2, 4], indirect=True)
    @pytest.mark.parametrize("controlled", [False, True])
    def test_bloq_counts(self, trotter: Trotterisation, controlled: bool):
        bloq = trotter.controlled() if controlled else trotter
        manual_counts = bloq.bloq_counts(generalizer=[ignore_split_join])
        decomp_counts = bloq.decompose_bloq().bloq_counts(
            generalizer=[ignore_split_join]
        )
        decomp_unpack = _flatten_trotterizedunitary(decomp_counts)
        assert manual_counts == decomp_unpack

    @pytest.mark.parametrize("trotter", [2, 4], indirect=True)
    @pytest.mark.parametrize("controlled", [False, True])
    def test_qubit_counts(self, trotter: Trotterisation, controlled: bool):
        bloq = trotter.controlled() if controlled else trotter
        manual_counts = logical_qubit_resources(bloq)
        decomp_counts = logical_qubit_resources(bloq.decompose_bloq())
        assert manual_counts == decomp_counts
