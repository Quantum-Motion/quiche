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
from numpy.typing import NDArray
from qualtran import QAny, Side
from qualtran.resource_counting.generalizers import ignore_split_join
from qualtran.testing import (
    assert_equivalent_bloq_counts,
)
from scipy.linalg import block_diag, expm

from quiche.core import Errors, Pauli, PauliSum, PauliWord
from quiche.resources import logical_qubit_resources
from quiche.resources.bloqs import (
    QDRIFT,
    LCUBlockEncodingWrapper,
    PauliWordRotation,
    SelectPauliLCUWrapper,
    Trotterisation,
)
from quiche.resources.bloqs.simulation import _PauliWordRotationSequence


@pytest.fixture
def h_small() -> PauliSum:
    """Hamiltonian for QDRIFT and Trotter tests."""
    return PauliSum(
        coefficients=(2.0, -1.0, 0.25, -0.1),
        words=(
            PauliWord(paulis=(Pauli.X, Pauli.Y), qubits=(0, 1)),
            PauliWord(paulis=(Pauli.Z, Pauli.Z), qubits=(0, 1)),
            PauliWord(paulis=(Pauli.Y,), qubits=(0,)),
            PauliWord(paulis=(Pauli.X,), qubits=(1,)),
        ),
        identity_coefficient=3.0,
    )


class TestSelectPauliLCUWrapper:
    """Tests for SelectPauliLCUWrapper."""

    @pytest.fixture
    def select(self, h2: PauliSum, budget: Errors) -> SelectPauliLCUWrapper:
        num_select_qubits = ceil(log2(h2.num_words))
        phase_bitsize = max(ceil(log2(2.0 * num_select_qubits / budget.state_prep)), 2)
        words = (word.to_cirq(h2.num_qubits) for word in h2.words)

        return SelectPauliLCUWrapper(
            selection_bitsize=num_select_qubits + phase_bitsize,
            target_bitsize=h2.num_qubits,
            select_unitaries=tuple(words),
        )

    @pytest.mark.parametrize("controlled", [False, True])
    def test_bloq_counts(self, select: SelectPauliLCUWrapper, *, controlled: bool):
        bloq = select.controlled() if controlled else select
        assert_equivalent_bloq_counts(bloq, generalizer=[ignore_split_join])

    @pytest.mark.parametrize("controlled", [False, True])
    def test_qubit_counts(self, select: SelectPauliLCUWrapper, *, controlled: bool):
        bloq = select.controlled() if controlled else select
        manual_counts = logical_qubit_resources(bloq)
        decomp_counts = logical_qubit_resources(bloq.decompose_bloq())
        assert manual_counts == decomp_counts


class TestLCUBlockEncodingWrapper:
    """Tests for LCUBlockEncodingWrapper."""

    @pytest.fixture
    def blockencoding(self, h2: PauliSum, budget: Errors) -> LCUBlockEncodingWrapper:
        num_select_qubits = ceil(log2(h2.num_words_with_identity))
        phase_bitsize = max(ceil(log2(2.0 * num_select_qubits / budget.state_prep)), 2)
        return LCUBlockEncodingWrapper.from_hamiltonian(h2, phase_bitsize)

    def test_signature(
        self, blockencoding: LCUBlockEncodingWrapper, h2: PauliSum, budget: Errors
    ):
        """Check bloq signature."""
        num_select_qubits = ceil(log2(h2.num_words_with_identity))
        phase_bitsize = max(ceil(log2(2.0 * num_select_qubits / budget.state_prep)), 2)
        sig = blockencoding.signature
        assert len(sig) == 3

        reg = sig[0]
        assert reg.name == "selection"
        assert reg.dtype == QAny(num_select_qubits)
        assert reg.side == Side.THRU

        reg = sig[1]
        assert reg.name == "phase_gradient"
        assert reg.dtype == QAny(phase_bitsize)
        assert reg.side == Side.THRU

        reg = sig[2]
        assert reg.name == "target"
        assert reg.dtype == QAny(h2.num_qubits)
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
        # There are a total of h2.num_words_with_identity non-zero coefficients because
        # the identity coefficient is non-zero. Test that these are indeed non-zero and
        # all others are zero.
        np.testing.assert_equal(
            prep_coeffs[: h2.num_words_with_identity] != 0, desired=True
        )
        np.testing.assert_allclose(prep_coeffs[h2.num_words_with_identity :], 0.0)

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
        # tested separately in test_zerocoefficients. Truncate after
        # h2.num_words_with_identity in order to count the identity contribution.
        true_unitaries = true_unitaries[: h2.num_words_with_identity]
        true_prep_coeffs = true_prep_coeffs[: h2.num_words_with_identity]

        # Set the target unitaries and coefficients
        target_unitaries = [u.to_cirq(h2.num_qubits) for u in h2.words] + [
            cirq.DensePauliString.eye(h2.num_qubits)
        ]
        target_coefficients = [*h2.coefficients, h2.identity_coefficient]

        # Assert the unitaries (and signs)
        for ii in range(len(target_unitaries)):
            target = target_unitaries[ii] * np.sign(target_coefficients[ii])
            error_msg = (
                f"Unitaries at index {ii} do not agree: "
                f"{true_unitaries[ii]} vs {target}."
            )
            assert true_unitaries[ii] == target, error_msg

        # Assert the coefficients
        np.testing.assert_allclose(
            h2.lam * true_prep_coeffs**2, np.abs(target_coefficients)
        )

    @pytest.mark.parametrize("controlled", [False, True])
    def test_bloq_counts(
        self, blockencoding: LCUBlockEncodingWrapper, *, controlled: bool
    ):
        bloq = blockencoding.controlled() if controlled else blockencoding
        assert_equivalent_bloq_counts(bloq, generalizer=[ignore_split_join])

    @pytest.mark.parametrize("controlled", [False, True])
    def test_qubit_counts(
        self, blockencoding: LCUBlockEncodingWrapper, *, controlled: bool
    ):
        bloq = blockencoding.controlled() if controlled else blockencoding
        manual_counts = logical_qubit_resources(bloq)
        decomp_counts = logical_qubit_resources(bloq.decompose_bloq())
        assert manual_counts == decomp_counts


class TestPauliWordRotation:
    @pytest.fixture
    def rotation(self) -> PauliWordRotation:
        qubits = (0, 2, 5)
        num_qubits = 7
        phase = 0.4
        word = PauliWord(paulis=(Pauli.Y, Pauli.Z, Pauli.X), qubits=qubits)
        return PauliWordRotation(word, phase, num_qubits)

    def test_signature(self, rotation: PauliWordRotation):
        sig = rotation.signature
        assert len(sig) == 1

        reg = sig[0]
        assert reg.name == "system"
        assert reg.dtype == QAny(rotation.num_qubits)
        assert reg.side == Side.THRU

    def test_wrong_num_qubits(self):
        word = PauliWord(paulis=(Pauli.X, Pauli.Y, Pauli.X), qubits=(0, 2, 7))
        with pytest.raises(ValueError, match="Target qubit 7 is out of range"):
            PauliWordRotation(word, 0.4, 7)

    def test_decomposition(self, rotation: PauliWordRotation):
        h = rotation.word._to_matrix(
            length=rotation.num_qubits, ignore_idle_qubits=False
        )
        u_target = expm(-1j * h * rotation.angle / 2)
        u = rotation.decompose_bloq().tensor_contract()
        np.testing.assert_allclose(u, u_target, atol=1e-12)

    @pytest.mark.parametrize("controlled", [False, True])
    def test_bloq_counts(self, rotation: PauliWordRotation, *, controlled: bool):
        bloq = rotation.controlled() if controlled else rotation
        assert_equivalent_bloq_counts(bloq, generalizer=[ignore_split_join])

    @pytest.mark.parametrize("controlled", [False, True])
    def test_qubit_counts(self, rotation: PauliWordRotation, *, controlled: bool):
        bloq = rotation.controlled() if controlled else rotation
        manual_counts = logical_qubit_resources(bloq)
        decomp_counts = logical_qubit_resources(bloq.decompose_bloq())
        assert manual_counts == decomp_counts


class TestPauliWordRotationSequence:
    @pytest.fixture
    def sequence(self) -> _PauliWordRotationSequence:
        num_qubits = 4
        word1 = PauliWord(paulis=(Pauli.X, Pauli.Y, Pauli.Z), qubits=(0, 2, 3))
        word2 = PauliWord(paulis=(Pauli.Y, Pauli.Z, Pauli.X), qubits=(1, 2, 3))
        word3 = PauliWord(paulis=(Pauli.Z,), qubits=(1,))
        rotations = (
            PauliWordRotation(word1, 0.3, num_qubits),
            PauliWordRotation(word2, -0.7, num_qubits),
            PauliWordRotation(word3, 0.5, num_qubits),
            PauliWordRotation(word1, 0.3, num_qubits),
        )
        return _PauliWordRotationSequence(rotations)

    def test_bloq_counts(self, sequence: _PauliWordRotationSequence):
        assert_equivalent_bloq_counts(sequence, generalizer=[ignore_split_join])

    def test_qubit_counts(self, sequence: _PauliWordRotationSequence):
        manual_counts = logical_qubit_resources(sequence)
        decomp_counts = logical_qubit_resources(sequence.decompose_bloq())
        assert manual_counts == decomp_counts


class TestQDRIFT:
    @pytest.fixture
    def qdrift(self, h2: PauliSum) -> QDRIFT:
        # Sum with repeated word for call graph regression testing
        pauli_sum = PauliSum(
            coefficients=(*h2.coefficients, h2.coefficients[-1]),
            words=(*h2.words, h2.words[-1]),
            identity_coefficient=h2.identity_coefficient,
        )
        return QDRIFT(pauli_sum, time=5, num_samples=20, seed=1024)

    def test_invalid_negative_num_samples(self, h2: PauliSum):
        with pytest.raises(ValueError, match="Choose positive num_samples"):
            QDRIFT(h2, time=5, num_samples=-10, seed=1024)

    def test_invalid_negative_time(self, h2: PauliSum):
        with pytest.raises(ValueError, match="Choose positive evolution time"):
            QDRIFT(h2, time=-5, num_samples=4, seed=1024)

    @pytest.mark.parametrize("controlled", [False, True])
    def test_bloq_counts(self, qdrift: QDRIFT, *, controlled: bool):
        bloq = qdrift.controlled() if controlled else qdrift
        assert_equivalent_bloq_counts(bloq, generalizer=[ignore_split_join])

    @pytest.mark.parametrize("controlled", [False, True])
    def test_qubit_counts(self, qdrift: QDRIFT, *, controlled: bool):
        bloq = qdrift.controlled() if controlled else qdrift
        manual_counts = logical_qubit_resources(bloq)
        decomp_counts = logical_qubit_resources(bloq.decompose_bloq())
        assert manual_counts == decomp_counts

    @pytest.mark.parametrize("controlled", [False, True])
    def test_analytic(self, h_small: PauliSum, *, controlled: bool):
        """Check QDRIFT against the exact matrix exponential."""
        time = 0.25
        qdrift = QDRIFT(h_small, time=time, num_samples=200, seed=1024)
        bloq = qdrift.controlled() if controlled else qdrift
        actual = bloq.tensor_contract()

        exponential = expm(-1j * time * h_small._to_matrix())
        expected = (
            block_diag(np.eye(len(exponential)), exponential)
            if controlled
            else exponential
        )

        np.testing.assert_allclose(actual, expected, atol=0.1)


class TestTrotterisation:
    @pytest.fixture
    def trotter(self, h2: PauliSum, request: pytest.FixtureRequest) -> Trotterisation:
        return Trotterisation(h2, time=5, num_steps=4, order=request.param)

    @pytest.mark.parametrize("num_steps", [-10, 0])
    def test_invalid_num_steps(self, h2: PauliSum, num_steps: int):
        with pytest.raises(ValueError, match="Choose positive num_steps"):
            Trotterisation(h2, time=1, num_steps=num_steps, order=2)

    @pytest.mark.parametrize(
        ("order", "error_msg"),
        [
            (-2, "positive Trotter order"),
            (0, "positive Trotter order"),
            (3, "order must be even"),
        ],
    )
    def test_invalid_trotter_order(self, h2: PauliSum, order: int, error_msg: str):
        with pytest.raises(ValueError, match=error_msg):
            Trotterisation(h2, time=0.2, num_steps=2, order=order)

    def test_coeffs_indices_lie_trotter(self, h2: PauliSum):
        expected_indices = list(range(h2.num_words))
        expected_coeffs = [1.0] * h2.num_words

        trotter = Trotterisation(h2, time=1.0, num_steps=3, order=1)
        np.testing.assert_equal(trotter.trotter_indices, expected_indices)
        np.testing.assert_allclose(trotter.trotter_coeffs, expected_coeffs)

    @pytest.mark.parametrize(
        ("order", "sweep_weights"),
        [
            (2, np.array([1.0])),
            (4, np.array([(u := 1 / (4 - 4 ** (1 / 3))), u, 1 - 4 * u, u, u])),
            (
                6,
                np.kron(
                    [(v := 1 / (4 - 4 ** (1 / 5))), v, 1 - 4 * v, v, v],
                    [u, u, 1 - 4 * u, u, u],
                ),
            ),
        ],
    )
    def test_coeffs_indices_suzuki(
        self, h2: PauliSum, order: int, sweep_weights: NDArray[np.float64]
    ):
        forward = list(range(h2.num_words))
        sweep = forward + forward[::-1]
        expected_indices = sweep * len(sweep_weights)
        expected_coeffs = np.repeat(sweep_weights, len(sweep)) / 2

        trotter = Trotterisation(h2, time=1.0, num_steps=3, order=order)
        np.testing.assert_equal(trotter.trotter_indices, expected_indices)
        np.testing.assert_allclose(trotter.trotter_coeffs, expected_coeffs)

    @pytest.mark.parametrize(("order", "atol"), [(1, 1e-1), (2, 1e-3), (4, 1e-5)])
    @pytest.mark.parametrize("controlled", [False, True])
    def test_unitary(
        self, h_small: PauliSum, order: int, atol: float, *, controlled: bool
    ):
        """Check Trotterisation against the exact matrix exponential."""
        time = 0.25
        trotter = Trotterisation(h_small, time=time, num_steps=3, order=order)
        bloq = trotter.controlled() if controlled else trotter
        actual = bloq.tensor_contract()

        exponential = expm(-1j * time * h_small._to_matrix())
        expected = (
            block_diag(np.eye(len(exponential)), exponential)
            if controlled
            else exponential
        )

        np.testing.assert_allclose(actual, expected, atol=atol)

    @pytest.mark.parametrize("trotter", [1, 2, 4, 6], indirect=True)
    @pytest.mark.parametrize("controlled", [False, True])
    def test_bloq_counts(self, trotter: Trotterisation, *, controlled: bool):
        bloq = trotter.controlled() if controlled else trotter
        assert_equivalent_bloq_counts(bloq, generalizer=[ignore_split_join])

    @pytest.mark.parametrize("trotter", [1, 2, 4, 6], indirect=True)
    @pytest.mark.parametrize("controlled", [False, True])
    def test_qubit_counts(self, trotter: Trotterisation, *, controlled: bool):
        bloq = trotter.controlled() if controlled else trotter
        manual_counts = logical_qubit_resources(bloq)
        decomp_counts = logical_qubit_resources(bloq.decompose_bloq())
        assert manual_counts == decomp_counts
