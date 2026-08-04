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

# TODO(Vasco): refactor these tests.

from math import ceil, log2

import cirq
import numpy as np
import pytest
from qualtran import Bloq, QAny, Side
from qualtran.bloqs.chemistry.trotter.trotterized_unitary import (
    TrotterizedUnitary,
)
from qualtran.resource_counting.generalizers import ignore_split_join
from qualtran.testing import (
    assert_equivalent_bloq_counts,
)
from scipy.linalg import expm

from quiche.core import Errors, Pauli, PauliSum, PauliWord
from quiche.hamlib import get_dataset, parse_hamiltonian
from quiche.resources import logical_qubit_resources
from quiche.resources.bloqs import (
    QDRIFT,
    LCUBlockEncodingWrapper,
    PauliWordRotation,
    SelectPauliLCUWrapper,
    Trotterisation,
)


def _load_h2() -> PauliSum:
    """Load the H2 Hamiltonian from python/tests/data/H2.hdf5."""
    filename = "python/tests/data/H2.hdf5"
    dataset = "ham_JW-4"  # choose smallest size to keep tests fast
    raw_data = get_dataset(filename, dataset)
    return parse_hamiltonian(raw_data)


def _default_budget() -> Errors:
    tot_error = 0.16
    return Errors(
        estimation=tot_error / 3.0,
        simulation=1.0,
        rotations=tot_error / 3.0,
        state_prep=tot_error / 3.0,
        overlap=1,
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


def _compare_manual_decomp_counts_trotter(bloq: Bloq) -> bool:
    """Compare call graphs where TrotterizedUnitary is unwrapped."""
    manual_counts = bloq.bloq_counts(generalizer=[ignore_split_join])
    decomp_counts = bloq.decompose_bloq().bloq_counts(generalizer=[ignore_split_join])
    decomp_unpack = _flatten_trotterizedunitary(decomp_counts)
    return manual_counts == decomp_unpack


class TestSelectPauliLCUWrapper:
    """Tests for PauliLCUWrapper."""

    h = _load_h2()
    budget = _default_budget()
    select_nqubits = ceil(log2(h.n_terms))
    phase_bitsize = max(ceil(log2(2.0 * select_nqubits / budget.state_prep)), 2)
    terms = []
    for term in h.terms:
        terms.append(term.to_cirq(h.n_qubits))
    select = SelectPauliLCUWrapper(
        selection_bitsize=select_nqubits + phase_bitsize,
        target_bitsize=h.n_qubits,
        select_unitaries=terms,
    )

    @pytest.mark.parametrize("bloq", [select, select.controlled()])
    def test_bloq_counts(self, bloq: Bloq):
        assert_equivalent_bloq_counts(bloq, generalizer=[ignore_split_join])

    @pytest.mark.parametrize("bloq", [select, select.controlled()])
    def test_qubit_counts(self, bloq: Bloq):
        manual_counts = logical_qubit_resources(bloq)
        decomp_counts = logical_qubit_resources(bloq.decompose_bloq())
        assert manual_counts == decomp_counts


class TestLCUBlockEncodingWrapper:
    """Tests for LCUBlockEncodingWrapper."""

    h = _load_h2()
    budget = _default_budget()
    select_nqubits = ceil(log2(h.n_terms))
    phase_bitsize = max(ceil(log2(2.0 * select_nqubits / budget.state_prep)), 2)
    blockencoding = LCUBlockEncodingWrapper.from_hamiltonian(h, phase_bitsize)

    def test_signature(self):
        """Check bloq signature."""
        sig = self.blockencoding.signature
        assert len(sig) == 2

        reg = sig[0]
        assert reg.name == "selection"
        assert reg.dtype == QAny(self.select_nqubits + self.phase_bitsize)
        assert reg.side == Side.THRU

        reg = sig[1]
        assert reg.name == "target"
        assert reg.dtype == QAny(self.h.n_qubits)
        assert reg.side == Side.THRU

    def test_small_phase_bitsize(self):
        phase_bitsize = 1
        error_msg = "Choose phase_bitsize at least 2"
        with pytest.raises(ValueError, match=error_msg):
            LCUBlockEncodingWrapper.from_hamiltonian(self.h, phase_bitsize)

    def test_zerocoefficients(self):
        prep_coeffs = self.blockencoding.prepare.stateprep.state_coefficients
        # The block encoding pads the coefficients to a power of two. All coefficients
        # beyond the ones needed for the Hamiltonian should be zero and have no effect.
        # There are a total of self.h.n_terms + 1 non-zero coefficients because the
        # identity coefficient is non-zero. Test that these are indeed non-zero and all
        # others are zero.
        np.testing.assert_equal(prep_coeffs[: self.h.n_terms + 1] != 0, True)
        np.testing.assert_allclose(prep_coeffs[self.h.n_terms + 1 :], 0.0)

    def test_selectunitaries(self):
        # Checks that the right unitaries and coefficients will be applied in the select
        # unitary.
        be = LCUBlockEncodingWrapper.from_hamiltonian(self.h, self.phase_bitsize)
        true_unitaries = be.select.select_unitaries
        true_prep_coeffs = np.array(
            be.prepare.stateprep.state_coefficients, dtype=complex
        )
        # Truncate to the non-zero terms. All truncated coefficients are zero, which is
        # tested separately in test_zerocoefficients. Truncate after self.h.n_terms+1 in
        # order to count the identity contribution.
        true_unitaries = true_unitaries[: self.h.n_terms + 1]
        true_prep_coeffs = true_prep_coeffs[: self.h.n_terms + 1]

        # Set the target unitaries and coefficients
        target_unitaries = [u.to_cirq(self.h.n_qubits) for u in self.h.terms] + [
            cirq.DensePauliString.eye(self.h.n_qubits)
        ]
        target_coefficients = [*self.h.coefficients, self.h.identity_coefficient]
        # Add the identity coefficient to the 1-norm
        lam = self.h.lam + abs(self.h.identity_coefficient)

        # Assert the unitaries
        for ii in range(len(target_unitaries)):
            err_msg = (
                f"Unitaries at index {ii} do not agree: "
                f"{true_unitaries[ii]} vs {target_unitaries[ii]}."
            )
            assert true_unitaries[ii] == target_unitaries[ii], err_msg

        # Assert the coefficients
        np.testing.assert_allclose(lam * true_prep_coeffs**2, target_coefficients)

    @pytest.mark.parametrize("bloq", [blockencoding, blockencoding.controlled()])
    def test_bloq_counts(self, bloq: Bloq):
        assert_equivalent_bloq_counts(bloq, generalizer=[ignore_split_join])

    @pytest.mark.parametrize("bloq", [blockencoding, blockencoding.controlled()])
    def test_qubit_counts(self, bloq: Bloq):
        manual_counts = logical_qubit_resources(bloq)
        decomp_counts = logical_qubit_resources(bloq.decompose_bloq())
        assert manual_counts == decomp_counts


class TestPauliWordRotation:
    n_qubits = 7
    qubits = (0, 2, 5)
    phase = 0.4
    word = PauliWord(terms=(Pauli.Y, Pauli.Z, Pauli.X), qubits=qubits)
    rot = PauliWordRotation(word, phase, n_qubits)

    def test_signature(self):
        sig = self.rot.signature
        assert len(sig) == 1

        reg = sig[0]
        assert reg.name == "system"
        assert reg.dtype == QAny(self.n_qubits)
        assert reg.side == Side.THRU

    def test_wrong_n_qubits(self):
        word = PauliWord(terms=(Pauli.X, Pauli.Y, Pauli.X), qubits=(0, 2, 7))
        rot = PauliWordRotation(word, 0.4, self.n_qubits)

        with pytest.raises(ValueError, match="Target qubit 7 is out of range"):
            rot.decompose_bloq()

    def test_decomposition(self):
        circ = self.rot.decompose_bloq().to_cirq_circuit(
            cirq_quregs={"system": cirq.LineQubit.range(self.n_qubits)},
        )
        h = self.word._to_matrix(ignore_idle_qubits=True)
        u = circ._unitary_()
        u_target = expm(-1j * h * self.phase / 2)
        np.testing.assert_allclose(u, u_target)

    @pytest.mark.parametrize("bloq", [rot, rot.controlled()])
    def test_bloq_counts(self, bloq: Bloq):
        assert_equivalent_bloq_counts(bloq, generalizer=[ignore_split_join])

    @pytest.mark.parametrize("bloq", [rot, rot.controlled()])
    def test_qubit_counts(self, bloq: Bloq):
        manual_counts = logical_qubit_resources(bloq)
        decomp_counts = logical_qubit_resources(bloq.decompose_bloq())
        assert manual_counts == decomp_counts


class TestQDRIFT:
    h = _load_h2()
    t = 5
    n_terms = 20
    seed = 1024
    qdrift = QDRIFT(h, t, n_terms, seed)

    def test_invalid_negative_nsteps(self):
        n_terms = -10
        t = 5
        with pytest.raises(ValueError, match="Choose positive n_terms"):
            QDRIFT(self.h, t, n_terms, self.seed)

    def test_invalid_negative_time(self):
        n_terms = 10
        t = -5
        with pytest.raises(ValueError, match="Choose positive evolution time"):
            QDRIFT(self.h, t, n_terms, self.seed)

    @pytest.mark.parametrize("bloq", [qdrift, qdrift.controlled()])
    def test_bloq_counts(self, bloq: Bloq):
        assert _compare_manual_decomp_counts_trotter(bloq)

    @pytest.mark.parametrize("bloq", [qdrift, qdrift.controlled()])
    def test_qubit_counts(self, bloq: Bloq):
        manual_counts = logical_qubit_resources(bloq)
        decomp_counts = logical_qubit_resources(bloq.decompose_bloq())
        assert manual_counts == decomp_counts


class TestTrotterisation:
    h = _load_h2()
    budget = _default_budget()
    t = 5
    n_steps = 10
    trotter_order2 = Trotterisation(h, t, n_steps, order=2)
    trotter_order4 = Trotterisation(h, t, n_steps, order=4)

    @pytest.mark.parametrize("n_steps", [-10, 0])
    def test_invalid_nsteps(self, n_steps: int):
        order = 2
        with pytest.raises(ValueError, match="Choose positive n_steps"):
            Trotterisation(self.h, self.t, n_steps, order)

    @pytest.mark.parametrize(
        ("order", "err_msg"),
        [
            (-2, "positive Trotter order"),
            (0, "positive Trotter order"),
            (3, "order must be even"),
        ],
    )
    def test_invalid_trotter_order(self, order: int, err_msg: str):
        with pytest.raises(ValueError, match=err_msg):
            Trotterisation(self.h, self.t, self.n_steps, order)

    def test_coeffs_indices_lie_trotter(self):
        word1 = PauliWord(terms=(Pauli.X, Pauli.Y, Pauli.Z), qubits=(0, 2, 3))
        h = PauliSum(
            coefficients=(5.0, 19.0, 10.0, 15.0),
            terms=(word1, word1, word1, word1),
            identity_coefficient=0,
        )

        order = 1
        trotterisation = Trotterisation(h, self.t, self.n_steps, order)
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

        order = 2
        trotterisation = Trotterisation(h, self.t, self.n_steps, order)
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
        trotterisation = Trotterisation(h, self.t, self.n_steps, order)
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
        n_qubits = h.n_qubits
        order = 1
        trotterisation = Trotterisation(h, self.t, self.n_steps, order)

        h1 = coeffs[0] * word1._to_matrix(ignore_idle_qubits=False)
        h2 = coeffs[1] * word2._to_matrix(ignore_idle_qubits=False)
        h3 = coeffs[2] * word3._to_matrix(ignore_idle_qubits=False)

        u_target = np.identity(2**n_qubits)
        for _ in range(trotterisation.n_steps):
            u_target = np.dot(u_target, expm(-1j * h3 * trotterisation.dt))
            u_target = np.dot(u_target, expm(-1j * h2 * trotterisation.dt))
            u_target = np.dot(u_target, expm(-1j * h1 * trotterisation.dt))

        circ = trotterisation.decompose_bloq().to_cirq_circuit(
            cirq_quregs={"simulation": cirq.LineQubit.range(n_qubits)},
        )
        u = circ._unitary_()
        np.testing.assert_allclose(u, u_target)

    def test_lt_fourqubits(self):
        word1 = PauliWord(terms=(Pauli.X, Pauli.Y, Pauli.Z), qubits=(0, 2, 3))
        word2 = PauliWord(terms=(Pauli.Y, Pauli.Z, Pauli.X), qubits=(1, 2, 3))
        h = PauliSum(
            coefficients=(5.0, 10.0), terms=(word1, word2), identity_coefficient=0
        )
        n_qubits = h.n_qubits
        order = 1
        trotterisation = Trotterisation(h, self.t, self.n_steps, order)

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
        u = circ._unitary_()

        np.testing.assert_allclose(u, u_target, atol=1e-15)

    def test_st_singlequbit(self):
        word1 = PauliWord(terms=(Pauli.Z,), qubits=(0,))
        word2 = PauliWord(terms=(Pauli.Y,), qubits=(0,))
        word3 = PauliWord(terms=(Pauli.X,), qubits=(0,))
        coeffs = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        terms = (word1, word2, word3)
        h = PauliSum(coefficients=coeffs, terms=terms, identity_coefficient=0)
        n_qubits = h.n_qubits
        order = 2
        trotterisation = Trotterisation(h, self.t, self.n_steps, order)

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
        u = circ._unitary_()

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
        order = 2
        trotterisation = Trotterisation(h, self.t, self.n_steps, order)

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
        u = circ._unitary_()

        np.testing.assert_allclose(u, u_target, atol=1e-15)

    def test_nonzero_identity_lt_fourqubits(self):
        word1 = PauliWord(terms=(Pauli.X, Pauli.Y, Pauli.Z), qubits=(0, 2, 3))
        word2 = PauliWord(terms=(Pauli.Y, Pauli.Z, Pauli.X), qubits=(1, 2, 3))
        h = PauliSum(
            coefficients=(0.5, 0.5), terms=(word1, word2), identity_coefficient=10
        )
        n_qubits = h.n_qubits
        order = 1
        trotterisation = Trotterisation(h, self.t, self.n_steps, order)

        h1 = h.coefficients[0] * word1._to_matrix(
            length=n_qubits, ignore_idle_qubits=False
        )
        h2 = h.coefficients[1] * word2._to_matrix(
            length=n_qubits, ignore_idle_qubits=False
        )

        u_target = expm(
            -1j * self.t * h.identity_coefficient * np.identity(2**n_qubits)
        )
        for _ in range(trotterisation.n_steps):
            u_target = np.dot(u_target, expm(-1j * h2 * trotterisation.dt))
            u_target = np.dot(u_target, expm(-1j * h1 * trotterisation.dt))

        circ = trotterisation.decompose_bloq().to_cirq_circuit(
            cirq_quregs={"simulation": cirq.LineQubit.range(n_qubits)},
        )
        u = circ._unitary_()

        np.testing.assert_allclose(u, u_target, atol=1e-15)

    @pytest.mark.parametrize(
        "bloq",
        [
            trotter_order2,
            trotter_order2.controlled(),
            trotter_order4,
            trotter_order4.controlled(),
        ],
    )
    def test_bloq_counts(self, bloq: Bloq):
        assert _compare_manual_decomp_counts_trotter(bloq)

    @pytest.mark.parametrize(
        "bloq",
        [
            trotter_order2,
            trotter_order2.controlled(),
            trotter_order4,
            trotter_order4.controlled(),
        ],
    )
    def test_qubit_counts(self, bloq: Bloq):
        manual_counts = logical_qubit_resources(bloq)
        decomp_counts = logical_qubit_resources(bloq.decompose_bloq())
        assert manual_counts == decomp_counts
