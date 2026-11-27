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

"""Tests for the DFTHC bloqs."""

import random

import attrs
import numpy as np
import pytest
from numpy.typing import ArrayLike
from qualtran import BloqBuilder
from qualtran.bloqs.basic_gates import CZ
from qualtran.resource_counting import (
    QECGatesCost,
    get_bloq_callee_counts,
    get_cost_value,
)
from qualtran.resource_counting.generalizers import ignore_alloc_free, ignore_split_join
from qualtran.simulation.classical_sim import ClassicalValT
from qualtran.testing import (
    assert_equivalent_bloq_counts,
    assert_valid_bloq_decomposition,
)

from quiche.core.electronic import DFTHCHamiltonian
from quiche.resources.bloqs.dfthc import (
    DFTHCBlockEncoding,
    DFTHCInnerBlockEncoding,
    DFTHCWalkOperator,
    InnerPrepareDFTHC,
    MajoranaDFTHC,
    OuterPrepareDFTHC,
    RotationPrepareDFTHC,
    SelectDFTHC,
)
from quiche.resources.bloqs.dfthc.encodings import (
    givens_angles,
    inner_qrom_coefficients,
    pack_integers,
    quantise_angles,
    rotation_qrom_angles,
    tail_qrom_angles,
)
from quiche.resources.bloqs.dfthc.qrom import (
    InnerQROAM,
    InnerQROM,
    InnerQROMTail,
    RotationQROM,
)
from quiche.resources.bloqs.dfthc.select import GivensLadder

# FeMoco-54 parameters, from G. H. Low et al., "Fast Quantum Simulation of Electronic
# Structure by Spectral Amplification", Phys. Rev. X, vol. 15, no. 4, p. 041016
N = 54
R, B, C = 10, 27, 27
B_ROT = B_K1 = B_K2 = 15
K_1 = K_4 = 2
K_2 = K_5 = 4
S = 7
N_SMALL, R_SMALL, B_SMALL, C_SMALL, B_ROT_SMALL = 4, 3, 2, 2, 6
B_K2_SMALL, K_2_SMALL = 3, 1

RESOLUTION = 4 * np.pi / 2**N_SMALL  # The angle of the least significant bit.


@pytest.fixture
def dummy_dfthc_hamiltonian() -> DFTHCHamiltonian:
    """Return a dummy DFTHCHamiltonian for use in tests."""
    # A symmetric one body term with eigenvalues of both signs.
    one_body = np.random.default_rng(1).normal(size=(N, N))
    return DFTHCHamiltonian(
        num_orbitals=N,
        num_ranks=R,
        num_bases=B,
        num_copies=C,
        unit_vectors=np.random.default_rng(0).normal(size=(R, B, N)),
        weight_vectors=np.ones((R, B, C)),
        identity_weights=np.zeros((R, C)),
        bliss_matrix=np.zeros((N, N)),
        factorised_one_body=one_body + one_body.T,
        const=0.0,
        bliss_constant=0.0,
        num_electrons=N,
    )


def random_rotation_data(
    num_ranks: int, num_bases: int, num_bits: int, seed: int = 0
) -> tuple[tuple[int, ...], ...]:
    """Get random rotations, indexed ``[rank][inner_index]``, of ``num_bits`` bits."""
    rng = random.Random(seed)  # noqa: S311 (not used for cryptography)
    return tuple(
        tuple(rng.getrandbits(num_bits) for _ in range(num_bases))
        for _ in range(num_ranks)
    )


def random_tail_data(
    num_orbitals: int, num_bits: int, seed: int = 0
) -> tuple[tuple[int, int], ...]:
    """Get random ``(rotations, G_1)`` pairs, with words of ``num_bits`` bits."""
    rng = random.Random(seed)  # noqa: S311 (not used for cryptography)
    return tuple(
        (rng.getrandbits(num_bits), rng.getrandbits(1)) for _ in range(num_orbitals)
    )


def decode_rotation_data(rotation_data: tuple[tuple[int, ...], ...]) -> np.ndarray:
    """Get the angles in the rotation words, as the Givens ladder splits them up."""
    partition = GivensLadder(N, B_ROT).partition
    values = [
        [partition.call_classically(x=word)[0] for word in row] for row in rotation_data
    ]
    # The register of an angle holds a fraction of 4 pi.
    return np.array(values, dtype=float) * 4 * np.pi / 2**B_ROT


def sampled_coefficients(qroam: InnerQROAM) -> np.ndarray:
    """Get the signed coefficients that alias sampling with the QROAM's data gives."""
    qrom = qroam.qrom
    assert qrom.inner_data is not None
    assert qrom.sign_data is not None
    num_inner, num_keep = qrom.num_inner, 2**qrom.num_bits_keep
    sampled = np.zeros((qrom.num_rc, num_inner))
    for x, (words, signs) in enumerate(
        zip(qrom.inner_data, qrom.sign_data, strict=True)
    ):
        for i, (word, s) in enumerate(zip(words, signs, strict=True)):
            alt, keep, d = map(int, qroam.alias_word_partition.call_classically(x=word))
            # Entry i is kept for `keep` of the values of sigma, and its alias is
            # swapped in for the rest, with the sign flipped by d.
            sampled[x, i] += keep * (-1) ** s
            sampled[x, alt] += (num_keep - keep) * (-1) ** (s ^ d)
    return sampled / (num_inner * num_keep)


class TestOuterPrepareDFTHC:
    """Tests for the outer index state preparation."""

    prep = OuterPrepareDFTHC(
        num_orbitals=N,
        num_ranks=R,
        num_copies=C,
        num_bits_keep=B_K1,
        num_bits_phase_grad=B_ROT,
        num_bits_amp_rotations=S,
        log_block_size=K_1,
        log_block_size_adjoint=K_5,
    )

    def test_decomposition(self):
        assert_valid_bloq_decomposition(self.prep)

    def test_bloq_counts(self):
        assert_equivalent_bloq_counts(
            self.prep, generalizer=[ignore_split_join, ignore_alloc_free]
        )


class TestInnerPrepareDFTHC:
    """Tests for the inner index state preparation."""

    prep = InnerPrepareDFTHC(
        num_orbitals=N,
        num_ranks=R,
        num_bases=B,
        num_copies=C,
        num_bits_keep=B_K2,
        num_bits_phase_grad=B_ROT,
        num_bits_amp_rotations=S,
        log_block_size=K_2,
        log_block_size_adjoint=K_4,
    )

    def test_decomposition(self):
        assert_valid_bloq_decomposition(self.prep)

    def test_bloq_counts(self):
        assert_equivalent_bloq_counts(
            self.prep, generalizer=[ignore_split_join, ignore_alloc_free]
        )

    def test_adjoint_omits_cz(self):
        # The CZ applies the sign factor (-1)^(f d) once, in the forward preparation,
        # so the unpreparation must not undo it.
        assert dict(get_bloq_callee_counts(self.prep))[CZ()] == 1
        assert CZ() not in dict(get_bloq_callee_counts(self.prep.adjoint()))


class TestRotationPrepareDFTHC:
    """Tests for the rotation angle state preparation."""

    prep = RotationPrepareDFTHC(
        num_orbitals=N,
        num_ranks=R,
        num_bases=B,
        num_bits_phase_grad=B_ROT,
    )

    def test_decomposition(self):
        assert_valid_bloq_decomposition(self.prep)

    def test_bloq_counts(self):
        assert_equivalent_bloq_counts(
            self.prep, generalizer=[ignore_split_join, ignore_alloc_free]
        )


class TestRotationQROM:
    """Tests for the QROM that loads the rotation words."""

    qrom = RotationQROM(
        num_orbitals=N_SMALL,
        num_ranks=R_SMALL,
        num_bases=B_SMALL,
        num_bits_phase_grad=B_ROT_SMALL,
        num_controls=1,
        rotation_data=random_rotation_data(
            R_SMALL, B_SMALL, (N_SMALL - 1) * B_ROT_SMALL
        ),
    )

    def test_decomposition(self):
        assert_valid_bloq_decomposition(self.qrom)

    def test_bloq_counts(self):
        assert_equivalent_bloq_counts(
            self.qrom, generalizer=[ignore_split_join, ignore_alloc_free]
        )

    def test_loads_rotation_data(self):
        data = tuple(
            tuple(1000 * (r + 1) + b for b in range(B_SMALL)) for r in range(R_SMALL)
        )
        qrom = RotationQROM(
            num_orbitals=N_SMALL,
            num_ranks=R_SMALL,
            num_bases=B_SMALL,
            num_bits_phase_grad=B_ROT_SMALL,
            num_controls=1,
            rotation_data=data,
        )
        cbloq = qrom.decompose_bloq()
        names = [reg.name for reg in qrom.signature.rights()]

        def lookup(**registers: int) -> dict[str, ClassicalValT]:
            return dict(zip(names, cbloq.call_classically(**registers), strict=True))

        for r in range(R_SMALL):
            for b in range(B_SMALL):
                loaded = lookup(
                    ctrl=1, inner_index=b, rank=r, rotations=0, non_identity=0
                )
                assert loaded["rotations"] == data[r][b]
                assert loaded["non_identity"] == 0

                idle = lookup(
                    ctrl=0, inner_index=b, rank=r, rotations=0, non_identity=0
                )
                assert idle["rotations"] == 0

        # b = B is the identity term, which flags instead of loading rotations.
        identity = lookup(
            ctrl=1, inner_index=B_SMALL, rank=0, rotations=0, non_identity=0
        )
        assert identity["rotations"] == 0
        assert identity["non_identity"] == 1

    @pytest.mark.parametrize(
        ("data", "error_msg"),
        [
            (((0, 0),) * (R_SMALL - 1), "must have shape"),
            (((0,),) * R_SMALL, "must have shape"),
            (((2 ** ((N_SMALL - 1) * B_ROT_SMALL), 0),) * R_SMALL, "must fit in"),
            (((-1, 0),) * R_SMALL, "must fit in"),
        ],
        ids=["too_few_ranks", "too_few_bases", "word_too_wide", "negative_word"],
    )
    def test_invalid_rotation_data(self, data: tuple, error_msg: str):
        with pytest.raises(ValueError, match=error_msg):
            RotationQROM(
                num_orbitals=N_SMALL,
                num_ranks=R_SMALL,
                num_bases=B_SMALL,
                num_bits_phase_grad=B_ROT_SMALL,
                rotation_data=data,
            )

    def test_bloq_counts_without_data(self):
        # Without data the call graph must still match the decomposition, which loads
        # zeros with a controlled XorK rather than the `(CNOT(), 0)` placeholder.
        assert_equivalent_bloq_counts(
            attrs.evolve(self.qrom, rotation_data=None),
            generalizer=[ignore_split_join, ignore_alloc_free],
        )


class TestInnerQROM:
    """Tests for the QROM that loads the alias sampling words of the SF terms."""

    # Distinct non-zero words, indexed [outer_index][inner_index], so that padding
    # and misplaced words are detected.
    inner_data = tuple(
        tuple(x * (B_SMALL + 1) + b + 1 for b in range(B_SMALL + 1))
        for x in range(R_SMALL * C_SMALL)
    )
    qrom = InnerQROM(
        num_orbitals=N_SMALL,
        num_ranks=R_SMALL,
        num_bases=B_SMALL,
        num_copies=C_SMALL,
        num_bits_keep=B_K2_SMALL,
        log_block_size=K_2_SMALL,
        num_controls=1,
        inner_data=inner_data,
    )

    def test_decomposition(self):
        assert_valid_bloq_decomposition(self.qrom)

    def test_bloq_counts(self):
        assert_equivalent_bloq_counts(
            self.qrom, generalizer=[ignore_split_join, ignore_alloc_free]
        )

    def test_loads_inner_data(self):
        cbloq = self.qrom.decompose_bloq()
        names = [reg.name for reg in self.qrom.signature.rights()]
        block_size, num_rc = self.qrom.block_size, R_SMALL * C_SMALL

        def lookup(**registers: int) -> dict[str, ClassicalValT]:
            empty = {"alias_words": np.zeros(block_size, dtype=int)}
            empty |= {"G_0": 0, "G_1": 0, "rank": 0}
            return dict(
                zip(names, cbloq.call_classically(**registers, **empty), strict=True)
            )

        for outer_index in range(num_rc + N_SMALL):
            for high in range(self.qrom.num_blocks):
                loaded = lookup(ctrl=1, outer_index=outer_index, inner_index_high=high)
                # Slot i holds entry high * block_size + i. Padding past B, and the
                # D_1/Q_1 terms loaded by `InnerQROMTail`, get nothing.
                expected = [
                    self.inner_data[outer_index][b]
                    if outer_index < num_rc and b <= B_SMALL
                    else 0
                    for b in range(high * block_size, (high + 1) * block_size)
                ]
                np.testing.assert_array_equal(loaded["alias_words"], expected)
                assert loaded["rank"] == (
                    outer_index // C_SMALL if outer_index < num_rc else 0
                )
                assert loaded["G_0"] == loaded["G_1"] == 0

                idle = lookup(ctrl=0, outer_index=outer_index, inner_index_high=high)
                np.testing.assert_array_equal(idle["alias_words"], 0)
                assert idle["rank"] == 0

    @pytest.mark.parametrize(
        ("field", "data", "error_msg"),
        [
            ("inner_data", ((0,) * (B_SMALL + 1),) * 5, "must have shape"),
            # Words are ceil(log(B + 1)) + b_k2 + 1 = 2 + 3 + 1 bits wide.
            ("inner_data", ((2**6,) * (B_SMALL + 1),) * 6, "must fit in"),
            ("inner_data", ((-1,) * (B_SMALL + 1),) * 6, "must fit in"),
            ("sign_data", ((0,) * B_SMALL,) * 6, "must have shape"),
            ("sign_data", ((2,) * (B_SMALL + 1),) * 6, "must be 0 or 1"),
        ],
        ids=[
            "too_few_rows",
            "word_too_wide",
            "negative_word",
            "sign_row_too_short",
            "sign_not_a_bit",
        ],
    )
    def test_invalid_inner_data(self, field: str, data: tuple, error_msg: str):
        with pytest.raises(ValueError, match=error_msg):
            attrs.evolve(self.qrom, **{field: data})

    def test_bloq_counts_without_data(self):
        # Without data the call graph must still match the decomposition, which
        # loads the rank and zero words with controlled XorKs.
        assert_equivalent_bloq_counts(
            attrs.evolve(self.qrom, inner_data=None),
            generalizer=[ignore_split_join, ignore_alloc_free],
        )


class TestInnerQROMTail:
    """Tests for the QROM that loads the D_1/Q_1 data."""

    qrom = InnerQROMTail(
        num_orbitals=N_SMALL,
        num_ranks=R_SMALL,
        num_copies=C_SMALL,
        num_bits_phase_grad=B_ROT_SMALL,
        num_controls=1,
        tail_data=random_tail_data(N_SMALL, (N_SMALL - 1) * B_ROT_SMALL),
    )

    def test_decomposition(self):
        assert_valid_bloq_decomposition(self.qrom)

    def test_bloq_counts(self):
        assert_equivalent_bloq_counts(
            self.qrom, generalizer=[ignore_split_join, ignore_alloc_free]
        )

    def test_loads_tail_data(self):
        data = tuple((1000 * (j + 1), j % 2) for j in range(N_SMALL))
        qrom = InnerQROMTail(
            num_orbitals=N_SMALL,
            num_ranks=R_SMALL,
            num_copies=C_SMALL,
            num_bits_phase_grad=B_ROT_SMALL,
            num_controls=1,
            tail_data=data,
        )
        cbloq = qrom.decompose_bloq()
        names = [reg.name for reg in qrom.signature.rights()]

        def lookup(**registers: int) -> dict[str, ClassicalValT]:
            return dict(zip(names, cbloq.call_classically(**registers), strict=True))

        num_rc = R_SMALL * C_SMALL
        for outer_index in range(num_rc + N_SMALL):
            loaded = lookup(
                ctrl=1, outer_index=outer_index, rotations=0, G_0=0, G_1=0, rank=0
            )
            # The SF terms are loaded by `InnerQROM`, so nothing is written for them.
            word, g_1 = data[outer_index - num_rc] if outer_index >= num_rc else (0, 0)
            assert loaded["rotations"] == word
            assert loaded["G_0"] == (outer_index >= num_rc)
            assert loaded["G_1"] == g_1
            assert loaded["rank"] == 0

            idle = lookup(
                ctrl=0, outer_index=outer_index, rotations=0, G_0=0, G_1=0, rank=0
            )
            assert idle["rotations"] == 0
            assert idle["G_0"] == 0

    @pytest.mark.parametrize(
        ("data", "error_msg"),
        [
            (((0, 0),) * (N_SMALL - 1), "must have"),
            (((2 ** ((N_SMALL - 1) * B_ROT_SMALL), 0),) * N_SMALL, "must fit in"),
            (((-1, 0),) * N_SMALL, "must fit in"),
            (((0, 2),) * N_SMALL, "must be 0 or 1"),
        ],
        ids=["too_few_entries", "word_too_wide", "negative_word", "g_1_not_a_bit"],
    )
    def test_invalid_tail_data(self, data: tuple, error_msg: str):
        with pytest.raises(ValueError, match=error_msg):
            InnerQROMTail(
                num_orbitals=N_SMALL,
                num_ranks=R_SMALL,
                num_copies=C_SMALL,
                num_bits_phase_grad=B_ROT_SMALL,
                tail_data=data,
            )

    def test_bloq_counts_without_data(self):
        # Without data the call graph must still match the decomposition, which
        # flips G_0 and loads zeros with controlled XorKs.
        assert_equivalent_bloq_counts(
            attrs.evolve(self.qrom, tail_data=None),
            generalizer=[ignore_split_join, ignore_alloc_free],
        )


class TestMajorana:
    """Tests for the Majorana operator."""

    majorana = MajoranaDFTHC()

    def test_decomposition(self):
        assert_valid_bloq_decomposition(self.majorana)

    def test_bloq_counts(self):
        assert_equivalent_bloq_counts(
            self.majorana, generalizer=[ignore_split_join, ignore_alloc_free]
        )

    @pytest.mark.parametrize(
        ("g_1", "expected"),
        [(0, [[0, 1], [0, 0]]), (1, [[0, 0], [1, 0]])],
        ids=["d_1", "q_1"],
    )
    def test_g_1_selects_d_1_or_q_1(self, g_1: int, expected: list[list[int]]):
        # Fix the controls G_0 = 1 (a D_1/Q_1 term), G_1, non_identity = 0 and s in
        # |+>, as the Hadamards around the Majorana operator do (Fig. 4).
        basis = np.eye(2)
        plus = np.ones(2) / np.sqrt(2)
        controls = np.kron(np.kron(np.kron(basis[1], basis[g_1]), basis[0]), plus)
        # Embed the target qubit after the controls, and keep only its operator.
        embed = np.kron(controls[:, None], np.eye(2))
        target_operator = embed.T @ self.majorana.tensor_contract() @ embed

        # D_1 applies a = (X + iY) / 2 and Q_1 applies a^dagger = (X - iY) / 2.
        np.testing.assert_allclose(target_operator, expected, atol=1e-12)


class TestGivensLadder:
    """Tests for the ladder of Givens rotations."""

    ladder = GivensLadder(num_orbitals=N, num_bits_phase_grad=B_ROT)

    def test_decomposition(self):
        assert_valid_bloq_decomposition(self.ladder)

    def test_bloq_counts(self):
        assert_equivalent_bloq_counts(
            self.ladder, generalizer=[ignore_split_join, ignore_alloc_free]
        )

    def test_cliffords_per_rung(self):
        cost = get_cost_value(self.ladder, QECGatesCost())
        # Per rung: 12 explicit Cliffords, plus 9 b_rot - 8 for each of the two
        # additions into the phase gradient (Qualtran 0.7 counts).
        assert cost.clifford == (N - 1) * (12 + 2 * (9 * B_ROT - 8))

    @pytest.mark.xfail(
        reason=(
            "Qualtran's RealGivensRotationByPhaseGradient differs from the rotation it "
            "documents by a Z on target_i, so the ladder does not equal the identity"
            "with a zero rotation angle. Resource counts are unaffected."
            "Remove once fixed upstream."
        ),
        strict=True,
    )
    def test_zero_angles_is_identity(self):
        ladder = GivensLadder(num_orbitals=2, num_bits_phase_grad=4)
        bb = BloqBuilder()
        system = bb.add_register("system", 2)
        rotations = bb.allocate(ladder.rotations_size)
        phase_grad = bb.allocate(4)
        rotations, phase_grad, system = bb.add(
            ladder, rotations=rotations, phase_grad=phase_grad, system=system
        )
        bb.free(rotations)
        bb.free(phase_grad)
        unitary = bb.finalize(system=system).tensor_contract()

        np.testing.assert_allclose(unitary / unitary[0, 0], np.eye(4), atol=1e-10)


class TestSelectDFTHC:
    """Tests for the SELECT operator."""

    select = SelectDFTHC(num_orbitals=N, num_bits_phase_grad=B_ROT)

    def test_decomposition(self):
        assert_valid_bloq_decomposition(self.select)

    def test_bloq_counts(self):
        assert_equivalent_bloq_counts(
            self.select, generalizer=[ignore_split_join, ignore_alloc_free]
        )


class TestDFTHCInnerBlockEncoding:
    """Tests for the inner block encoding."""

    inner = DFTHCInnerBlockEncoding(
        num_orbitals=N,
        num_ranks=R,
        num_bases=B,
        num_copies=C,
        num_bits_keep_inner=B_K2,
        num_bits_phase_grad=B_ROT,
        num_bits_amp_rotations=S,
        log_block_size_inner=K_2,
        log_block_size_inner_adjoint=K_4,
    )

    def test_decomposition(self):
        assert_valid_bloq_decomposition(self.inner)

    def test_bloq_counts(self):
        assert_equivalent_bloq_counts(
            self.inner, generalizer=[ignore_split_join, ignore_alloc_free]
        )


class TestDFTHCBlockEncoding:
    """Tests for the full DFTHC block encoding."""

    block_encoding = DFTHCBlockEncoding(
        num_orbitals=N,
        num_ranks=R,
        num_bases=B,
        num_copies=C,
        num_bits_keep_inner=B_K2,
        num_bits_keep_outer=B_K1,
        num_bits_phase_grad=B_ROT,
        num_bits_amp_rotations=S,
        log_block_size_inner=K_2,
        log_block_size_inner_adjoint=K_4,
        log_block_size_outer=K_1,
        log_block_size_outer_adjoint=K_5,
        rotation_data=random_rotation_data(R, B, (N - 1) * B_ROT),
        tail_data=random_tail_data(N, (N - 1) * B_ROT),
    )

    def test_decomposition(self):
        assert_valid_bloq_decomposition(self.block_encoding)

    def test_bloq_counts(self):
        assert_equivalent_bloq_counts(
            self.block_encoding, generalizer=[ignore_split_join, ignore_alloc_free]
        )

    def test_from_dfthc_hamiltonian(self, dummy_dfthc_hamiltonian: DFTHCHamiltonian):
        block_encoding_from_fh = DFTHCBlockEncoding.from_dfthc_hamiltonian(
            hamiltonian=dummy_dfthc_hamiltonian,
            num_bits_keep_inner=B_K2,
            num_bits_keep_outer=B_K1,
            num_bits_phase_grad=B_ROT,
            num_bits_amp_rotations=S,
            log_block_size_inner=K_2,
            log_block_size_inner_adjoint=K_4,
            log_block_size_outer=K_1,
            log_block_size_outer_adjoint=K_5,
        )

        # Apart from the data, it is the bloq built from the parameters.
        no_data = {
            "rotation_data": None,
            "tail_data": None,
            "inner_data": None,
            "sign_data": None,
        }
        assert attrs.evolve(self.block_encoding, **no_data) == attrs.evolve(
            block_encoding_from_fh, **no_data
        )

        # Alias sampling with the words and their signs gives the SF coefficients.
        qroam = block_encoding_from_fh.inner.inner_prep.qroam
        coefficients = inner_qrom_coefficients(dummy_dfthc_hamiltonian)
        np.testing.assert_allclose(
            sampled_coefficients(qroam),
            coefficients / np.abs(coefficients).sum(axis=1, keepdims=True),
            rtol=0,
            atol=1 / ((B + 1) * 2**B_K2),
        )
        assert qroam.qrom.inner_data == block_encoding_from_fh.inner_data
        assert qroam.qrom.sign_data == block_encoding_from_fh.sign_data

        assert block_encoding_from_fh.rotation_data is not None
        # The angles the ladder reads from the words are those of the Hamiltonian.
        np.testing.assert_allclose(
            np.exp(1j * decode_rotation_data(block_encoding_from_fh.rotation_data)),
            np.exp(1j * rotation_qrom_angles(dummy_dfthc_hamiltonian)),
            rtol=0,
            atol=2 * np.pi / 2**B_ROT,
        )
        qrom = block_encoding_from_fh.inner.rotation_prep.qrom
        assert qrom.rotation_data == block_encoding_from_fh.rotation_data

        # The tail words hold the eigenvector angles, paired with their G_1 bits.
        words, g_1 = zip(*block_encoding_from_fh.tail_data, strict=True)
        angles, negative = tail_qrom_angles(dummy_dfthc_hamiltonian)
        np.testing.assert_allclose(
            np.exp(1j * decode_rotation_data((words,))[0]),
            np.exp(1j * angles),
            rtol=0,
            atol=2 * np.pi / 2**B_ROT,
        )
        np.testing.assert_array_equal(g_1, negative)
        tail_qrom = block_encoding_from_fh.inner.inner_prep.tail_qrom
        assert tail_qrom.tail_data == block_encoding_from_fh.tail_data


class TestDFTHCWalkOperator:
    """Tests for the DFTHC walk operator."""

    walk = DFTHCWalkOperator(
        num_orbitals=N,
        num_ranks=R,
        num_bases=B,
        num_copies=C,
        num_bits_keep_inner=B_K2,
        num_bits_keep_outer=B_K1,
        num_bits_phase_grad=B_ROT,
        num_bits_amp_rotations=S,
        log_block_size_inner=K_2,
        log_block_size_inner_adjoint=K_4,
        log_block_size_outer=K_1,
        log_block_size_outer_adjoint=K_5,
        rotation_data=random_rotation_data(R, B, (N - 1) * B_ROT),
        tail_data=random_tail_data(N, (N - 1) * B_ROT),
    )

    def test_decomposition(self):
        assert_valid_bloq_decomposition(self.walk)

    def test_bloq_counts(self):
        assert_equivalent_bloq_counts(
            self.walk, generalizer=[ignore_split_join, ignore_alloc_free]
        )

    def test_from_dfthc_hamiltonian(self, dummy_dfthc_hamiltonian: DFTHCHamiltonian):
        walk_from_fh = DFTHCWalkOperator.from_dfthc_hamiltonian(
            hamiltonian=dummy_dfthc_hamiltonian,
            num_bits_keep_inner=B_K2,
            num_bits_keep_outer=B_K1,
            num_bits_phase_grad=B_ROT,
            num_bits_amp_rotations=S,
            log_block_size_inner=K_2,
            log_block_size_inner_adjoint=K_4,
            log_block_size_outer=K_1,
            log_block_size_outer_adjoint=K_5,
        )

        # Apart from the data, it is the bloq built from the parameters.
        no_data = {
            "rotation_data": None,
            "tail_data": None,
            "inner_data": None,
            "sign_data": None,
        }
        assert attrs.evolve(self.walk, **no_data) == attrs.evolve(
            walk_from_fh, **no_data
        )

        # Alias sampling with the words and their signs gives the SF coefficients.
        qroam = walk_from_fh.block_encoding.inner.inner_prep.qroam
        coefficients = inner_qrom_coefficients(dummy_dfthc_hamiltonian)
        np.testing.assert_allclose(
            sampled_coefficients(qroam),
            coefficients / np.abs(coefficients).sum(axis=1, keepdims=True),
            rtol=0,
            atol=1 / ((B + 1) * 2**B_K2),
        )
        assert qroam.qrom.inner_data == walk_from_fh.inner_data
        assert qroam.qrom.sign_data == walk_from_fh.sign_data

        assert walk_from_fh.rotation_data is not None
        # The angles the ladder reads from the words are those of the Hamiltonian.
        np.testing.assert_allclose(
            np.exp(1j * decode_rotation_data(walk_from_fh.rotation_data)),
            np.exp(1j * rotation_qrom_angles(dummy_dfthc_hamiltonian)),
            rtol=0,
            atol=2 * np.pi / 2**B_ROT,
        )
        qrom = walk_from_fh.block_encoding.inner.rotation_prep.qrom
        assert qrom.rotation_data == walk_from_fh.rotation_data

        # The tail words hold the eigenvector angles, paired with their G_1 bits.
        words, g_1 = zip(*walk_from_fh.tail_data, strict=True)
        angles, negative = tail_qrom_angles(dummy_dfthc_hamiltonian)
        np.testing.assert_allclose(
            np.exp(1j * decode_rotation_data((words,))[0]),
            np.exp(1j * angles),
            rtol=0,
            atol=2 * np.pi / 2**B_ROT,
        )
        np.testing.assert_array_equal(g_1, negative)
        tail_qrom = walk_from_fh.block_encoding.inner.inner_prep.tail_qrom
        assert tail_qrom.tail_data == walk_from_fh.tail_data


class TestQuantiseAngles:
    """Tests for converting angles to integers."""

    def test_givens_angles_generation(self):
        # Un-normalised vectors with mixed signs, including the last coordinate.
        vectors = 3.7 * np.random.default_rng(0).normal(size=(2, 4, 3))
        angles = givens_angles(vectors)

        assert angles.shape == (2, 4, 2)
        # Invert the Givens ladder: u_j = (prod_{k<j} sin t_k) cos t_j for j < N - 1,
        # and u_{N-1} = prod_k sin t_k.
        sines = np.cumprod(np.sin(angles), axis=-1)
        prefix = np.concatenate([np.ones((2, 4, 1)), sines[..., :-1]], axis=-1)
        rebuilt = np.concatenate([prefix * np.cos(angles), sines[..., -1:]], axis=-1)
        expected = vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)
        np.testing.assert_allclose(rebuilt, expected, atol=1e-12)

    @pytest.mark.parametrize(
        ("angles", "expected"),
        [
            ([0.0, RESOLUTION, 2 * RESOLUTION, 4 * RESOLUTION], [0, 1, 2, 4]),
            ([-2 * RESOLUTION, 8 * RESOLUTION, 9 * RESOLUTION], [6, 0, 1]),
            ([0.4 * RESOLUTION, 0.6 * RESOLUTION], [0, 1]),
            ([7.9 * RESOLUTION], [0]),
        ],
        ids=[
            "exact_multiples",
            "negative_and_beyond_2pi",
            "rounds_to_nearest",
            "rounds_up_and_wraps_to_zero",
        ],
    )
    def test_values(self, angles: list[float], expected: list[int]):
        np.testing.assert_array_equal(quantise_angles(angles, N_SMALL), expected)

    @pytest.mark.parametrize("num_bits", [6, 12, 15])
    def test_accuracy(self, num_bits: int):
        rng = np.random.default_rng(0)
        angles = rng.uniform(-4 * np.pi, 4 * np.pi, size=(5, 4, 7))

        integers = quantise_angles(angles, num_bits)

        assert integers.shape == angles.shape
        assert integers.min() >= 0
        assert integers.max() < 2 ** (num_bits - 1)
        # Decoding gives back the angle up to half the angle of the least significant
        # bit, compared as points on the circle since rotations are periodic in 2 pi.
        decoded = integers * 4 * np.pi / 2**num_bits
        np.testing.assert_allclose(
            np.exp(1j * decoded),
            np.exp(1j * angles),
            rtol=0,
            atol=2 * np.pi / 2**num_bits,
        )

    @pytest.mark.parametrize("num_bits", [1, 0, -3])
    def test_too_few_bits(self, num_bits: int):
        with pytest.raises(ValueError, match="must be at least 2"):
            quantise_angles([0.5], num_bits)

    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_non_finite_angles(self, value: float):
        with pytest.raises(ValueError, match="must be finite"):
            quantise_angles([0.5, value], N_SMALL)

    @pytest.mark.parametrize(
        ("angle", "neighbours"),
        [(0.3, {0, 1}), (1.7, {1, 2}), (7.9, {7, 0})],  # In steps; 8 wraps to 0.
        ids=["lower_half", "upper_half", "rounds_up_and_wraps_to_zero"],
    )
    def test_unbiased_rounding_neighbours(self, angle: float, neighbours: set[int]):
        rng = np.random.default_rng(0)
        integers = quantise_angles(np.full(1000, angle * RESOLUTION), N_SMALL, rng=rng)
        assert set(integers.tolist()) == neighbours

    def test_unbiased_rounding_mean(self):
        angles = np.full(100_000, 0.3 * RESOLUTION)
        integers = quantise_angles(angles, N_SMALL, rng=np.random.default_rng(0))
        # Rounded up with probability equal to the fractional part, 0.3.
        np.testing.assert_allclose(integers.mean(), 0.3, atol=0.01)

    def test_unbiased_rounding_reproducible(self):
        angles = np.random.default_rng(1).uniform(0, 2 * np.pi, size=100)
        first = quantise_angles(angles, N_SMALL, rng=np.random.default_rng(5))
        second = quantise_angles(angles, N_SMALL, rng=np.random.default_rng(5))
        np.testing.assert_array_equal(first, second)


class TestPackIntegers:
    """Tests for packing integers into a single word."""

    def test_order(self):
        # The first integer takes the most significant bits.
        assert pack_integers([1, 2, 3], num_bits=4) == 0x123

    def test_large_words(self):
        # FeMoco-54 sizes, where each word exceeds 64 bits.
        num_bits, num_angles = 15, 53
        rng = np.random.default_rng(1)
        integers = rng.integers(0, 2 ** (num_bits - 1), size=num_angles)

        word = pack_integers(integers, num_bits)

        assert isinstance(word, int)
        mask = 2**num_bits - 1
        unpacked = [
            (word >> (num_bits * (num_angles - 1 - k))) & mask
            for k in range(num_angles)
        ]
        np.testing.assert_array_equal(unpacked, integers)

    @pytest.mark.parametrize(
        "integers",
        [[16], [-1], np.array([16]), np.array([-1])],
        ids=["list_too_large", "list_negative", "array_too_large", "array_negative"],
    )
    def test_entries_must_fit(self, integers: ArrayLike):
        with pytest.raises(ValueError, match="to fit in the word"):
            pack_integers(integers, num_bits=4)
