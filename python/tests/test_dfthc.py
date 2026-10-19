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

import attrs
import numpy as np
import pytest
from qualtran.resource_counting.generalizers import ignore_alloc_free, ignore_split_join
from qualtran.simulation.classical_sim import ClassicalValT
from qualtran.testing import (
    assert_equivalent_bloq_counts,
    assert_valid_bloq_decomposition,
)

from quiche.core._encodings import angles_to_integer_representation, pack_into_word
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
from quiche.resources.bloqs.dfthc.qrom import RotationQROM

# FeMoco-54 parameters, from G. H. Low et al., "Fast Quantum Simulation of Electronic
# Structure by Spectral Amplification", Phys. Rev. X, vol. 15, no. 4, p. 041016
N = 54
R, B, C = 10, 27, 27
B_ROT = B_K1 = B_K2 = 15
K_1 = K_4 = 2
K_2 = K_5 = 4
S = 7
N_SMALL, R_SMALL, B_SMALL, B_ROT_SMALL = 4, 3, 2, 6

@pytest.fixture
def dummy_dfthc_hamiltonian() -> DFTHCHamiltonian:
    """Return a dummy DFTHCHamiltonian for use in tests."""
    return DFTHCHamiltonian(
        num_orbitals=N,
        num_ranks=R,
        num_bases=B,
        num_copies=C,
        unit_vectors=np.random.default_rng(0).normal(size=(R, B, N)),
        weight_vectors=np.ones((R, B, C)),
        bliss_matrix=np.zeros((N, N)),
        factorised_one_body=np.eye(N),
        const=0.0,
        num_electrons=N,
    )

@pytest.fixture
def dummy_rotation_data(
    dummy_dfthc_hamiltonian: DFTHCHamiltonian,
) -> tuple[tuple[int, ...], ...]:
    """Return the rotation words of the dummy DFTHCHamiltonian, with B_ROT bits."""
    integers = angles_to_integer_representation(
        dummy_dfthc_hamiltonian.rotation_angles(), B_ROT
    )
    return tuple(
        tuple(int(word) for word in row) for row in pack_into_word(integers, B_ROT)
    )


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


class TestMajorana:
    """Tests for the Majorana operator."""

    majorana = MajoranaDFTHC()

    def test_decomposition(self):
        assert_valid_bloq_decomposition(self.majorana)

    def test_bloq_counts(self):
        assert_equivalent_bloq_counts(
            self.majorana, generalizer=[ignore_split_join, ignore_alloc_free]
        )


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
    )

    def test_from_dfthc_hamiltonian(
        self,
        dummy_dfthc_hamiltonian: DFTHCHamiltonian,
        dummy_rotation_data: tuple[tuple[int, ...], ...],
    ):
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

        # Apart from the rotation data, it is the bloq built from the parameters.
        assert self.block_encoding == attrs.evolve(
            block_encoding_from_fh, rotation_data=None
        )
        np.testing.assert_array_equal(
            block_encoding_from_fh.rotation_data, dummy_rotation_data
        )
        # The rotation data reaches the QROM that loads it.
        qrom = block_encoding_from_fh.inner.rotation_prep.qrom
        assert qrom.rotation_data == block_encoding_from_fh.rotation_data


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
    )

    def test_from_dfthc_hamiltonian(
        self,
        dummy_dfthc_hamiltonian: DFTHCHamiltonian,
        dummy_rotation_data: tuple[tuple[int, ...], ...],
    ):
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

        # Apart from the rotation data, it is the bloq built from the parameters.
        assert self.walk == attrs.evolve(walk_from_fh, rotation_data=None)
        np.testing.assert_array_equal(walk_from_fh.rotation_data, dummy_rotation_data)
        qrom = walk_from_fh.block_encoding.inner.rotation_prep.qrom
        assert qrom.rotation_data == walk_from_fh.rotation_data
