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

import numpy as np
from qualtran.resource_counting.generalizers import ignore_alloc_free, ignore_split_join
from qualtran.testing import (
    assert_equivalent_bloq_counts,
    assert_valid_bloq_decomposition,
)

from quiche.core.electronic import FactorisedHamiltonian
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

# FeMoco-54 parameters, from G. H. Low et al., 
# "Fast Quantum Simulation of Electronic
# Structure by Spectral Amplification", 
# Phys. Rev. X, vol. 15, no. 4, p. 041016
N = 54
R, B, C = 10, 27, 27
B_ROT = B_K1 = B_K2 = 15
K_1 = K_4 = 2
K_2 = K_5 = 4
S = 7

dummy_factorised_hamiltonian = FactorisedHamiltonian(
    N=N,
    R=R,
    B=B,
    C=C,
    U=np.ones((R, B, N)),  # non-zero, so unit-vector validation (if added) passes
    W=np.ones((R, B, C)),
    bliss_matrix=np.zeros((N, N)),
    h1=np.eye(N),
    const=0.0,
    electrons=N,
    job_id=0,
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

    block_encoding_from_fh = DFTHCBlockEncoding.FromFactorisedHamiltonian(
        hamiltonian=dummy_factorised_hamiltonian,
        num_bits_keep_inner=B_K2,
        num_bits_keep_outer=B_K1,
        num_bits_phase_grad=B_ROT,
        num_bits_amp_rotations=S,
        log_block_size_inner=K_2,
        log_block_size_inner_adjoint=K_4,
        log_block_size_outer=K_1,
        log_block_size_outer_adjoint=K_5,
    )

    def test_decomposition(self):
        assert_valid_bloq_decomposition(self.block_encoding)
        assert_valid_bloq_decomposition(self.block_encoding_from_fh)

    def test_bloq_counts(self):
        assert_equivalent_bloq_counts(
            self.block_encoding, generalizer=[ignore_split_join, ignore_alloc_free]
        )
        assert_equivalent_bloq_counts(
            self.block_encoding_from_fh,
            generalizer=[ignore_split_join, ignore_alloc_free],
        )


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

    walk_from_fh = DFTHCBlockEncoding.FromFactorisedHamiltonian(
        hamiltonian=dummy_factorised_hamiltonian,
        num_bits_keep_inner=B_K2,
        num_bits_keep_outer=B_K1,
        num_bits_phase_grad=B_ROT,
        num_bits_amp_rotations=S,
        log_block_size_inner=K_2,
        log_block_size_inner_adjoint=K_4,
        log_block_size_outer=K_1,
        log_block_size_outer_adjoint=K_5,
    )

    def test_decomposition(self):
        assert_valid_bloq_decomposition(self.walk)
        assert_valid_bloq_decomposition(self.walk_from_fh)

    def test_bloq_counts(self):
        assert_equivalent_bloq_counts(
            self.walk, generalizer=[ignore_split_join, ignore_alloc_free]
        )
        assert_equivalent_bloq_counts(
            self.walk_from_fh, generalizer=[ignore_split_join, ignore_alloc_free]
        )


class TestEquivalence:
    kwargs = {
        "num_bits_keep_inner": B_K2,
        "num_bits_keep_outer": B_K1,
        "num_bits_phase_grad": B_ROT,
        "num_bits_amp_rotations": S,
        "log_block_size_inner": K_2,
        "log_block_size_inner_adjoint": K_4,
        "log_block_size_outer": K_1,
        "log_block_size_outer_adjoint": K_5,
    }

    def test_from_hamiltonian_equivalencet(self):

        assert DFTHCBlockEncoding.FromFactorisedHamiltonian(
            dummy_factorised_hamiltonian, **self.kwargs
        ) == DFTHCBlockEncoding(
            num_orbitals=N, num_ranks=R, num_bases=B, num_copies=C, **self.kwargs
        )
