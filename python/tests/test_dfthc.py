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


@pytest.fixture
def dummy_dfthc_hamiltonian() -> DFTHCHamiltonian:
    """Return a dummy DFTHCHamiltonian for use in tests."""
    return DFTHCHamiltonian(
        num_orbitals=N,
        num_ranks=R,
        num_bases=B,
        num_copies=C,
        unit_vectors=np.ones((R, B, N)),  # Non-zero, so unit-vector validation passes.
        weight_vectors=np.ones((R, B, C)),
        bliss_matrix=np.zeros((N, N)),
        factorised_one_body=np.eye(N),
        const=0.0,
        num_electrons=N,
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

    def test_from_dfthc_hamiltonian_equivalence(
        self, dummy_dfthc_hamiltonian: DFTHCHamiltonian
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
        # The rotation data is tested separately in `TestRotationData`.
        assert self.block_encoding == attrs.evolve(
            block_encoding_from_fh, rotation_data=None
        )

    def test_decomposition(self):
        assert_valid_bloq_decomposition(self.block_encoding)

    def test_bloq_counts(self):
        assert_equivalent_bloq_counts(
            self.block_encoding, generalizer=[ignore_split_join, ignore_alloc_free]
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

    def test_from_dfthc_hamiltonian_equivalence(
        self, dummy_dfthc_hamiltonian: DFTHCHamiltonian
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
        # The rotation data is tested separately in `TestRotationData`.
        assert self.walk == attrs.evolve(walk_from_fh, rotation_data=None)

    def test_decomposition(self):
        assert_valid_bloq_decomposition(self.walk)

    def test_bloq_counts(self):
        assert_equivalent_bloq_counts(
            self.walk, generalizer=[ignore_split_join, ignore_alloc_free]
        )


class TestRotationData:
    """Tests for loading the Givens rotation words into the rotation QROM."""

    # A small system, so the QROM can be simulated.
    norb, nranks, nbases, ncopies, brot = 4, 3, 2, 2, 6

    @pytest.fixture
    def hamiltonian(self) -> DFTHCHamiltonian:
        rng = np.random.default_rng(7)
        return DFTHCHamiltonian(
            num_orbitals=self.norb,
            num_ranks=self.nranks,
            num_bases=self.nbases,
            num_copies=self.ncopies,
            unit_vectors=rng.normal(size=(self.nranks, self.nbases, self.norb)),
            weight_vectors=np.ones((self.nranks, self.nbases, self.ncopies)),
            bliss_matrix=np.zeros((self.norb, self.norb)),
            factorised_one_body=np.eye(self.norb),
            const=0.0,
            num_electrons=self.norb,
        )

    @pytest.fixture(params=[DFTHCBlockEncoding, DFTHCWalkOperator])
    def bloq(
        self, request: pytest.FixtureRequest, hamiltonian: DFTHCHamiltonian
    ) -> DFTHCBlockEncoding | DFTHCWalkOperator:
        return request.param.from_dfthc_hamiltonian(
            hamiltonian=hamiltonian,
            num_bits_keep_inner=3,
            num_bits_keep_outer=3,
            num_bits_phase_grad=self.brot,
            num_bits_amp_rotations=3,
            log_block_size_inner=1,
            log_block_size_inner_adjoint=1,
            log_block_size_outer=1,
            log_block_size_outer_adjoint=1,
        )

    def test_factory_encodes_rotation_angles(
        self, hamiltonian: DFTHCHamiltonian, bloq: DFTHCBlockEncoding
    ):
        angles = hamiltonian.rotation_angles()
        data = bloq.rotation_data

        assert data is not None
        assert len(data) == self.nranks
        assert all(len(row) == self.nbases for row in data)
        mask = 2**self.brot - 1
        for r in range(self.nranks):
            for b in range(self.nbases):
                # The first angle is in the most significant bits.
                q = [
                    (data[r][b] >> (self.brot * (self.norb - 2 - k))) & mask
                    for k in range(self.norb - 1)
                ]
                decoded = np.array(q) * 4 * np.pi / 2**self.brot
                error = np.abs(np.angle(np.exp(1j * (decoded - angles[r, b]))))
                assert error.max() <= 2 * np.pi / 2**self.brot + 1e-12

    def test_data_reaches_qrom(self, bloq: DFTHCBlockEncoding | DFTHCWalkOperator):
        block_encoding = (
            bloq.block_encoding if isinstance(bloq, DFTHCWalkOperator) else bloq
        )

        qrom = block_encoding.inner.rotation_prep.qrom

        assert bloq.rotation_data is not None
        assert qrom.rotation_data == bloq.rotation_data

    def test_qrom_loads_table(self):
        data = tuple(
            tuple(1000 * (r + 1) + b for b in range(self.nbases))
            for r in range(self.nranks)
        )
        qrom = RotationQROM(
            num_orbitals=self.norb,
            num_ranks=self.nranks,
            num_bases=self.nbases,
            num_bits_phase_grad=self.brot,
            num_controls=1,
            rotation_data=data,
        )
        cbloq = qrom.decompose_bloq()
        names = [reg.name for reg in qrom.signature.rights()]

        def lookup(**registers: int) -> dict[str, ClassicalValT]:
            return dict(zip(names, cbloq.call_classically(**registers), strict=True))

        for r in range(self.nranks):
            for b in range(self.nbases):
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
            ctrl=1, inner_index=self.nbases, rank=0, rotations=0, non_identity=0
        )
        assert identity["rotations"] == 0
        assert identity["non_identity"] == 1

    @pytest.mark.parametrize(
        ("data", "match"),
        [
            (((0, 0),) * (nranks - 1), "shape"),  # Too few ranks.
            (((0,),) * nranks, "shape"),  # Too few bases.
            (((2 ** ((norb - 1) * brot), 0),) * nranks, "fit in"),  # Too wide.
            (((-1, 0),) * nranks, "fit in"),  # Negative.
        ],
    )
    def test_qrom_rotation_data_validation(self, data: tuple, match: str):
        with pytest.raises(ValueError, match=match):
            RotationQROM(
                num_orbitals=self.norb,
                num_ranks=self.nranks,
                num_bases=self.nbases,
                num_bits_phase_grad=self.brot,
                rotation_data=data,
            )
