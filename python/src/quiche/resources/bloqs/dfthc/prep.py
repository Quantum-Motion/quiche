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

"""State preparation routines for the DFTHC Hamiltonian block encoding."""

from functools import cached_property

import attrs
from qualtran import (
    Bloq,
    BloqBuilder,
    QAny,
    QBit,
    Register,
    Side,
    Signature,
    SoquetT,
)
from qualtran.bloqs.arithmetic import LessThanEqual
from qualtran.bloqs.basic_gates import CZ, CSwap, Hadamard, OnEach, XGate
from qualtran.bloqs.data_loading import QROAMClean
from qualtran.bloqs.state_preparation import PrepareUniformSuperposition
from qualtran.resource_counting import (
    BloqCountDictT,
    SympySymbolAllocator,
)
from qualtran.symbolics import bit_length

from quiche.resources.bloqs.dfthc.qrom import (
    InnerQROAM,
    InnerQROMTail,
    RotationQROM,
)
from quiche.resources.bloqs.dfthc.utils import extract_soqs


@attrs.frozen
class PrepareUniformSuperpositionDFTHC(Bloq):
    """
    Uniform superposition preparation with success flag.

    Placeholder for the amplitude-amplified equal superposition preparation of
    [Low2025]_.

    Parameters
    ----------
    size : int
        Number of terms in the superposition.
    num_bits_amp_rotations : int
        Number of bits used for amplitude amplification rotations during equal state
        preparation, :math:`s` in [Low2025]_.
    num_bits_phase_grad : int
        Total number of bits for persistent phase gradient register, :math:`b_{rot}` in
        [Low2025]_.

    Registers
    ---------
    target : QAny
        Register to apply the uniform superposition preparation to.
    success : QBit
        Success flag for the uniform state preparation.
    phase_grad : QAny
        Phase gradient register used throughout the circuit.
    amp_rotation : QBit
        Qubit rotated in amplitude amplification step of equal superposition
        preparation.

    References
    ----------
    .. [Low2025] G. H. Low et al., "Fast Quantum Simulation of Electronic Structure by
       Spectral Amplification", Phys. Rev. X, vol. 15, no. 4, p. 041016, Oct. 2025.

    """

    size: int

    num_bits_amp_rotations: int
    num_bits_phase_grad: int

    @property
    def signature(self) -> Signature:
        """Define input and/or output registers of the bloq."""
        return Signature(
            [
                Register("target", QAny(bit_length(self.size - 1))),
                Register("success", QBit()),
                Register("phase_grad", QAny(self.num_bits_phase_grad)),
                Register("amp_rotation", QBit()),
            ]
        )

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        **soqs: SoquetT,
    ) -> dict[str, SoquetT]:
        """Implement bloq decomposition into sub-bloqs."""
        soqs["target"] = bb.add(
            PrepareUniformSuperposition(self.size),
            target=soqs["target"],
        )
        soqs["success"] = bb.add(XGate(), q=soqs["success"])
        return soqs


@attrs.frozen
class OuterPrepareDFTHC(Bloq):
    """
    Outer state preparation for the DFTHC block encoding.

    Parameters
    ----------
    num_orbitals : int
        Number of spatial orbitals, :math:`N` in [Low2025]_.
    num_ranks : int
        DFTHC rank, :math:`R` in [Low2025]_.
    num_copies : int
        Number of copies per rank component, :math:`C` in [Low2025]_.

    num_bits_keep : int
        Number of bits for coherent alias sampling keep probability, :math:`b_{k1}` in
        [Low2025]_.
    num_bits_phase_grad : int
        Total number of bits for persistent phase gradient register, :math:`b_{rot}` in
        [Low2025]_.
    num_bits_amp_rotations : int
        Number of bits used for amplitude amplification rotations during equal state
        preparation, :math:`s` in [Low2025]_.

    log_block_size : int
        Log of the block size for QROAM, :math:`k_1` in [Low2025]_.
    log_block_size_adjoint : int
        Log of the block size for QROAM adjoint, :math:`k_5` in [Low2025]_.

    Registers
    ---------
    outer_index : QAny
        Register representing the outer index, :math:`x_o`.

    amp_rotation : QBit
        Qubit rotated in amplitude amplification step of equal superposition
        preparation.
    succ_outer : QBit
        Success flag for the uniform state preparation over the outer index,
        :math:`succ_{x_o}` in [Low2025]_.

    phase_grad : QAny
        Phase gradient register used throughout the circuit.
    alt : QAny, RIGHT
        Alias indices for coherent alias sampling.
    keep : QAny, RIGHT
        Keep thresholds for coherent alias sampling.
    keep_compare_outer : QAny
        Uniform superposition register compared against `keep` in the outer alias
        sampling inequality test.
    compare : QBit, RIGHT
        Result of the alias sampling inequality test.

    junk_target0_ : QAny, RIGHT
        Unselected `alt` words from the other QROAM blocks.
    junk_target1_ : QAny, RIGHT
        Unselected `keep` words from the other QROAM blocks.

    References
    ----------
    .. [Low2025] G. H. Low et al., "Fast Quantum Simulation of Electronic Structure by
       Spectral Amplification", Phys. Rev. X, vol. 15, no. 4, p. 041016, Oct. 2025.

    """

    num_orbitals: int
    num_ranks: int
    num_copies: int

    num_bits_keep: int
    num_bits_phase_grad: int
    num_bits_amp_rotations: int

    log_block_size: int
    log_block_size_adjoint: int

    def __attrs_post_init__(self) -> None:
        """Validate inputs."""
        if self.num_bits_amp_rotations > self.num_bits_phase_grad:
            err_msg = (
                "Phase gradient register not large enough for number of rotation bits."
            )
            raise ValueError(err_msg)

    @property
    def num_outer(self) -> int:
        """Number of terms for the outer index, :math:`N + RC`."""
        return self.num_orbitals + self.num_ranks * self.num_copies

    @property
    def num_bits_outer(self) -> int:
        """Number of bits needed to represent the outer index."""
        return bit_length(self.num_outer - 1)

    @property
    def signature(self) -> Signature:
        """Define input and/or output registers of the bloq."""
        return Signature(
            [
                Register("outer_index", QAny(self.num_bits_outer)),
                Register("amp_rotation", QBit()),
                Register("succ_outer", QBit()),
                Register("phase_grad", QAny(self.num_bits_phase_grad)),
                Register("alt", QAny(self.num_bits_outer), side=Side.RIGHT),
                Register("keep", QAny(self.num_bits_keep), side=Side.RIGHT),
                Register("keep_compare_outer", QAny(self.num_bits_keep)),
                Register("compare", QBit(), side=Side.RIGHT),
                *self.qroam.junk_registers,
            ]
        )

    @property
    def uniform_prep(self) -> PrepareUniformSuperpositionDFTHC:
        """Get the equal superposition preparation over the outer indices."""
        return PrepareUniformSuperpositionDFTHC(
            size=self.num_outer,
            num_bits_phase_grad=self.num_bits_phase_grad,
            num_bits_amp_rotations=self.num_bits_amp_rotations,
        )

    @property
    def hadamards(self) -> Bloq:
        """Get the Hadamards for preparing the alias sampling comparison."""
        return OnEach(self.num_bits_keep, Hadamard())

    @property
    def qroam(self) -> QROAMClean:
        """Get the QROAM loading the alias sampling data."""
        return QROAMClean.build_from_bitsize(
            (self.num_outer,),
            (self.num_bits_outer, self.num_bits_keep),
            log_block_sizes=(self.log_block_size,),
        )

    @property
    def inequality_check(self) -> LessThanEqual:
        """Get the alias sampling inequality test."""
        return LessThanEqual(self.num_bits_keep, self.num_bits_keep)

    @property
    def cswap(self) -> CSwap:
        """Get the controlled swap between `outer_index` and `alt`."""
        return CSwap(self.num_bits_outer)

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Build call graph for OuterPrepareDFTHC."""
        return {
            self.uniform_prep: 1,
            self.qroam: 1,
            self.inequality_check: 1,
            self.cswap: 1,
            self.hadamards: 1,
        }

    def adjoint(self) -> Bloq:
        """Get the unpreparation that undoes this state preparation."""
        return OuterPrepareDFTHCAdjoint(self)

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        **soqs: SoquetT,
    ) -> dict[str, SoquetT]:
        """Implement bloq decomposition into sub-bloqs."""
        (
            soqs["outer_index"],
            soqs["succ_outer"],
            soqs["phase_grad"],
            soqs["amp_rotation"],
        ) = bb.add(
            self.uniform_prep,
            target=soqs["outer_index"],
            success=soqs["succ_outer"],
            phase_grad=soqs["phase_grad"],
            amp_rotation=soqs["amp_rotation"],
        )

        soqs |= bb.add_d(self.qroam, selection=soqs.pop("outer_index"))
        soqs["outer_index"] = soqs.pop("selection")
        soqs["alt"] = soqs.pop("target0_")
        soqs["keep"] = soqs.pop("target1_")

        soqs["keep_compare_outer"] = bb.add(
            self.hadamards,
            q=soqs["keep_compare_outer"],
        )

        soqs["keep"], soqs["keep_compare_outer"], soqs["compare"] = bb.add(
            self.inequality_check,
            x=soqs["keep"],
            y=soqs["keep_compare_outer"],
            target=bb.allocate(1),
        )

        soqs["compare"], soqs["outer_index"], soqs["alt"] = bb.add(
            self.cswap,
            ctrl=soqs["compare"],
            x=soqs["outer_index"],
            y=soqs["alt"],
        )

        return soqs


@attrs.frozen
class OuterPrepareDFTHCAdjoint(Bloq):
    """
    Adjoint of `OuterPrepareDFTHC`.

    Parameters
    ----------
    prepare : OuterPrepareDFTHC
        State preparation undone by this bloq.

    Registers
    ---------
    `prepare`'s registers, with left- and right-registers swapped.

    References
    ----------
    .. [Low2025] G. H. Low et al., "Fast Quantum Simulation of Electronic Structure by
       Spectral Amplification", Phys. Rev. X, vol. 15, no. 4, p. 041016, Oct. 2025.

    """

    prepare: OuterPrepareDFTHC

    @cached_property
    def signature(self) -> Signature:
        """Define input and/or output registers of the bloq."""
        return self.prepare.signature.adjoint()

    @property
    def qroam_adjoint(self) -> Bloq:
        """Get the QROAM adjoint with the required block sizes."""
        return self.prepare.qroam.adjoint().with_log_block_sizes(
            (self.prepare.log_block_size_adjoint,)
        )

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Build call graph for OuterPrepareDFTHCAdjoint."""
        return {
            self.prepare.uniform_prep.adjoint(): 1,
            self.qroam_adjoint: 1,
            self.prepare.inequality_check.adjoint(): 1,
            self.prepare.cswap.adjoint(): 1,
            self.prepare.hadamards: 1,
        }

    def adjoint(self) -> Bloq:
        """Get the state preparation undone by this unpreparation."""
        return self.prepare


@attrs.frozen
class InnerPrepareDFTHC(Bloq):
    """
    Inner state preparation for the DFTHC block encoding.

    Parameters
    ----------
    num_orbitals : int
        Number of spatial orbitals, :math:`N` in [Low2025]_.
    num_ranks : int
        DFTHC rank, :math:`R` in [Low2025]_.
    num_bases : int
        Number of bases per rank component, :math:`B` in [Low2025]_.
    num_copies : int
        Number of copies per rank component, :math:`C` in [Low2025]_.

    num_bits_keep : int
        Number of bits for coherent alias sampling keep probability, :math:`b_{k2}` in
        [Low2025]_.
    num_bits_phase_grad : int
        Total number of bits for persistent phase gradient register, :math:`b_{rot}` in
        [Low2025]_.
    num_bits_amp_rotations : int
        Number of bits used for amplitude amplification rotations during equal state
        preparation, :math:`s` in [Low2025]_.

    log_block_size : int
        Log of the block size for QROAM, :math:`k_2` in [Low2025]_.
    log_block_size_adjoint : int
        Log of the block size for QROAM adjoint, :math:`k_4` in [Low2025]_.

    Registers
    ---------
    outer_index : QAny
        Register representing the outer index, :math:`x_o`.
    inner_index : QAny
        Register representing the inner index, :math:`b`.

    succ_inner : QBit
        Success flag for inner uniform state preparation.
    amp_rotation : QBit
        Qubit rotated in amplitude amplification step of equal superposition
        preparation.
    phase_grad : QAny
        Phase gradient register used throughout the circuit.
    keep_compare_inner : QAny
        Uniform superposition register compared against `keep` in the inner alias
        sampling inequality test.

    alt : QAny, RIGHT
        Alias indices for coherent alias sampling.
    keep : QAny, RIGHT
        Keep thresholds for coherent alias sampling.
    sign : QBit, RIGHT
        Flag for the alias term's sign.
    compare : QBit, RIGHT
        Result of the alias sampling inequality test.

    G_0 : QBit, RIGHT
        Qubit selecting between SF and :math:`D_1/Q_1`.
    G_1 : QBit, RIGHT
        Qubit selecting between :math:`D_1` and :math:`Q_1`.
    rank : QAny, RIGHT
        Register representing the rank, :math:`r`.

    rotations : QAny, RIGHT
        Givens rotation angles for the basis rotation.

    References
    ----------
    .. [Low2025] G. H. Low et al., "Fast Quantum Simulation of Electronic Structure by
       Spectral Amplification", Phys. Rev. X, vol. 15, no. 4, p. 041016, Oct. 2025.

    """

    num_orbitals: int
    num_ranks: int
    num_bases: int
    num_copies: int

    num_bits_keep: int
    num_bits_phase_grad: int
    num_bits_amp_rotations: int

    log_block_size: int
    log_block_size_adjoint: int

    def __attrs_post_init__(self) -> None:
        """Validate inputs."""
        if self.num_bits_amp_rotations > self.num_bits_phase_grad:
            err_msg = (
                "Phase gradient register not large enough for number of rotation bits."
            )
            raise ValueError(err_msg)

    @property
    def num_outer(self) -> int:
        """Number of terms for the outer index, :math:`N + RC`."""
        return self.num_orbitals + self.num_ranks * self.num_copies

    @property
    def num_bits_outer(self) -> int:
        """Number of bits needed to represent the outer index."""
        return bit_length(self.num_outer - 1)

    @property
    def num_inner(self) -> int:
        """Number of terms for the inner index, :math:`B + 1`."""
        return self.num_bases + 1

    @property
    def num_bits_inner(self) -> int:
        """Number of bits needed to represent the inner index."""
        return bit_length(self.num_inner - 1)

    @property
    def num_bits_rank(self) -> int:
        """Number of bits needed to represent the rank, :math:`r`."""
        return bit_length(self.num_ranks - 1)

    @property
    def rotations_size(self) -> int:
        """Rotation register size, :math:`(N - 1) b_{rot}`."""
        return (self.num_orbitals - 1) * self.num_bits_phase_grad

    @cached_property
    def index_registers(self) -> tuple[Register, ...]:
        """Registers used for indexing LCU terms."""
        return (
            Register("outer_index", QAny(self.num_bits_outer)),
            Register("inner_index", QAny(self.num_bits_inner)),
        )

    @cached_property
    def misc_registers(self) -> tuple[Register, ...]:
        """Other registers used in the state preparation."""
        return (
            Register("succ_inner", QBit()),
            Register("amp_rotation", QBit()),
            Register("phase_grad", QAny(self.num_bits_phase_grad)),
            Register("keep_compare_inner", QAny(self.num_bits_keep)),
        )

    @cached_property
    def alias_registers(self) -> tuple[Register, ...]:
        """Registers used for coherent alias sampling."""
        return (
            Register("alt", QAny(self.num_bits_inner), side=Side.RIGHT),
            Register("keep", QAny(self.num_bits_keep), side=Side.RIGHT),
            Register("sign", QBit(), side=Side.RIGHT),
            Register("compare", QBit(), side=Side.RIGHT),
        )

    @property
    def signature(self) -> Signature:
        """Define input and/or output registers of the bloq."""
        return Signature(
            [
                *self.index_registers,
                *self.alias_registers,
                *self.misc_registers,
                *self.qroam.selector_registers,
                Register("rotations", QAny(self.rotations_size), side=Side.RIGHT),
            ]
        )

    @property
    def uniform_prep(self) -> PrepareUniformSuperpositionDFTHC:
        """Get the equal superposition preparation over the inner indices."""
        return PrepareUniformSuperpositionDFTHC(
            size=self.num_inner,
            num_bits_phase_grad=self.num_bits_phase_grad,
            num_bits_amp_rotations=self.num_bits_amp_rotations,
        )

    @property
    def hadamards(self) -> Bloq:
        """Get the Hadamards for preparing the alias sampling comparison."""
        return OnEach(self.num_bits_keep, Hadamard())

    @property
    def qroam(self) -> InnerQROAM:
        """Get the QROAM loading the alias sampling data."""
        return InnerQROAM(
            num_orbitals=self.num_orbitals,
            num_ranks=self.num_ranks,
            num_bases=self.num_bases,
            num_copies=self.num_copies,
            num_bits_keep=self.num_bits_keep,
            log_block_size=self.log_block_size,
            log_block_size_adjoint=self.log_block_size_adjoint,
        )

    @property
    def tail_qrom(self) -> InnerQROMTail:
        """Get the QROM loading the rotations for the :math:`D_1/Q_1` terms."""
        return InnerQROMTail(
            num_orbitals=self.num_orbitals,
            num_ranks=self.num_ranks,
            num_copies=self.num_copies,
            num_bits_phase_grad=self.num_bits_phase_grad,
        )

    @property
    def inequality_check(self) -> LessThanEqual:
        """Get the alias sampling inequality test."""
        return LessThanEqual(self.num_bits_keep, self.num_bits_keep)

    @property
    def cswap(self) -> CSwap:
        """Get the controlled swap between `inner_index` and `alt`."""
        return CSwap(self.num_bits_inner)

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Build call graph for InnerPrepareDFTHC."""
        return {
            self.uniform_prep: 1,
            self.qroam: 1,
            self.tail_qrom: 1,
            self.inequality_check: 1,
            self.cswap: 1,
            self.hadamards: 1,
            CZ(): 1,
        }

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        **soqs: SoquetT,
    ) -> dict[str, SoquetT]:
        """Implement bloq decomposition into sub-bloqs."""
        (
            soqs["inner_index"],
            soqs["succ_inner"],
            soqs["phase_grad"],
            soqs["amp_rotation"],
        ) = bb.add(
            self.uniform_prep,
            target=soqs["inner_index"],
            success=soqs["succ_inner"],
            phase_grad=soqs["phase_grad"],
            amp_rotation=soqs["amp_rotation"],
        )

        soqs |= bb.add_d(
            self.qroam,
            outer_index=soqs.pop("outer_index"),
            inner_index=soqs.pop("inner_index"),
        )

        # Allocate `rotations` only after the QROAM has freed its junk, so the tail
        # can reuse those qubits. G, r are shared: x_o selects the SF or the tail
        # branch, never both.
        soqs["rotations"] = bb.allocate(self.rotations_size)
        soqs |= bb.add_d(self.tail_qrom, **extract_soqs(self.tail_qrom, soqs))

        soqs["compare"] = bb.allocate(1)

        soqs["keep_compare_inner"] = bb.add(
            self.hadamards,
            q=soqs["keep_compare_inner"],
        )

        soqs["keep"], soqs["keep_compare_inner"], soqs["compare"] = bb.add(
            self.inequality_check,
            x=soqs["keep"],
            y=soqs["keep_compare_inner"],
            target=soqs["compare"],
        )

        soqs["compare"], soqs["inner_index"], soqs["alt"] = bb.add(
            self.cswap,
            ctrl=soqs["compare"],
            x=soqs["inner_index"],
            y=soqs["alt"],
        )

        soqs["compare"], soqs["sign"] = bb.add(
            CZ(),
            q1=soqs["compare"],
            q2=soqs["sign"],
        )

        return soqs


@attrs.frozen
class RotationPrepareDFTHC(Bloq):
    """
    Rotation preparation for the DFTHC block encoding.

    Parameters
    ----------
    num_orbitals : int
        Number of spatial orbitals, :math:`N` in [Low2025]_.
    num_ranks : int
        DFTHC rank, :math:`R` in [Low2025]_.
    num_bases : int
        Number of bases per rank component, :math:`B` in [Low2025]_.

    num_bits_phase_grad : int
        Total number of bits for persistent phase gradient register, :math:`b_{rot}` in
        [Low2025]_.

    Registers
    ---------
    inner_index : QAny
        Register representing the inner index, :math:`b`.

    G_0 : QBit
        Qubit selecting between SF and :math:`D_1/Q_1`.

    rank : QAny
        Register representing the rank, :math:`r`.
    non_identity : QBit, RIGHT
        Non-identity flag qubit.
    rotations : QAny
        Givens rotation angles for the basis rotation.

    References
    ----------
    .. [Low2025] G. H. Low et al., "Fast Quantum Simulation of Electronic Structure by
       Spectral Amplification", Phys. Rev. X, vol. 15, no. 4, p. 041016, Oct. 2025.

    """

    num_orbitals: int
    num_ranks: int
    num_bases: int

    num_bits_phase_grad: int

    @property
    def num_inner(self) -> int:
        """Number of terms for the inner index, :math:`B + 1`."""
        return self.num_bases + 1

    @property
    def num_bits_inner(self) -> int:
        """Number of bits needed to represent the inner index."""
        return bit_length(self.num_inner - 1)

    @property
    def rotations_size(self) -> int:
        """Rotation register size, :math:`(N - 1) b_{rot}`."""
        return (self.num_orbitals - 1) * self.num_bits_phase_grad

    @property
    def num_bits_rank(self) -> int:
        """Number of bits needed to represent the rank, :math:`r`."""
        return bit_length(self.num_ranks - 1)

    @property
    def signature(self) -> Signature:
        """Define input and/or output registers of the bloq."""
        return Signature(
            [
                Register("inner_index", QAny(self.num_bits_inner)),
                Register("G_0", QBit()),
                Register("rank", QAny(self.num_bits_rank)),
                Register("non_identity", QBit(), side=Side.RIGHT),
                Register("rotations", QAny(self.rotations_size)),
            ]
        )

    @property
    def qrom(self) -> RotationQROM:
        """Get the QROM loading the rotation data."""
        return RotationQROM(
            num_orbitals=self.num_orbitals,
            num_ranks=self.num_ranks,
            num_bases=self.num_bases,
            num_bits_phase_grad=self.num_bits_phase_grad,
            num_controls=1,
        )

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Build call graph for RotationPrepareDFTHC."""
        return {self.qrom: 1, XGate(): 3}

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        **soqs: SoquetT,
    ) -> dict[str, SoquetT]:
        """Implement bloq decomposition into sub-bloqs."""
        soqs["non_identity"] = bb.add(XGate(), q=bb.allocate(1))

        soqs["G_0"] = bb.add(XGate(), q=soqs["G_0"])

        (
            soqs["G_0"],
            soqs["inner_index"],
            soqs["rank"],
            soqs["rotations"],
            soqs["non_identity"],
        ) = bb.add(
            self.qrom,
            ctrl=soqs["G_0"],
            inner_index=soqs["inner_index"],
            rank=soqs["rank"],
            rotations=soqs["rotations"],
            non_identity=soqs["non_identity"],
        )

        soqs["G_0"] = bb.add(XGate(), q=soqs["G_0"])

        return soqs
