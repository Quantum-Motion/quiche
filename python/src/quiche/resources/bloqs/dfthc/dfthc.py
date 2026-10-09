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

"""Main routines for the block encoding of the DFTHC Hamiltonian."""

from functools import cached_property
from typing import TYPE_CHECKING, Self

import attrs
from qualtran import (
    Bloq,
    BloqBuilder,
    CtrlSpec,
    QAny,
    QBit,
    Register,
    Signature,
    SoquetT,
)
from qualtran.bloqs.block_encoding import BlockEncoding
from qualtran.bloqs.mcmt import ControlledViaAnd
from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity
from qualtran.bloqs.reflections.reflection_using_prepare import ReflectionUsingPrepare
from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare
from qualtran.resource_counting import (
    BloqCountDictT,
    SympySymbolAllocator,
)
from qualtran.symbolics import bit_length

from quiche.resources.bloqs.dfthc.prep import (
    InnerPrepareDFTHC,
    OuterPrepareDFTHC,
    RotationPrepareDFTHC,
)
from quiche.resources.bloqs.dfthc.select import SelectDFTHC
from quiche.resources.bloqs.dfthc.utils import extract_soqs

if TYPE_CHECKING:
    from quiche.core.electronic import DFTHCHamiltonian


@attrs.frozen
class DFTHCInnerBlockEncoding(BlockEncoding):
    r"""
    Inner block encoding for the DFTHC block encoding.

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

    num_bits_keep_inner : int
        Number of bits for inner coherent alias sampling keep probability,
        :math:`b_{k2}` in [Low2025]_.
    num_bits_phase_grad : int
        Total number of bits for persistent phase gradient register, :math:`b_{rot}` in
        [Low2025]_.
    num_bits_amp_rotations : int
        Number of bits used for amplitude amplification rotations during equal state
        preparation, :math:`s` in [Low2025]_.

    log_block_size_inner : int
        Log of the block size for inner QROAM, :math:`k_2` in [Low2025]_.
    log_block_size_inner_adjoint : int
        Log of the block size for inner QROAM adjoint, :math:`k_4` in [Low2025]_.

    Registers
    ---------
    inner_index : QAny
        Register representing the inner index, :math:`b`.
    keep_compare_inner : QAny
        Uniform superposition register compared against `keep` in the inner alias
        sampling inequality test.
    amp_rotation_inner : QBit
        Qubit rotated in amplitude amplification step of the inner equal superposition
        preparation.

    spin_0 : QBit
        Spin control for SF terms.
    s : QBit
        Superposition qubit selecting :math:`X` or :math:`iY` for :math:`D_1/Q_1`,
        :math:`\varsigma` in [Low2026]_.

    phase_grad : QAny
        Phase gradient register used throughout the circuit.

    system_down : QAny
        Spin-down system register.
    system_up : QAny
        Spin-up system register.

    outer_index : QAny
        Register representing the outer index, :math:`x_o`.
    succ_outer : QBit
        Success flag for the uniform state preparation over the outer index,
        :math:`succ_{x_o}` in [Low2025]_.
    spin_1 : QBit
        Spin control for :math:`D_1/Q_1` terms.

    References
    ----------
    .. [Low2025] G. H. Low et al., "Fast Quantum Simulation of Electronic Structure by
       Spectral Amplification", Phys. Rev. X, vol. 15, no. 4, p. 041016, Oct. 2025.
    .. [Low2026] G. H. Low et al., "A Denser Planar Surface Code", May 28, 2026,
       arXiv: 2605.30455.

    """

    num_orbitals: int
    num_ranks: int
    num_bases: int
    num_copies: int

    num_bits_keep_inner: int
    num_bits_phase_grad: int
    num_bits_amp_rotations: int

    log_block_size_inner: int
    log_block_size_inner_adjoint: int

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

    @cached_property
    def ancilla_bitsize(self) -> int:
        """Total number of bits used for ancilla registers."""
        return sum(r.total_bits() for r in self.ancilla_registers)

    @cached_property
    def resource_bitsize(self) -> int:
        """Total number of bits used for resource registers."""
        return sum(r.total_bits() for r in self.resource_registers)

    @cached_property
    def system_bitsize(self) -> int:
        """Total number of bits used for system registers."""
        return sum(r.total_bits() for r in self.system_registers)

    @property
    def ancilla_registers(self) -> tuple[Register, ...]:
        """Registers projected onto the signal state of the block encoding."""
        return (
            Register("inner_index", QAny(self.num_bits_inner)),
            Register("keep_compare_inner", QAny(self.num_bits_keep_inner)),
            Register("amp_rotation_inner", QBit()),
            Register("spin_0", QBit()),
            Register("s", QBit()),
        )

    @property
    def resource_registers(self) -> tuple[Register, ...]:
        """Registers used by, and returned unchanged by, the block encoding."""
        return (Register("phase_grad", QAny(self.num_bits_phase_grad)),)

    @property
    def system_registers(self) -> tuple[Register, ...]:
        """Registers the block encoding acts on."""
        return (
            Register("system_down", QAny(self.num_orbitals)),
            Register("system_up", QAny(self.num_orbitals)),
        )

    @cached_property
    def signature(self) -> Signature:
        """Define input and/or output registers of the bloq."""
        return Signature(
            [
                *self.ancilla_registers,
                *self.resource_registers,
                *self.system_registers,
                Register("outer_index", QAny(self.num_bits_outer)),
                Register("succ_outer", QBit()),
                Register("spin_1", QBit()),
            ]
        )

    @property
    def alpha(self) -> float:
        """Get the normalisation factor of the block encoding."""
        raise NotImplementedError

    @property
    def epsilon(self) -> float:
        """Get the precision of the block encoding."""
        raise NotImplementedError

    @property
    def signal_state(self) -> BlackBoxPrepare:
        """Get the signal state of the block encoding."""
        return BlackBoxPrepare(PrepareIdentity(self.ancilla_registers))

    @property
    def inner_prep(self) -> InnerPrepareDFTHC:
        """Get the inner index state preparation for the block encoding."""
        return InnerPrepareDFTHC(
            num_orbitals=self.num_orbitals,
            num_ranks=self.num_ranks,
            num_bases=self.num_bases,
            num_copies=self.num_copies,
            num_bits_keep=self.num_bits_keep_inner,
            num_bits_phase_grad=self.num_bits_phase_grad,
            num_bits_amp_rotations=self.num_bits_amp_rotations,
            log_block_size=self.log_block_size_inner,
            log_block_size_adjoint=self.log_block_size_inner_adjoint,
        )

    @property
    def rotation_prep(self) -> RotationPrepareDFTHC:
        """Get the rotation state preparation for the block encoding."""
        return RotationPrepareDFTHC(
            num_orbitals=self.num_orbitals,
            num_ranks=self.num_ranks,
            num_bases=self.num_bases,
            num_bits_phase_grad=self.num_bits_phase_grad,
        )

    @property
    def select(self) -> SelectDFTHC:
        """Get the SELECT operator for the block encoding."""
        return SelectDFTHC(
            num_orbitals=self.num_orbitals,
            num_bits_phase_grad=self.num_bits_phase_grad,
        )

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Build call graph for DFTHCInnerBlockEncoding."""
        return {
            self.inner_prep: 1,
            self.rotation_prep: 1,
            self.select: 1,
            self.rotation_prep.adjoint(): 1,
            self.inner_prep.adjoint(): 1,
        }

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        **soqs: SoquetT,
    ) -> dict[str, SoquetT]:
        """Implement bloq decomposition into sub-bloqs."""
        soqs |= {"succ_inner": bb.allocate(1), "spin": bb.allocate(1)}

        soqs["amp_rotation"] = soqs.pop("amp_rotation_inner")
        soqs |= bb.add_d(self.inner_prep, **extract_soqs(self.inner_prep, soqs))

        soqs |= bb.add_d(self.rotation_prep, **extract_soqs(self.rotation_prep, soqs))

        soqs |= bb.add_d(self.select, **extract_soqs(self.select, soqs))

        soqs |= bb.add_d(
            self.rotation_prep.adjoint(),
            **extract_soqs(self.rotation_prep.adjoint(), soqs),
        )
        soqs |= bb.add_d(
            self.inner_prep.adjoint(),
            **extract_soqs(self.inner_prep.adjoint(), soqs),
        )
        soqs["amp_rotation_inner"] = soqs.pop("amp_rotation")

        bb.free(soqs.pop("succ_inner"))
        bb.free(soqs.pop("spin"))

        return soqs


@attrs.frozen
class DFTHCBlockEncoding(BlockEncoding):
    r"""
    Complete DFTHC Hamiltonian block encoding.

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

    num_bits_keep_inner : int
        Number of bits for inner coherent alias sampling keep probability,
        :math:`b_{k2}` in [Low2025]_.
    num_bits_keep_outer : int
        Number of bits for outer coherent alias sampling keep probability,
        :math:`b_{k1}` in [Low2025]_.
    num_bits_phase_grad : int
        Total number of bits for persistent phase gradient register, :math:`b_{rot}` in
        [Low2025]_.
    num_bits_amp_rotations : int
        Number of bits used for amplitude amplification rotations during equal state
        preparation, :math:`s` in [Low2025]_.

    log_block_size_inner : int
        Log of the block size for inner QROAM, :math:`k_2` in [Low2025]_.
    log_block_size_inner_adjoint : int
        Log of the block size for inner QROAM adjoint, :math:`k_4` in [Low2025]_.
    log_block_size_outer : int
        Log of the block size for outer QROAM, :math:`k_1` in [Low2025]_.
    log_block_size_outer_adjoint : int
        Log of the block size for outer QROAM adjoint, :math:`k_5` in [Low2025]_.

    num_controls : int, optional
        Number of control qubits (default = 0).

    Registers
    ---------
    ctrl : QAny, optional
        Control qubits.

    outer_index : QAny
        Register representing the outer index, :math:`x_o`.
    inner_index : QAny
        Register representing the inner index, :math:`b`.

    keep_compare_outer : QAny
        Uniform superposition register compared against `keep` in the outer alias
        sampling inequality test.
    keep_compare_inner : QAny
        Uniform superposition register compared against `keep` in the inner alias
        sampling inequality test.

    amp_rotation_outer : QBit
        Qubit rotated in amplitude amplification step of the outer equal superposition
        preparation.
    amp_rotation_inner : QBit
        Qubit rotated in amplitude amplification step of the inner equal superposition
        preparation.

    spin_0 : QBit
        Spin control for SF terms.
    spin_1 : QBit
        Spin control for :math:`D_1/Q_1` terms.
    s : QBit
        Superposition qubit selecting :math:`X` or :math:`iY` for :math:`D_1/Q_1`,
        :math:`\varsigma` in [Low2026]_.

    phase_grad : QAny
        Phase gradient register used throughout the circuit.

    system_down : QAny
        Spin-down system register.
    system_up : QAny
        Spin-up system register.

    References
    ----------
    .. [Low2025] G. H. Low et al., "Fast Quantum Simulation of Electronic Structure by
       Spectral Amplification", Phys. Rev. X, vol. 15, no. 4, p. 041016, Oct. 2025.
    .. [Low2026] G. H. Low et al., "A Denser Planar Surface Code", May 28, 2026,
       arXiv: 2605.30455.

    """

    num_orbitals: int
    num_ranks: int
    num_bases: int
    num_copies: int

    num_bits_keep_inner: int
    num_bits_keep_outer: int
    num_bits_phase_grad: int
    num_bits_amp_rotations: int

    log_block_size_inner: int
    log_block_size_inner_adjoint: int
    log_block_size_outer: int
    log_block_size_outer_adjoint: int

    num_controls: int = 0

    def __attrs_post_init__(self) -> None:
        """Validate inputs."""
        if self.num_bits_amp_rotations > self.num_bits_phase_grad:
            err_msg = (
                "Phase gradient register not large enough for number of rotation bits."
            )
            raise ValueError(err_msg)

    @classmethod
    def FromDFTHCHamiltonian(  # noqa: N802, PLR0913
        cls,
        hamiltonian: "DFTHCHamiltonian",
        *,
        num_bits_keep_inner: int,
        num_bits_keep_outer: int,
        num_bits_phase_grad: int,
        num_bits_amp_rotations: int,
        log_block_size_inner: int,
        log_block_size_inner_adjoint: int,
        log_block_size_outer: int,
        log_block_size_outer_adjoint: int,
        num_controls: int = 0,
    ) -> Self:
        """
        Complete DFTHC Hamiltonian block encoding.

        Parameters
        ----------
        hamiltonian : DFTHCHamiltonian
            A DFTHCHamiltonian object that contains parameters of a
            second quantised Hamiltonian, after factorisation.
        num_bits_keep_inner : int
            Number of bits for inner coherent alias sampling keep probability,
            :math:`b_{k2}` in ref. [1].
        num_bits_keep_outer : int
            Number of bits for outer coherent alias sampling keep probability,
            :math:`b_{k1}` in ref. [1].
        num_bits_phase_grad : int
            Total number of bits for persistent phase gradient register,
            :math:`b_{rot}` in ref. [1].
        num_bits_amp_rotations : int
            Number of bits used for amplitude amplification rotations during equal state
            preparation, :math:`s` in ref. [1].

        log_block_size_inner : int
            Log of the block size for inner QROAM, :math:`k_2` in ref. [1].
        log_block_size_inner_adjoint : int
            Log of the block size for inner QROAM adjoint, :math:`k_4` in ref. [1].
        log_block_size_outer : int
            Log of the block size for outer QROAM, :math:`k_1` in ref. [1].
        log_block_size_outer_adjoint : int
            Log of the block size for outer QROAM adjoint, :math:`k_5` in ref. [1].

        num_controls : int, optional
            Number of control qubits (default = 0).

        Registers
        ---------
        `block_encoding`'s registers, preceded by `ctrl` if `num_controls` > 0.

        References
        ----------
            [1] G. H. Low et al., "Fast Quantum Simulation of Electronic Structure by
                Spectral Amplification", Phys. Rev. X, vol. 15, no. 4,
                p. 041016, Oct. 2025.

        """
        return cls(
            num_orbitals=hamiltonian.num_orbitals,
            num_ranks=hamiltonian.num_ranks,
            num_bases=hamiltonian.num_bases,
            num_copies=hamiltonian.num_copies,
            num_bits_keep_inner=num_bits_keep_inner,
            num_bits_keep_outer=num_bits_keep_outer,
            num_bits_phase_grad=num_bits_phase_grad,
            num_bits_amp_rotations=num_bits_amp_rotations,
            log_block_size_inner=log_block_size_inner,
            log_block_size_inner_adjoint=log_block_size_inner_adjoint,
            log_block_size_outer=log_block_size_outer,
            log_block_size_outer_adjoint=log_block_size_outer_adjoint,
            num_controls=num_controls,
        )

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

    @cached_property
    def ancilla_bitsize(self) -> int:
        """Total number of bits used for ancilla registers."""
        return sum(r.total_bits() for r in self.ancilla_registers)

    @cached_property
    def resource_bitsize(self) -> int:
        """Total number of bits used for resource registers."""
        return sum(r.total_bits() for r in self.resource_registers)

    @cached_property
    def system_bitsize(self) -> int:
        """Total number of bits used for system registers."""
        return sum(r.total_bits() for r in self.system_registers)

    @property
    def control_registers(self) -> tuple[Register, ...]:
        """Control registers of the block encoding."""
        return (Register("ctrl", QAny(self.num_controls)),) if self.num_controls else ()

    @property
    def ancilla_registers(self) -> tuple[Register, ...]:
        """Registers projected onto the signal state of the block encoding."""
        return (
            Register("outer_index", QAny(self.num_bits_outer)),
            Register("inner_index", QAny(self.num_bits_inner)),
            Register("keep_compare_outer", QAny(self.num_bits_keep_outer)),
            Register("keep_compare_inner", QAny(self.num_bits_keep_inner)),
            Register("amp_rotation_outer", QBit()),
            Register("amp_rotation_inner", QBit()),
            Register("spin_0", QBit()),
            Register("spin_1", QBit()),
            Register("s", QBit()),
        )

    @property
    def resource_registers(self) -> tuple[Register, ...]:
        """Registers used by, and returned unchanged by, the block encoding."""
        return (Register("phase_grad", QAny(self.num_bits_phase_grad)),)

    @property
    def system_registers(self) -> tuple[Register, ...]:
        """Registers the block encoding acts on."""
        return (
            Register("system_down", QAny(self.num_orbitals)),
            Register("system_up", QAny(self.num_orbitals)),
        )

    @cached_property
    def signature(self) -> Signature:
        """Define input and/or output registers of the bloq."""
        return Signature(
            [
                *self.control_registers,
                *self.ancilla_registers,
                *self.resource_registers,
                *self.system_registers,
            ]
        )

    @property
    def alpha(self) -> float:
        """Get the normalisation factor of the block encoding."""
        raise NotImplementedError

    @property
    def epsilon(self) -> float:
        """Get the precision of the block encoding."""
        raise NotImplementedError

    @property
    def signal_state(self) -> BlackBoxPrepare:
        """Get the signal state of the block encoding."""
        return BlackBoxPrepare(PrepareIdentity(self.ancilla_registers))

    @property
    def outer_prep(self) -> OuterPrepareDFTHC:
        """Get the outer index state preparation for the block encoding."""
        return OuterPrepareDFTHC(
            num_orbitals=self.num_orbitals,
            num_ranks=self.num_ranks,
            num_copies=self.num_copies,
            num_bits_keep=self.num_bits_keep_outer,
            num_bits_phase_grad=self.num_bits_phase_grad,
            num_bits_amp_rotations=self.num_bits_amp_rotations,
            log_block_size=self.log_block_size_outer,
            log_block_size_adjoint=self.log_block_size_outer_adjoint,
        )

    @property
    def inner(self) -> DFTHCInnerBlockEncoding:
        """Get the inner block encoding."""
        return DFTHCInnerBlockEncoding(
            num_bases=self.num_bases,
            num_orbitals=self.num_orbitals,
            num_ranks=self.num_ranks,
            num_copies=self.num_copies,
            num_bits_keep_inner=self.num_bits_keep_inner,
            num_bits_phase_grad=self.num_bits_phase_grad,
            num_bits_amp_rotations=self.num_bits_amp_rotations,
            log_block_size_inner=self.log_block_size_inner,
            log_block_size_inner_adjoint=self.log_block_size_inner_adjoint,
        )

    @property
    def t2_reflection(self) -> Bloq:
        """Get the reflection used for the Chebyshev polynomial (:math:`T_2`)."""
        reflection = ReflectionUsingPrepare(
            PrepareIdentity(self.inner.ancilla_registers),
            control_val=1,
            global_phase=-1,
        )

        if self.num_controls:
            return ControlledViaAnd(
                reflection,
                CtrlSpec(qdtypes=QAny(self.num_controls), cvs=2**self.num_controls - 1),
            )
        return reflection

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Build call graph for DFTHCBlockEncoding."""
        return {
            self.outer_prep: 1,
            self.inner: 1,
            self.t2_reflection: 1,
            self.inner.adjoint(): 1,
            self.outer_prep.adjoint(): 1,
        }

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        **soqs: SoquetT,
    ) -> dict[str, SoquetT]:
        """Implement bloq decomposition into sub-bloqs."""
        soqs |= {"succ_outer": bb.allocate(1)}

        soqs["amp_rotation"] = soqs.pop("amp_rotation_outer")
        soqs |= bb.add_d(self.outer_prep, **extract_soqs(self.outer_prep, soqs))

        soqs |= bb.add_d(self.inner, **extract_soqs(self.inner, soqs))

        soqs["control"] = soqs.pop("succ_outer")
        soqs |= bb.add_d(self.t2_reflection, **extract_soqs(self.t2_reflection, soqs))
        soqs["succ_outer"] = soqs.pop("control")

        soqs |= bb.add_d(
            self.inner.adjoint(), **extract_soqs(self.inner.adjoint(), soqs)
        )
        soqs |= bb.add_d(
            self.outer_prep.adjoint(), **extract_soqs(self.outer_prep.adjoint(), soqs)
        )
        soqs["amp_rotation_outer"] = soqs.pop("amp_rotation")

        bb.free(soqs.pop("succ_outer"))

        return soqs


@attrs.frozen
class DFTHCWalkOperator(Bloq):
    """
    Qubitisation walk operator for the DFTHC block encoding.

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

    num_bits_keep_inner : int
        Number of bits for inner coherent alias sampling keep probability,
        :math:`b_{k2}` in [Low2025]_.
    num_bits_keep_outer : int
        Number of bits for outer coherent alias sampling keep probability,
        :math:`b_{k1}` in [Low2025]_.
    num_bits_phase_grad : int
        Total number of bits for persistent phase gradient register, :math:`b_{rot}` in
        [Low2025]_.
    num_bits_amp_rotations : int
        Number of bits used for amplitude amplification rotations during equal state
        preparation, :math:`s` in [Low2025]_.

    log_block_size_inner : int
        Log of the block size for inner QROAM, :math:`k_2` in [Low2025]_.
    log_block_size_inner_adjoint : int
        Log of the block size for inner QROAM adjoint, :math:`k_4` in [Low2025]_.
    log_block_size_outer : int
        Log of the block size for outer QROAM, :math:`k_1` in [Low2025]_.
    log_block_size_outer_adjoint : int
        Log of the block size for outer QROAM adjoint, :math:`k_5` in [Low2025]_.

    num_controls : int, optional
        Number of control qubits (default = 0).

    Registers
    ---------
    `block_encoding`'s registers, preceded by `ctrl` if `num_controls` > 0.

    References
    ----------
    .. [Low2025] G. H. Low et al., "Fast Quantum Simulation of Electronic Structure by
       Spectral Amplification", Phys. Rev. X, vol. 15, no. 4, p. 041016, Oct. 2025.

    Examples
    --------
    Using the FeMoco-54 parameters from [Low2025]_:

    >>> walk = DFTHCWalkOperator(
    ...     num_orbitals=54,
    ...     num_ranks=10,
    ...     num_bases=27,
    ...     num_copies=27,
    ...     num_bits_keep_inner=15,
    ...     num_bits_keep_outer=15,
    ...     num_bits_phase_grad=15,
    ...     num_bits_amp_rotations=7,
    ...     log_block_size_inner=4,
    ...     log_block_size_inner_adjoint=2,
    ...     log_block_size_outer=2,
    ...     log_block_size_outer_adjoint=4,
    ... )

    """

    num_orbitals: int
    num_ranks: int
    num_bases: int
    num_copies: int

    num_bits_keep_inner: int
    num_bits_keep_outer: int
    num_bits_phase_grad: int
    num_bits_amp_rotations: int

    log_block_size_inner: int
    log_block_size_inner_adjoint: int
    log_block_size_outer: int
    log_block_size_outer_adjoint: int

    num_controls: int = 0

    @classmethod
    def FromDFTHCHamiltonian(  # noqa: N802, PLR0913
        cls,
        hamiltonian: "DFTHCHamiltonian",
        *,
        num_bits_keep_inner: int,
        num_bits_keep_outer: int,
        num_bits_phase_grad: int,
        num_bits_amp_rotations: int,
        log_block_size_inner: int,
        log_block_size_inner_adjoint: int,
        log_block_size_outer: int,
        log_block_size_outer_adjoint: int,
        num_controls: int = 0,
    ) -> Self:
        """
        Qubitisation walk operator for the DFTHC block encoding.

        Parameters
        ----------
        hamiltonian : DFTHCHamiltonian
            A DFTHCHamiltonian object that contains parameters of a
            second quantised Hamiltonian, after factorisation.
        num_bits_keep_inner : int
            Number of bits for inner coherent alias sampling keep probability,
            :math:`b_{k2}` in ref. [1].
        num_bits_keep_outer : int
            Number of bits for outer coherent alias sampling keep probability,
            :math:`b_{k1}` in ref. [1].
        num_bits_phase_grad : int
            Total number of bits for persistent phase gradient register,
            :math:`b_{rot}` in ref. [1].
        num_bits_amp_rotations : int
            Number of bits used for amplitude amplification rotations during equal state
            preparation, :math:`s` in ref. [1].

        log_block_size_inner : int
            Log of the block size for inner QROAM, :math:`k_2` in ref. [1].
        log_block_size_inner_adjoint : int
            Log of the block size for inner QROAM adjoint, :math:`k_4` in ref. [1].
        log_block_size_outer : int
            Log of the block size for outer QROAM, :math:`k_1` in ref. [1].
        log_block_size_outer_adjoint : int
            Log of the block size for outer QROAM adjoint, :math:`k_5` in ref. [1].

        num_controls : int, optional
            Number of control qubits (default = 0).

        Registers
        ---------
        `block_encoding`'s registers, preceded by `ctrl` if `num_controls` > 0.

        References
        ----------
            [1] G. H. Low et al., "Fast Quantum Simulation of Electronic Structure by
                Spectral Amplification", Phys. Rev. X, vol. 15, no. 4,
                p. 041016, Oct. 2025.

        """
        return cls(
            num_orbitals=hamiltonian.num_orbitals,
            num_ranks=hamiltonian.num_ranks,
            num_bases=hamiltonian.num_bases,
            num_copies=hamiltonian.num_copies,
            num_bits_keep_inner=num_bits_keep_inner,
            num_bits_keep_outer=num_bits_keep_outer,
            num_bits_phase_grad=num_bits_phase_grad,
            num_bits_amp_rotations=num_bits_amp_rotations,
            log_block_size_inner=log_block_size_inner,
            log_block_size_inner_adjoint=log_block_size_inner_adjoint,
            log_block_size_outer=log_block_size_outer,
            log_block_size_outer_adjoint=log_block_size_outer_adjoint,
            num_controls=num_controls,
        )

    @property
    def control_registers(self) -> tuple[Register, ...]:
        """Control registers of the walk operator."""
        return (Register("ctrl", QAny(self.num_controls)),) if self.num_controls else ()

    @cached_property
    def signature(self) -> Signature:
        """Define input and/or output registers of the bloq."""
        return Signature(
            [
                *self.control_registers,
                *self.block_encoding.signature,
            ]
        )

    @property
    def block_encoding(self) -> DFTHCBlockEncoding:
        """Get the block encoding that will be qubitised."""
        return DFTHCBlockEncoding(
            num_orbitals=self.num_orbitals,
            num_ranks=self.num_ranks,
            num_bases=self.num_bases,
            num_copies=self.num_copies,
            num_bits_keep_inner=self.num_bits_keep_inner,
            num_bits_keep_outer=self.num_bits_keep_outer,
            num_bits_phase_grad=self.num_bits_phase_grad,
            num_bits_amp_rotations=self.num_bits_amp_rotations,
            log_block_size_inner=self.log_block_size_inner,
            log_block_size_inner_adjoint=self.log_block_size_inner_adjoint,
            log_block_size_outer=self.log_block_size_outer,
            log_block_size_outer_adjoint=self.log_block_size_outer_adjoint,
            num_controls=0,
        )

    @property
    def reflection(self) -> Bloq:
        """Get the reflection that turns the block encoding into a walk operator."""
        reflection = ReflectionUsingPrepare(
            PrepareIdentity(self.block_encoding.ancilla_registers),
            global_phase=-1,
        )
        if self.num_controls:
            return ControlledViaAnd(
                reflection,
                CtrlSpec(qdtypes=QAny(self.num_controls), cvs=2**self.num_controls - 1),
            )
        return reflection

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Build call graph for DFTHCWalkOperator."""
        return {self.block_encoding: 1, self.reflection: 1}

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        **soqs: SoquetT,
    ) -> dict[str, SoquetT]:
        """Implement bloq decomposition into sub-bloqs."""
        soqs |= bb.add_d(self.block_encoding, **extract_soqs(self.block_encoding, soqs))
        soqs |= bb.add_d(self.reflection, **extract_soqs(self.reflection, soqs))
        return soqs
