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

"""Custom optimised data loading methods for DFTHC Hamiltonian block encoding."""

from __future__ import annotations

from collections import Counter
from functools import cached_property
from math import ceil
from typing import TYPE_CHECKING, override

from cirq import X, Z
from qualtran import (
    Bloq,
    BloqBuilder,
    BQUInt,
    QAny,
    QBit,
    Register,
    Side,
    Signature,
    SoquetT,
)
from qualtran.bloqs.arithmetic import XorK
from qualtran.bloqs.basic_gates import CNOT, XGate
from qualtran.bloqs.bookkeeping import Partition
from qualtran.bloqs.multiplexers.apply_gate_to_lth_target import ApplyGateToLthQubit
from qualtran.bloqs.multiplexers.unary_iteration_bloq import UnaryIterationGate
from qualtran.bloqs.swap_network import SwapWithZero
from qualtran.symbolics import bit_length

from quiche.resources.bloqs.dfthc.encodings import (
    InnerData,
    InnerSignData,
    RotationData,
    TailData,
    as_inner_data,
    as_inner_sign_data,
    as_rotation_data,
    as_tail_data,
)
from quiche.resources.bloqs.dfthc.utils import extract_soqs

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from cirq import OP_TREE, DecompositionContext, Gate, Qid
    from numpy.typing import NDArray
    from qualtran.resource_counting import (
        BloqCountDictT,
        BloqCountT,
        SympySymbolAllocator,
    )
    from qualtran.symbolics import SymbolicInt

    QidArray = NDArray[Qid]  # ty: ignore[invalid-type-arguments]

import attrs


@attrs.frozen
class QROMAdjoint(Bloq):
    """
    Measurement-based uncomputation of QROM loads.

    Supports non-rectangular as well as non-power-of-2 data shapes.

    For ``InnerQROAM``, the phase fixup also applies the sign :math:`(-1)^{s_i}` of
    the SF coefficients, from ``qrom.qrom.sign_data``, after the alias swap is undone.
    This only changes which phases are flipped, so the cost is the same.

    Parameters
    ----------
    qrom : InnerQROMTail | InnerQROAM | RotationQROM
        Data lookup whose output is erased.
    unary_size : int
        Size of the binary-to-unary conversion.
    iteration_size : int
        Number of entries in the unary iteration applying the phase fixups.

    Registers
    ---------
    `qrom`'s registers, with left- and right-registers swapped.

    References
    ----------
    .. [Low2025] G. H. Low et al., "Fast Quantum Simulation of Electronic Structure by
       Spectral Amplification", Phys. Rev. X, vol. 15, no. 4, p. 041016, Oct. 2025.
    """

    qrom: InnerQROMTail | InnerQROAM | RotationQROM
    unary_size: int
    iteration_size: int

    @staticmethod
    def _z_gate(_: int) -> Gate:
        """Z gate at every leaf; named function so equal bloqs compare correctly."""
        return Z

    @staticmethod
    def _x_gate(_: int) -> Gate:
        """X gate at every leaf; named function so equal bloqs compare correctly."""
        return X

    @staticmethod
    def _unary_iteration(
        size: int,
        gate: Callable[[int], Gate],
        control_regs: tuple[Register, ...] = (),
    ) -> ApplyGateToLthQubit:
        """Unary iteration applying a gate at each leaf."""
        return ApplyGateToLthQubit(
            Register("selection", BQUInt(bit_length(size - 1), size)),
            nth_gate=gate,
            control_regs=control_regs,
        )

    @staticmethod
    def phase_iteration(
        size: int,
        control_regs: tuple[Register, ...] = (),
    ) -> ApplyGateToLthQubit:
        """Unary iteration applying Z at each leaf."""
        return QROMAdjoint._unary_iteration(size, QROMAdjoint._z_gate, control_regs)

    @staticmethod
    def binary_to_unary(size: int) -> ApplyGateToLthQubit:
        """Binary-to-unary conversion of register, i.e. a QROM loading the identity."""
        return QROMAdjoint._unary_iteration(size, QROMAdjoint._x_gate)

    @cached_property
    def signature(self) -> Signature:
        """Define input and/or output registers of the bloq."""
        return self.qrom.signature.adjoint()

    @cached_property
    def control_registers(self) -> tuple[Register, ...]:
        """Control registers of the erasure (same as the lookup)."""
        return self.qrom.control_registers

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Build call graph for QROMAdjoint."""
        counts = {self.phase_iteration(self.iteration_size, self.control_registers): 1}
        if self.unary_size > 1:
            counts[self.binary_to_unary(self.unary_size)] = 1
        return counts

    def adjoint(self) -> Bloq:
        """Get the data lookup undone by this erasure."""
        return self.qrom


@attrs.frozen
class InnerQROM(UnaryIterationGate):
    """
    QROM loading the alias sampling data, G and rank for the SF terms.

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

    log_block_size : int
        Log of the block size for QROAM, :math:`k_2` in [Low2025]_.

    num_controls : int, optional
        Number of control qubits (default = 0).

    inner_data : tuple[tuple[int, ...], ...] | None, optional
        Alias sampling words loaded for the SF terms, indexed
        ``[outer_index][inner_index]``, as packed by ``encode_inner_data``. If None
        (default), placeholder data is used, which is sufficient for Toffoli resource
        estimation.
    sign_data : tuple[tuple[int, ...], ...] | None, optional
        Sign bits of the SF coefficients, indexed ``[outer_index][inner_index]``, as
        computed by ``inner_qrom_signs``. They are not loaded, but applied by the
        phase fixup of the erasure (``QROMAdjoint``). If None (default), no signs
        are specified, which is sufficient for Toffoli resource estimation.

    Registers
    ---------
    outer_index : BQUInt
        Register representing the outer index, :math:`x_o`.
    inner_index_high : BQUInt
        High bits of the inner index, selecting the QROAM block.

    alias_words : QAny
        Block of alias sampling words.

    G_0 : QBit
        Qubit selecting between SF and :math:`D_1/Q_1`.
    G_1 : QBit
        Qubit selecting between :math:`D_1` and :math:`Q_1`.
    rank : QAny
        Register representing the rank, :math:`r`.

    ctrl : QAny, optional
        Control qubits.

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

    log_block_size: int

    num_controls: int = 0

    inner_data: InnerData | None = attrs.field(
        default=None, repr=False, converter=as_inner_data
    )
    sign_data: InnerSignData | None = attrs.field(
        default=None, repr=False, converter=as_inner_sign_data
    )

    def __attrs_post_init__(self) -> None:
        """Validate the shape and bit widths of the alias sampling and sign data."""
        for name, data in (
            ("inner_data", self.inner_data),
            ("sign_data", self.sign_data),
        ):
            if data is not None and (
                len(data) != self.num_rc
                or any(len(row) != self.num_inner for row in data)
            ):
                err_msg = (
                    f"`{name}` must have shape ({self.num_rc}, {self.num_inner}) "
                    "indexed [outer_index][inner_index]."
                )
                raise ValueError(err_msg)

        if self.inner_data is not None and any(
            not 0 <= word < 2**self.alias_word_size
            for row in self.inner_data
            for word in row
        ):
            err_msg = f"`inner_data` words must fit in {self.alias_word_size} bits."
            raise ValueError(err_msg)

        if self.sign_data is not None and any(
            bit not in (0, 1) for row in self.sign_data for bit in row
        ):
            err_msg = "`sign_data` bits must be 0 or 1."
            raise ValueError(err_msg)

    @property
    def num_rc(self) -> int:
        """Number of SF terms, :math:`RC`."""
        return self.num_ranks * self.num_copies

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
    def block_size(self) -> int:
        """Number of terms within a single block."""
        return 2**self.log_block_size

    @property
    def num_blocks(self) -> int:
        """Number of blocks covering the inner index range."""
        return ceil(self.num_inner / self.block_size)

    @property
    def num_bits_blocks(self) -> int:
        """Number of bits needed to represent the block index."""
        return bit_length(self.num_blocks - 1)

    @property
    def alias_word_size(self) -> int:
        r"""
        Number of bits in one alias sampling word.

        This is :math:`b_2 = \lceil \log(B + 1) \rceil + b_{k2}` (eq. (B32) of
        [Low2025]_) plus the sign bit :math:`d_i` of ``encode_inner_data``.
        """
        return self.num_bits_inner + self.num_bits_keep + 1

    @cached_property
    def control_registers(self) -> tuple[Register, ...]:
        """Control registers of the QROM."""
        return (Register("ctrl", QAny(self.num_controls)),) if self.num_controls else ()

    @cached_property
    def selection_registers(self) -> tuple[Register, ...]:
        """Index registers iterated over by unary iteration."""
        return (
            Register("outer_index", BQUInt(self.num_bits_outer, self.num_outer)),
            Register("inner_index_high", BQUInt(self.num_bits_blocks, self.num_blocks)),
        )

    @cached_property
    def target_registers(self) -> tuple[Register, ...]:
        """Registers the looked-up data is written into."""
        return (
            Register("alias_words", QAny(self.alias_word_size), (self.block_size,)),
            Register("G_0", QBit()),
            Register("G_1", QBit()),
            Register("rank", QAny(self.num_bits_rank)),
        )

    @override
    def nth_operation(
        self,
        context: DecompositionContext,
        control: Qid,
        *,
        outer_index: int,
        inner_index_high: int,
        alias_words: QidArray,
        G_0: QidArray,
        G_1: QidArray,
        rank: QidArray,
    ) -> Iterator[OP_TREE]:  # ty: ignore[invalid-method-override]
        # The D_1/Q_1 tail (x_o >= RC) belongs to `InnerQROMTail`
        if outer_index >= self.num_rc:
            return

        # G = (0, 0) marks an SF term, so G_0 and G_1 are left as they are. The rank
        # follows from the outer index x_o = r * C + c, for every index here.
        yield (
            XorK(QAny(self.num_bits_rank), outer_index // self.num_copies)
            .on(*rank)
            .controlled_by(control)
        )

        # Slot i of the block holds entry b = inner_index_high * 2**k_2 + i, and
        # slots past b = B in the last block are padding
        for i in range(self.block_size):
            inner_index = inner_index_high * self.block_size + i
            data = (
                0
                if self.inner_data is None or inner_index >= self.num_inner
                else self.inner_data[outer_index][inner_index]
            )
            yield (
                XorK(QAny(self.alias_word_size), data)
                .on(*alias_words[i])
                .controlled_by(control)
            )

    @override
    def _break_early(
        self,
        selection_index_prefix: tuple[int, ...],
        l: SymbolicInt,  # noqa: E741 (required for override)
        r: SymbolicInt,
    ) -> bool:
        # Skip terms belonging to `InnerQROMTail`.
        if not selection_index_prefix:
            return l >= self.num_rc

        return selection_index_prefix[0] >= self.num_rc

    def nth_operation_callgraph(
        self, outer_index: int, inner_index_high: int
    ) -> set[BloqCountT]:
        """Get the data loading cost for each index."""
        if outer_index >= self.num_rc:
            return set()

        # Same data as `nth_operation`, so counts match the decomposition.
        # Counter to prevent merging of identical bloqs.
        counts = Counter[Bloq]()
        counts[
            XorK(QAny(self.num_bits_rank), outer_index // self.num_copies).controlled()
        ] += 1
        for i in range(self.block_size):
            inner_index = inner_index_high * self.block_size + i
            data = (
                0
                if self.inner_data is None or inner_index >= self.num_inner
                else self.inner_data[outer_index][inner_index]
            )
            counts[XorK(QAny(self.alias_word_size), data).controlled()] += 1
        return set(counts.items())


@attrs.frozen
class InnerQROMTail(UnaryIterationGate):
    """
    Inner QROM continuation loading rotations and G for the :math:`D_1/Q_1` terms.

    Parameters
    ----------
    num_orbitals : int
        Number of spatial orbitals, :math:`N` in [Low2025]_.
    num_ranks : int
        DFTHC rank, :math:`R` in [Low2025]_.
    num_copies : int
        Number of copies per rank component, :math:`C` in [Low2025]_.

    num_bits_phase_grad : int
        Total number of bits for persistent phase gradient register, :math:`b_{rot}` in
        [Low2025]_.
    num_controls : int, optional
        Number of control qubits (default = 0).

    tail_data : tuple[tuple[int, int], ...] | None, optional
        ``(rotations, G_1)`` pairs loaded for the :math:`D_1/Q_1` terms, indexed
        ``[outer_index - RC]``, with each rotation word :math:`(N - 1) b_{rot}` bits
        wide and `G_1` = 1 for the :math:`Q_1` terms. If None (default),
        placeholder data is used, which is sufficient for Toffoli resource estimation.

    Registers
    ---------
    outer_index : BQUInt
        Register representing the outer index, :math:`x_o`.

    rotations : QAny
        Givens rotation angles for the basis rotation.

    G_0 : QBit
        Qubit selecting between SF and :math:`D_1/Q_1`.
    G_1 : QBit
        Qubit selecting between :math:`D_1` and :math:`Q_1`.
    rank : QAny
        Register representing the rank, :math:`r`. Not written here, as it is only
        used for SF terms.

    ctrl : QAny, optional
        Control qubits.

    References
    ----------
    .. [Low2025] G. H. Low et al., "Fast Quantum Simulation of Electronic Structure by
       Spectral Amplification", Phys. Rev. X, vol. 15, no. 4, p. 041016, Oct. 2025.
    """

    num_orbitals: int
    num_ranks: int
    num_copies: int

    num_bits_phase_grad: int

    num_controls: int = 0

    tail_data: TailData | None = attrs.field(
        default=None, repr=False, converter=as_tail_data
    )

    def __attrs_post_init__(self) -> None:
        """Validate the length and bit widths of the tail data."""
        if self.tail_data is None:
            return

        if len(self.tail_data) != self.num_orbitals:
            err_msg = (
                f"`tail_data` must have {self.num_orbitals} entries indexed "
                "[outer_index - RC]."
            )
            raise ValueError(err_msg)

        if any(not 0 <= word < 2**self.rotations_size for word, _ in self.tail_data):
            err_msg = f"`tail_data` words must fit in {self.rotations_size} bits."
            raise ValueError(err_msg)

        if any(g_1 not in (0, 1) for _, g_1 in self.tail_data):
            err_msg = "`tail_data` G_1 bits must be 0 or 1."
            raise ValueError(err_msg)

    @property
    def num_rc(self) -> int:
        """Number of SF terms, :math:`RC`."""
        return self.num_ranks * self.num_copies

    @property
    def num_outer(self) -> int:
        """Number of terms for the outer index, :math:`N + RC`."""
        return self.num_orbitals + self.num_ranks * self.num_copies

    @property
    def num_bits_outer(self) -> int:
        """Number of bits needed to represent the outer index."""
        return bit_length(self.num_outer - 1)

    @property
    def num_bits_rank(self) -> int:
        """Number of bits needed to represent the rank, :math:`r`."""
        return bit_length(self.num_ranks - 1)

    @property
    def rotations_size(self) -> int:
        """Rotation register size, :math:`(N - 1) b_{rot}`."""
        return (self.num_orbitals - 1) * self.num_bits_phase_grad

    @cached_property
    def control_registers(self) -> tuple[Register, ...]:
        """Control registers of the QROM."""
        return (Register("ctrl", QAny(self.num_controls)),) if self.num_controls else ()

    @cached_property
    def selection_registers(self) -> tuple[Register, ...]:
        """Index registers iterated over by unary iteration."""
        return (Register("outer_index", BQUInt(self.num_bits_outer, self.num_outer)),)

    @cached_property
    def target_registers(self) -> tuple[Register, ...]:
        """Registers the looked-up data is written into."""
        return (
            Register("rotations", QAny(self.rotations_size)),
            Register("G_0", QBit()),
            Register("G_1", QBit()),
            Register("rank", QAny(self.num_bits_rank)),
        )

    @override
    def nth_operation(
        self,
        context: DecompositionContext,
        control: Qid,
        *,
        outer_index: int,
        rotations: QidArray,
        G_0: QidArray,
        G_1: QidArray,
        rank: QidArray,
    ) -> Iterator[OP_TREE]:  # ty: ignore[invalid-method-override]
        if outer_index < self.num_rc:
            return

        rotation_word, g_1 = (
            (0, 0)
            if self.tail_data is None
            else self.tail_data[outer_index - self.num_rc]
        )

        # G_0 = 1 marks a D_1/Q_1 term, which holds for every index here.
        yield XGate().on(*G_0).controlled_by(control)
        yield XorK(QBit(), g_1).on(*G_1).controlled_by(control)
        yield (
            XorK(QAny(self.rotations_size), rotation_word)
            .on(*rotations)
            .controlled_by(control)
        )

    @override
    def _break_early(
        self,
        selection_index_prefix: tuple[int, ...],
        l: SymbolicInt,  # noqa: E741 (required for override)
        r: SymbolicInt,
    ) -> bool:
        # Skip terms belonging to `InnerQROM`.
        return r <= self.num_rc

    def nth_operation_callgraph(self, outer_index: int) -> set[BloqCountT]:
        """Get the data loading cost for each index."""
        if outer_index < self.num_rc:
            return set()

        # Same data as `nth_operation`, so counts match the decomposition.
        rotation_word, g_1 = (
            (0, 0)
            if self.tail_data is None
            else self.tail_data[outer_index - self.num_rc]
        )
        return {
            (CNOT(), 1),
            (XorK(QBit(), g_1).controlled(), 1),
            (XorK(QAny(self.rotations_size), rotation_word).controlled(), 1),
        }

    def adjoint(self) -> Bloq:
        """Get the measurement-based erasure of this QROM."""
        return QROMAdjoint(self, unary_size=1, iteration_size=self.num_orbitals)


@attrs.frozen
class InnerQROAM(Bloq):
    """
    QROAM loading the alias sampling data and G and rank for inner state preparation.

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

    log_block_size : int
        Log of the block size for QROAM, :math:`k_2` in [Low2025]_.
    log_block_size_adjoint : int
        Log of the block size for QROAM adjoint, :math:`k_4` in [Low2025]_.

    num_controls : int, optional
        Number of control qubits (default = 0).

    inner_data : tuple[tuple[int, ...], ...] | None, optional
        Alias sampling words loaded for the SF terms, indexed
        ``[outer_index][inner_index]``, as packed by ``encode_inner_data``. If None
        (default), placeholder data is used, which is sufficient for Toffoli resource
        estimation.
    sign_data : tuple[tuple[int, ...], ...] | None, optional
        Sign bits of the SF coefficients, indexed ``[outer_index][inner_index]``, as
        computed by ``inner_qrom_signs``. They are not loaded, but applied by the
        phase fixup of the erasure (``QROMAdjoint``). If None (default), no signs
        are specified, which is sufficient for Toffoli resource estimation.

    Registers
    ---------
    outer_index : QAny
        Register representing the outer index, :math:`x_o`.
    inner_index : QAny
        Register representing the inner index, :math:`b`.

    alt : QAny, RIGHT
        Alias indices for coherent alias sampling.
    keep : QAny, RIGHT
        Keep thresholds for coherent alias sampling.
    sign : QBit, RIGHT
        1 if the alias term and the sampled entry have opposite signs, :math:`d_i` of
        ``encode_inner_data``.

    G_0 : QBit, RIGHT
        Qubit selecting between SF and :math:`D_1/Q_1`.
    G_1 : QBit, RIGHT
        Qubit selecting between :math:`D_1` and :math:`Q_1`.
    rank : QAny, RIGHT
        Register representing the rank, :math:`r`.

    ctrl : QAny, optional
        Control qubits.

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

    log_block_size: int
    log_block_size_adjoint: int

    num_controls: int = 0

    inner_data: InnerData | None = attrs.field(
        default=None, repr=False, converter=as_inner_data
    )
    sign_data: InnerSignData | None = attrs.field(
        default=None, repr=False, converter=as_inner_sign_data
    )

    def __attrs_post_init__(self) -> None:
        """Validate inputs."""
        if not 1 <= self.log_block_size < self.num_bits_inner:
            err_msg = (
                f"log_block_size (k_2) must satisfy 1 <= k_2 < {self.num_bits_inner} "
                f"(bits of b), got {self.log_block_size}."
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
    def num_rc(self) -> int:
        """Number of SF terms, :math:`RC`."""
        return self.num_ranks * self.num_copies

    @property
    def num_bits_rank(self) -> int:
        """Number of bits needed to represent the rank, :math:`r`."""
        return bit_length(self.num_ranks - 1)

    @property
    def block_size(self) -> int:
        """Number of terms within a single block."""
        return 2**self.log_block_size

    @property
    def block_size_adjoint(self) -> int:
        """Number of terms within a single block of the adjoint."""
        return 2**self.log_block_size_adjoint

    @cached_property
    def control_registers(self) -> tuple[Register, ...]:
        """Control registers of the QROAM."""
        return (Register("ctrl", QAny(self.num_controls)),) if self.num_controls else ()

    @cached_property
    def index_registers(self) -> tuple[Register, ...]:
        """Index registers iterated over."""
        return (
            Register("outer_index", QAny(self.num_bits_outer)),
            Register("inner_index", QAny(self.num_bits_inner)),
        )

    @cached_property
    def alias_registers(self) -> tuple[Register, ...]:
        """Registers used for coherent alias sampling."""
        return (
            Register("alt", QAny(self.num_bits_inner), side=Side.RIGHT),
            Register("keep", QAny(self.num_bits_keep), side=Side.RIGHT),
            Register("sign", QBit(), side=Side.RIGHT),
        )

    @cached_property
    def selector_registers(self) -> tuple[Register, ...]:
        """Registers used to select the operator."""
        return (
            Register("G_0", QBit(), side=Side.RIGHT),
            Register("G_1", QBit(), side=Side.RIGHT),
            Register("rank", QAny(self.num_bits_rank), side=Side.RIGHT),
        )

    @cached_property
    def signature(self) -> Signature:
        """Define input and/or output registers of the bloq."""
        return Signature(
            [
                *self.index_registers,
                *self.alias_registers,
                *self.selector_registers,
                *self.control_registers,
            ]
        )

    @cached_property
    def qrom(self) -> InnerQROM:
        """QROM loading the alias sampling words."""
        return InnerQROM(
            num_orbitals=self.num_orbitals,
            num_ranks=self.num_ranks,
            num_bases=self.num_bases,
            num_copies=self.num_copies,
            num_bits_keep=self.num_bits_keep,
            log_block_size=self.log_block_size,
            num_controls=self.num_controls,
            inner_data=self.inner_data,
            sign_data=self.sign_data,
        )

    @cached_property
    def swap(self) -> SwapWithZero:
        """Swap bringing the addressed alias word to the front."""
        return SwapWithZero(
            (self.log_block_size,),
            self.qrom.alias_word_size,
            self.block_size,
        )

    @cached_property
    def index_partition(self) -> Partition:
        """Partition of the index into high (block) and low (offset) bits."""
        high_dtype = BQUInt(self.qrom.num_bits_blocks, self.qrom.num_blocks)
        low_dtype = BQUInt(self.log_block_size, self.block_size)

        num_bits = self.num_bits_inner
        registers = (
            Register("inner_index_high", high_dtype),
            Register("inner_index_low", low_dtype),
        )
        return Partition(num_bits, registers)

    @cached_property
    def qrom_targets_partition(self) -> Partition:
        """Partition of one register into the QROM's target registers."""
        registers = self.qrom.target_registers
        num_bits = sum(reg.total_bits() for reg in registers)
        return Partition(num_bits, registers)

    @cached_property
    def alias_word_partition(self) -> Partition:
        """Partition of a word into the individual alias sampling registers."""
        return Partition(self.qrom.alias_word_size, self.alias_registers)

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Build call graph for InnerQROAM."""
        return {self.qrom: 1, self.swap: 1}

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        **soqs: SoquetT,
    ) -> dict[str, SoquetT]:
        """Implement bloq decomposition into sub-bloqs."""
        soqs |= bb.add_d(self.index_partition, x=soqs.pop("inner_index"))

        qrom_targets = bb.allocate(self.qrom_targets_partition.n)
        soqs |= bb.add_d(self.qrom_targets_partition, x=qrom_targets)

        soqs |= bb.add_d(self.qrom, **extract_soqs(self.qrom, soqs))

        soqs["inner_index_low"], alias_words = bb.add(
            self.swap,
            selection=soqs.pop("inner_index_low"),
            targets=soqs.pop("alias_words"),
        )

        target_word, *junk_words = alias_words

        for junk in junk_words:
            bb.free(junk, dirty=True)

        soqs |= bb.add_d(self.alias_word_partition, x=target_word)

        soqs["inner_index"] = bb.add(
            self.index_partition.adjoint(),
            **extract_soqs(self.index_partition.adjoint(), soqs),
        )

        return soqs

    def adjoint(self) -> Bloq:
        """Get the measurement-based erasure of this QROAM."""
        return QROMAdjoint(
            self,
            unary_size=self.block_size_adjoint * self.num_inner,
            iteration_size=ceil(self.num_rc / self.block_size_adjoint),
        )


@attrs.frozen
class RotationQROM(UnaryIterationGate):
    """
    QROM loading the Majorana basis rotation angles indexed by inner and rank indices.

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

    num_controls : int, optional
        Number of control qubits (default = 0).

    rotation_data : tuple[tuple[int, ...], ...] | None, optional
        Packed Givens rotation words loaded by the rotation QROM, indexed
        ``[rank][inner_index]`` with each word :math:`(N - 1) b_{rot}` bits wide.
        If None (default), placeholder data is used, which is sufficient for
        Toffoli resource estimation.

    Registers
    ---------
    inner_index : BQUInt
        Register representing the inner index, :math:`b`.

    rank : BQUInt
        Register representing the rank, :math:`r`.

    rotations : QAny
        Givens rotation angles for the basis rotation.

    non_identity : QBit
        Non-identity flag qubit.

    ctrl : QAny, optional
        Control qubits.

    References
    ----------
    .. [Low2025] G. H. Low et al., "Fast Quantum Simulation of Electronic Structure by
       Spectral Amplification", Phys. Rev. X, vol. 15, no. 4, p. 041016, Oct. 2025.
    """

    num_orbitals: int
    num_ranks: int
    num_bases: int

    num_bits_phase_grad: int

    num_controls: int = 0

    rotation_data: RotationData | None = attrs.field(
        default=None, repr=False, converter=as_rotation_data
    )

    def __attrs_post_init__(self) -> None:
        """Validate the shape and bit width of the rotation data."""
        if self.rotation_data is None:
            return

        if len(self.rotation_data) != self.num_ranks or any(
            len(row) != self.num_bases for row in self.rotation_data
        ):
            err_msg = (
                f"`rotation_data` must have shape ({self.num_ranks}, "
                f"{self.num_bases}) indexed [rank][inner_index]."
            )
            raise ValueError(err_msg)

        if any(
            not 0 <= word < 2**self.rotations_size
            for row in self.rotation_data
            for word in row
        ):
            err_msg = f"`rotation_data` words must fit in {self.rotations_size} bits."
            raise ValueError(err_msg)

    @property
    def rotations_size(self) -> int:
        """Rotation register size, :math:`(N - 1) b_{rot}`."""
        return (self.num_orbitals - 1) * self.num_bits_phase_grad

    @property
    def num_bits_rank(self) -> int:
        """Number of bits needed to represent the rank, :math:`r`."""
        return bit_length(self.num_ranks - 1)

    @cached_property
    def control_registers(self) -> tuple[Register, ...]:
        """Control registers of the QROM."""
        return (Register("ctrl", QAny(self.num_controls)),) if self.num_controls else ()

    @cached_property
    def selection_registers(self) -> tuple[Register, ...]:
        """Index registers iterated over by unary iteration."""
        return (
            Register(
                "inner_index", BQUInt(bit_length(self.num_bases), self.num_bases + 1)
            ),
            Register("rank", BQUInt(self.num_bits_rank, self.num_ranks)),
        )

    @cached_property
    def target_registers(self) -> tuple[Register, ...]:
        """Registers the looked-up data is written into."""
        return (
            Register("rotations", QAny(self.rotations_size)),
            Register("non_identity", QBit()),
        )

    @override
    def nth_operation(
        self,
        context: DecompositionContext,
        control: Qid,
        *,
        inner_index: int,
        rank: int,
        rotations: Sequence[Qid],
        non_identity: Sequence[Qid],
    ) -> Iterator[OP_TREE]:  # ty: ignore[invalid-method-override]
        # For b = B output to the flag (check whether to invert or not)
        if inner_index == self.num_bases:
            yield XGate().on(*non_identity).controlled_by(control)

        # For b < B iterate over r and output rotations
        else:
            data = (
                0
                if self.rotation_data is None
                else self.rotation_data[rank][inner_index]
            )
            yield (
                XorK(QAny(self.rotations_size), data)
                .on(*rotations)
                .controlled_by(control)
            )

    @override
    def _break_early(
        self,
        selection_index_prefix: tuple[int, ...],
        l: SymbolicInt,  # noqa: E741 (required for override)
        r: SymbolicInt,
    ) -> bool:

        # Break early if we are iterating over the inner loop, and the outer loop
        # index is greater than or equal to B (skipping those iterations)
        return (
            len(selection_index_prefix) == 1
            and selection_index_prefix[0] >= self.num_bases
        )

    def nth_operation_callgraph(self, inner_index: int, rank: int) -> set[BloqCountT]:
        """Get the data loading cost for each index."""
        # Flag qubit: a single CNOT.
        if inner_index == self.num_bases:
            return {(CNOT(), 1)}

        # Same placeholder data as `nth_operation`, so counts match the decomposition.
        data = (
            0 if self.rotation_data is None else self.rotation_data[rank][inner_index]
        )
        return {(XorK(QAny(self.rotations_size), data).controlled(), 1)}

    def adjoint(self) -> Bloq:
        """Get the measurement-based erasure of this QROM."""
        return QROMAdjoint(
            self,
            unary_size=min(self.num_ranks, self.num_bases),
            iteration_size=max(self.num_ranks, self.num_bases),
        )
