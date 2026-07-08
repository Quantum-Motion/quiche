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

"""Hamiltonian simulation routines."""

from __future__ import annotations

from collections import Counter
from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from numpy.typing import NDArray

    from quiche.core.paulis import PauliSum, PauliWord

from math import ceil, log2, pi, sqrt
from random import choices, getstate, seed, setstate
from typing import Self

import attrs
import numpy as np
from attrs import evolve
from cirq import DensePauliString
from qualtran import (
    AddControlledT,
    Bloq,
    BloqBuilder,
    CtrlSpec,
    QAny,
    QBit,
    Register,
    Side,
    Signature,
    SoquetT,
    make_ctrl_system_with_correct_metabloq,
)
from qualtran.bloqs.basic_gates import (
    CNOT,
    CRz,
    GlobalPhase,
    Hadamard,
    Rz,
    SGate,
    XGate,
)
from qualtran.bloqs.block_encoding import LCUBlockEncoding
from qualtran.bloqs.chemistry.trotter.trotterized_unitary import (
    TrotterizedUnitary,
)
from qualtran.bloqs.mcmt import And
from qualtran.bloqs.multiplexers.apply_lth_bloq import ApplyLthBloq
from qualtran.bloqs.multiplexers.select_pauli_lcu import SelectPauliLCU
from qualtran.bloqs.reflections.reflection_using_prepare import ReflectionUsingPrepare
from qualtran.bloqs.state_preparation.state_preparation_via_rotation import (
    StatePreparationViaRotations,
)
from qualtran.cirq_interop import CirqGateAsBloq
from qualtran.resource_counting import (
    BloqCountDictT,
    CostKey,
    QubitCount,
    SympySymbolAllocator,
)

from quiche.core import Pauli, PauliWord

from .state_prep import PrepareFromStatePrep


def _pauli_to_z_string(
    word: PauliWord, bb: BloqBuilder, qs: NDArray[SoquetT]
) -> NDArray[SoquetT]:
    """First half of transforming X, Y Pauli gates to Z and add to BloqBuilder."""
    # Using H Z H = X and (SH) Z (SH)^\dagger = Y, apply the operators preceding the
    # application of the centre Z.
    for q, t in zip(word.qubits, word.terms, strict=True):
        if t is Pauli.X:
            qs[q] = bb.add(Hadamard(), q=qs[q])
        if t is Pauli.Y:
            qs[q] = bb.add(SGate(is_adjoint=True), q=qs[q])
            qs[q] = bb.add(Hadamard(), q=qs[q])

    return qs


def _adjoint_pauli_to_z_string(
    word: PauliWord, bb: BloqBuilder, qs: NDArray[SoquetT]
) -> NDArray[SoquetT]:
    """Second half of transforming X, Y Pauli gates to Z and add to BloqBuilder."""
    # Using H Z H = X and (SH) Z (SH)^\dagger = Y, apply the operators following the
    # application of the centre Z.
    for q, t in zip(word.qubits, word.terms, strict=True):
        if t is Pauli.X:
            qs[q] = bb.add(Hadamard(), q=qs[q])
        if t is Pauli.Y:
            qs[q] = bb.add(Hadamard(), q=qs[q])
            qs[q] = bb.add(SGate(), q=qs[q])

    return qs


def _make_generic_add_controlled(cbloq: Bloq) -> Callable:
    """Create a generic function to add the custom controlled bloq cbloq."""

    def add_controlled(
        bb: BloqBuilder, ctrl_soqs: Sequence[SoquetT], in_soqs: dict[str, SoquetT]
    ) -> tuple[Iterable[SoquetT], Iterable[SoquetT]]:
        # Combine the ctrl soquets with the input soquets
        in_soqs |= {"ctrl": ctrl_soqs}
        # Add the given controlled bloq using the BloqBuilder
        new_out_d = bb.add_d(cbloq, **in_soqs)
        # Extract the ctrl soquets from the resulting soquets. Using .pop()
        # simultaneously modifies the soquet register new_out_d such that only the
        # non-control soquets remain.
        ctrl_soqs = tuple(new_out_d.pop(creg_name) for creg_name in ["ctrl"])
        # return the control soquets and the non-control soquets
        return ctrl_soqs, new_out_d.values()

    return add_controlled


@attrs.frozen
class SOSSABlockEncoding(Bloq):
    """
    Routine for constructing the block encoding of the SOSSA representation.

    If the Hamiltonian can be written as

        H + beta * I = sum_{j=0}^{B-1} A_j† A_j,

    implements the block encoding

        BE[2 H_SOSSA† H_SOSSA / lambda - 1]
                = BE[H_SOSSA/sqrt(lambda)]† REF_a BE[H_SOSSA/sqrt(lambda)]

    described in Lemma 4 in King et al., Phys. Rev. Lett. 136, 110601.

    BE[H_SOSSA/sqrt(lambda)] is implemented in SOSSASqrtBlockEncoding. The reflection
    operator REF_a only acts on the selection qubits of the inner block encodings, and
    it is a reflection about the identity.

    Properties
    ----------
    inner_block_encodings : tuple[LCUBlockEncodingWrapper, ...]
        Block encodings BE[A_j / a_j]
    inner_normalisation_constants : tuple[float, ...]
        Normalisation of the inner block encodings a_j
    outer_phase_bitsize : int
        Phase bitsize for the PREPARE operator
    num_inner_be_ancillas : int
        Number of ancillas of the inner block encodings BE[A_j / a_j]
    num_data : int
        Number of data qubits

    """

    inner_block_encodings: tuple[LCUBlockEncodingWrapper, ...]  # Be[Aj / aj]
    inner_normalisation_constants: tuple[float, ...]  # aj

    outer_phase_bitsize: int

    num_inner_be_ancillas: int  # register a, includes phase bit register as well
    num_data: int  # number of system qubits

    @property
    def signature(self) -> Signature:
        """Define input and output registers of the bloq."""
        return Signature.build(
            outer_be_ancillas=self.num_outer_be_ancillas,
            inner_be_ancillas=self.num_inner_be_ancillas,
            data=self.num_data,
        )

    @property
    def num_outer_be_ancillas(self) -> int:
        """Return number of outer block encoding ancillas."""
        return self.outer_phase_bitsize + self.outer_select_nqubits

    @property
    def num_inner_phase_ancillas(self) -> int:
        """Number of ancillas for representing phase in the inner block encoding."""
        # All the inner block encodings have the same register, so it does not matter
        # which one we use.
        return self.inner_block_encodings[0].prepare.phase_bitsize

    @property
    def num_inner_select_ancillas(self) -> int:
        """Number of ancillas for select operation in the inner block encoding."""
        return self.num_inner_be_ancillas - self.num_inner_phase_ancillas

    @property
    def outer_select_nqubits(self) -> int:
        """Calculate number of select qubits for outer block encoding."""
        return ceil(log2(len(self.inner_block_encodings)))

    @cached_property
    def squareroot_operator(self) -> SOSSASqrtBlockEncoding:
        """Return the SOSSA square root operator."""
        return SOSSASqrtBlockEncoding(
            self.inner_block_encodings,
            self.inner_normalisation_constants,
            self.outer_phase_bitsize,
            self.num_inner_be_ancillas,
            self.num_data,
        )

    @property
    def reflection(self) -> ReflectionUsingPrepare:
        """A reflection around the zero state on the inner ancilla qubits."""
        return ReflectionUsingPrepare.reflection_around_zero(
                bitsizes=(self.num_inner_select_ancillas,),
            )

    def build_composite_bloq(
        self, bb: BloqBuilder, **soqs: SoquetT
    ) -> dict[str, SoquetT]:
        """Build bloq decomposition for SOSSABlockEncoding."""
        outer_be = soqs["outer_be_ancillas"]
        inner_be = soqs["inner_be_ancillas"]
        data = soqs["data"]

        root = self.squareroot_operator

        # add square root block encoding
        outer_be, inner_be, data = bb.add(
            root,
            outer_be_ancillas=outer_be,
            inner_be_ancillas=inner_be,
            data=data,
        )

        # To prepare for the reflection, split up the inner ancillas. The reflection
        # only acts on the qubits involved in the unary operation and not the ones used
        # for preparing the phases.
        inner_be = bb.split(inner_be)
        inner_be_phase = inner_be[:self.num_inner_phase_ancillas]
        inner_be_select = bb.join(inner_be[self.num_inner_phase_ancillas:])

        # Add reflection around the zero state in the inner ancillas
        inner_be_select = bb.add(self.reflection, reg0_=inner_be_select)

        # join registers again
        inner_be_select = bb.split(inner_be_select)
        to_join = list(inner_be_phase) + list(inner_be_select)
        inner_be = bb.join(to_join)

        # add adjoint square root block encoding
        outer_be, inner_be, data = bb.add(
            root.adjoint(),
            outer_be_ancillas=outer_be,
            inner_be_ancillas=inner_be,
            data=data,
        )

        return {
            "outer_be_ancillas": outer_be,
            "inner_be_ancillas": inner_be,
            "data": data,
        }

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Build call graph for SOSSABlockEncoding."""
        return {
            self.squareroot_operator : 1,
            self.squareroot_operator.adjoint() : 1,
            self.reflection : 1
        }

@attrs.frozen
class SOSSASqrtBlockEncoding(Bloq):
    """
    Prepare the block encoding of the square root operator within the SOSSA formalism.

    If the Hamiltonian can be written as

        H + beta * I = sum_{j=0}^{B-1} A_j† A_j,

    then implements the square root operator as described in Lemma 3 in
    King et al., Phys. Rev. Lett. 136, 110601, with

        BE[H_SOSSA/sqrt(lambda)] = SELECT x PREPARE

    where the SELECT and PREPARE operations are

            PREPARE |0>_b = 1/sqrt(lambda) sum_{j=0}^{B-1} a_j |j>_b

            SELECT = sum_{j=0}^{B-1} |j>_b <j|_b x BE[A_j / a_j]

    Properties
    ----------
    inner_block_encodings : tuple[LCUBlockEncodingWrapper, ...]
        Block encodings BE[A_j / a_j]
    inner_normalisation_constants : tuple[float, ...]
        Normalisation of the inner block encodings a_j
    outer_phase_bitsize : int
        Phase bitsize for the PREPARE operator
    num_inner_be_ancillas : int
        Number of ancillas of the inner block encodings BE[A_j / a_j]
    num_data : int
        Number of data qubits

    """

    # TODO: so far assumes all inner block encodings have the same number of ancillas

    inner_block_encodings: tuple[LCUBlockEncodingWrapper, ...]  # Be[Aj / aj]
    inner_normalisation_constants: tuple[float, ...]  # aj

    outer_phase_bitsize: int

    num_inner_be_ancillas: int  # register a, includes phase bit register as well
    num_data: int  # number of system qubits

    @property
    def signature(self) -> Signature:
        """Define input and output registers of the bloq."""
        return Signature.build(
            outer_be_ancillas=self.num_outer_be_ancillas,
            inner_be_ancillas=self.num_inner_be_ancillas,
            data=self.num_data,
        )

    @property
    def num_outer_be_ancillas(self) -> int:
        """Return number of outer block encoding ancillas."""
        return self.outer_phase_bitsize + self.outer_select_nqubits

    @property
    def outer_select_nqubits(self) -> int:
        """Calculate number of select qubits for outer block encoding."""
        return ceil(log2(len(self.inner_block_encodings)))

    @cached_property
    def lamda(self) -> float:
        """Return the block normalisation factor lambda."""
        lamda = 0.
        for aj in self.inner_normalisation_constants:
            lamda += aj**2
        return lamda

    @property
    def prepare(self) -> PrepareFromStatePrep:
        """Create the PREPARE operator for the outer block encoding."""
        coefficients = [
            aj / sqrt(self.lamda) for aj in self.inner_normalisation_constants
        ]
        # pad the coefficients if necessary
        nterms = len(coefficients)
        if log2(nterms) % 1 > 0:
            nadd = int(2**self.outer_select_nqubits - nterms)
            coefficients = np.append(coefficients, np.zeros(nadd, dtype=np.float64))

        prepare_op = StatePreparationViaRotations(
            state_coefficients=coefficients,
            phase_bitsize=self.outer_phase_bitsize,
            control_bitsize=0,
        )
        return PrepareFromStatePrep(
            stateprep=prepare_op,
            phase_bitsize=self.outer_phase_bitsize,
            select_nqubits=self.outer_select_nqubits,
        )

    @property
    def select(self) -> ApplyLthBloq:
        """Create the SELECT operator for the outer block encoding."""
        return ApplyLthBloq(ops=self.inner_block_encodings)

    def build_composite_bloq(
        self, bb: BloqBuilder, **soqs: SoquetT
    ) -> dict[str, SoquetT]:
        """Build bloq decomposition for SOSSASqrtBlockEncoding."""
        outer_be = soqs["outer_be_ancillas"]
        inner_be = soqs["inner_be_ancillas"]
        data = soqs["data"]

        # Prepare on the outer ancilla ('b' register)
        outer_be = bb.add(self.prepare, selection=outer_be)

        # Split outer_be into qubits used for bit-wise phase approximation and for
        # unary iteration. The select operator only uses the unary iteration qubits.
        outer_be = bb.split(outer_be)
        outer_be_phase = outer_be[:self.outer_phase_bitsize]
        outer_be_select = bb.join(outer_be[self.outer_phase_bitsize:])

        # Apply the SELECT operator on the new selection qubits. It inherits the system
        # and ancilla register from self.inner_block_encodings.
        outer_be_select, inner_be, data = bb.add(
            self.select, selection=outer_be_select, ancilla=inner_be, system=data
        )

        # The outgoing register must have the same sizes as the incoming, so join the
        # qubits for the outer block encoding back up.
        outer_be_select = bb.split(outer_be_select)
        to_join = list(outer_be_phase) + list(outer_be_select)
        outer_be = bb.join(to_join)

        return {
            "outer_be_ancillas": outer_be,
            "inner_be_ancillas": inner_be,
            "data": data,
        }

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Build the call graph for SOSSASqrtBlockEncoding."""
        return {
            self.select : 1,
            self.prepare : 1
        }

@attrs.frozen
class SelectPauliLCUWrapper(SelectPauliLCU):
    """Wrapper for SelectPauliLCU for resource counting."""

    # Implements the Select operator described in Phys. Rev. X 8, 041015 (2018)

    def my_static_costs(self, cost_key: CostKey) -> int:
        """Return hard-coded qubit counts."""
        if isinstance(cost_key, QubitCount):
            nterms = len(self.select_unitaries)
            n_select_qubits = ceil(log2(nterms))
            # The controlled version of SelectPauliLCUWrapper is obtained by setting the
            # attribute control_val.
            if self.control_val is None:
                # There is at most a ladder of (n_select_qubits - 1) and_bloqs coming
                # from the unary iteration, which needs n_select_qubits - 1 ancilla.
                # This cost is added to the select and target bloqs, which act on
                # selection_bitsize + target_bitsize qubits.
                return (
                    self.selection_bitsize + self.target_bitsize + n_select_qubits - 1
                )
            # If the bloq is controlled, there are two additional ancilla: One is the
            # control ancilla, and the second is from an additional and_bloq in the
            # unary iteration.
            return self.selection_bitsize + self.target_bitsize + n_select_qubits + 1
        return NotImplemented

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Build call graph for SelectPauliLCU."""
        # Some of the logic in here could be streamlined, but for now leaving it
        # relatively explicit.
        nterms = len(self.select_unitaries)
        bloq_counts = {
            And(cv1=1, cv2=0): nterms - 2,
            And().adjoint(): nterms - 2,
            CNOT(): nterms - 2,
            XGate(): 2,
        }
        for term in self.select_unitaries:
            # Extract the gate corresponding to the relevant Pauli term and transform
            # from cirq to qualtran object.
            bloq = CirqGateAsBloq(term.sparse().gate).controlled()
            # The bloq has no information about which qubits the Paulis act on, and
            # there can be duplicates (e.g. X(0)X(1) and X(0)X(2)). So either create
            # a new key or add to an existing count.
            bloq_counts[bloq] = bloq_counts.get(bloq, 0) + 1

        # Add relevant bloqs if SelectPauliLCU is controlled.
        if self.control_val is not None:
            bloq_counts[And(cv1=1, cv2=0)] += 1
            bloq_counts[And().adjoint()] += 1
            bloq_counts[CNOT()] += 1
            bloq_counts.pop(XGate())

        # Remove empty bloqs.
        if nterms == 2 and self.control_val is None:
            bloq_counts.pop(And(cv1=1, cv2=0))
            bloq_counts.pop(And().adjoint())
            bloq_counts.pop(CNOT())
        return bloq_counts


@attrs.frozen
class LCUBlockEncodingWrapper(LCUBlockEncoding):
    """Class that welds the Qualtran to the QUICHE bloq."""

    @classmethod
    def from_hamiltonian(cls,
                         h: PauliSum,
                         phase_bitsize: int,
                         n_data_qubits: None | int = None
                        ) -> Self:
        """Process the input arguments and return an LCU block encoding."""
        #############
        # VALIDATION
        #############
        # Check phase_bitsize large enough.
        if phase_bitsize < 2:
            error_msg = "Choose phase_bitsize at least 2."
            raise ValueError(error_msg)

        # If number of qubits is not specified during constructor, derive it from the
        # Hamiltonian. This is to allow for cases where block encodings with different
        # h.n_qubits should be concatenated.
        if n_data_qubits is None:
            n_data_qubits = h.n_qubits

        #############
        # COMPUTE BLOQ
        #############
        terms = [u.to_cirq(n_data_qubits) for u in h.terms]
        nterms = h.n_terms
        lam = h.lam
        coeffs = np.array(h.coefficients, dtype=complex)

        # Add the identity term in the Hamiltonian, if needed
        if h.identity_coefficient != 0.:
            terms.append(DensePauliString.eye(n_data_qubits))
            nterms += 1
            lam += abs(h.identity_coefficient)
            coeffs = np.append(coeffs, [h.identity_coefficient])

        prep_coeffs = np.sqrt(np.array(coeffs, dtype=complex) / lam)

        # find the number of select qubits
        select_nqubits = max(ceil(log2(nterms)), 1)

        # pad coefficients if necessary
        if log2(nterms) % 1 > 0:
            nadd = int(2**select_nqubits - nterms)
            id_string = DensePauliString.eye(n_data_qubits)
            terms += [id_string] * nadd
            prep_coeffs = np.append(prep_coeffs, np.zeros(nadd, dtype=np.float64))

        # create SELECT and PREP operators
        select = SelectPauliLCUWrapper(
            selection_bitsize=select_nqubits,
            target_bitsize=h.n_qubits,
            select_unitaries=terms,
        )

        prepare_op = StatePreparationViaRotations(
            state_coefficients=prep_coeffs,
            phase_bitsize=phase_bitsize,
        )
        prepare = PrepareFromStatePrep(
            stateprep=prepare_op,
            phase_bitsize=phase_bitsize,
            select_nqubits=select_nqubits,
        )

        return cls(prepare=prepare, select=select)

    @cached_property
    def signature(self) -> Signature:
        # Base class has register 'selection', which would lead to a duplicate when
        # wrapping an instance inside ApplyLthBloq. Instead, we use 'ancilla' for this
        # register and 'system' for the data qubits in analogy with the signature of
        # LinearCombination.
        return Signature.build_from_dtypes(
            ctrl=QAny(1) if self.control_val else QAny(0),
            ancilla=QAny(self.ancilla_bitsize),
            system=QAny(self.system_bitsize),
        )

    def get_ctrl_system(self, ctrl_spec: 'CtrlSpec') -> tuple['Bloq', 'AddControlledT']:
        from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv_from_bloqs

        return get_ctrl_system_1bit_cv_from_bloqs(
            self,
            ctrl_spec,
            current_ctrl_bit=1 if self.control_val else None,
            bloq_with_ctrl=evolve(self, control_val=1),
            ctrl_reg_name="ctrl",
        )

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        **soqs: SoquetT,
    ) -> dict[str, SoquetT]:
        # Call the super bloq decomposition (from LCUBlockEncoding) to accomodate for
        # the different naming of the registers.
        if self.control_val:
            soqs = super().build_composite_bloq(bb=bb, target=soqs["system"], selection=soqs["ancilla"], ctrl=soqs["ctrl"])
            return { "ancilla": soqs["selection"], "system": soqs["target"],"ctrl": soqs["ctrl"]}
        soqs = super().build_composite_bloq(bb=bb, target=soqs["system"], selection=soqs["ancilla"])
        return {"ancilla": soqs["selection"], "system": soqs["target"]}

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Build call graph for LCUBlockEncodingWrapper."""
        return {
            self.prepare: 1,
            self.prepare.adjoint(): 1,
            (self.select if self.control_val is None else self.select.controlled()): 1,
        }


@attrs.frozen
class PauliWordRotation(Bloq):
    """Routine constructing the Pauli phasor for a given Pauli."""

    word: PauliWord
    angle: float
    n_qubits: int

    def my_static_costs(self, cost_key: CostKey) -> int:
        """Return hard-coded qubit counts."""
        if isinstance(cost_key, QubitCount):
            # This bloq needs a rotation gate, which may require ancilla qubits. But
            # the rotation gate is counted separately and the number of ancilla qubits
            # is calculated in post-processing.
            # So only the data qubits are counted here.
            return self.n_qubits
        return NotImplemented

    @property
    def signature(self) -> Signature:
        """Define input and/or output registers of the bloq."""
        return Signature.build(system=self.n_qubits)

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        **soqs: SoquetT,
    ) -> dict[str, SoquetT]:
        """Implement bloq decomposition into sub-bloqs."""
        if max(self.word.qubits) >= self.n_qubits:
            message = (
                f"Target qubit {max(self.word.qubits)} is out of range for a "
                f"{self.n_qubits} qubit register."
            )
            raise ValueError(message)

        qs = bb.split(soqs["system"])

        qs = _pauli_to_z_string(self.word, bb, qs)

        gw = self.word.greatest_qubit

        for q in self.word.qubits:
            if q < gw:
                qs[q], qs[gw] = bb.add(CNOT(), ctrl=qs[q], target=qs[gw])

        # Rz assumes that self.angle contains any necessary scaling factors
        qs[gw] = bb.add(Rz(self.angle), q=qs[gw])

        # Note that the order of CNOTs doesn't matter since they commute
        for q in self.word.qubits:
            if q < gw:
                qs[q], qs[gw] = bb.add(CNOT(), ctrl=qs[q], target=qs[gw])

        qs = _adjoint_pauli_to_z_string(self.word, bb, qs)

        return {"system": bb.join(qs)}

    def get_ctrl_system(self, ctrl_spec: CtrlSpec) -> tuple[Bloq, AddControlledT]:
        """Override function to get controlled bloq."""
        cbloq = CTRLPauliWordRotation.from_pauliwordrotation(self)

        if ctrl_spec == CtrlSpec():
            add_controlled = _make_generic_add_controlled(cbloq)
            return cbloq, add_controlled

        return make_ctrl_system_with_correct_metabloq(self, ctrl_spec=ctrl_spec)

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Build call graph for PauliWorldRotation."""
        n_x = np.sum([term == Pauli.X for term in self.word.terms])
        n_y = np.sum([term == Pauli.Y for term in self.word.terms])
        return {
            CNOT(): 2 * (len(self.word.terms) - 1),
            SGate(is_adjoint=True): n_y,
            SGate(): n_y,
            Hadamard(): 2 * (n_x + n_y),
            Rz(self.angle): 1,
        }


@attrs.frozen
class CTRLPauliWordRotation(Bloq):
    """Routine constructing the controlled Pauli phasor for a given Pauli."""

    word: PauliWord
    angle: float
    n_qubits: int
    n_controls: int

    @classmethod
    def from_pauliwordrotation(cls, rotation: PauliWordRotation) -> Self:
        """Initialise instance from PauliWordRotation."""
        return cls(rotation.word, rotation.angle, rotation.n_qubits, n_controls=1)

    def my_static_costs(self, cost_key: CostKey) -> int:
        """Return hard-coded qubit counts."""
        if isinstance(cost_key, QubitCount):
            # This bloq needs a rotation gate, which may require ancilla qubits. But
            # the rotation gate is counted separately and the number of ancilla qubits
            # is calculated in post-processing.
            # So only the data qubits and controls are counted here.
            return self.n_qubits + self.n_controls
        return NotImplemented

    @property
    def signature(self) -> Signature:
        """Define input and/or output registers of the bloq."""
        return Signature(
            [
                Register("ctrl", dtype=QBit(), side=Side.THRU),
                Register("system", dtype=QAny(self.n_qubits), side=Side.THRU),
            ]
        )

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        **soqs: SoquetT,
    ) -> dict[str, SoquetT]:
        """Implement bloq decomposition into sub-bloqs."""
        if max(self.word.qubits) >= self.n_qubits:
            message = (
                f"Target qubit {max(self.word.qubits)} is out of range for a "
                f"{self.n_qubits} qubit register."
            )
            raise ValueError(message)

        ctrl = soqs["ctrl"]
        qs = bb.split(soqs["system"])

        qs = _pauli_to_z_string(self.word, bb, qs)

        gw = self.word.greatest_qubit

        for q in self.word.qubits:
            if q < gw:
                qs[q], qs[gw] = bb.add(CNOT(), ctrl=qs[q], target=qs[gw])

        # CRz assumes that self.angle contains any necessary scaling factors
        ctrl, qs[gw] = bb.add(CRz(self.angle), ctrl=ctrl, q=qs[gw])

        # Note that the order of CNOTs doesn't matter since they commute
        for q in self.word.qubits:
            if q < gw:
                qs[q], qs[gw] = bb.add(CNOT(), ctrl=qs[q], target=qs[gw])

        qs = _adjoint_pauli_to_z_string(self.word, bb, qs)

        return {"ctrl": ctrl, "system": bb.join(qs)}

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Build call graph for CTRLPauliWorldRotation."""
        n_x = np.sum([term == Pauli.X for term in self.word.terms])
        n_y = np.sum([term == Pauli.Y for term in self.word.terms])
        return {
            CNOT(): 2 * (len(self.word.terms) - 1),
            SGate(is_adjoint=True): n_y,
            SGate(): n_y,
            Hadamard(): 2 * (n_x + n_y),
            CRz(self.angle): 1,
        }


@attrs.frozen
class QDRIFT(Bloq):
    """QDRIFT implementation for a given linear combination of Paulis."""

    h: PauliSum
    t: float
    n_terms: int
    seed: None | int | float | str | bytes | bytearray = None

    def __attrs_post_init__(self) -> Self:
        """Validate attributes."""
        if self.n_terms < 1:
            error_msg = "Choose positive n_terms."
            raise ValueError(error_msg)

        if self.t < 0:
            error_msg = "Choose positive evolution time."
            raise ValueError(error_msg)

        return self

    def my_static_costs(self, cost_key: CostKey) -> int:
        """Return hard-coded qubit counts."""
        if isinstance(cost_key, QubitCount):
            # This bloq needs a number of rotation gates, which may require ancilla
            # qubits. However, the rotation gates are counted separately and the number
            # of ancilla qubits is calculated in post-processing.
            # So only the data qubits are counted.
            return self.n_qubits
        return NotImplemented

    @property
    def signature(self) -> Signature:
        """Define input and/or output registers of the bloq."""
        return Signature.build(simulation=self.h.n_qubits)

    def __str__(self) -> str:
        """Get human-readable representation."""
        return f"QDRIFT(h, t={self.t}, n_terms={self.n_terms}, seed={self.seed})"

    __repr__ = __str__

    @property
    def positive_coefficients(self) -> tuple[float, ...]:
        """Get absolute value of the coefficients."""
        return tuple(map(abs, self.h.coefficients))

    @property
    def lam(self) -> float:
        """Get the 1-norm of the coefficients."""
        return sum(self.positive_coefficients)

    @property
    def n_qubits(self) -> int:
        """Get number of qubits of bloq."""
        return self.h.n_qubits

    @property
    def dt(self) -> float:
        """Get timestep for each operator."""
        return self.t * self.lam / self.n_terms

    def sample_term_indices(self) -> tuple[int, ...]:
        """Generate random sequence for Hamiltonian sampling."""
        rng_state = getstate()
        if self.seed:
            seed(self.seed)
        c = choices(range(self.h.n_terms), self.positive_coefficients, k=self.n_terms)  # noqa: S311
        setstate(rng_state)
        return tuple(c)

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        **soqs: SoquetT,
    ) -> dict[str, SoquetT]:
        """Implement bloq decomposition into sub-bloqs."""
        simulation = soqs["simulation"]

        # Build the PauliWordRotation for each term in the Hamiltonian.
        bloqs = tuple(
            PauliWordRotation(t, self.dt, self.n_qubits) for t in self.h.terms
        )
        indices = self.sample_term_indices()
        # The frequency with which indices are sampled already accounts for the term's
        # coefficient in the Hamiltonian, so only need to record the sign of the
        # coefficient.
        coeffs = tuple(-1 if self.h.coefficients[i] < 0 else 1 for i in indices)
        # Use the sampled indices and the pre-built individual propagators to build the
        # full propagator.
        t = TrotterizedUnitary(bloqs, indices, coeffs, self.dt)
        simulation = bb.add(t, system=simulation)

        # Add the constant term as a global phase.
        bb.add(GlobalPhase(exponent=-self.h.identity_coefficient / pi))
        return {"simulation": simulation}

    def get_ctrl_system(self, ctrl_spec: CtrlSpec) -> tuple[Bloq, AddControlledT]:
        """Override function to get controlled bloq."""
        cbloq = CTRLQDRIFT(self)

        if ctrl_spec == CtrlSpec():
            add_controlled = _make_generic_add_controlled(cbloq)
            return cbloq, add_controlled

        return make_ctrl_system_with_correct_metabloq(self, ctrl_spec=ctrl_spec)

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Compute call graph for CTRLQDRIFT."""
        # Calculate a Counter that counts the frequency of each index in the sampled
        # configuration.
        sampled_indices = self.sample_term_indices()
        index_counts = Counter(sampled_indices)

        bloq_counts = {}
        # For each index in the Counter, add the relevant bloq to the count.
        for idx, count in index_counts.items():
            word = self.h.terms[idx]
            sign = -1 if self.h.coefficients[idx] < 0 else 1
            angle = sign * self.dt

            # Create the corresponding Bloq. Double the angle because Qualtran
            # convention halves it
            gate = PauliWordRotation(word, angle=2 * angle, n_qubits=self.n_qubits)

            bloq_counts[gate] = count

        # Add the global phase.
        bloq_counts[GlobalPhase(exponent=-self.h.identity_coefficient / pi)] = 1
        return bloq_counts


@attrs.frozen
class CTRLQDRIFT(Bloq):
    """Controlled QDRIFT application."""

    simulation: QDRIFT
    n_controls: int = 1

    def my_static_costs(self, cost_key: CostKey) -> int:
        """Return hard-coded qubit counts."""
        if isinstance(cost_key, QubitCount):
            # This bloq needs a number of rotation gates, which may require ancilla
            # qubits. However, the rotation gates are counted separately and the number
            # of ancilla qubits is calculated in post-processing.
            # So only the data qubits and controls are counted.
            return self.simulation.n_qubits + self.n_controls
        return NotImplemented

    @property
    def signature(self) -> Signature:
        """Define input and/or output registers of the bloq."""
        return Signature(
            [
                Register(
                    "ctrl",
                    dtype=QAny(self.n_controls),
                    side=Side.THRU,
                ),
                Register(
                    "simulation",
                    dtype=QAny(self.simulation.n_qubits),
                    side=Side.THRU,
                ),
            ],
        )

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        **soqs: SoquetT,
    ) -> dict[str, SoquetT]:
        """Implement bloq decomposition into sub-bloqs."""
        ctrl = soqs["ctrl"]
        simulation = soqs["simulation"]

        # Build controlled bloqs for each term in the Hamiltonian. Otherwise proceeds
        # the same as build_composite_bloq in QDRIFT.
        bloqs = tuple(
            PauliWordRotation(
                t,
                self.simulation.dt,
                self.simulation.n_qubits,
            ).controlled()
            for t in self.simulation.h.terms
        )
        indices = self.simulation.sample_term_indices()
        coeffs = tuple(
            -1 if self.simulation.h.coefficients[i] < 0 else 1 for i in indices
        )
        t = TrotterizedUnitary(bloqs, indices, coeffs, self.simulation.dt)
        ctrl, simulation = bb.add(t, ctrl=ctrl, system=simulation)

        # Add the constant term as controlled global phase
        ctrl = bb.add(
            GlobalPhase(
                exponent=-self.simulation.h.identity_coefficient / pi
            ).controlled(),
            q=ctrl,
        )
        return {"ctrl": ctrl, "simulation": simulation}

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Compute call graph for CTRLQDRIFT."""
        # See build_call_graph in QDRIFT for an explanation.
        sampled_indices = self.simulation.sample_term_indices()
        index_counts = Counter(sampled_indices)

        bloq_counts = {}

        for idx, count in index_counts.items():
            word = self.simulation.h.terms[idx]
            sign = -1 if self.simulation.h.coefficients[idx] < 0 else 1
            angle = sign * self.simulation.dt

            # Create the corresponding Bloq. Double the angle because Qualtran
            # convention halves it
            gate = PauliWordRotation(
                word, angle=2 * angle, n_qubits=self.simulation.n_qubits
            ).controlled()

            bloq_counts[gate] = count

        bloq_counts[
            GlobalPhase(
                exponent=-self.simulation.h.identity_coefficient / pi
            ).controlled()
        ] = 1
        return bloq_counts


@attrs.frozen
class Trotterisation(Bloq):
    """General Trotter product formulae."""

    h: PauliSum
    t: float  # total evolution time
    n_steps: int
    order: int

    def __attrs_post_init__(self) -> Self:
        """Validate attributes."""
        if self.n_steps < 1:
            error_msg = "Choose positive n_steps."
            raise ValueError(error_msg)

        if self.t < 0:
            error_msg = "Choose positive evolution time."
            raise ValueError(error_msg)

        if self.order < 1:
            error_msg = "Choose positive Trotter order."
            raise ValueError(error_msg)

        if self.order > 1 and self.order % 2 == 1:
            error_msg = "Suzuki-Trotter order must be even."
            raise ValueError(error_msg)

        return self

    def my_static_costs(self, cost_key: CostKey) -> int:
        """Return hard-coded qubit counts."""
        if isinstance(cost_key, QubitCount):
            # This bloq needs a number of rotation gates, which may require ancilla
            # qubits. However, the rotation gates are counted separately and the number
            # of ancilla qubits is calculated in post-processing.
            # So only the data qubits are counted.
            return self.n_qubits
        return NotImplemented

    def __str__(self) -> str:
        """Get human-readable representation."""
        method = "lie-trotter" if self.order == 1 else "suzuki-trotter"
        return (
            f"Trotterisation(h, method={method}, order = {self.order}, t={self.t}, "
            f"n_steps={self.n_steps})"
        )

    __repr__ = __str__

    @property
    def signature(self) -> Signature:
        """Define input and/or output registers of the bloq."""
        return Signature.build(simulation=self.h.n_qubits)

    @property
    def _trotter_error_bound(self) -> float:
        """Compute an error bound. The total error is trotter_error_bound * t^{p+1}."""
        return self.h.lam ** (self.order + 1.0)

    @property
    def n_qubits(self) -> int:
        """Get number of qubits of bloq."""
        return self.h.n_qubits

    @property
    def dt(self) -> float:
        """Determine the time step size in each step of the Trotterization."""
        return self.t / self.n_steps

    def get_coeffs_indices(
        self, order: int | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
        """Return the coefficients and indices for the Trotter product."""
        # If order is not passed, use self.order
        if order is None:
            order = self.order

        # If order is 1 or 2, return the appropriate coefficients and indices. Otherwise
        # call the function recursively with decreased order.
        if order == 1:
            coeffs = np.ones(self.h.n_terms)
            indices = np.arange(self.h.n_terms)
        elif order == 2:
            coeffs = 0.5 * np.ones(2 * self.h.n_terms - 1)
            coeffs[self.h.n_terms - 1] = 1.0
            indices = np.concatenate(
                [np.arange(self.h.n_terms - 1), np.arange(self.h.n_terms)[::-1]]
            )
        else:
            uk = 1.0 / (4.0 - 4.0 ** (1.0 / (order - 1.0)))
            coeffs_lower, indices_lower = self.get_coeffs_indices(order - 2)
            coeffs_lr = np.concatenate(
                [
                    uk * coeffs_lower[:-1],
                    [2 * uk * coeffs_lower[-1]],
                    uk * coeffs_lower[-2::-1],
                ]
            )
            coeffs_c = (1 - 4 * uk) * coeffs_lower

            coeffs = np.concatenate(
                [
                    coeffs_lr[:-1],
                    [coeffs_lr[-1] + coeffs_c[0]],
                    coeffs_c[1:-1],
                    [coeffs_lr[-1] + coeffs_c[0]],
                    coeffs_lr[-2::-1],
                ]
            )
            indices = np.concatenate([indices_lower, np.tile(indices_lower[1:], 4)])

        return coeffs, indices

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        **soqs: SoquetT,
    ) -> dict[str, SoquetT]:
        """Implement the decomposition into sub-bloqs."""
        simulation = soqs["simulation"]

        # Build the PauliWordRotation for each term in the Hamiltonian.
        bloqs = tuple(
            PauliWordRotation(t, self.dt, self.n_qubits) for t in self.h.terms
        )
        # Extract coefficients and indices for the given Trotter formula.
        coeffs, indices = self.get_coeffs_indices(self.order)

        # Multiply the coefficients from the Trotter formula by the associated term's
        # coefficient in the Hamiltonian to get its effective coefficient.
        for i in range(len(indices)):
            coeffs[i] *= self.h.coefficients[indices[i]]
        coeffs, indices = tuple(coeffs), tuple(indices)

        # Use the sampled indices, the effective coefficients and the pre-built
        # individual propagators to build the full propagator
        u = TrotterizedUnitary(bloqs, indices, coeffs, self.dt)
        # The Trotter formula is applied self.n_steps times.
        for _ in range(self.n_steps):
            simulation = bb.add(u, system=simulation)

        # Add the constant term as a global phase.
        bb.add(GlobalPhase(exponent=-self.h.identity_coefficient * self.t / pi))
        return {"simulation": simulation}

    def get_ctrl_system(self, ctrl_spec: CtrlSpec) -> tuple[Bloq, AddControlledT]:
        """Override function to get controlled bloq."""
        cbloq = CTRLTrotterisation(self)

        if ctrl_spec == CtrlSpec():
            add_controlled = _make_generic_add_controlled(cbloq)
            return cbloq, add_controlled

        return make_ctrl_system_with_correct_metabloq(self, ctrl_spec=ctrl_spec)

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Compute call graph for Trotterisation."""
        # Calculate a Counter that counts the frequency of each index in the sampled
        # configuration. For general Trotter products, the rotation angle depends on the
        # sampled coefficient so it needs to be included in the Counter.
        coeffs, indices = self.get_coeffs_indices(self.order)
        joint_coeffs_indices = zip(coeffs, indices, strict=True)

        index_counts = Counter(joint_coeffs_indices)
        bloq_counts = {}

        # For each index in the Counter, add the relevant bloq to the count.
        for joint_coeff_idx, count in index_counts.items():
            coeff = joint_coeff_idx[0]
            idx = joint_coeff_idx[1]
            word = self.h.terms[idx]
            angle = coeff * self.h.coefficients[idx] * self.dt

            # Create the corresponding Bloq. Double the angle because Qualtran
            # convention halves it
            gate = PauliWordRotation(word, angle=2 * angle, n_qubits=self.n_qubits)

            bloq_counts[gate] = count * self.n_steps

        # Add the global phase.
        bloq_counts[
            GlobalPhase(exponent=-self.h.identity_coefficient * self.t / pi)
        ] = 1
        return bloq_counts


@attrs.frozen
class CTRLTrotterisation(Bloq):
    """Controlled QDRIFT application."""

    simulation: Trotterisation
    n_controls: int = 1

    def my_static_costs(self, cost_key: CostKey) -> int:
        """Return hard-coded qubit counts."""
        if isinstance(cost_key, QubitCount):
            # This bloq needs a number of rotation gates, which may require ancilla
            # qubits. However, the rotation gates are counted separately and the number
            # of ancilla qubits is calculated in post-processing.
            # So only the data qubits and controls are counted.
            return self.simulation.n_qubits + self.n_controls
        return NotImplemented

    @property
    def signature(self) -> Signature:
        """Define input and/or output registers of the bloq."""
        return Signature(
            [
                Register(
                    "ctrl",
                    dtype=QAny(self.n_controls),
                    side=Side.THRU,
                ),
                Register(
                    "simulation",
                    dtype=QAny(self.simulation.n_qubits),
                    side=Side.THRU,
                ),
            ],
        )

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        **soqs: SoquetT,
    ) -> dict[str, SoquetT]:
        """Implement the decomposition into sub-bloqs."""
        ctrl = soqs["ctrl"]
        simulation = soqs["simulation"]

        # Build controlled bloqs for each term in the Hamiltonian. Otherwise proceeds
        # the same as build_composite_bloq in Trotterisation.
        bloqs = tuple(
            PauliWordRotation(
                t, self.simulation.dt, self.simulation.n_qubits
            ).controlled()
            for t in self.simulation.h.terms
        )
        coeffs, indices = self.simulation.get_coeffs_indices(self.simulation.order)
        for i in range(len(indices)):
            coeffs[i] *= self.simulation.h.coefficients[indices[i]]
        coeffs, indices = tuple(coeffs), tuple(indices)
        u = TrotterizedUnitary(bloqs, indices, coeffs, self.simulation.dt)
        for _ in range(self.simulation.n_steps):
            ctrl, simulation = bb.add(u, ctrl=ctrl, system=simulation)

        # Add controlled global phase.
        ctrl = bb.add(
            GlobalPhase(
                exponent=-self.simulation.h.identity_coefficient
                * self.simulation.t
                / pi
            ).controlled(),
            q=ctrl,
        )
        return {"ctrl": ctrl, "simulation": simulation}

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Compute call graph for CTRLTrotterisation."""
        # See build_call_graph in Trotterisation for an explanation.
        coeffs, indices = self.simulation.get_coeffs_indices(self.simulation.order)
        joint_coeffs_indices = zip(coeffs, indices, strict=True)

        index_counts = Counter(joint_coeffs_indices)
        bloq_counts = {}

        for joint_coeff_idx, count in index_counts.items():
            coeff = joint_coeff_idx[0]
            idx = joint_coeff_idx[1]
            word = self.simulation.h.terms[idx]
            angle = coeff * self.simulation.h.coefficients[idx] * self.simulation.dt

            # Create the corresponding Bloq. Double the angle because Qualtran
            # convention halves it
            gate = PauliWordRotation(
                word, angle=2 * angle, n_qubits=self.simulation.n_qubits
            ).controlled()

            bloq_counts[gate] = count * self.simulation.n_steps

        bloq_counts[
            GlobalPhase(
                exponent=-self.simulation.h.identity_coefficient
                * self.simulation.t
                / pi
            ).controlled()
        ] = 1
        return bloq_counts
