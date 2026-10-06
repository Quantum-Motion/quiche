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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from quiche.core.paulis import PauliSum, PauliWord

from functools import cached_property
from math import ceil, copysign, log2, pi
from typing import Self

import attrs
import numpy as np
from cirq import DensePauliString
from qualtran import (
    AddControlledT,
    Bloq,
    BloqBuilder,
    CtrlSpec,
    QAny,
    QBit,
    Register,
    Signature,
    SoquetT,
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
from qualtran.bloqs.mcmt import And
from qualtran.bloqs.mcmt.specialized_ctrl import (
    get_ctrl_system_1bit_cv_from_bloqs,
)
from qualtran.bloqs.multiplexers.select_pauli_lcu import SelectPauliLCU
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


@attrs.frozen
class SelectPauliLCUWrapper(SelectPauliLCU):
    """
    Wrapper for ``SelectPauliLCU`` with analytic resource counting costs.

    Defines analytic ``call_graph`` and QubitCount ``static_costs`` enabling resource
    costs and counts to be generated without performing a full explicit decomposition of
    the bloq.

    Costs and implementation follow that of [Babbush2018]_.

    References
    ----------
    .. [Babbush2018] R. Babbush et al., 'Encoding Electronic Spectra in Quantum Circuits
        with Linear T Complexity', Phys. Rev. X, vol. 8, no. 4, p. 041015, Oct. 2018,
        doi:10.1103/PhysRevX.8.041015.

    """

    def my_static_costs(self, cost_key: CostKey) -> int:
        """Return qubit counts (not including any rotation-synthesis ancillas)."""
        if isinstance(cost_key, QubitCount):
            num_terms = len(self.select_unitaries)
            num_select_qubits = ceil(log2(num_terms))
            # The controlled version of SelectPauliLCUWrapper is obtained by setting the
            # attribute control_val.
            if self.control_val is None:
                # There is at most a ladder of (num_select_qubits - 1) and_bloqs coming
                # from the unary iteration, which needs num_select_qubits - 1 ancilla.
                # This cost is added to the select and target bloqs, which act on
                # selection_bitsize + target_bitsize qubits.
                return (
                    self.selection_bitsize + self.target_bitsize + num_select_qubits - 1
                )
            # If the bloq is controlled, there are two additional ancilla: One is the
            # control ancilla, and the second is from an additional and_bloq in the
            # unary iteration.
            return self.selection_bitsize + self.target_bitsize + num_select_qubits + 1
        return NotImplemented

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Build call graph for SelectPauliLCU."""
        num_terms = len(self.select_unitaries)
        num_and = num_terms - 1 if self.control_val is not None else num_terms - 2

        bloq_counts = {}

        if num_and > 0:
            bloq_counts[And(cv1=1, cv2=0)] = num_and
            bloq_counts[And().adjoint()] = num_and
            bloq_counts[CNOT()] = num_and

        if self.control_val is None:
            bloq_counts[XGate()] = 2

        for term in self.select_unitaries:
            # Extract the gate corresponding to the relevant Pauli term and transform
            # from cirq to qualtran object.
            bloq = CirqGateAsBloq(term.sparse().gate).controlled()
            # The bloq has no information about which qubits the Paulis act on, and
            # there can be duplicates (e.g. X(0)X(1) and X(0)X(2)). So either create
            # a new key or add to an existing count.
            bloq_counts[bloq] = bloq_counts.get(bloq, 0) + 1

        return bloq_counts


@attrs.frozen
class LCUBlockEncodingWrapper(LCUBlockEncoding):
    r"""
    ``LCUBlockEncoding`` of a QUICHE ``PauliSum``.

    For :math:`H = c_0 I + \sum_{j=1}^{L} h_j P_j`, block encodes
    :math:`H / \lambda` with :math:`\lambda = |c_0| + \sum_j |h_j|`.

    Extends Qualtran's ``LCUBlockEncoding`` with a ``from_hamiltonian``
    constructor and an analytic ``build_call_graph``, so resource counts don't
    require a full decomposition.
    """

    @classmethod
    def from_hamiltonian(cls, pauli_sum: PauliSum, phase_bitsize: int) -> Self:
        """
        Construct the LCU block encoding of a ``PauliSum``.

        Parameters
        ----------
        pauli_sum : PauliSum
            Hamiltonian to block encode.
        phase_bitsize : int
            Size of the register storing the rotation angles in the PREPARE state
            preparation.

        Returns
        -------
        LCUBlockEncodingWrapper
            Block encoding of ``pauli_sum``.

        Raises
        ------
        ValueError
            If the ``phase_bitsize`` is less than 2.

        """
        if phase_bitsize < 2:
            error_msg = "Choose phase_bitsize at least 2."
            raise ValueError(error_msg)

        terms = [u.to_cirq(pauli_sum.num_qubits) for u in pauli_sum.words]
        num_terms = pauli_sum.num_words_with_identity
        lam = pauli_sum.lam
        coeffs = np.array(pauli_sum.coefficients)

        if pauli_sum.has_identity:
            terms.append(DensePauliString.eye(pauli_sum.num_qubits))
            coeffs = np.append(coeffs, [pauli_sum.identity_coefficient])

        terms = [
            term if c >= 0 else -term for term, c in zip(terms, coeffs, strict=True)
        ]
        prep_coeffs = np.sqrt(np.abs(coeffs) / lam)

        # find the number of select qubits
        num_select_qubits = ceil(log2(num_terms))

        # pad coefficients if necessary
        if log2(num_terms) % 1 > 0:
            num_add = int(2**num_select_qubits - num_terms)
            id_string = DensePauliString.eye(pauli_sum.num_qubits)
            terms += [id_string] * num_add
            prep_coeffs = np.append(prep_coeffs, np.zeros(num_add, dtype=np.float64))

        select = SelectPauliLCUWrapper(
            selection_bitsize=num_select_qubits,
            target_bitsize=pauli_sum.num_qubits,
            select_unitaries=tuple(terms),
        )

        prepare_op = StatePreparationViaRotations(
            state_coefficients=prep_coeffs,
            phase_bitsize=phase_bitsize,
        )
        prepare = PrepareFromStatePrep(
            stateprep=prepare_op,
            phase_bitsize=phase_bitsize,
            num_select_qubits=num_select_qubits,
        )

        return cls(prepare=prepare, select=select)

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Build call graph for LCUBlockEncodingWrapper."""
        return {
            self.prepare: 1,
            self.prepare.adjoint(): 1,
            (self.select if self.control_val is None else self.select.controlled()): 1,
        }


@attrs.frozen
class PauliWordRotation(Bloq):
    r"""
    Multi-qubit rotation generated by a given ``PauliWord``.

    For a PauliWord :math:`P` and angle :math:`\theta`, implements
    :math:`\exp(-i \theta P / 2)`.

    Also known as a Pauli gadget or Pauli phasor.

    Parameters
    ----------
    word : PauliWord
        PauliWord defining the rotation axis.
    angle : float
        Angle/coefficient applied within the exponential.
    num_qubits : int
        Explicit number of qubits which the rotation should act on.

    is_controlled : bool, optional
        Whether to generate the optimised, singly-controlled version of the bloq
        (default: False).

    Registers
    ---------
    ctrl : QBit, optional
        Control qubit, if ``is_controlled`` is ``True``.
    system : QAny
        Qubits representing the target system.

    Raises
    ------
    ValueError
        If the highest qubit included in ``word`` is greater than or equal to the
        specified ``num_qubits``.

    """

    word: PauliWord
    angle: float
    num_qubits: int

    is_controlled: bool = False

    def __attrs_post_init__(self) -> None:
        """Validate attributes."""
        if self.word.greatest_qubit >= self.num_qubits:
            message = (
                f"Target qubit {self.word.greatest_qubit} is out of range for a "
                f"{self.num_qubits} qubit register."
            )
            raise ValueError(message)

    @property
    def control_registers(self) -> tuple[Register, ...]:
        """Registers holding the control qubit, if any."""
        return (Register("ctrl", dtype=QBit()),) if self.is_controlled else ()

    @property
    def num_controls(self) -> int:
        """Number of control qubits (1 if controlled, 0 otherwise)."""
        return 1 if self.is_controlled else 0

    @property
    def signature(self) -> Signature:
        """Define input and/or output registers of the bloq."""
        return Signature(
            [*self.control_registers, Register("system", dtype=QAny(self.num_qubits))]
        )

    def __str__(self) -> str:
        """Get human-readable representation."""
        name = "C[PauliWordRotation]" if self.is_controlled else "PauliWordRotation"
        return f"{name}(word={self.word}, angle={self.angle})"

    def get_ctrl_system(self, ctrl_spec: CtrlSpec) -> tuple[Bloq, AddControlledT]:
        """Override function to get controlled bloq."""
        return get_ctrl_system_1bit_cv_from_bloqs(
            self,
            ctrl_spec,
            current_ctrl_bit=1 if self.is_controlled else None,
            bloq_with_ctrl=attrs.evolve(self, is_controlled=True),
            ctrl_reg_name="ctrl",
        )

    def my_static_costs(self, cost_key: CostKey) -> int:
        """Return qubit counts (not including any rotation-synthesis ancillas)."""
        if isinstance(cost_key, QubitCount):
            return self.num_qubits + self.num_controls
        return NotImplemented

    @cached_property
    def to_z_basis_gates(self) -> tuple[tuple[int, tuple[Bloq, ...]], ...]:
        """Gates mapping Pauli X, Y gates to Z basis."""
        gate_map = {
            Pauli.X: (Hadamard(),),
            Pauli.Y: (SGate(is_adjoint=True), Hadamard()),
        }

        return tuple(
            (q, gate_map[p])
            for q, p in zip(self.word.qubits, self.word.paulis, strict=True)
            if p in gate_map
        )

    @property
    def cnot_control_qubits(self) -> tuple[int, ...]:
        """Control qubits to use for each CNOT."""
        greatest = self.word.greatest_qubit
        return tuple(q for q in self.word.qubits if q < greatest)

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        **soqs: SoquetT,
    ) -> dict[str, SoquetT]:
        """Implement bloq decomposition into sub-bloqs."""
        greatest_qubit = self.word.greatest_qubit
        system_qubits = bb.split(soqs["system"])

        for q, ops in self.to_z_basis_gates:
            for op in ops:
                system_qubits[q] = bb.add(op, q=system_qubits[q])

        for q in self.cnot_control_qubits:
            system_qubits[q], system_qubits[greatest_qubit] = bb.add(
                CNOT(),
                ctrl=system_qubits[q],
                target=system_qubits[greatest_qubit],
            )

        if self.is_controlled:
            ctrl = soqs["ctrl"]
            ctrl, system_qubits[greatest_qubit] = bb.add(
                CRz(self.angle),
                ctrl=ctrl,
                q=system_qubits[greatest_qubit],
            )
        else:
            system_qubits[greatest_qubit] = bb.add(
                Rz(self.angle),
                q=system_qubits[greatest_qubit],
            )

        for q in self.cnot_control_qubits:
            system_qubits[q], system_qubits[greatest_qubit] = bb.add(
                CNOT(),
                ctrl=system_qubits[q],
                target=system_qubits[greatest_qubit],
            )

        for q, ops in self.to_z_basis_gates:
            for op in reversed(ops):
                system_qubits[q] = bb.add(op.adjoint(), q=system_qubits[q])

        soqs = {"system": bb.join(system_qubits)}
        if self.is_controlled:
            soqs["ctrl"] = ctrl

        return soqs

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Build call graph for PauliWordRotation."""
        num_x = sum(pauli is Pauli.X for pauli in self.word.paulis)
        num_y = sum(pauli is Pauli.Y for pauli in self.word.paulis)
        num_cnot = 2 * (len(self.word.paulis) - 1)

        bloq_counts = {}

        if self.is_controlled:
            bloq_counts[CRz(self.angle)] = 1
        else:
            bloq_counts[Rz(self.angle)] = 1

        if num_cnot > 0:
            bloq_counts[CNOT()] = num_cnot

        if num_y:
            bloq_counts[SGate(is_adjoint=True)] = num_y
            bloq_counts[SGate()] = num_y

        if num_x + num_y:
            bloq_counts[Hadamard()] = 2 * (num_x + num_y)

        return bloq_counts


@attrs.frozen
class QDRIFT(Bloq):
    r"""
    Randomised unitary time evolution propagator for a given Hamiltonian.

    Approximates the time evolution of the Hamiltonian
    :math:`H = c_0 I + \sum_{j=1}^{L} h_j P_j` by the randomised product of
    [Campbell2019]_

    .. math::

        e^{-iHt} \approx e^{-i c_0 t}
        \prod_{n=1}^{N} e^{-i \tau \, \mathrm{sgn}(h_{j_n}) P_{j_n}},
        \qquad \tau = \frac{\lambda t}{N},

    where :math:`N` is ``num_samples``, :math:`\lambda = \sum_j |h_j|`, and each
    index :math:`j_n` is sampled independently with probability
    :math:`p_j = |h_j| / \lambda`. The identity term is applied as a global
    phase.

    Parameters
    ----------
    sum : PauliSum
        Hamiltonian to construct the propagator for.
    time : float
        Total evolution time.
    num_samples : int
        Total number of QDRIFT samples to use for randomisation.
    seed : int
        Seed used for reproducible randomisation. Must be nonnegative.

    is_controlled : bool, optional
        Whether to generate the optimised, singly-controlled version of the bloq
        (default: False).

    Registers
    ---------
    ctrl : QBit, optional
        Control qubit, if ``is_controlled`` is ``True``.
    system : QAny
        Qubits representing the target system.

    Raises
    ------
    ValueError
        If ``num_samples`` or ``time`` aren't positive.

    References
    ----------
    .. [Campbell2019] E. Campbell, 'Random Compiler for Fast Hamiltonian Simulation',
        Phys. Rev. Lett., vol. 123, no. 7, p. 070503, Aug. 2019,
        doi: 10.1103/PhysRevLett.123.070503.

    """

    sum: PauliSum
    time: float
    num_samples: int
    seed: int = attrs.field(
        validator=[
            attrs.validators.instance_of((int, np.integer)),
            attrs.validators.ge(0),
        ]
    )

    is_controlled: bool = False

    def __attrs_post_init__(self) -> None:
        """Validate attributes."""
        if self.num_samples < 1:
            error_msg = "Choose positive num_samples."
            raise ValueError(error_msg)

        if self.time <= 0:
            error_msg = "Choose positive evolution time."
            raise ValueError(error_msg)

    @property
    def control_registers(self) -> tuple[Register, ...]:
        """Registers holding the control qubit, if any."""
        return (Register("ctrl", dtype=QBit()),) if self.is_controlled else ()

    @property
    def num_controls(self) -> int:
        """Number of control qubits (1 if controlled, 0 otherwise)."""
        return 1 if self.is_controlled else 0

    @property
    def signature(self) -> Signature:
        """Define input and/or output registers of the bloq."""
        return Signature(
            [*self.control_registers, Register("system", dtype=QAny(self.num_qubits))]
        )

    def __str__(self) -> str:
        """Get human-readable representation."""
        name = "C[QDRIFT]" if self.is_controlled else "QDRIFT"
        return (
            f"{name}(sum, time={self.time}, num_samples={self.num_samples}, "
            f"seed={self.seed})"
        )

    __repr__ = __str__

    def get_ctrl_system(self, ctrl_spec: CtrlSpec) -> tuple[Bloq, AddControlledT]:
        """Override function to get controlled bloq."""
        return get_ctrl_system_1bit_cv_from_bloqs(
            self,
            ctrl_spec,
            current_ctrl_bit=1 if self.is_controlled else None,
            bloq_with_ctrl=attrs.evolve(self, is_controlled=True),
            ctrl_reg_name="ctrl",
        )

    def my_static_costs(self, cost_key: CostKey) -> int:
        """Return qubit counts (not including any rotation-synthesis ancillas)."""
        if isinstance(cost_key, QubitCount):
            return self.num_qubits + self.num_controls
        return NotImplemented

    @property
    def positive_coefficients(self) -> tuple[float, ...]:
        """Get absolute value of the coefficients."""
        # Identity deliberately not included here (applied as a global phase)
        return tuple(map(abs, self.sum.coefficients))

    @property
    def lam(self) -> float:
        """Get the 1-norm of the coefficients."""
        return sum(self.positive_coefficients)

    @property
    def num_qubits(self) -> int:
        """Get number of qubits of bloq."""
        return self.sum.num_qubits

    @property
    def dt(self) -> float:
        """Get timestep for each operator."""
        return self.time * self.lam / self.num_samples

    @cached_property
    def sampled_indices(self) -> NDArray[np.int_]:
        """Random sequence of indices for Hamiltonian sampling."""
        rng = np.random.default_rng(self.seed)
        probabilities = np.asarray(self.positive_coefficients) / self.lam
        return rng.choice(
            self.sum.num_words,
            size=self.num_samples,
            p=probabilities,
        )

    def get_rotation(self, index: int) -> PauliWordRotation:
        """Get the individual PauliWordRotation for a given term."""
        return PauliWordRotation(
            word=self.sum.words[index],
            angle=copysign(2 * self.dt, self.sum.coefficients[index]),
            num_qubits=self.num_qubits,
            is_controlled=self.is_controlled,
        )

    @property
    def phase(self) -> Bloq:
        """Get the phase contribution from the identity coefficient."""
        phase = GlobalPhase(exponent=-self.sum.identity_coefficient * self.time / pi)
        return phase.controlled() if self.is_controlled else phase

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        **soqs: SoquetT,
    ) -> dict[str, SoquetT]:
        """Implement bloq decomposition into sub-bloqs."""
        for idx in self.sampled_indices:
            soqs = bb.add_d(self.get_rotation(idx), **soqs)

        if self.is_controlled:
            soqs["ctrl"] = bb.add(self.phase, q=soqs["ctrl"])
        else:
            bb.add(self.phase)
        return soqs

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Compute call graph for QDRIFT."""
        counts = Counter(self.sampled_indices)
        return {
            **{self.get_rotation(i): n for i, n in counts.items()},
            self.phase: 1,
        }


@attrs.frozen
class PauliWordRotationSequence(Bloq):
    """
    Ordered sequence of ``PauliWordRotation`` bloqs, applied as a single bloq.

    Used by ``Trotterisation`` to enable the call graph to be calculated once instead
    of for each step.

    Qualtran provides a similar ``TrotterizedUnitary`` bloq. However, it takes
    bloqs parameterised by an ``angle`` attribute and overwrites it for each term.

    Parameters
    ----------
    rotations : tuple[PauliWordRotation, ...]
        Ordered sequence of rotations to apply. All must share the same signature.

    Raises
    ------
    ValueError
        If ``rotations`` is empty, or if the rotations do not all share the same
        signature.

    """

    rotations: tuple[PauliWordRotation, ...]

    def __attrs_post_init__(self) -> None:
        """Validate attributes."""
        if not self.rotations:
            error_msg = "Rotation sequence must have at least one rotation."
            raise ValueError(error_msg)

        if any(r.signature != self.rotations[0].signature for r in self.rotations):
            error_msg = "All rotations must share the same signature."
            raise ValueError(error_msg)

    @property
    def signature(self) -> Signature:
        """Define input and/or output registers of the bloq."""
        return self.rotations[0].signature

    def __str__(self) -> str:
        """Get human-readable representation."""
        return f"PauliWordRotationSequence(num_rotations={len(self.rotations)})"

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        **soqs: SoquetT,
    ) -> dict[str, SoquetT]:
        """Implement bloq decomposition into sub-bloqs."""
        for rotation in self.rotations:
            soqs = bb.add_d(rotation, **soqs)
        return soqs

    def my_static_costs(self, cost_key: CostKey) -> int:
        """Return qubit counts (not including any rotation-synthesis ancillas)."""
        if isinstance(cost_key, QubitCount):
            return self.signature.n_qubits()
        return NotImplemented

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Build call graph for PauliWordRotationSequence."""
        return Counter(self.rotations)


@attrs.frozen
class Trotterisation(Bloq):
    r"""
    Trotterised unitary time evolution propagator for a given Hamiltonian.

    Approximates the time evolution of the Hamiltonian
    :math:`H = c_0 I + \sum_{j=1}^{L} h_j P_j` by

    .. math::

        e^{-iHt} \approx e^{-i c_0 t} \, S_p(\Delta t)^{n},
        \qquad \Delta t = \frac{t}{n},

    where :math:`n` is ``num_steps`` and :math:`S_p` is the order-:math:`p` product
    formula. The identity term is applied as a global phase.

    For ``order=1``, the Lie-Trotter formula is used,

    .. math::

        S_1(\Delta t) = \prod_{j=1}^{L} e^{-i h_j P_j \Delta t}.

    For ``order=2``, the symmetric (Strang) formula is used,

    .. math::

        S_2(\Delta t) = \prod_{j=L}^{1} e^{-i h_j P_j \Delta t / 2}
                        \prod_{j=1}^{L} e^{-i h_j P_j \Delta t / 2},

    and higher even orders follow Suzuki's recursion,

    .. math::

        S_{2k}(\Delta t) = S_{2k-2}(u_k \Delta t)^2 \,
                           S_{2k-2}\big((1 - 4u_k)\Delta t\big) \,
                           S_{2k-2}(u_k \Delta t)^2,
        \qquad u_k = \frac{1}{4 - 4^{1/(2k-1)}}.

    Parameters
    ----------
    sum : PauliSum
        Hamiltonian to construct the propagator for.
    time : float
        Total evolution time.
    num_steps : int
        Number of Trotter steps the evolution is split into.
    order : int
        Order of the Trotter formula to construct. Must be 1 or a positive even integer.

    is_controlled : bool, optional
        Whether to generate the optimised, singly-controlled version of the bloq
        (default: False).

    Registers
    ---------
    ctrl : QBit, optional
        Control qubit, if ``is_controlled`` is ``True``.
    system : QAny
        Qubits representing the target system.

    Raises
    ------
    ValueError
        If ``num_steps``, ``time`` or ``order`` aren't positive.
        If ``order`` isn't 1 or a positive even integer.

    """

    sum: PauliSum
    time: float
    num_steps: int
    order: int

    is_controlled: bool = False

    def __attrs_post_init__(self) -> None:
        """Validate attributes."""
        if self.num_steps < 1:
            error_msg = "Choose positive num_steps."
            raise ValueError(error_msg)

        if self.time <= 0:
            error_msg = "Choose positive evolution time."
            raise ValueError(error_msg)

        if self.order < 1:
            error_msg = "Choose positive Trotter order."
            raise ValueError(error_msg)

        if self.order > 1 and self.order % 2 == 1:
            error_msg = "Suzuki-Trotter order must be even."
            raise ValueError(error_msg)

    @property
    def control_registers(self) -> tuple[Register, ...]:
        """Registers holding the control qubit, if any."""
        return (Register("ctrl", dtype=QBit()),) if self.is_controlled else ()

    @property
    def num_controls(self) -> int:
        """Number of control qubits (1 if controlled, 0 otherwise)."""
        return 1 if self.is_controlled else 0

    @property
    def signature(self) -> Signature:
        """Define input and/or output registers of the bloq."""
        return Signature(
            [*self.control_registers, Register("system", dtype=QAny(self.num_qubits))]
        )

    def __str__(self) -> str:
        """Get human-readable representation."""
        name = "C[Trotterisation]" if self.is_controlled else "Trotterisation"
        method = "lie-trotter" if self.order == 1 else "suzuki-trotter"
        return (
            f"{name}(sum, method={method}, order={self.order}, time={self.time}, "
            f"num_steps={self.num_steps})"
        )

    __repr__ = __str__

    def get_ctrl_system(self, ctrl_spec: CtrlSpec) -> tuple[Bloq, AddControlledT]:
        """Override function to get controlled bloq."""
        return get_ctrl_system_1bit_cv_from_bloqs(
            self,
            ctrl_spec,
            current_ctrl_bit=1 if self.is_controlled else None,
            bloq_with_ctrl=attrs.evolve(self, is_controlled=True),
            ctrl_reg_name="ctrl",
        )

    def my_static_costs(self, cost_key: CostKey) -> int:
        """Return qubit counts (not including any rotation-synthesis ancillas)."""
        if isinstance(cost_key, QubitCount):
            return self.num_qubits + self.num_controls
        return NotImplemented

    @property
    def num_qubits(self) -> int:
        """Get number of qubits of bloq."""
        return self.sum.num_qubits

    @property
    def dt(self) -> float:
        """Determine the time step size in each step of the Trotterisation."""
        return self.time / self.num_steps

    @property
    def trotter_coeffs(self) -> NDArray[np.float64]:
        """Get the coefficient for each term in the Trotterised propagator."""
        if self.order == 1:
            coeffs = np.ones(self.sum.num_words)
        else:
            coeffs = np.full(2 * self.sum.num_words, 0.5)
            for k in range(4, self.order + 1, 2):
                uk = 1.0 / (4.0 - 4.0 ** (1.0 / (k - 1.0)))
                coeffs = np.kron([uk, uk, 1 - 4 * uk, uk, uk], coeffs)

        return coeffs

    @property
    def trotter_indices(self) -> NDArray[np.int_]:
        """Get the indices for each term in the Trotterised propagator."""
        forward = np.arange(self.sum.num_words)
        if self.order == 1:
            indices = forward
        else:
            sweep = np.concatenate([forward, forward[::-1]])
            indices = np.tile(sweep, 5 ** ((self.order - 2) // 2))

        return indices

    def get_rotation(self, index: int, coeff: float) -> PauliWordRotation:
        """Get the individual PauliWordRotation for a given term and coefficient."""
        # Pauli rotation applies exp(-i θ P/2), so double the angle to get exp(-i θ P)
        angle = 2 * coeff * self.sum.coefficients[index] * self.dt
        return PauliWordRotation(
            word=self.sum.words[index],
            angle=angle,
            num_qubits=self.num_qubits,
            is_controlled=self.is_controlled,
        )

    @cached_property
    def rotation_sequence(self) -> PauliWordRotationSequence:
        """Construct the sequence of rotations for each step in the Trotterisation."""
        return PauliWordRotationSequence(
            tuple(
                self.get_rotation(i, c)
                for c, i in zip(self.trotter_coeffs, self.trotter_indices, strict=True)
            )
        )

    @property
    def phase(self) -> Bloq:
        """Get the phase contribution from the identity coefficient."""
        phase = GlobalPhase(exponent=-self.sum.identity_coefficient * self.time / pi)
        return phase.controlled() if self.is_controlled else phase

    def build_composite_bloq(
        self, bb: BloqBuilder, **soqs: SoquetT
    ) -> dict[str, SoquetT]:
        """Implement the decomposition into sub-bloqs."""
        for _ in range(self.num_steps):
            soqs = bb.add_d(self.rotation_sequence, **soqs)

        if self.is_controlled:
            soqs["ctrl"] = bb.add(self.phase, q=soqs["ctrl"])
        else:
            bb.add(self.phase)
        return soqs

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Compute call graph for Trotterisation."""
        return {self.rotation_sequence: self.num_steps, self.phase: 1}
