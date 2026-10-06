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

"""State preparation routines."""

import attrs
from qualtran import Bloq, BloqBuilder, QAny, Register, Side, Signature, SoquetT
from qualtran.bloqs.basic_gates import XGate
from qualtran.bloqs.bookkeeping import Allocate
from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle
from qualtran.bloqs.state_preparation.state_preparation_via_rotation import (
    StatePreparationViaRotations,
)
from qualtran.resource_counting import (
    BloqCountDictT,
    CostKey,
    QubitCount,
    SympySymbolAllocator,
)


@attrs.frozen
class IdentityStatePrep(Bloq):
    r"""
    Trivial preparation of the all-zero state :math:`|0\rangle^{\otimes n}`.

    Parameters
    ----------
    num_qubits : int
        Number of qubits, :math:`n`, in register to initialise.

    Registers
    ---------
    q : QAny, RIGHT
        Newly-allocated qubit register initialised to zero state.

    """

    num_qubits: int

    def my_static_costs(self, cost_key: CostKey) -> int:
        """Return qubit counts (not including any rotation-synthesis ancillas)."""
        if isinstance(cost_key, QubitCount):
            return self.num_qubits
        return NotImplemented

    @property
    def signature(self) -> Signature:
        """Define input and/or output registers of the bloq."""
        return Signature([Register("q", dtype=QAny(self.num_qubits), side=Side.RIGHT)])

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        **_soqs: SoquetT,
    ) -> dict[str, SoquetT]:
        """Implement bloq decomposition into sub-bloqs."""
        return {"q": bb.allocate(self.num_qubits)}

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Build call graph."""
        return {Allocate(QAny(self.num_qubits)): 1}


@attrs.frozen
class BitstringStatePrep(Bloq):
    """
    Computational basis state preparation from a given bitstring.

    Parameters
    ----------
    bitstring : tuple[int, ...]
        Binary string representing the target computational basis state.
        Bit ``i`` is applied to qubit ``i`` of the output register.

    Registers
    ---------
    q : QAny, RIGHT
        Newly-allocated qubit register initialised to given computational basis state.

    Raises
    ------
    ValueError
        If ``bitstring`` contains values other than 0 or 1.

    """

    bitstring: tuple[int, ...]

    def __attrs_post_init__(self) -> None:
        """Input validator."""
        if not all(i in {0, 1} for i in self.bitstring):
            error_msg = "Invalid bitstring."
            raise ValueError(error_msg)

    @property
    def signature(self) -> Signature:
        """Define input and/or output registers of the bloq."""
        return Signature([Register("q", dtype=QAny(self.num_qubits), side=Side.RIGHT)])

    @property
    def num_qubits(self) -> int:
        """Calculate number of qubits."""
        return len(self.bitstring)

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        **soqs: SoquetT,  # noqa: ARG002
    ) -> dict[str, SoquetT]:
        """Implement bloq decomposition into sub-bloqs."""
        q = bb.allocate(self.num_qubits)
        qs = bb.split(q)

        for i in range(self.num_qubits):
            if self.bitstring[i]:
                qs[i] = bb.add(XGate(), q=qs[i])

        return {"q": bb.join(qs)}

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Build call graph."""
        bloq_counts = {Allocate(QAny(self.num_qubits)): 1}

        if num_x := sum(self.bitstring):
            bloq_counts[XGate()] = num_x

        return bloq_counts

    def my_static_costs(self, cost_key: CostKey) -> int:
        """Return qubit counts (not including any rotation-synthesis ancillas)."""
        if isinstance(cost_key, QubitCount):
            return self.num_qubits
        return NotImplemented


@attrs.frozen
class PrepareFromStatePrep(PrepareOracle):
    r"""
    ``PrepareOracle`` interface wrapper for ``StatePreparationViaRotations`` bloq.

    Implements the action

    .. math::

        \mathrm{PREP}|0\rangle^{\otimes n} = \sum_{j=0}^{2^n - 1} c_j |j\rangle,

    where :math:`n` is ``num_select_qubits`` and :math:`c_j` are the state
    coefficients of ``stateprep``. The phase gradient register is returned unchanged.

    Parameters
    ----------
    stateprep : StatePreparationViaRotations
        State preparation bloq. Its target register must have ``num_select_qubits``
        qubits and its phase gradient register must have ``phase_bitsize`` qubits.
    phase_bitsize : int
        Number of qubits used for phase gradient state.
    num_select_qubits : int
        Number of qubits, :math:`n`, on which to prepare the PREP state.

    Registers
    ---------
    selection : QAny
        Qubits used to represent the selection index for the LCU.
    phase_gradient : QAny
        Qubits used for phase gradient states for rotation synthesis.

    """

    stateprep: StatePreparationViaRotations
    phase_bitsize: int
    num_select_qubits: int

    @property
    def selection_registers(self) -> tuple[Register, ...]:
        """Get selection (index) register."""
        return (Register("selection", QAny(self.num_select_qubits)),)

    @property
    def junk_registers(self) -> tuple[Register, ...]:
        """Get the junk (phase gradient) register."""
        return (Register("phase_gradient", QAny(self.phase_bitsize)),)

    # We must implement the required abstract method to expose the concrete circuit
    def build_prepare_circuit(self) -> Bloq:
        """Get the state preparation bloq. Required by the PrepareOracle interface."""
        return self.stateprep

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        **soqs: SoquetT,
    ) -> dict[str, SoquetT]:
        """Implement decomposition into sub-bloqs using stateprep bloq decomposition."""
        selection, phase_gradient = bb.add(
            self.stateprep,
            target_state=soqs["selection"],
            phase_gradient=soqs["phase_gradient"],
        )
        return {"selection": selection, "phase_gradient": phase_gradient}

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Build call graph for PrepareFromStatePrep."""
        return {self.stateprep: 1}
