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

from typing import Self

import attrs
import numpy as np
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
    """Routine for trivial state preparation."""

    n_qubits: int

    def my_static_costs(self, cost_key: "CostKey") -> int:
        """Return hard-coded qubit counts."""
        if isinstance(cost_key, QubitCount):
            # Only data qubits are needed for this state preparation.
            return self.n_qubits
        return NotImplemented

    @property
    def signature(self) -> Signature:
        """Define input and/or output registers of the bloq."""
        return Signature([Register("q", dtype=QAny(self.n_qubits), side=Side.RIGHT)])

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        **_soqs: SoquetT,
    ) -> dict[str, SoquetT]:
        """Implement bloq decomposition into sub-bloqs."""
        return {"q": bb.allocate(self.n_qubits)}

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Build call graph."""
        return {Allocate(QAny(self.n_qubits)): 1}


@attrs.frozen
class BitstringStatePrep(Bloq):
    """Routine to prepare an arbitrary computational basis state."""

    bitstring: tuple[int, ...]

    def __attrs_post_init__(self) -> Self:
        """Input validator."""
        if not all(i in {0, 1} for i in self.bitstring):
            err_msg = "Invalid bitstring."
            raise ValueError(err_msg)

        return self

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
        return {
            Allocate(QAny(self.num_qubits)): 1,
            XGate(): sum(self.bitstring),
        }

    def my_static_costs(self, cost_key: "CostKey") -> int:
        """Return hard-coded qubit counts."""
        if isinstance(cost_key, QubitCount):
            # Only data qubits are needed for this state preparation.
            return self.num_qubits
        return NotImplemented


@attrs.frozen
class PrepareFromStatePrep(PrepareOracle):
    r"""
    PREP routine for a given phase-gradient state preparation bloq.

    Implements the action PREP |0> = \sum_{j=1}^L c_j |j>

    Properties
    stateprep: State preparation bloq
    phase_bitsize: Number of qubits used for phase gradient
    select_nqubits: Number of qubits on which to prepare the PREP state.
                    Note that L must be equal to 2**select_nqubits.
    """

    stateprep: StatePreparationViaRotations
    phase_bitsize: int
    select_nqubits: int

    @property
    def selection_registers(self) -> tuple[Register, ...]:
        """Get selection (index) register."""
        return (Register("selection", QAny(self.phase_bitsize + self.select_nqubits)),)

    # We must implement the required abstract method to expose the concrete circuit
    def build_prepare_circuit(self) -> Bloq:
        """Get the state preparation bloq. Required by the PrepareOracle interface."""
        return self.stateprep

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        **soqs: SoquetT,
    ) -> dict[str, SoquetT]:
        """
        Implement decomposition into sub-bloqs using state prep bloq decompositon.

        Because the state prep unitary uses registers 'target_state' and
        'phase_gradient', the register 'selection' needs to be split up in the process.
        """
        # split the selection register into the state prep register
        xs = bb.split(soqs["selection"])
        target_state = bb.join(xs[: self.select_nqubits])
        phase_gradient = bb.join(xs[self.select_nqubits :])

        target_state, phase_gradient = bb.add(
            self.stateprep, target_state=target_state, phase_gradient=phase_gradient
        )

        xs = np.concatenate([bb.split(target_state), bb.split(phase_gradient)])
        result = bb.join(xs)

        return {"selection": result}

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Build call graph for PrepareFromStatePrep."""
        return {
            self.stateprep: 1,
        }
