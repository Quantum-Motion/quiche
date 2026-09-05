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

"""Quantum phase estimation routines."""

import abc
import numbers
from collections.abc import Callable
from typing import Self

import attrs
import sympy
from qualtran import (
    Bloq,
    BloqBuilder,
    CtrlSpec,
    QAny,
    QBit,
    QUInt,
    Register,
    Signature,
    SoquetT,
)
from qualtran.bloqs.basic_gates import Hadamard, Power, Rz, SGate
from qualtran.bloqs.bookkeeping import Allocate, Free
from qualtran.bloqs.phase_estimation import RectangularWindowState
from qualtran.bloqs.qft import QFTTextBook
from qualtran.bloqs.qubitization.qubitization_walk_operator import (
    QubitizationWalkOperator,
)
from qualtran.resource_counting import (
    BloqCountDictT,
    CostKey,
    QubitCount,
    SympySymbolAllocator,
    get_cost_value,
)

from .simulation import QDRIFT, Trotterisation


@attrs.frozen
class _SingleAncillaQPE(Bloq):
    """
    Base class for single-ancilla QPE algorithms.

    Provides the common Hadamard-test structure used for Naive QPE, Kitaev QPE and
    Iterative QPE.

    |0>   ---H---[S†]------•------[Rz]---H---- meas
                           |
    |psi> -------------U^exponent-------------

    `S†` applied if measuring imaginary component.
    `Rz` used for feedback rotation in Iterative QPE (enabled according to
        the `apply_feedback` property).

    The exponent of the propagator is also determined by the `exponent` property.

    Properties
    ----------
    simulation : Bloq
        Bloq implementing the Hamiltonian simulation.
    mode : {'re', 'im'}
        Specifies whether to measure the real or imaginary part of the expectation
        value.
    """

    simulation: Bloq
    mode: str

    def __attrs_post_init__(self) -> Self:
        """Input validator."""
        if not isinstance(self.exponent, numbers.Integral) or self.exponent < 1:
            err_msg = "Exponent must be positive integer."
            raise ValueError(err_msg)
        if self.mode not in ("re", "im"):
            err_msg = "Measurement mode must be either 're' or 'im'."
            raise ValueError(err_msg)

        return self

    @property
    @abc.abstractmethod
    def exponent(self) -> int:
        """Return the power to which the propagator is raised."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def apply_feedback(self) -> bool:
        """Return whether a feedback rotation precedes the final Hadamard or not."""
        raise NotImplementedError

    @property
    def controlled_propagator(self) -> Bloq:
        """Return the controlled propagator `C[U^exponent]`."""
        return (
            self.simulation.controlled()
            if self.exponent == 1
            else Power(self.simulation.controlled(), self.exponent)
        )

    @property
    def n_simulation_qubits(self) -> int:
        """Return number of qubits used for Hamiltonian simulation."""
        return self.simulation.signature[0].total_bits()

    @property
    def n_estimation_bits(self) -> int:
        """Return number of estimation qubits."""
        return 1

    @property
    def signature(self) -> Signature:
        """Define input and/or output registers of the bloq."""
        return Signature([Register("simulation", dtype=QAny(self.n_simulation_qubits))])

    def my_static_costs(self, cost_key: "CostKey") -> int:
        """Return hard-coded qubit counts."""
        if isinstance(cost_key, QubitCount) and (
            isinstance(self.simulation, (QDRIFT, Trotterisation))
        ):
            # This bloq only needs the data qubits and one ancilla. So far assumes that
            # will not be initialised with a Qubitisation simulation bloq.
            return self.n_simulation_qubits + 1
        return NotImplemented

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        **soqs: SoquetT,
    ) -> dict[str, SoquetT]:
        """Implement bloq decomposition into sub-bloqs."""
        simulation = soqs["simulation"]

        estimation = bb.add(RectangularWindowState(self.n_estimation_bits))

        if self.mode == "im":
            estimation = bb.add(SGate(is_adjoint=True), q=estimation)

        estimation, simulation = bb.add(
            self.controlled_propagator,
            ctrl=estimation,
            simulation=simulation,
        )

        if self.apply_feedback:
            theta = sympy.Symbol("a")
            estimation = bb.add(Rz(theta), q=estimation)

        estimation = bb.add(Hadamard(), q=estimation)

        bb.free(estimation)
        return {"simulation": simulation}

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Build call graph for single-ancilla QPE."""
        bloq_counts = {
            RectangularWindowState(self.n_estimation_bits): 1,
            self.controlled_propagator: 1,
            Hadamard(): 1,
            Free(QBit()): 1,
        }

        if self.mode == "im":
            bloq_counts[SGate(is_adjoint=True)] = 1

        if self.apply_feedback:
            theta = sympy.Symbol("a")
            bloq_counts[Rz(theta)] = 1

        return bloq_counts


@attrs.frozen
class NaiveQPE(_SingleAncillaQPE):
    """
    Naive, single-ancilla phase estimation (Hadamard test).

    Uses a controlled application of the propagator U to estimate its expectation value
    in a provided initial state. Only one ancilla is used during the Hadamard test.

    If the measurement mode is "re", the bloq represents the following circuit:

    |0>   ------H------•--------H----- meas
                       |
    |psi> -------------U--------------

    and if it is "im", the bloq represents the following circuit:

    |0>   ---H---S†----•------- H ---- meas
                       |
    |psi> -------------U--------------

    Properties
    ----------
    simulation : Bloq
        Bloq implementing the Hamiltonian simulation.
    mode : {'re', 'im'}
        Specifies whether to measure the real or imaginary part of the expectation
        value.

    Resources
    ----------
    The bloq uses simulation qubits and a single ancilla qubit.

    """

    simulation: Bloq
    mode: str

    @property
    def exponent(self) -> int:
        """Power to which the propagator is raised (always 1 for Naive QPE)."""
        return 1

    @property
    def apply_feedback(self) -> bool:
        """Whether a feedback rotation is applied (`False` for Naive QPE)."""
        return False


@attrs.frozen
class KitaevQPE(_SingleAncillaQPE):
    """
    Kitaev single-ancilla phase estimation.

    For a unitary operator U and an initial state that approximates some eigenstate
    |psi>, estimate the phase theta in U|psi> = e^{2 pi i theta} |psi>. The Kitaev QPE
    estimates the k-th digit of the phase by applying the propagator U^(2^k) for
    increasing k values.
    This bloq implements the QPE for a single selection of k. It needs to be called
    multiple times with different k values to reflect a full phase estimation.

    If the measurement mode is "re", the bloq represents the following circuit:

    |0>   ------H------•--------H----- meas
                       |
    |psi> -------------U^(2^k)--------

    and if it is "im", the bloq represents the following circuit:

    |0>   ---H---S†----•------- H ---- meas
                       |
    |psi> -------------U^(2^k)--------

    Properties
    ----------
    simulation : Bloq
        Bloq implementing the Hamiltonian simulation.
    k : int
        Controls the exponent of the propagator U as described above.
    mode : {'re', 'im'}
        Specifies whether to measure the real or imaginary part of the expectation
        value.
    Resources
    ----------
    The bloq uses simulation qubits and a single ancilla qubit.

    """

    simulation: Bloq
    k: int
    mode: str

    @property
    def exponent(self) -> int:
        """Power to which the propagator is raised (2^k for Kitaev QPE)."""
        return 2**self.k

    @property
    def apply_feedback(self) -> bool:
        """Whether a feedback rotation is applied (`False` for Kitaev QPE)."""
        return False


@attrs.frozen
class IterativeQPE(_SingleAncillaQPE):
    """
    Iterative phase estimation.

    For a unitary operator U and an initial state that approximates some eigenstate
    |psi>, estimate the phase theta in U|psi> = e^{2 pi i theta} |psi>. In iterative
    QPE, a rotation around z is inserted between the controlled propagator and the
    second Hadamard gate. The rotation angle depends on previous measurements, which we
    cannot evaluate in Qualtran. Instead we use symbolic angles.
    This bloq implements the QPE for a single selection of k. It needs to be called
    multiple times with different k values to reflect a full phase estimation.

    If the measurement mode is "re", the bloq represents the following circuit:

    |0>   ------H------•----Rz----H----- meas
                       |
    |psi> -------------U^(2^k)----------

    and if it is "im", the bloq represents the following circuit:

    |0>   ---H---S†----•----Rz---H---- meas
                       |
    |psi> -------------U^(2^k)----------

    Properties
    ----------
    simulation : Bloq
        Bloq implementing the Hamiltonian simulation.
    k : int
        Controls the exponent of the propagator U as described above.
    mode : {'re', 'im'}
        Specifies whether to measure the real or imaginary part of the expectation
        value.
    Resources
    ----------
    The bloq uses simulation qubits and a single ancilla qubit.

    """

    simulation: Bloq
    k: int
    mode: str

    @property
    def exponent(self) -> int:
        """Power to which the propagator is raised (2^k for Iterative QPE)."""
        return 2**self.k

    @property
    def apply_feedback(self) -> bool:
        """Whether a feedback rotation is applied (`True` for Iterative QPE)."""
        return True


@attrs.frozen
class TextbookQPE(Bloq):
    """End-to-end QPE routine including state preparation, simulation and QFT."""

    simulation_factory: Callable[[int], Bloq]

    num_data: int  # system qubits
    num_qpe_ancillas: int  # ancilla used for phase estimation
    num_other_ancillas: int  # other ancilla used e.g. for block encoding

    def my_static_costs(self, cost_key: "CostKey") -> int:
        """Return hard-coded qubit counts."""
        # There are three stages to the QPE:
        # 1. State preparation on simulation and estimation registers
        #    The static cost currently assumes that there are no ancilla
        #    used for this part.
        # 2. Controlled time propagation
        #    If the propagation is done with Trotter methods, then there are no ancilla
        #    due to the time evolution, and the width of this part is equal to the
        #    number of data qubits plus number of estimation qubits.
        # 3. Inverse QFT on estimation register
        #    The QFT needs one ancilla qubit for the PhaseGradientUnitary. Note that
        #    this is independent of the number of estimation qubits: There is a number
        #    of ancilla allocated, but they are all allocated sequentially and thus only
        #    contribute one ancilla.
        # FullQPE is only really used with Trotter methods, so there is no accounting
        # for qubitisation in here.
        #
        # First, generate a factory instance in order to determine which type of QPE
        # we are doing.
        factory_instance = self.simulation_factory(0)
        if isinstance(cost_key, QubitCount):
            if isinstance(factory_instance, TrotterLadder):
                # As described above, the number of qubits is equal to the number of
                # simulation plus estimation qubits, plus the ancilla needed for the
                # inverse QFT.
                return (
                    self.num_data + self.num_other_ancillas + self.num_qpe_ancillas + 1
                )
            if isinstance(factory_instance, QubitisationLadder):
                # The Trotter and qubitisation approaches only differ in step 2.
                # In qubitisation:
                # First, a walk operator is applied on the data + ancilla qubits, being
                # controlled from one of the estimation ancilla. Subsequently, each
                # time propagation involves a reflection operator controlled by one of
                # the estimation ancilla, followed by an (uncontrolled) walk operator,
                # followed by an adjoint reflection controlled by an estimation ancilla.
                # The qubit count only cares about the widest part of the algorithm.
                # This can be found by finding the width of the controlled reflection
                # and the width of the controlled walk.
                n_reflect = get_cost_value(
                    self.simulation_factory(0).walk.reflect.controlled(
                        ctrl_spec=CtrlSpec(cvs=(0,))
                    ),
                    QubitCount(),
                )
                n_walk = get_cost_value(
                    self.simulation_factory(0).walk.controlled(), QubitCount()
                )
                # Calculate the ancillas allocated and deallocated on the fly by
                # subtracting the qubits that are kept throughout the algorithm.
                tmp_ancillas_reflect = n_reflect - self.num_other_ancillas - 1
                tmp_ancillas_walk = n_walk - self.num_other_ancillas - self.num_data - 1
                tmp_ancillas_qft = 1
                # get the temporary ancillas for the QubitisationLadder
                tmp_ancillas_prop = max(tmp_ancillas_reflect, tmp_ancillas_walk)
                # get overall temporary ancillas
                tmp_ancillas = max(tmp_ancillas_prop, tmp_ancillas_qft)
                # The width of the QPE up to the QFT is then the number of temporary
                # ancillas plus the permanent ancillas plus the data qubits.
                return (
                    self.num_qpe_ancillas
                    + self.num_other_ancillas
                    + self.num_data
                    + tmp_ancillas
                )

        return NotImplemented

    @property
    def signature(self) -> Signature:
        """Define input and/or output registers of the bloq."""
        return Signature.build(data=self.num_data)

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        **soqs: SoquetT,
    ) -> dict[str, SoquetT]:
        """Implement bloq decomposition into sub-bloqs."""
        data = soqs["data"]

        # Hadamard on ancillas
        qpe_ancillas = bb.add(RectangularWindowState(self.num_qpe_ancillas))

        if self.num_other_ancillas > 0:
            other_ancillas = bb.allocate(self.num_other_ancillas)

            # Generic unitary ladder constructor
            for i in range(self.num_qpe_ancillas):
                data, qpe_ancillas, other_ancillas = bb.add(
                    self.simulation_factory(i),
                    data=data,
                    qpe_ancillas=qpe_ancillas,
                    other_ancillas=other_ancillas,
                )

            bb.free(other_ancillas)

        else:
            for i in range(self.num_qpe_ancillas):
                data, qpe_ancillas = bb.add(
                    self.simulation_factory(i),
                    data=data,
                    qpe_ancillas=qpe_ancillas,
                )

        # Inverse QFT
        iqft = QFTTextBook(self.num_qpe_ancillas).adjoint()
        qpe_ancillas = bb.add(iqft, q=qpe_ancillas)

        bb.free(qpe_ancillas)

        return {"data": data}

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Build call graph for TextbookQPE."""
        bloq_counts = {RectangularWindowState(self.num_qpe_ancillas): 1}
        for ii in range(self.num_qpe_ancillas):
            bloq_counts[self.simulation_factory(index=ii)] = 1
        bloq_counts[QFTTextBook(self.num_qpe_ancillas).adjoint()] = 1
        # Bookkeeping terms.
        # TODO(Annina): Need to find out why QPE ancilla are QUInt type and other
        # ancilla are QAny type (otherwise call graphs differ from call graphs obtained
        # from decomposition). Probably something to do with the allocation.
        if self.num_other_ancillas > 0:
            bloq_counts[Allocate(QAny(self.num_other_ancillas))] = 1
            bloq_counts[Free(QAny(self.num_other_ancillas))] = 1
        bloq_counts[Free(QUInt(self.num_qpe_ancillas))] = 1
        return bloq_counts


@attrs.frozen
class TrotterLadder(Bloq):
    """Routine to construct controlled Trotter evolution operator as needed for QPE."""

    # Index must be first parameter
    index: int
    simulation: Trotterisation | QDRIFT

    num_data: int
    num_qpe_ancillas: int

    def my_static_costs(self, cost_key: "CostKey") -> int:
        """Return hard-coded qubit counts."""
        if isinstance(cost_key, QubitCount) and (
            isinstance(self.simulation, (Trotterisation, QDRIFT))
        ):
            # The Trotter bloq needs a number of rotation gates. Although it may require
            # ancilla qubits, the required number is calculated in post-processing.
            # So the total number of qubits is simply the number of data qubits plus the
            # number of ancillas.
            return self.num_data + self.num_qpe_ancillas
        return NotImplemented

    @property
    def signature(self) -> Signature:
        """Define input and/or output registers of the bloq."""
        return Signature.build(data=self.num_data, qpe_ancillas=self.num_qpe_ancillas)

    def build_composite_bloq(
        self, bb: BloqBuilder, **soqs: SoquetT
    ) -> dict[str, SoquetT]:
        """Implement bloq decomposition into sub-bloqs."""
        data = soqs["data"]
        qpe_ancillas = soqs["qpe_ancillas"]
        ancilla_qubits = bb.split(qpe_ancillas)

        # Generate the controlled simulation bloq and add it to the BloqBuilder. For
        # Trotter methods, the process is the same for each index.
        c_simulation = self.simulation.controlled()
        ancilla_qubits[self.index], data = bb.add(
            Power(c_simulation, 2**self.index),
            ctrl=ancilla_qubits[self.index],
            simulation=data,
        )

        qpe_ancillas = bb.join(ancilla_qubits)

        return {"data": data, "qpe_ancillas": qpe_ancillas}

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Build call graph for TrotterLadder."""
        return {
            Power(self.simulation.controlled(), 2**self.index): 1,
        }


@attrs.frozen
class QubitisationLadder(Bloq):
    """Routine to construct controlled Qubitisation operator as needed for QPE."""

    # Index must be first parameter
    index: int
    walk: QubitizationWalkOperator

    num_data: int
    num_qpe_ancillas: int
    num_selection_ancillas: int

    @property
    def signature(self) -> Signature:
        """Define input and/or output registers of the bloq."""
        return Signature.build(
            data=self.num_data,
            qpe_ancillas=self.num_qpe_ancillas,
            other_ancillas=self.num_selection_ancillas,
        )

    # Decomposition derived from qualtran/bloqs/phase_estimation/qubitization_qpe.py
    # Copyright 2024 Google LLC.
    # Licensed under the Apache License 2.0.
    # Modified to `Ladder` Bloq structure with index parameter. Further modified to
    # include signal state preparation and correct the exponent of the walk operator.
    def build_composite_bloq(
        self, bb: BloqBuilder, **soqs: SoquetT
    ) -> dict[str, SoquetT]:
        """Implement bloq decomposition into sub-bloqs."""
        target = soqs["data"]
        qpe_ancillas = soqs["qpe_ancillas"]
        selection = soqs["other_ancillas"]
        qpe_ancilla_qubits = bb.split(qpe_ancillas)

        reflect_controlled = self.walk.reflect.controlled(ctrl_spec=CtrlSpec(cvs=0))
        walk_controlled = self.walk.controlled()

        # Apply the suitable bloq depending on the index. If index is 0, then apply the
        # controlled walk operator. Otherwise, apply controlled reflections and a power
        # of the (uncontrolled) walk operator.
        if self.index == 0:
            # Prepare signal state for block encoding ancillas if needed
            selection = bb.add(
                self.walk.block_encoding.signal_state,
                selection=selection,
            )

            qpe_ancilla_qubits[0], selection, target = bb.add(
                walk_controlled,
                ctrl=qpe_ancilla_qubits[0],
                selection=selection,
                target=target,
            )

        else:
            qpe_ancilla_qubits[self.index], selection = bb.add(
                reflect_controlled,
                control=qpe_ancilla_qubits[self.index],
                selection=selection,
            )

            selection, target = bb.add(
                Power(self.walk, 2 ** (self.index - 1)),
                selection=selection,
                target=target,
            )

            qpe_ancilla_qubits[self.index], selection = bb.add(
                reflect_controlled,
                control=qpe_ancilla_qubits[self.index],
                selection=selection,
            )

        return {
            "data": target,
            "qpe_ancillas": bb.join(qpe_ancilla_qubits),
            "other_ancillas": selection,
        }

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Build call graph for QubitisationLadder."""
        reflect_controlled = self.walk.reflect.controlled(ctrl_spec=CtrlSpec(cvs=0))
        walk_controlled = self.walk.controlled()
        signal_state = self.walk.block_encoding.signal_state

        bloq_counts = {}
        if self.index == 0:
            bloq_counts[walk_controlled] = 1
            bloq_counts[signal_state] = 1
        else:
            bloq_counts[Power(self.walk, 2 ** (self.index - 1))] = 1
            bloq_counts[reflect_controlled] = 2
        return bloq_counts
