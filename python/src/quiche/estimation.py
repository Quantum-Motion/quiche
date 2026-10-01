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

"""
Phase estimation algorithms, each wrapping a Hamiltonian simulation method.

Each algorithm is specified either by its size (ancillas or rounds) or by a target
estimation error, never both; the other half is resolved on construction, as for the
simulation methods in `quiche.simulation`. `error` is then the algorithm's own
estimation error, and `total_error` adds the simulation error to it as an upper bound.

The data register width is `hamiltonian.n_qubits`, so the Hamiltonian is assumed to
have no unused trailing qubits: the state preparation must act on exactly that many
qubits. Use `PauliSum.compact()` to drop unused qubits first.

The `to_qualtran`/`to_quest`/`to_cudaq` methods build the QPE part only, expecting an
externally-prepared state on the data register; compose it with a state preparation
(e.g. `quiche.state_prep.HartreeFock`) using each backend's own tools.
"""

from math import inf
from typing import Annotated

from pydantic import Field, PositiveFloat, PositiveInt
from pydantic.dataclasses import dataclass
from qualtran import Bloq

from quiche._validation import require_exactly_one
from quiche.budget.estimation import (
    get_kitaev_qpe_error,
    get_kitaev_qpe_rounds,
    get_textbook_qpe_ancillas,
    get_textbook_qpe_error,
)
from quiche.cudaq import CudaqKernel
from quiche.dispatch.spec import Spec
from quiche.quest import QuestRoutine
from quiche.simulation import SimulationMethod

# Magnitude of the overlap of the prepared initial state with the target eigenstate.
type Overlap = Annotated[float, Field(gt=0, le=1)]


class _PhaseEstimation(Spec):
    """Qubit counts, total error and backend lowering shared by all algorithms."""

    simulation: SimulationMethod
    error: float

    @property
    def num_data(self) -> int:
        """Get the number of qubits in the data register."""
        return self.simulation.hamiltonian.n_qubits

    @property
    def num_qpe_ancillas(self) -> int:
        """Get the number of QPE ancillas."""
        return 1

    @property
    def num_simulation_ancillas(self) -> int:
        """Get the number of ancillas used by the simulation."""
        return self.simulation.num_ancillas

    @property
    def num_qubits(self) -> int:
        """Get the total number of qubits."""
        return self.num_data + self.num_qpe_ancillas + self.num_simulation_ancillas

    @property
    def total_error(self) -> float:
        """Get the union-bound total of the estimation and simulation errors."""
        return self.error + self.simulation.error

    # The lowering modules match on the classes defined here, so they are imported
    # lazily to avoid a circular import.
    def to_qualtran(self) -> Bloq:
        """
        Build the Qualtran Bloq implementing the QPE algorithm.

        The returned Bloq expects an externally-prepared state on its `data` register;
        compose it with a state-preparation Bloq to get a runnable circuit.
        """
        from quiche.dispatch.lowering import qualtran  # noqa: PLC0415

        return qualtran.estimation(self)

    def to_quest(self) -> QuestRoutine:
        """
        Generate the QuEST routine implementing the QPE algorithm.

        The returned routine expects the Qureg to already be in an externally-prepared
        initial state; compose it (e.g. via `QuestRoutine.extend`) with a state
        preparation routine to get a runnable simulation.
        """
        from quiche.dispatch.lowering import quest  # noqa: PLC0415

        return quest.estimation(self)

    def to_cudaq(self) -> CudaqKernel:
        """
        Build the CUDA-Q kernel implementing the QPE algorithm.

        The returned kernel has signature `(data: cudaq.qview) -> float`: it operates
        in place on an already-allocated data register (the same contract as
        `HartreeFock.to_cudaq()`'s kernel). Compose it with a state-preparation kernel
        by allocating the register once and calling both kernels on it inside one
        top-level kernel, e.g.:

        ```python
        prep = hf.to_cudaq()
        qpe_kernel = qpe.to_cudaq()
        num_data = qpe.num_data

        @cudaq.kernel
        def run() -> float:
            data = cudaq.qvector(num_data)
            prep(data)
            return qpe_kernel(data)

        energy = run()                               # a single shot
        energies = cudaq.run(run, shots_count=100)    # or many, as a list[float]
        ```

        What the returned `float` *means* depends on the algorithm: `Textbook`/
        `Iterative` decode a genuine one-shot energy estimate in-kernel. `Naive`
        instead returns the raw single-shot ancilla outcome (`+1.0`/`-1.0`) - a
        Hadamard test's one bit carries no phase information on its own; average many
        shots (`statistics.mean(cudaq.run(run, shots_count=n))`) to estimate
        `Re`/`Im(<psi|U|psi>)`. See `quiche.cudaq.estimation.textbook_qpe_kernel`/
        `qubitised_qpe_kernel`/`naive_qpe_kernel`/`iterative_qpe_kernel` for the exact
        contract of each.

        `Kitaev` raises `NotImplementedError`: in this repo's sense (matching QuEST's/
        Qualtran's naming) it's per-round expectation-value estimation via repeated
        measurement with no feedback rotation, combined classically across rounds - a
        materially different, bigger piece of work than a single in-kernel decode, not
        yet built. `Qubitised` simulation also still raises for `Iterative`
        specifically (deferred separately).
        """
        from quiche.dispatch.lowering import cudaq  # noqa: PLC0415

        return cudaq.estimation(self)


@dataclass(frozen=True)
class Textbook(_PhaseEstimation):
    """
    Textbook QPE: multi-ancilla, using C-U^(2^k) gates and the inverse QFT.

    Give exactly one of `num_ancillas` or the estimation `error`.
    """

    simulation: SimulationMethod
    overlap: Overlap
    num_ancillas: PositiveInt | None = None
    error: PositiveFloat | None = None

    def __post_init__(self) -> None:
        """Resolve the ancillas or estimation error from the other."""
        require_exactly_one(num_ancillas=self.num_ancillas, error=self.error)
        if self.num_ancillas is None:
            num_ancillas = get_textbook_qpe_ancillas(self.error, self.overlap)
            object.__setattr__(self, "num_ancillas", num_ancillas)
        error = get_textbook_qpe_error(self.num_ancillas, self.overlap)
        object.__setattr__(self, "error", error)

    @property
    def num_qpe_ancillas(self) -> int:
        """Get the number of QPE ancillas."""
        return self.num_ancillas


@dataclass(frozen=True)
class _SingleAncilla(_PhaseEstimation):
    """Single-ancilla QPE over `num_rounds` rounds of C-U^(2^k)."""

    simulation: SimulationMethod
    overlap: Overlap
    num_rounds: PositiveInt | None = None
    error: PositiveFloat | None = None

    def __post_init__(self) -> None:
        """Resolve the rounds or estimation error from the other."""
        require_exactly_one(num_rounds=self.num_rounds, error=self.error)
        if self.num_rounds is None:
            num_rounds = get_kitaev_qpe_rounds(self.error, self.overlap)
            object.__setattr__(self, "num_rounds", num_rounds)
        error = get_kitaev_qpe_error(self.num_rounds, self.overlap)
        object.__setattr__(self, "error", error)


@dataclass(frozen=True)
class Kitaev(_SingleAncilla):
    """
    Kitaev QPE: single-ancilla, using C-U^(2^k) gates.

    Give exactly one of `num_rounds` or the estimation `error`.
    """


@dataclass(frozen=True)
class Iterative(_SingleAncilla):
    """
    Iterative QPE: single-ancilla, using C-U^(2^k) gates and Rz for feedback.

    Give exactly one of `num_rounds` or the estimation `error`.
    """


@dataclass(frozen=True)
class Naive(_PhaseEstimation):
    """
    Naive QPE: single-ancilla Hadamard test of a single C-U.

    One shot carries a single bit, so there is no a-priori bound on the estimation
    error: `error` is infinite, and so is `total_error`.
    """

    simulation: SimulationMethod

    @property
    def error(self) -> float:
        """Get the estimation error, which is unbounded for a Hadamard test."""
        return inf


type EstimationMethod = Textbook | Kitaev | Iterative | Naive
