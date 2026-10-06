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
estimation `error`, never both; the other half is resolved on construction, as for the
simulation methods in `quiche.simulation`. All errors are energies, in the units of the
Hamiltonian's coefficients (Hartree for HamLib or FCIDUMP inputs).

The modelled protocol runs QPE `repetitions` times and keeps the minimum energy
reading. `error` is the algorithm's own energy resolution, `total_error` adds the
simulation's energy error to it, and `success_probability` lower-bounds the probability
that the minimum reading lies within `total_error` of the ground-state energy. The
bits are `precision_bits` (setting `error`), plus `ceil(log2(1 / overlap))` bits that
keep the minimum over `~1/overlap` repetitions reliable, plus `extra_ancillas` bits that
set the per-run tail probability. See `quiche.budget.estimation` for the bounds and
their references.

The data register width is `hamiltonian.n_qubits`, so the Hamiltonian is assumed to
have no unused trailing qubits: the state preparation must act on exactly that many
qubits. Use `PauliSum.compact()` to drop unused qubits first.

The `to_qualtran`/`to_quest`/`to_cudaq` methods build the QPE part only, expecting an
externally-prepared state on the data register; compose it with a state preparation
(e.g. `quiche.state_prep.HartreeFock`) using each backend's own tools.
"""

from collections.abc import Callable
from math import inf, nan
from typing import Annotated, Literal

from pydantic import Field, PositiveFloat, PositiveInt
from pydantic.dataclasses import dataclass
from qualtran import Bloq

from quiche._validation import require_exactly_one
from quiche.budget.estimation import (
    MIN_EXTRA_ANCILLAS,
    get_default_repetitions,
    get_estimation_error,
    get_overlap_bits,
    get_precision_bits,
    get_success_probability,
    get_tail_probability,
)
from quiche.cudaq import CudaqKernel
from quiche.dispatch.spec import Spec
from quiche.quest import QuestRoutine
from quiche.simulation import SimulationMethod

# Squared overlap |<psi|E0>|^2 of the prepared state with the ground state.
type Overlap = Annotated[float, Field(gt=0, le=1)]
type ExtraAncillas = Annotated[int, Field(ge=MIN_EXTRA_ANCILLAS)]


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
        """Get the bound on the energy error: estimation plus simulation error."""
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

    def to_cudaq(
        self,
        state_prep: CudaqKernel | None = None,
        *,
        kernel: Literal["all", "circuit", "post"] = "all",
    ) -> CudaqKernel | Callable:
        """
        Build the CUDA-Q kernel implementing the QPE algorithm.

        `kernel` selects what is built:

        - `"all"` (default): the whole algorithm in one kernel, including measuring and
          decoding the energy in-kernel (described below). This is the form for
          hardware-style execution and for algorithms that need mid-circuit feedback,
          such as `Iterative`. Simulators run it once per shot via `cudaq.run`.
        - `"circuit"`: `Textbook` only. The QPE circuit followed by a measurement of the
          QPE ancillas, with no measurement-dependent logic or return value:
          `() -> None` given `state_prep`, otherwise `(data: cudaq.qview) -> None`.
          `cudaq.sample` simulates it once for any number of shots, and it suits
          compilers that reject measurement feedback, such as CUDA-Q Logical.
        - `"post"`: `Textbook` only. A host-side function (not a kernel) turning the
          `cudaq.sample` result of the `"circuit"` kernel into energies, keeping the
          minimum over each block of `repetitions` shots:

          ```python
          circuit = qpe.to_cudaq(hf.to_cudaq(), kernel="circuit")
          post = qpe.to_cudaq(kernel="post")
          energies = post(cudaq.sample(circuit, shots_count=shots * qpe.repetitions))
          ```

        The rest of this docstring describes `"all"`.

        Given a `state_prep` kernel `(qubits: cudaq.qview) -> None`, the returned kernel
        has signature `() -> float` and runs the whole modelled protocol in-kernel: for
        each of the `repetitions`, it allocates a fresh data register, prepares it, runs
        one QPE, and keeps the minimum energy, which it returns. This is what
        `success_probability` describes. `Naive` has no minimum to take, so raises.

        Without `state_prep`, the returned kernel runs a single QPE and has signature
        `(data: cudaq.qview) -> float`: it operates
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

        return cudaq.estimation(self, state_prep, kernel)


class _Bounded(_PhaseEstimation):
    """Error and success-probability bookkeeping for the bit-by-bit algorithms."""

    overlap: float
    extra_ancillas: int
    repetitions: int

    @property
    def num_bits(self) -> int:
        """Get the number of phase bits measured per run."""
        raise NotImplementedError

    def _resolve(self, bits_field: str) -> None:
        """Resolve the bits or error from the other, then the repetitions."""
        require_exactly_one(
            **{bits_field: getattr(self, bits_field), "error": self.error}
        )
        protection = get_overlap_bits(self.overlap) + self.extra_ancillas
        energy_scale = self.simulation.energy_scale
        if getattr(self, bits_field) is None:
            precision_bits = get_precision_bits(energy_scale, self.error)
            object.__setattr__(self, bits_field, precision_bits + protection)
        if self.precision_bits < 1:
            msg = (
                f"Need at least {protection + 1} bits for overlap {self.overlap} and "
                f"{self.extra_ancillas} extra ancillas, got {self.num_bits}."
            )
            raise ValueError(msg)
        error = get_estimation_error(energy_scale, self.precision_bits)
        object.__setattr__(self, "error", error)
        if self.repetitions is None:
            repetitions = get_default_repetitions(self.overlap, self.tail_probability)
            object.__setattr__(self, "repetitions", repetitions)

    @property
    def precision_bits(self) -> int:
        """Get the number of bits setting the energy resolution `error`."""
        return self.num_bits - get_overlap_bits(self.overlap) - self.extra_ancillas

    @property
    def tail_probability(self) -> float:
        """Get the probability that one run reads outside the precision window."""
        return get_tail_probability(
            get_overlap_bits(self.overlap) + self.extra_ancillas
        )

    @property
    def applications(self) -> int:
        """Get the number of controlled applications of the simulation per run."""
        return 2**self.num_bits - 1

    @property
    def success_probability(self) -> float:
        """Get a lower bound on the probability that the minimum is within the error."""
        return get_success_probability(
            overlap=self.overlap,
            tail_probability=self.tail_probability,
            repetitions=self.repetitions,
            applications=self.applications,
            channel_error=self.simulation.channel_error,
        )


@dataclass(frozen=True)
class Textbook(_Bounded):
    """
    Textbook QPE: multi-ancilla, using C-U^(2^k) gates and the inverse QFT.

    Give exactly one of `num_ancillas` or the energy `error`. `overlap` is the squared
    overlap of the prepared state with the ground state. `repetitions` defaults to the
    fewest runs missing the ground state with probability at most 5%.
    """

    simulation: SimulationMethod
    overlap: Overlap
    num_ancillas: PositiveInt | None = None
    error: PositiveFloat | None = None
    extra_ancillas: ExtraAncillas = 4
    repetitions: PositiveInt | None = None

    def __post_init__(self) -> None:
        """Resolve the ancillas or estimation error from the other."""
        self._resolve("num_ancillas")

    @property
    def num_bits(self) -> int:
        """Get the number of phase bits measured per run."""
        return self.num_ancillas

    @property
    def num_qpe_ancillas(self) -> int:
        """Get the number of QPE ancillas."""
        return self.num_ancillas


@dataclass(frozen=True)
class _SingleAncilla(_Bounded):
    """Single-ancilla QPE over `num_rounds` rounds of C-U^(2^k)."""

    simulation: SimulationMethod
    overlap: Overlap
    num_rounds: PositiveInt | None = None
    error: PositiveFloat | None = None
    extra_ancillas: ExtraAncillas = 4
    repetitions: PositiveInt | None = None

    def __post_init__(self) -> None:
        """Resolve the rounds or estimation error from the other."""
        self._resolve("num_rounds")

    @property
    def num_bits(self) -> int:
        """Get the number of phase bits measured per run."""
        return self.num_rounds


@dataclass(frozen=True)
class Kitaev(_SingleAncilla):
    """
    Kitaev QPE: single-ancilla, using C-U^(2^k) gates.

    Give exactly one of `num_rounds` or the energy `error`. The error and success
    bookkeeping is that of `Iterative`; the QuEST backend idealises each round with
    exact expectation values, so its readout is deterministic.
    """


@dataclass(frozen=True)
class Iterative(_SingleAncilla):
    """
    Iterative QPE: single-ancilla, using C-U^(2^k) gates and Rz for feedback.

    Give exactly one of `num_rounds` or the energy `error`. Its readout has the same
    distribution as `Textbook` with as many ancillas as rounds (semiclassical QFT).
    """


@dataclass(frozen=True)
class Naive(_PhaseEstimation):
    """
    Naive QPE: single-ancilla Hadamard test of a single C-U.

    One shot carries a single bit, so there is no a-priori bound on the estimation
    error: `error` and `total_error` are infinite, and `success_probability` is
    undefined (`nan`).
    """

    simulation: SimulationMethod

    @property
    def error(self) -> float:
        """Get the estimation error, which is unbounded for a Hadamard test."""
        return inf

    @property
    def success_probability(self) -> float:
        """Get the success probability, undefined for a single Hadamard test."""
        return nan


type EstimationMethod = Textbook | Kitaev | Iterative | Naive
