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
Hamiltonian simulation methods for phase estimation algorithms.

`error` is an energy, in the units of the Hamiltonian's coefficients (Hartree for HamLib
or FCIDUMP inputs): a rigorous bound on how far each eigenvalue of the Hamiltonian that
is actually simulated lies from one of the true Hamiltonian. It does not grow with the
number of times the simulation is applied. `Trotter` and `Qubitised` are specified
either by their parameters or by a target `error`, never both; the other half is
resolved on construction, after which `error` is the bound achieved by the resolved
parameters - at most the requested target, since parameters are rounded up.

`QDRIFT` is random, so it simulates no fixed Hamiltonian: its `error` is zero and its
inaccuracy is instead the per-application `channel_error`, which lowers the phase
estimation's `success_probability`.

`energy_scale` is the energy per cycle of measured phase, used by the phase estimation
algorithms to convert between phase precision and energy error.

See `quiche.budget.simulation` for the bounds and their references.
"""

from math import pi

from pydantic import ConfigDict, PositiveFloat, PositiveInt
from pydantic.dataclasses import dataclass

from quiche._validation import require_exactly_one
from quiche.budget.simulation import (
    get_alpha,
    get_qdrift_channel_error,
    get_qubitisation_error,
    get_qubitisation_index_ancillas,
    get_qubitisation_phase_ancillas,
    get_simulation_time,
    get_trotter_error,
    get_trotter_reps,
    get_trotter_unitary_error,
)
from quiche.core import PauliSum, Seed


@dataclass(frozen=True)
class Trotter:
    """
    Trotter-Suzuki product formula of a given `order` (1, or even).

    `time` defaults to `pi / alpha`, the longest time for which the measured phase
    cannot wrap. Give exactly one of `reps` (Trotter steps) or the energy `error`.
    """

    hamiltonian: PauliSum
    order: PositiveInt = 2
    time: PositiveFloat | None = None
    reps: PositiveInt | None = None
    error: PositiveFloat | None = None

    def __post_init__(self) -> None:
        """Resolve the steps or error from the other."""
        if self.order > 1 and self.order % 2 == 1:
            msg = f"Trotter order must be 1 or even, got {self.order}."
            raise ValueError(msg)
        require_exactly_one(reps=self.reps, error=self.error)
        if self.time is None:
            object.__setattr__(self, "time", get_simulation_time(self.hamiltonian))
        if self.reps is None:
            reps = get_trotter_reps(self.hamiltonian, self.time, self.order, self.error)
            object.__setattr__(self, "reps", reps)
        error = get_trotter_error(self.hamiltonian, self.time, self.order, self.reps)
        object.__setattr__(self, "error", error)

    @property
    def unitary_error(self) -> float:
        """Get the operator-norm error of one application of the evolution."""
        return get_trotter_unitary_error(
            self.hamiltonian, self.time, self.order, self.reps
        )

    @property
    def channel_error(self) -> float:
        """Get the per-application channel error, zero for a deterministic formula."""
        return 0.0

    @property
    def energy_scale(self) -> float:
        """Get the energy per cycle of measured phase."""
        return 2 * pi / self.time

    @property
    def num_ancillas(self) -> int:
        """Get the number of ancillas used by the simulation."""
        return 0


# `Seed` admits `bytearray`, which pydantic has no schema for.
@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class QDRIFT:
    """
    QDRIFT randomised product formula of `reps` sampled terms per application.

    `time` defaults as for `Trotter`. `seed` fixes the sampled term sequence.
    """

    hamiltonian: PauliSum
    reps: PositiveInt
    time: PositiveFloat | None = None
    seed: Seed = None

    def __post_init__(self) -> None:
        """Resolve the default time."""
        if self.time is None:
            object.__setattr__(self, "time", get_simulation_time(self.hamiltonian))

    @property
    def error(self) -> float:
        """Get the energy error, zero since no fixed Hamiltonian is simulated."""
        return 0.0

    @property
    def channel_error(self) -> float:
        """Get the diamond-distance error of one application."""
        return get_qdrift_channel_error(self.hamiltonian, self.time, self.reps)

    @property
    def energy_scale(self) -> float:
        """Get the energy per cycle of measured phase."""
        return 2 * pi / self.time

    @property
    def num_ancillas(self) -> int:
        """Get the number of ancillas used by the simulation."""
        return 0


@dataclass(frozen=True)
class Qubitised:
    """
    Qubitisation walk operator over an LCU block encoding.

    Give exactly one of `num_phase_ancillas` (Prepare angle precision) or the energy
    `error`. The index register is always sized to select every term. Only Qualtran
    models the finite Prepare precision; the QuEST and CUDA-Q backends prepare exactly.
    """

    hamiltonian: PauliSum
    num_phase_ancillas: PositiveInt | None = None
    error: PositiveFloat | None = None

    def __post_init__(self) -> None:
        """Resolve the phase ancillas or error from the other."""
        require_exactly_one(
            num_phase_ancillas=self.num_phase_ancillas, error=self.error
        )
        if self.num_phase_ancillas is None:
            num_phase_ancillas = get_qubitisation_phase_ancillas(
                self.hamiltonian, self.error
            )
            object.__setattr__(self, "num_phase_ancillas", num_phase_ancillas)
        error = get_qubitisation_error(self.hamiltonian, self.num_phase_ancillas)
        object.__setattr__(self, "error", error)

    @property
    def channel_error(self) -> float:
        """Get the per-application channel error, zero for a deterministic walk."""
        return 0.0

    @property
    def energy_scale(self) -> float:
        """Get the largest energy per cycle of phase in the `alpha cos` decoding."""
        return 2 * pi * get_alpha(self.hamiltonian)

    @property
    def num_index_ancillas(self) -> int:
        """Get the number of ancillas selecting the Hamiltonian term."""
        return get_qubitisation_index_ancillas(self.hamiltonian)

    @property
    def num_ancillas(self) -> int:
        """Get the number of ancillas used by the simulation."""
        return self.num_index_ancillas + self.num_phase_ancillas


type SimulationMethod = Trotter | QDRIFT | Qubitised
