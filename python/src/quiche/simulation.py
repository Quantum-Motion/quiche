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

Each method is specified either by its parameters or by a target error, never both.
The other half is resolved on construction, after which `error` is the bound achieved
by the resolved parameters - at most the requested target, since parameters are
rounded up.
"""

from pydantic import ConfigDict, PositiveFloat, PositiveInt
from pydantic.dataclasses import dataclass

from quiche._validation import require_exactly_one
from quiche.budget.simulation import (
    get_qdrift_error,
    get_qdrift_reps,
    get_qubitisation_error,
    get_qubitisation_index_ancillas,
    get_qubitisation_phase_ancillas,
    get_simulation_time,
    get_trotter_error,
    get_trotter_reps,
)
from quiche.core import PauliSum, Seed


@dataclass(frozen=True)
class Trotter:
    """
    Trotter-Suzuki product formula of a given `order`.

    `time` defaults to the time giving the correct phase period for QPE (pi / lam).
    Give exactly one of `reps` (Trotter steps) or `error`.
    """

    hamiltonian: PauliSum
    order: PositiveInt = 2
    time: PositiveFloat | None = None
    reps: PositiveInt | None = None
    error: PositiveFloat | None = None

    def __post_init__(self) -> None:
        """Resolve the steps or error from the other."""
        require_exactly_one(reps=self.reps, error=self.error)
        if self.time is None:
            object.__setattr__(self, "time", get_simulation_time(self.hamiltonian))
        if self.reps is None:
            reps = get_trotter_reps(self.hamiltonian, self.time, self.order, self.error)
            object.__setattr__(self, "reps", reps)
        error = get_trotter_error(self.hamiltonian, self.time, self.order, self.reps)
        object.__setattr__(self, "error", error)

    @property
    def num_ancillas(self) -> int:
        """Get the number of ancillas used by the simulation."""
        return 0


# `Seed` admits `bytearray`, which pydantic has no schema for.
@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class QDRIFT:
    """
    QDRIFT randomised product formula.

    `time` defaults as for `Trotter`. Give exactly one of `reps` (sampled terms) or
    `error`. `seed` fixes the sampled term sequence.
    """

    hamiltonian: PauliSum
    time: PositiveFloat | None = None
    reps: PositiveInt | None = None
    error: PositiveFloat | None = None
    seed: Seed = None

    def __post_init__(self) -> None:
        """Resolve the repetitions or error from the other."""
        require_exactly_one(reps=self.reps, error=self.error)
        if self.time is None:
            object.__setattr__(self, "time", get_simulation_time(self.hamiltonian))
        if self.reps is None:
            reps = get_qdrift_reps(self.hamiltonian, self.time, self.error)
            object.__setattr__(self, "reps", reps)
        error = get_qdrift_error(self.hamiltonian, self.time, self.reps)
        object.__setattr__(self, "error", error)

    @property
    def num_ancillas(self) -> int:
        """Get the number of ancillas used by the simulation."""
        return 0


@dataclass(frozen=True)
class Qubitised:
    """
    Qubitisation walk operator over an LCU block encoding.

    Give exactly one of `num_phase_ancillas` (Prepare rotation precision) or
    `prepare_error`. The index register is always sized to select every term.
    """

    hamiltonian: PauliSum
    num_phase_ancillas: PositiveInt | None = None
    prepare_error: PositiveFloat | None = None

    def __post_init__(self) -> None:
        """Resolve the phase ancillas or Prepare error from the other."""
        require_exactly_one(
            num_phase_ancillas=self.num_phase_ancillas,
            prepare_error=self.prepare_error,
        )
        if self.num_phase_ancillas is None:
            num_phase_ancillas = get_qubitisation_phase_ancillas(
                self.hamiltonian, self.prepare_error
            )
            object.__setattr__(self, "num_phase_ancillas", num_phase_ancillas)
        error = get_qubitisation_error(self.hamiltonian, self.num_phase_ancillas)
        object.__setattr__(self, "prepare_error", error)

    @property
    def error(self) -> float:
        """Get the simulation error, which is the Prepare error."""
        return self.prepare_error

    @property
    def num_index_ancillas(self) -> int:
        """Get the number of ancillas selecting the Hamiltonian term."""
        return get_qubitisation_index_ancillas(self.hamiltonian)

    @property
    def num_ancillas(self) -> int:
        """Get the number of ancillas used by the simulation."""
        return self.num_index_ancillas + self.num_phase_ancillas


type SimulationMethod = Trotter | QDRIFT | Qubitised
