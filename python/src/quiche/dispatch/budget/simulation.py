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

"""Methods for determining Hamiltonian simulation parameters from error budgets."""

from math import ceil, log2, pi

from quiche.core import Errors, PauliSum


def get_simulation_time(paulis: PauliSum) -> float:
    """Get the simulation time required for correct period in Trotter methods."""
    return pi / paulis.lam


def get_qdrift_params(paulis: PauliSum, errors: Errors) -> tuple[float, int]:
    """Get the QDRIFT settings to simulate a given PauliSum within a set Error."""
    time = get_simulation_time(paulis)
    # See Eq. (3) and discussion underneath in arxiv:arXiv:1811.08017 [ Official
    # reference: Campbell, Phys. Rev. Lett. 123 (2019)]
    reps = ceil(2 * paulis.lam**2 * time**2 / errors.simulation)

    return (time, reps)


# TODO(Vasco): implement the order calculation
def get_trotter_params(paulis: PauliSum, errors: Errors) -> tuple[float, int, int]:
    """Get the Trotter settings to simulate a given PauliSum within a set Error."""
    time = get_simulation_time(paulis)
    order = 2
    reps = ceil(
        (paulis.lam ** (order + 1) * time ** (order + 1) / errors.simulation)
        ** (1.0 / order)
    )

    return (time, order, reps)


def get_qubitisation_ancillas(paulis: PauliSum, errors: Errors) -> tuple[int, int]:
    """Get the Qubitisation settings to simulate a given PauliSum within a set Error."""
    num_index_ancillas = ceil(log2(paulis.n_terms))
    num_phase_ancillas = max(
        ceil(log2(2.0 * num_index_ancillas / errors.state_prep)),
        2,
    )

    return (num_index_ancillas, num_phase_ancillas)
