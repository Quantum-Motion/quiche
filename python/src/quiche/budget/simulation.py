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
Methods for determining Hamiltonian simulation parameters from error budgets.

The simulation error is the error incurred during one instance of the Hamiltonian
simulation. For Trotter methods it is the Trotter error; for qubitisation it is the
error of the Prepare operation in the walk operator.
"""

# TODO(Annina): If the simulation is used multiple times (e.g. in QPE), the simulation
# error should account for the total error, not just the error of a single simulation
# as it is currently.

from math import ceil, log2, pi

from quiche.core import PauliSum

_MIN_PHASE_ANCILLAS = 2


def get_simulation_time(paulis: PauliSum) -> float:
    """Get the simulation time required for correct period in Trotter methods."""
    return pi / paulis.lam


def get_qdrift_reps(paulis: PauliSum, time: float, error: float) -> int:
    """Get the QDRIFT repetitions to simulate a given PauliSum within `error`."""
    # See Eq. (3) and discussion underneath in arxiv:arXiv:1811.08017 [ Official
    # reference: Campbell, Phys. Rev. Lett. 123 (2019)]
    return ceil(2 * paulis.lam**2 * time**2 / error)


def get_qdrift_error(paulis: PauliSum, time: float, reps: int) -> float:
    """Get the error of simulating a given PauliSum with `reps` QDRIFT repetitions."""
    return 2 * paulis.lam**2 * time**2 / reps


def get_trotter_reps(paulis: PauliSum, time: float, order: int, error: float) -> int:
    """Get the Trotter steps to simulate a given PauliSum within `error`."""
    return ceil(
        (paulis.lam ** (order + 1) * time ** (order + 1) / error) ** (1.0 / order)
    )


def get_trotter_error(paulis: PauliSum, time: float, order: int, reps: int) -> float:
    """Get the error of simulating a given PauliSum with `reps` Trotter steps."""
    return (paulis.lam * time) ** (order + 1) / reps**order


def get_qubitisation_index_ancillas(paulis: PauliSum) -> int:
    """Get the number of index ancillas needed to select every term of a PauliSum."""
    return ceil(log2(paulis.n_terms))


def get_qubitisation_phase_ancillas(paulis: PauliSum, error: float) -> int:
    """Get the Prepare rotation ancillas to block-encode a PauliSum within `error`."""
    num_index_ancillas = get_qubitisation_index_ancillas(paulis)
    if num_index_ancillas == 0:
        return _MIN_PHASE_ANCILLAS
    return max(
        ceil(log2(2.0 * num_index_ancillas / error)),
        _MIN_PHASE_ANCILLAS,
    )


def get_qubitisation_error(paulis: PauliSum, num_phase_ancillas: int) -> float:
    """Get the Prepare error of block-encoding a PauliSum with `num_phase_ancillas`."""
    num_index_ancillas = get_qubitisation_index_ancillas(paulis)
    return 2.0 * num_index_ancillas / 2**num_phase_ancillas
