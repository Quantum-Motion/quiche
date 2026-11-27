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

"""Resource calculations for quantum circuits and routines."""

from math import ceil, log2
from typing import Literal

import attrs
from qualtran import Bloq
from qualtran.resource_counting import (
    GateCounts,
    QECGatesCost,
    QubitCount,
    get_cost_value,
)
from qualtran.resource_counting.generalizers import ignore_split_join


def logical_gate_resources(circuit: Bloq) -> GateCounts:
    """Calculate estimated logical gate cost of a bloq."""
    # Ignores soquet joining and splitting operations needed within Qualtran to join
    # bloqs with different signatures.
    return get_cost_value(circuit, QECGatesCost(), generalizer=[ignore_split_join])


def logical_qubit_resources(circuit: Bloq) -> int:
    """Calculate estimated logical qubit cost of a bloq."""
    # Ignores soquet joining and splitting operations needed within Qualtran to join
    # bloqs with different signatures.
    return get_cost_value(circuit, QubitCount(), generalizer=[ignore_split_join])


# TODO(Annina): Implement other synthesis methods and add capabilities to account for
# additional ancilla due to synthesis methods. Although and bloqs also incur additional
# ancillas, these are handled in the my_static_cost subroutine of each bloq and do not
# need to be accounted for during postprocessing like rotations.
def logical_rotations_to_tgates(
    gates: GateCounts,
    total_error: float,
    rotation_synthesis: Literal["direct"],
) -> GateCounts:
    """
    Update gate counts converting rotation gates to T gates.

    Parameters
    ----------
    gates : GateCounts
        Gate count dictionary possibly including rotations to convert.
    total_error : float
        Maximum allowable total rotation synthesis error. Subadditivity of error
        contributions assumed, (naively) splitting errors equally across all rotations.
        Must be positive.
    rotation_synthesis : Literal["direct"]
        Rotation synthesis algorithm to use for costing.

    Returns
    -------
    GateCounts
        Gate count dictionary with any rotations converted to T gates.

    Raises
    ------
    ValueError
        If ``total_error`` isn't positive.
        If ``rotation_synthesis`` isn't one of the accepted rotation synthesis
        algorithms.
    """
    if total_error <= 0:
        error_msg = "Total rotation synthesis error must be positive."
        raise ValueError(error_msg)

    num_rotations = int(gates.rotation)
    if num_rotations == 0:
        return gates

    if rotation_synthesis == "direct":
        error_per_rotation = total_error / num_rotations
        t_per_rotation = max(0, ceil(3 * log2(1 / error_per_rotation)))
        total_t = num_rotations * t_per_rotation
    else:
        error_msg = f"Rotation synthesis method {rotation_synthesis} not recognised."
        raise ValueError(error_msg)

    return attrs.evolve(gates, rotation=0, t=gates.t + total_t)
