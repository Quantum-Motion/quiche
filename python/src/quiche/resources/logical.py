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

from math import log2
from typing import Literal

from qualtran import Bloq
from qualtran.resource_counting import (
    GateCounts,
    QECGatesCost,
    QubitCount,
    get_cost_value,
)
from qualtran.resource_counting.generalizers import ignore_split_join


def logical_gate_resources(circuit: Bloq) -> GateCounts:
    """Calculate estimated logical gate cost of a bloq provided an error budget."""
    # Ignores soquet joining and splitting operations needed within Qualtran to join
    # bloqs with different signatures.
    return get_cost_value(circuit, QECGatesCost(), generalizer=[ignore_split_join])


def logical_qubit_resources(circuit: Bloq) -> int:
    """Calculate estimated logical gate cost of a bloq provided an error budget."""
    # Ignores soquet joining and splitting operations needed within Qualtran to join
    # bloqs with different signatures.
    return get_cost_value(circuit, QubitCount(), generalizer=[ignore_split_join])


# TODO(Annina): Implement other synthesis methods and add capabilities to account for
# additional ancilla due to synthesis methods. Although and bloqs also incur additional
# ancillas, these are handled in the my_static_cost subroutine of each bloq and do not
# need to be accounted for during postprocessing like rotations.


def logical_rotations_to_tgates(
    gates: GateCounts,
    error: float,
    rotation_synthesis: Literal["direct"],
) -> GateCounts:
    """
    Transform rotation gates to T gates up to a given synthesis error.

    Parameters
    ----------
    gates : GateCounts
        Logical gate counts. ``gates.rotation`` is the number of arbitrary-angle
        single-qubit rotations to synthesise.
    error : float
        Total synthesis error budget across all rotations, as an operator-norm
        distance. Errors are assumed to add linearly (triangle inequality).
    rotation_synthesis : Literal["direct"]
        Synthesis method. Only ``"direct"`` is supported currently.

    Returns
    -------
    GateCounts
        Gate counts with ``rotation`` set to 0 and the synthesised T gates added
        to ``t``. If there are no rotations, ``gates`` is returned unchanged.

    Raises
    ------
    ValueError
        If ``rotation_synthesis`` is not recognised.

    References
    ----------
    .. [RossSelinger2016] N. J. Ross and P. Selinger, 'Optimal ancilla-free Clifford+T
        approximation of z-rotations', Jun. 15, 2016, arXiv: arXiv:1403.2975. doi:
        10.48550/arXiv.1403.2975.

    """
    gc_dict = gates.asdict()
    num_rotations = int(gates.rotation)
    if num_rotations == 0:
        # nothing to be done, return original gates
        return gates

    # Depending on the rotation synthesis method, convert the rotations to the number
    # of T gates.
    if rotation_synthesis == "direct":
        # Calculate the error allowed per rotation.
        eps_per_rotation = error / num_rotations
        ts_per_rotation = int(3 * log2(1 / eps_per_rotation))
        total_ts = num_rotations * ts_per_rotation
    else:
        error_msg = f"Rotation synthesis method {rotation_synthesis} not recognised."
        raise ValueError(error_msg)

    # Now that the rotations have been converted, set number of rotations to zero and
    # add the calculated number of T gates to the total.
    gc_dict["rotation"] = 0
    gc_dict["t"] += total_ts

    return GateCounts(**gc_dict)
