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

"""Methods for determining phase estimation parameters from error budgets."""

from math import ceil, log2

# Extra ancillas beyond the precision and overlap terms, from the textbook
# success-probability analysis.
_EXTRA_ANCILLAS = 4


def _overlap_ancillas(overlap: float) -> int:
    """Get the ancillas needed to compensate for an imperfect initial-state overlap."""
    return ceil(log2(1 / overlap))


# TODO(Annina): Decide whether estimation error should be input in Ha. In that case
# we will have to convert it to a dimensionless error in the phase.
def get_textbook_qpe_ancillas(error: float, overlap: float) -> int:
    """Get the number of ancillas required for Textbook QPE."""
    # TODO(Annina): add the one-norm of the Hamiltonian.
    return ceil(log2(1 / error)) + _overlap_ancillas(overlap) + _EXTRA_ANCILLAS


def get_textbook_qpe_error(num_ancillas: int, overlap: float) -> float:
    """Get the estimation error achieved by Textbook QPE with `num_ancillas`."""
    precision_bits = num_ancillas - _overlap_ancillas(overlap) - _EXTRA_ANCILLAS
    if precision_bits < 0:
        minimum = _overlap_ancillas(overlap) + _EXTRA_ANCILLAS
        msg = (
            f"Textbook QPE needs at least {minimum} ancillas for overlap {overlap}, "
            f"got {num_ancillas}."
        )
        raise ValueError(msg)
    return 2.0**-precision_bits


def get_kitaev_qpe_rounds(error: float, overlap: float) -> int:
    """Get the number of rounds for Kitaev single-ancilla QPE."""
    return get_textbook_qpe_ancillas(error, overlap)


def get_kitaev_qpe_error(num_rounds: int, overlap: float) -> float:
    """Get the estimation error achieved by Kitaev single-ancilla QPE."""
    return get_textbook_qpe_error(num_rounds, overlap)
