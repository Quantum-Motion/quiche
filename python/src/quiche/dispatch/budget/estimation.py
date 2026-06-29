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

from quiche.core.errors import Errors


def get_textbook_qpe_ancillas(e: Errors) -> int:
    """Get the number of ancillas required for Textbook QPE."""
    # TODO(Annina): add the one-norm of the Hamiltonian.
    return ceil(log2(1 / e.estimation)) + ceil(log2(1 / e.overlap)) + 4


def get_kitaev_qpe_rounds(e: Errors) -> int:
    """Get the number of rounds for Kitaev single-ancilla QPE."""
    return get_textbook_qpe_ancillas(e)
