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

"""Backend-neutral QDRIFT term sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from quiche.core.paulis import PauliSum

# A nonnegative integer seed, or None to draw fresh entropy from the OS.
Seed = int | None


def _seed_sequences(seed: Seed) -> list[np.random.SeedSequence]:
    """Independent seed sequences for count and order sampling."""
    return np.random.SeedSequence(seed).spawn(2)


def _sample_counts(
    paulis: PauliSum, reps: int, seed_sequence: np.random.SeedSequence
) -> NDArray[np.int_]:
    # Identity deliberately not included (applied as a global phase).
    positive_coefficients = tuple(map(abs, paulis.coefficients))
    probabilities = np.asarray(positive_coefficients) / sum(positive_coefficients)
    rng = np.random.default_rng(seed_sequence)
    return rng.multinomial(reps, probabilities)


def sample_qdrift_counts(paulis: PauliSum, reps: int, seed: Seed) -> NDArray[np.int_]:
    """
    Draw how many times each word of a PauliSum appears in `reps` QDRIFT samples.

    Word `j` is sampled with probability `|paulis.coefficients[j]| / lam`, where `lam`
    is the 1-norm of the non-identity coefficients. For a given (non-None) `seed` the
    counts are consistent with `sample_qdrift_indices`.
    """
    return _sample_counts(paulis, reps, _seed_sequences(seed)[0])


def sample_qdrift_indices(
    paulis: PauliSum, reps: int, seed: Seed = None
) -> NDArray[np.int_]:
    """
    Draw `reps` QDRIFT term indices for a PauliSum.

    Term `j` is sampled with probability `|paulis.coefficients[j]| / lam`,
    independently each draw. Shared by every backend that lowers a QDRIFT spec, so
    they realise the same random product formula for a given seed - the QuEST
    backend samples independently in C++ and cannot be made to agree, but the
    Qualtran and CUDA-Q backends can and should.

    The term counts are drawn first (see `sample_qdrift_counts`) and then randomly
    ordered, using independent streams spawned from `seed`.
    """
    count_sequence, order_sequence = _seed_sequences(seed)
    counts = _sample_counts(paulis, reps, count_sequence)
    indices = np.repeat(np.arange(len(paulis.coefficients)), counts)
    return np.random.default_rng(order_sequence).permutation(indices)
