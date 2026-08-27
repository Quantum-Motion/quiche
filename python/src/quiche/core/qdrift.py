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

from random import Random

from quiche.core.paulis import PauliSum

# Any value accepted by `random.Random.seed`.
Seed = int | float | str | bytes | bytearray | None


def sample_qdrift_indices(
    paulis: PauliSum, reps: int, seed: Seed = None
) -> tuple[int, ...]:
    """
    Draw `reps` QDRIFT term indices for a PauliSum.

    Term `j` is sampled with probability `|paulis.coefficients[j]| / paulis.lam`,
    independently each draw. Shared by every backend that lowers a QDRIFT spec, so
    they realise the same random product formula for a given seed - the QuEST
    backend samples independently in C++ and cannot be made to agree, but the
    Qualtran and CUDA-Q backends can and should.

    Uses a private `random.Random` instance rather than reseeding the module-global
    generator: thread-safe, and `seed=0` is honoured (a truthiness check such as
    `if seed:` would wrongly treat it as unseeded).
    """
    weights = tuple(map(abs, paulis.coefficients))
    return tuple(Random(seed).choices(range(paulis.n_terms), weights, k=reps))  # noqa: S311
