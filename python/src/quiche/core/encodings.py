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

"""Binary word encodings of rotation angles for QROM data loading."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def angles_to_integer_representation(
    angles: ArrayLike, num_bits: int, *, rng: np.random.Generator | None = None
) -> NDArray[np.int64]:
    """
    Angles (radians) -> integers q in [0, 2**(num_bits - 1)).

    x = (theta mod 2*pi) / (4*pi)    in [0, 1/2)
    q = x * 2**num_bits              (rounded to an integer)
    q = q mod 2**(num_bits - 1)      (rounding can land exactly on 2**(num_bits - 1))
    """
    x = np.mod(np.asarray(angles, dtype=float), 2 * np.pi) / (4 * np.pi)
    q = x * 2**num_bits
    q = np.rint(q) if rng is None else np.floor(q + rng.random(q.shape))
    return np.mod(q.astype(np.int64), 2 ** (num_bits - 1))


def pack_into_word(q: ArrayLike, num_bits: int) -> NDArray[np.object_]:
    """
    Pack integers along the last axis.

    W = sum_k q_k * 2**(num_bits * (K - 1 - k))
    """
    q = np.asarray(q)
    num_angles = q.shape[-1]
    words = np.empty(q.shape[:-1], dtype=object)  # Python ints: rows can exceed 64 bits
    for idx in np.ndindex(words.shape):
        words[idx] = sum(
            int(q[(*idx, k)]) << (num_bits * (num_angles - 1 - k))
            for k in range(num_angles)
        )
    return words
