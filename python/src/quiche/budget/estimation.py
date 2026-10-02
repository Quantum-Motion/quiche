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
Error bounds and success probabilities for phase estimation.

The protocol modelled here runs QPE `repetitions` times on the prepared state and keeps
the minimum energy reading. With overlap `p0` (the squared overlap of the prepared state
with the ground state), each run collapses onto the ground state with probability `p0`,
so roughly `1/p0` runs are needed. Each run also has a small chance of a spuriously low
reading; over `1/p0` runs these add up unless each run's tail probability shrinks by a
factor of `p0`, which costs `log2(1/p0)` extra ancillas [LinTong2022, Sec. I A].

Energy errors are in the units of the Hamiltonian's coefficients.

References
----------
[NielsenChuang] Nielsen, Chuang, "Quantum Computation and Quantum Information",
    Sec. 5.2.1, Eq. (5.35).
[LinTong2022] Lin, Tong, "Heisenberg-limited ground-state energy estimation for early
    fault-tolerant quantum computers", PRX Quantum 3, 010318 (2022).
[Campbell2019] Campbell, "Random compiler for fast Hamiltonian simulation",
    Phys. Rev. Lett. 123, 070503 (2019), App. E.

"""

from math import ceil, log, log2

# Smallest number of bits beyond the precision bits for which the tail bound holds.
MIN_EXTRA_ANCILLAS = 2
# Default probability of missing the ground state in every repetition.
DEFAULT_MISS_PROBABILITY = 0.05


def get_precision_bits(energy_scale: float, error: float) -> int:
    """
    Get the fewest phase bits resolving energies to within `error`.

    `energy_scale` is the energy per cycle of phase, so `b` bits resolve
    `energy_scale / 2^b`.
    """
    return max(1, ceil(log2(energy_scale / error)))


def get_estimation_error(energy_scale: float, precision_bits: int) -> float:
    """Get the energy resolution of `precision_bits` phase bits."""
    return energy_scale / 2**precision_bits


def get_overlap_bits(overlap: float) -> int:
    """Get the extra bits protecting the minimum over `~1/overlap` repetitions."""
    return ceil(log2(1 / overlap))


def get_tail_probability(extra_bits: int) -> float:
    """
    Get the probability that one QPE run misses the precision window.

    With `extra_bits` bits beyond the precision bits, the phase is accurate to the
    precision bits with probability at least `1 - 1 / (2 (2^p - 2))` [NielsenChuang].
    """
    if extra_bits < MIN_EXTRA_ANCILLAS:
        msg = f"Need at least {MIN_EXTRA_ANCILLAS} extra bits, got {extra_bits}."
        raise ValueError(msg)
    return 1 / (2 * (2**extra_bits - 2))


def get_default_repetitions(overlap: float, tail_probability: float) -> int:
    """Get the fewest repetitions missing the ground state with probability <= 5%."""
    hit = overlap * (1 - tail_probability)
    if hit >= 1:
        return 1
    return max(1, ceil(log(DEFAULT_MISS_PROBABILITY) / log(1 - hit)))


def get_success_probability(
    *,
    overlap: float,
    tail_probability: float,
    repetitions: int,
    applications: int,
    channel_error: float,
) -> float:
    """
    Get a lower bound on the probability that the minimum reading is within the error.

    By the union bound, the protocol fails only if no repetition lands on the ground
    state within the precision window, if any repetition reads spuriously outside it,
    or if the simulation channel error changes any repetition's outcome. The last term
    is `2 x` the total diamond-distance error of the controlled applications
    [Campbell2019, Eq. (E4)].
    """
    miss = (1 - overlap * (1 - tail_probability)) ** repetitions
    tails = repetitions * tail_probability
    channel = 2 * repetitions * applications * channel_error
    return max(0.0, 1 - miss - tails - channel)
