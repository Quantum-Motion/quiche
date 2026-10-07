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
Error bounds for Hamiltonian simulation methods.

Energy errors are in the units of the Hamiltonian's coefficients (Hartree for HamLib or
FCIDUMP inputs). Throughout, `lam` is the 1-norm of the non-identity coefficients and
`alpha = lam + |identity_coefficient|` bounds the spectral radius of the Hamiltonian.

References
----------
[Childs2021] Childs, Su, Tran, Wiebe, Zhu, "Theory of Trotter error with commutator
    scaling", Phys. Rev. X 11, 011020 (2021), arXiv:1912.08854.
[Schubert2023] Schubert, Mendl, "Trotter error with commutator scaling for the
    Fermi-Hubbard model", arXiv:2306.10603 (2023).
[Campbell2019] Campbell, "Random compiler for fast Hamiltonian simulation",
    Phys. Rev. Lett. 123, 070503 (2019), arXiv:1811.08017.
[Bhatia1984] Bhatia, Davis, "A bound for the spectral variation of a unitary operator",
    Linear Multilinear Algebra 15, 71 (1984).

"""

from functools import lru_cache
from math import asin, ceil, exp, factorial, log2, pi, sin

from quiche.core import PauliSum

from ._commutators import first_order_constant, second_order_constant

_MIN_PHASE_ANCILLAS = 2

# Largest number of terms for which the commutator Trotter constant is computed, per
# order. Order 1 costs O(n_terms^2 n_qubits) time; order 2 costs O(n_terms^3) time and
# O(n_terms^2) memory, a few seconds and ~200 MB at the cap. Larger Hamiltonians fall
# back to the coefficient-magnitude bound.
_MAX_COMMUTATOR_TERMS = {1: 20_000, 2: 5_000}


def get_alpha(paulis: PauliSum) -> float:
    """Get the 1-norm of all coefficients, including the identity coefficient."""
    return paulis.lam + abs(paulis.identity_coefficient)


def get_simulation_time(paulis: PauliSum) -> float:
    """
    Get the default evolution time for phase estimation, `pi / alpha`.

    Every eigenvalue `E` then satisfies `|E| t / (2 pi) <= 1/2`, so the measured phase
    never wraps, even when a backend folds the identity coefficient into the phase.
    """
    return pi / get_alpha(paulis)


@lru_cache(maxsize=32)
def get_trotter_constant(paulis: PauliSum, order: int) -> float:
    """
    Get `C` such that a Trotter step of size `dt` has unitary error `<= C dt^(order+1)`.

    The identity term commutes with everything and is applied exactly, so it is
    excluded. Orders 1 and 2 use the exact nested commutators of the Pauli terms
    [Childs2021, Props. 9-10], as evaluated in [Schubert2023] (see `_commutators`),
    which assumes the terms are applied in `PauliSum` order, as every backend does.
    Hamiltonians above `_MAX_COMMUTATOR_TERMS`, and higher orders, use
    `get_trotter_norm_constant`.
    """
    if paulis.n_terms <= _MAX_COMMUTATOR_TERMS.get(order, 0):
        if order == 1:
            return first_order_constant(paulis)
        return second_order_constant(paulis)
    return get_trotter_norm_constant(paulis, order)


def get_trotter_norm_constant(paulis: PauliSum, order: int) -> float:
    """
    Get the Trotter constant `C` from the coefficient magnitudes alone.

    Uses `||[A, B]|| <= 2 ||A|| ||B||`, so it holds for any term order and is never
    smaller than the commutator constant of `get_trotter_constant`.

    - Order 1: [Childs2021, Prop. 15].
    - Order 2: [Childs2021, Prop. 16].
    - Even order `p >= 4`: [Childs2021, Lemma 6, Eq. (22)] for a Suzuki formula with
      `2 * 5^(p/2 - 1)` stages; this also bounds formulas with fewer stages, such as the
      Forest-Ruth fourth-order formula used by CUDA-Q.
    """
    magnitudes = [abs(c) for c in paulis.coefficients]
    lam = paulis.lam
    if order == 1:
        return (lam**2 - sum(h**2 for h in magnitudes)) / 2
    if order == 2:
        nested = sum(h * (lam - h) ** 2 for h in magnitudes)
        inner = sum(h**2 * (lam - h) for h in magnitudes)
        return nested / 3 + inner / 6
    if order % 2 == 0:
        stages = 2 * 5 ** (order // 2 - 1)
        return (stages ** (order + 1) + 1) * lam ** (order + 1) / factorial(order + 1)
    msg = f"Trotter order must be 1 or even, got {order}."
    raise ValueError(msg)


def get_trotter_unitary_error(
    paulis: PauliSum, time: float, order: int, reps: int
) -> float:
    """Get the unitary error bound of `reps` Trotter steps simulating time `time`."""
    constant = get_trotter_constant(paulis, order)
    return reps * constant * (time / reps) ** (order + 1)


def unitary_to_energy_error(unitary_error: float, time: float) -> float:
    """
    Convert a unitary error at time `time` to an energy error.

    If `||U - exp(-iHt)|| <= eps`, the eigenvalues of `U` can be matched to those of
    `exp(-iHt)` with chordal distance at most `eps` [Bhatia1984], so each eigenvalue of
    the effective Hamiltonian `U = exp(-i H_eff t)` is within `2 arcsin(eps / 2) / t`
    of one of `H`. Repeated applications of `U` share the same `H_eff`, so this does
    not grow with the number of applications.
    """
    return 2 * asin(min(unitary_error, 2.0) / 2) / time


def energy_to_unitary_error(error: float, time: float) -> float:
    """Get the largest unitary error at time `time` within energy error `error`."""
    return 2 * sin(min(error * time, pi) / 2)


def get_trotter_error(paulis: PauliSum, time: float, order: int, reps: int) -> float:
    """Get the energy error bound of `reps` Trotter steps simulating time `time`."""
    unitary_error = get_trotter_unitary_error(paulis, time, order, reps)
    return unitary_to_energy_error(unitary_error, time)


def get_trotter_reps(paulis: PauliSum, time: float, order: int, error: float) -> int:
    """Get the fewest Trotter steps simulating `time` within energy error `error`."""
    unitary_error = energy_to_unitary_error(error, time)
    constant = get_trotter_constant(paulis, order)
    reps = max(1, ceil((constant * time ** (order + 1) / unitary_error) ** (1 / order)))
    # Correct floating-point rounding in the closed form, in either direction.
    while get_trotter_error(paulis, time, order, reps) > error:
        reps += 1
    while reps > 1 and get_trotter_error(paulis, time, order, reps - 1) <= error:
        reps -= 1
    return reps


def get_qdrift_channel_error(paulis: PauliSum, time: float, reps: int) -> float:
    """
    Get the diamond-distance error of one QDRIFT application of `reps` sampled terms.

    [Campbell2019, Eqs. (10)-(11)], with the diamond distance including a factor 1/2.
    """
    return 2 * paulis.lam**2 * time**2 / reps * exp(2 * paulis.lam * time / reps)


def get_qubitisation_index_ancillas(paulis: PauliSum) -> int:
    """Get the number of index ancillas needed to select every term of a PauliSum."""
    return ceil(log2(paulis.n_terms))


def get_prepare_levels(paulis: PauliSum) -> int:
    """
    Get the number of rotation levels in the Qualtran Prepare state preparation.

    `LCUBlockEncodingWrapper.from_hamiltonian` adds the identity as an extra term when
    its coefficient is nonzero.
    """
    n_terms = paulis.n_terms + (paulis.identity_coefficient != 0)
    return max(1, ceil(log2(n_terms)))


def get_qubitisation_error(paulis: PauliSum, num_phase_ancillas: int) -> float:
    """
    Get the energy error of block-encoding a PauliSum with finite-precision Prepare.

    The Prepare state is built by `levels` layers of amplitude rotations plus one layer
    of phase rotations, each angle stored in `num_phase_ancillas` bits, so the angle
    error is at most `pi / 2^b` and the state error at most `(levels + 1) pi / 2^(b+1)`.
    The 1-norm error of the term weights is at most twice that, and each weight
    multiplies `alpha`.
    """
    levels = get_prepare_levels(paulis)
    return get_alpha(paulis) * (levels + 1) * pi / 2**num_phase_ancillas


def get_qubitisation_phase_ancillas(paulis: PauliSum, error: float) -> int:
    """Get the fewest Prepare angle bits block-encoding a PauliSum within `error`."""
    levels = get_prepare_levels(paulis)
    bits = ceil(log2(get_alpha(paulis) * (levels + 1) * pi / error))
    return max(bits, _MIN_PHASE_ANCILLAS)
