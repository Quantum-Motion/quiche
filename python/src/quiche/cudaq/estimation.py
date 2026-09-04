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

"""Quantum phase estimation kernels for the CUDA-Q backend."""

# CUDA-Q's intrinsic gates (h, r1, swap, mz, ...) are not real Python names -
# they're recognised by the @cudaq.kernel AST bridge by bare-name pattern
# matching (confirmed: `hasattr(cudaq, "h")` is False), so ruff sees every
# in-kernel gate call as undefined. This file is otherwise plain kernel
# construction, so a file-level suppression is low-risk. The kernel factories
# below also take many named, optional simulation parameters, matching the
# equally wide `getPhaseTextbook*` signatures in quiche.quest.estimation.
# ruff: noqa: F821, PLR0913, PLR0917

from math import pi

import numpy as np

from quiche.core import PauliSum, Seed, Simulation
from quiche.cudaq._runtime import CudaqKernel, load_cudaq
from quiche.cudaq.simulation import (
    qdrift_evolution,
    qubitised_encoding,
    trotter_evolution,
)


def inverse_qft_kernel() -> CudaqKernel:
    """
    Build a `(qubits: cudaq.qview)` kernel applying the inverse QFT.

    CUDA-Q Algorithms has no QFT primitive; this is a from-scratch
    implementation (bit-reversal swaps, then the standard controlled-rotation
    ladder). Validated directly as a unitary matrix - built from
    `cudaq.get_state` on every computational basis input - against the
    closed-form `(1/sqrt(N)) exp(-2*pi*i*x*y/N)`: an exact match already in
    CUDA-Q's own qubit convention (qubit `i` = bit `i`), with no bit-reversal
    of the *output* needed. `textbook_qpe_kernel`'s controlled-power ladder
    assigns ancilla `k` to `U^(2^k)` specifically because of this convention -
    the more common "qubit 0 = most significant bit" textbook-diagram
    assignment is wrong for this particular circuit.
    """
    cudaq, _ = load_cudaq()

    @cudaq.kernel
    def inverse_qft(qubits: cudaq.qview) -> None:
        """Apply the inverse QFT: bit-reversal swaps, then rotations and Hadamards."""
        n = qubits.size()
        for i in range(n // 2):
            swap(qubits[i], qubits[n - 1 - i])
        for j in range(n):
            for k in range(j):
                angle = -pi / (2.0 ** (j - k))
                r1.ctrl(angle, qubits[j], qubits[k])
            h(qubits[j])

    return inverse_qft


def textbook_qpe_kernel(
    hamiltonian: PauliSum,
    simulation: Simulation,
    num_qpe_ancillas: int,
    time: float,
    reps: int,
    order: int = 2,
    *,
    seed: Seed = None,
    n_qubits: int | None = None,
) -> CudaqKernel:
    """
    Build the Textbook QPE kernel for `Simulation.Trotter`/`Simulation.QDRIFT`.

    The returned kernel has signature `(data: cudaq.qview) -> float`: like
    `bitstring_kernel`/`inverse_qft_kernel`, it operates in place on an
    already-allocated register rather than accepting a `cudaq.State` - the
    caller allocates the register (`cudaq.qvector(n_qubits)`), runs a
    state-preparation kernel on it (mirroring `to_qualtran`'s "compose with a
    state-preparation Bloq" and `to_quest`'s "compose with a state preparation
    routine" contracts), then this kernel, all within one compiled top-level
    kernel - see `QPESpec.to_cudaq`'s docstring for a worked example. `data`
    must be at least `n_qubits` wide, the caller's responsibility (same
    requirement as `apply_trotter`/`bitstring_kernel`).

    The energy recovery is done *inside* the kernel, not left to the caller:
    after the inverse QFT, the ancilla register is measured and decoded to a
    phase (unsigned integer `y`, ancilla `k` weighted `2**k`, wrapped to
    `[-0.5, 0.5)`), then converted to an energy via `-phase * (2*pi/time)` -
    the minus sign because `U = exp(-i H t)` kicks back phase
    `exp(-i * eigenvalue * t)`, so the measured phase is
    `-eigenvalue * t / (2*pi)` (mod 1), not `+eigenvalue * t / (2*pi)`
    (confirmed directly: an uncorrected `+` sign recovers +0.98 for `H = Z`'s
    `|1>` eigenstate, whose true eigenvalue is -1). `identity_coefficient` is
    NOT added again here - the per-rung `r1` correction below already folds it
    into the measured phase, so doing both double-counts it (also confirmed
    directly: with the addition, `H = Z + 0.3*I`'s `|1>` eigenstate recovers
    -0.35 instead of its true eigenvalue -0.7; dropping the addition and
    refining `num_qpe_ancillas` converges cleanly to -0.7). This is all
    classical kernel-mode arithmetic on the measured bits, confirmed to compile
    and execute directly (no `cudaq.sample`/host-side decode needed). This is one
    shot's own energy estimate; call the kernel directly for a single value, or
    `cudaq.run(kernel, data, shots_count=n)` to collect `n` of them (a
    `list[float]`) when `data` isn't already a clean eigenstate and some
    aggregation - e.g. the mode - across shots is needed. `to_qualtran`/
    `to_quest` don't have an equivalent: nothing analogous to CUDA-Q's
    in-kernel classical control flow exists for a Bloq or a QuEST routine, so
    their energy recovery stays the caller's job.

    Ladder rung `k` (`k = 0, ..., num_qpe_ancillas - 1`) applies
    controlled-U^(2^k) by repeating the base `(time, reps, order)`-parametrized
    controlled block `2**k` times - mathematically identical to scaling `time`
    and `reps` by `2**k` in one call (same total step count, same `dt`), but
    avoids needing a fresh, longer QDRIFT sample per rung. Cost grows as `2^k`
    per rung (`2^n - 1` total blocks) - inherent to the textbook algorithm, not
    specific to this implementation (QuEST's C++ backend has the same growth).
    `iterative_qpe_kernel`'s single-ancilla, sequential-round construction has
    the *same* total block count (also `2^n - 1`, summed the other way round)
    - its saving over this ladder is qubit count (`1` ancilla instead of
    `num_qpe_ancillas`), not gate count; an earlier version of this docstring
    said "cheaper", which was imprecise about which resource.

    `identity_coefficient` (dropped from the circuit by `trotter_evolution`/
    `qdrift_evolution` - see their docstrings) is restored per rung with a
    single `r1` phase gate on that rung's ancilla, since `r1(t) = diag(1,
    e^(i*t))` naturally applies only when the ancilla is `|1>` - this alone is
    what makes the measured phase reflect the full Hamiltonian's eigenvalue,
    not just the non-identity part (see the decode note above).

    `Simulation.Qubitised` raises `NotImplementedError`: it has no `(time,
    reps, order, seed)` shape to share with this function (no evolution-time
    concept for a qubitisation walk operator) - use `qubitised_qpe_kernel`
    directly instead.
    """
    cudaq, _ = load_cudaq()

    match simulation:
        case Simulation.Trotter:
            evolution = trotter_evolution(hamiltonian, n_qubits=n_qubits)
            base_steps, base_order = reps, order
        case Simulation.QDRIFT:
            evolution = qdrift_evolution(
                hamiltonian, reps, seed=seed, n_qubits=n_qubits
            )
            base_steps, base_order = 1, 1
        case Simulation.Qubitised:
            msg = (
                "Simulation.Qubitised has no (time, reps, order, seed) shape "
                "to share with Trotter/QDRIFT - use "
                "quiche.cudaq.estimation.qubitised_qpe_kernel directly."
            )
            raise NotImplementedError(msg)

    coefficients = evolution.coefficients
    words = [cudaq.pauli_word(word) for word in evolution.words]
    identity_coefficient = evolution.identity_coefficient
    # A plain Python float, computed host-side: kernel-mode has no `float()` cast
    # (confirmed - it fails to compile), so the divisor is precomputed here and
    # captured as a constant rather than derived from `num_qpe_ancillas` in-kernel.
    dimension = float(1 << num_qpe_ancillas)

    from cudaq_algorithms.trotter import apply_trotter  # noqa: PLC0415

    inverse_qft = inverse_qft_kernel()

    @cudaq.kernel
    def qpe(data: cudaq.qview) -> float:
        """Ladder controlled-U^(2^k), inverse QFT, then decode the energy in-kernel."""
        ancilla = cudaq.qvector(num_qpe_ancillas)
        h(ancilla)
        for k in range(num_qpe_ancillas):
            power = 1 << k
            for _ in range(power):
                cudaq.control(
                    apply_trotter,
                    ancilla[k],
                    coefficients,
                    words,
                    time,
                    base_steps,
                    base_order,
                    data,
                )
            r1(-identity_coefficient * time * power, ancilla[k])
        inverse_qft(ancilla)

        y = 0.0
        for k in range(num_qpe_ancillas):
            if mz(ancilla[k]):
                y += 1 << k
        phase = y / dimension
        if phase >= 0.5:
            phase -= 1.0
        return -phase * (2.0 * pi / time)

    return qpe


def qubitised_qpe_kernel(
    hamiltonian: PauliSum,
    num_qpe_ancillas: int,
    *,
    n_qubits: int | None = None,
) -> CudaqKernel:
    """
    Build the Textbook QPE kernel for `Simulation.Qubitised`.

    Same contract as `textbook_qpe_kernel`: signature `(data: cudaq.qview) ->
    float`, operates in place on an already-allocated register, decodes one
    shot's own energy estimate in-kernel (call directly for a single value, or
    `cudaq.run(kernel, data, shots_count=n)` for `n` of them). A separate
    function rather than a branch inside `textbook_qpe_kernel`, because
    Qubitised has no `(time, reps, order, seed)` shape to share - matching
    both other backends: QuEST's `getPhaseTextbookQubitised` is separate from
    `getPhaseTextbookTrotter`/`getPhaseTextbookQDRIFT`, and Qualtran's
    `QubitisationLadder` Bloq is separate from `TrotterLadder` (which Trotter
    and QDRIFT share).

    Builds `qubitised_encoding(hamiltonian, n_qubits=n_qubits)` fresh and uses
    its own `.num_ancilla`/`.alpha` as the sole source of truth for register
    width and the decode formula - deliberately not any externally-supplied
    ancilla count. `get_qubitisation_ancillas` (`dispatch/budget/simulation.py`)
    is already known to under-count relative to `PauliLCU.num_ancilla` whenever
    the identity coefficient is nonzero (see `TestQubitisedEncoding.test_num_ancilla`
    in `test_cudaq.py`); sidestepping it here matches the pattern
    `qubitised_kernel`/`qubitised_controlled_kernel` (in `quiche.cudaq.simulation`)
    already established.

    Mechanism: `cudaq.control()` cannot wrap the qubitisation walk step at all
    (confirmed during `textbook_qpe_kernel`'s own research - it silently
    produces wrong results on kernels that call other kernels internally,
    which the walk step does), so this uses a different technique entirely -
    no `cudaq.control()` anywhere. Instead, one combined `[control,
    lcu_ancillas...]` register is allocated once outside the ladder, and each
    rung "CNOT-borrows" its control value in from the QPE ancilla register
    (`x.ctrl(ancilla[k], combined[0])`, undone by the same gate after, safe
    since `combined[0]` is never a gate target anywhere else - confirmed by
    tracing every gate in `PauliLCU`'s controlled-select/controlled-reflection
    kernels) around `power` raw calls to `encoding.controlled_walk_step_kernel()`
    (the library's own hand-rolled controlled primitive, not `cudaq.control()`).
    `encoding.prepare_kernel()` is called exactly ONCE, before the whole
    ladder, and `unprepare_kernel()` is never called at all - confirmed
    necessary, not just simpler: each `controlled_walk_step_kernel()` call
    already contains its own internal uncontrolled-unprepare -> controlled-
    reflection -> uncontrolled-prepare cycle (the standard "reflect about
    PREPARE" construction), so it's self-sufficient once primed; wrapping each
    *rung's* `power` steps in its own outer prepare/unprepare pair (mirroring
    `Walk.controlled_kernel`'s single-block structure) does NOT compose across
    multiple rungs, since `unprepare` only reliably returns the LCU ancilla to
    `|0>` when reversing an untouched `prepare(|0>)` - after real controlled
    steps it's left spread/entangled, corrupting the next rung's `prepare`.
    Not uncomputing at the end is fine because only `ancilla` gets measured,
    mirroring `textbook_qpe_kernel` never uncomputing `data`.

    Decode: the walk `W` block-encodes `-H/alpha` (confirmed:
    `qubitised_walk`'s own docstring, and the vendored library's own
    `test_single_term_walk_keeps_minus_h_over_alpha_sign`), so `W`'s
    eigenvalues are `e^(+-i*theta)` with `cos(theta) = -eigenvalue/alpha`.
    QPE measures `phase = y / 2**num_qpe_ancillas` directly - no wrap to
    `[-0.5, 0.5)` needed here (unlike `textbook_qpe_kernel`'s *linear* phase
    decode): `cos` is already even and periodic over the full `[0, 1)` range,
    so the standard +-theta QPE ambiguity is harmless. No separate
    `identity_coefficient` correction either (unlike `textbook_qpe_kernel`'s
    per-rung `r1`): `PauliLCU.alpha` already folds it in (confirmed via
    `test_alpha_includes_identity`), so there's no dropped global phase to
    restore. `cos()`/`math.cos()` do not work inside a `@cudaq.kernel` body
    (confirmed via the AST bridge's call dispatch - no handler for bare or
    `math.`-prefixed calls); `np.cos(...)` does (dispatches to a real MLIR
    math op via the numpy-specific call path), hence the `numpy` import here.

    Both the ladder construction and the decode formula (specifically "no
    separate identity correction needed") were validated end to end against
    real `cudaq` execution and known eigenvalues before being written here -
    `H = Z`'s `|0>` eigenstate (eigenvalue +1, alpha=1) recovers exactly 1.0;
    `H = Z + I`'s `|1>` eigenstate (eigenvalue -1+1=0, alpha=2) recovers
    exactly 0.0, confirming no double-count; a two-qubit `H = ZI - IZ`'s
    `|11>` eigenstate (eigenvalue 0, alpha=2) also recovers exactly 0.0.
    """
    cudaq, _ = load_cudaq()

    encoding = qubitised_encoding(hamiltonian, n_qubits=n_qubits)
    n_anc = encoding.num_ancilla
    alpha = encoding.alpha
    # See textbook_qpe_kernel's identical comment: no float() cast in-kernel.
    dimension = float(1 << num_qpe_ancillas)

    prep = encoding.prepare_kernel()
    controlled_step = encoding.controlled_walk_step_kernel()
    inverse_qft = inverse_qft_kernel()

    @cudaq.kernel
    def qpe(data: cudaq.qview) -> float:
        """Ladder controlled-W^(2^k), inverse QFT, then decode the energy in-kernel."""
        ancilla = cudaq.qvector(num_qpe_ancillas)
        h(ancilla)
        combined = cudaq.qvector(1 + n_anc)
        prep(combined.back(n_anc))
        for k in range(num_qpe_ancillas):
            power = 1 << k
            x.ctrl(ancilla[k], combined[0])
            for _ in range(power):
                controlled_step(combined, data)
            x.ctrl(ancilla[k], combined[0])
        inverse_qft(ancilla)

        y = 0.0
        for k in range(num_qpe_ancillas):
            if mz(ancilla[k]):
                y += 1 << k
        phase = y / dimension
        return -alpha * np.cos(2.0 * pi * phase)

    return qpe


def naive_qpe_kernel(
    hamiltonian: PauliSum,
    simulation: Simulation,
    time: float,
    reps: int,
    order: int = 2,
    *,
    mode: str = "re",
    seed: Seed = None,
    n_qubits: int | None = None,
) -> CudaqKernel:
    """
    Build the Naive (Hadamard-test) QPE kernel for `Simulation.Trotter`/`QDRIFT`.

    Single ancilla, single round: `h(ancilla)`, `[sdg(ancilla)` if `mode="im"]`,
    controlled-U (`base_steps`/`base_order`-parametrized, no ladder - just one
    application), the same per-rung identity correction as `textbook_qpe_kernel`
    at `power=1` (no ladder means no `power` multiplier), `h(ancilla)`, `mz`.

    Returns `(data: cudaq.qview) -> float`, but the float means something
    genuinely different here than in `textbook_qpe_kernel`/`qubitised_qpe_kernel`:
    it's the raw single-shot ancilla outcome (`+1.0` if measured `|0>`, `-1.0`
    if `|1>` - the `Z` eigenvalue), **not** a decoded energy. A single
    Hadamard-test shot carries one bit, not a phase - average many shots
    (`statistics.mean(cudaq.run(kernel, data, shots_count=n))`) to estimate
    `Re`/`Im(<psi|U|psi>)`. This is not an inconsistency with the other
    kernels in this file; it's what a Hadamard test actually measures - there
    is no single-shot "energy" to decode here at all.

    `mode` is resolved with an ordinary host-level `if`, before the kernel is
    built - choosing between two small, separately-written kernel bodies,
    not any in-kernel string comparison (unverified and unnecessary, since
    `mode` is fully known before kernel construction). `S†` is `sdg(ancilla)`
    - a dedicated kernel-mode intrinsic, confirmed distinct from (but
    equivalent to) `s.adj(ancilla)`. The ancilla is a bare `cudaq.qubit()`
    (idiomatic for a single qubit in CUDA-Q's own examples), not a size-1
    `cudaq.qvector`.

    `Simulation.Qubitised` raises `NotImplementedError`: same reasoning as
    `textbook_qpe_kernel`/`qubitised_qpe_kernel` - no `(time, reps, order,
    seed)` shape to share - use `qubitised_naive_qpe_kernel` directly.
    """
    cudaq, _ = load_cudaq()

    match simulation:
        case Simulation.Trotter:
            evolution = trotter_evolution(hamiltonian, n_qubits=n_qubits)
            base_steps, base_order = reps, order
        case Simulation.QDRIFT:
            evolution = qdrift_evolution(
                hamiltonian, reps, seed=seed, n_qubits=n_qubits
            )
            base_steps, base_order = 1, 1
        case Simulation.Qubitised:
            msg = (
                "Simulation.Qubitised has no (time, reps, order, seed) shape "
                "to share with Trotter/QDRIFT - use "
                "quiche.cudaq.estimation.qubitised_naive_qpe_kernel directly."
            )
            raise NotImplementedError(msg)

    coefficients = evolution.coefficients
    words = [cudaq.pauli_word(word) for word in evolution.words]
    identity_coefficient = evolution.identity_coefficient

    from cudaq_algorithms.trotter import apply_trotter  # noqa: PLC0415

    if mode == "im":

        @cudaq.kernel
        def qpe_im(data: cudaq.qview) -> float:
            """Hadamard test (imaginary part): H, S dagger, controlled-U, H, measure."""
            ancilla = cudaq.qubit()
            h(ancilla)
            sdg(ancilla)
            cudaq.control(
                apply_trotter,
                ancilla,
                coefficients,
                words,
                time,
                base_steps,
                base_order,
                data,
            )
            r1(-identity_coefficient * time, ancilla)
            h(ancilla)
            if mz(ancilla):
                return -1.0
            return 1.0

        return qpe_im

    @cudaq.kernel
    def qpe_re(data: cudaq.qview) -> float:
        """Hadamard test (real part): H, controlled-U, H, measure."""
        ancilla = cudaq.qubit()
        h(ancilla)
        cudaq.control(
            apply_trotter,
            ancilla,
            coefficients,
            words,
            time,
            base_steps,
            base_order,
            data,
        )
        r1(-identity_coefficient * time, ancilla)
        h(ancilla)
        if mz(ancilla):
            return -1.0
        return 1.0

    return qpe_re


def qubitised_naive_qpe_kernel(
    hamiltonian: PauliSum,
    *,
    mode: str = "re",
    n_qubits: int | None = None,
) -> CudaqKernel:
    """
    Build the Naive (Hadamard-test) QPE kernel for `Simulation.Qubitised`.

    Same `+1.0`/`-1.0` single-shot contract as `naive_qpe_kernel` (see its
    docstring - this is not a decoded energy). Reuses `qubitised_qpe_kernel`'s
    exact CNOT-borrow mechanism at `power=1` (a single application, no ladder,
    no `k` loop): `prepare_kernel()` once, CNOT-borrow the ancilla into
    `combined[0]`, one `controlled_walk_step_kernel()` call, CNOT-borrow back
    out. No identity correction (unchanged from `qubitised_qpe_kernel` -
    `PauliLCU.alpha` already folds it in).

    Note: this measures `Re`/`Im(<psi|W|psi>)` - the *walk* operator `W`, not
    `H/alpha` directly. `Walk.moment()`/`Walk.moments()` (in
    `cudaq_algorithms.qubitization`) already give more direct access to
    Chebyshev-moment expectation values without a Hadamard-test ancilla at
    all - this function exists for interface consistency with
    `naive_qpe_kernel` across all three `Simulation` variants, not because
    it's the most efficient way to extract this information for Qubitised
    specifically.
    """
    cudaq, _ = load_cudaq()

    encoding = qubitised_encoding(hamiltonian, n_qubits=n_qubits)
    n_anc = encoding.num_ancilla
    prep = encoding.prepare_kernel()
    controlled_step = encoding.controlled_walk_step_kernel()

    if mode == "im":

        @cudaq.kernel
        def qpe_im(data: cudaq.qview) -> float:
            """Hadamard test (imaginary part) on the qubitisation walk operator."""
            ancilla = cudaq.qubit()
            h(ancilla)
            sdg(ancilla)
            combined = cudaq.qvector(1 + n_anc)
            prep(combined.back(n_anc))
            x.ctrl(ancilla, combined[0])
            controlled_step(combined, data)
            x.ctrl(ancilla, combined[0])
            h(ancilla)
            if mz(ancilla):
                return -1.0
            return 1.0

        return qpe_im

    @cudaq.kernel
    def qpe_re(data: cudaq.qview) -> float:
        """Hadamard test (real part) on the qubitisation walk operator."""
        ancilla = cudaq.qubit()
        h(ancilla)
        combined = cudaq.qvector(1 + n_anc)
        prep(combined.back(n_anc))
        x.ctrl(ancilla, combined[0])
        controlled_step(combined, data)
        x.ctrl(ancilla, combined[0])
        h(ancilla)
        if mz(ancilla):
            return -1.0
        return 1.0

    return qpe_re


def iterative_qpe_kernel(
    hamiltonian: PauliSum,
    simulation: Simulation,
    num_rounds: int,
    time: float,
    reps: int,
    order: int = 2,
    *,
    seed: Seed = None,
    n_qubits: int | None = None,
) -> CudaqKernel:
    """
    Build the Iterative QPE kernel for `Simulation.Trotter`/`Simulation.QDRIFT`.

    Single ancilla, classical Rz feedback - IPEA, Dobsicek et al. 2007. Same
    evolution setup as `textbook_qpe_kernel`. Returns
    `(data: cudaq.qview) -> float`, decoding a genuine one-shot energy
    estimate in-kernel (same contract as `textbook_qpe_kernel`, unlike
    `naive_qpe_kernel` - IPEA extracts the *whole* phase within one
    continuous shot, no averaging needed).

    Reuses ONE physical ancilla qubit across `num_rounds` sequential rounds
    via `reset(...)`, rather than `num_qpe_ancillas` separate ones - the
    resource-saving point of this algorithm relative to `textbook_qpe_kernel`
    is qubit count, not gate count (both cost `2**num_rounds - 1` total
    controlled-block applications).

    Round order and feedback formula match QuEST's C++ `getPhaseIterativeInner`
    (`cpp/src/qpe.cpp`) - most-significant exponent round first (`index` from
    `num_rounds - 1` down to `0`), running `phase` accumulating
    `1 / 2**(index+1)` per round whose measured bit is `1` - but using a
    **genuine** per-shot `mz()` decision, unlike QuEST's deterministic
    `calcProbOfQubitOutcome(...) >= 0.5` threshold (a simulator-only proxy
    for what a real stochastic outcome would give). The identity correction
    (same derivation as `textbook_qpe_kernel`'s per-rung `r1`, at this round's
    own `power`) and the IPEA feedback correction are both diagonal `r1`
    phase shifts applied to `ancilla` at the same point in the circuit, so
    they combine additively into one `r1` call (`r1(a)*r1(b) = r1(a+b)`).

    The round loop deliberately does **not** use a descending `range(num_rounds
    - 1, -1, -1)`: confirmed empirically (against a known eigenvalue) that a
    negative-step `range()` combined with this reset/accumulator/nested-loop
    pattern produces genuinely non-deterministic, wrong results, even for an
    exact eigenstate - a real, non-obvious CUDA-Q compiler limitation, not a
    logic bug in the algorithm (isolated by testing a fully-unrolled version
    against the looped version, and an ascending loop with a host-computed
    descending index against the raw descending range - only the descending
    range failed). The loop below counts up (`round_number` from `0` to
    `num_rounds - 1`) and computes `index = num_rounds - 1 - round_number`
    itself.

    `Simulation.Qubitised` raises `NotImplementedError`: combining this
    reset/feedback construction with the already-complex CNOT-borrow
    controlled mechanism needs its own separate validation, deferred.
    """
    cudaq, _ = load_cudaq()

    match simulation:
        case Simulation.Trotter:
            evolution = trotter_evolution(hamiltonian, n_qubits=n_qubits)
            base_steps, base_order = reps, order
        case Simulation.QDRIFT:
            evolution = qdrift_evolution(
                hamiltonian, reps, seed=seed, n_qubits=n_qubits
            )
            base_steps, base_order = 1, 1
        case Simulation.Qubitised:
            msg = (
                "Simulation.Qubitised is not yet implemented for Iterative "
                "QPE in the CUDA-Q backend - it needs its own validation of "
                "the reset/feedback construction combined with the "
                "CNOT-borrow controlled mechanism."
            )
            raise NotImplementedError(msg)

    coefficients = evolution.coefficients
    words = [cudaq.pauli_word(word) for word in evolution.words]
    identity_coefficient = evolution.identity_coefficient

    from cudaq_algorithms.trotter import apply_trotter  # noqa: PLC0415

    @cudaq.kernel
    def qpe(data: cudaq.qview) -> float:
        """Single ancilla, num_rounds sequential rounds with classical Rz feedback."""
        ancilla = cudaq.qubit()
        phase = 0.0
        for round_number in range(num_rounds):
            index = num_rounds - 1 - round_number
            reset(ancilla)
            h(ancilla)
            power = 1 << index
            for _ in range(power):
                cudaq.control(
                    apply_trotter,
                    ancilla,
                    coefficients,
                    words,
                    time,
                    base_steps,
                    base_order,
                    data,
                )
            r1(
                -identity_coefficient * time * power - 2.0 * pi * phase * power,
                ancilla,
            )
            h(ancilla)
            if mz(ancilla):
                phase += 1.0 / (1 << (index + 1))
        if phase >= 0.5:
            phase -= 1.0
        return -phase * (2.0 * pi / time)

    return qpe
