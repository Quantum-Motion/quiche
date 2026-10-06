# Changelog

Changelog format based on [Keep a Changelog](https://keepachangelog.com/).
Versioning based on [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `quiche.simulation` (`Trotter`, `QDRIFT`, `Qubitised`) and `quiche.estimation` (`Textbook`, `Kitaev`, `Iterative`, `Naive`) method objects. Each one takes either its parameters or a target error, works out the other when it's created, and reports the `error` its final parameters achieve. Estimation objects hold their simulation and the initial-state `overlap`, report `total_error`, and build the QPE circuit with `to_qualtran`/`to_quest`/`to_cudaq`.
- Errors are energies in the units of the Hamiltonian (Hartree for HamLib/FCIDUMP inputs), with rigorous bounds: Childs et al. (2021) for Trotter, Campbell (2019) for QDRIFT, and the Prepare angle precision for qubitisation. `total_error` bounds the energy error of the estimate.
- `success_probability` on the estimation objects: a lower bound on the probability that the minimum over `repetitions` QPE runs lies within `total_error` of the ground-state energy. `overlap` is the squared overlap with the ground state; `extra_ancillas` (default 4) and `repetitions` (default: 95% chance of reaching the ground state) are configurable.
- `to_cudaq(state_prep=...)` builds a `() -> float` kernel that runs the repetitions in-kernel and returns the minimum energy.
- `to_cudaq(kernel="circuit")` and `to_cudaq(kernel="post")` for Textbook QPE. `"circuit"` is the QPE circuit ending in a measurement of the QPE ancillas, with no measurement-dependent logic, so `cudaq.sample` simulates it once for any number of shots and compilers that reject feedback can take it. `"post"` decodes the sampled bitstrings to energies on the host, keeping the minimum over each block of `repetitions` shots. The default `kernel="all"` is unchanged. New CUDA-Q builders: `textbook_qpe_circuit`, `qubitised_qpe_circuit`, `measured_circuit_kernel` and `sample_post_processor`. The CUDA-Q QPE builders gain `*_core` variants that take a reusable work register, plus `single_run_kernel` and `repeated_minimum_kernel`.
- `quiche.state_prep.HartreeFock`, built from an `occupation` and a required `mapping` (or with `HartreeFock.closed_shell`), with an optional `qubits` selection of the mapped state.
- `PauliSum.unused_qubits` and `PauliSum.compact()` for dropping qubits that no term acts on.
- Inverse error-budget formulas in `quiche.budget` (e.g. `get_trotter_reps`, `get_qubitisation_error`).

### Changed
- Renamed `applyMultiStateControlledPhaseShift` to `applyMultiQubitStatePhaseShift` and `applyMultiStateControlledQubitPhaseFlip` to `applyMultiQubitStatePhaseFlip`.
- The CUDA-Q builders `textbook_qpe_kernel`, `naive_qpe_kernel`, `iterative_qpe_kernel` and `simulation_kernel` take a simulation object instead of a `Simulation` enum plus `hamiltonian`/`time`/`reps`/`order`/`seed`.
- Moved `quiche.dispatch.budget` to `quiche.budget`; its functions take error values instead of an `Errors` budget.
- `logical_rotations_to_tgates` takes a `rotation_error` value instead of an `Errors` budget.
- The default evolution time is `pi / (lam + |identity_coefficient|)`, so the measured phase can no longer wrap when a backend (CUDA-Q) folds the identity coefficient into it.
- `QDRIFT` takes `reps` only: its error reduces `success_probability` rather than shifting the energy. `Qubitised` takes `error` (an energy) in place of `prepare_error`. `Trotter` rejects odd orders above 1.

### Deprecated

### Removed
- `QPESpec`, `HartreeFockSpec`, `HartreeFockState`, `Errors` and the `PhaseEstimation`/`Simulation` enums, replaced by the objects above.

### Fixed

### Security

## [0.0.1] - 2026-06-30

### Added

- Initial public release
