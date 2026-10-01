# Changelog

Changelog format based on [Keep a Changelog](https://keepachangelog.com/).
Versioning based on [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `quiche.simulation` (`Trotter`, `QDRIFT`, `Qubitised`) and `quiche.estimation` (`Textbook`, `Kitaev`, `Iterative`, `Naive`) method objects. Each one takes either its parameters or a target error, works out the other when it's created, and reports the `error` its final parameters achieve. Estimation objects hold their simulation and the initial-state `overlap`, report `total_error`, and build the QPE circuit with `to_qualtran`/`to_quest`/`to_cudaq`.
- `quiche.state_prep.HartreeFock`, with an optional `qubits` selection of the mapped state.
- `PauliSum.unused_qubits` and `PauliSum.compact()` for dropping qubits that no term acts on.
- Inverse error-budget formulas in `quiche.budget` (e.g. `get_trotter_error`, `get_textbook_qpe_error`).

### Changed
- Renamed `applyMultiStateControlledPhaseShift` to `applyMultiQubitStatePhaseShift` and `applyMultiStateControlledQubitPhaseFlip` to `applyMultiQubitStatePhaseFlip`.
- The CUDA-Q builders `textbook_qpe_kernel`, `naive_qpe_kernel`, `iterative_qpe_kernel` and `simulation_kernel` take a simulation object instead of a `Simulation` enum plus `hamiltonian`/`time`/`reps`/`order`/`seed`.
- Moved `quiche.dispatch.budget` to `quiche.budget`; its functions take error values instead of an `Errors` budget.
- `logical_rotations_to_tgates` takes a `rotation_error` value instead of an `Errors` budget.

### Deprecated

### Removed
- `QPESpec`, `HartreeFockSpec`, `Errors` and the `PhaseEstimation`/`Simulation` enums, replaced by the method objects above.

### Fixed

### Security

## [0.0.1] - 2026-06-30

### Added

- Initial public release
