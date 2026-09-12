# Changelog

Changelog format based on [Keep a Changelog](https://keepachangelog.com/).
Versioning based on [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Added `py.typed` marker for type checkers.
- Added `qpe::getEnergyFromTrotterPhase` and `qpe::getEnergyFromQubitisationPhase` functions for converting a QPE phase to an energy.
- Added `logical_rotations_to_tgates` to `quiche.resources` exports.
- Added `PauliSum.has_identity`, `PauliSum.n_terms_with_identity`, `PauliSum.without_identity()` and `PauliSum.split_identity()`.

### Changed
- Renamed `applyMultiStateControlledPhaseShift` to `applyMultiQubitStatePhaseShift` and `applyMultiStateControlledQubitPhaseFlip` to `applyMultiQubitStatePhaseFlip`.
- Changed `quiche` submodule imports to lazy imports.
- Deferred `cirq` import in `quiche.core` to `PauliWord.to_cirq()`.
- Enabled bytecode compilation for `uv` installs.
- Raised minimum build requirements to CMake 3.26, scikit-build-core 1.0, and nanobind 2.10.
- Pinned fetched C++ dependencies (QuEST, Catch2, nanobind) using checksummed release archives where applicable.
- Exposed namespaced C++ library target `quiche::quiche`.
- Moved `pyproject.toml` to the repository root.
- Improved Windows build compatibility.
- `.controlled()` on `PauliWordRotation`, `QDRIFT` and `Trotterisation` now constructs the same bloq but with `is_controlled=True` rather than a separate class.
- `PauliWordRotation` now validates its target qubit range on construction rather than on decomposition.
- Tightened `PauliWord` validation to reject repeated target qubits.
- Tightened `PauliSum` validation to require at least one term.
- Added electron and qubit count validation to `getHartreeFockStateJW`, `getHartreeFockStateBK` and `getHartreeFockStateParity`.
- Made `cloneWithoutIdentity` reject `PauliStrSum`s containing only identity terms.
- Renamed the `QuESTEnv` binding methods `syncQuESTEnv` and `isQuESTEnvInit` to `sync` and `isInit`.
- Added missing `const` qualifiers to the control arguments of `applyMultiControlledCoeffsPrep` and `applyMultiControlledPauliStrSumPrep`.
- `PauliSum.identity_coefficient` now defaults to `0.0`.
- `PauliSum.lam` is now the 1-norm of the whole operator (including the identity coefficient).
- `PauliSum.to_quest()` now includes the identity term rather than dropping it. Call `PauliSum.without_identity()` first to recover the previous behaviour.
- Changed `QDRIFT` sampling to use its own `numpy` generator, requiring an explicit integer `seed` (sampled terms will differ for a given seed from previous sampling).
- `QPESpec` now takes `seed` as a field rather than an `extras` entry (required when simulating with `QDRIFT`).
- `PrepareFromStatePrep` now exposes the phase gradient as a junk register, adding a `phase_gradient` register to the block encoding signature.
- `QDRIFT` and `Trotterisation` now reject a zero evolution time.

### Deprecated

### Removed
- Removed the C++ Hamlib module (keeping just the Python one), along with the corresponding example and unit tests.
- Removed the `QUICHE_BUILD_HAMLIB` build flag and the HDF5 dependency for the C++ backend.
- Removed `CTRLPauliWordRotation`, `CTRLQDRIFT` and `CTRLTrotterisation` (use `.controlled()` on the corresponding bloq instead).
- Removed the (empty) `quiche.dispatch.budget.state_prep` module.
- Removed the dangling `StatePrep` entry from `quiche.core.__all__`.
- Removed `QDRIFT.sample_term_indices()` (use the `sampled_indices` property instead).

### Fixed
- Removed zero-count sub-bloqs from call graphs.
- Fixed `QDRIFT` incorrectly ignoring a `0` seed.
- Fixed Python stable-ABI wheel builds (added missing `Development.SABIModule` component).
- Fixed `get_bk_state` raising `IndexError` for single-orbital systems.
- Fixed `get_jw_state` returning its input rather than an integer array as its signature declares.
- Fixed incorrectly bound `QuESTEnv.sync` and `QuESTEnv.isInit`.
- Rebound `PauliStrSum.fromFile`, `PauliStrSum.fromReversedFile` and the `Qureg` creation methods (`createDensityQureg`, `createForcedQureg`, `createForcedDensityQureg` and `createCustomQureg`) as static methods.
- Fixed `QDRIFT` dropping the evolution time from the identity term's global phase.
- Fixed `QDRIFT` sampling from the global `random` state.
- Fixed `LCUBlockEncodingWrapper` discarding the signs of negative coefficients.
- Fixed `LCUBlockEncodingWrapper` including the phase gradient qubits in the index register for SELECT.

### Security

## [0.0.1] - 2026-06-30

### Added

- Initial public release
