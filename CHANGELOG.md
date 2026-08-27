# Changelog

Changelog format based on [Keep a Changelog](https://keepachangelog.com/).
Versioning based on [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Added `qpe::getEnergyFromTrotterPhase` and `qpe::getEnergyFromQubitisationPhase` functions for converting a QPE phase to an energy.

### Changed
- Renamed `applyMultiStateControlledPhaseShift` to `applyMultiQubitStatePhaseShift` and `applyMultiStateControlledQubitPhaseFlip` to `applyMultiQubitStatePhaseFlip`.
- Changed `quiche` submodule imports to lazy imports.
- Deferred `cirq` import in `quiche.core` to `PauliWord.to_cirq()`.
- Enabled bytecode compilation for `uv` installs.

### Deprecated

### Removed
- Removed the C++ Hamlib module (keeping just the Python one), along with the corresponding example and unit tests.
- Removed the `QUICHE_BUILD_HAMLIB` build flag and the HDF5 dependency for the C++ backend.

### Fixed
- Removed zero-count sub-bloqs from call graphs.
- Fixed `QDRIFT` incorrectly ignoring a `0` seed.

### Security

## [0.0.1] - 2026-06-30

### Added

- Initial public release
