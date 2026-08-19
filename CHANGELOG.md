# Changelog

Changelog format based on [Keep a Changelog](https://keepachangelog.com/).
Versioning based on [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

### Changed
- Renamed `applyMultiStateControlledPhaseShift` to `applyMultiQubitStatePhaseShift` and `applyMultiStateControlledQubitPhaseFlip` to `applyMultiQubitStatePhaseFlip`.

### Deprecated

### Removed
- Removed the C++ Hamlib module (keeping just the Python one), along with the corresponding example and unit tests.
- Removed the `QUICHE_BUILD_HAMLIB` build flag and the HDF5 dependency for the C++ backend.

### Fixed

### Security

## [0.0.1] - 2026-06-30

### Added

- Initial public release
