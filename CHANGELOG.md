# Changelog

Changelog format based on [Keep a Changelog](https://keepachangelog.com/).
Versioning based on [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

### Changed
- Renamed `applyMultiStateControlledPhaseShift` to `applyMultiQubitStatePhaseShift` and `applyMultiStateControlledQubitPhaseFlip` to `applyMultiQubitStatePhaseFlip`.
- Changed `quiche` submodule imports to lazy imports.
- Deferred `cirq` import in `quiche.core` to `PauliWord.to_cirq()`.
- Enabled bytecode compilation for `uv` installs.

### Deprecated

### Removed

### Fixed

### Security

## [0.0.1] - 2026-06-30

### Added

- Initial public release
