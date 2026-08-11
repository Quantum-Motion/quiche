# Changelog

Changelog format based on [Keep a Changelog](https://keepachangelog.com/).
Versioning based on [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Added `py.typed` marker for type checkers.

### Changed
- Renamed `applyMultiStateControlledPhaseShift` to `applyMultiQubitStatePhaseShift` and `applyMultiStateControlledQubitPhaseFlip` to `applyMultiQubitStatePhaseFlip`.
- Raised minimum build requirements to CMake 3.24, scikit-build-core 1.0, and nanobind 2.10.
- Pinned fetched C++ dependencies (QuEST, HDF5, Catch2, nanobind) using checksummed release archives where applicable.
- Exposed namespaced C++ library target `quiche::quiche`.
- Moved `pyproject.toml` to the repository root.
- Improved Windows build compatibility.

### Deprecated

### Removed

### Fixed
- Fixed Python stable-ABI wheel builds (added missing `Development.SABIModule` component).

### Security

## [0.0.1] - 2026-06-30

### Added

- Initial public release
