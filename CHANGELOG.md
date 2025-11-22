# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2025-11-23

### Added

- `MathematicalLandmarkDataset` now takes `ptype` argument.

## [1.2.1] - 2025-10-27

### Fixed

- Integer indexing `MathematicalLandmarkDataset` is now correctly done.
- `MathematicalLandmarkDataset` with `m=1` now correctly returns 1-D profiles.
- Indexing a single data now returns non-vector element.

## [1.2.0] - 2025-10-27

### Changed

- `MathematicalLandmarkDataset` now returns point at x=0 as an additional landmark.

## [1.1.0] - 2025-10-15

### Added

- `PseudoLandmarkDataset` class.
- `MathematicalLandmarkDataset` class.

## [1.0.0] - 2025-10-14

### Added

- `ProfileDataset` class.