# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), 
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Planned features for future releases.

## [0.2.1] - 2025-01-27
### Added
- Enhanced documentation with comprehensive examples and tutorials
- Updated Read the Docs configuration for improved build process
- Added support for PyTorch, SHAP, and LIME optional dependencies
- Comprehensive installation documentation
- **Comprehensive REST API**: 15+ new API endpoints for filtering, feature extraction, respiratory analysis, and more
  - Signal filtering endpoints (Butterworth, adaptive)
  - Feature extraction (time-domain, frequency-domain, HRV)
  - Respiratory rate estimation
  - Signal quality assessment
  - Transform operations (FFT, Wavelet)
  - Health report generation
  - Batch processing support
- Separate coverage targets in Makefile for vitalDSP core and webapp

### Changed
- Improved test suite with better coverage
- Updated GitHub CI/CD workflow
- Enhanced code quality and linting standards
- Updated dependencies to latest stable versions
- Docker and Docker Compose configurations with version tags

### Fixed
- Fixed GitHub Actions CI configuration
- Resolved linting issues across the codebase
- Improved test coverage and reliability

## [Version X.X.X] - YYYY-MM-DD
### Added
- Initial implementation of the `filtering/` module, including moving average, Gaussian, and Butterworth filters.
- Added `xxx.py` for real-time adaptive filtering.
- Introduced `xxx.py` using ML-inspired loss functions.

### Changed
- Updated `README.md` with new usage examples.
- Refactored `wavelet_transform.py` to improve performance.

### Deprecated
- Deprecated `xxx.py` in favor of `xxx.py`.

### Removed
- Removed `xxx.py` as it is no longer supported.

### Fixed
- Fixed a bug in `xxx.py` where it missed peaks at the signal boundaries.

### Security
- Patched a vulnerability in `setup.py` related to dependency versions.

## [Version X.X.X] - YYYY-MM-DD
### Added
- Added `xxx/` module with xxx band power analysis and ERP detection.
- Implemented `xxx.py` for automated signal processing pipeline construction.
- CI/CD integration via GitHub Actions with a `ci.yml` workflow.

### Changed
- Enhanced `xxx.py` to support multi-channel signals.

### Deprecated
- Deprecated `xxx.py` in favor of more advanced transformations.

### Removed
- Removed outdated test files from the `tests/` directory.

### Fixed
- Addressed memory leak in `xxx.py`.

### Security
- Improved handling of sensitive data in `xxx.py`.
