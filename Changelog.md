# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), 
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Planned features for future releases.

## [0.2.1] - 2025-01-27

### Added
- **Enhanced documentation** with comprehensive examples and tutorials
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
- **Comprehensive test coverage improvements**:
  - Added 200+ new unit tests across webapp modules
  - Test coverage for `transform_callbacks.py` (6% → 60%+)
  - Test coverage for `upload_callbacks.py` (39% → 50%+)
  - Test coverage for `pipeline_callbacks.py` (17% → 40%+)
  - Test coverage for `signal_filtering_callbacks.py` (25% → 50%+)
  - Test coverage for `quality_sqi_functions.py` (5% → 56%)
  - Test coverage for `header_monitoring_callbacks.py` (20% → 74%)
  - Test coverage for `export_utils.py` (12% → 80%)
  - Test coverage for `signal_type_detection.py` (11% → 90%)
  - Test coverage for `plot_utils.py` (22% → 94%)
  - Test coverage for `progress_tracker.py` (24% → 85%)
  - Test coverage for `enhanced_data_service.py` (25% → 44%)
  - Test coverage for `export_components.py` (0% → 100%)
  - Test coverage for `theme_callbacks.py` (27% → 60%+)
  - Test coverage for `progress_components.py` (31% → 60%+)
  - Test coverage for `progress_indicator.py` (19% → 60%+)
- **New test files**:
  - `test_transform_callbacks_analyze.py` - Comprehensive transform callback testing
  - `test_upload_callbacks_load_data.py` - Data loading and format handling tests
  - `test_pipeline_callbacks_stages.py` - Complete pipeline stage testing
  - `test_signal_filtering_callbacks_helpers.py` - Filtering helper functions tests
  - `test_quality_callbacks_comprehensive.py` - Quality assessment callback tests
  - `test_quality_sqi_functions_comprehensive.py` - SQI computation tests
  - `test_time_domain_callbacks_extended.py` - Time domain analysis tests
  - `test_export_callbacks_comprehensive.py` - Export functionality tests
  - `test_export_utils_comprehensive.py` - Export utility tests
  - `test_signal_type_detection_comprehensive.py` - Signal type detection tests
  - `test_plot_utils_comprehensive.py` - Plot optimization tests
  - `test_progress_tracker_comprehensive.py` - Progress tracking tests
  - `test_enhanced_data_service_comprehensive.py` - Enhanced data service tests
  - `test_export_components_comprehensive.py` - Export UI components tests
  - `test_progress_components_comprehensive.py` - Progress UI components tests
  - `test_progress_indicator_comprehensive.py` - Progress indicator tests
  - `test_theme_callbacks_comprehensive.py` - Theme management tests
  - `test_header_monitoring_callbacks_comprehensive.py` - System monitoring tests
- **Enhanced data services**:
  - Improved `ChunkedDataService` with better error handling
  - Enhanced `MemoryMappedDataService` for large file support
  - Better `ProgressiveDataLoader` implementation
  - Improved `LRUCache` memory management
- **Export improvements**:
  - Enhanced CSV, JSON, and Excel export utilities
  - New export UI components for consistent user experience
  - Better error handling in export functions
- **Signal processing improvements**:
  - Enhanced signal type detection with 90%+ test coverage
  - Improved plot data optimization utilities
  - Better signal quality assessment functions

### Changed
- Improved test suite with significantly better coverage (overall webapp coverage increased from 37% to 44%+)
- Updated GitHub CI/CD workflow with NumPy compatibility fixes
- Enhanced code quality and linting standards
- Updated dependencies to latest stable versions
- Docker and Docker Compose configurations with version tags
- **CI/CD improvements**:
  - Added NumPy version constraints to prevent TensorFlow compatibility issues
  - Created `constraints.txt` file for dependency management
  - Enhanced CI workflow with better dependency resolution
  - Added NumPy version verification steps

### Fixed
- **Critical CI/CD fixes**:
  - Fixed NumPy 2.x compatibility issue with TensorFlow in GitHub Actions
  - Resolved `AttributeError: _ARRAY_API not found` by constraining NumPy to <2.0
  - Added proper dependency constraints to prevent version conflicts
- Fixed GitHub Actions CI configuration
- Resolved linting issues across the codebase
- Improved test coverage and reliability
- Fixed function signature mismatches in test files
- Resolved monkey-patching issues in enhanced filtering callbacks tests
- Fixed import path issues in various test files
- Corrected assertion errors in test files for Dash components

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
