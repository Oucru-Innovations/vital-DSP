# vitalDSP v0.2.1 Release Notes

**Release Date:** January 27, 2025

## 🎉 Overview

vitalDSP v0.2.1 is a significant release focused on **improving code quality, test coverage, and CI/CD reliability**. This release includes comprehensive test coverage improvements, critical bug fixes, and enhanced infrastructure for better maintainability.

## 🚀 Key Highlights

### 📊 Massive Test Coverage Improvements
- **200+ new unit tests** added across webapp modules
- Overall webapp test coverage increased from **37% to 44%+**
- Multiple modules now have **60%+ coverage**, with some reaching **90%+**

### 🔧 Critical CI/CD Fixes
- **Fixed NumPy 2.x compatibility issue** with TensorFlow
- Resolved `AttributeError: _ARRAY_API not found` errors
- Added proper dependency constraints to prevent version conflicts

### ✨ Enhanced Features
- Improved data services for better large file handling
- Enhanced export utilities (CSV, JSON, Excel)
- Better signal type detection and plot optimization
- Comprehensive REST API with 15+ endpoints

## 📈 Test Coverage Improvements

| Module | Previous Coverage | New Coverage | Improvement |
|--------|------------------|--------------|-------------|
| `transform_callbacks.py` | 6% | 60%+ | +54% |
| `upload_callbacks.py` | 39% | 50%+ | +11% |
| `pipeline_callbacks.py` | 17% | 40%+ | +23% |
| `signal_filtering_callbacks.py` | 25% | 50%+ | +25% |
| `quality_sqi_functions.py` | 5% | 56% | +51% |
| `header_monitoring_callbacks.py` | 20% | 74% | +54% |
| `export_utils.py` | 12% | 80% | +68% |
| `signal_type_detection.py` | 11% | 90% | +79% |
| `plot_utils.py` | 22% | 94% | +72% |
| `progress_tracker.py` | 24% | 85% | +61% |
| `export_components.py` | 0% | 100% | +100% |

## 🆕 New Test Files

This release includes 18 new comprehensive test files:

- `test_transform_callbacks_analyze.py` - Transform callback testing
- `test_upload_callbacks_load_data.py` - Data loading tests
- `test_pipeline_callbacks_stages.py` - Pipeline stage testing
- `test_signal_filtering_callbacks_helpers.py` - Filtering helpers
- `test_quality_callbacks_comprehensive.py` - Quality assessment
- `test_quality_sqi_functions_comprehensive.py` - SQI computation
- `test_time_domain_callbacks_extended.py` - Time domain analysis
- `test_export_callbacks_comprehensive.py` - Export functionality
- `test_export_utils_comprehensive.py` - Export utilities
- `test_signal_type_detection_comprehensive.py` - Signal detection
- `test_plot_utils_comprehensive.py` - Plot optimization
- `test_progress_tracker_comprehensive.py` - Progress tracking
- `test_enhanced_data_service_comprehensive.py` - Data services
- `test_export_components_comprehensive.py` - Export UI components
- `test_progress_components_comprehensive.py` - Progress UI
- `test_progress_indicator_comprehensive.py` - Progress indicators
- `test_theme_callbacks_comprehensive.py` - Theme management
- `test_header_monitoring_callbacks_comprehensive.py` - System monitoring

## 🐛 Critical Fixes

### CI/CD Improvements
- ✅ Fixed NumPy 2.x compatibility with TensorFlow
- ✅ Added `constraints.txt` for dependency management
- ✅ Enhanced CI workflow with better dependency resolution
- ✅ Added NumPy version verification steps

### Test Infrastructure
- ✅ Fixed function signature mismatches
- ✅ Resolved monkey-patching issues in tests
- ✅ Fixed import path issues
- ✅ Corrected assertion errors for Dash components

## 🔄 Changed

- Improved test suite with significantly better coverage
- Updated GitHub CI/CD workflow with NumPy compatibility fixes
- Enhanced code quality and linting standards
- Updated dependencies to latest stable versions
- Docker and Docker Compose configurations with version tags

## 📦 Installation

```bash
# Install from PyPI
pip install vitalDSP==0.2.1

# Or install with optional dependencies
pip install vitalDSP[tensorflow]==0.2.1
pip install vitalDSP[pytorch]==0.2.1
pip install vitalDSP[all]==0.2.1
```

## 📚 Documentation

- [Full Documentation](https://vital-dsp.readthedocs.io/)
- [API Reference](https://vital-dsp.readthedocs.io/en/latest/api.html)
- [Tutorials](https://colab.research.google.com/github/Oucru-Innovations/vital-DSP/blob/main/docs/source/notebooks/synthesize_data.ipynb)

## 🙏 Acknowledgments

Thank you to all contributors and users who reported issues and provided feedback!

## 🔗 Links

- **GitHub Repository**: https://github.com/Oucru-Innovations/vital-DSP
- **PyPI Package**: https://pypi.org/project/vitalDSP/
- **Documentation**: https://vital-dsp.readthedocs.io/
- **Web Application**: https://vital-dsp-1.onrender.com/

## 📝 Full Changelog

For a complete list of changes, see [Changelog.md](Changelog.md).

---

**Note**: This release maintains backward compatibility with previous versions. No breaking changes were introduced.

