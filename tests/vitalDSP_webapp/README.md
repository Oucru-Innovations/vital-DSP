# vitalDSP Webapp Test Suite

This directory contains comprehensive tests for the vitalDSP webapp, mirroring the source code structure from `src/vitalDSP_webapp/`.

## Test Structure

The test structure follows the exact same organization as the source code:

```
tests/vitalDSP_webapp/
├── api/                           # API endpoint tests
│   ├── __init__.py
│   └── test_endpoints.py
├── callbacks/                     # Callback function tests
│   ├── __init__.py
│   ├── analysis/                  # Analysis callback tests
│   │   ├── __init__.py
│   │   ├── test_advanced_callbacks.py
│   │   └── test_frequency_filtering_callbacks.py
│   ├── core/                      # Core callback tests
│   │   ├── __init__.py
│   │   ├── test_app_callbacks.py
│   │   └── test_page_routing_callbacks.py
│   ├── features/                  # Feature extraction callback tests
│   │   ├── __init__.py
│   │   ├── test_features_callbacks.py
│   │   └── test_physiological_callbacks.py
│   ├── legacy/                    # Legacy callback tests
│   │   └── __init__.py
│   ├── utils/                     # Utility callback tests
│   │   └── __init__.py
│   └── visualization/             # Visualization callback tests
│       ├── __init__.py
│       └── test_visualization_callbacks.py
├── config/                        # Configuration tests
│   ├── __init__.py
│   ├── test_logging_config.py
│   └── test_settings.py
├── layout/                        # Layout component tests
│   ├── __init__.py
│   ├── common/                    # Common layout component tests
│   │   ├── __init__.py
│   │   ├── test_footer.py
│   │   ├── test_header.py
│   │   └── test_sidebar.py
│   └── pages/                     # Page layout tests
│       ├── __init__.py
│       ├── test_analysis_pages.py
│       └── test_upload_page.py
├── models/                        # Data model tests
│   ├── __init__.py
│   └── test_signal_processing.py
├── services/                      # Service layer tests
│   ├── __init__.py
│   ├── test_settings_service.py
│   └── data/                      # Data service tests
│       ├── __init__.py
│       └── test_data_service.py
├── utils/                         # Utility function tests
│   ├── __init__.py
│   ├── test_data_processor.py
│   ├── test_error_handler.py
│   └── test_file_utils.py
├── __init__.py                    # Main test package
├── conftest.py                    # Test configuration and fixtures
├── requirements.txt               # Test dependencies
├── test_app.py                    # Main application tests
├── test_run_webapp.py             # Webapp runner tests
└── README.md                      # This file
```

## Test Categories

### 1. Unit Tests
- **API Tests**: Test individual API endpoints
- **Callback Tests**: Test Dash callback functions
- **Service Tests**: Test business logic services
- **Utility Tests**: Test helper functions

### 2. Integration Tests
- **Layout Tests**: Test component integration
- **Model Tests**: Test data flow between components
- **Configuration Tests**: Test system configuration

### 3. Mock Tests
- **Signal Processing**: Test with synthetic physiological data
- **File Operations**: Test file upload/download functionality
- **Error Handling**: Test error scenarios and recovery

## Running the Tests

### Prerequisites
Install the test dependencies:
```bash
pip install -r tests/vitalDSP_webapp/requirements.txt
```

### Basic Test Execution
```bash
# Run all webapp tests
pytest tests/vitalDSP_webapp/

# Run specific test category
pytest tests/vitalDSP_webapp/callbacks/
pytest tests/vitalDSP_webapp/services/

# Run specific test file
pytest tests/vitalDSP_webapp/test_app.py
```

### Advanced Test Execution
```bash
# Run with coverage report
pytest tests/vitalDSP_webapp/ --cov=src/vitalDSP_webapp --cov-report=html

# Run with verbose output
pytest tests/vitalDSP_webapp/ -v

# Run tests in parallel
pytest tests/vitalDSP_webapp/ -n auto

# Run tests with timeout
pytest tests/vitalDSP_webapp/ --timeout=30
```

### Test Discovery
```bash
# List all available tests
pytest tests/vitalDSP_webapp/ --collect-only

# List tests in a specific file
pytest tests/vitalDSP_webapp/test_app.py --collect-only
```

## Test Fixtures

The `conftest.py` file provides several useful fixtures:

- **`sample_ppg_data`**: Synthetic PPG signal data
- **`sample_ecg_data`**: Synthetic ECG signal data  
- **`sample_respiratory_data`**: Synthetic respiratory data
- **`mock_dash_app`**: Mock Dash application
- **`temp_upload_dir`**: Temporary directory for file tests
- **`mock_file_upload`**: Mock file upload object
- **`sample_analysis_parameters`**: Sample analysis parameters
- **`mock_signal_processor`**: Mock signal processing service

## Writing New Tests

### Test File Naming Convention
- Test files should be named `test_*.py`
- Test classes should be named `Test*`
- Test methods should be named `test_*`

### Example Test Structure
```python
"""
Tests for [component name]
"""
import pytest
from unittest.mock import Mock, patch


class TestComponentName:
    """Test class for [component name]"""
    
    def test_functionality_name(self):
        """Test description"""
        # Arrange
        # Act
        # Assert
        pass
```

### Best Practices
1. **Use descriptive test names** that explain what is being tested
2. **Follow AAA pattern**: Arrange, Act, Assert
3. **Use appropriate fixtures** for common test data
4. **Mock external dependencies** to isolate unit tests
5. **Test both success and failure scenarios**
6. **Keep tests focused** on a single piece of functionality

## Continuous Integration

The test suite is designed to work with CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run Tests
  run: |
    pip install -r tests/vitalDSP_webapp/requirements.txt
    pytest tests/vitalDSP_webapp/ --cov=src/vitalDSP_webapp --cov-report=xml
```

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure the source code is in the Python path
2. **Missing Dependencies**: Install all requirements from `requirements.txt`
3. **Test Discovery Issues**: Check that test files follow naming conventions
4. **Mock Issues**: Ensure proper patching of external dependencies

### Debug Mode
```bash
# Run tests with debug output
pytest tests/vitalDSP_webapp/ -s -v --tb=long

# Run single test with debugger
pytest tests/vitalDSP_webapp/test_app.py::TestVitalDSPWebapp::test_app_initialization -s
```

## Contributing

When adding new tests:
1. Follow the existing structure and naming conventions
2. Add appropriate fixtures to `conftest.py` if needed
3. Update this README if adding new test categories
4. Ensure tests pass before submitting changes

## Coverage Goals

- **Line Coverage**: Target >90%
- **Branch Coverage**: Target >85%
- **Function Coverage**: Target >95%

Run coverage analysis to track progress:
```bash
pytest tests/vitalDSP_webapp/ --cov=src/vitalDSP_webapp --cov-report=term-missing
```
