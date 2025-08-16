# vitalDSP Webapp - Refactored Architecture

## Overview

The vitalDSP webapp has been completely refactored to improve maintainability, scalability, and code organization. This document outlines the new architecture and key improvements.

## New Architecture

### 1. Configuration Management (`config/`)

- **`settings.py`**: Centralized configuration using dataclasses
  - App metadata, server settings, file upload limits
  - UI styling constants and theme management
  - Signal quality thresholds and column mapping patterns
  - Environment-specific configurations

- **`logging_config.py`**: Comprehensive logging setup
  - Rotating file logs with configurable sizes
  - Environment-specific log levels (dev/prod/test)
  - Performance and data operation logging
  - Error context logging

### 2. Data Services (`services/`)

- **`data_service.py`**: Centralized data management
  - Data storage with unique IDs and metadata
  - Column mapping storage and retrieval
  - Analysis results management
  - Data export functionality
  - Session management and cleanup

### 3. Utility Modules (`utils/`)

- **`data_processor.py`**: Data processing utilities
  - File validation and reading
  - Column auto-detection
  - Signal quality assessment
  - Sample data generation
  - Column mapping validation

- **`error_handler.py`**: Comprehensive error handling
  - Custom exception classes
  - Standardized error alerts
  - Safe function execution
  - Data validation utilities

### 4. Refactored Callbacks (`callbacks/`)

- **`upload_callbacks.py`**: Streamlined upload handling
  - Uses data service for storage
  - Leverages data processor utilities
  - Improved error handling and logging
  - Configuration-driven behavior

## Key Improvements

### 1. **Separation of Concerns**
- Data processing logic separated from UI callbacks
- Configuration centralized and easily modifiable
- Error handling standardized across components

### 2. **Data Management**
- Proper data storage with unique identifiers
- Column mapping persistence
- Analysis results tracking
- Session-based data isolation

### 3. **Error Handling**
- Custom exception classes for different error types
- Consistent error messages and user feedback
- Comprehensive logging with context
- Safe execution patterns

### 4. **Configuration Management**
- Environment-specific settings
- Centralized constants and thresholds
- Easy customization without code changes
- Type-safe configuration using dataclasses

### 5. **Logging and Monitoring**
- Rotating log files with size limits
- Performance metrics logging
- Data operation tracking
- Error context preservation

## Usage Examples

### Running the Application

```python
from vitalDSP_webapp.app import run_app

# Run with default settings
run_app()

# Run with custom settings
run_app(debug=True, host="127.0.0.1", port=8080)
```

### Configuration Management

```python
from vitalDSP_webapp.config.settings import app_config, update_config

# Access configuration
print(f"App name: {app_config.APP_NAME}")
print(f"Max file size: {app_config.MAX_FILE_SIZE}")

# Update configuration
update_config(DEBUG=True, PORT=9000)
```

### Data Service Usage

```python
from vitalDSP_webapp.services.data_service import get_data_service

data_service = get_data_service()

# Store data
data_id = data_service.store_data(df, metadata)

# Retrieve data
df = data_service.get_data(data_id)

# Store column mapping
data_service.store_column_mapping(data_id, column_mapping)
```

### Error Handling

```python
from vitalDSP_webapp.utils.error_handler import safe_execute, create_error_alert

# Safe function execution
result = safe_execute(
    risky_function, 
    arg1, arg2, 
    default_return="fallback",
    error_context="Data processing"
)

# Create error alerts
error_alert = create_error_alert(
    error, 
    title="Processing Error",
    show_details=True
)
```

## Migration Guide

### For Existing Code

1. **Update imports** to use new module structure
2. **Replace direct data handling** with data service calls
3. **Use configuration constants** instead of hard-coded values
4. **Implement proper error handling** using new utilities
5. **Add logging** for better debugging and monitoring

### Before (Old Code)

```python
# Hard-coded values
MAX_FILE_SIZE = 100 * 1024 * 1024
SIDEBAR_WIDTH = 280

# Direct data handling
data_store = {"dataframe": df.to_dict("records")}

# Basic error handling
except Exception as e:
    return f"Error: {str(e)}"
```

### After (New Code)

```python
# Configuration-driven
from vitalDSP_webapp.config.settings import app_config
max_file_size = app_config.MAX_FILE_SIZE
sidebar_width = app_config.SIDEBAR_WIDTH

# Service-based data handling
from vitalDSP_webapp.services.data_service import get_data_service
data_service = get_data_service()
data_id = data_service.store_data(df, metadata)

# Comprehensive error handling
from vitalDSP_webapp.utils.error_handler import create_error_alert
except Exception as e:
    logger.error(f"Error in data processing: {str(e)}")
    return create_error_alert(e, "Data Processing Error")
```

## Development Guidelines

### 1. **Configuration First**
- Always use configuration constants instead of hard-coded values
- Add new settings to the appropriate configuration class
- Use environment variables for sensitive settings

### 2. **Service Layer**
- Business logic should go in service modules
- Callbacks should only handle UI state and user interactions
- Use data service for all data operations

### 3. **Error Handling**
- Use custom exception classes for different error types
- Always log errors with context
- Provide user-friendly error messages

### 4. **Logging**
- Use appropriate log levels (DEBUG, INFO, WARNING, ERROR)
- Include context in log messages
- Log performance metrics for slow operations

### 5. **Testing**
- Test service functions independently
- Mock external dependencies
- Test error conditions and edge cases

## Performance Considerations

### 1. **Data Management**
- Large datasets are stored efficiently with metadata
- Column mappings are cached for quick access
- Analysis results are stored separately to avoid duplication

### 2. **Memory Usage**
- Data is stored as records to minimize memory overhead
- Log files are rotated to prevent disk space issues
- Session cleanup prevents memory leaks

### 3. **Scalability**
- Service-based architecture allows for easy scaling
- Configuration can be updated without code changes
- Logging provides insights for performance optimization

## Future Enhancements

### 1. **Database Integration**
- Replace in-memory storage with persistent database
- Add data versioning and history tracking
- Implement user authentication and data isolation

### 2. **API Enhancements**
- Add RESTful endpoints for external integrations
- Implement real-time data streaming
- Add webhook support for notifications

### 3. **Advanced Analytics**
- Add machine learning model integration
- Implement batch processing for large datasets
- Add real-time signal quality monitoring

### 4. **User Experience**
- Add progress indicators for long operations
- Implement undo/redo functionality
- Add keyboard shortcuts and accessibility features

## Conclusion

The refactored vitalDSP webapp provides a solid foundation for future development with improved maintainability, better error handling, and comprehensive logging. The new architecture separates concerns effectively and makes the codebase easier to understand and extend.
