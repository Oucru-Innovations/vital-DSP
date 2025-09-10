# vitalDSP Webapp - Refactored Architecture

This document describes the new modular architecture of the vitalDSP webapp after refactoring.

## Overview

The webapp has been refactored from a monolithic structure to a clean, modular architecture that improves maintainability, readability, and extensibility.

## New Directory Structure

```
src/vitalDSP_webapp/
├── callbacks/                    # Callback functions organized by functionality
│   ├── core/                    # Core application callbacks
│   │   ├── app_callbacks.py     # Basic app functionality (sidebar, etc.)
│   │   ├── page_routing_callbacks.py  # Page navigation
│   │   └── upload_callbacks.py  # File upload handling
│   ├── analysis/                # Analysis-related callbacks
│   │   ├── vitaldsp_callbacks.py      # Core vitalDSP analysis
│   │   ├── frequency_filtering_callbacks.py  # Filtering operations
│   │   └── respiratory_callbacks.py   # Respiratory analysis
│   ├── features/                # Feature extraction callbacks
│   │   ├── physiological_callbacks.py # Physiological features
│   │   ├── features_callbacks.py      # Advanced features
│   │   └── preview_callbacks.py       # Data preview
│   └── utils/                   # Utility callbacks
├── layout/                      # Layout components organized by type
│   ├── common/                  # Reusable layout components
│   │   ├── header.py           # Application header
│   │   ├── sidebar.py          # Navigation sidebar
│   │   └── footer.py           # Application footer
│   ├── pages/                   # Page-specific layouts
│   │   ├── upload_page.py      # Upload page layout
│   │   └── analysis_pages.py   # Analysis page layouts
│   └── components/              # Reusable UI components
├── services/                    # Business logic and data services
│   ├── core/                    # Core services
│   ├── analysis/                # Analysis services
│   └── data/                    # Data management services
│       └── data_service.py     # Data loading and processing
├── utils/                       # Utility functions and helpers
│   └── data_processor.py       # Data processing utilities
├── legacy/                      # Original code preserved for reference
│   ├── callbacks/               # Original callback files
│   ├── layout/                  # Original layout files
│   └── services/                # Original service files
├── config/                      # Configuration files
├── api/                         # FastAPI endpoints
├── assets/                      # Static assets
├── models/                      # Data models
└── app.py                       # Main application entry point
```

## Key Improvements

### 1. **Modular Callbacks**
- **Core callbacks**: Basic application functionality like sidebar toggling and page routing
- **Analysis callbacks**: Signal processing and analysis operations
- **Feature callbacks**: Feature extraction and engineering
- **Utility callbacks**: Helper functions and utilities

### 2. **Organized Layouts**
- **Common components**: Reusable UI elements like header, sidebar, footer
- **Page layouts**: Specific layouts for each application page
- **Component layouts**: Modular UI components for reuse

### 3. **Service Layer**
- **Data services**: Centralized data management and processing
- **Analysis services**: Signal processing algorithms and methods
- **Core services**: Application-wide functionality

### 4. **Utility Functions**
- **Data processing**: File handling, validation, and processing utilities
- **Helper functions**: Common operations and calculations

## Benefits of the New Structure

### **Maintainability**
- Clear separation of concerns
- Easier to locate and modify specific functionality
- Reduced code duplication

### **Readability**
- Logical organization by functionality
- Consistent naming conventions
- Clear import paths

### **Extensibility**
- Easy to add new features
- Modular design allows independent development
- Clear interfaces between components

### **Testing**
- Easier to write unit tests for specific modules
- Isolated functionality for better test coverage
- Clear dependencies and interfaces

## Migration Notes

### **Legacy Code**
- All original code has been preserved in the `legacy/` directory
- Original functionality has been recreated in the new structure
- No functionality has been lost during refactoring

### **Import Changes**
- All imports have been updated to use the new structure
- The main `app.py` file now imports from the modular structure
- Callback registration follows the new organization

### **File Naming**
- Files are named descriptively based on their functionality
- Consistent naming conventions across all modules
- Clear indication of purpose and scope

## Usage Examples

### **Adding New Callbacks**
```python
# In callbacks/analysis/new_analysis_callbacks.py
def register_new_analysis_callbacks(app):
    @app.callback(...)
    def new_callback(...):
        # Implementation
        pass

# In callbacks/analysis/__init__.py
from .new_analysis_callbacks import register_new_analysis_callbacks

# In callbacks/__init__.py
from .analysis.new_analysis_callbacks import register_new_analysis_callbacks
```

### **Adding New Layouts**
```python
# In layout/pages/new_page.py
def new_page_layout():
    return html.Div([...])

# In layout/pages/__init__.py
from .new_page import new_page_layout

# In layout/__init__.py
from .pages.new_page import new_page_layout
```

### **Adding New Services**
```python
# In services/analysis/new_service.py
class NewAnalysisService:
    def analyze(self, data):
        # Implementation
        pass

# In services/__init__.py
from .analysis.new_service import NewAnalysisService
```

## Best Practices

### **Module Organization**
- Keep related functionality together
- Use clear, descriptive names
- Maintain consistent structure across modules

### **Import Management**
- Use relative imports within packages
- Keep `__init__.py` files clean and organized
- Avoid circular imports

### **Code Style**
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Include comprehensive docstrings

### **Error Handling**
- Implement proper error handling in all callbacks
- Use logging for debugging and monitoring
- Provide user-friendly error messages

## Future Enhancements

### **Planned Improvements**
- Add more comprehensive error handling
- Implement caching for expensive operations
- Add unit tests for all modules
- Create API documentation

### **Extensibility Points**
- Plugin system for custom analysis methods
- Configuration-driven feature toggles
- Customizable UI themes and layouts

## Conclusion

The refactored architecture provides a solid foundation for future development while maintaining all existing functionality. The modular structure makes the codebase more maintainable, testable, and extensible.

For questions or issues with the new structure, please refer to the legacy code in the `legacy/` directory or contact the development team.
