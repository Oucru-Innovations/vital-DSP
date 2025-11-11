"""
Test cases for vitalDSP/__init__.py module.

This test file covers the ImportError exception handling in the __init__.py file
to ensure complete code coverage.
"""

import pytest
import sys
from unittest.mock import patch, MagicMock
from io import StringIO


class TestInitImportError:
    """Test ImportError handling in vitalDSP/__init__.py"""
    
    def test_import_error_handling(self):
        """Test that ImportError is properly handled when modules cannot be imported.
        
        This test covers lines 113-115 in src/vitalDSP/__init__.py.
        """
        # Save original modules
        original_modules = sys.modules.copy()
        
        try:
            # Remove vitalDSP from sys.modules if it exists to force reimport
            modules_to_remove = [key for key in list(sys.modules.keys()) if key.startswith('vitalDSP')]
            for module in modules_to_remove:
                del sys.modules[module]
            
            # Create a custom import hook that raises ImportError for the specific module
            original_import = __import__
            call_tracker = {'in_vitaldsp_init': False}
            
            def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
                # Detect if we're in vitalDSP.__init__.py by checking the file path
                if globals:
                    file_path = globals.get('__file__', '')
                    if 'vitalDSP' in str(file_path) and '__init__.py' in str(file_path):
                        call_tracker['in_vitaldsp_init'] = True
                        # Raise ImportError for the first import in try block
                        if name.endswith('filtering.signal_filtering') or (name.endswith('filtering') and fromlist and 'signal_filtering' in fromlist):
                            raise ImportError("Mocked import error")
                
                # Use real import for everything else
                return original_import(name, globals, locals, fromlist, level)
            
            with patch('builtins.__import__', side_effect=mock_import):
                # Import vitalDSP - should handle ImportError gracefully
                import vitalDSP
                
                # Check that __all__ is empty when ImportError occurs
                assert vitalDSP.__all__ == []
                
        finally:
            # Restore original modules
            sys.modules.clear()
            sys.modules.update(original_modules)
    
    def test_import_error_with_print_output(self, capsys):
        """Test that ImportError prints warning message.
        
        This test covers lines 113-115 in src/vitalDSP/__init__.py, specifically
        the print statement on line 114.
        """
        # Save original modules
        original_modules = {k: v for k, v in sys.modules.items() if k.startswith('vitalDSP')}
        
        try:
            # Remove vitalDSP from sys.modules if it exists
            modules_to_remove = [key for key in list(sys.modules.keys()) if key.startswith('vitalDSP')]
            for module in modules_to_remove:
                del sys.modules[module]
            
            # Patch __import__ to raise ImportError
            original_import = __import__
            error_message = "Test import error message"
            
            def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
                # Check if we're importing from vitalDSP.__init__.py
                if globals:
                    file_path = str(globals.get('__file__', ''))
                    if 'vitalDSP' in file_path and '__init__.py' in file_path:
                        # Check if this is the import we want to fail
                        if name.endswith('filtering.signal_filtering') or (name.endswith('filtering') and fromlist and 'signal_filtering' in fromlist):
                            raise ImportError(error_message)
                
                # Use real import for everything else
                return original_import(name, globals, locals, fromlist, level)
            
            with patch('builtins.__import__', side_effect=mock_import):
                import vitalDSP
                
                # Capture stdout to check if warning was printed
                captured = capsys.readouterr()
                
                # Check that warning message was printed
                assert "Warning: Some vitalDSP modules could not be imported" in captured.out
                assert error_message in captured.out
                
                # Check that __all__ is empty
                assert vitalDSP.__all__ == []
                
        finally:
            # Restore only vitalDSP modules
            for module in list(sys.modules.keys()):
                if module.startswith('vitalDSP'):
                    del sys.modules[module]
            sys.modules.update(original_modules)
    
    def test_normal_import_success(self):
        """Test that normal import works correctly (baseline test).
        
        This ensures the test setup doesn't break normal imports.
        """
        # This should work normally - don't clear modules to avoid scipy issues
        import vitalDSP
        
        # Check that __all__ is populated when import succeeds
        assert len(vitalDSP.__all__) > 0
        assert "SignalFiltering" in vitalDSP.__all__
        assert "TimeDomainFeatures" in vitalDSP.__all__
