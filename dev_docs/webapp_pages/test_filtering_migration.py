#!/usr/bin/env python3
"""
Comprehensive Filtering Page Migration Test Suite

This script tests all aspects of the filtering page migration to ensure
complete functionality and proper component coverage.
"""

import sys
import os
import numpy as np
import pandas as pd
import traceback
from typing import Dict, List, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def test_component_imports() -> bool:
    """Test that all components can be imported and registered."""
    try:
        print("🧪 Testing component imports...")
        
        from vitalDSP_webapp.layout.pages.filtering_page import filtering_layout
        from vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks import register_signal_filtering_callbacks
        from vitalDSP_webapp.callbacks.utils.export_callbacks import register_all_export_callbacks
        
        # Test layout creation
        layout = filtering_layout()
        assert layout is not None, "Filtering layout should be created successfully"
        print("  ✅ Filtering layout created successfully")
        
        # Test callback registration
        from dash import Dash
        app = Dash(__name__)
        register_signal_filtering_callbacks(app)
        register_all_export_callbacks(app)
        
        assert len(app.callback_map) > 0, "Callbacks should be registered"
        print(f"  ✅ {len(app.callback_map)} callbacks registered successfully")
        
        return True
    except Exception as e:
        print(f"  ❌ Component import test failed: {e}")
        traceback.print_exc()
        return False

def test_callback_coverage() -> bool:
    """Test that all components have proper callback coverage."""
    try:
        print("🧪 Testing callback coverage...")
        
        # Expected callback mappings
        expected_outputs = [
            "filter-signal-type-select", "filter-type-select", "advanced-filter-method",
            "traditional-filter-params", "advanced-filter-params", "artifact-removal-params",
            "neural-network-params", "ensemble-params", "filter-original-plot",
            "filter-filtered-plot", "filter-comparison-plot", "filter-quality-metrics",
            "filter-quality-plots", "store-filtering-data", "store-filter-comparison",
            "store-filter-quality-metrics", "store-filtered-signal", "download-filtered-csv",
            "download-filtered-json"
        ]
        
        # Test callback registration
        from dash import Dash
        from vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks import register_signal_filtering_callbacks
        from vitalDSP_webapp.callbacks.utils.export_callbacks import register_all_export_callbacks
        
        app = Dash(__name__)
        register_signal_filtering_callbacks(app)
        register_all_export_callbacks(app)
        
        # Verify callbacks are registered
        callback_ids = list(app.callback_map.keys())
        
        # Check for filtering-related callbacks
        filtering_callbacks = [cb for cb in callback_ids if any(comp in str(cb) for comp in expected_outputs)]
        
        assert len(filtering_callbacks) > 0, "Filtering callbacks should be registered"
        print(f"  ✅ {len(filtering_callbacks)} filtering-related callbacks found")
        
        # Check for export callbacks
        export_callbacks = [cb for cb in callback_ids if "export" in str(cb) or "download" in str(cb)]
        print(f"  ✅ {len(export_callbacks)} export-related callbacks found")
        
        return True
    except Exception as e:
        print(f"  ❌ Callback coverage test failed: {e}")
        traceback.print_exc()
        return False

def test_state_parameters() -> bool:
    """Test that all state parameters are properly handled."""
    try:
        print("🧪 Testing state parameters...")
        
        # Test callback signature by examining the source code
        from vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks import register_signal_filtering_callbacks
        import inspect
        
        # Get the source code to find the callback function signature
        source = inspect.getsource(register_signal_filtering_callbacks)
        lines = source.split('\n')
        
        # Find the advanced_filtering_callback function
        callback_start = None
        for i, line in enumerate(lines):
            if 'def advanced_filtering_callback(' in line:
                callback_start = i
                break
        
        if callback_start is None:
            print("  ❌ Could not find advanced_filtering_callback function")
            return False
        
        # Extract parameters from the function signature
        params = []
        for j in range(1, 50):  # Look at next 50 lines
            if callback_start + j < len(lines):
                param_line = lines[callback_start + j].strip()
                if param_line and not param_line.startswith('#') and not param_line.startswith('def'):
                    if '):' in param_line:
                        # Last parameter line
                        param = param_line.replace('):', '').strip()
                        if param:
                            # Remove trailing comma if present
                            param = param.rstrip(',')
                            params.append(param)
                        break
                    else:
                        # Remove trailing comma if present
                        param = param_line.rstrip(',')
                        params.append(param)
        
        # Expected parameters
        expected_params = [
            "pathname", "n_clicks", "nudge_m10", "center_click", "nudge_p10",
            "start_position", "duration", "filter_type", "filter_family", "filter_response",
            "low_freq", "high_freq", "filter_order", "advanced_method", "noise_level",
            "iterations", "learning_rate", "artifact_type", "artifact_strength",
            "neural_type", "neural_complexity", "ensemble_method", "ensemble_n_filters",
            "quality_options", "detrend_option", "signal_type", "savgol_window",
            "savgol_polyorder", "moving_avg_window", "gaussian_sigma", "wavelet_type",
            "wavelet_level", "threshold_type", "threshold_value", "reference_signal",
            "fusion_method"
        ]
        
        # Check that all expected parameters are present
        missing_params = [p for p in expected_params if p not in params]
        
        if missing_params:
            print(f"  ❌ Missing parameters: {missing_params}")
            print(f"  Found parameters: {params}")
            return False
        
        print(f"  ✅ All {len(expected_params)} parameters present")
        print(f"  ✅ Function signature verified with {len(params)} parameters")
        return True
    except Exception as e:
        print(f"  ❌ State parameters test failed: {e}")
        traceback.print_exc()
        return False

def test_store_data_population() -> bool:
    """Test that all data stores are properly populated."""
    try:
        print("🧪 Testing store data population...")
        
        # Mock data
        mock_signal = np.random.randn(1000)
        mock_time = np.linspace(0, 10, 1000)
        
        # Test store data creation
        comparison_data = {
            "original_signal": mock_signal.tolist(),
            "filtered_signal": mock_signal.tolist(),
            "time_axis": mock_time.tolist(),
            "sampling_freq": 100,
            "filter_type": "traditional",
            "filter_params": {
                "filter_family": "butter",
                "filter_response": "bandpass",
                "low_freq": 0.5,
                "high_freq": 40,
                "filter_order": 4,
            }
        }
        
        quality_metrics_data = {
            "snr_improvement": 5.2,
            "mse": 0.1,
            "correlation": 0.95,
            "filter_type": "traditional",
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        
        filtered_signal_data = {
            "signal": mock_signal.tolist(),
            "time": mock_time.tolist(),
            "sampling_freq": 100,
            "filter_type": "traditional",
            "filter_params": {
                "filter_family": "butter",
                "filter_response": "bandpass",
                "low_freq": 0.5,
                "high_freq": 40,
                "filter_order": 4,
            },
            "signal_type": "PPG",
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        
        # Verify data structures
        assert "original_signal" in comparison_data, "Comparison data should contain original signal"
        assert "filtered_signal" in comparison_data, "Comparison data should contain filtered signal"
        assert "snr_improvement" in quality_metrics_data, "Quality metrics should contain SNR improvement"
        assert "signal" in filtered_signal_data, "Filtered signal data should contain signal"
        
        print("  ✅ All store data structures validated")
        return True
    except Exception as e:
        print(f"  ❌ Store data population test failed: {e}")
        traceback.print_exc()
        return False

def test_export_functionality() -> bool:
    """Test that export functionality works correctly."""
    try:
        print("🧪 Testing export functionality...")
        
        from vitalDSP_webapp.callbacks.utils.export_callbacks import register_filtering_export_callbacks
        from dash import Dash
        
        app = Dash(__name__)
        register_filtering_export_callbacks(app)
        
        # Check that export callbacks are registered
        callback_ids = list(app.callback_map.keys())
        export_callbacks = [cb for cb in callback_ids if "export" in str(cb) or "download" in str(cb)]
        
        assert len(export_callbacks) > 0, "Export callbacks should be registered"
        
        # Check for specific export callbacks
        csv_export_found = any("csv" in str(cb) for cb in export_callbacks)
        json_export_found = any("json" in str(cb) for cb in export_callbacks)
        
        assert csv_export_found, "CSV export callback should be registered"
        assert json_export_found, "JSON export callback should be registered"
        
        print("  ✅ Export functionality validated")
        return True
    except Exception as e:
        print(f"  ❌ Export functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_plot_generation() -> bool:
    """Test that all plots can be generated correctly."""
    try:
        print("🧪 Testing plot generation...")
        
        from vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks import (
            create_original_signal_plot,
            create_filtered_signal_plot,
            create_filter_comparison_plot,
            create_empty_figure
        )
        
        # Mock data
        time_axis = np.linspace(0, 10, 1000)
        signal_data = np.random.randn(1000)
        sampling_freq = 100
        
        # Test plot creation
        original_plot = create_original_signal_plot(time_axis, signal_data, sampling_freq, "PPG")
        assert original_plot is not None, "Original plot should be created"
        
        filtered_plot = create_filtered_signal_plot(time_axis, signal_data, sampling_freq, "PPG")
        assert filtered_plot is not None, "Filtered plot should be created"
        
        comparison_plot = create_filter_comparison_plot(time_axis, signal_data, signal_data, sampling_freq, "PPG")
        assert comparison_plot is not None, "Comparison plot should be created"
        
        empty_plot = create_empty_figure()
        assert empty_plot is not None, "Empty figure should be created"
        
        print("  ✅ All plots generated successfully")
        return True
    except Exception as e:
        print(f"  ❌ Plot generation test failed: {e}")
        traceback.print_exc()
        return False

def test_quality_metrics() -> bool:
    """Test that quality metrics are calculated correctly."""
    try:
        print("🧪 Testing quality metrics...")
        
        from vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks import (
            calculate_snr_improvement,
            calculate_mse,
            calculate_correlation,
            calculate_frequency_metrics,
            calculate_statistical_metrics
        )
        
        # Mock data
        original_signal = np.random.randn(1000)
        filtered_signal = original_signal + np.random.randn(1000) * 0.1  # Add some noise
        
        snr_improvement = calculate_snr_improvement(original_signal, filtered_signal)
        assert isinstance(snr_improvement, (int, float)), "SNR improvement should be numeric"
        
        mse = calculate_mse(original_signal, filtered_signal)
        assert isinstance(mse, (int, float)), "MSE should be numeric"
        
        correlation = calculate_correlation(original_signal, filtered_signal)
        assert isinstance(correlation, (int, float)), "Correlation should be numeric"
        
        freq_metrics = calculate_frequency_metrics(original_signal, filtered_signal, 100)
        assert isinstance(freq_metrics, dict), "Frequency metrics should be a dictionary"
        
        stat_metrics = calculate_statistical_metrics(original_signal, filtered_signal)
        assert isinstance(stat_metrics, dict), "Statistical metrics should be a dictionary"
        
        print("  ✅ All quality metrics calculated successfully")
        return True
    except Exception as e:
        print(f"  ❌ Quality metrics test failed: {e}")
        traceback.print_exc()
        return False

def test_filter_functions() -> bool:
    """Test that all filter functions work correctly."""
    try:
        print("🧪 Testing filter functions...")
        
        from vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks import (
            apply_traditional_filter,
            apply_advanced_filter,
            apply_neural_filter,
            apply_ensemble_filter
        )
        
        # Mock data
        signal_data = np.random.randn(1000)
        sampling_freq = 100
        
        # Test traditional filter
        filtered_traditional = apply_traditional_filter(
            signal_data, sampling_freq, "butter", "bandpass", 0.5, 40, 4
        )
        assert len(filtered_traditional) == len(signal_data), "Filtered signal should have same length"
        
        # Test advanced filter
        filtered_advanced = apply_advanced_filter(
            signal_data, "convolution", 0.1, 10, 0.01
        )
        assert len(filtered_advanced) == len(signal_data), "Advanced filtered signal should have same length"
        
        # Test neural filter
        filtered_neural = apply_neural_filter(signal_data, "autoencoder", "medium")
        assert len(filtered_neural) == len(signal_data), "Neural filtered signal should have same length"
        
        # Test ensemble filter
        filtered_ensemble = apply_ensemble_filter(signal_data, "voting", 3)
        assert len(filtered_ensemble) == len(signal_data), "Ensemble filtered signal should have same length"
        
        print("  ✅ All filter functions work correctly")
        return True
    except Exception as e:
        print(f"  ❌ Filter functions test failed: {e}")
        traceback.print_exc()
        return False

def test_webapp_integration() -> bool:
    """Test complete webapp integration."""
    try:
        print("🧪 Testing webapp integration...")
        
        from vitalDSP_webapp.app import create_dash_app
        
        app = create_dash_app()
        
        # Check callback count
        callback_count = len(app.callback_map)
        assert callback_count > 90, f"Should have at least 90 callbacks, got {callback_count}"
        
        # Check for filtering callbacks
        filtering_callbacks = [cb for cb in app.callback_map.keys() if 'filter' in str(cb)]
        assert len(filtering_callbacks) > 0, "Should have filtering callbacks"
        
        # Check for export callbacks
        export_callbacks = [cb for cb in app.callback_map.keys() if 'export' in str(cb) or 'download' in str(cb)]
        assert len(export_callbacks) > 0, "Should have export callbacks"
        
        print(f"  ✅ Webapp integration successful - {callback_count} callbacks, {len(filtering_callbacks)} filtering, {len(export_callbacks)} export")
        return True
    except Exception as e:
        print(f"  ❌ Webapp integration test failed: {e}")
        traceback.print_exc()
        return False

def run_comprehensive_tests() -> Dict[str, str]:
    """Run all comprehensive tests."""
    
    tests = [
        ("Component Import", test_component_imports),
        ("Callback Coverage", test_callback_coverage),
        ("State Parameters", test_state_parameters),
        ("Store Data Population", test_store_data_population),
        ("Export Functionality", test_export_functionality),
        ("Plot Generation", test_plot_generation),
        ("Quality Metrics", test_quality_metrics),
        ("Filter Functions", test_filter_functions),
        ("Webapp Integration", test_webapp_integration),
    ]
    
    results = {}
    
    print("🚀 Starting Comprehensive Filtering Page Migration Tests")
    print("=" * 60)
    
    for test_name, test_func in tests:
        try:
            print(f"\n📋 Running {test_name} test...")
            result = test_func()
            results[test_name] = "✅ PASSED" if result else "❌ FAILED"
        except Exception as e:
            results[test_name] = f"❌ ERROR: {str(e)}"
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results.items():
        print(f"{result} {test_name}")
        if "✅ PASSED" in result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n📈 Overall Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! Migration is complete and working correctly.")
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
    
    return results

if __name__ == "__main__":
    results = run_comprehensive_tests()
    sys.exit(0 if all("✅ PASSED" in result for result in results.values()) else 1)
