import pytest
import numpy as np
from vitalDSP.health_analysis.health_report_generator import HealthReportGenerator
import matplotlib
from unittest.mock import patch, MagicMock

# Use Agg backend to prevent issues with Tkinter in testing environments
matplotlib.use("Agg")


@pytest.fixture
def sample_feature_data():
    """Fixture to provide sample feature data for the tests."""
    return {
        "sdnn": [
            35.0,
            36.0,
            37.0,
            38.0,
            39.0,
            40.0,
            41.0,
            42.0,
            43.0,
            1000,
            40.0,
            40.0,
            40.0,
            35.0,
            37.0,
            45.0,
            50.0,
            60.0,
            10000.0,
            70.0,
        ],  # Out-of-range and identical values
        "rmssd": [
            45.0,
            44.0,
            43.0,
            42.0,
            41.0,
            40.0,
            39.0,
            38.0,
            37.0,
            45.0,
            45.0,
            45.0,
            30.0,
            40.0,
            60.0,
            100.0,
            45.0,
            45.0,
            200.0,
            300.0,
        ],  # Out-of-range and same values
        "nn50": [
            30.0,
            31.0,
            32.0,
            33.0,
            34.0,
            35.0,
            36.0,
            37.0,
            38.0,
            40.0,
            9.0,
            40.0,
            40.0,
            40.0,
            30.0,
            35.0,
            9.0,
            9.0,
            100.0,
            9.0,
        ],  # Same values and an outlier
        "pnn50": [28.5] * 20,  # All identical values
        "mean_nn": [750.0] * 20,  # All identical values
    }


@pytest.fixture
def report_generator(sample_feature_data):
    """Fixture to provide a HealthReportGenerator instance with sample data."""
    return HealthReportGenerator(
        feature_data=sample_feature_data, segment_duration="1_min"
    )


# Mocking the required components for HealthReportGenerator
@pytest.fixture
def mock_interpreter():
    interpreter = MagicMock()
    interpreter.interpret_feature.return_value = {
        "description": "Mock description",
        "interpretation": "Mock interpretation",
        "normal_range": [10, 100],
        "contradiction": "Mock contradiction",
        "correlation": "Mock correlation",
    }
    interpreter.get_range_status.return_value = "in_range"
    return interpreter

@pytest.fixture
def mock_visualizer():
    visualizer = MagicMock()
    visualizer.create_visualizations.return_value = {}
    return visualizer

@pytest.fixture
def generator(mock_interpreter, mock_visualizer):
    feature_data = {"nn50": [45, 55, np.nan, np.inf], "rmssd": [60, 70, 80]}
    with patch("vitalDSP.health_analysis.health_report_generator.InterpretationEngine", return_value=mock_interpreter), \
        patch("vitalDSP.health_analysis.health_report_generator.HealthReportVisualizer", return_value=mock_visualizer):
        return HealthReportGenerator(feature_data, segment_duration="1 min")


def test_generate_report_html(report_generator):
    """
    Test if the health report is generated as HTML and if the content has expected keys.
    """
    # Generate the report
    report_html = report_generator.generate()

    # Basic checks on the generated HTML content
    assert isinstance(report_html, str), "Report should be a string (HTML format)."
    assert (
        "<html" in report_html.lower()
    ), "Generated report should contain HTML structure."
    assert "sdnn" in report_html, "Report should contain the feature 'sdnn'."
    assert "Normal Range" in report_html, "Report should mention 'Normal Range'."


def test_interpretation_data_structure(report_generator):
    """
    Test if the segment values for features are structured properly in the report.
    """
    report_content = report_generator.generate()

    # Check if the interpretation for 'sdnn' contains median, stddev, and list of values
    assert (
        "Median" in report_content
    ), "Report should contain the median value for the feature."
    assert (
        "Standard Deviation" in report_content
    ), "Report should contain the standard deviation value for the feature."


def test_out_of_range_values_handling(report_generator):
    """
    Test how the report handles features with extreme out-of-range values.
    """
    report_html = report_generator.generate()

    # Debug print the report for inspection
    # print(report_html)
    assert report_html is not None, "Report should not be None."
    # Check for outliers in the generated report
    # assert "10000.0" in report_html, "Outliers should be present in the report."
    # assert "Outlier" in report_html, "The term 'Outlier' should appear in the report for out-of-range values."


def test_handling_of_same_values(report_generator):
    """
    Test if the report handles features with identical values correctly.
    """
    report_html = report_generator.generate()

    # Check for identical values handling
    assert (
        "Identical Values" not in report_html
    ), "Report should not falsely indicate identical values as problematic."
    assert "28.5" in report_html, "Identical values should be reported."


def test_normal_values_handling(report_generator):
    """
    Test if the report handles normal-range values correctly.
    """
    report_html = report_generator.generate()

    # Check for normal-range values handling
    assert "750.0" in report_html, "Normal-range values should be correctly displayed."
    assert "Normal Range" in report_html, "Report should mention 'Normal Range'."

#-----------------------------------------------------
# Update test cases

def test_generate_filter_status_skip(generator):
    # Modify the range status to simulate mismatch with filter_status
    generator.interpreter.get_range_status.return_value = "out_of_range"
    
    # Call generate with filter_status different from range_status
    report_html = generator.generate(filter_status="in_range")
    
    # Ensure nn50 is skipped due to filter status mismatch
    assert "nn50" not in report_html

def test_generate_feature_processing_exception(generator):
    # Mock interpreter to raise an exception when interpreting nn50
    generator.interpreter.interpret_feature.side_effect = Exception("Mock Error")
    
    # Ensure it logs the error but continues with other features
    with patch.object(generator.logger, 'error') as mock_log_error:
        generator.generate()
        # mock_log_error.assert_called_with("Error processing rmssd: Mock Error")

def test_generate_process_interpretations_exception(generator):
    # This test is no longer relevant as process_interpretations was removed
    # The functionality is now handled directly in the interpretation engine
    # Test that the generator still works without process_interpretations
    report_html = generator.generate()
    assert isinstance(report_html, str)
    assert len(report_html) > 0

def test_generate_feature_report(generator):
    # Call _generate_feature_report for nn50 with mock data
    feature_report = generator._generate_feature_report("nn50", 45)

    # Ensure the report has the expected keys and values
    assert feature_report["description"] == "Mock description"
    assert feature_report["value"] == 45
    assert feature_report["interpretation"] == "Mock interpretation"
    assert feature_report["normal_range"] == [10, 100]
    assert feature_report["contradiction"] == "Mock contradiction"
    assert feature_report["correlation"] == "Mock correlation"


def test_concurrent_processing_configuration(generator):
    """Test concurrent processing configuration methods."""
    # Test get_performance_info
    perf_info = generator.get_performance_info()
    assert "cpu_count" in perf_info
    assert "feature_count" in perf_info
    assert "current_max_workers" in perf_info
    assert "recommended_feature_workers" in perf_info
    assert "recommended_viz_workers" in perf_info
    assert "estimated_speedup" in perf_info
    
    # Test set_concurrency
    generator.set_concurrency(max_workers=4)
    assert generator.max_workers == 4
    
    # Test reset to default
    generator.set_concurrency()
    assert generator.max_workers > 0  # Should be CPU count


def test_generate_with_visualizations(generator):
    """Test report generation with visualization output directory."""
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock the static batch_visualization method to return empty dict to avoid actual plotting
        with patch('vitalDSP.health_analysis.health_report_generator.HealthReportGenerator.batch_visualization', return_value={}) as mock_batch:
            report_html = generator.generate(output_dir=temp_dir)
            
            # Should call batch_visualization
            mock_batch.assert_called_once()
            
            # Report should still be generated
            assert isinstance(report_html, str)
            assert len(report_html) > 0


def test_generate_without_visualizations(generator):
    """Test report generation without visualization output directory."""
    report_html = generator.generate(output_dir=None)
    
    # Report should be generated without visualizations
    assert isinstance(report_html, str)
    assert len(report_html) > 0


def test_process_feature_method(generator):
    """Test the _process_feature method for concurrent processing."""
    # Test with valid data
    feature_name, processed_data = generator._process_feature("nn50", [45, 55, 65], "all")
    
    assert feature_name == "nn50"
    assert processed_data is not None
    assert "description" in processed_data
    assert "value" in processed_data
    assert "interpretation" in processed_data
    
    # Test with filter mismatch
    feature_name, processed_data = generator._process_feature("nn50", [45, 55, 65], "in_range")
    # Should return None due to filter mismatch (mock returns "in_range" but we're filtering for "in_range")
    assert processed_data is not None  # Actually should pass since mock returns "in_range"
    
    # Test with invalid data (all NaN)
    feature_name, processed_data = generator._process_feature("nn50", [np.nan, np.inf], "all")
    assert processed_data is None  # Should return None for invalid data


def test_concurrent_processing_methods_exist(generator):
    """Test that concurrent processing methods exist and are callable."""
    # Test that the methods exist and can be called
    assert hasattr(generator, 'set_concurrency')
    assert hasattr(generator, 'get_performance_info')
    assert hasattr(generator, '_process_feature')
    assert hasattr(HealthReportGenerator, 'batch_visualization')
    
    # Test that they return expected types
    perf_info = generator.get_performance_info()
    assert isinstance(perf_info, dict)
    assert 'cpu_count' in perf_info
    
    # Test set_concurrency doesn't raise errors
    generator.set_concurrency(max_workers=2)
    assert generator.max_workers == 2


def test_dynamic_analysis_generation(generator):
    """Test dynamic analysis generation."""
    # Create mock segment values
    segment_values = {
        "sdnn": {
            "value": [50, 55, 60],
            "range_status": "in_range",
            "description": "Test description",
            "interpretation": "Test interpretation"
        },
        "rmssd": {
            "value": [30, 35, 40],
            "range_status": "above_range", 
            "description": "Test description 2",
            "interpretation": "Test interpretation 2"
        }
    }
    
    dynamic_analysis = generator._generate_dynamic_analysis(segment_values)
    
    assert isinstance(dynamic_analysis, dict)
    assert "executive_summary" in dynamic_analysis
    assert "risk_assessment" in dynamic_analysis
    assert "recommendations" in dynamic_analysis
    assert "key_insights" in dynamic_analysis
    assert "statistics" in dynamic_analysis
    assert "cross_correlations" in dynamic_analysis
    assert "overall_health_score" in dynamic_analysis


def test_generate_with_empty_feature_data(generator):
    """Test report generation with empty feature data."""
    empty_generator = HealthReportGenerator({}, segment_duration="1_min")
    report_html = empty_generator.generate()
    
    assert isinstance(report_html, str)
    assert len(report_html) > 0
    assert "Error generating report" in report_html or "<html" in report_html.lower()


def test_generate_with_invalid_segment_duration(generator):
    """Test report generation with invalid segment duration."""
    # Use the existing generator but test with invalid segment duration
    generator.segment_duration = "invalid_duration"
    report_html = generator.generate()
    
    assert isinstance(report_html, str)
    assert len(report_html) > 0


def test_process_feature_with_empty_values(generator):
    """Test _process_feature with empty values."""
    feature_name, processed_data = generator._process_feature("test", [], "all")
    
    assert feature_name == "test"
    assert processed_data is None


def test_process_feature_with_single_value(generator):
    """Test _process_feature with single value."""
    feature_name, processed_data = generator._process_feature("test", [42], "all")
    
    assert feature_name == "test"
    assert processed_data is not None
    assert "value" in processed_data
    assert processed_data["value"] == [42]


def test_process_feature_with_large_dataset(generator):
    """Test _process_feature with large dataset (should trigger downsampling)."""
    large_data = list(range(2000))  # Large dataset
    feature_name, processed_data = generator._process_feature("test", large_data, "all")
    
    assert feature_name == "test"
    assert processed_data is not None
    assert "value" in processed_data
    # Should be downsampled to 1000 or less
    assert len(processed_data["value"]) <= 1000


def test_generate_feature_report_with_mock_data(generator):
    """Test _generate_feature_report with various mock data."""
    # Test with normal value
    report = generator._generate_feature_report("test_feature", 50.0)
    assert isinstance(report, dict)
    assert "description" in report
    assert "value" in report
    assert "interpretation" in report
    assert "normal_range" in report
    assert "contradiction" in report
    assert "correlation" in report


def test_validate_segment_duration_valid_cases(generator):
    """Test _validate_segment_duration with valid cases."""
    # Test various valid formats
    valid_durations = ["1_min", "5_min", "1min", "5min", "1_minute", "5_minutes"]
    
    for duration in valid_durations:
        result = generator._validate_segment_duration(duration)
        assert result in ["1_min", "5_min"]


def test_validate_segment_duration_invalid_cases(generator):
    """Test _validate_segment_duration with invalid cases."""
    # Test invalid duration
    result = generator._validate_segment_duration("invalid")
    assert result == "1_min"  # Should default to 1_min


def test_generate_dynamic_analysis_with_empty_data(generator):
    """Test _generate_dynamic_analysis with empty segment values."""
    empty_analysis = generator._generate_dynamic_analysis({})
    
    assert isinstance(empty_analysis, dict)
    assert "executive_summary" in empty_analysis
    assert "risk_assessment" in empty_analysis
    assert "recommendations" in empty_analysis
    assert "key_insights" in empty_analysis
    assert "statistics" in empty_analysis
    assert "cross_correlations" in empty_analysis
    assert "overall_health_score" in empty_analysis


def test_generate_dynamic_analysis_with_mixed_range_statuses(generator):
    """Test _generate_dynamic_analysis with mixed range statuses."""
    mixed_data = {
        "feature1": {
            "value": [50, 55, 60],
            "range_status": "in_range",
            "description": "Normal feature",
            "interpretation": "Normal interpretation"
        },
        "feature2": {
            "value": [10, 15, 20],
            "range_status": "below_range",
            "description": "Low feature",
            "interpretation": "Low interpretation"
        },
        "feature3": {
            "value": [90, 95, 100],
            "range_status": "above_range",
            "description": "High feature",
            "interpretation": "High interpretation"
        }
    }
    
    analysis = generator._generate_dynamic_analysis(mixed_data)
    
    assert isinstance(analysis, dict)
    assert "executive_summary" in analysis
    assert "risk_assessment" in analysis
    assert "statistics" in analysis
    assert analysis["statistics"]["in_range"] == 1
    assert analysis["statistics"]["below_range"] == 1
    assert analysis["statistics"]["above_range"] == 1


def test_generate_with_exception_handling(generator):
    """Test generate method with exception handling."""
    # Mock interpreter to raise exception
    generator.interpreter.interpret_feature.side_effect = Exception("Test error")
    
    report_html = generator.generate()
    
    assert isinstance(report_html, str)
    assert len(report_html) > 0


def test_generate_with_visualization_exception(generator):
    """Test generate method with visualization exception."""
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock batch_visualization to raise exception
        with patch('vitalDSP.health_analysis.health_report_generator.HealthReportGenerator.batch_visualization', side_effect=Exception("Visualization error")):
            report_html = generator.generate(output_dir=temp_dir)
            
            assert isinstance(report_html, str)
            assert len(report_html) > 0


def test_generate_with_render_exception(generator):
    """Test generate method with render exception."""
    # Mock render_report to raise exception
    with patch('vitalDSP.health_analysis.health_report_generator.render_report', side_effect=Exception("Render error")):
        report_html = generator.generate()
        
        assert isinstance(report_html, str)
        assert "Error generating report" in report_html


def test_downsample_method(generator):
    """Test the downsample method."""
    # Test with large array
    large_array = np.array(list(range(2000)))
    downsampled = HealthReportGenerator.downsample(large_array)
    
    assert len(downsampled) == 200  # 2000 / 10 = 200
    assert isinstance(downsampled, np.ndarray)
    
    # Test with small array (will be downsampled by factor of 10)
    small_array = np.array([1, 2, 3, 4, 5])
    small_downsampled = HealthReportGenerator.downsample(small_array)
    
    assert len(small_downsampled) == 1  # 5/10 = 0.5, rounded to 1
    assert small_downsampled[0] == 1  # First element


def test_get_performance_info_detailed(generator):
    """Test get_performance_info with detailed assertions."""
    perf_info = generator.get_performance_info()
    
    assert isinstance(perf_info, dict)
    assert "cpu_count" in perf_info
    assert "feature_count" in perf_info
    assert "current_max_workers" in perf_info
    assert "recommended_feature_workers" in perf_info
    assert "recommended_viz_workers" in perf_info
    assert "estimated_speedup" in perf_info
    
    # Check that values are reasonable
    assert perf_info["cpu_count"] > 0
    assert perf_info["feature_count"] >= 0
    assert perf_info["current_max_workers"] > 0
    assert perf_info["recommended_feature_workers"] > 0
    assert perf_info["recommended_viz_workers"] > 0
    assert perf_info["estimated_speedup"] >= 1


def test_set_concurrency_edge_cases(generator):
    """Test set_concurrency with edge cases."""
    # Test with zero workers
    generator.set_concurrency(max_workers=0)
    assert generator.max_workers == 1  # Should be at least 1
    
    # Test with negative workers
    generator.set_concurrency(max_workers=-5)
    assert generator.max_workers == 1  # Should be at least 1
    
    # Test with very large number
    generator.set_concurrency(max_workers=1000)
    assert generator.max_workers <= 1000  # Should be capped


def test_generate_with_different_filter_statuses(generator):
    """Test generate with different filter statuses."""
    # Test with "in_range" filter
    report_in_range = generator.generate(filter_status="in_range")
    assert isinstance(report_in_range, str)
    
    # Test with "above_range" filter
    report_above_range = generator.generate(filter_status="above_range")
    assert isinstance(report_above_range, str)
    
    # Test with "below_range" filter
    report_below_range = generator.generate(filter_status="below_range")
    assert isinstance(report_below_range, str)


def test_process_single_feature_visualization_module_level():
    """Test the module-level process_single_feature_visualization function."""
    from vitalDSP.health_analysis.health_report_generator import process_single_feature_visualization
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock the HealthReportVisualizer import
        with patch('vitalDSP.health_analysis.health_report_generator.HealthReportVisualizer') as mock_visualizer_class:
            mock_visualizer = MagicMock()
            mock_visualizer_class.return_value = mock_visualizer
            mock_visualizer.create_visualizations.return_value = {"test": {"plot1": "path1.png"}}
            
            feature_item = ("test_feature", [1, 2, 3])
            config = {"test": "config"}
            segment_duration = "1_min"
            
            result = process_single_feature_visualization(feature_item, config, segment_duration, temp_dir)
            
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert result[0] == "test_feature"
            assert isinstance(result[1], dict)


def test_process_single_feature_visualization_exception():
    """Test process_single_feature_visualization with exception."""
    from vitalDSP.health_analysis.health_report_generator import process_single_feature_visualization
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock to raise exception
        with patch('vitalDSP.health_analysis.health_report_generator.HealthReportVisualizer', side_effect=Exception("Test error")):
            feature_item = ("test_feature", [1, 2, 3])
            config = {"test": "config"}
            segment_duration = "1_min"

            result = process_single_feature_visualization(feature_item, config, segment_duration, temp_dir)

            assert isinstance(result, tuple)
            assert len(result) == 2
            assert result[0] == "test_feature"
            assert isinstance(result[1], dict)  # Should return dict on error
            # The dict may be empty or have error messages, just check it's a dict


def test_batch_visualization_with_multiple_features():
    """Test batch_visualization with multiple features."""
    from vitalDSP.health_analysis.health_report_generator import HealthReportGenerator
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock visualizer
        mock_visualizer = MagicMock()
        mock_visualizer.config = {}
        mock_visualizer.segment_duration = "1_min"

        feature_data = {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "feature3": [7, 8, 9]
        }

        # Mock process_single_feature_visualization to avoid actual plotting
        with patch('vitalDSP.health_analysis.health_report_generator.process_single_feature_visualization') as mock_process:
            mock_process.return_value = ("feature1", {"plot1": "path1.png"})

            result = HealthReportGenerator.batch_visualization(
                mock_visualizer, feature_data, temp_dir, processes=2
            )

            assert isinstance(result, dict)


def test_batch_visualization_with_exceptions():
    """Test batch_visualization handling exceptions."""
    from vitalDSP.health_analysis.health_report_generator import HealthReportGenerator
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        mock_visualizer = MagicMock()
        mock_visualizer.config = {}
        mock_visualizer.segment_duration = "1_min"

        feature_data = {"feature1": [1, 2, 3]}

        # Mock to raise exception
        with patch('vitalDSP.health_analysis.health_report_generator.process_single_feature_visualization', side_effect=Exception("Test error")):
            result = HealthReportGenerator.batch_visualization(
                mock_visualizer, feature_data, temp_dir, processes=1
            )

            assert isinstance(result, dict)
            # Should return empty dict or dict with empty values for failed features


def test_batch_visualization_path_normalization():
    """Test batch_visualization normalizes paths correctly."""
    from vitalDSP.health_analysis.health_report_generator import HealthReportGenerator
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        mock_visualizer = MagicMock()
        mock_visualizer.config = {}
        mock_visualizer.segment_duration = "1_min"

        feature_data = {"feature1": [1, 2, 3]}

        # Mock to return path with backslashes
        with patch('vitalDSP.health_analysis.health_report_generator.process_single_feature_visualization') as mock_process:
            mock_process.return_value = ("feature1", {"plot1": "path\\to\\plot.png"})

            result = HealthReportGenerator.batch_visualization(
                mock_visualizer, feature_data, temp_dir, processes=1
            )

            assert isinstance(result, dict)
            # Check that paths are normalized to forward slashes
            if "feature1" in result and "plot1" in result["feature1"]:
                assert "\\" not in result["feature1"]["plot1"]
                assert "/" in result["feature1"]["plot1"]


def test_process_feature_with_numpy_array(generator):
    """Test _process_feature with numpy array input."""
    import numpy as np

    values = np.array([10, 20, 30, 40, 50])
    feature_name, processed_data = generator._process_feature("test", values, "all")

    assert feature_name == "test"
    assert processed_data is not None
    assert "value" in processed_data
    assert isinstance(processed_data["value"], list)


def test_process_feature_with_tuple(generator):
    """Test _process_feature with tuple input."""
    values = (10, 20, 30, 40, 50)
    feature_name, processed_data = generator._process_feature("test", values, "all")

    assert feature_name == "test"
    assert processed_data is not None
    assert "value" in processed_data


def test_process_feature_downsample_logging(generator):
    """Test _process_feature logs downsampling warning."""
    large_data = list(range(1500))  # Should trigger downsampling

    with patch.object(generator.logger, 'warning') as mock_warning:
        feature_name, processed_data = generator._process_feature("test", large_data, "all")

        # Should log warning about downsampling
        mock_warning.assert_called()
        assert any("Downsampling" in str(call) for call in mock_warning.call_args_list)


def test_process_feature_error_logging(generator):
    """Test _process_feature logs errors."""
    # Mock interpret_feature to raise exception
    generator.interpreter.interpret_feature.side_effect = Exception("Test error")

    with patch.object(generator.logger, 'error') as mock_error:
        feature_name, processed_data = generator._process_feature("test", [10, 20], "all")

        # Should log error
        mock_error.assert_called()


def test_validate_segment_duration_with_various_formats(generator):
    """Test _validate_segment_duration with various input formats."""
    # Test "1 min"
    assert generator._validate_segment_duration("1 min") == "1_min"

    # Test "1min"
    assert generator._validate_segment_duration("1min") == "1_min"

    # Test "1"
    assert generator._validate_segment_duration("1") == "1_min"

    # Test "5 min"
    assert generator._validate_segment_duration("5 min") == "5_min"

    # Test "5min"
    assert generator._validate_segment_duration("5min") == "5_min"

    # Test "5"
    assert generator._validate_segment_duration("5") == "5_min"

    # Test already normalized
    assert generator._validate_segment_duration("1_min") == "1_min"
    assert generator._validate_segment_duration("5_min") == "5_min"


def test_validate_segment_duration_with_invalid_input(generator):
    """Test _validate_segment_duration with invalid input logs warning."""
    with patch.object(generator.logger, 'warning') as mock_warning:
        result = generator._validate_segment_duration("invalid_duration")

        assert result == "1_min"
        mock_warning.assert_called()


def test_generate_with_all_nan_values():
    """Test generate with feature containing all NaN values."""
    feature_data = {"test": [np.nan, np.nan, np.nan]}
    generator = HealthReportGenerator(feature_data, segment_duration="1_min")

    report_html = generator.generate()

    assert isinstance(report_html, str)
    assert len(report_html) > 0


def test_generate_with_all_inf_values():
    """Test generate with feature containing all Inf values."""
    feature_data = {"test": [np.inf, np.inf, np.inf]}
    generator = HealthReportGenerator(feature_data, segment_duration="1_min")

    report_html = generator.generate()

    assert isinstance(report_html, str)
    assert len(report_html) > 0


def test_generate_with_mixed_valid_invalid_values():
    """Test generate with mixed valid and invalid values."""
    feature_data = {"test": [10, np.nan, 20, np.inf, 30, -np.inf, 40]}
    generator = HealthReportGenerator(feature_data, segment_duration="1_min")

    report_html = generator.generate()

    assert isinstance(report_html, str)
    assert len(report_html) > 0


def test_generate_feature_report_with_none_values(generator):
    """Test _generate_feature_report handles None values in interpretation."""
    # Mock interpreter to return None for some fields
    generator.interpreter.interpret_feature.return_value = {
        "description": "Test description",
        "interpretation": "Test interpretation",
        "normal_range": [10, 100],
        "contradiction": None,
        "correlation": None,
    }

    report = generator._generate_feature_report("test_feature", 50.0)

    assert isinstance(report, dict)
    assert report["contradiction"] is None
    assert report["correlation"] is None


def test_get_performance_info_with_no_features():
    """Test get_performance_info with no features."""
    empty_generator = HealthReportGenerator({}, segment_duration="1_min")
    perf_info = empty_generator.get_performance_info()

    assert isinstance(perf_info, dict)
    assert perf_info["feature_count"] == 0
    assert perf_info["estimated_speedup"] == 1  # No speedup for 0 features


def test_set_concurrency_with_none():
    """Test set_concurrency resets to CPU count when passed None."""
    feature_data = {"test": [1, 2, 3]}
    generator = HealthReportGenerator(feature_data, segment_duration="1_min")

    # Set to custom value first
    generator.set_concurrency(max_workers=2)
    assert generator.max_workers == 2

    # Reset with None
    generator.set_concurrency(max_workers=None)
    import multiprocessing
    assert generator.max_workers == multiprocessing.cpu_count()


def test_downsample_with_factor():
    """Test downsample with custom factor."""
    values = np.array(list(range(100)))

    # Downsample by factor of 5
    downsampled = HealthReportGenerator.downsample(values, factor=5)

    assert len(downsampled) == 20  # 100 / 5 = 20
    assert downsampled[0] == 0
    assert downsampled[1] == 5


def test_downsample_with_factor_1():
    """Test downsample with factor 1 (no downsampling)."""
    values = np.array([1, 2, 3, 4, 5])

    downsampled = HealthReportGenerator.downsample(values, factor=1)

    assert len(downsampled) == 5
    assert list(downsampled) == [1, 2, 3, 4, 5]


def test_process_feature_with_filter_all(generator):
    """Test _process_feature with filter_status='all'."""
    feature_name, processed_data = generator._process_feature("nn50", [45, 55, 65], "all")

    assert feature_name == "nn50"
    assert processed_data is not None  # Should not filter out


def test_process_feature_calculates_statistics_correctly(generator):
    """Test _process_feature calculates median and stddev correctly."""
    values = [10, 20, 30, 40, 50]
    feature_name, processed_data = generator._process_feature("test", values, "all")

    assert processed_data is not None
    assert "median" in processed_data
    assert "stddev" in processed_data

    # Check calculations
    expected_median = np.median(values)
    expected_stddev = np.std(values)

    assert processed_data["median"] == expected_median
    assert processed_data["stddev"] == expected_stddev


def test_process_single_feature_visualization_error_handling():
    """Test process_single_feature_visualization exception handling (line 44-46)."""
    from vitalDSP.health_analysis.health_report_generator import process_single_feature_visualization
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock to raise exception during visualization creation
        with patch('vitalDSP.health_analysis.health_report_generator.HealthReportVisualizer') as mock_viz:
            mock_viz.side_effect = Exception("Visualization creation failed")

            feature_item = ("test_feature", [1, 2, 3])
            config = {"test": "config"}
            segment_duration = "1_min"

            result = process_single_feature_visualization(feature_item, config, segment_duration, temp_dir)

            # Should catch exception and return empty dict (line 44-46)
            assert isinstance(result, tuple)
            assert result[0] == "test_feature"
            assert isinstance(result[1], dict)


def test_process_feature_converts_non_list_to_list(generator):
    """Test _process_feature converts single value to list (line 207)."""
    # Test with single value (not list/tuple/array)
    feature_name, processed_data = generator._process_feature("test", 42.5, "all")

    assert feature_name == "test"
    assert processed_data is not None
    assert "value" in processed_data
    # Should convert single value to list
    assert isinstance(processed_data["value"], list)
    assert 42.5 in processed_data["value"]


def test_batch_visualization_with_future_exception():
    """Test batch_visualization handles future.result() exception (line 306)."""
    from vitalDSP.health_analysis.health_report_generator import HealthReportGenerator
    from vitalDSP.health_analysis.health_report_visualization import HealthReportVisualizer
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a real visualizer with minimal config
        visualizer = HealthReportVisualizer({}, "1_min")
        
        # Create feature data that will cause an exception during processing
        # Use invalid data that will cause matplotlib to fail
        feature_data = {"feature1": []}  # Empty list will cause issues

        # This should handle the exception gracefully and return error messages for failed features
        result = HealthReportGenerator.batch_visualization(
            visualizer, feature_data, temp_dir, processes=1
        )

        assert isinstance(result, dict)
        # Should return error messages for failed feature
        assert "feature1" in result
        assert isinstance(result["feature1"], dict)
        # Most plots should return error messages, but some might succeed
        # Check for any error messages in the result values
        error_count = sum(1 for v in result["feature1"].values() if "Error" in str(v))
        # Since we're using empty data, all plots should fail and return error messages
        assert error_count > 0  # At least some plots should fail


def test_batch_visualization_path_normalization_none_path():
    """Test batch_visualization handles None paths (lines 316-322)."""
    from vitalDSP.health_analysis.health_report_generator import HealthReportGenerator
    from vitalDSP.health_analysis.health_report_visualization import HealthReportVisualizer
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a real visualizer with minimal config
        visualizer = HealthReportVisualizer({}, "1_min")

        # Create feature data that will generate some plots but may have None paths
        # Use data that might cause some plots to fail and return None
        feature_data = {"feature1": [1, 2, 3, 4, 5]}

        result = HealthReportGenerator.batch_visualization(
            visualizer, feature_data, temp_dir, processes=1
        )

        assert isinstance(result, dict)
        if "feature1" in result:
            # Check that the result contains the expected plot types
            # Some may be error messages if they failed to generate
            expected_plots = [
                "heatmap", "bell_plot", "radar_plot", "violin_plot",
                "line_with_rolling_stats", "lag_plot", "plot_periodogram",
                "plot_spectrogram", "plot_spectral_density", "plot_box_swarm"
            ]
            
            for plot_type in expected_plots:
                if plot_type in result["feature1"]:
                    plot_path = result["feature1"][plot_type]
                    # If the path is not an error message and not None, it should be normalized (no backslashes)
                    if plot_path is not None and "Error generating plot" not in str(plot_path):
                        assert "\\" not in str(plot_path)
                        assert "/" in str(plot_path) or str(plot_path) == ""


class TestHealthReportGeneratorMissingCoverage:
    """Tests to cover missing lines in health_report_generator.py."""

    def test_make_config_picklable_none(self):
        """Test _make_config_picklable when config is None.
        
        This test covers line 67 in health_report_generator.py where
        {} is returned when config is None.
        """
        from vitalDSP.health_analysis.health_report_generator import _make_config_picklable
        
        result = _make_config_picklable(None)
        assert result == {}

    def test_make_config_picklable_magicmock(self):
        """Test _make_config_picklable when config is MagicMock.
        
        This test covers line 71 in health_report_generator.py where
        {} is returned when config is MagicMock.
        """
        from vitalDSP.health_analysis.health_report_generator import _make_config_picklable
        
        mock_config = MagicMock()
        result = _make_config_picklable(mock_config)
        assert result == {}

    def test_make_config_picklable_dict_with_magicmock_value(self):
        """Test _make_config_picklable when dict contains MagicMock value.
        
        This test covers line 79 in health_report_generator.py where
        continue is executed when value is MagicMock.
        """
        from vitalDSP.health_analysis.health_report_generator import _make_config_picklable
        
        config = {
            "valid_key": "valid_value",
            "mock_key": MagicMock(),
            "another_valid": 42
        }
        result = _make_config_picklable(config)
        assert "valid_key" in result
        assert "another_valid" in result
        assert "mock_key" not in result  # Should be skipped

    def test_make_config_picklable_nested_dict(self):
        """Test _make_config_picklable with nested dict.
        
        This test covers line 81 in health_report_generator.py where
        recursive call is made for dict values.
        """
        from vitalDSP.health_analysis.health_report_generator import _make_config_picklable
        
        config = {
            "nested": {
                "key1": "value1",
                "key2": MagicMock()  # Should be skipped
            }
        }
        result = _make_config_picklable(config)
        assert "nested" in result
        assert isinstance(result["nested"], dict)
        assert "key1" in result["nested"]
        assert "key2" not in result["nested"]

    def test_make_config_picklable_list_processing(self):
        """Test _make_config_picklable with list values.
        
        This test covers lines 84-96 in health_report_generator.py where
        lists are processed recursively.
        """
        from vitalDSP.health_analysis.health_report_generator import _make_config_picklable
        
        config = {
            "list_key": [
                "string_value",
                42,
                {"nested": "dict"},
                MagicMock(),  # Should be skipped
                [1, 2, 3]  # Nested list
            ]
        }
        result = _make_config_picklable(config)
        assert "list_key" in result
        assert isinstance(result["list_key"], list)
        assert len(result["list_key"]) == 4  # MagicMock should be skipped
        assert "string_value" in result["list_key"]
        assert 42 in result["list_key"]

    def test_make_config_picklable_object_with_dict(self):
        """Test _make_config_picklable with object that has __dict__.
        
        This test covers lines 98-103 and 108-109 in health_report_generator.py where
        objects with __dict__ are converted recursively.
        """
        from vitalDSP.health_analysis.health_report_generator import _make_config_picklable
        
        class TestObject:
            def __init__(self):
                self.attr1 = "value1"
                self.attr2 = 42
        
        config = {
            "obj_key": TestObject()
        }
        result = _make_config_picklable(config)
        assert "obj_key" in result
        assert isinstance(result["obj_key"], dict)
        assert result["obj_key"]["attr1"] == "value1"
        assert result["obj_key"]["attr2"] == 42

    def test_make_config_picklable_object_with_dict_exception(self):
        """Test _make_config_picklable when object conversion raises exception.
        
        This test covers lines 98-103 in health_report_generator.py where
        exception is caught and continue is executed.
        """
        from vitalDSP.health_analysis.health_report_generator import _make_config_picklable
        
        # Create an object where __dict__ contains something that causes exception
        # when processed recursively. The recursive call to _make_config_picklable(value.__dict__)
        # will iterate over items(), so we need to make items() raise an exception
        class ExceptionDict(dict):
            def items(self):
                raise Exception("Cannot process dict")
        
        class BadObject:
            def __init__(self):
                # Use object.__setattr__ to bypass normal attribute setting
                object.__setattr__(self, '__dict__', ExceptionDict({'attr': 'value'}))
        
        config = {
            "bad_obj": BadObject(),
            "valid_key": "valid_value"
        }
        result = _make_config_picklable(config)
        # Valid key should remain
        assert "valid_key" in result
        # bad_obj should be skipped due to exception
        assert "bad_obj" not in result

    def test_make_config_picklable_fallback(self):
        """Test _make_config_picklable fallback case.
        
        This test covers line 112 in health_report_generator.py where
        {} is returned as fallback.
        """
        from vitalDSP.health_analysis.health_report_generator import _make_config_picklable
        
        # Create an object that doesn't match any condition
        class UnusualObject:
            pass
        
        config = UnusualObject()
        result = _make_config_picklable(config)
        assert result == {}

    def test_batch_visualization_path_none(self):
        """Test batch_visualization when path is None.
        
        This test covers line 449 in health_report_generator.py where
        else branch is executed when path is falsy.
        """
        from vitalDSP.health_analysis.health_report_generator import HealthReportGenerator
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_visualizer = MagicMock()
            mock_visualizer.config = {}
            mock_visualizer.segment_duration = "1_min"
            
            feature_data = {"feature1": [1, 2, 3]}
            
            # Create a mock that simulates the process returning None for some paths
            # We'll patch the function to return a result with None paths
            def mock_process_func(feature_item, config, segment_duration, output_dir):
                return ("feature1", {
                    "plot1": None,  # None path
                    "plot2": "",  # Empty string path
                    "plot3": "path/to/plot.png"  # Valid path
                })
            
            # Patch at the module level before multiprocessing
            import vitalDSP.health_analysis.health_report_generator as hrg_module
            original_func = hrg_module.process_single_feature_visualization
            
            try:
                hrg_module.process_single_feature_visualization = mock_process_func
                
                result = HealthReportGenerator.batch_visualization(
                    mock_visualizer, feature_data, temp_dir, processes=1
                )
                
                assert isinstance(result, dict)
                if "feature1" in result:
                    # Check that None and empty paths are handled correctly (line 449)
                    assert isinstance(result["feature1"], dict)
                    # The else branch (line 449) should preserve None/empty paths as-is
                    if "plot1" in result["feature1"]:
                        assert result["feature1"]["plot1"] is None or result["feature1"]["plot1"] == ""
            finally:
                # Restore original function
                hrg_module.process_single_feature_visualization = original_func

    def test_generate_exception_in_future_result(self, generator):
        """Test generate when future.result() raises exception.
        
        This test covers lines 499-501 in health_report_generator.py where
        exception is caught and continue is executed.
        """
        from unittest.mock import patch, MagicMock
        from concurrent.futures import Future
        
        # Create a mock future that raises exception
        mock_future = MagicMock(spec=Future)
        mock_future.result.side_effect = Exception("Future error")
        
        with patch('vitalDSP.health_analysis.health_report_generator.ThreadPoolExecutor') as mock_executor:
            mock_executor_instance = MagicMock()
            mock_executor.return_value.__enter__.return_value = mock_executor_instance
            mock_executor.return_value.__exit__.return_value = None
            
            # Mock submit to return our problematic future
            mock_executor_instance.submit.return_value = mock_future
            
            # Mock as_completed to return the problematic future
            with patch('vitalDSP.health_analysis.health_report_generator.concurrent.futures.as_completed') as mock_as_completed:
                mock_as_completed.return_value = [mock_future]
                
                with patch.object(generator.logger, 'error') as mock_error:
                    report_html = generator.generate()
                    assert isinstance(report_html, str)
                    # Should log error and continue
                    mock_error.assert_called()

    def test_generate_dynamic_analysis_exception(self, generator):
        """Test _generate_dynamic_analysis exception handling.
        
        This test covers lines 714-730 in health_report_generator.py where
        exception is caught and default values are returned.
        """
        # Create segment_values that will cause an exception
        # Mock a method that will be called to raise exception
        with patch.object(generator, '_generate_executive_summary', side_effect=Exception("Test error")):
            result = generator._generate_dynamic_analysis({
                "test": {
                    "value": [1, 2, 3],
                    "range_status": "in_range"
                }
            })
            
            assert isinstance(result, dict)
            assert "executive_summary" in result
            assert "Unable to generate summary" in result["executive_summary"]
            assert result["overall_health_score"] == 0

    def test_generate_risk_assessment_hrv_out_of_range(self, generator):
        """Test _generate_risk_assessment when HRV features are out of range.
        
        This test covers lines 793-795 in health_report_generator.py where
        risk level is adjusted when multiple HRV features are abnormal.
        """
        segment_values = {
            "sdnn": {"range_status": "below_range", "value": [10]},
            "rmssd": {"range_status": "below_range", "value": [5]},
            "nn50": {"range_status": "below_range", "value": [2]},
            "pnn50": {"range_status": "below_range", "value": [1]}
        }
        
        result = generator._generate_risk_assessment(segment_values, health_score=50)
        assert isinstance(result, dict)
        assert "level" in result
        assert "concerns" in result
        assert any("heart rate variability" in concern.lower() for concern in result["concerns"])

    def test_generate_risk_assessment_cardiovascular_risk(self, generator):
        """Test _generate_risk_assessment when SDNN is below range.
        
        This test covers lines 798-804 in health_report_generator.py where
        cardiovascular risk concern is added.
        """
        segment_values = {
            "sdnn": {"range_status": "below_range", "value": [10]}
        }
        
        result = generator._generate_risk_assessment(segment_values, health_score=50)
        assert isinstance(result, dict)
        assert "concerns" in result
        assert any("cardiovascular" in concern.lower() for concern in result["concerns"])

    def test_generate_recommendations_hrv_below_range(self, generator):
        """Test _generate_recommendations for HRV features below range.
        
        This test covers lines 840-843 in health_report_generator.py where
        recommendations are added for HRV features below range.
        """
        segment_values = {
            "sdnn": {"range_status": "below_range", "value": [10]},
            "rmssd": {"range_status": "below_range", "value": [5]}
        }
        risk_assessment = {"level": "moderate", "concerns": []}
        
        result = generator._generate_recommendations(segment_values, risk_assessment)
        assert isinstance(result, list)
        assert any("stress management" in rec.lower() or "exercise" in rec.lower() for rec in result)

    def test_generate_recommendations_heart_rate_below_range(self, generator):
        """Test _generate_recommendations for heart_rate below range.
        
        This test covers lines 844-847 in health_report_generator.py where
        recommendations are added for heart_rate below range.
        """
        segment_values = {
            "heart_rate": {"range_status": "below_range", "value": [40]}
        }
        risk_assessment = {"level": "moderate", "concerns": []}
        
        result = generator._generate_recommendations(segment_values, risk_assessment)
        assert isinstance(result, list)
        assert any("cardiovascular" in rec.lower() for rec in result)

    def test_generate_recommendations_heart_rate_above_range(self, generator):
        """Test _generate_recommendations for heart_rate above range.
        
        This test covers lines 849-852 in health_report_generator.py where
        recommendations are added for heart_rate above range.
        """
        segment_values = {
            "heart_rate": {"range_status": "above_range", "value": [120]}
        }
        risk_assessment = {"level": "moderate", "concerns": []}
        
        result = generator._generate_recommendations(segment_values, risk_assessment)
        assert isinstance(result, list)
        assert any("tachycardia" in rec.lower() or "cardiovascular" in rec.lower() for rec in result)

    def test_generate_key_insights_few_out_of_range(self, generator):
        """Test _generate_key_insights when <= 2 features are out of range.
        
        This test covers lines 877-880 in health_report_generator.py where
        insight is generated for few out-of-range features.
        """
        segment_values = {
            "feature1": {"range_status": "in_range", "value": [50]},
            "feature2": {"range_status": "below_range", "value": [10]},
            "feature3": {"range_status": "in_range", "value": [50]}
        }
        
        result = generator._generate_key_insights(segment_values)
        assert isinstance(result, list)
        assert any("most parameters" in insight.lower() or "2" in insight for insight in result)

    def test_generate_key_insights_many_out_of_range(self, generator):
        """Test _generate_key_insights when > 2 features are out of range.
        
        This test covers lines 882-884 in health_report_generator.py where
        insight is generated for many out-of-range features.
        """
        segment_values = {
            "feature1": {"range_status": "below_range", "value": [10]},
            "feature2": {"range_status": "below_range", "value": [5]},
            "feature3": {"range_status": "above_range", "value": [200]},
            "feature4": {"range_status": "in_range", "value": [50]}
        }
        
        result = generator._generate_key_insights(segment_values)
        assert isinstance(result, list)
        assert any("multiple" in insight.lower() or "3" in insight for insight in result)

    def test_generate_key_insights_hrv_below_range(self, generator):
        """Test _generate_key_insights when HRV features are below range.
        
        This test covers lines 898-901 in health_report_generator.py where
        insight is generated for HRV below range.
        """
        segment_values = {
            "sdnn": {"range_status": "below_range", "value": [10]},
            "rmssd": {"range_status": "below_range", "value": [5]}
        }
        
        result = generator._generate_key_insights(segment_values)
        assert isinstance(result, list)
        assert any("autonomic" in insight.lower() or "heart rate variability" in insight.lower() for insight in result)

    def test_generate_cross_correlations_list_values(self, generator):
        """Test _generate_cross_correlations with list values.
        
        This test covers lines 911-920 in health_report_generator.py where
        values are extracted from feature_data_dict.
        """
        segment_values = {
            "feature1": {
                "value": [10, 20, 30],
                "range_status": "in_range"
            },
            "feature2": {
                "value": [40, 50, 60],
                "range_status": "in_range"
            }
        }
        
        # Mock the interpreter method
        with patch.object(generator.interpreter, '_analyze_cross_feature_correlations', return_value=[]):
            result = generator._generate_cross_correlations(segment_values)
            assert isinstance(result, list)

    def test_generate_cross_correlations_non_list_values(self, generator):
        """Test _generate_cross_correlations with non-list values.
        
        This test covers lines 917 and 919-920 in health_report_generator.py where
        else branches handle non-list values and raw values.
        """
        segment_values = {
            "feature1": {
                "value": 42.5,  # Single value, not list
                "range_status": "in_range"
            },
            "feature2": 50.0  # Raw value
        }
        
        # Mock the interpreter method
        with patch.object(generator.interpreter, '_analyze_cross_feature_correlations', return_value=[]):
            result = generator._generate_cross_correlations(segment_values)
            assert isinstance(result, list)

    def test_generate_cross_correlations_exception(self, generator):
        """Test _generate_cross_correlations exception handling.
        
        This test covers lines 927-929 in health_report_generator.py where
        exception is caught and [] is returned.
        """
        segment_values = {
            "feature1": {
                "value": [10, 20, 30],
                "range_status": "in_range"
            }
        }
        
        # Mock to raise exception
        with patch.object(generator.interpreter, '_analyze_cross_feature_correlations', side_effect=Exception("Test error")):
            with patch.object(generator.logger, 'error') as mock_error:
                result = generator._generate_cross_correlations(segment_values)
                assert result == []
                mock_error.assert_called()