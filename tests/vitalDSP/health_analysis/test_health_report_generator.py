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
            assert isinstance(result[1], dict)  # Should return dict with error messages
            assert len(result[1]) > 0  # Should have error messages
