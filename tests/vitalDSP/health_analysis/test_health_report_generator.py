import pytest
from unittest.mock import patch, MagicMock
from vitalDSP.health_analysis.health_report_generator import HealthReportGenerator

@pytest.fixture
def mock_feature_data():
    """Fixture for mock feature data"""
    return {
        "sdnn": 35.0,
        "rmssd": 45.0,
        "pnn50": 28.5,
        "mean_nn": 750.0,
    }

@pytest.fixture
def mock_interpretation():
    """Fixture for mock interpretation data"""
    return {
        "description": "Test description",
        "interpretation": "Test interpretation",
        "normal_range": [20, 60],
        "contradiction": {"test_feature": "Test contradiction"},
        "correlation": {"test_feature": "Test correlation"},
    }

@pytest.fixture
def mock_visualizations():
    """Fixture for mock visualizations"""
    return {
        "sdnn": {
            "heatmap": "visualizations/sdnn_heatmap.png",
            "bell_plot": "visualizations/sdnn_bell_plot.png",
        },
        "rmssd": {
            "heatmap": "visualizations/rmssd_heatmap.png",
            "bell_plot": "visualizations/rmssd_bell_plot.png",
        },
    }

# Test for initialization of HealthReportGenerator
def test_health_report_generator_initialization(mock_feature_data):
    generator = HealthReportGenerator(feature_data=mock_feature_data, segment_duration="1 min")
    assert generator.feature_data == mock_feature_data
    assert generator.segment_duration == "1 min"
    assert isinstance(generator.interpreter, object)  # Check if interpreter was initialized
    assert isinstance(generator.visualizer, object)  # Check if visualizer was initialized

# Test for generating health report with mocked dependencies
@patch("vitalDSP.health_analysis.health_report_generator.InterpretationEngine")
@patch("vitalDSP.health_analysis.health_report_generator.HealthReportVisualizer")
@patch("vitalDSP.health_analysis.health_report_generator.render_report")
def test_generate_health_report(mock_render_report, MockVisualizer, MockInterpreter, mock_feature_data, mock_interpretation, mock_visualizations):
    # Set up mock interpreter and visualizer behavior
    mock_interpreter_instance = MockInterpreter.return_value
    mock_interpreter_instance.interpret_feature.return_value = mock_interpretation
    mock_interpreter_instance.get_range_status.return_value = "in_range"

    mock_visualizer_instance = MockVisualizer.return_value
    mock_visualizer_instance.create_visualizations.return_value = mock_visualizations

    # Mock the render_report function
    mock_render_report.return_value = "<html>Test Report</html>"

    # Initialize the report generator
    generator = HealthReportGenerator(feature_data=mock_feature_data, segment_duration="1 min")

    # Generate the report
    report_html = generator.generate(filter_status="all")

    # Assertions
    mock_interpreter_instance.interpret_feature.assert_called()
    mock_visualizer_instance.create_visualizations.assert_called_once_with(mock_feature_data)
    mock_render_report.assert_called_once()

    assert report_html == "<html>Test Report</html>"

# Test for filtering feature statuses in the health report generation
@patch("vitalDSP.health_analysis.health_report_generator.InterpretationEngine")
@patch("vitalDSP.health_analysis.health_report_generator.HealthReportVisualizer")
@patch("vitalDSP.health_analysis.health_report_generator.render_report")
def test_generate_health_report_filtered(mock_render_report, MockVisualizer, MockInterpreter, mock_feature_data, mock_interpretation, mock_visualizations):
    # Set up mock interpreter and visualizer behavior
    mock_interpreter_instance = MockInterpreter.return_value
    mock_interpreter_instance.interpret_feature.return_value = mock_interpretation
    mock_interpreter_instance.get_range_status.return_value = "below_range"  # Simulate feature out of range

    mock_visualizer_instance = MockVisualizer.return_value
    mock_visualizer_instance.create_visualizations.return_value = mock_visualizations

    # Mock the render_report function
    mock_render_report.return_value = "<html>Test Report</html>"

    # Initialize the report generator
    generator = HealthReportGenerator(feature_data=mock_feature_data, segment_duration="1 min")

    # Generate the report with filter_status="in_range" (should exclude below_range features)
    report_html = generator.generate(filter_status="in_range")

    # Assertions
    mock_interpreter_instance.get_range_status.assert_called()
    assert report_html == "<html>Test Report</html>"

# Test the private method _generate_feature_report
@patch("vitalDSP.health_analysis.health_report_generator.InterpretationEngine")
def test_generate_feature_report(MockInterpreter, mock_feature_data, mock_interpretation):
    mock_interpreter_instance = MockInterpreter.return_value
    mock_interpreter_instance.interpret_feature.return_value = mock_interpretation
    mock_interpreter_instance.config = {  # Mock the config to return a dictionary
        "sdnn": {
            "description": "Standard Deviation of NN intervals",
            "normal_range": [15, 40],
            "interpretation": {
                "in_range": "SDNN is within the normal range.",
                "below_range": "SDNN is below the normal range.",
                "above_range": "SDNN is above the normal range."
            }
        }
    }

    # Initialize the report generator
    generator = HealthReportGenerator(feature_data=mock_feature_data, segment_duration="1 min")

    # Generate the report for a specific feature
    feature_report = generator._generate_feature_report("sdnn", 35.0)

    # Assertions
    assert feature_report["description"] == "Test description"
    assert feature_report["value"] == 35.0
    assert feature_report["interpretation"] == "Test interpretation"
    assert feature_report["normal_range"] == [20, 60]
    assert "contradiction" in feature_report
    assert "correlation" in feature_report
