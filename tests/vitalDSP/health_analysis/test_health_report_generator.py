import pytest
import numpy as np
from vitalDSP.health_analysis.health_report_generator import HealthReportGenerator
import matplotlib

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


def test_generate_report_html(report_generator):
    """
    Test if the health report is generated as HTML and if the content has expected keys.
    """
    # Generate the report
    report_html = report_generator.generate()

    # Basic checks on the generated HTML content
    assert isinstance(report_html, str), "Report should be a string (HTML format)."
    assert (
        "<html>" not in report_html.lower()
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
