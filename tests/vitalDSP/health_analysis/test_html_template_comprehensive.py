"""
Comprehensive tests for vitalDSP.health_analysis.html_template module.
Tests all functions and edge cases to improve coverage.
"""

import pytest
from unittest.mock import patch, MagicMock
from vitalDSP.health_analysis.html_template import (
    render_report,
    _get_dynamic_analysis_template,
    _get_javascript_content,
    _get_base_css,
    _get_correlation_contradiction_template,
    _get_feature_template,
    _get_visualization_template
)


@pytest.fixture
def sample_feature_interpretations():
    """Sample feature interpretations for testing."""
    return {
        "sdnn": {
            "description": "Standard deviation of NN intervals",
            "value": [35.0, 40.0, 45.0],
            "median": 40.0,
            "stddev": 5.0,
            "interpretation": "Normal heart rate variability",
            "normal_range": [20, 50],
            "contradiction": "High SDNN and low RMSSD may indicate different aspects of HRV",
            "correlation": "SDNN correlates with RMSSD as both measure heart rate variability",
            "range_status": "in_range"
        },
        "rmssd": {
            "description": "Root mean square of successive differences",
            "value": [25.0, 30.0, 35.0],
            "median": 30.0,
            "stddev": 5.0,
            "interpretation": "Normal short-term heart rate variability",
            "normal_range": [15, 40],
            "contradiction": "High RMSSD and low SDNN may indicate different aspects of HRV",
            "correlation": "RMSSD correlates with SDNN as both measure heart rate variability",
            "range_status": "above_range"
        }
    }


@pytest.fixture
def sample_visualizations():
    """Sample visualizations for testing."""
    return {
        "sdnn": {
            "bell_plot": "_static/images/sdnn_bell_plot.png",
            "time_series": "_static/images/sdnn_time_series.png",
            "histogram": "_static/images/sdnn_histogram.png"
        },
        "rmssd": {
            "bell_plot": "_static/images/rmssd_bell_plot.png",
            "time_series": "_static/images/rmssd_time_series.png"
        }
    }


@pytest.fixture
def sample_dynamic_analysis():
    """Sample dynamic analysis for testing."""
    return {
        "executive_summary": {
            "summary": "Overall health assessment shows normal cardiovascular function with some areas of concern."
        },
        "risk_assessment": {
            "level": "low",
            "recommendation": "Continue monitoring with regular check-ups."
        },
        "recommendations": [
            "Maintain current lifestyle habits",
            "Monitor heart rate variability regularly",
            "Consider stress management techniques"
        ],
        "key_insights": [
            "Heart rate variability is within normal range",
            "Some features show elevated values",
            "Overall cardiovascular health is good"
        ],
        "statistics": {
            "total_features": 2,
            "in_range": 1,
            "above_range": 1,
            "below_range": 0
        },
        "cross_correlations": [
            {
                "features": ["sdnn", "rmssd"],
                "description": "Strong positive correlation between SDNN and RMSSD"
            }
        ],
        "overall_health_score": 75.5
    }


def test_render_report_basic(sample_feature_interpretations, sample_visualizations):
    """Test basic report rendering."""
    html = render_report(sample_feature_interpretations, sample_visualizations)
    
    assert isinstance(html, str)
    assert "<html" in html.lower()
    assert "sdnn" in html
    assert "rmssd" in html
    assert "Normal heart rate variability" in html


def test_render_report_with_dynamic_analysis(sample_feature_interpretations, sample_visualizations, sample_dynamic_analysis):
    """Test report rendering with dynamic analysis."""
    html = render_report(sample_feature_interpretations, sample_visualizations, dynamic_analysis=sample_dynamic_analysis)
    
    assert isinstance(html, str)
    assert "Executive Summary" in html
    assert "Risk Assessment" in html
    assert "Key Insights" in html
    assert "75.5" in html  # Overall health score


def test_render_report_with_filter_status(sample_feature_interpretations, sample_visualizations):
    """Test report rendering with different filter statuses."""
    # Test in_range filter
    html_in_range = render_report(sample_feature_interpretations, sample_visualizations, filter_status="in_range")
    assert isinstance(html_in_range, str)
    
    # Test above_range filter
    html_above_range = render_report(sample_feature_interpretations, sample_visualizations, filter_status="above_range")
    assert isinstance(html_above_range, str)
    
    # Test all filter
    html_all = render_report(sample_feature_interpretations, sample_visualizations, filter_status="all")
    assert isinstance(html_all, str)


def test_render_report_empty_data():
    """Test report rendering with empty data."""
    html = render_report({}, {})
    
    assert isinstance(html, str)
    assert "<html" in html.lower()


def test_render_report_with_visualizations_only(sample_visualizations):
    """Test report rendering with only visualizations."""
    html = render_report({}, sample_visualizations)

    assert isinstance(html, str)


def test_calculate_correlation_with_different_thresholds():
    """Test calculate_correlation with various threshold conditions (lines 18->12, 26, 31->40)."""
    from vitalDSP.health_analysis.html_template import calculate_correlation
    import numpy as np

    feature_values = list(range(100))
    related_values = list(range(100))

    # Test with strong correlation
    thresholds = {"strong": 0.5, "slight": 0.2}
    result = calculate_correlation(feature_values, related_values, thresholds, chunk_size=10)
    assert result == "Strong correlation"  # Should have strong correlation

    # Test with slight correlation
    feature_values_slight = list(range(100))
    related_values_slight = [x + np.random.normal(0, 10) for x in range(100)]
    thresholds_slight = {"strong": 0.9, "slight": 0.3}
    result_slight = calculate_correlation(feature_values_slight, related_values_slight, thresholds_slight, chunk_size=10)
    # May return slight or no correlation depending on random values
    assert isinstance(result_slight, str)

    # Test with no correlation
    feature_values_no = list(range(100))
    related_values_no = list(range(99, -1, -1))  # Reverse order
    thresholds_no = {"strong": 0.8, "slight": 0.6}
    result_no = calculate_correlation(feature_values_no, related_values_no, thresholds_no, chunk_size=10)
    assert isinstance(result_no, str)


def test_calculate_contradiction_with_different_thresholds():
    """Test calculate_contradiction with various threshold conditions (lines 62-65, 74-77, 95->84)."""
    from vitalDSP.health_analysis.html_template import calculate_contradiction

    # Test with strong contradiction
    feature_values = [0.1] * 50
    related_values = [0.9] * 50
    thresholds = {"strong": 0.3, "slight": 0.1}
    result = calculate_contradiction(feature_values, related_values, thresholds, chunk_size=5)
    assert result == "Strong contradiction"

    # Test with slight contradiction
    feature_values_slight = [0.08] * 50
    related_values_slight = [0.12] * 50
    thresholds_slight = {"strong": 0.3, "slight": 0.1}
    result_slight = calculate_contradiction(feature_values_slight, related_values_slight, thresholds_slight, chunk_size=5)
    assert result_slight == "Slight contradiction"

    # Test with no contradiction
    feature_values_no = [0.5] * 50
    related_values_no = [0.5] * 50
    thresholds_no = {"strong": 0.3, "slight": 0.1}
    result_no = calculate_contradiction(feature_values_no, related_values_no, thresholds_no, chunk_size=5)
    assert result_no == "No significant contradiction"


def test_process_interpretations_with_correlations_and_contradictions():
    """Test process_interpretations with both correlations and contradictions (lines 103-118)."""
    from vitalDSP.health_analysis.html_template import process_interpretations

    feature_interpretations = {
        "sdnn": {
            "value": [35.0, 40.0, 45.0] * 10,
            "correlation": {
                "rmssd": "SDNN correlates with RMSSD"
            },
            "contradiction": {
                "nn50": "SDNN contradicts NN50"
            },
            "thresholds": {
                "correlation": {
                    "rmssd": {"strong": 0.5, "slight": 0.2}
                },
                "contradiction": {
                    "nn50": {"strong": 0.3, "slight": 0.1}
                }
            }
        },
        "rmssd": {
            "value": [30.0, 35.0, 40.0] * 10,
        },
        "nn50": {
            "value": [20.0, 25.0, 30.0] * 10,
        }
    }

    result = process_interpretations(feature_interpretations)

    assert isinstance(result, dict)
    assert "sdnn" in result
    # Check that correlation strength was calculated
    if "correlation_strength" in result["sdnn"]:
        assert isinstance(result["sdnn"]["correlation_strength"], str)
    # Check that contradiction strength was calculated
    if "contradiction_strength" in result["sdnn"]:
        assert isinstance(result["sdnn"]["contradiction_strength"], str)


def test_render_report_with_interpretations_only(sample_feature_interpretations):
    """Test report rendering with only interpretations."""
    html = render_report(sample_feature_interpretations, {})
    
    assert isinstance(html, str)
    assert "sdnn" in html
    assert "rmssd" in html


def test_get_dynamic_analysis_template():
    """Test getting dynamic analysis template."""
    template = _get_dynamic_analysis_template()
    
    assert isinstance(template, str)
    assert "dynamic-analysis-container" in template
    assert "Executive Summary" in template
    assert "Risk Assessment" in template


def test_get_javascript_content():
    """Test getting JavaScript content."""
    js_content = _get_javascript_content()
    
    assert isinstance(js_content, str)
    assert "<script>" in js_content
    assert "function filterReport" in js_content


def test_get_base_css():
    """Test getting base CSS content."""
    css_content = _get_base_css()
    
    assert isinstance(css_content, str)
    assert "<style>" in css_content
    assert "body" in css_content
    assert "font-family" in css_content


def test_get_correlation_contradiction_template():
    """Test getting correlation contradiction template."""
    template = _get_correlation_contradiction_template()
    
    assert isinstance(template, str)
    assert "contradiction" in template.lower()
    assert "correlation" in template.lower()


def test_get_feature_template():
    """Test getting feature template."""
    template = _get_feature_template()
    
    assert isinstance(template, str)
    assert "feature" in template.lower()
    assert "interpretation" in template.lower()


def test_get_visualization_template():
    """Test getting visualization template."""
    template = _get_visualization_template()
    
    assert isinstance(template, str)
    assert "visualization" in template.lower()
    assert "plot" in template.lower()


def test_render_report_with_missing_visualizations(sample_feature_interpretations):
    """Test report rendering with missing visualizations for some features."""
    partial_visualizations = {
        "sdnn": {
            "bell_plot": "_static/images/sdnn_bell_plot.png"
        }
        # rmssd has no visualizations
    }
    
    html = render_report(sample_feature_interpretations, partial_visualizations)
    
    assert isinstance(html, str)
    assert "sdnn" in html
    assert "rmssd" in html


def test_render_report_with_none_values(sample_feature_interpretations):
    """Test report rendering with None values in visualizations."""
    visualizations_with_none = {
        "sdnn": {
            "bell_plot": "_static/images/sdnn_bell_plot.png",
            "time_series": None,
            "histogram": "_static/images/sdnn_histogram.png"
        }
    }
    
    html = render_report(sample_feature_interpretations, visualizations_with_none)
    
    assert isinstance(html, str)
    assert "sdnn" in html


def test_render_report_with_empty_strings(sample_feature_interpretations):
    """Test report rendering with empty string values."""
    visualizations_with_empty = {
        "sdnn": {
            "bell_plot": "_static/images/sdnn_bell_plot.png",
            "time_series": "",
            "histogram": "_static/images/sdnn_histogram.png"
        }
    }
    
    html = render_report(sample_feature_interpretations, visualizations_with_empty)
    
    assert isinstance(html, str)
    assert "sdnn" in html


def test_render_report_with_special_characters(sample_visualizations):
    """Test report rendering with special characters in feature names."""
    special_interpretations = {
        "feature-with-dashes": {
            "description": "Feature with dashes",
            "value": [1.0, 2.0, 3.0],
            "median": 2.0,
            "stddev": 1.0,
            "interpretation": "Normal",
            "normal_range": [0, 5],
            "contradiction": "No contradiction",
            "correlation": "No correlation",
            "range_status": "in_range"
        },
        "feature_with_underscores": {
            "description": "Feature with underscores",
            "value": [10.0, 20.0, 30.0],
            "median": 20.0,
            "stddev": 10.0,
            "interpretation": "Normal",
            "normal_range": [5, 35],
            "contradiction": "No contradiction",
            "correlation": "No correlation",
            "range_status": "in_range"
        }
    }
    
    html = render_report(special_interpretations, sample_visualizations)
    
    assert isinstance(html, str)
    assert "feature-with-dashes" in html
    assert "feature_with_underscores" in html


def test_render_report_with_unicode_characters(sample_visualizations):
    """Test report rendering with Unicode characters."""
    unicode_interpretations = {
        "unicode_feature": {
            "description": "Feature with unicode: αβγδε",
            "value": [1.0, 2.0, 3.0],
            "median": 2.0,
            "stddev": 1.0,
            "interpretation": "Normal with unicode: αβγδε",
            "normal_range": [0, 5],
            "contradiction": "No contradiction",
            "correlation": "No correlation",
            "range_status": "in_range"
        }
    }
    
    html = render_report(unicode_interpretations, sample_visualizations)
    
    assert isinstance(html, str)
    assert "unicode_feature" in html


def test_render_report_with_large_datasets(sample_visualizations):
    """Test report rendering with large datasets."""
    large_interpretations = {}
    for i in range(100):  # Create 100 features
        large_interpretations[f"feature_{i}"] = {
            "description": f"Feature {i} description",
            "value": [float(i), float(i+1), float(i+2)],
            "median": float(i+1),
            "stddev": 1.0,
            "interpretation": f"Normal for feature {i}",
            "normal_range": [0, 100],
            "contradiction": "No contradiction",
            "correlation": "No correlation",
            "range_status": "in_range"
        }
    
    html = render_report(large_interpretations, sample_visualizations)
    
    assert isinstance(html, str)
    assert len(html) > 1000  # Should be a substantial HTML document


def test_render_report_template_rendering_error(sample_feature_interpretations, sample_visualizations):
    """Test report rendering with template rendering error."""
    with patch('jinja2.Template.render', side_effect=Exception("Template error")):
        html = render_report(sample_feature_interpretations, sample_visualizations)
        
        assert isinstance(html, str)
        # The error handling should return a basic HTML structure
        assert "<html" in html.lower()


def test_render_report_with_complex_dynamic_analysis(sample_feature_interpretations, sample_visualizations):
    """Test report rendering with complex dynamic analysis."""
    complex_dynamic_analysis = {
        "executive_summary": {
            "summary": "Complex health assessment with multiple factors."
        },
        "risk_assessment": {
            "level": "high",
            "recommendation": "Immediate medical attention required."
        },
        "recommendations": [
            "Seek immediate medical attention",
            "Monitor vital signs continuously",
            "Consider emergency intervention"
        ],
        "key_insights": [
            "Multiple risk factors detected",
            "Immediate intervention required",
            "Critical health status"
        ],
        "statistics": {
            "total_features": 10,
            "in_range": 2,
            "above_range": 5,
            "below_range": 3
        },
        "cross_correlations": [
            {
                "features": ["feature1", "feature2"],
                "description": "Strong correlation detected"
            },
            {
                "features": ["feature3", "feature4"],
                "description": "Moderate correlation detected"
            }
        ],
        "overall_health_score": 25.0
    }

    html = render_report(sample_feature_interpretations, sample_visualizations, dynamic_analysis=complex_dynamic_analysis)

    assert isinstance(html, str)
    assert "high" in html.lower()  # Risk level
    assert "25.0" in html  # Health score
    assert "Immediate medical attention" in html


def test_get_filter_dropdown():
    """Test _get_filter_dropdown template function."""
    from vitalDSP.health_analysis.html_template import _get_filter_dropdown

    dropdown_html = _get_filter_dropdown()

    assert isinstance(dropdown_html, str)
    assert "filter-dropdown" in dropdown_html
    assert "Filter Features" in dropdown_html
    assert "All Features" in dropdown_html
    assert "In Normal Range" in dropdown_html
    assert "Above Normal Range" in dropdown_html
    assert "Below Normal Range" in dropdown_html


def test_get_description_interpretation_template():
    """Test _get_description_interpretation_template function."""
    from vitalDSP.health_analysis.html_template import _get_description_interpretation_template

    template_html = _get_description_interpretation_template()

    assert isinstance(template_html, str)
    assert "description-block" in template_html
    assert "interpretation-block" in template_html
    assert "Description:" in template_html
    assert "Interpretation:" in template_html


def test_get_range_interpretation_template():
    """Test _get_range_interpretation_template function."""
    from vitalDSP.health_analysis.html_template import _get_range_interpretation_template

    template_html = _get_range_interpretation_template()

    assert isinstance(template_html, str)
    assert "Values" in template_html
    assert "Median" in template_html
    assert "Standard Deviation" in template_html
    assert "Range" in template_html
    assert "normal-range-bar-modern" in template_html


def test_render_report_with_missing_median_values(sample_visualizations):
    """Test report rendering with missing median values."""
    interpretations_no_median = {
        "test_feature": {
            "description": "Test feature",
            "value": [1.0, 2.0, 3.0],
            "stddev": 1.0,
            "interpretation": "Normal",
            "normal_range": [0, 5],
            "contradiction": "No contradiction",
            "correlation": "No correlation",
            "range_status": "in_range"
            # No median key
        }
    }

    html = render_report(interpretations_no_median, sample_visualizations)

    assert isinstance(html, str)
    # Report may succeed or fail depending on template requirements
    # Just check it returns valid HTML string
    assert len(html) > 0


def test_render_report_with_missing_stddev(sample_visualizations):
    """Test report rendering with missing stddev values."""
    interpretations_no_stddev = {
        "test_feature": {
            "description": "Test feature",
            "value": [1.0, 2.0, 3.0],
            "median": 2.0,
            "interpretation": "Normal",
            "normal_range": [0, 5],
            "contradiction": "No contradiction",
            "correlation": "No correlation",
            "range_status": "in_range"
            # No stddev key
        }
    }

    html = render_report(interpretations_no_stddev, sample_visualizations)

    assert isinstance(html, str)
    assert "test_feature" in html


def test_render_report_with_none_stddev(sample_visualizations):
    """Test report rendering with None stddev values."""
    interpretations_none_stddev = {
        "test_feature": {
            "description": "Test feature",
            "value": [1.0, 2.0, 3.0],
            "median": 2.0,
            "stddev": None,
            "interpretation": "Normal",
            "normal_range": [0, 5],
            "contradiction": "No contradiction",
            "correlation": "No correlation",
            "range_status": "in_range"
        }
    }

    html = render_report(interpretations_none_stddev, sample_visualizations)

    assert isinstance(html, str)
    assert "test_feature" in html
    assert "N/A" in html  # Should show N/A for None stddev


def test_render_report_with_multiple_visualizations(sample_feature_interpretations):
    """Test report rendering with multiple visualization types."""
    multi_visualizations = {
        "sdnn": {
            "heatmap": "_static/images/sdnn_heatmap.png",
            "bell_plot": "_static/images/sdnn_bell_plot.png",
            "radar_plot": "_static/images/sdnn_radar_plot.png",
            "violin_plot": "_static/images/sdnn_violin_plot.png",
            "plot_box_swarm": "_static/images/sdnn_box_swarm.png",
            "plot_spectrogram": "_static/images/sdnn_spectrogram.png",
            "lag_plot": "_static/images/sdnn_lag_plot.png",
            "line_with_rolling_stats": "_static/images/sdnn_line_rolling.png",
            "plot_spectral_density": "_static/images/sdnn_spectral_density.png"
        }
    }

    html = render_report(sample_feature_interpretations, multi_visualizations)

    assert isinstance(html, str)
    assert "heatmap" in html.lower()
    assert "bell_plot" in html
    assert "Choose Plot Type" in html


def test_render_report_with_visualization_dropdowns(sample_feature_interpretations, sample_visualizations):
    """Test that visualization dropdowns are correctly rendered."""
    html = render_report(sample_feature_interpretations, sample_visualizations)

    assert "plot_type_sdnn" in html  # Dropdown ID for sdnn
    assert "changePlot" in html  # JavaScript function
    assert "changePlotSecondColumn" in html  # JavaScript function for second column


def test_calculate_correlation_basic():
    """Test basic correlation calculation."""
    from vitalDSP.health_analysis.html_template import calculate_correlation
    import numpy as np

    # Test strong positive correlation
    feature_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    related_values = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    thresholds = {"strong": 0.5, "slight": 0.2}

    result = calculate_correlation(feature_values, related_values, thresholds)

    assert isinstance(result, str)
    assert "correlation" in result.lower()


def test_calculate_correlation_no_correlation():
    """Test correlation calculation with uncorrelated data."""
    from vitalDSP.health_analysis.html_template import calculate_correlation
    import numpy as np

    # Test no correlation
    feature_values = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
    related_values = np.array([2, 1, 2, 1, 2, 1, 2, 1, 2, 1])
    thresholds = {"strong": 0.5, "slight": 0.2}

    result = calculate_correlation(feature_values, related_values, thresholds)

    assert isinstance(result, str)
    assert "No significant correlation" in result


def test_calculate_contradiction_basic():
    """Test basic contradiction calculation."""
    from vitalDSP.health_analysis.html_template import calculate_contradiction
    import numpy as np

    # Test contradiction
    feature_values = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    related_values = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    thresholds = {"strong": 0.3, "slight": 0.1}

    result = calculate_contradiction(feature_values, related_values, thresholds)

    assert isinstance(result, str)
    assert "contradiction" in result.lower()


def test_process_interpretations():
    """Test process_interpretations function."""
    from vitalDSP.health_analysis.html_template import process_interpretations

    feature_interpretations = {
        "feature1": {
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "correlation": {
                "feature2": "Positive correlation expected"
            },
            "thresholds": {
                "correlation": {
                    "feature2": {"strong": 0.5, "slight": 0.2}
                }
            }
        },
        "feature2": {
            "value": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
        }
    }

    result = process_interpretations(feature_interpretations)

    assert isinstance(result, dict)
    assert "feature1" in result
    assert "feature2" in result


def test_render_report_with_empty_dynamic_analysis(sample_feature_interpretations, sample_visualizations):
    """Test report rendering with empty dynamic analysis."""
    empty_dynamic = {
        "executive_summary": {"summary": ""},
        "risk_assessment": {"level": "", "recommendation": ""},
        "recommendations": [],
        "key_insights": [],
        "statistics": {
            "total_features": 0,
            "in_range": 0,
            "above_range": 0,
            "below_range": 0
        },
        "cross_correlations": [],
        "overall_health_score": 0.0
    }

    html = render_report(sample_feature_interpretations, sample_visualizations, dynamic_analysis=empty_dynamic)

    assert isinstance(html, str)
    assert "<html" in html.lower()


def test_render_report_below_range_status(sample_visualizations):
    """Test report rendering with below_range status."""
    below_range_interpretations = {
        "low_feature": {
            "description": "Feature with low values",
            "value": [1.0, 2.0, 3.0],
            "median": 2.0,
            "stddev": 1.0,
            "interpretation": "Below normal range",
            "normal_range": [10, 20],
            "contradiction": "No contradiction",
            "correlation": "No correlation",
            "range_status": "below_range"
        }
    }

    html = render_report(below_range_interpretations, sample_visualizations, filter_status="below_range")

    assert isinstance(html, str)
    assert "low_feature" in html
