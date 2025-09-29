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
    assert "<html" in html.lower()


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
