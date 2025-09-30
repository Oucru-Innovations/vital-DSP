"""
Comprehensive tests for vitalDSP.health_analysis.interpretation_engine module.
Tests all methods and edge cases to improve coverage.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
from vitalDSP.health_analysis.interpretation_engine import InterpretationEngine


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        "sdnn": {
            "normal_range": {
                "1_min": [20, 50],
                "5_min": [50, 130]
            },
            "description": "Standard deviation of NN intervals",
            "interpretation": {
                "in_range": "Normal heart rate variability",
                "below_range": "Low heart rate variability",
                "above_range": "High heart rate variability"
            },
            "contradiction": {
                "rmssd": "High SDNN and low RMSSD may indicate different aspects of HRV"
            },
            "correlation": {
                "rmssd": "SDNN correlates with RMSSD as both measure heart rate variability"
            },
            "thresholds": {
                "correlation": {
                    "rmssd": 0.7
                }
            }
        },
        "rmssd": {
            "normal_range": {
                "1_min": [10, 30],
                "5_min": [20, 100]
            },
            "description": "Root mean square of successive differences",
            "interpretation": {
                "in_range": "Normal short-term heart rate variability",
                "below_range": "Low short-term heart rate variability",
                "above_range": "High short-term heart rate variability"
            },
            "contradiction": {
                "sdnn": "High RMSSD and low SDNN may indicate different aspects of HRV"
            },
            "correlation": {
                "sdnn": "RMSSD correlates with SDNN as both measure heart rate variability"
            }
        }
    }


@pytest.fixture
def engine(mock_config):
    """Create InterpretationEngine instance with mock config."""
    with patch('vitalDSP.health_analysis.interpretation_engine.yaml.safe_load', return_value=mock_config):
        return InterpretationEngine()


def test_interpret_feature_in_range(engine):
    """Test interpreting feature with in-range value."""
    result = engine.interpret_feature("sdnn", 35.0, "1_min")
    
    assert isinstance(result, dict)
    assert "description" in result
    assert "normal_range" in result
    assert "interpretation" in result
    assert "contradiction" in result
    assert "correlation" in result
    assert result["normal_range"] == [20, 50]


def test_interpret_feature_below_range(engine):
    """Test interpreting feature with below-range value."""
    result = engine.interpret_feature("sdnn", 15.0, "1_min")
    
    assert isinstance(result, dict)
    assert "interpretation" in result
    assert "below" in result["interpretation"].lower() or "low" in result["interpretation"].lower()


def test_interpret_feature_above_range(engine):
    """Test interpreting feature with above-range value."""
    result = engine.interpret_feature("sdnn", 60.0, "1_min")
    
    assert isinstance(result, dict)
    assert "interpretation" in result
    assert "above" in result["interpretation"].lower() or "high" in result["interpretation"].lower()


def test_interpret_feature_unknown_feature(engine):
    """Test interpreting unknown feature."""
    result = engine.interpret_feature("unknown_feature", 50.0, "1_min")
    
    assert isinstance(result, dict)
    assert "description" in result
    # Unknown features only return description, not interpretation
    assert "interpretation" not in result


def test_get_range_status_in_range(engine):
    """Test getting range status for in-range value."""
    status = engine.get_range_status("sdnn", 35.0, "1_min")
    assert status == "in_range"


def test_get_range_status_below_range(engine):
    """Test getting range status for below-range value."""
    status = engine.get_range_status("sdnn", 15.0, "1_min")
    assert status == "below_range"


def test_get_range_status_above_range(engine):
    """Test getting range status for above-range value."""
    status = engine.get_range_status("sdnn", 60.0, "1_min")
    assert status == "above_range"


def test_get_range_status_unknown_feature(engine):
    """Test getting range status for unknown feature."""
    status = engine.get_range_status("unknown_feature", 50.0, "1_min")
    assert status == "unknown"


def test_get_feature_contradiction(engine):
    """Test getting feature contradiction."""
    contradiction = engine._get_feature_contradiction("sdnn")
    assert isinstance(contradiction, str)
    assert len(contradiction) > 0


def test_get_feature_contradiction_unknown(engine):
    """Test getting contradiction for unknown feature."""
    contradiction = engine._get_feature_contradiction("unknown_feature")
    assert contradiction == "No contradiction found."


def test_get_feature_correlation(engine):
    """Test getting feature correlation."""
    correlation = engine._get_feature_correlation("sdnn")
    assert isinstance(correlation, str)
    assert len(correlation) > 0


def test_get_feature_correlation_unknown(engine):
    """Test getting correlation for unknown feature."""
    correlation = engine._get_feature_correlation("unknown_feature")
    assert correlation == "No correlation found."


def test_generate_dynamic_interpretation(engine):
    """Test generating dynamic interpretation."""
    interpretation = engine._generate_dynamic_interpretation(
        "sdnn", 35.0, [20, 50], engine.config["sdnn"], "1_min"
    )
    
    assert isinstance(interpretation, str)
    assert len(interpretation) > 0


def test_get_deviation_severity(engine):
    """Test getting deviation severity."""
    # Test in range (deviation = 0)
    severity = engine._get_deviation_severity(0.0)
    assert severity == "mild"
    
    # Test mild deviation (deviation = 0.3)
    severity = engine._get_deviation_severity(0.3)
    assert severity == "mild"
    
    # Test moderate deviation (deviation = 0.7)
    severity = engine._get_deviation_severity(0.7)
    assert severity == "moderate"
    
    # Test severe deviation (deviation = 1.5)
    severity = engine._get_deviation_severity(1.5)
    assert severity == "severe"
    
    # Test critical deviation (deviation = 2.5)
    severity = engine._get_deviation_severity(2.5)
    assert severity == "critical"


def test_get_contextual_interpretation(engine):
    """Test getting contextual interpretation."""
    # Pass a string severity instead of dict
    severity = "optimal"
    interpretation = engine._get_contextual_interpretation(
        "sdnn", 35.0, "normal", severity, "1_min"
    )
    
    assert isinstance(interpretation, str)
    assert len(interpretation) > 0


def test_generate_dynamic_description(engine):
    """Test generating dynamic description."""
    description = engine._generate_dynamic_description(
        "sdnn", 35.0, [20, 50], engine.config["sdnn"], "1_min"
    )
    
    assert isinstance(description, str)
    assert len(description) > 0


def test_generate_dynamic_correlation(engine):
    """Test generating dynamic correlation."""
    correlation = engine._generate_dynamic_correlation(
        "sdnn", 35.0, [20, 50], engine.config["sdnn"], "1_min"
    )
    
    assert isinstance(correlation, str)
    assert len(correlation) > 0


def test_get_correlation_context(engine):
    """Test getting correlation context."""
    context = engine._get_correlation_context(
        "sdnn", 35.0, [20, 50], engine.config["sdnn"]["thresholds"]["correlation"]
    )
    
    assert isinstance(context, str)


def test_analyze_cross_feature_correlations(engine):
    """Test analyzing cross-feature correlations."""
    segment_values = {
        "sdnn": {"value": 35.0, "range_status": "in_range"},
        "rmssd": {"value": 25.0, "range_status": "in_range"}
    }
    
    correlations = engine._analyze_cross_feature_correlations(segment_values, "1_min")
    
    assert isinstance(correlations, list)


def test_analyze_hrv_correlations(engine):
    """Test analyzing HRV correlations."""
    segment_values = {
        "sdnn": {"value": 35.0, "range_status": "in_range"},
        "rmssd": {"value": 25.0, "range_status": "in_range"}
    }
    
    hrv_features = ["sdnn", "rmssd"]
    correlations = engine._analyze_hrv_correlations(hrv_features, segment_values, "1_min")
    
    assert isinstance(correlations, list)


def test_analyze_sdnn_rmssd_correlation(engine):
    """Test analyzing SDNN-RMSSD correlation."""
    correlation = engine._analyze_sdnn_rmssd_correlation(35.0, 25.0, "1_min")
    
    assert correlation is None or isinstance(correlation, dict)


def test_analyze_rmssd_nn50_correlation(engine):
    """Test analyzing RMSSD-NN50 correlation."""
    correlation = engine._analyze_rmssd_nn50_correlation(25.0, 15.0, "1_min")
    
    assert correlation is None or isinstance(correlation, dict)


def test_analyze_sdnn_nn50_correlation(engine):
    """Test analyzing SDNN-NN50 correlation."""
    correlation = engine._analyze_sdnn_nn50_correlation(35.0, 15.0, "1_min")
    
    assert correlation is None or isinstance(correlation, dict)


def test_analyze_other_correlations(engine):
    """Test analyzing other correlations."""
    segment_values = {
        "feature1": {"value": [10.0], "range_status": "in_range"},
        "feature2": {"value": [20.0], "range_status": "in_range"}
    }
    
    correlations = engine._analyze_other_correlations(segment_values, "1_min")
    
    assert isinstance(correlations, list)


def test_get_relative_position(engine):
    """Test getting relative position."""
    position = engine._get_relative_position(35.0, [20, 50])
    assert isinstance(position, float)
    assert 0.0 <= position <= 1.0


def test_interpret_feature_with_different_segment_durations(engine):
    """Test interpreting feature with different segment durations."""
    # Test 1_min
    result_1min = engine.interpret_feature("sdnn", 35.0, "1_min")
    assert isinstance(result_1min, dict)
    
    # Test 5_min
    result_5min = engine.interpret_feature("sdnn", 35.0, "5_min")
    assert isinstance(result_5min, dict)


def test_interpret_feature_with_edge_values(engine):
    """Test interpreting feature with edge values."""
    # Test exact boundary values
    result_low = engine.interpret_feature("sdnn", 20.0, "1_min")
    result_high = engine.interpret_feature("sdnn", 50.0, "1_min")
    
    assert isinstance(result_low, dict)
    assert isinstance(result_high, dict)


def test_interpret_feature_with_nan_values(engine):
    """Test interpreting feature with NaN values."""
    result = engine.interpret_feature("sdnn", float('nan'), "1_min")
    
    assert isinstance(result, dict)
    assert "description" in result


def test_interpret_feature_with_inf_values(engine):
    """Test interpreting feature with infinite values."""
    result = engine.interpret_feature("sdnn", float('inf'), "1_min")
    
    assert isinstance(result, dict)
    assert "description" in result


def test_config_loading_error():
    """Test InterpretationEngine with config loading error."""
    with patch('vitalDSP.health_analysis.interpretation_engine.yaml.safe_load', side_effect=Exception("Config error")):
        with patch('builtins.open', mock_open(read_data="test: data")):
            with pytest.raises(FileNotFoundError, match="Error loading feature_config.yml"):
                engine = InterpretationEngine()


def test_config_file_not_found():
    """Test InterpretationEngine with missing config file."""
    with patch('builtins.open', side_effect=FileNotFoundError("File not found")):
        engine = InterpretationEngine()
        # Should still work with default config
        result = engine.interpret_feature("test", 50.0, "1_min")
        assert isinstance(result, dict)
