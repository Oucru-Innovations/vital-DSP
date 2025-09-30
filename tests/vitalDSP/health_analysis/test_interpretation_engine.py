import pytest
import numpy as np
from unittest.mock import patch, mock_open
from vitalDSP.health_analysis.interpretation_engine import InterpretationEngine

# Sample YAML configuration for testing purposes
# Define a sample configuration for mocking
sample_config = {
    "rmssd": {
        "description": "Root Mean Square of the Difference between Successive Peaks.",
        "normal_range": {"1_min": [20, 60], "5_min": [40, 100]},
        "interpretation": {
            "below_range": "Low RMSSD suggests reduced parasympathetic activity.",
            "in_range": "RMSSD is within the normal range.",
            "above_range": "High RMSSD may indicate strong vagal tone.",
        },
        "contradiction": {
            "lf_hf_ratio": "High RMSSD and high LF/HF ratio may suggest imbalance."
        },
        "correlation": {"hf_power": "RMSSD positively correlates with HF power."},
    }
}


@pytest.fixture
def engine_with_mocked_config():
    """Fixture to initialize InterpretationEngine with mocked config."""
    with patch.object(
        InterpretationEngine, "load_feature_config", return_value=sample_config
    ):
        return InterpretationEngine()


def test_yaml_loading_custom_path(engine_with_mocked_config):
    """Test loading of YAML configuration file from a custom path."""
    engine = engine_with_mocked_config
    assert "rmssd" in engine.config


def test_get_range_status_in_range(engine_with_mocked_config):
    """Test getting range status for a value within normal range."""
    engine = engine_with_mocked_config
    status = engine.get_range_status("rmssd", 50, segment_duration="1_min")
    assert status == "in_range"


def test_get_range_status_below_range(engine_with_mocked_config):
    """Test getting range status for a value below normal range."""
    engine = engine_with_mocked_config
    status = engine.get_range_status("rmssd", 10, segment_duration="1_min")
    assert status == "below_range"


def test_get_range_status_above_range(engine_with_mocked_config):
    """Test getting range status for a value above normal range."""
    engine = engine_with_mocked_config
    status = engine.get_range_status("rmssd", 70, segment_duration="1_min")
    assert status == "above_range"


def test_interpret_feature_in_range(engine_with_mocked_config):
    """Test feature interpretation for a value within normal range."""
    engine = engine_with_mocked_config
    interpretation = engine.interpret_feature("rmssd", 50, segment_duration="1_min")
    # Check that the interpretation contains the base interpretation and dynamic context
    assert "RMSSD is within the normal range" in interpretation["interpretation"]
    assert "Your RMSSD of 50.0 ms" in interpretation["interpretation"]


def test_interpret_feature_below_range(engine_with_mocked_config):
    """Test feature interpretation for a value below normal range."""
    engine = engine_with_mocked_config
    interpretation = engine.interpret_feature("rmssd", 10, segment_duration="1_min")
    # Check that the interpretation contains the base interpretation and dynamic context
    assert "Low RMSSD suggests reduced parasympathetic activity" in interpretation["interpretation"]
    assert "Your RMSSD of 10.0 ms" in interpretation["interpretation"]


def test_interpret_feature_above_range(engine_with_mocked_config):
    """Test feature interpretation for a value above normal range."""
    engine = engine_with_mocked_config
    interpretation = engine.interpret_feature("rmssd", 70, segment_duration="1_min")
    # Check that the interpretation contains the base interpretation and dynamic context
    assert "High RMSSD may indicate strong vagal tone" in interpretation["interpretation"]
    assert "Your RMSSD of 70.0 ms" in interpretation["interpretation"]


def test_parse_inf_values(engine_with_mocked_config):
    """Test the parsing of 'inf' and '-inf' values."""
    engine = engine_with_mocked_config
    assert engine._parse_inf_values("inf") == np.inf
    assert engine._parse_inf_values("-inf") == -np.inf
    assert engine._parse_inf_values(50) == 50  # No conversion needed


def test_get_feature_contradiction(engine_with_mocked_config):
    """Test retrieving contradiction for a feature."""
    engine = engine_with_mocked_config
    contradiction = engine._get_feature_contradiction("rmssd")
    assert contradiction == "High RMSSD and high LF/HF ratio may suggest imbalance."


def test_get_feature_correlation(engine_with_mocked_config):
    """Test retrieving correlation for a feature."""
    engine = engine_with_mocked_config
    correlation = engine._get_feature_correlation("rmssd")
    assert correlation == "RMSSD positively correlates with HF power."


def test_get_range_status_unknown_feature(engine_with_mocked_config):
    """Test handling of unknown feature names."""
    engine = engine_with_mocked_config
    status = engine.get_range_status("unknown_feature", 50, segment_duration="1 min")
    assert status == "unknown"


def test_interpret_feature_unknown_feature(engine_with_mocked_config):
    """Test interpretation of an unknown feature."""
    engine = engine_with_mocked_config
    interpretation = engine.interpret_feature(
        "unknown_feature", 50, segment_duration="1 min"
    )
    assert (
        "Feature 'unknown_feature' not found in configuration."
        in interpretation["description"]
    )

# Update Test Cases
# Sample YAML data for testing
sample_yaml = """
NN50:
  normal_range:
    "1 min": [40, 150]
    "5 min": [30, 100]
  interpretation:
    in_range: "Normal parasympathetic activity."
    below_range: "Low parasympathetic activity."
    above_range: "High parasympathetic activity."
  description: "Number of significant changes in heart rate."
  contradiction: "Low NN50 contradicts high RMSSD."
  correlation: "Positively correlated with RMSSD."
"""

def test_load_yaml_success():
    # Test the _load_yaml function success case
    engine = InterpretationEngine()
    with patch("builtins.open", mock_open(read_data=sample_yaml)):
        data = engine._load_yaml("fake_path.yaml")
        assert "NN50" in data
        assert data["NN50"]["normal_range"]["1 min"] == [40, 150]

def test_load_yaml_file_not_found():
    # Test _load_yaml when the file is not found
    engine = InterpretationEngine()
    with pytest.raises(FileNotFoundError, match="YAML file not found: missing.yaml"):
        engine._load_yaml("missing.yaml")

def test_load_yaml_parse_error():
    # Test _load_yaml when the YAML parsing fails
    engine = InterpretationEngine()
    malformed_yaml = "bad_yaml: [this is : not correct"
    with patch("builtins.open", mock_open(read_data=malformed_yaml)):
        with pytest.raises(Exception, match="Error parsing YAML file"):
            engine._load_yaml("fake_path.yaml")

def test_init_custom_yaml():
    # Test __init__ when providing a custom YAML file
    with patch("builtins.open", mock_open(read_data=sample_yaml)):
        engine = InterpretationEngine("fake_path.yaml")
        assert "NN50" in engine.config

def test_load_feature_config_raises_exception():
    # Mocking the open_text function to raise an Exception
    with patch("importlib.resources.open_text", side_effect=Exception("Test error")):
        with pytest.raises(FileNotFoundError, match="Error loading feature_config.yml: Test error"):
            engine = InterpretationEngine()

def test_get_range_status_no_normal_range():
    # Test get_range_status when no normal range is available
    engine = InterpretationEngine()
    engine.config = {
        "NN50": {
            "normal_range": {
                "5 min": [30, 100]  # Only has "5 min", not "1 min"
            }
        }
    }
    assert engine.get_range_status("NN50", 50) == "unknown"

def test_get_range_status_inf_handling():
    # Test get_range_status with '-inf' and 'inf' in normal range
    engine = InterpretationEngine()
    engine.config = {
        "NN50": {
            "normal_range": {
                "1 min": ["-inf", "inf"]
            }
        }
    }
    status = engine.get_range_status("NN50", 50)
    assert status == "in_range"  # Since the value is within -inf to inf

def test_interpret_feature_no_normal_range():
    # Test interpret_feature when no normal range is available
    engine = InterpretationEngine()
    engine.config = {
        "NN50": {
            "normal_range": {}
        }
    }
    result = engine.interpret_feature("NN50", 50)
    assert result == {"normal_range": "Normal range for 'NN50' not available."}

def test_parse_inf_values():
    # Test _parse_inf_values for handling '-inf'
    engine = InterpretationEngine()
    assert engine._parse_inf_values("-inf") == -np.inf
    assert engine._parse_inf_values("inf") == np.inf
    assert engine._parse_inf_values(100) == 100


def test_interpret_multiple_features_basic():
    """Test interpret_multiple_features with basic input."""
    engine = InterpretationEngine()
    feature_data = {"sdnn": 45.0, "rmssd": 30.0}

    result = engine.interpret_multiple_features(feature_data, "1 min")

    assert isinstance(result, dict)
    assert "individual_interpretations" in result
    assert "cross_feature_patterns" in result
    assert "cross_correlations" in result
    assert "warnings" in result
    assert "overall_assessment" in result


def test_interpret_multiple_features_with_exception():
    """Test interpret_multiple_features with exception handling."""
    engine = InterpretationEngine()

    # Pass invalid data to trigger exception
    result = engine.interpret_multiple_features({}, "1 min")

    assert isinstance(result, dict)
    assert "individual_interpretations" in result
    assert "overall_assessment" in result


def test_analyze_cross_feature_correlations():
    """Test _analyze_cross_feature_correlations method."""
    engine = InterpretationEngine()
    feature_data = {"sdnn": 45.0, "rmssd": 30.0, "nn50": 40.0}

    correlations = engine._analyze_cross_feature_correlations(feature_data, "1 min")

    assert isinstance(correlations, list)


def test_analyze_hrv_correlations():
    """Test _analyze_hrv_correlations method."""
    engine = InterpretationEngine()
    hrv_features = ["sdnn", "rmssd", "nn50"]
    feature_data = {"sdnn": 45.0, "rmssd": 30.0, "nn50": 40.0}

    correlations = engine._analyze_hrv_correlations(hrv_features, feature_data, "1 min")

    assert isinstance(correlations, list)


def test_analyze_sdnn_rmssd_correlation():
    """Test _analyze_sdnn_rmssd_correlation method."""
    engine = InterpretationEngine()

    # Test strong positive correlation
    correlation = engine._analyze_sdnn_rmssd_correlation(45.0, 30.0, "1 min")

    assert isinstance(correlation, dict)
    assert "features" in correlation
    assert "type" in correlation
    assert "description" in correlation
    assert "strength" in correlation


def test_analyze_rmssd_nn50_correlation():
    """Test _analyze_rmssd_nn50_correlation method."""
    engine = InterpretationEngine()

    correlation = engine._analyze_rmssd_nn50_correlation(30.0, 40.0, "1 min")

    assert isinstance(correlation, dict)
    assert "features" in correlation
    assert "type" in correlation
    assert "description" in correlation
    assert "strength" in correlation


def test_analyze_sdnn_nn50_correlation():
    """Test _analyze_sdnn_nn50_correlation method."""
    engine = InterpretationEngine()

    correlation = engine._analyze_sdnn_nn50_correlation(45.0, 40.0, "1 min")

    assert isinstance(correlation, dict)
    assert "features" in correlation
    assert "type" in correlation
    assert "description" in correlation
    assert "strength" in correlation


def test_get_relative_position():
    """Test _get_relative_position method."""
    engine = InterpretationEngine()

    # Test value in middle of range
    position = engine._get_relative_position(50, [0, 100])
    assert position == 0.5

    # Test value at start
    position = engine._get_relative_position(0, [0, 100])
    assert position == 0.0

    # Test value at end
    position = engine._get_relative_position(100, [0, 100])
    assert position == 1.0

    # Test equal min and max
    position = engine._get_relative_position(50, [50, 50])
    assert position == 0.5


def test_analyze_cross_feature_patterns():
    """Test _analyze_cross_feature_patterns method."""
    engine = InterpretationEngine()

    # Test with multiple out-of-range HRV features
    feature_data = {"sdnn": 10.0, "rmssd": 5.0, "nn50": 2.0}
    patterns = engine._analyze_cross_feature_patterns(feature_data, "1 min")

    assert isinstance(patterns, list)


def test_generate_cross_feature_warnings():
    """Test _generate_cross_feature_warnings method."""
    engine = InterpretationEngine()

    # Test with contradictory patterns
    feature_data = {"sdnn": 45.0, "rmssd": 10.0}
    warnings = engine._generate_cross_feature_warnings(feature_data, "1 min")

    assert isinstance(warnings, list)


def test_generate_cross_feature_warnings_extreme_values():
    """Test _generate_cross_feature_warnings with extreme values."""
    engine = InterpretationEngine()

    # Test with extremely low value
    feature_data = {"sdnn": 1.0}
    warnings = engine._generate_cross_feature_warnings(feature_data, "1 min")

    assert isinstance(warnings, list)
    # Warnings may or may not be generated depending on configuration
    # Just verify it returns a list


def test_generate_overall_assessment():
    """Test _generate_overall_assessment method."""
    engine = InterpretationEngine()

    # Test with all in-range
    interpretations = {
        "feature1": {"range_status": "in_range"},
        "feature2": {"range_status": "in_range"}
    }
    assessment = engine._generate_overall_assessment(interpretations)
    assert "excellent" in assessment.lower()

    # Test with empty interpretations
    assessment = engine._generate_overall_assessment({})
    assert "No features" in assessment


def test_get_correlation_context():
    """Test _get_correlation_context method."""
    engine = InterpretationEngine()

    # Test below normal range
    context = engine._get_correlation_context("sdnn", 10.0, [20, 50], {})
    assert isinstance(context, str)

    # Test above normal range
    context = engine._get_correlation_context("sdnn", 60.0, [20, 50], {})
    assert isinstance(context, str)

    # Test within optimal range
    context = engine._get_correlation_context("sdnn", 35.0, [20, 50], {})
    assert isinstance(context, str)


def test_generate_dynamic_correlation():
    """Test _generate_dynamic_correlation method."""
    engine = InterpretationEngine()
    feature_info = {
        "correlation": "Base correlation info",
        "thresholds": {"correlation": {}}
    }

    correlation = engine._generate_dynamic_correlation(
        "sdnn", 45.0, [20, 50], feature_info, "1 min"
    )

    assert isinstance(correlation, str)
    assert "Base correlation info" in correlation


def test_generate_dynamic_interpretation():
    """Test _generate_dynamic_interpretation method."""
    engine = InterpretationEngine()
    feature_info = {
        "interpretation": {
            "below_range": "Below normal",
            "in_range": "Normal",
            "above_range": "Above normal"
        }
    }

    # Test below range
    interpretation = engine._generate_dynamic_interpretation(
        "sdnn", 10.0, [20, 50], feature_info, "1 min"
    )
    assert isinstance(interpretation, str)

    # Test in range
    interpretation = engine._generate_dynamic_interpretation(
        "sdnn", 35.0, [20, 50], feature_info, "1 min"
    )
    assert isinstance(interpretation, str)

    # Test above range
    interpretation = engine._generate_dynamic_interpretation(
        "sdnn", 60.0, [20, 50], feature_info, "1 min"
    )
    assert isinstance(interpretation, str)


def test_get_deviation_severity():
    """Test _get_deviation_severity method."""
    engine = InterpretationEngine()

    assert engine._get_deviation_severity(0.3) == "mild"
    assert engine._get_deviation_severity(0.7) == "moderate"
    assert engine._get_deviation_severity(1.5) == "severe"
    assert engine._get_deviation_severity(2.5) == "critical"


def test_get_contextual_interpretation():
    """Test _get_contextual_interpretation method."""
    engine = InterpretationEngine()

    # Test for sdnn below range
    context = engine._get_contextual_interpretation(
        "sdnn", 10.0, "below", "critical", "1 min"
    )
    assert isinstance(context, str)
    assert "sdnn" in context.lower() or "SDNN" in context

    # Test for rmssd above range
    context = engine._get_contextual_interpretation(
        "rmssd", 100.0, "above", "severe", "1 min"
    )
    assert isinstance(context, str)


def test_generate_dynamic_description():
    """Test _generate_dynamic_description method."""
    engine = InterpretationEngine()
    feature_info = {"description": "Test feature description"}

    # Test below range
    description = engine._generate_dynamic_description(
        "test", 5.0, [10, 20], feature_info, "1 min"
    )
    assert isinstance(description, str)
    assert "Test feature description" in description

    # Test above range
    description = engine._generate_dynamic_description(
        "test", 25.0, [10, 20], feature_info, "1 min"
    )
    assert isinstance(description, str)

    # Test in optimal range
    description = engine._generate_dynamic_description(
        "test", 15.0, [10, 20], feature_info, "1 min"
    )
    assert isinstance(description, str)


def test_analyze_other_correlations():
    """Test _analyze_other_correlations method."""
    engine = InterpretationEngine()
    feature_data = {"feature1": 10.0, "feature2": 20.0}

    correlations = engine._analyze_other_correlations(feature_data, "1 min")

    assert isinstance(correlations, list)


def test_interpret_feature_with_different_segment_durations():
    """Test interpret_feature with different segment durations."""
    engine = InterpretationEngine()

    # Test with 1 min
    result_1min = engine.interpret_feature("sdnn", 45.0, "1 min")
    assert isinstance(result_1min, dict)

    # Test with 5 min
    result_5min = engine.interpret_feature("sdnn", 45.0, "5 min")
    assert isinstance(result_5min, dict)


def test_get_range_status_with_edge_values():
    """Test get_range_status with edge values."""
    engine = InterpretationEngine()
    engine.config = {
        "test_feature": {
            "normal_range": {"1 min": [10, 20]}
        }
    }

    # Test exact min
    status = engine.get_range_status("test_feature", 10, "1 min")
    assert status == "in_range"

    # Test exact max
    status = engine.get_range_status("test_feature", 20, "1 min")
    assert status == "in_range"

    # Test just below min
    status = engine.get_range_status("test_feature", 9.99, "1 min")
    assert status == "below_range"

    # Test just above max
    status = engine.get_range_status("test_feature", 20.01, "1 min")
    assert status == "above_range"


def test_get_feature_contradiction_with_string():
    """Test _get_feature_contradiction with string contradiction."""
    engine = InterpretationEngine()
    engine.config = {
        "test_feature": {
            "contradiction": "String contradiction"
        }
    }

    contradiction = engine._get_feature_contradiction("test_feature")
    assert contradiction == "String contradiction"


def test_get_feature_correlation_with_string():
    """Test _get_feature_correlation with string correlation."""
    engine = InterpretationEngine()
    engine.config = {
        "test_feature": {
            "correlation": "String correlation"
        }
    }

    correlation = engine._get_feature_correlation("test_feature")
    assert correlation == "String correlation"


def test_interpret_multiple_features_with_cardiovascular_risk():
    """Test interpret_multiple_features detecting cardiovascular risk."""
    engine = InterpretationEngine()

    # Create data that should trigger cardiovascular risk pattern
    feature_data = {"sdnn": 10.0, "heart_rate": 100.0}
    result = engine.interpret_multiple_features(feature_data, "1 min")

    assert isinstance(result, dict)
    assert "cross_feature_patterns" in result


def test_interpret_multiple_features_with_single_feature():
    """Test interpret_multiple_features with single feature."""
    engine = InterpretationEngine()
    feature_data = {"sdnn": 45.0}

    result = engine.interpret_multiple_features(feature_data, "1 min")

    assert isinstance(result, dict)
    assert "individual_interpretations" in result
    assert "sdnn" in result["individual_interpretations"]


def test_parse_inf_values_with_lower_case():
    """Test _parse_inf_values handles lowercase inf/-inf (lines 220->222)."""
    engine = InterpretationEngine()

    # Test lowercase
    assert engine._parse_inf_values("inf") == np.inf
    assert engine._parse_inf_values("-inf") == -np.inf

    # Test uppercase
    assert engine._parse_inf_values("INF") == np.inf
    assert engine._parse_inf_values("-INF") == -np.inf

    # Test mixed case
    assert engine._parse_inf_values("Inf") == np.inf
    assert engine._parse_inf_values("-Inf") == -np.inf


def test_analyze_hrv_correlations_with_dict_values():
    """Test _analyze_hrv_correlations handles dict values (lines 411-421)."""
    engine = InterpretationEngine()

    # Test with dict values containing 'value' key
    feature_data = {
        "sdnn": {"value": 45.0, "other": "data"},
        "rmssd": {"value": 35.0, "other": "data"}
    }
    hrv_features = ["sdnn", "rmssd"]

    correlations = engine._analyze_hrv_correlations(hrv_features, feature_data, "1_min")

    assert isinstance(correlations, list)
    # Should have analyzed SDNN-RMSSD correlation
    assert len(correlations) > 0


def test_interpret_multiple_features_with_exception():
    """Test interpret_multiple_features exception handling (lines 632-633)."""
    engine = InterpretationEngine()

    # Mock interpret_feature to raise exception
    with patch.object(engine, 'interpret_feature', side_effect=Exception("Test error")):
        feature_data = {"sdnn": 45.0}
        result = engine.interpret_multiple_features(feature_data, "1 min")

        # Should return error dict
        assert isinstance(result, dict)
        assert "individual_interpretations" in result
        assert "warnings" in result
        assert len(result["warnings"]) > 0


def test_analyze_cross_feature_patterns_autonomic_dysfunction():
    """Test _analyze_cross_feature_patterns detects autonomic dysfunction (lines 655-671)."""
    engine = InterpretationEngine()

    # Create data with multiple HRV features out of range
    feature_data = {
        "sdnn": 5.0,   # Very low
        "rmssd": 3.0,  # Very low
        "nn50": 2.0    # Very low
    }

    patterns = engine._analyze_cross_feature_patterns(feature_data, "1_min")

    assert isinstance(patterns, list)
    # Should detect autonomic dysfunction pattern
    autonomic_patterns = [p for p in patterns if p.get("type") == "autonomic_dysfunction"]
    assert len(autonomic_patterns) > 0


def test_analyze_cross_feature_patterns_cardiovascular_risk():
    """Test _analyze_cross_feature_patterns detects cardiovascular risk (line 680)."""
    engine = InterpretationEngine()

    # Create data that should trigger cardiovascular risk
    # Both features need to be "below_range" to trigger the pattern
    feature_data = {
        "sdnn": 5.0,      # Below range (normal range is 20-50 for 1_min)
        "mean_nn": 500.0  # Below range (normal range is 700-900 for 1_min, so 500 = high heart rate)
    }

    patterns = engine._analyze_cross_feature_patterns(feature_data, "1_min")

    assert isinstance(patterns, list)
    # Should detect cardiovascular risk pattern
    cardio_patterns = [p for p in patterns if p.get("type") == "cardiovascular_risk"]
    assert len(cardio_patterns) > 0


def test_generate_cross_feature_warnings_contradictory_patterns():
    """Test _generate_cross_feature_warnings detects contradictions (line 705)."""
    engine = InterpretationEngine()

    # Create contradictory data (SDNN below, RMSSD above)
    feature_data = {
        "sdnn": 10.0,   # Below range
        "rmssd": 80.0   # Above range
    }

    warnings = engine._generate_cross_feature_warnings(feature_data, "1_min")

    assert isinstance(warnings, list)
    # Should generate warning about contradictory patterns
    assert len(warnings) > 0


def test_generate_cross_feature_warnings_extreme_values():
    """Test _generate_cross_feature_warnings detects extreme values (lines 718-724)."""
    engine = InterpretationEngine()

    # Create data with extremely low values
    feature_data = {
        "sdnn": 1.0,  # Extremely low (normal is 20-50)
    }

    warnings = engine._generate_cross_feature_warnings(feature_data, "1_min")

    assert isinstance(warnings, list)
    # Should generate warning about extreme values
    extreme_warnings = [w for w in warnings if "Extremely" in w]
    assert len(extreme_warnings) > 0


def test_generate_cross_feature_warnings_extreme_high_values():
    """Test _generate_cross_feature_warnings detects extremely high values (lines 723-724)."""
    engine = InterpretationEngine()

    # Create data with extremely high values
    feature_data = {
        "sdnn": 200.0,  # Extremely high (normal is 20-50, so 200 > 50*2)
    }

    warnings = engine._generate_cross_feature_warnings(feature_data, "1_min")

    assert isinstance(warnings, list)
    # Should generate warning about extreme values
    extreme_warnings = [w for w in warnings if "Extremely" in w]
    assert len(extreme_warnings) > 0