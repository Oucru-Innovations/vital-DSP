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
    assert contradiction == {
        "lf_hf_ratio": "High RMSSD and high LF/HF ratio may suggest imbalance."
    }


def test_get_feature_correlation(engine_with_mocked_config):
    """Test retrieving correlation for a feature."""
    engine = engine_with_mocked_config
    correlation = engine._get_feature_correlation("rmssd")
    assert correlation == {"hf_power": "RMSSD positively correlates with HF power."}


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