import pytest
import numpy as np
from unittest.mock import patch
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
    assert interpretation["interpretation"] == "RMSSD is within the normal range."


def test_interpret_feature_below_range(engine_with_mocked_config):
    """Test feature interpretation for a value below normal range."""
    engine = engine_with_mocked_config
    interpretation = engine.interpret_feature("rmssd", 10, segment_duration="1_min")
    assert (
        interpretation["interpretation"]
        == "Low RMSSD suggests reduced parasympathetic activity."
    )


def test_interpret_feature_above_range(engine_with_mocked_config):
    """Test feature interpretation for a value above normal range."""
    engine = engine_with_mocked_config
    interpretation = engine.interpret_feature("rmssd", 70, segment_duration="1_min")
    assert (
        interpretation["interpretation"] == "High RMSSD may indicate strong vagal tone."
    )


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
