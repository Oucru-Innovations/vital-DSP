import pytest
import numpy as np
from vitalDSP.transforms.event_related_potential import EventRelatedPotential


@pytest.fixture
def sample_signal():
    """Create a sample signal for testing"""
    np.random.seed(42)
    return np.sin(np.linspace(0, 10, 1000))


def test_init_default_params(sample_signal):
    """Test initialization with default parameters"""
    stimulus_times = np.array([100, 300, 500])
    erp = EventRelatedPotential(sample_signal, stimulus_times)
    
    assert np.array_equal(erp.signal, sample_signal)
    assert np.array_equal(erp.stimulus_times, stimulus_times)
    assert erp.pre_stimulus == 100  # 0.1 * 1000
    assert erp.post_stimulus == 400  # 0.4 * 1000
    assert erp.sample_rate == 1000


def test_init_custom_params(sample_signal):
    """Test initialization with custom parameters"""
    stimulus_times = np.array([100, 300, 500])
    erp = EventRelatedPotential(
        sample_signal, 
        stimulus_times, 
        pre_stimulus=0.2, 
        post_stimulus=0.5, 
        sample_rate=500
    )
    
    assert erp.pre_stimulus == 100  # 0.2 * 500
    assert erp.post_stimulus == 250  # 0.5 * 500
    assert erp.sample_rate == 500


def test_compute_erp_basic(sample_signal):
    """Test basic ERP computation"""
    stimulus_times = np.array([100, 300, 500])
    erp = EventRelatedPotential(sample_signal, stimulus_times)
    erp_result = erp.compute_erp()
    
    assert isinstance(erp_result, np.ndarray)
    assert len(erp_result) == erp.pre_stimulus + erp.post_stimulus
    assert len(erp_result) == 500  # 100 + 400


def test_compute_erp_start_out_of_bounds():
    """Test ERP computation when start goes out of bounds (start < 0).
    
    This test covers line 122 in event_related_potential.py where
    the continue statement is executed when start < 0.
    """
    signal = np.sin(np.linspace(0, 10, 1000))
    # Use stimulus time that's too early (pre_stimulus will cause start < 0)
    stimulus_times = np.array([50])  # With pre_stimulus=100, start = 50 - 100 = -50 < 0
    
    erp = EventRelatedPotential(signal, stimulus_times, pre_stimulus=0.1, post_stimulus=0.4, sample_rate=1000)
    
    # This should skip the invalid segment and raise ValueError because no valid segments remain
    with pytest.raises(ValueError, match="No valid ERP segments were found"):
        erp.compute_erp()


def test_compute_erp_end_out_of_bounds():
    """Test ERP computation when end goes out of bounds (end > len(signal)).
    
    This test covers line 122 in event_related_potential.py where
    the continue statement is executed when end > len(signal).
    """
    signal = np.sin(np.linspace(0, 10, 1000))
    # Use stimulus time that's too late (post_stimulus will cause end > len(signal))
    stimulus_times = np.array([950])  # With post_stimulus=400, end = 950 + 400 = 1350 > 1000
    
    erp = EventRelatedPotential(signal, stimulus_times, pre_stimulus=0.1, post_stimulus=0.4, sample_rate=1000)
    
    # This should skip the invalid segment and raise ValueError because no valid segments remain
    with pytest.raises(ValueError, match="No valid ERP segments were found"):
        erp.compute_erp()


def test_compute_erp_both_out_of_bounds():
    """Test ERP computation when both start and end are out of bounds."""
    signal = np.sin(np.linspace(0, 10, 100))
    # Use stimulus time that causes both start < 0 and end > len(signal)
    stimulus_times = np.array([10])  # With pre_stimulus=100, post_stimulus=400, both will be out of bounds
    
    erp = EventRelatedPotential(signal, stimulus_times, pre_stimulus=0.1, post_stimulus=0.4, sample_rate=1000)
    
    with pytest.raises(ValueError, match="No valid ERP segments were found"):
        erp.compute_erp()


def test_compute_erp_mixed_valid_invalid():
    """Test ERP computation with mix of valid and invalid stimulus times.
    
    This test covers line 122 where some segments are skipped due to bounds,
    but valid segments still exist.
    """
    signal = np.sin(np.linspace(0, 10, 1000))
    # Mix of valid and invalid stimulus times
    # First is invalid (start < 0), second is valid, third is invalid (end > len(signal))
    stimulus_times = np.array([50, 500, 950])
    
    erp = EventRelatedPotential(signal, stimulus_times, pre_stimulus=0.1, post_stimulus=0.4, sample_rate=1000)
    erp_result = erp.compute_erp()
    
    # Should only use the valid segment (stimulus_time=500)
    assert isinstance(erp_result, np.ndarray)
    assert len(erp_result) == erp.pre_stimulus + erp.post_stimulus


def test_compute_erp_no_valid_segments():
    """Test ERP computation when no valid segments are found.
    
    This test covers lines 131-133 in event_related_potential.py where
    ValueError is raised when erps list is empty.
    """
    signal = np.sin(np.linspace(0, 10, 100))
    # All stimulus times are out of bounds
    stimulus_times = np.array([10, 20, 30])  # All will cause out-of-bounds segments
    
    erp = EventRelatedPotential(signal, stimulus_times, pre_stimulus=0.1, post_stimulus=0.4, sample_rate=1000)
    
    with pytest.raises(ValueError, match="No valid ERP segments were found. Check stimulus times and signal length."):
        erp.compute_erp()


def test_compute_erp_empty_stimulus_times():
    """Test ERP computation with empty stimulus_times array."""
    signal = np.sin(np.linspace(0, 10, 1000))
    stimulus_times = np.array([])
    
    erp = EventRelatedPotential(signal, stimulus_times)
    
    with pytest.raises(ValueError, match="No valid ERP segments were found"):
        erp.compute_erp()


def test_compute_erp_single_valid_segment(sample_signal):
    """Test ERP computation with a single valid segment"""
    stimulus_times = np.array([500])
    erp = EventRelatedPotential(sample_signal, stimulus_times)
    erp_result = erp.compute_erp()
    
    assert isinstance(erp_result, np.ndarray)
    assert len(erp_result) == erp.pre_stimulus + erp.post_stimulus
    # Should be equal to the single segment
    expected_segment = sample_signal[500 - erp.pre_stimulus:500 + erp.post_stimulus]
    np.testing.assert_array_equal(erp_result, expected_segment)


def test_compute_erp_multiple_valid_segments(sample_signal):
    """Test ERP computation with multiple valid segments"""
    # Use stimulus times that are all valid (within bounds)
    # With pre_stimulus=100 and post_stimulus=400, we need:
    # start >= 0 and end <= 1000
    # So: stimulus_time >= 100 and stimulus_time <= 600
    stimulus_times = np.array([200, 300, 400, 500])
    erp = EventRelatedPotential(sample_signal, stimulus_times)
    erp_result = erp.compute_erp()
    
    assert isinstance(erp_result, np.ndarray)
    assert len(erp_result) == erp.pre_stimulus + erp.post_stimulus
    
    # Verify that the result is reasonable (not all zeros, finite values)
    assert np.all(np.isfinite(erp_result))
    assert not np.allclose(erp_result, 0)  # Should not be all zeros
    
    # Verify all segments are valid
    segment_length = erp.pre_stimulus + erp.post_stimulus
    for st in stimulus_times:
        start = st - erp.pre_stimulus
        end = st + erp.post_stimulus
        assert start >= 0 and end <= len(sample_signal), f"Segment at {st} should be valid"
        assert end - start == segment_length, f"Segment at {st} should have correct length"


def test_compute_erp_different_sample_rates():
    """Test ERP computation with different sample rates"""
    signal = np.sin(np.linspace(0, 10, 5000))  # Longer signal for higher sample rate
    stimulus_times = np.array([500, 1500, 2500])
    
    # Test with higher sample rate
    erp = EventRelatedPotential(signal, stimulus_times, pre_stimulus=0.1, post_stimulus=0.4, sample_rate=2000)
    erp_result = erp.compute_erp()
    
    assert isinstance(erp_result, np.ndarray)
    assert len(erp_result) == erp.pre_stimulus + erp.post_stimulus
    assert erp.pre_stimulus == 200  # 0.1 * 2000
    assert erp.post_stimulus == 800  # 0.4 * 2000

