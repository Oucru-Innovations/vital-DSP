import pytest
import numpy as np
from vitalDSP.transforms.beats_transformation import RRTransformation
from vitalDSP.utils.common import find_peaks


# Mock the find_peaks function if needed to simulate peak detection (comment this out if using actual peak detection)
def mock_find_peaks(signal, height=None, distance=None):
    return np.array([100, 200, 300, 400, 500])


@pytest.fixture
def sample_ecg_signal():
    # Simulate a clean ECG signal with regular peaks
    signal = np.sin(np.linspace(0, 10 * np.pi, 1000))  # Sine wave signal
    return signal


@pytest.fixture
def rr_transformation(sample_ecg_signal):
    # Create an instance of the RRTransformation with mock data
    return RRTransformation(signal=sample_ecg_signal, fs=100, signal_type="ecg")


def test_compute_rr_intervals_valid_case(rr_transformation):
    # Test RR interval computation in a valid case
    rr_intervals = rr_transformation.compute_rr_intervals()
    assert isinstance(rr_intervals, np.ndarray)
    assert len(rr_intervals) > 0
    assert rr_intervals.mean() > 0, "RR intervals should be positive"


def test_compute_rr_intervals_no_peaks():
    # Test if no peaks are detected
    signal = np.zeros(1000)  # Flatline signal
    rr_transformation = RRTransformation(signal, fs=100)
    with pytest.raises(ValueError, match="No peaks detected in the signal"):
        rr_transformation.compute_rr_intervals()


def test_remove_invalid_rr_intervals(rr_transformation):
    # Test invalid RR interval removal
    rr_intervals = np.array(
        [0.2, 0.9, 1.2, 1.5, 2.5]
    )  # Simulated RR intervals with three invalid
    filtered_rr_intervals = rr_transformation.remove_invalid_rr_intervals(rr_intervals)
    assert (
        np.isnan(filtered_rr_intervals).sum() == 3
    ), "Three invalid intervals should be NaN"


def test_remove_invalid_rr_intervals_sudden_change(rr_transformation):
    # Test invalid RR interval removal with sudden change detection
    rr_intervals = np.array(
        [0.8, 0.82, 0.85, 1.2, 0.9]
    )  # Sudden increase between two intervals
    filtered_rr_intervals = rr_transformation.remove_invalid_rr_intervals(
        rr_intervals, sudden_change_threshold=0.2
    )
    assert (
        np.isnan(filtered_rr_intervals).sum() > 0
    ), "Sudden changes should be marked as invalid"


def test_impute_rr_intervals_linear(rr_transformation):
    # Test linear interpolation imputation
    rr_intervals = np.array([0.8, np.nan, 0.82, np.nan, np.nan, 0.85, 0.83])
    imputed_intervals = rr_transformation.impute_rr_intervals(
        rr_intervals, method="linear"
    )
    assert not np.isnan(
        imputed_intervals
    ).any(), "All NaNs should be imputed with linear interpolation"
    imputed_intervals = rr_transformation.impute_rr_intervals(
        rr_intervals, method="spline"
    )
    assert not np.isnan(
        imputed_intervals
    ).any(), "All NaNs should be imputed with linear interpolation"
    imputed_intervals = rr_transformation.impute_rr_intervals(
        rr_intervals, method="mean"
    )
    assert not np.isnan(
        imputed_intervals
    ).any(), "All NaNs should be imputed with linear interpolation"
    imputed_intervals = rr_transformation.impute_rr_intervals(
        rr_intervals, method="median"
    )
    assert not np.isnan(
        imputed_intervals
    ).any(), "All NaNs should be imputed with linear interpolation"
    imputed_intervals = rr_transformation.impute_rr_intervals(
        rr_intervals, method="forward_fill"
    )
    assert not np.isnan(
        imputed_intervals
    ).any(), "All NaNs should be imputed with linear interpolation"
    imputed_intervals = rr_transformation.impute_rr_intervals(
        rr_intervals, method="backward_fill"
    )
    assert not np.isnan(
        imputed_intervals
    ).any(), "All NaNs should be imputed with linear interpolation"
    imputed_intervals = rr_transformation.impute_rr_intervals(
        rr_intervals, method="rolling_mean"
    )
    assert not np.isnan(
        imputed_intervals
    ).any(), "All NaNs should be imputed with linear interpolation"

    intervals = np.array([0.8, np.nan, 0.82, 0.84, np.nan, 0.83, 0.81, np.nan])
    imputed_intervals = rr_transformation.impute_rr_intervals(
        intervals, method="adaptive", order=3
    )
    # Check if spline interpolation was used by asserting no NaN values remain
    assert not np.isnan(
        imputed_intervals
    ).any(), "Spline interpolation failed to impute NaN values correctly"


def test_impute_rr_intervals_adaptive(rr_transformation):
    # Test adaptive imputation
    rr_intervals = np.array([0.8, np.nan, 0.82, np.nan, np.nan, 0.85, 0.83])
    imputed_intervals = rr_transformation.impute_rr_intervals(
        rr_intervals, method="adaptive"
    )
    assert not np.isnan(
        imputed_intervals
    ).any(), "All NaNs should be imputed adaptively"


def test_process_rr_intervals(rr_transformation):
    # Test the full RR interval processing pipeline
    processed_rr_intervals = rr_transformation.process_rr_intervals()
    assert isinstance(
        processed_rr_intervals, np.ndarray
    ), "Processed RR intervals should be an array"
    assert not np.isnan(
        processed_rr_intervals
    ).any(), "All invalid RR intervals should be imputed"


def test_reconsider_trends(rr_transformation):
    # Test the reconsider_trends method
    rr_intervals = np.array([0.8, np.nan, 0.82, np.nan, 0.85, 0.83])
    rr_intervals_with_trends = rr_transformation._reconsider_trends(rr_intervals)
    assert (
        np.isnan(rr_intervals_with_trends).sum() == 2
    ), "Trend reconsideration should only handle gradual trends"


def test_invalid_signal_length():
    # Generate a synthetic ECG-like signal with detectable peaks
    fs = 100  # Sampling frequency in Hz
    t = np.linspace(0, 2, fs * 2)  # 2 seconds of data
    ecg_signal = np.sin(2 * np.pi * 1.7 * t)  # Simulated sine wave with peaks

    # Ensure the signal length is greater than the padlen required by the filter
    rr_transformation = RRTransformation(signal=ecg_signal, fs=fs)

    try:
        rr_intervals = rr_transformation.compute_rr_intervals()
        assert len(rr_intervals) > 0, "RR intervals should be computed from the peaks."
    except ValueError:
        pytest.fail(
            "Signal length should be long enough and peaks should be detected for RR interval computation."
        )


def test_invalid_imputation_method(rr_transformation):
    # Test for unsupported imputation method
    rr_intervals = np.array([0.8, np.nan, 0.82, 0.85])
    with pytest.raises(ValueError, match="Unsupported imputation method"):
        rr_transformation.impute_rr_intervals(rr_intervals, method="unsupported_method")
