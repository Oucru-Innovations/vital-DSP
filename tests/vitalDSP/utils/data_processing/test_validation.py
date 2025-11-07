"""
Comprehensive tests for validation.py module.

Tests cover all validation functions and edge cases.
"""

import pytest
import numpy as np
from vitalDSP.utils.data_processing.validation import (
    SignalValidator,
    validate_signal_input,
    validate_signal_length,
    validate_frequency_range,
    validate_positive_parameter,
)


class TestSignalValidator:
    """Tests for SignalValidator class."""

    def test_validate_signal_basic(self):
        """Test basic signal validation."""
        signal = np.array([1, 2, 3, 4, 5])
        result = SignalValidator.validate_signal(signal)
        assert isinstance(result, np.ndarray)
        assert len(result) == 5

    def test_validate_signal_list_input(self):
        """Test validation with list input."""
        signal = [1, 2, 3, 4, 5]
        result = SignalValidator.validate_signal(signal)
        assert isinstance(result, np.ndarray)

    def test_validate_signal_conversion_error(self):
        """Test validation with unconvertible input."""
        # Test lines 91-94: exception during conversion
        class Unconvertible:
            def __iter__(self):
                raise RuntimeError("Cannot convert")
        
        with pytest.raises((TypeError, ValueError), match="(must be array-like|len)"):
            SignalValidator.validate_signal(Unconvertible())

    def test_validate_signal_empty_not_allowed(self):
        """Test validation with empty signal not allowed."""
        signal = np.array([])
        with pytest.raises(ValueError, match="cannot be empty"):
            SignalValidator.validate_signal(signal, allow_empty=False)

    def test_validate_signal_empty_allowed(self):
        """Test validation with empty signal allowed."""
        signal = np.array([])
        result = SignalValidator.validate_signal(signal, allow_empty=True)
        assert len(result) == 0

    def test_validate_signal_min_length(self):
        """Test validation with minimum length requirement."""
        signal = np.array([1, 2])
        with pytest.raises(ValueError, match="minimum required"):
            SignalValidator.validate_signal(signal, min_length=5)

    def test_validate_signal_nan_not_allowed(self):
        """Test validation with NaN values not allowed."""
        signal = np.array([1, 2, np.nan, 4])
        with pytest.raises(ValueError, match="NaN values"):
            SignalValidator.validate_signal(signal, allow_nan=False)

    def test_validate_signal_nan_allowed(self):
        """Test validation with NaN values allowed."""
        signal = np.array([1, 2, np.nan, 4])
        result = SignalValidator.validate_signal(signal, allow_nan=True)
        assert len(result) == 4

    def test_validate_signal_inf_not_allowed(self):
        """Test validation with infinite values not allowed."""
        signal = np.array([1, 2, np.inf, 4])
        with pytest.raises(ValueError, match="infinite values"):
            SignalValidator.validate_signal(signal, allow_inf=False)

    def test_validate_signal_inf_allowed(self):
        """Test validation with infinite values allowed."""
        signal = np.array([1, 2, np.inf, 4])
        result = SignalValidator.validate_signal(signal, allow_inf=True)
        assert len(result) == 4

    def test_validate_signal_pair_same_length(self):
        """Test validation of signal pair with same length required."""
        signal1 = np.array([1, 2, 3])
        signal2 = np.array([4, 5, 6])
        sig1, sig2 = SignalValidator.validate_signal_pair(signal1, signal2)
        assert len(sig1) == len(sig2)

    def test_validate_signal_pair_different_length(self):
        """Test validation of signal pair with different lengths."""
        signal1 = np.array([1, 2, 3])
        signal2 = np.array([4, 5, 6, 7])
        with pytest.raises(ValueError, match="length"):
            SignalValidator.validate_signal_pair(signal1, signal2, require_same_length=True)

    def test_validate_signal_pair_different_length_allowed(self):
        """Test validation of signal pair with different lengths allowed."""
        signal1 = np.array([1, 2, 3])
        signal2 = np.array([4, 5, 6, 7])
        sig1, sig2 = SignalValidator.validate_signal_pair(
            signal1, signal2, require_same_length=False
        )
        assert len(sig1) != len(sig2)

    def test_validate_frequency_parameters_basic(self):
        """Test basic frequency parameter validation."""
        cutoff = 10.0
        fs = 100.0
        result_cutoff, result_fs = SignalValidator.validate_frequency_parameters(cutoff, fs)
        assert result_cutoff == cutoff
        assert result_fs == fs

    def test_validate_frequency_parameters_invalid_fs(self):
        """Test validation with invalid sampling frequency."""
        with pytest.raises(ValueError, match="must be positive"):
            SignalValidator.validate_frequency_parameters(10.0, fs=0)

    def test_validate_frequency_parameters_list_cutoff(self):
        """Test validation with list cutoff frequencies."""
        cutoff = [5.0, 15.0]
        fs = 100.0
        result_cutoff, result_fs = SignalValidator.validate_frequency_parameters(
            cutoff, fs, filter_type="bandpass"
        )
        assert len(result_cutoff) == 2

    def test_validate_frequency_parameters_list_negative(self):
        """Test validation with negative cutoff in list."""
        cutoff = [5.0, -15.0]
        fs = 100.0
        with pytest.raises(ValueError, match="must be positive"):
            SignalValidator.validate_frequency_parameters(cutoff, fs)

    def test_validate_frequency_parameters_list_above_nyquist(self):
        """Test validation with cutoff above Nyquist frequency."""
        cutoff = [5.0, 60.0]  # 60 > 100/2 = 50
        fs = 100.0
        with pytest.raises(ValueError, match="Nyquist frequency"):
            SignalValidator.validate_frequency_parameters(cutoff, fs)

    def test_validate_frequency_parameters_bandpass_wrong_count(self):
        """Test validation with bandpass requiring 2 frequencies."""
        cutoff = [5.0]  # Only 1 frequency for bandpass
        fs = 100.0
        with pytest.raises(ValueError, match="exactly 2 cutoff frequencies"):
            SignalValidator.validate_frequency_parameters(cutoff, fs, filter_type="bandpass")

    def test_validate_frequency_parameters_bandstop_wrong_count(self):
        """Test validation with bandstop requiring 2 frequencies."""
        cutoff = [5.0]  # Only 1 frequency for bandstop
        fs = 100.0
        with pytest.raises(ValueError, match="exactly 2 cutoff frequencies"):
            SignalValidator.validate_frequency_parameters(cutoff, fs, filter_type="bandstop")

    def test_validate_frequency_parameters_single_negative(self):
        """Test validation with single negative cutoff."""
        with pytest.raises(ValueError, match="must be positive"):
            SignalValidator.validate_frequency_parameters(-10.0, fs=100.0)

    def test_validate_frequency_parameters_single_above_nyquist(self):
        """Test validation with single cutoff above Nyquist."""
        with pytest.raises(ValueError, match="Nyquist frequency"):
            SignalValidator.validate_frequency_parameters(60.0, fs=100.0)

    def test_validate_filter_order_basic(self):
        """Test basic filter order validation."""
        order = 5
        result = SignalValidator.validate_filter_order(order)
        assert result == 5

    def test_validate_filter_order_non_integer(self):
        """Test validation with non-integer order."""
        with pytest.raises(TypeError, match="must be integer"):
            SignalValidator.validate_filter_order(5.5)

    def test_validate_filter_order_negative(self):
        """Test validation with negative order."""
        with pytest.raises(ValueError, match="must be positive"):
            SignalValidator.validate_filter_order(-5)

    def test_validate_filter_order_too_large(self):
        """Test validation with order exceeding maximum."""
        with pytest.raises(ValueError, match="exceeds maximum"):
            SignalValidator.validate_filter_order(25, max_order=20)

    def test_validate_window_parameters_basic(self):
        """Test basic window parameter validation."""
        window_size = 10
        signal_length = 100
        result = SignalValidator.validate_window_parameters(window_size, signal_length)
        assert result == 10

    def test_validate_window_parameters_non_integer(self):
        """Test validation with non-integer window size."""
        with pytest.raises(TypeError, match="must be integer"):
            SignalValidator.validate_window_parameters(10.5, signal_length=100)

    def test_validate_window_parameters_too_small(self):
        """Test validation with window size below minimum."""
        with pytest.raises(ValueError, match="minimum"):
            SignalValidator.validate_window_parameters(1, signal_length=100, min_window_size=2)

    def test_validate_window_parameters_too_large(self):
        """Test validation with window size larger than signal."""
        with pytest.raises(ValueError, match="signal length"):
            SignalValidator.validate_window_parameters(150, signal_length=100)

    def test_validate_nn_intervals_basic(self):
        """Test basic NN interval validation."""
        nn_intervals = np.array([800, 850, 900, 820])
        result = SignalValidator.validate_nn_intervals(nn_intervals)
        assert isinstance(result, np.ndarray)

    def test_validate_nn_intervals_negative(self):
        """Test validation with negative NN intervals."""
        nn_intervals = np.array([800, -850, 900])
        with pytest.raises(ValueError, match="must be positive"):
            SignalValidator.validate_nn_intervals(nn_intervals)

    def test_validate_nn_intervals_too_large(self):
        """Test validation with NN intervals exceeding 3000ms."""
        nn_intervals = np.array([800, 3500, 900])
        # Should warn but not raise
        with pytest.warns(UserWarning, match="exceed 3000ms"):
            result = SignalValidator.validate_nn_intervals(nn_intervals)
        assert isinstance(result, np.ndarray)

    def test_validate_nn_intervals_too_small(self):
        """Test validation with NN intervals below 300ms."""
        nn_intervals = np.array([800, 250, 900])
        # Should warn but not raise
        with pytest.warns(UserWarning, match="less than 300ms"):
            result = SignalValidator.validate_nn_intervals(nn_intervals)
        assert isinstance(result, np.ndarray)

    def test_validate_threshold_basic(self):
        """Test basic threshold validation."""
        threshold = 0.5
        result = SignalValidator.validate_threshold(threshold)
        assert result == 0.5

    def test_validate_threshold_non_numeric(self):
        """Test validation with non-numeric threshold."""
        with pytest.raises(TypeError, match="must be numeric"):
            SignalValidator.validate_threshold("0.5")

    def test_validate_threshold_absolute_negative(self):
        """Test validation with negative absolute threshold."""
        with pytest.raises(ValueError, match="non-negative"):
            SignalValidator.validate_threshold(-0.5, threshold_type="absolute")

    def test_validate_threshold_relative_too_low(self):
        """Test validation with relative threshold below 0."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            SignalValidator.validate_threshold(-0.1, threshold_type="relative")

    def test_validate_threshold_relative_too_high(self):
        """Test validation with relative threshold above 1."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            SignalValidator.validate_threshold(1.5, threshold_type="relative")


class TestValidateSignalInput:
    """Tests for validate_signal_input decorator."""

    def test_validate_signal_input_decorator(self):
        """Test the validate_signal_input decorator."""
        @validate_signal_input
        def test_func(self_or_none, signal):
            return len(signal)

        # Test with valid signal
        signal = np.array([1, 2, 3, 4, 5])
        result = test_func(None, signal)
        assert result == 5

    def test_validate_signal_input_decorator_invalid(self):
        """Test decorator with invalid signal."""
        @validate_signal_input
        def test_func(self_or_none, signal):
            return len(signal)

        # Test with invalid signal (empty)
        signal = np.array([])
        with pytest.raises(ValueError):
            test_func(None, signal)


class TestConvenienceFunctions:
    """Tests for convenience validation functions."""

    def test_validate_signal_length_basic(self):
        """Test validate_signal_length function."""
        signal = np.array([1, 2, 3, 4, 5])
        result = validate_signal_length(signal, min_length=3)
        assert isinstance(result, np.ndarray)

    def test_validate_signal_length_too_short(self):
        """Test validate_signal_length with too short signal."""
        signal = np.array([1, 2])
        with pytest.raises(ValueError):
            validate_signal_length(signal, min_length=5)

    def test_validate_frequency_range_basic(self):
        """Test validate_frequency_range function."""
        cutoff = 10.0
        fs = 100.0
        result_cutoff, result_fs = validate_frequency_range(cutoff, fs)
        assert result_cutoff == cutoff
        assert result_fs == fs

    def test_validate_frequency_range_invalid(self):
        """Test validate_frequency_range with invalid parameters."""
        with pytest.raises(ValueError):
            validate_frequency_range(60.0, fs=100.0)  # Above Nyquist

    def test_validate_positive_parameter_basic(self):
        """Test validate_positive_parameter function."""
        result = validate_positive_parameter(5.0, "test_param")
        assert result == 5.0

    def test_validate_positive_parameter_negative(self):
        """Test validate_positive_parameter with negative value."""
        with pytest.raises(ValueError, match="must be positive"):
            validate_positive_parameter(-5.0, "test_param")

    def test_validate_positive_parameter_zero(self):
        """Test validate_positive_parameter with zero."""
        with pytest.raises(ValueError, match="must be positive"):
            validate_positive_parameter(0.0, "test_param")

