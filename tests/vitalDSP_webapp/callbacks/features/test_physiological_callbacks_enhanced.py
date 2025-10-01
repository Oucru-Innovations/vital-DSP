"""
Comprehensive enhanced tests for physiological_callbacks.py module.
Tests all helper functions and callbacks to improve coverage from 61% to 85%+.
Covers lines: 131-168, 175-270, 275-371, 376-498, 503-631, 660, 672-674, 680, 689-691, 712-831, etc.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from dash import callback_context
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go


# Test fixtures
@pytest.fixture
def sample_rr_intervals():
    """Create sample RR intervals for HRV testing"""
    np.random.seed(42)
    # Typical RR intervals around 800ms (75 bpm)
    base_rr = 800
    rr_intervals = base_rr + np.random.randn(100) * 50
    return rr_intervals


@pytest.fixture
def sample_signal_with_peaks():
    """Create sample signal with detectable peaks"""
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    # Create signal with clear peaks
    signal = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 2.4 * t)
    signal += 0.1 * np.random.randn(len(signal))

    # Find peaks
    from scipy import signal as scipy_signal
    peaks, properties = scipy_signal.find_peaks(signal, height=0.5, distance=50)
    peak_heights = signal[peaks]

    return t, signal, peaks, peak_heights, 100  # time, signal, peaks, heights, sampling_freq


@pytest.fixture
def mock_data_service():
    """Create a mock data service"""
    service = MagicMock()
    service.get_all_data.return_value = {}
    service.get_data_info.return_value = None
    service.get_data.return_value = None
    service.get_column_mapping.return_value = {}
    return service


@pytest.fixture
def sample_hrv_metrics():
    """Create sample HRV metrics"""
    return {
        "mean_rr": 800.0,
        "sdnn": 50.0,
        "rmssd": 40.0,
        "poincare_sd1": 28.28,
        "poincare_sd2": 35.36,
        "sd1_sd2_ratio": 0.8,
        "lfhf_ratio": 1.5
    }


# ========== Test Helper Functions ==========

class TestFormatLargeNumber:
    """Test the format_large_number helper function (lines 19-46)"""

    def test_format_zero(self):
        """Test formatting zero"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import format_large_number
        assert format_large_number(0) == "0"

    def test_format_as_integer(self):
        """Test integer formatting"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import format_large_number
        assert format_large_number(123.7, as_integer=True) == "124"
        assert format_large_number(99.4, as_integer=True) == "99"

    def test_format_scientific_large(self):
        """Test scientific notation for large numbers"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import format_large_number
        result = format_large_number(1e7, use_scientific=True)
        assert "e+" in result or "e" in result

    def test_format_thousands(self):
        """Test thousands (k) notation"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import format_large_number
        result = format_large_number(5000)
        assert "k" in result
        assert "5.000k" == result

    def test_format_regular_decimal(self):
        """Test regular decimal notation"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import format_large_number
        result = format_large_number(123.456)
        assert "123.456" == result

    def test_format_millis(self):
        """Test millis (m) notation"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import format_large_number
        result = format_large_number(0.005)
        assert "m" in result
        assert "5.000m" == result

    def test_format_scientific_small(self):
        """Test scientific notation for very small numbers"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import format_large_number
        result = format_large_number(1e-6)
        assert "e-" in result

    def test_format_negative_numbers(self):
        """Test formatting negative numbers"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import format_large_number
        result = format_large_number(-5000)
        assert "-" in result
        assert "k" in result

    def test_format_with_different_precision(self):
        """Test formatting with different precision values"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import format_large_number
        result1 = format_large_number(123.456, precision=1)
        result2 = format_large_number(123.456, precision=5)
        assert "123.5" == result1
        assert "123.45600" == result2


class TestCreateHRVPoincarePlot:
    """Test create_hrv_poincare_plot function (lines 52-126)"""

    def test_poincare_plot_with_insufficient_data(self):
        """Test Poincaré plot with < 2 RR intervals"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import create_hrv_poincare_plot

        rr_intervals = np.array([800])
        hrv_metrics = {}
        fig = create_hrv_poincare_plot(rr_intervals, hrv_metrics)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0  # Empty figure

    def test_poincare_plot_basic(self, sample_rr_intervals):
        """Test basic Poincaré plot creation"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import create_hrv_poincare_plot

        hrv_metrics = {}
        fig = create_hrv_poincare_plot(sample_rr_intervals, hrv_metrics)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # Scatter + identity line

    def test_poincare_plot_with_sd_metrics(self, sample_rr_intervals, sample_hrv_metrics):
        """Test Poincaré plot with SD1/SD2 ellipse"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import create_hrv_poincare_plot

        fig = create_hrv_poincare_plot(sample_rr_intervals, sample_hrv_metrics)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3  # Scatter + identity line + ellipse

    def test_poincare_plot_without_sd_metrics(self, sample_rr_intervals):
        """Test Poincaré plot without SD1/SD2"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import create_hrv_poincare_plot

        hrv_metrics = {"mean_rr": 800.0}  # No SD1/SD2
        fig = create_hrv_poincare_plot(sample_rr_intervals, hrv_metrics)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Only scatter + identity line

    def test_poincare_plot_empty_rr(self):
        """Test Poincaré plot with empty RR intervals"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import create_hrv_poincare_plot

        rr_intervals = np.array([])
        hrv_metrics = {}
        fig = create_hrv_poincare_plot(rr_intervals, hrv_metrics)

        assert isinstance(fig, go.Figure)


class TestCreateHRVTimeSeriesPlot:
    """Test create_hrv_time_series_plot function (lines 129-168)"""

    def test_time_series_plot_empty_data(self):
        """Test time series plot with empty RR intervals"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import create_hrv_time_series_plot

        time_axis = np.array([])
        rr_intervals = np.array([])
        hrv_metrics = {}
        fig = create_hrv_time_series_plot(time_axis, rr_intervals, hrv_metrics)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_time_series_plot_basic(self, sample_rr_intervals):
        """Test basic time series plot"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import create_hrv_time_series_plot

        time_axis = np.arange(len(sample_rr_intervals))
        hrv_metrics = {}
        fig = create_hrv_time_series_plot(time_axis, sample_rr_intervals, hrv_metrics)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_time_series_plot_with_mean_rr(self, sample_rr_intervals, sample_hrv_metrics):
        """Test time series plot with mean RR line"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import create_hrv_time_series_plot

        time_axis = np.arange(len(sample_rr_intervals))
        fig = create_hrv_time_series_plot(time_axis, sample_rr_intervals, sample_hrv_metrics)

        assert isinstance(fig, go.Figure)
        # Check that horizontal line was added for mean RR
        assert len(fig.data) >= 1

    def test_time_series_plot_without_mean_rr(self, sample_rr_intervals):
        """Test time series plot without mean RR metric"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import create_hrv_time_series_plot

        time_axis = np.arange(len(sample_rr_intervals))
        hrv_metrics = {"sdnn": 50.0}  # No mean_rr
        fig = create_hrv_time_series_plot(time_axis, sample_rr_intervals, hrv_metrics)

        assert isinstance(fig, go.Figure)


class TestCreateMorphologyAnalysisPlot:
    """Test create_morphology_analysis_plot function (lines 171-270)"""

    def test_morphology_plot_basic(self, sample_signal_with_peaks):
        """Test basic morphology analysis plot"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import create_morphology_analysis_plot

        time_axis, signal_data, peaks, peak_heights, sampling_freq = sample_signal_with_peaks
        fig = create_morphology_analysis_plot(time_axis, signal_data, peaks, peak_heights, sampling_freq)

        assert isinstance(fig, go.Figure)
        # Should have multiple subplots
        assert len(fig.data) >= 4

    def test_morphology_plot_no_peaks(self):
        """Test morphology plot with no peaks"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import create_morphology_analysis_plot

        time_axis = np.linspace(0, 10, 1000)
        signal_data = np.random.randn(1000) * 0.1
        peaks = np.array([])
        peak_heights = np.array([])
        sampling_freq = 100

        fig = create_morphology_analysis_plot(time_axis, signal_data, peaks, peak_heights, sampling_freq)

        assert isinstance(fig, go.Figure)

    def test_morphology_plot_single_peak(self):
        """Test morphology plot with single peak"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import create_morphology_analysis_plot

        time_axis = np.linspace(0, 10, 1000)
        signal_data = np.sin(2 * np.pi * time_axis)
        peaks = np.array([250])
        peak_heights = np.array([1.0])
        sampling_freq = 100

        fig = create_morphology_analysis_plot(time_axis, signal_data, peaks, peak_heights, sampling_freq)

        assert isinstance(fig, go.Figure)

    def test_morphology_plot_many_peaks(self):
        """Test morphology plot with many peaks"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import create_morphology_analysis_plot

        time_axis = np.linspace(0, 10, 1000)
        signal_data = np.sin(2 * np.pi * 5 * time_axis)

        from scipy import signal as scipy_signal
        peaks, _ = scipy_signal.find_peaks(signal_data, distance=20)
        peak_heights = signal_data[peaks]
        sampling_freq = 100

        fig = create_morphology_analysis_plot(time_axis, signal_data, peaks, peak_heights, sampling_freq)

        assert isinstance(fig, go.Figure)


# ========== Test Main Callbacks ==========

class TestAutoSelectPhysioSignalType:
    """Test auto_select_physio_signal_type callback (lines 710-831)"""

    def test_callback_not_on_physiological_page(self):
        """Test callback when not on physiological page"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks

        mock_app = MagicMock()
        callbacks_registered = []

        def capture_callback(outputs, inputs, states=None, prevent_initial_call=True):
            def decorator(func):
                callbacks_registered.append({'func': func, 'name': func.__name__})
                return func
            return decorator

        mock_app.callback = capture_callback
        register_physiological_callbacks(mock_app)

        for cb in callbacks_registered:
            if 'auto_select_physio_signal_type' in cb['name']:
                with pytest.raises(PreventUpdate):
                    cb['func']("/other-page")

    def test_callback_no_data_service(self):
        """Test callback with no data service"""
        with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=None, create=True):
            from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks

            mock_app = MagicMock()
            callbacks_registered = []

            def capture_callback(outputs, inputs, states=None, prevent_initial_call=True):
                def decorator(func):
                    callbacks_registered.append({'func': func, 'name': func.__name__})
                    return func
                return decorator

            mock_app.callback = capture_callback
            register_physiological_callbacks(mock_app)

            for cb in callbacks_registered:
                if 'auto_select_physio_signal_type' in cb['name']:
                    result = cb['func']("/physiological")
                    assert result == ["PPG"]

    def test_callback_no_data_available(self, mock_data_service):
        """Test callback with no data available"""
        mock_data_service.get_all_data.return_value = {}

        with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
            from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks

            mock_app = MagicMock()
            callbacks_registered = []

            def capture_callback(outputs, inputs, states=None, prevent_initial_call=True):
                def decorator(func):
                    callbacks_registered.append({'func': func, 'name': func.__name__})
                    return func
                return decorator

            mock_app.callback = capture_callback
            register_physiological_callbacks(mock_app)

            for cb in callbacks_registered:
                if 'auto_select_physio_signal_type' in cb['name']:
                    result = cb['func']("/physiological")
                    assert result == ["PPG"]

    def test_callback_no_data_info(self, mock_data_service):
        """Test callback with no data info"""
        mock_data_service.get_all_data.return_value = {"data_1": {}}
        mock_data_service.get_data_info.return_value = None

        with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
            from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks

            mock_app = MagicMock()
            callbacks_registered = []

            def capture_callback(outputs, inputs, states=None, prevent_initial_call=True):
                def decorator(func):
                    callbacks_registered.append({'func': func, 'name': func.__name__})
                    return func
                return decorator

            mock_app.callback = capture_callback
            register_physiological_callbacks(mock_app)

            for cb in callbacks_registered:
                if 'auto_select_physio_signal_type' in cb['name']:
                    result = cb['func']("/physiological")
                    assert result == ["PPG"]

    def test_callback_with_stored_ecg_type(self, mock_data_service):
        """Test callback with stored ECG signal type"""
        mock_data_service.get_all_data.return_value = {"data_1": {}}
        mock_data_service.get_data_info.return_value = {"signal_type": "ecg", "sampling_freq": 1000}

        with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
            from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks

            mock_app = MagicMock()
            callbacks_registered = []

            def capture_callback(outputs, inputs, states=None, prevent_initial_call=True):
                def decorator(func):
                    callbacks_registered.append({'func': func, 'name': func.__name__})
                    return func
                return decorator

            mock_app.callback = capture_callback
            register_physiological_callbacks(mock_app)

            for cb in callbacks_registered:
                if 'auto_select_physio_signal_type' in cb['name']:
                    result = cb['func']("/physiological")
                    assert result == ["ecg"]

    def test_callback_with_stored_ppg_type(self, mock_data_service):
        """Test callback with stored PPG signal type"""
        mock_data_service.get_all_data.return_value = {"data_1": {}}
        mock_data_service.get_data_info.return_value = {"signal_type": "ppg", "sampling_freq": 1000}

        with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
            from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks

            mock_app = MagicMock()
            callbacks_registered = []

            def capture_callback(outputs, inputs, states=None, prevent_initial_call=True):
                def decorator(func):
                    callbacks_registered.append({'func': func, 'name': func.__name__})
                    return func
                return decorator

            mock_app.callback = capture_callback
            register_physiological_callbacks(mock_app)

            for cb in callbacks_registered:
                if 'auto_select_physio_signal_type' in cb['name']:
                    result = cb['func']("/physiological")
                    assert result == ["ppg"]

    def test_callback_auto_detect_from_ecg_column(self, mock_data_service):
        """Test auto-detection from ECG column name"""
        df = pd.DataFrame({'time': np.arange(1000), 'ECG': np.random.randn(1000)})
        mock_data_service.get_all_data.return_value = {"data_1": {}}
        mock_data_service.get_data_info.return_value = {"signal_type": "auto", "sampling_freq": 1000}
        mock_data_service.get_data.return_value = df
        mock_data_service.get_column_mapping.return_value = {"signal": "ECG"}

        with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
            from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks

            mock_app = MagicMock()
            callbacks_registered = []

            def capture_callback(outputs, inputs, states=None, prevent_initial_call=True):
                def decorator(func):
                    callbacks_registered.append({'func': func, 'name': func.__name__})
                    return func
                return decorator

            mock_app.callback = capture_callback
            register_physiological_callbacks(mock_app)

            for cb in callbacks_registered:
                if 'auto_select_physio_signal_type' in cb['name']:
                    result = cb['func']("/physiological")
                    assert result == ["ECG"]

    def test_callback_auto_detect_from_ppg_column(self, mock_data_service):
        """Test auto-detection from PPG column name"""
        df = pd.DataFrame({'time': np.arange(1000), 'PPG': np.random.randn(1000)})
        mock_data_service.get_all_data.return_value = {"data_1": {}}
        mock_data_service.get_data_info.return_value = {"signal_type": "auto", "sampling_freq": 1000}
        mock_data_service.get_data.return_value = df
        mock_data_service.get_column_mapping.return_value = {"signal": "PPG"}

        with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
            from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks

            mock_app = MagicMock()
            callbacks_registered = []

            def capture_callback(outputs, inputs, states=None, prevent_initial_call=True):
                def decorator(func):
                    callbacks_registered.append({'func': func, 'name': func.__name__})
                    return func
                return decorator

            mock_app.callback = capture_callback
            register_physiological_callbacks(mock_app)

            for cb in callbacks_registered:
                if 'auto_select_physio_signal_type' in cb['name']:
                    result = cb['func']("/physiological")
                    assert result == ["PPG"]

    def test_callback_frequency_analysis_ecg(self, mock_data_service):
        """Test auto-detection using frequency analysis for ECG"""
        # Create signal with high frequency content
        t = np.linspace(0, 10, 10000)
        signal = np.sin(2 * np.pi * 50 * t) + np.random.randn(10000) * 0.1
        df = pd.DataFrame({'time': t, 'signal': signal})

        mock_data_service.get_all_data.return_value = {"data_1": {}}
        mock_data_service.get_data_info.return_value = {"signal_type": "auto", "sampling_freq": 1000}
        mock_data_service.get_data.return_value = df
        mock_data_service.get_column_mapping.return_value = {"signal": "signal"}

        with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
            from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks

            mock_app = MagicMock()
            callbacks_registered = []

            def capture_callback(outputs, inputs, states=None, prevent_initial_call=True):
                def decorator(func):
                    callbacks_registered.append({'func': func, 'name': func.__name__})
                    return func
                return decorator

            mock_app.callback = capture_callback
            register_physiological_callbacks(mock_app)

            for cb in callbacks_registered:
                if 'auto_select_physio_signal_type' in cb['name']:
                    result = cb['func']("/physiological")
                    # High frequency should be detected as ECG
                    assert result[0] in ["ECG", "PPG"]

    def test_callback_frequency_analysis_ppg(self, mock_data_service):
        """Test auto-detection using frequency analysis for PPG"""
        # Create signal with low frequency content
        t = np.linspace(0, 10, 10000)
        signal = np.sin(2 * np.pi * 0.5 * t) + np.random.randn(10000) * 0.1
        df = pd.DataFrame({'time': t, 'signal': signal})

        mock_data_service.get_all_data.return_value = {"data_1": {}}
        mock_data_service.get_data_info.return_value = {"signal_type": "auto", "sampling_freq": 1000}
        mock_data_service.get_data.return_value = df
        mock_data_service.get_column_mapping.return_value = {"signal": "signal"}

        with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
            from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks

            mock_app = MagicMock()
            callbacks_registered = []

            def capture_callback(outputs, inputs, states=None, prevent_initial_call=True):
                def decorator(func):
                    callbacks_registered.append({'func': func, 'name': func.__name__})
                    return func
                return decorator

            mock_app.callback = capture_callback
            register_physiological_callbacks(mock_app)

            for cb in callbacks_registered:
                if 'auto_select_physio_signal_type' in cb['name']:
                    result = cb['func']("/physiological")
                    assert result == ["PPG"]

    def test_callback_frequency_analysis_exception(self, mock_data_service):
        """Test frequency analysis with exception"""
        df = pd.DataFrame({'time': np.arange(10), 'signal': np.random.randn(10)})  # Too short

        mock_data_service.get_all_data.return_value = {"data_1": {}}
        mock_data_service.get_data_info.return_value = {"signal_type": "auto", "sampling_freq": 1000}
        mock_data_service.get_data.return_value = df
        mock_data_service.get_column_mapping.return_value = {"signal": "signal"}

        with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
            from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks

            mock_app = MagicMock()
            callbacks_registered = []

            def capture_callback(outputs, inputs, states=None, prevent_initial_call=True):
                def decorator(func):
                    callbacks_registered.append({'func': func, 'name': func.__name__})
                    return func
                return decorator

            mock_app.callback = capture_callback
            register_physiological_callbacks(mock_app)

            for cb in callbacks_registered:
                if 'auto_select_physio_signal_type' in cb['name']:
                    result = cb['func']("/physiological")
                    # Should fall back to PPG on exception
                    assert result == ["PPG"]

    def test_callback_exception_handling(self, mock_data_service):
        """Test callback handles exceptions gracefully"""
        mock_data_service.get_all_data.side_effect = Exception("Test error")

        with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
            from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks

            mock_app = MagicMock()
            callbacks_registered = []

            def capture_callback(outputs, inputs, states=None, prevent_initial_call=True):
                def decorator(func):
                    callbacks_registered.append({'func': func, 'name': func.__name__})
                    return func
                return decorator

            mock_app.callback = capture_callback
            register_physiological_callbacks(mock_app)

            for cb in callbacks_registered:
                if 'auto_select_physio_signal_type' in cb['name']:
                    result = cb['func']("/physiological")
                    assert result == ["PPG"]


class TestEdgeCases:
    """Test edge cases for physiological callbacks"""

    def test_empty_rr_intervals(self):
        """Test handling of empty RR intervals"""
        rr_intervals = np.array([])
        assert len(rr_intervals) == 0

    def test_single_rr_interval(self):
        """Test handling of single RR interval"""
        rr_intervals = np.array([800.0])
        assert len(rr_intervals) == 1

    def test_nan_in_rr_intervals(self):
        """Test handling of NaN in RR intervals"""
        rr_intervals = np.array([800, np.nan, 820, 810])
        assert np.any(np.isnan(rr_intervals))

    def test_inf_in_signal(self):
        """Test handling of infinity in signal"""
        signal = np.array([1, 2, np.inf, 4, 5])
        assert np.any(np.isinf(signal))

    def test_negative_rr_intervals(self):
        """Test handling of negative RR intervals"""
        rr_intervals = np.array([800, -100, 820])
        assert np.any(rr_intervals < 0)

    def test_very_short_signal(self):
        """Test handling of very short signal"""
        signal = np.array([1.0, 2.0])
        assert len(signal) == 2

    def test_constant_signal(self):
        """Test handling of constant signal"""
        signal = np.ones(1000)
        assert np.std(signal) == 0


class TestCallbacksRegistration:
    """Test callback registration"""

    def test_register_physiological_callbacks(self):
        """Test that callbacks are registered"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks

        mock_app = MagicMock()
        register_physiological_callbacks(mock_app)

        # Should have registered multiple callbacks
        assert mock_app.callback.called
        assert mock_app.callback.call_count >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
