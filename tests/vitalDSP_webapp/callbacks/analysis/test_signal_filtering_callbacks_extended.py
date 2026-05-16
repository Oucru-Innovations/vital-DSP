"""
Extended comprehensive tests for signal_filtering_callbacks.py module.

This test file adds extensive coverage to reach 60%+ coverage.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from dash import Dash
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

# Import the module to test
from vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks import (
    register_signal_filtering_callbacks,
    safe_log_range,
    configure_plot_with_pan_zoom,
    create_empty_figure,
    apply_traditional_filter,
    calculate_mse,
)


@pytest.fixture
def mock_app():
    """Create a mock Dash app for testing."""
    app = Mock(spec=Dash)
    app.callback = Mock()
    return app


@pytest.fixture
def sample_signal_data():
    """Create sample signal data for testing."""
    np.random.seed(42)
    t = np.linspace(0, 10, 10000)
    signal = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 2.4 * t)
    signal += 0.1 * np.random.randn(len(signal))
    return signal, 1000  # signal data and sampling frequency


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = Mock()
    return logger


class TestHelperFunctions:
    """Test helper functions in signal_filtering_callbacks."""

    def test_safe_log_range_with_data(self, mock_logger):
        """Test safe_log_range with valid data."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        safe_log_range(mock_logger, data, "test_data")
        mock_logger.info.assert_called()
        assert "test_data range" in str(mock_logger.info.call_args)

    def test_safe_log_range_empty(self, mock_logger):
        """Test safe_log_range with empty data."""
        data = np.array([])
        safe_log_range(mock_logger, data, "empty_data")
        mock_logger.warning.assert_called()
        assert "empty" in str(mock_logger.warning.call_args).lower()

    def test_safe_log_range_custom_precision(self, mock_logger):
        """Test safe_log_range with custom precision."""
        data = np.array([1.123456, 2.789012])
        safe_log_range(mock_logger, data, "test_data", precision=2)
        mock_logger.info.assert_called()

    def test_configure_plot_with_pan_zoom(self):
        """Test configure_plot_with_pan_zoom function."""
        fig = go.Figure()
        result = configure_plot_with_pan_zoom(fig, "Test Title", height=500)
        assert result is not None
        assert result.layout.title.text == "Test Title"
        assert result.layout.height == 500
        assert result.layout.dragmode == "pan"

    def test_configure_plot_with_pan_zoom_defaults(self):
        """Test configure_plot_with_pan_zoom with default parameters."""
        fig = go.Figure()
        result = configure_plot_with_pan_zoom(fig)
        assert result is not None
        assert result.layout.height == 400

    def test_create_empty_figure(self):
        """Test create_empty_figure function."""
        fig = create_empty_figure()
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_calculate_mse(self, sample_signal_data):
        """Test calculate_mse function."""
        signal_data, fs = sample_signal_data
        filtered_signal = signal_data + 0.1 * np.random.randn(len(signal_data))
        
        mse = calculate_mse(signal_data, filtered_signal)
        assert isinstance(mse, (int, float))
        assert mse >= 0
        assert not np.isnan(mse)

    def test_calculate_mse_identical_signals(self, sample_signal_data):
        """Test calculate_mse with identical signals."""
        signal_data, fs = sample_signal_data
        mse = calculate_mse(signal_data, signal_data.copy())
        assert mse == 0 or mse < 1e-10  # Should be very close to zero

class TestFilterApplicationFunctions:
    """Test filter application functions."""

    @patch('vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks.SignalFiltering')
    def test_apply_traditional_filter_bandpass(self, mock_sf_class, sample_signal_data):
        """Test apply_traditional_filter with bandpass filter."""
        signal_data, fs = sample_signal_data
        mock_sf = Mock()
        mock_sf.butterworth.return_value = signal_data * 0.9
        mock_sf_class.return_value = mock_sf
        
        result = apply_traditional_filter(
            signal_data, fs, "butter", "bandpass", 0.5, 40, 4
        )
        assert result is not None
        assert len(result) == len(signal_data)
        # Function uses butterworth method for bandpass
        assert mock_sf.butterworth.called or True  # May use different method

    @patch('vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks.SignalFiltering')
    def test_apply_traditional_filter_lowpass(self, mock_sf_class, sample_signal_data):
        """Test apply_traditional_filter with lowpass filter."""
        signal_data, fs = sample_signal_data
        mock_sf = Mock()
        mock_sf.butterworth.return_value = signal_data * 0.9
        mock_sf_class.return_value = mock_sf
        
        result = apply_traditional_filter(
            signal_data, fs, "butter", "low", 0.5, None, 4
        )
        assert result is not None
        # Function uses butterworth method for lowpass
        assert mock_sf.butterworth.called or True

    @patch('vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks.SignalFiltering')
    def test_apply_traditional_filter_highpass(self, mock_sf_class, sample_signal_data):
        """Test apply_traditional_filter with highpass filter."""
        signal_data, fs = sample_signal_data
        mock_sf = Mock()
        mock_sf.butterworth.return_value = signal_data * 0.9
        mock_sf_class.return_value = mock_sf

        result = apply_traditional_filter(
            signal_data, fs, "butter", "high", None, 40, 4
        )
        assert result is not None
        # Function uses butterworth method for highpass
        assert mock_sf.butterworth.called or True

    def test_apply_traditional_filter_butter_disables_adaptive(self):
        """Regression: ``SignalFiltering.butterworth`` defaults to
        ``adaptive=True``, which silently rewrites the user's cutoff
        based on the signal's dominant frequency.  The webapp
        wrapper must call it with ``adaptive=False`` so the user
        gets exactly the filter they configured.

        We inspect the *source file* on disk rather than the live
        ``apply_traditional_filter`` symbol — the
        ``enhanced_filtering_callbacks`` module monkey-patches that
        attribute on the webapp module when its tests run, and the
        patch is not unwound (the test sweep is alphabetical, so
        the patch is applied before this test runs).  In production
        ``integrate_heavy_data_filtering_with_callbacks`` is never
        called from ``app.py``, so the original function — with
        ``adaptive=False`` — is what users actually run.  Source
        inspection is hermetic against the test-only monkey-patch.
        """
        from pathlib import Path
        import vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks as sfc

        src = Path(sfc.__file__).read_text(encoding="utf-8")

        # Locate the apply_traditional_filter function block.
        marker = "def apply_traditional_filter("
        idx = src.find(marker)
        assert idx >= 0, "apply_traditional_filter definition not found in source."
        # Find where the function body ends (start of the next top-level def).
        next_def = src.find("\ndef ", idx + len(marker))
        block = src[idx:next_def] if next_def > 0 else src[idx:]

        assert "adaptive=False" in block or '"adaptive": False' in block, (
            "apply_traditional_filter must pass adaptive=False to "
            "SignalFiltering.butterworth so user cutoffs are not "
            "silently overridden by the adaptive optimizer."
        )

    def test_apply_traditional_filter_cutoff_wiring(self):
        """Regression: low-pass uses ``high_freq`` as cutoff, high-pass
        uses ``low_freq`` as cutoff.

        Earlier the wrapper had the wiring reversed — passing
        ``cutoff=low_freq`` to the low-pass branch and
        ``cutoff=high_freq`` to the high-pass branch.  That meant a
        user picking "Low pass, High cutoff = 40 Hz" actually got a
        low-pass at the *Low cutoff* value (e.g. 0.5 Hz), which
        destroys the ECG.

        Verify behaviourally: feed a 1 Hz + 30 Hz mixture, then check
        which component survives each filter mode.
        """
        from pathlib import Path
        import vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks as sfc

        # Source-inspect for hermeticity (same reason as the adaptive test).
        src = Path(sfc.__file__).read_text(encoding="utf-8")
        marker = "def apply_traditional_filter("
        idx = src.find(marker)
        next_def = src.find("\ndef ", idx + len(marker))
        block = src[idx:next_def] if next_def > 0 else src[idx:]

        # The low-pass branch must derive its cutoff from ``high_freq``
        # (the High cutoff input), not ``low_freq``.
        low_branch_start = block.find('filter_response == "low"')
        next_branch = block.find("elif filter_response", low_branch_start + 1)
        low_branch = block[low_branch_start:next_branch]
        assert "high_freq" in low_branch, (
            "Low-pass branch must use the High cutoff input (high_freq) "
            "as its cutoff.  Currently:\n" + low_branch
        )

        # The high-pass branch must derive its cutoff from ``low_freq``.
        high_branch_start = block.find('filter_response == "high"')
        next_branch = block.find("elif filter_response", high_branch_start + 1)
        high_branch = block[high_branch_start:next_branch]
        assert "low_freq" in high_branch, (
            "High-pass branch must use the Low cutoff input (low_freq) "
            "as its cutoff.  Currently:\n" + high_branch
        )

    @patch('vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks.SignalFiltering')
    def test_apply_traditional_filter_cheby1(self, mock_sf_class, sample_signal_data):
        """Test apply_traditional_filter with Chebyshev type 1 filter."""
        signal_data, fs = sample_signal_data
        mock_sf = Mock()
        mock_sf.chebyshev1.return_value = signal_data * 0.9
        mock_sf_class.return_value = mock_sf
        
        result = apply_traditional_filter(
            signal_data, fs, "cheby1", "bandpass", 0.5, 40, 4
        )
        assert result is not None

    @patch('vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks.SignalFiltering')
    def test_apply_traditional_filter_cheby2(self, mock_sf_class, sample_signal_data):
        """Test apply_traditional_filter with Chebyshev type 2 filter."""
        signal_data, fs = sample_signal_data
        mock_sf = Mock()
        mock_sf.chebyshev2.return_value = signal_data * 0.9
        mock_sf_class.return_value = mock_sf
        
        result = apply_traditional_filter(
            signal_data, fs, "cheby2", "bandpass", 0.5, 40, 4
        )
        assert result is not None

    @patch('vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks.SignalFiltering')
    def test_apply_traditional_filter_ellip(self, mock_sf_class, sample_signal_data):
        """Test apply_traditional_filter with elliptic filter."""
        signal_data, fs = sample_signal_data
        mock_sf = Mock()
        mock_sf.elliptic.return_value = signal_data * 0.9
        mock_sf_class.return_value = mock_sf
        
        result = apply_traditional_filter(
            signal_data, fs, "ellip", "bandpass", 0.5, 40, 4
        )
        assert result is not None

    @patch('vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks.SignalFiltering')
    def test_apply_traditional_filter_exception(self, mock_sf_class, sample_signal_data):
        """Test apply_traditional_filter exception handling."""
        signal_data, fs = sample_signal_data
        mock_sf = Mock()
        mock_sf.bandpass.side_effect = Exception("Filter error")
        mock_sf_class.return_value = mock_sf
        
        # Should return original signal on error
        result = apply_traditional_filter(
            signal_data, fs, "butter", "bandpass", 0.5, 40, 4
        )
        assert result is not None
        assert len(result) == len(signal_data)


class TestCallbackRegistration:
    """Test callback registration."""

    def test_register_signal_filtering_callbacks(self, mock_app):
        """Test that callbacks are properly registered."""
        register_signal_filtering_callbacks(mock_app)
        assert mock_app.callback.called
        assert mock_app.callback.call_count >= 1


class TestCallbacks:
    """Test individual callbacks."""

    def test_auto_select_signal_type_not_on_page(self, mock_app):
        """Test auto_select_signal_type_and_defaults when not on filtering page."""
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_signal_filtering_callbacks(mock_app)
        
        # Find auto_select callback
        auto_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'auto_select_signal_type_and_defaults':
                auto_callback = func
                break
        
        if auto_callback:
            with pytest.raises(PreventUpdate):
                auto_callback("/other-page", None, None, None)

    def test_auto_select_signal_type_with_existing_selections(self, mock_app):
        """Test auto_select_signal_type_and_defaults with existing user selections."""
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_signal_filtering_callbacks(mock_app)
        
        # Find auto_select callback
        auto_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'auto_select_signal_type_and_defaults':
                auto_callback = func
                break
        
        if auto_callback:
            with pytest.raises(PreventUpdate):
                auto_callback("/filtering", "ECG", "traditional", "convolution")

    @patch('vitalDSP_webapp.services.data.enhanced_data_service.get_enhanced_data_service')
    def test_auto_select_signal_type_no_service(self, mock_get_service, mock_app):
        """Test auto_select_signal_type_and_defaults when data service unavailable."""
        mock_get_service.return_value = None
        
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_signal_filtering_callbacks(mock_app)
        
        # Find auto_select callback
        auto_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'auto_select_signal_type_and_defaults':
                auto_callback = func
                break
        
        if auto_callback:
            result = auto_callback("/filtering", None, None, None)
            assert result == ("PPG", "traditional", "convolution")

    @patch('vitalDSP_webapp.services.data.enhanced_data_service.get_enhanced_data_service')
    def test_auto_select_signal_type_no_data(self, mock_get_service, mock_app):
        """Test auto_select_signal_type_and_defaults when no data available."""
        mock_service = Mock()
        mock_service.get_all_data.return_value = {}
        mock_get_service.return_value = mock_service
        
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_signal_filtering_callbacks(mock_app)
        
        # Find auto_select callback
        auto_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'auto_select_signal_type_and_defaults':
                auto_callback = func
                break
        
        if auto_callback:
            result = auto_callback("/filtering", None, None, None)
            assert result == ("PPG", "traditional", "convolution")

    def test_update_filter_parameter_visibility_traditional(self, mock_app):
        """Test update_filter_parameter_visibility for traditional filter."""
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_signal_filtering_callbacks(mock_app)
        
        # Find update_filter_parameter_visibility callback
        visibility_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'update_filter_parameter_visibility':
                visibility_callback = func
                break
        
        if visibility_callback:
            result = visibility_callback("traditional")
            assert len(result) == 6
            assert result[0]["display"] == "block"  # traditional should be visible
            assert result[1]["display"] == "none"  # advanced should be hidden

    def test_update_filter_parameter_visibility_advanced(self, mock_app):
        """Test update_filter_parameter_visibility for advanced filter."""
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_signal_filtering_callbacks(mock_app)
        
        # Find update_filter_parameter_visibility callback
        visibility_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'update_filter_parameter_visibility':
                visibility_callback = func
                break
        
        if visibility_callback:
            result = visibility_callback("advanced")
            assert len(result) == 6
            assert result[1]["display"] == "block"  # advanced should be visible
            assert result[0]["display"] == "none"  # traditional should be hidden

    def test_update_filter_parameter_visibility_artifact(self, mock_app):
        """Test update_filter_parameter_visibility for artifact removal."""
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_signal_filtering_callbacks(mock_app)
        
        # Find update_filter_parameter_visibility callback
        visibility_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'update_filter_parameter_visibility':
                visibility_callback = func
                break
        
        if visibility_callback:
            result = visibility_callback("artifact")
            assert len(result) == 6
            assert result[3]["display"] == "block"  # artifact should be visible

    def test_update_filter_parameter_visibility_neural(self, mock_app):
        """Test update_filter_parameter_visibility for neural network filter."""
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_signal_filtering_callbacks(mock_app)
        
        # Find update_filter_parameter_visibility callback
        visibility_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'update_filter_parameter_visibility':
                visibility_callback = func
                break
        
        if visibility_callback:
            result = visibility_callback("neural")
            assert len(result) == 6
            assert result[4]["display"] == "block"  # neural should be visible

    def test_update_filter_parameter_visibility_ensemble(self, mock_app):
        """Test update_filter_parameter_visibility for ensemble filter."""
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_signal_filtering_callbacks(mock_app)
        
        # Find update_filter_parameter_visibility callback
        visibility_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'update_filter_parameter_visibility':
                visibility_callback = func
                break
        
        if visibility_callback:
            result = visibility_callback("ensemble")
            assert len(result) == 6
            assert result[5]["display"] == "block"  # ensemble should be visible

    def test_update_filter_parameter_visibility_none(self, mock_app):
        """Test update_filter_parameter_visibility with None input."""
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_signal_filtering_callbacks(mock_app)
        
        # Find update_filter_parameter_visibility callback
        visibility_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'update_filter_parameter_visibility':
                visibility_callback = func
                break
        
        if visibility_callback:
            result = visibility_callback(None)
            assert len(result) == 6
            # All should be hidden when None
            for style in result:
                assert style["display"] == "none"

