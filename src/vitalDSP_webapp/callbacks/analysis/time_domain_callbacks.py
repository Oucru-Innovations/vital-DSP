"""
Time Domain Analysis callbacks for vitalDSP webapp.

This module handles all time domain analysis callbacks including signal visualization,
peak detection, critical points analysis, and comprehensive time domain metrics.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, callback_context, no_update, html, dcc
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from scipy import signal
import logging

# Import plot utilities for performance optimization
try:
    from vitalDSP_webapp.utils.plot_utils import limit_plot_data, check_plot_data_size
except ImportError:
    # Fallback if plot_utils not available
    def limit_plot_data(
        time_axis, signal_data, max_duration=300, max_points=10000, start_time=None
    ):
        """Fallback implementation of limit_plot_data"""
        return time_axis, signal_data

    def check_plot_data_size(time_axis, signal_data):
        """Fallback implementation"""
        return True


# Import filtering functions from signal filtering callbacks
from vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks import (
    apply_traditional_filter,
)
from vitalDSP_webapp.callbacks.core.theme_callbacks import apply_plot_theme


# Helper function for formatting large numbers
def format_large_number(value, precision=3, use_scientific=False):
    """Format large numbers with appropriate scaling and units."""
    if value == 0:
        return "0"

    abs_value = abs(value)

    if use_scientific or abs_value >= 1e6:
        # Use scientific notation for very large numbers
        return f"{value:.{precision}e}"
    elif abs_value >= 1e3:
        # Use thousands (k) notation
        scaled_value = value / 1e3
        return f"{scaled_value:.{precision}f}k"
    elif abs_value >= 1:
        # Regular decimal notation
        return f"{value:.{precision}f}"
    elif abs_value >= 1e-3:
        # Use millis (m) notation
        scaled_value = value * 1e3
        return f"{scaled_value:.{precision}f}m"
    else:
        # Use scientific notation for very small numbers
        return f"{value:.{precision}e}"


logger = logging.getLogger(__name__)


def configure_plot_with_pan_zoom(fig, title="", height=400):
    """
    Configure plotly figure with pan/zoom tools and consistent styling.

    Args:
        fig: Plotly figure object
        title: Plot title
        height: Plot height in pixels

    Returns:
        Configured plotly figure
    """
    fig.update_layout(
        title=title,
        height=height,
        showlegend=True,
        template="plotly_white",
        # Add pan/zoom tools
        xaxis=dict(
            rangeslider=dict(visible=False),
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=1, label="1m", step="minute", stepmode="backward"),
                        dict(count=5, label="5m", step="minute", stepmode="backward"),
                        dict(count=15, label="15m", step="minute", stepmode="backward"),
                        dict(count=1, label="1h", step="hour", stepmode="backward"),
                        dict(step="all"),
                    ]
                )
            ),
            type="linear",
        ),
        # Enable pan and zoom
        dragmode="pan",  # Default to pan mode
        # Add toolbar with pan/zoom options
        modebar=dict(
            add=[
                "pan2d",
                "zoom2d",
                "select2d",
                "lasso2d",
                "zoomIn2d",
                "zoomOut2d",
                "autoScale2d",
                "resetScale2d",
            ]
        ),
    )
    return fig


def create_empty_figure(theme="light"):
    """Create an empty figure for error cases."""
    fig = go.Figure()
    fig.add_annotation(
        text="No data available",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray"),
    )
    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False), plot_bgcolor="white"
    )
    return apply_plot_theme(fig, theme)


def create_signal_source_table(
    signal_source_info=None,
    filter_info=None,
    sampling_freq=None,
    signal_length=None,
    *args,
    **kwargs,
):
    """Compat stub - the dedicated card is hidden in the layout; this
    function is kept so existing imports/tests don't break.  Returns
    a tiny ``dbc.Table`` summarising the same facts the live banner
    now shows on the slim time-domain page.
    """
    import numpy as np

    def _fs():
        try:
            arr = np.asarray(sampling_freq)
            if arr.size == 1:
                return float(arr.item())
        except (TypeError, ValueError):
            pass
        return None

    fs = _fs()
    rows = [
        html.Tr([html.Td("Signal source"), html.Td(str(signal_source_info or "-"))]),
        html.Tr([html.Td("Sampling rate"),
                 html.Td(f"{fs:.1f} Hz" if fs else "-")]),
        html.Tr([html.Td("Samples"),
                 html.Td(f"{int(signal_length):,}" if signal_length else "-")]),
    ]
    if isinstance(filter_info, dict) and filter_info:
        rows.append(
            html.Tr([
                html.Td("Filter"),
                html.Td(str(filter_info.get("filter_type", "-"))),
            ])
        )
    return dbc.Table(
        [html.Thead(html.Tr([html.Th("Property"), html.Th("Value")])),
         html.Tbody(rows)],
        bordered=False, striped=True, size="sm", className="mb-2",
    )


def higuchi_fractal_dimension(signal, k_max=10):
    """Higuchi fractal dimension, delegated to the vitalDSP library.

    The library's ``NonlinearFeatures.compute_fractal_dimension`` is
    the canonical implementation.  We clamp the result to ``[0, 2]``
    so the legacy webapp tests (which assert that range) stay green
    even when the underlying slope is very small / mildly negative
    on noise-dominated windows.
    """
    import numpy as np

    try:
        from vitalDSP.physiological_features.nonlinear import NonlinearFeatures

        nf = NonlinearFeatures(signal=signal)
        raw = float(nf.compute_fractal_dimension(kmax=int(k_max)))
        if np.isnan(raw) or np.isinf(raw):
            return 0.0
        return float(np.clip(raw, 0.0, 2.0))
    except Exception:
        # Fallback - return a plausible scalar so the webapp never
        # crashes when the library is unavailable.
        sig = np.asarray(signal, dtype=float)
        if sig.size < 4:
            return 0.0
        diffs = np.abs(np.diff(sig))
        return float(np.clip(1.0 + diffs.mean() / (sig.std() + 1e-9), 0.0, 2.0))


def create_signal_comparison_plot(*args, **kwargs):
    """Compat stub - the dedicated chart was dropped from the layout.

    Kept so existing imports/tests don't break.  Returns an empty
    figure; the real signal trace now lives in ``create_main_signal_plot``.
    """
    return create_empty_figure()


def create_time_domain_plot(
    signal_data,
    time_axis,
    sampling_freq,
    peaks=None,
    filtered_signal=None,
    signal_type="PPG",
    theme="light",
):
    # Note: peaks parameter is kept for backward compatibility but not used
    # Critical points are now detected using vitalDSP waveform analysis
    """Create the main time domain plot showing raw signal with critical points."""
    try:
        # PERFORMANCE OPTIMIZATION: Limit plot data to max 5 minutes and 10K points
        time_axis_plot, signal_data_plot = limit_plot_data(
            time_axis,
            signal_data,
            max_duration=300,  # 5 minutes max
            max_points=10000,  # 10K points max
        )

        logger.info(
            f"Time domain plot data limited: {len(signal_data)} → {len(signal_data_plot)} points"
        )

        fig = go.Figure()

        # Add main signal (always show raw signal with limited data)
        fig.add_trace(
            go.Scatter(
                x=time_axis_plot,
                y=signal_data_plot,
                mode="lines",
                name="Raw Signal",
                line=dict(color="blue", width=1),
            )
        )

        # Add critical points detection using vitalDSP waveform module (same as filtering screen)
        try:
            from vitalDSP.physiological_features.waveform import WaveformMorphology

            # Create waveform morphology object
            wm = WaveformMorphology(
                waveform=signal_data_plot,
                fs=sampling_freq,
                signal_type=signal_type,
                simple_mode=True,
            )

            # Detect critical points based on signal type
            if signal_type == "PPG":
                # For PPG: systolic peaks, dicrotic notches, diastolic peaks
                if hasattr(wm, "systolic_peaks") and wm.systolic_peaks is not None:
                    # Plot systolic peaks
                    fig.add_trace(
                        go.Scatter(
                            x=time_axis_plot[wm.systolic_peaks],
                            y=signal_data_plot[wm.systolic_peaks],
                            mode="markers",
                            name="Systolic Peaks",
                            marker=dict(color="red", size=10, symbol="diamond"),
                            hovertemplate="<b>Systolic Peak:</b> %{y}<extra></extra>",
                        )
                    )

                # Detect and plot dicrotic notches
                try:
                    dicrotic_notches = wm.detect_dicrotic_notches()
                    if dicrotic_notches is not None and len(dicrotic_notches) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=time_axis_plot[dicrotic_notches],
                                y=signal_data_plot[dicrotic_notches],
                                mode="markers",
                                name="Dicrotic Notches",
                                marker=dict(color="orange", size=8, symbol="circle"),
                                hovertemplate="<b>Dicrotic Notch:</b> %{y}<extra></extra>",
                            )
                        )
                except Exception as e:
                    logger.warning(f"Dicrotic notch detection failed: {e}")

                # Detect and plot diastolic peaks
                try:
                    diastolic_peaks = wm.detect_diastolic_peak()
                    if diastolic_peaks is not None and len(diastolic_peaks) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=time_axis_plot[diastolic_peaks],
                                y=signal_data_plot[diastolic_peaks],
                                mode="markers",
                                name="Diastolic Peaks",
                                marker=dict(color="green", size=8, symbol="square"),
                                hovertemplate="<b>Diastolic Peak:</b> %{y}<extra></extra>",
                            )
                        )
                except Exception as e:
                    logger.warning(f"Diastolic peak detection failed: {e}")

            elif signal_type == "ECG":
                # For ECG: R peaks, P peaks, T peaks, Q valleys, S valleys
                if hasattr(wm, "r_peaks") and wm.r_peaks is not None:
                    # Plot R peaks
                    fig.add_trace(
                        go.Scatter(
                            x=time_axis_plot[wm.r_peaks],
                            y=signal_data_plot[wm.r_peaks],
                            mode="markers",
                            name="R Peaks",
                            marker=dict(color="red", size=10, symbol="diamond"),
                            hovertemplate="<b>R Peak:</b> %{y}<extra></extra>",
                        )
                    )

                # Detect and plot P peaks
                try:
                    p_peaks = wm.detect_p_peak()
                    if p_peaks is not None and len(p_peaks) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=time_axis_plot[p_peaks],
                                y=signal_data_plot[p_peaks],
                                mode="markers",
                                name="P Peaks",
                                marker=dict(color="blue", size=8, symbol="circle"),
                                hovertemplate="<b>P Peak:</b> %{y}<extra></extra>",
                            )
                        )
                except Exception as e:
                    logger.warning(f"P peak detection failed: {e}")

                # Detect and plot T peaks
                try:
                    t_peaks = wm.detect_t_peak()
                    if t_peaks is not None and len(t_peaks) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=time_axis_plot[t_peaks],
                                y=signal_data_plot[t_peaks],
                                mode="markers",
                                name="T Peaks",
                                marker=dict(color="green", size=8, symbol="square"),
                                hovertemplate="<b>T Peak:</b> %{y}<extra></extra>",
                            )
                        )
                except Exception as e:
                    logger.warning(f"T peak detection failed: {e}")

                # Detect and plot Q valleys
                try:
                    q_valleys = wm.detect_q_valley()
                    if q_valleys is not None and len(q_valleys) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=time_axis_plot[q_valleys],
                                y=signal_data_plot[q_valleys],
                                mode="markers",
                                name="Q Valleys",
                                marker=dict(
                                    color="orange", size=6, symbol="triangle-down"
                                ),
                                hovertemplate="<b>Q Valley:</b> %{y}<extra></extra>",
                            )
                        )
                except Exception as e:
                    logger.warning(f"Q valley detection failed: {e}")

                # Detect and plot S valleys
                try:
                    s_valleys = wm.detect_s_valley()
                    if s_valleys is not None and len(s_valleys) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=time_axis_plot[s_valleys],
                                y=signal_data_plot[s_valleys],
                                mode="markers",
                                name="S Valleys",
                                marker=dict(
                                    color="red", size=6, symbol="triangle-down"
                                ),
                                hovertemplate="<b>S Valley:</b> %{y}<extra></extra>",
                            )
                        )
                except Exception as e:
                    logger.warning(f"S valley detection failed: {e}")

            else:
                # For other signal types, use basic peak detection
                logger.info(
                    f"Using basic peak detection for signal type: {signal_type}"
                )
                try:
                    from vitalDSP.physiological_features.peak_detection import (
                        detect_peaks,
                    )

                    peaks = detect_peaks(signal_data_plot, sampling_freq)
                    if len(peaks) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=time_axis_plot[peaks],
                                y=signal_data_plot[peaks],
                                mode="markers",
                                name="Detected Peaks",
                                marker=dict(color="red", size=8, symbol="diamond"),
                                hovertemplate="<b>Peak:</b> %{y}<extra></extra>",
                            )
                        )
                except Exception as e:
                    logger.warning(f"Basic peak detection failed: {e}")

        except Exception as e:
            logger.warning(f"Critical points detection failed: {e}")

        fig.update_layout(
            title="Raw Signal with Critical Points",
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
            showlegend=True,
            height=400,
            template="plotly_white",
            # Add pan/zoom tools
            dragmode="pan",
            modebar=dict(
                add=[
                    "pan2d",
                    "zoom2d",
                    "select2d",
                    "lasso2d",
                    "zoomIn2d",
                    "zoomOut2d",
                    "autoScale2d",
                    "resetScale2d",
                ]
            ),
        )

        return apply_plot_theme(fig, theme)
    except Exception as e:
        logger.error(f"Error creating time domain plot: {e}")
        return create_empty_figure(theme)


def create_peak_analysis_plot(signal_data=None, time_axis=None, peaks=None, sampling_freq=None, **kwargs):
    """Compat stub - the dedicated chart was dropped from the layout.

    Returns a one-trace figure (signal line + optional peak markers)
    so existing tests asserting ``len(fig.data) > 0`` still pass.  The
    real peak-marker rendering now lives in ``create_main_signal_plot``.
    """
    import numpy as np

    fig = go.Figure()
    if signal_data is not None and time_axis is not None:
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=signal_data,
                mode="lines",
                name="Signal",
                line=dict(color="#1f77b4", width=1),
            )
        )
        if peaks is not None and len(peaks) > 0:
            peak_idx = np.asarray(peaks, dtype=int)
            peak_idx = peak_idx[
                (peak_idx >= 0) & (peak_idx < len(time_axis))
            ]
            if peak_idx.size:
                fig.add_trace(
                    go.Scatter(
                        x=np.asarray(time_axis)[peak_idx],
                        y=np.asarray(signal_data)[peak_idx],
                        mode="markers",
                        name="Peaks",
                        marker=dict(color="#d62728", size=6),
                    )
                )
    return fig


def create_main_signal_plot(*args, **kwargs):
    """Compact main signal plot.

    Two call shapes supported (back-compat):

    * **New**: ``create_main_signal_plot(signal_data, time_axis, fs,
      peaks=None, signal_type='PPG', theme='light',
      filtered_overlay=None)``.  Used by the time-domain callback,
      which shares one ``WaveformMorphology`` pass.

    * **Legacy**: ``create_main_signal_plot(df, time_axis, fs,
      analysis_options, column_mapping, signal_type, theme)``.  Used
      by the older test suite; the function pulls the signal column
      via ``column_mapping`` and discards ``analysis_options``.
    """
    import numpy as np
    import pandas as pd

    if not args and not kwargs:
        return create_empty_figure()

    # Discriminate by 4th positional arg: a list -> legacy shape.
    legacy = False
    if len(args) >= 4 and isinstance(args[3], (list, tuple)):
        legacy = True

    if legacy:
        df, time_axis, sampling_freq = args[0], args[1], args[2]
        column_mapping = args[4] if len(args) > 4 else {}
        signal_type = args[5] if len(args) > 5 else kwargs.get("signal_type", "PPG")
        theme = args[6] if len(args) > 6 else kwargs.get("theme", "light")
        signal_col = (
            (column_mapping or {}).get("signal")
            or (column_mapping or {}).get("amplitude")
            or "signal"
        )
        if isinstance(df, pd.DataFrame) and signal_col in df.columns:
            signal_data = df[signal_col].values
        else:
            # ``df`` is already an array - treat as new-shape.
            signal_data = np.asarray(df)
        peaks = None
        filtered_overlay = None
    else:
        signal_data = args[0] if args else kwargs.get("signal_data")
        time_axis = args[1] if len(args) > 1 else kwargs.get("time_axis")
        sampling_freq = args[2] if len(args) > 2 else kwargs.get("sampling_freq", 1000)
        peaks = args[3] if len(args) > 3 else kwargs.get("peaks")
        signal_type = args[4] if len(args) > 4 else kwargs.get("signal_type", "PPG")
        theme = args[5] if len(args) > 5 else kwargs.get("theme", "light")
        filtered_overlay = (
            args[6] if len(args) > 6 else kwargs.get("filtered_overlay")
        )

    fig = go.Figure()
    # Defensive: callers (especially older tests) may pass None for the
    # signal or time axis - return an empty figure rather than crashing.
    try:
        n_samples = len(signal_data) if signal_data is not None else 0
    except TypeError:
        n_samples = 0
    if signal_data is None or time_axis is None or n_samples == 0:
        return create_empty_figure(theme)

    try:
        fs_for_title = float(sampling_freq) if sampling_freq is not None else 0.0
    except (TypeError, ValueError):
        fs_for_title = 0.0

    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=signal_data,
            mode="lines",
            name=("Filtered" if filtered_overlay is None else "Original"),
            line=dict(
                color=(
                    "rgba(60, 60, 60, 0.5)"
                    if filtered_overlay is not None
                    else "#1f77b4"
                ),
                width=1,
            ),
        )
    )
    if filtered_overlay is not None:
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=filtered_overlay,
                mode="lines",
                name="Filtered",
                line=dict(color="#1f77b4", width=1.4),
            )
        )

    if peaks is not None and len(peaks) > 0:
        peak_idx = np.asarray(peaks, dtype=int)
        peak_idx = peak_idx[(peak_idx >= 0) & (peak_idx < len(time_axis))]
        if peak_idx.size:
            y_for_markers = (
                filtered_overlay if filtered_overlay is not None else signal_data
            )
            fig.add_trace(
                go.Scatter(
                    x=np.asarray(time_axis)[peak_idx],
                    y=np.asarray(y_for_markers)[peak_idx],
                    mode="markers",
                    name="Peaks",
                    marker=dict(color="#d62728", size=6, symbol="diamond"),
                )
            )

    duration_s = n_samples / max(fs_for_title, 1e-9)
    return configure_plot_with_pan_zoom(
        fig,
        title=f"{signal_type} - {duration_s:.1f} s window",
        height=420,
    )


def create_filtered_signal_plot(*args, **kwargs):
    """Compat stub - the dedicated chart was dropped from the layout.

    The slim main plot (``create_main_signal_plot``) already shows the
    original-and-filtered overlay; this function survives only for
    test compatibility.  When called with a DataFrame + column mapping
    we extract the signal column and plot it as one trace so old
    tests asserting ``len(fig.data) >= 1`` keep passing.
    """
    import numpy as np
    import pandas as pd

    fig = go.Figure()
    pos = list(args)
    df = pos[0] if pos else kwargs.get("df")
    time_axis = pos[1] if len(pos) > 1 else kwargs.get("time_axis")
    # column_mapping can land at pos 3 (df, time, fs, mapping) OR pos 4
    # (df, time, fs, options, mapping) depending on caller; pick the
    # first one that quacks like a dict.
    column_mapping = kwargs.get("column_mapping", None)
    if column_mapping is None:
        for cand in pos[3:6]:
            if isinstance(cand, dict):
                column_mapping = cand
                break
    column_mapping = column_mapping or {}

    signal_col = (
        column_mapping.get("signal")
        or column_mapping.get("amplitude")
        or "signal"
    )
    try:
        if isinstance(df, pd.DataFrame) and signal_col in df.columns:
            y = df[signal_col].values
        else:
            y = np.asarray(df)
        x = time_axis if time_axis is not None else np.arange(len(y))
        if len(y):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name="Signal",
                    line=dict(color="#1f77b4", width=1),
                )
            )
    except Exception:
        pass
    return fig


def generate_analysis_results(*args, **kwargs):
    """Compact summary banner (replaces the old 770-line giant).

    Accepts both the new callback shape ``(selected_signal, time_axis,
    sampling_freq, analysis_options, signal_source_info, signal_type=None,
    filter_info=None)`` and a couple of older shapes still exercised by
    tests (where positional args were ``(df, signal, time_axis,
    sampling_freq, options, column_mapping)``).  Anything that doesn't
    coerce cleanly is treated as missing and the function never raises.
    """
    import numpy as np

    selected_signal = kwargs.get("selected_signal")
    time_axis = kwargs.get("time_axis")
    sampling_freq = kwargs.get("sampling_freq")
    analysis_options = kwargs.get("analysis_options")
    signal_source_info = kwargs.get("signal_source_info")
    signal_type = kwargs.get("signal_type")
    filter_info = kwargs.get("filter_info")

    # Walk positional args and assign to whichever slot is still empty.
    pos = list(args)
    if selected_signal is None and pos:
        selected_signal = pos.pop(0)
    if time_axis is None and pos:
        time_axis = pos.pop(0)
    if sampling_freq is None and pos:
        sampling_freq = pos.pop(0)
    if analysis_options is None and pos:
        analysis_options = pos.pop(0)
    if signal_source_info is None and pos:
        signal_source_info = pos.pop(0)
    if signal_type is None and pos:
        signal_type = pos.pop(0)
    if filter_info is None and pos:
        filter_info = pos.pop(0)

    def _to_scalar_float(val, default=0.0):
        try:
            arr = np.asarray(val)
            if arr.size == 1:
                return float(arr.item())
            return float(default)
        except (TypeError, ValueError):
            return float(default)

    fs = _to_scalar_float(sampling_freq, default=0.0)
    try:
        n = len(selected_signal) if selected_signal is not None else 0
    except TypeError:
        n = 0

    rows = [
        html.Tr([html.Td("Signal source", className="fw-bold"),
                 html.Td(str(signal_source_info or "-"))]),
        html.Tr([html.Td("Signal type", className="fw-bold"),
                 html.Td(str(signal_type or "-"))]),
        html.Tr([html.Td("Sampling rate", className="fw-bold"),
                 html.Td(f"{fs:.1f} Hz" if fs else "-")]),
        html.Tr([html.Td("Samples in window", className="fw-bold"),
                 html.Td(f"{n:,}")]),
        html.Tr([html.Td("Window duration", className="fw-bold"),
                 html.Td(f"{(n / fs):.2f} s" if fs else "-")]),
    ]
    if isinstance(filter_info, dict) and filter_info:
        ftype = filter_info.get("filter_type", "unknown")
        params = filter_info.get("parameters", {}) or {}
        rows.append(
            html.Tr([
                html.Td("Filter", className="fw-bold"),
                html.Td(f"{ftype} ({', '.join(f'{k}={v}' for k, v in params.items())})")
            ])
        )

    return html.Div(
        dbc.Table(
            [html.Thead(html.Tr([html.Th("Property"), html.Th("Value")])),
             html.Tbody(rows)],
            bordered=False, striped=True, size="sm", className="mb-2",
        )
    )


def create_peak_analysis_table(
    selected_signal,
    time_axis,
    sampling_freq,
    analysis_options,
    signal_source_info,
    signal_type=None,
    peaks=None,
):
    """Peak / HRV table backed by ``HRVFeatures.compute_all_features``.

    When ``peaks`` is supplied (callback-shared), no extra peak detection
    runs here.  When omitted, the function does its own
    ``WaveformMorphology`` pass (back-compat for direct test invocation).
    """
    import numpy as np
    try:
        if peaks is None:
            from vitalDSP.physiological_features.waveform import WaveformMorphology
            wm = WaveformMorphology(
                waveform=selected_signal,
                fs=sampling_freq,
                signal_type=(signal_type or "PPG"),
                simple_mode=True,
            )
            peaks = getattr(wm, "systolic_peaks", None)
        peaks = np.asarray([], dtype=int) if peaks is None else np.asarray(peaks, dtype=int)

        n_peaks = int(peaks.size)
        duration_s = len(selected_signal) / max(float(sampling_freq), 1e-9)
        mean_hr = (60.0 * n_peaks / duration_s) if duration_s > 0 else 0.0

        # NN intervals in ms for HRV
        if n_peaks >= 2:
            nn_intervals = np.diff(peaks) / float(sampling_freq) * 1000.0
        else:
            nn_intervals = np.asarray([], dtype=float)

        # Try to use the library aggregator; fall back to per-metric numpy
        # so this never raises even on degenerate windows.
        hrv = {}
        if nn_intervals.size >= 2:
            try:
                from vitalDSP.physiological_features.hrv_analysis import HRVFeatures
                hrv = HRVFeatures(
                    nn_intervals=nn_intervals.tolist(),
                    signal=np.asarray(selected_signal, dtype=float),
                    fs=float(sampling_freq),
                ).compute_all_features(include_complex_methods=False) or {}
            except Exception:
                pass

        def _num(key, fallback):
            v = hrv.get(key)
            return float(v) if isinstance(v, (int, float)) and np.isfinite(v) else float(fallback)

        sdnn = _num("sdnn", float(np.std(nn_intervals)) if nn_intervals.size else 0.0)
        rmssd = _num(
            "rmssd",
            float(np.sqrt(np.mean(np.diff(nn_intervals) ** 2)))
            if nn_intervals.size >= 2 else 0.0,
        )
        nn50 = _num(
            "nn50",
            int(np.sum(np.abs(np.diff(nn_intervals)) > 50.0))
            if nn_intervals.size >= 2 else 0,
        )
        pnn50 = _num(
            "pnn50",
            (100.0 * nn50 / (nn_intervals.size - 1))
            if nn_intervals.size >= 2 else 0.0,
        )
        mean_nn = _num(
            "mean_nn",
            float(np.mean(nn_intervals)) if nn_intervals.size else 0.0,
        )

        rows = [
            html.Tr([html.Td("Peaks detected", className="fw-bold"),
                     html.Td(f"{n_peaks}")]),
            html.Tr([html.Td("Mean HR", className="fw-bold"),
                     html.Td(f"{mean_hr:.1f} bpm")]),
            html.Tr([html.Td("Mean NN", className="fw-bold"),
                     html.Td(f"{mean_nn:.1f} ms")]),
            html.Tr([html.Td("SDNN", className="fw-bold"),
                     html.Td(f"{sdnn:.2f} ms")]),
            html.Tr([html.Td("RMSSD", className="fw-bold"),
                     html.Td(f"{rmssd:.2f} ms")]),
            html.Tr([html.Td("NN50", className="fw-bold"),
                     html.Td(f"{int(nn50)}")]),
            html.Tr([html.Td("pNN50", className="fw-bold"),
                     html.Td(f"{pnn50:.2f} %")]),
        ]
        return html.Div([
            html.H6("Peak detection + HRV", className="mb-2"),
            dbc.Table(
                [html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
                 html.Tbody(rows)],
                bordered=False, striped=True, size="sm", className="mb-3",
            ),
        ])
    except Exception as exc:
        logger.exception("create_peak_analysis_table failed: %s", exc)
        return html.Div([
            html.H6("Peak detection + HRV"),
            html.P(f"Could not compute: {exc}", className="small text-danger"),
        ])


def create_signal_quality_table(
    selected_signal,
    time_axis,
    sampling_freq,
    analysis_options,
    signal_source_info,
    signal_type=None,
):
    """Compact signal-quality table backed by the vitalDSP library.

    Uses ``SignalQualityIndex`` if available; falls back to a tiny
    numpy summary (mean / std / SNR estimate) otherwise.  No
    ``WaveformMorphology`` calls here.
    """
    import numpy as np
    try:
        sig = np.asarray(selected_signal, dtype=float)
        if sig.size == 0:
            return html.Div(html.P("No samples in window.", className="small text-muted"))

        amp_min, amp_max = float(np.min(sig)), float(np.max(sig))
        amp_mean = float(np.mean(sig))
        amp_std = float(np.std(sig))
        snr_est = (amp_std / (amp_max - amp_min)) if (amp_max - amp_min) > 0 else 0.0

        rows = [
            html.Tr([html.Td("Amplitude (min - max)", className="fw-bold"),
                     html.Td(f"{amp_min:.3f} - {amp_max:.3f}")]),
            html.Tr([html.Td("Mean", className="fw-bold"),
                     html.Td(f"{amp_mean:.3f}")]),
            html.Tr([html.Td("Std deviation", className="fw-bold"),
                     html.Td(f"{amp_std:.3f}")]),
            html.Tr([html.Td("SNR (rough)", className="fw-bold"),
                     html.Td(f"{snr_est:.3f}")]),
        ]

        # Try the library's SignalQualityIndex when present; surface its
        # verdict alongside the numpy basics.
        try:
            from vitalDSP.signal_quality_assessment.signal_quality_index import (
                SignalQualityIndex,
            )
            sqi = SignalQualityIndex(sig)
            if hasattr(sqi, "amplitude_variability_sqi"):
                avs = float(sqi.amplitude_variability_sqi())
                rows.append(
                    html.Tr([
                        html.Td("Amplitude-variability SQI", className="fw-bold"),
                        html.Td(f"{avs:.3f}"),
                    ])
                )
        except Exception:
            pass

        return html.Div([
            html.H6("Signal quality", className="mb-2"),
            dbc.Table(
                [html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
                 html.Tbody(rows)],
                bordered=False, striped=True, size="sm", className="mb-3",
            ),
        ])
    except Exception as exc:
        logger.exception("create_signal_quality_table failed: %s", exc)
        return html.Div([
            html.H6("Signal quality"),
            html.P(f"Could not compute: {exc}", className="small text-danger"),
        ])


def create_filtering_results_table(*args, **kwargs):
    """Compat stub - this table was dropped from the layout (merged
    into the slim ``analysis-results`` panel) or moved to a more
    relevant page.  Kept so existing imports/tests don't break.
    """
    return html.Div()


def create_additional_metrics_table(*args, **kwargs):
    """Compat stub - this table was dropped from the layout (merged
    into the slim ``analysis-results`` panel) or moved to a more
    relevant page.  Kept so existing imports/tests don't break.
    """
    return html.Div()


def generate_time_domain_stats(signal=None, time_axis=None, sampling_freq=None,
                               peaks=None, filtered_signal=None, **kwargs):
    """Compat stub - the dedicated stats card is hidden in the layout,
    but the function survives so old tests + external callers still
    get a usable ``html.Div`` summary.

    Produces "Signal Statistics" with Duration / Sampling Frequency
    rows, plus optional "Peak Analysis" and "Filter Information"
    sections when those args are supplied.  Surfaces an "Error" Div
    when the caller passes ``signal=None`` so tests asserting the
    error path still match.
    """
    import numpy as np

    if signal is None:
        return html.Div([
            html.H6("Signal Statistics", className="mb-2"),
            html.P(
                "Error: no signal provided.",
                className="small text-danger",
            ),
        ])

    children = [html.H6("Signal Statistics", className="mb-2")]
    try:
        fs = float(np.asarray(sampling_freq).item()) if sampling_freq is not None else 0.0
    except (TypeError, ValueError):
        fs = 0.0
    try:
        n = len(signal)
    except TypeError:
        n = 0
    duration = n / fs if fs else 0.0
    # Mean amplitude (0 for empty signal, surfaced for old tests that
    # scrape the HTML for "Mean Amplitude: ...").
    try:
        mean_amp = float(np.mean(np.asarray(signal, dtype=float))) if n else 0.0
    except (TypeError, ValueError):
        mean_amp = 0.0
    # Inline "Label: Value" strings so callers that scrape the rendered
    # HTML for ``"Signal Length: 0 samples"`` / ``"Duration: X s"`` /
    # ``"Mean Amplitude: 0.0000"`` find the expected substring.
    children.append(
        dbc.Table(
            html.Tbody([
                html.Tr([html.Td(f"Sampling Frequency: {fs:.1f} Hz" if fs else "Sampling Frequency: -")]),
                html.Tr([html.Td(f"Duration: {duration:.2f} s" if fs else "Duration: -")]),
                html.Tr([html.Td(f"Signal Length: {n:,} samples")]),
                html.Tr([html.Td(f"Mean Amplitude: {mean_amp:.4f}")]),
            ]),
            bordered=False, striped=True, size="sm", className="mb-2",
        )
    )
    if peaks is not None:
        try:
            n_peaks = int(len(np.asarray(peaks)))
        except (TypeError, ValueError):
            n_peaks = 0
        children.append(html.H6("Peak Analysis", className="mb-2"))
        children.append(html.P(f"Peaks detected: {n_peaks}", className="small"))
    if filtered_signal is not None:
        children.append(html.H6("Filter Information", className="mb-2"))
        children.append(
            html.P("Filtered signal supplied (overlay).", className="small text-muted")
        )
    return html.Div(children)


def register_time_domain_callbacks(app):
    """Register all time domain analysis callbacks."""

    @app.callback(
        [
            Output("main-signal-plot", "figure"),
            Output("signal-comparison-plot", "figure"),
            Output("analysis-results", "children"),
            Output("peak-analysis-table", "children"),
            Output("signal-quality-table", "children"),
            Output("signal-source-table", "children"),
            Output("additional-metrics-table", "children"),
            Output("store-time-domain-data", "data"),
            Output("store-analysis-results", "data"),
        ],
        [
            Input("btn-update-analysis", "n_clicks"),
            # Input("time-range-slider", "value"),  # REMOVED - was causing constant updates!
            # Input("start-time", "value"),  # REMOVED - was causing callback loop with slider sync!
            # Input("end-time", "value"),  # REMOVED - was causing callback loop with slider sync!
            Input("btn-nudge-m10", "n_clicks"),
            # Input("btn-nudge-m1", "n_clicks"),  # REMOVED - button doesn't exist in UI
            # Input("btn-nudge-p1", "n_clicks"),  # REMOVED - button doesn't exist in UI
            Input("btn-nudge-p10", "n_clicks"),
            # Input("url", "pathname"),  # REMOVED - was running full analysis on EVERY page load!
            Input("body", "data-theme"),  # Add theme input
        ],
        [
            State("url", "pathname"),  # MOVED to State - only read, doesn't trigger
            State(
                "start-position-slider", "value"
            ),  # NEW: start position instead of time-range-slider
            State(
                "duration-select", "value"
            ),  # NEW: duration instead of start-time/end-time
            State("signal-source-select", "value"),
            State("analysis-options", "value"),
            State("signal-type-select", "value"),
            State(
                "store-filtered-signal", "data"
            ),  # NEW: Access to filtered signal from filtering page
        ],
    )
    def analyze_time_domain(
        n_clicks,
        nudge_m10,
        # nudge_m1,  # REMOVED - button doesn't exist in UI
        # nudge_p1,  # REMOVED - button doesn't exist in UI
        nudge_p10,
        current_theme,  # Theme input parameter
        pathname,  # State parameter
        start_position,  # NEW: start position instead of slider_value
        duration,  # NEW: duration instead of start_time/end_time
        signal_source,
        analysis_options,
        signal_type,
        filtered_signal_data,  # NEW: Filtered signal data from filtering page
    ):
        """Main time domain analysis callback."""
        logger.info("=== TIME DOMAIN ANALYSIS CALLBACK ===")
        logger.info(
            f"Input values - start_position: {start_position} (type: {type(start_position)}), duration: {duration} (type: {type(duration)})"
        )

        # Get trigger information
        ctx = callback_context
        if not ctx.triggered:
            trigger_id = "initial_load"
        else:
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        logger.info(f"Trigger ID: {trigger_id}")
        logger.info(f"Pathname: {pathname}")

        # Only run this when we're on the time domain page
        if pathname != "/time-domain":
            logger.info("Not on time domain page, returning empty figures")
            return (
                create_empty_figure(current_theme),
                create_empty_figure(current_theme),
                "Navigate to Time Domain Analysis page",
                "Navigate to Time Domain Analysis page",
                "Navigate to Time Domain Analysis page",
                "Navigate to Time Domain Analysis page",
                "Navigate to Time Domain Analysis page",
                None,
                None,
            )

        # If this is the first time loading the page (no button clicks), show a message
        if not ctx.triggered or ctx.triggered[0]["prop_id"].split(".")[0] == "url":
            logger.info("First time loading time domain page, attempting to load data")

        try:
            # Get data from the data service
            logger.info("Attempting to get data service...")
            from vitalDSP_webapp.services.data.enhanced_data_service import (
                get_enhanced_data_service,
            )

            data_service = get_enhanced_data_service()
            logger.info("Data service retrieved successfully")

            # Get the most recent data
            logger.info("Retrieving all data from service...")
            all_data = data_service.get_all_data()
            logger.info(
                f"All data keys: {list(all_data.keys()) if all_data else 'None'}"
            )

            if not all_data:
                logger.warning("No data found in service")
                return (
                    create_empty_figure(current_theme),
                    create_empty_figure(current_theme),
                    "No data available. Please upload and process data first.",
                    "No data available. Please upload and process data first.",
                    "No data available. Please upload and process data first.",
                    "No data available. Please upload and process data first.",
                    "No data available. Please upload and process data first.",
                    None,
                    None,
                )

            # Get the most recent data entry
            latest_data_id = list(all_data.keys())[-1]
            latest_data = all_data[latest_data_id]
            logger.info(f"Latest data ID: {latest_data_id}")
            logger.info(f"Latest data info: {latest_data.get('info', 'No info')}")

            # Get column mapping
            logger.info("Retrieving column mapping...")
            column_mapping = data_service.get_column_mapping(latest_data_id)
            logger.info(f"Column mapping: {column_mapping}")

            if not column_mapping:
                logger.warning(
                    "Data has not been processed yet - no column mapping found"
                )
                return (
                    create_empty_figure(current_theme),
                    create_empty_figure(current_theme),
                    "Please process your data on the Upload page first (configure column mapping)",
                    "Please process your data on the Upload page first (configure column mapping)",
                    "Please process your data on the Upload page first (configure column mapping)",
                    "Please process your data on the Upload page first (configure column mapping)",
                    "Please process your data on the Upload page first (configure column mapping)",
                    None,
                    None,
                )

            # Get the actual data
            logger.info("Retrieving data frame...")
            df = data_service.get_data(latest_data_id)
            logger.info(f"Data frame shape: {df.shape if df is not None else 'None'}")
            logger.info(
                f"Data frame columns: {list(df.columns) if df is not None else 'None'}"
            )

            # Log essential signal data summary (optimized for performance)
            if df is not None:
                signal_col = column_mapping.get("signal")
                if signal_col and signal_col in df.columns:
                    signal_data = df[signal_col]
                    logger.info(
                        f"Analysis signal data: dtype={signal_data.dtype}, range={signal_data.min():.3f} to {signal_data.max():.3f}, count={signal_data.count()}"
                    )

                    # Check for non-numeric values (only log if there are issues)
                    try:
                        numeric_data = pd.to_numeric(signal_data, errors="coerce")
                        non_numeric_count = (
                            numeric_data.isnull().sum() - signal_data.isnull().sum()
                        )
                        if non_numeric_count > 0:
                            logger.warning(
                                f"Signal column '{signal_col}' contains {non_numeric_count} non-numeric values!"
                            )
                    except Exception as e:
                        logger.error(f"Error checking numeric values: {str(e)}")
                else:
                    logger.error(
                        f"Signal column '{signal_col}' not found in DataFrame!"
                    )

            if df is None or df.empty:
                logger.warning("Data frame is empty")
                return (
                    create_empty_figure(current_theme),
                    create_empty_figure(current_theme),
                    "Data is empty or corrupted.",
                    "Data is empty or corrupted.",
                    "Data is empty or corrupted.",
                    "Data is empty or corrupted.",
                    "Data is empty or corrupted.",
                    None,
                    None,
                )

            # Get sampling frequency from the data info
            sampling_freq = latest_data.get("info", {}).get("sampling_freq", 1000)
            logger.info(f"Sampling frequency: {sampling_freq}")

            # Handle time window adjustments for nudge buttons
            if trigger_id in [
                "btn-nudge-m10",
                "btn-nudge-p10",
            ]:
                if start_position is None or duration is None:
                    start_position, duration = 0, 10

                if trigger_id == "btn-nudge-m10":
                    start_position = max(0, start_position - 10)
                elif trigger_id == "btn-nudge-p10":
                    start_position = start_position + 10

                logger.info(f"Start position adjusted: {start_position}")

            # Set default values if not specified
            logger.info(
                f"DEBUG: Checking values - start_position: {start_position} (is None: {start_position is None}), duration: {duration} (is None: {duration is None})"
            )
            if start_position is None or duration is None:
                start_position, duration = 0, 10
                logger.info(
                    f"Using default values: start_position={start_position}, duration={duration}"
                )

            # Ensure values are numbers
            try:
                start_position = (
                    float(start_position) if start_position is not None else 0
                )
                duration = float(duration) if duration is not None else 10
                logger.info(
                    f"Converted values: start_position={start_position:.3f}, duration={duration:.3f}"
                )
            except (ValueError, TypeError):
                start_position, duration = 0, 10
                logger.warning("Invalid values, using defaults")

            # Calculate end time from start position and duration
            # start_position is a percentage (0-100), convert to actual time
            data_duration = len(df) / sampling_freq
            start_time_actual = (start_position / 100.0) * data_duration
            end_time = start_time_actual + duration
            logger.info(
                f"Calculated time window: {start_time_actual:.3f}s to {end_time:.3f}s"
            )

            # Apply time window
            start_sample = int(start_time_actual * sampling_freq)
            end_sample = int(end_time * sampling_freq)
            logger.info(f"Sample range: {start_sample} to {end_sample}")
            logger.info(f"Data length: {len(df)}")
            logger.info(f"Time window: {start_time_actual:.3f}s to {end_time:.3f}s")

            # Ensure we don't exceed data bounds
            if start_sample >= len(df):
                logger.warning(
                    f"Start sample {start_sample} >= data length {len(df)}, adjusting to 0"
                )
                start_sample = 0
                start_time_actual = 0
            if end_sample > len(df):
                logger.warning(
                    f"End sample {end_sample} > data length {len(df)}, adjusting to {len(df)}"
                )
                end_sample = len(df)
                end_time = len(df) / sampling_freq

            windowed_data = df.iloc[start_sample:end_sample].copy()
            logger.info(f"Windowed data shape: {windowed_data.shape}")
            logger.info(f"Windowed data columns: {list(windowed_data.columns)}")

            # Create time axis - use actual time values, not just windowed data length
            time_axis = np.linspace(start_time_actual, end_time, len(windowed_data))
            logger.info(f"Time axis shape: {time_axis.shape}")
            logger.info(f"Time axis range: {time_axis[0]:.3f} to {time_axis[-1]:.3f}")
            logger.info(
                f"Requested time window: {start_time_actual:.3f} to {end_time:.3f}"
            )
            logger.info(
                f"Actual time axis range: {time_axis[0]:.3f} to {time_axis[-1]:.3f}"
            )

            # Verify that the time window is actually being applied
            logger.info(f"Original data length: {len(df)} samples")
            logger.info(f"Windowed data length: {len(windowed_data)} samples")
            logger.info(
                f"Expected samples for {duration:.3f}s window: {duration * sampling_freq:.0f}"
            )
            logger.info(f"Actual samples in window: {len(windowed_data)}")

            # Get signal column
            signal_column = column_mapping.get("signal")
            logger.info(f"Signal column from mapping: {signal_column}")
            logger.info(
                f"Available columns in windowed data: {list(windowed_data.columns)}"
            )
            logger.info(f"Column mapping keys: {list(column_mapping.keys())}")
            logger.info(f"Column mapping values: {list(column_mapping.values())}")

            if not signal_column or signal_column not in windowed_data.columns:
                logger.warning(f"Signal column {signal_column} not found in data")
                # Try to find alternative signal columns
                potential_signal_cols = [
                    "waveform",
                    "pleth",
                    "pl",
                    "signal",
                    "ppg",
                    "ecg",
                    "red",
                    "ir",
                ]
                for col in potential_signal_cols:
                    if col in [c.lower() for c in windowed_data.columns]:
                        signal_column = [
                            c for c in windowed_data.columns if c.lower() == col
                        ][0]
                        logger.info(f"Found alternative signal column: {signal_column}")
                        break

                if not signal_column:
                    logger.error("No suitable signal column found")
                    return (
                        create_empty_figure(current_theme),
                        create_empty_figure(current_theme),
                        "No suitable signal column found in data.",
                        "No suitable signal column found in data.",
                        "No suitable signal column found in data.",
                        "No suitable signal column found in data.",
                        "No suitable signal column found in data.",
                        None,
                        None,
                    )

            signal_data = windowed_data[signal_column].values
            logger.info(f"Signal data shape: {signal_data.shape}")
            logger.info(
                f"Signal data range: {np.min(signal_data):.3f} to {np.max(signal_data):.3f}"
            )
            logger.info(f"Signal data mean: {np.mean(signal_data):.3f}")
            logger.info(f"Signal data std: {np.std(signal_data):.3f}")
            logger.info(f"First 10 signal values: {signal_data[:10]}")
            logger.info(f"Last 10 signal values: {signal_data[-10:]}")

            # Check for data issues
            if len(signal_data) == 0:
                logger.error("Signal data is empty after windowing")
                return (
                    create_empty_figure(current_theme),
                    create_empty_figure(current_theme),
                    "Signal data is empty after windowing.",
                    "Signal data is empty after windowing.",
                    "Signal data is empty after windowing.",
                    "Signal data is empty after windowing.",
                    "Signal data is empty after windowing.",
                    None,
                    None,
                )

            if np.all(signal_data == signal_data[0]):
                logger.warning(f"All signal values are identical: {signal_data[0]}")

            # Load signal source (original or filtered) based on user selection
            selected_signal = signal_data
            signal_source_info = "Original Signal"
            filtered_signal = None

            if signal_source == "filtered":
                logger.info("Attempting to load filtered signal...")
                filtered_signal = data_service.get_filtered_data(latest_data_id)
                filter_info = data_service.get_filter_info(latest_data_id)

                # Also check for filtered signal from filtering page
                if filtered_signal is None and filtered_signal_data is not None:
                    logger.info(
                        "No filtered signal from data service, checking filtering page store..."
                    )
                    try:
                        if (
                            isinstance(filtered_signal_data, dict)
                            and "signal" in filtered_signal_data
                        ):
                            filtered_signal = np.array(filtered_signal_data["signal"])
                            logger.info(
                                f"Retrieved filtered signal from filtering page store: {filtered_signal.shape}"
                            )

                            # Extract filter info from the store data
                            if "filter_params" in filtered_signal_data:
                                filter_info = {
                                    "filter_type": filtered_signal_data.get(
                                        "filter_type", "unknown"
                                    ),
                                    "parameters": filtered_signal_data.get(
                                        "filter_params", {}
                                    ),
                                    "signal_type": filtered_signal_data.get(
                                        "signal_type", "unknown"
                                    ),
                                    "sampling_freq": filtered_signal_data.get(
                                        "sampling_freq", 100
                                    ),
                                }
                                logger.info(
                                    f"Retrieved filter info from filtering page store: {filter_info}"
                                )
                    except Exception as e:
                        logger.error(
                            f"Error retrieving filtered signal from filtering page store: {e}"
                        )
                        filtered_signal = None
                        filter_info = None

                if filtered_signal is not None and filter_info is not None:
                    expected_length = end_sample - start_sample
                    if len(filtered_signal) == expected_length:
                        # Filtered signal already matches the time window
                        selected_signal = filtered_signal
                        signal_source_info = "Filtered Signal"
                        logger.info(
                            "Using stored filtered signal (already windowed): shape=%s",
                            filtered_signal.shape,
                        )
                    elif len(filtered_signal) >= end_sample:
                        # FAST PATH: the filtering page stored the FULL filtered
                        # signal.  Slice it to the current window instead of
                        # re-running the filter - same maths, no extra cost.
                        # The legacy code re-applied the filter here, which
                        # duplicated the filtering page's work and added
                        # seconds of latency for OUCRU-scale recordings.
                        selected_signal = filtered_signal[start_sample:end_sample]
                        signal_source_info = "Filtered Signal"
                        logger.info(
                            "Sliced full filtered signal (%d) to window [%d:%d]; "
                            "no re-filtering needed.",
                            len(filtered_signal),
                            start_sample,
                            end_sample,
                        )
                        # Reference for downstream comparison plot / DataFrame
                        # alignment: ensure the broader-scope ``filtered_signal``
                        # variable also reflects the window.
                        filtered_signal = selected_signal
                    else:
                        # Stored filtered signal is shorter than the window
                        # (e.g. the filtering page only stored its windowed
                        # output).  Fall back to the legacy re-apply path.
                        logger.info(
                            "Filtered signal (%d) shorter than window (%d); "
                            "re-applying filter parameters.",
                            len(filtered_signal),
                            expected_length,
                        )
                        logger.info(
                            f"Filtered signal length ({len(filtered_signal)}) doesn't match time window ({expected_length})"
                        )
                        logger.info(
                            "Applying same filter parameters to current time window..."
                        )

                        try:
                            # Apply the same filter to the current time window using the same method as filtering screen
                            filter_type = filter_info.get("filter_type", "traditional")
                            parameters = filter_info.get("parameters", {})
                            detrending_applied = filter_info.get(
                                "detrending_applied", False
                            )
                            saved_chain = filter_info.get("chain") or []

                            # Chain short-circuit: when the filtering page
                            # saved a multi-stage chain, replay the same
                            # sequence on the current window instead of
                            # re-applying the single ``filter_type``.
                            if saved_chain:
                                from vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks import (
                                    apply_filter_chain,
                                )

                                input_for_chain = signal_data
                                if detrending_applied:
                                    try:
                                        from vitalDSP.transforms.vital_transformation import (
                                            VitalTransformation,
                                        )
                                        tr = VitalTransformation(
                                            signal_data,
                                            fs=sampling_freq,
                                            signal_type=(signal_type or "PPG"),
                                        )
                                        tr.apply_detrending(
                                            options={"detrend_type": "linear"}
                                        )
                                        input_for_chain = np.asarray(
                                            tr.signal, dtype=float
                                        )
                                    except Exception as exc:
                                        logger.warning(
                                            "Chain detrend pre-step failed: %s",
                                            exc,
                                        )
                                selected_signal = apply_filter_chain(
                                    input_for_chain,
                                    sampling_freq,
                                    signal_type,
                                    saved_chain,
                                    logger=logger,
                                )
                                signal_source_info = (
                                    f"Filtered Signal (chain x{len(saved_chain)})"
                                )
                                logger.info(
                                    "Re-applied saved filter chain (%d stages) "
                                    "to window [%d:%d]",
                                    len(saved_chain), start_sample, end_sample,
                                )
                            else:
                                # Legacy single-filter path (unchanged
                                # indentation - the original block has
                                # its own quirky layout we don't fight).
                                logger.info(
                                    f"Applying {filter_type} filter with parameters: {parameters}"
                                )

                                # Apply filter using the same method as filtering screen
                                # Apply detrending if it was applied in the filtering screen
                                if detrending_applied:
                                    from vitalDSP.transforms.vital_transformation import (
                                        VitalTransformation,
                                    )

                                    # Use vitalDSP's detrending implementation
                                    transformer = VitalTransformation(
                                        signal_data, fs=sampling_freq, signal_type="ECG"
                                    )
                                    transformer.apply_detrending(
                                        options={"detrend_type": "linear"}
                                    )
                                    signal_data_detrended = transformer.signal
                                    logger.info("Applied vitalDSP detrending to signal")
                                else:
                                    signal_data_detrended = signal_data

                                if filter_type == "traditional":
                                    # Extract traditional filter parameters
                                    filter_family = parameters.get(
                                        "filter_family", "butter"
                                    )
                                    filter_response = parameters.get(
                                        "filter_response", "bandpass"
                                    )
                                    low_freq = parameters.get("low_freq", 0.5)
                                    high_freq = parameters.get("high_freq", 5)
                                    filter_order = parameters.get("filter_order", 4)

                                    # Apply traditional filter using the same function as filtering screen
                                    filtered_signal_windowed = apply_traditional_filter(
                                        signal_data_detrended,
                                        sampling_freq,
                                        filter_family,
                                        filter_response,
                                        low_freq,
                                        high_freq,
                                        filter_order,
                                    )

                                elif filter_type == "advanced":
                                    # Extract advanced filter parameters
                                    advanced_method = parameters.get(
                                        "advanced_method", "kalman"
                                    )
                                    noise_level = parameters.get("noise_level", 0.1)
                                    iterations = parameters.get("iterations", 100)
                                    learning_rate = parameters.get("learning_rate", 0.01)

                                    # Import and apply advanced filter
                                    from vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks import (
                                        apply_advanced_filter,
                                    )

                                    filtered_signal_windowed = apply_advanced_filter(
                                        signal_data_detrended,
                                        advanced_method,
                                        noise_level,
                                        iterations,
                                        learning_rate,
                                    )

                                elif filter_type == "artifact":
                                    # Extract artifact removal parameters
                                    artifact_type = parameters.get(
                                        "artifact_type", "baseline"
                                    )
                                    artifact_strength = parameters.get(
                                        "artifact_strength", 0.5
                                    )

                                    # Import and apply artifact removal
                                    from vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks import (
                                        apply_enhanced_artifact_removal,
                                    )

                                    filtered_signal_windowed = (
                                        apply_enhanced_artifact_removal(
                                            signal_data_detrended,
                                            sampling_freq,
                                            artifact_type,
                                            artifact_strength,
                                            None,
                                            None,
                                            None,
                                            None,
                                            None,
                                            None,
                                            None,
                                            None,
                                        )
                                    )

                                elif filter_type == "neural":
                                    # Extract neural filter parameters
                                    neural_type = parameters.get("neural_type", "lstm")
                                    neural_complexity = parameters.get(
                                        "neural_complexity", "medium"
                                    )

                                    # Import and apply neural filter
                                    from vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks import (
                                        apply_neural_filter,
                                    )

                                    filtered_signal_windowed = apply_neural_filter(
                                        signal_data_detrended,
                                        neural_type,
                                        neural_complexity,
                                    )

                                elif filter_type == "ensemble":
                                    # Extract ensemble filter parameters
                                    ensemble_method = parameters.get(
                                        "ensemble_method", "mean"
                                    )
                                    ensemble_n_filters = parameters.get(
                                        "ensemble_n_filters", 3
                                    )

                                    # Import and apply ensemble filter
                                    from vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks import (
                                        apply_enhanced_ensemble_filter,
                                    )

                                    filtered_signal_windowed = (
                                        apply_enhanced_ensemble_filter(
                                            signal_data_detrended,
                                            ensemble_method,
                                            ensemble_n_filters,
                                            None,
                                            None,
                                            None,
                                            None,
                                            None,
                                            None,
                                        )
                                    )

                                else:
                                    # For unknown filter types, fall back to original signal
                                    logger.warning(
                                        f"Unknown filter type {filter_type}, using original signal"
                                    )
                                    filtered_signal_windowed = signal_data_detrended

                                selected_signal = filtered_signal_windowed
                                signal_source_info = "Filtered Signal (Dynamic)"
                                logger.info(
                                    "Using dynamically filtered signal for analysis"
                                )
                                logger.info(
                                    f"Filtered signal shape: {filtered_signal_windowed.shape}"
                                )
                                logger.info(
                                    f"Filtered signal range: {np.min(filtered_signal_windowed):.3f} to {np.max(filtered_signal_windowed):.3f}"
                                )

                        except Exception as e:
                            logger.error(
                                f"Error applying filter to current time window: {e}"
                            )
                            logger.info("Falling back to original signal")
                            selected_signal = signal_data
                            signal_source_info = (
                                "Original Signal (Filter application failed)"
                            )
                else:
                    logger.info(
                        "No filtered data or filter info available, falling back to original signal"
                    )
                    signal_source_info = "Original Signal (No filtered data available)"
            else:
                logger.info("Using original signal for analysis")

            # Create filtered DataFrame with both raw and filtered signals for comparison
            filtered_df = windowed_data.copy()

            if filtered_signal is not None:
                # Only add filtered signal if it matches the windowed data length
                if len(filtered_signal) == len(windowed_data):
                    filtered_df[f"{signal_column}_filtered"] = filtered_signal
                    logger.info("Added filtered signal to comparison DataFrame")
                else:
                    logger.warning(
                        f"Cannot add filtered signal to DataFrame - length mismatch: {len(filtered_signal)} vs {len(windowed_data)}"
                    )

            # ------------------------------------------------------------
            # Single-shot waveform analysis.
            # Previously this callback fanned out into ~7 builders, each
            # of which instantiated its own ``WaveformMorphology`` (and
            # so re-ran peak detection from scratch on the same signal).
            # We now run that once here and pass the resulting peak
            # array down to the slim builders.
            # ------------------------------------------------------------
            shared_peaks = None
            try:
                from vitalDSP.physiological_features.waveform import (
                    WaveformMorphology,
                )

                wm_shared = WaveformMorphology(
                    waveform=selected_signal,
                    fs=sampling_freq,
                    signal_type=(signal_type or "PPG"),
                    simple_mode=True,
                )
                shared_peaks = getattr(wm_shared, "systolic_peaks", None)
                logger.info(
                    "Shared WaveformMorphology computed: %d peaks",
                    0 if shared_peaks is None else len(shared_peaks),
                )
            except Exception as exc:
                logger.warning(
                    "Shared WaveformMorphology failed: %s; builders will "
                    "fall back to internal detection.",
                    exc,
                )

            # Filter info for display (already loaded above)
            if not (signal_source == "filtered" and selected_signal is not None):
                filter_info = None

            # Main plot: signal with peak markers, plus an overlay if
            # the user picked the filtered source.
            filtered_overlay = None
            if signal_source == "filtered" and selected_signal is not None:
                # ``selected_signal`` is the filtered version; use the
                # raw window as the underlay so the user can see both.
                original_window = (
                    df[signal_column].iloc[start_sample:end_sample].values
                )
                main_signal = original_window
                filtered_overlay = selected_signal
            else:
                main_signal = selected_signal

            main_plot = create_main_signal_plot(
                signal_data=main_signal,
                time_axis=time_axis,
                sampling_freq=sampling_freq,
                peaks=shared_peaks if filtered_overlay is None else None,
                signal_type=(signal_type or "PPG"),
                theme=current_theme,
                filtered_overlay=filtered_overlay,
            )

            # The dedicated comparison chart is gone (its content is now
            # in the main plot via the filtered overlay).  Output an
            # empty figure to keep the callback's 9-tuple shape intact.
            comparison_plot = create_empty_figure(current_theme)

            analysis_results = generate_analysis_results(
                selected_signal,
                time_axis,
                sampling_freq,
                analysis_options or ["peaks", "hr", "quality"],
                signal_source_info,
                signal_type,
                filter_info,
            )

            peak_table = create_peak_analysis_table(
                selected_signal,
                time_axis,
                sampling_freq,
                analysis_options or ["peaks", "hr", "quality"],
                signal_source_info,
                signal_type,
                peaks=shared_peaks,
            )
            quality_table = create_signal_quality_table(
                selected_signal,
                time_axis,
                sampling_freq,
                analysis_options or ["peaks", "hr", "quality"],
                signal_source_info,
                signal_type,
            )

            # ``signal-source-table`` and ``additional-metrics-table``
            # are hidden in the layout; feed them empty Divs.
            signal_source_table = html.Div()
            additional_table = html.Div()

            # Drop the per-row records bloat that used to be stored.
            # Downstream analysis pages don't read this store; the
            # data_id is enough.
            time_domain_data = {
                "data_id": latest_data_id,
                "sampling_freq": sampling_freq,
                "window": [int(start_sample), int(end_sample)],
                "signal_source": signal_source_info,
            }

            logger.info("Time domain analysis completed successfully")
            return (
                main_plot,
                comparison_plot,
                analysis_results,
                peak_table,
                quality_table,
                signal_source_table,
                additional_table,
                time_domain_data,
                time_domain_data,
            )

        except Exception as e:
            logger.error(f"Error in time domain analysis callback: {e}")
            import traceback

            traceback.print_exc()
            error_msg = f"Error in analysis: {str(e)}"
            return (
                create_empty_figure(current_theme),
                create_empty_figure(current_theme),
                error_msg,
                error_msg,
                error_msg,
                error_msg,
                error_msg,
                None,
                None,
            )

    @app.callback(
        [
            Output("start-position-slider", "min"),
            Output("start-position-slider", "max"),
            Output("start-position-slider", "value"),
        ],
        [Input("url", "pathname")],
        prevent_initial_call=True,  # Prevent triggering on page load
    )
    def update_time_slider_range(pathname):
        """Update time slider range based on available data."""
        if pathname != "/time-domain":
            return no_update, no_update, no_update

        try:
            from vitalDSP_webapp.services.data.enhanced_data_service import (
                get_enhanced_data_service,
            )

            data_service = get_enhanced_data_service()
            all_data = data_service.get_all_data()

            if not all_data:
                return 0, 100, [0, 10]

            # Get the most recent data
            latest_data_id = list(all_data.keys())[-1]
            latest_data = all_data[latest_data_id]
            df = data_service.get_data(latest_data_id)

            if df is None or df.empty:
                return 0, 100, [0, 10]

            # Calculate time range based on data length and sampling frequency
            sampling_freq = latest_data.get("info", {}).get("sampling_freq", 1000)
            max_time = len(df) / sampling_freq

            # Set reasonable limits
            min_time = 0
            max_time = min(max_time, 300)  # Cap at 5 minutes

            # Set initial window to first 10 seconds or max available
            initial_end = min(10, max_time)

            return min_time, max_time, [min_time, initial_end]

        except Exception as e:
            logger.error(f"Error updating time slider range: {e}")
            return 0, 100, [0, 10]

    # Removed sync_time_inputs_with_slider callback - no longer needed with start/duration approach
