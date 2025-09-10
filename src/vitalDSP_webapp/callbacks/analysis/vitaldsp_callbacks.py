"""
Analysis callbacks for vitalDSP core functionality.

This module handles the main vitalDSP analysis callbacks including time domain,
frequency domain, and signal processing operations.
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

logger = logging.getLogger(__name__)


def create_empty_figure():
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
    return fig


def higuchi_fractal_dimension(signal, k_max=8):
    """Calculate Higuchi fractal dimension."""
    try:
        N = len(signal)
        L = []
        x = []

        for k in range(1, k_max + 1):
            Lk = 0
            for m in range(k):
                Lmk = 0
                for i in range(1, int((N - m) / k)):
                    Lmk += abs(signal[m + i * k] - signal[m + (i - 1) * k])
                Lmk = Lmk * (N - 1) / (k**2 * int((N - m) / k))
                Lk += Lmk
            Lk /= k
            L.append(Lk)
            x.append([np.log(1.0 / k), 1])

        # Linear fit to get slope
        x = np.array(x)
        L = np.log(L)
        slope = np.polyfit(x[:, 0], L, 1)[0]
        return slope
    except Exception as e:
        logger.error(f"Error calculating Higuchi fractal dimension: {e}")
        return 0


def create_time_domain_plot(
    signal_data, time_axis, sampling_freq, peaks=None, filtered_signal=None
):
    """Create the main time domain plot."""
    try:
        fig = go.Figure()

        # Add main signal
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=signal_data,
                mode="lines",
                name="Original Signal",
                line=dict(color="blue", width=1),
            )
        )

        # Add filtered signal if available
        if filtered_signal is not None:
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=filtered_signal,
                    mode="lines",
                    name="Filtered Signal",
                    line=dict(color="red", width=1, dash="dash"),
                )
            )

        # Add peaks if available
        if peaks is not None and len(peaks) > 0:
            fig.add_trace(
                go.Scatter(
                    x=time_axis[peaks],
                    y=signal_data[peaks],
                    mode="markers",
                    name="Detected Peaks",
                    marker=dict(color="red", size=8, symbol="diamond"),
                )
            )

        fig.update_layout(
            title="Time Domain Signal Analysis",
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
            showlegend=True,
            height=400,
        )

        return fig
    except Exception as e:
        logger.error(f"Error creating time domain plot: {e}")
        return create_empty_figure()


def create_peak_analysis_plot(signal_data, time_axis, peaks, sampling_freq):
    """Create peak analysis plot."""
    try:
        if peaks is None or len(peaks) == 0:
            return create_empty_figure()

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Peak Detection", "Peak Intervals"),
            vertical_spacing=0.1,
        )

        # Peak detection plot
        fig.add_trace(
            go.Scatter(x=time_axis, y=signal_data, mode="lines", name="Signal"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=time_axis[peaks],
                y=signal_data[peaks],
                mode="markers",
                name="Peaks",
                marker=dict(color="red", size=8),
            ),
            row=1,
            col=1,
        )

        # Peak intervals plot
        if len(peaks) > 1:
            intervals = np.diff(peaks) / sampling_freq
            interval_times = time_axis[peaks[1:]]
            fig.add_trace(
                go.Scatter(
                    x=interval_times,
                    y=intervals,
                    mode="lines+markers",
                    name="Peak Intervals",
                    line=dict(color="green"),
                ),
                row=2,
                col=1,
            )

        fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
        fig.update_yaxes(title_text="Interval (seconds)", row=2, col=1)

        fig.update_layout(height=400, showlegend=True)
        return fig

    except Exception as e:
        logger.error(f"Error creating peak analysis plot: {e}")
        return create_empty_figure()


def create_main_signal_plot(
    signal_data,
    time_axis,
    sampling_freq,
    analysis_options,
    column_mapping,
    signal_type="PPG",
):
    """Create the main time domain plot with critical points detection."""
    try:
        fig = go.Figure()

        # Use column mapping to get the correct columns
        time_col = column_mapping.get("time")
        signal_col = column_mapping.get("signal")

        if not time_col or not signal_col:
            logger.warning("Missing time or signal column in column mapping")
            return create_empty_figure()

        if time_col not in signal_data.columns or signal_col not in signal_data.columns:
            logger.warning(f"Columns {time_col} or {signal_col} not found in data")
            return create_empty_figure()

        # Get signal data
        signal_values = signal_data[signal_col].values
        time_values = signal_data[time_col].values if time_col != "index" else time_axis

        # Add main signal
        fig.add_trace(
            go.Scatter(
                x=time_values,
                y=signal_values,
                mode="lines",
                name="Original Signal",
                line=dict(color="blue", width=1),
            )
        )

        # Add critical points detection if enabled
        if "critical_points" in analysis_options:
            try:
                # Import vitalDSP waveform morphology
                from vitalDSP.physiological_features.waveform import WaveformMorphology

                # Create waveform morphology object
                wm = WaveformMorphology(
                    waveform=signal_values,
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
                                x=time_values[wm.systolic_peaks],
                                y=signal_values[wm.systolic_peaks],
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
                                    x=time_values[dicrotic_notches],
                                    y=signal_values[dicrotic_notches],
                                    mode="markers",
                                    name="Dicrotic Notches",
                                    marker=dict(
                                        color="orange", size=8, symbol="circle"
                                    ),
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
                                    x=time_values[diastolic_peaks],
                                    y=signal_values[diastolic_peaks],
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
                                x=time_values[wm.r_peaks],
                                y=signal_values[wm.r_peaks],
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
                                    x=time_values[p_peaks],
                                    y=signal_values[p_peaks],
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
                                    x=time_values[t_peaks],
                                    y=signal_values[t_peaks],
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
                                    x=time_values[q_valleys],
                                    y=signal_values[q_valleys],
                                    mode="markers",
                                    name="Q Valleys",
                                    marker=dict(
                                        color="purple", size=8, symbol="triangle-down"
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
                                    x=time_values[s_valleys],
                                    y=signal_values[s_valleys],
                                    mode="markers",
                                    name="S Valleys",
                                    marker=dict(
                                        color="orange", size=8, symbol="triangle-down"
                                    ),
                                    hovertemplate="<b>S Valley:</b> %{y}<extra></extra>",
                                )
                            )
                    except Exception as e:
                        logger.warning(f"S valley detection failed: {e}")

            except Exception as e:
                logger.error(f"Critical points detection failed: {e}")

        # Add basic peak detection if enabled (fallback)
        elif "peaks" in analysis_options:
            try:
                from scipy.signal import find_peaks

                peaks, _ = find_peaks(
                    signal_values,
                    height=0.5 * np.max(signal_values),
                    distance=int(0.5 * sampling_freq),
                )
                if len(peaks) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=time_values[peaks],
                            y=signal_values[peaks],
                            mode="markers",
                            name="Detected Peaks",
                            marker=dict(color="red", size=8, symbol="diamond"),
                            hovertemplate="<b>Peak:</b> %{y}<extra></extra>",
                        )
                    )
            except Exception as e:
                logger.error(f"Peak detection failed: {e}")

        fig.update_layout(
            title=f"Time Domain Signal Analysis - {signal_type}",
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
            showlegend=True,
            height=400,
            template="plotly_white",
            margin=dict(l=40, r=40, t=60, b=40),
            hovermode="closest",
        )

        return fig
    except Exception as e:
        logger.error(f"Error creating time domain plot: {e}")
        return create_empty_figure()


def create_filtered_signal_plot(
    filtered_data,
    time_axis,
    sampling_freq,
    column_mapping,
    signal_type="PPG",
    analysis_options=None,
):
    """Create the filtered signal plot showing both raw and filtered signals with critical points."""
    try:
        fig = go.Figure()

        # Use column mapping to get the correct columns
        time_col = column_mapping.get("time")
        signal_col = column_mapping.get("signal")

        if not time_col or not signal_col:
            logger.warning("Missing time or signal column in column mapping")
            return create_empty_figure()

        if (
            time_col not in filtered_data.columns
            or signal_col not in filtered_data.columns
        ):
            logger.warning(
                f"Columns {time_col} or {signal_col} not found in filtered data"
            )
            return create_empty_figure()

        # Get signal data
        signal_values = filtered_data[signal_col].values
        time_values = (
            filtered_data[time_col].values if time_col != "index" else time_axis
        )

        # Plot raw signal
        fig.add_trace(
            go.Scatter(
                x=time_values,
                y=signal_values,
                mode="lines",
                name="Raw Signal",
                line=dict(color="#2E86AB", width=1.5),
                opacity=0.7,
                hovertemplate="<b>Time:</b> %{x}<br><b>Raw:</b> %{y}<extra></extra>",
            )
        )

        # Plot filtered signal if available
        filtered_col = f"{signal_col}_filtered"
        if filtered_col in filtered_data.columns:
            filtered_values = filtered_data[filtered_col].values
            fig.add_trace(
                go.Scatter(
                    x=time_values,
                    y=filtered_values,
                    mode="lines",
                    name="Filtered Signal",
                    line=dict(color="#E63946", width=2),
                    hovertemplate="<b>Time:</b> %{x}<br><b>Filtered:</b> %{y}<extra></extra>",
                )
            )
            
            # Use the filtered signal for critical points detection
            signal_for_analysis = filtered_values
        else:
            # If no filtered signal available, use raw signal for analysis
            signal_for_analysis = signal_values

        # Add critical points detection if enabled
        if analysis_options and "critical_points" in analysis_options:
            try:
                # Import vitalDSP waveform morphology
                from vitalDSP.physiological_features.waveform import (
                    WaveformMorphology,
                )

                # Create waveform morphology object for the signal to analyze
                wm = WaveformMorphology(
                    waveform=signal_for_analysis,
                    fs=sampling_freq,
                    signal_type=signal_type,
                    simple_mode=True,
                )

                # Detect critical points based on signal type
                if signal_type == "PPG":
                    # For PPG: systolic peaks, dicrotic notches, diastolic peaks
                    if (
                        hasattr(wm, "systolic_peaks")
                        and wm.systolic_peaks is not None
                    ):
                        # Plot systolic peaks on the analyzed signal
                        fig.add_trace(
                            go.Scatter(
                                x=time_values[wm.systolic_peaks],
                                y=signal_for_analysis[wm.systolic_peaks],
                                mode="markers",
                                name="Systolic Peaks",
                                marker=dict(color="red", size=10, symbol="diamond"),
                                hovertemplate="<b>Systolic Peak:</b> %{y}<extra></extra>",
                            )
                        )

                    # Detect and plot dicrotic notches
                    try:
                        dicrotic_notches = wm.detect_dicrotic_notches()
                        if (
                            dicrotic_notches is not None
                            and len(dicrotic_notches) > 0
                        ):
                            fig.add_trace(
                                go.Scatter(
                                    x=time_values[dicrotic_notches],
                                    y=signal_for_analysis[dicrotic_notches],
                                    mode="markers",
                                    name="Dicrotic Notches",
                                    marker=dict(
                                        color="orange", size=8, symbol="circle"
                                    ),
                                    hovertemplate="<b>Dicrotic Notch:</b> %{y}<extra></extra>",
                                )
                            )
                    except Exception as e:
                        logger.warning(
                            f"Dicrotic notch detection failed: {e}"
                        )

                    # Detect and plot diastolic peaks
                    try:
                        diastolic_peaks = wm.detect_diastolic_peak()
                        if diastolic_peaks is not None and len(diastolic_peaks) > 0:
                            fig.add_trace(
                                go.Scatter(
                                    x=time_values[diastolic_peaks],
                                    y=signal_for_analysis[diastolic_peaks],
                                    mode="markers",
                                    name="Diastolic Peaks",
                                    marker=dict(
                                        color="green", size=8, symbol="square"
                                    ),
                                    hovertemplate="<b>Diastolic Peak:</b> %{y}<extra></extra>",
                                )
                            )
                    except Exception as e:
                        logger.warning(
                            f"Diastolic peak detection failed: {e}"
                        )

                elif signal_type == "ECG":
                    # For ECG: R peaks, P peaks, T peaks, Q valleys, S valleys
                    if hasattr(wm, "r_peaks") and wm.r_peaks is not None:
                        # Plot R peaks
                        fig.add_trace(
                            go.Scatter(
                                x=time_values[wm.r_peaks],
                                y=signal_for_analysis[wm.r_peaks],
                                mode="markers",
                                name="R Peaks",
                                marker=dict(color="red", size=10, symbol="diamond"),
                                hovertemplate="<b>R Peak:</b> %{y}<extra></extra>",
                            )
                        )

                    # Detect and plot other ECG features
                    try:
                        p_peaks = wm.detect_p_peak()
                        if p_peaks is not None and len(p_peaks) > 0:
                            fig.add_trace(
                                go.Scatter(
                                    x=time_values[p_peaks],
                                    y=signal_for_analysis[p_peaks],
                                    mode="markers",
                                    name="P Peaks",
                                    marker=dict(
                                        color="blue", size=8, symbol="circle"
                                    ),
                                    hovertemplate="<b>P Peak:</b> %{y}<extra></extra>",
                                )
                            )
                    except Exception as e:
                        logger.warning(f"P peak detection failed: {e}")

                    try:
                        t_peaks = wm.detect_t_peak()
                        if t_peaks is not None and len(t_peaks) > 0:
                            fig.add_trace(
                                go.Scatter(
                                    x=time_values[t_peaks],
                                    y=signal_for_analysis[t_peaks],
                                    mode="markers",
                                    name="T Peaks",
                                    marker=dict(
                                        color="green", size=8, symbol="square"
                                    ),
                                    hovertemplate="<b>T Peak:</b> %{y}<extra></extra>",
                                )
                            )
                    except Exception as e:
                        logger.warning(f"T peak detection failed: {e}")

            except Exception as e:
                logger.error(
                    f"Critical points detection failed: {e}"
                )

        fig.update_layout(
            title=f"Signal Comparison: Raw vs Filtered - {signal_type}",
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
            template="plotly_white",
            height=400,
            margin=dict(l=40, r=40, t=60, b=40),
            showlegend=True,
            hovermode="closest",
        )

        return fig
    except Exception as e:
        logger.error(f"Error creating filtered signal plot: {e}")
        return create_empty_figure()


def generate_analysis_results(
    raw_data, filtered_data, time_axis, sampling_freq, analysis_options, column_mapping
):
    """Generate analysis results and insights."""
    if not analysis_options:
        return "No analysis options selected"

    results = []

    try:
        # Find signal column
        signal_col = column_mapping.get("signal")
        if not signal_col or signal_col not in raw_data.columns:
            return "Signal column not found in data"

        raw_signal = raw_data[signal_col].values
        filtered_signal = filtered_data if filtered_data is not None else raw_signal

        # Basic statistics
        results.append(html.H6("📊 Signal Statistics", className="mb-2"))
        results.append(
            dbc.Table(
                [
                    html.Thead(
                        [
                            html.Tr(
                                [
                                    html.Th("Metric", className="text-center"),
                                    html.Th("Value", className="text-center"),
                                    html.Th("Unit", className="text-center"),
                                ]
                            )
                        ]
                    ),
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td("Mean", className="fw-bold"),
                                    html.Td(
                                        f"{np.mean(raw_signal):.2f}",
                                        className="text-end",
                                    ),
                                    html.Td("Signal Units", className="text-muted"),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Standard Deviation", className="fw-bold"),
                                    html.Td(
                                        f"{np.std(raw_signal):.2f}",
                                        className="text-end",
                                    ),
                                    html.Td("Signal Units", className="text-muted"),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Minimum", className="fw-bold"),
                                    html.Td(
                                        f"{np.min(raw_signal):.2f}",
                                        className="text-end",
                                    ),
                                    html.Td("Signal Units", className="text-muted"),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Maximum", className="fw-bold"),
                                    html.Td(
                                        f"{np.max(raw_signal):.2f}",
                                        className="text-end",
                                    ),
                                    html.Td("Signal Units", className="text-muted"),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("RMS", className="fw-bold"),
                                    html.Td(
                                        f"{np.sqrt(np.mean(raw_signal**2)):.2f}",
                                        className="text-end",
                                    ),
                                    html.Td("Signal Units", className="text-muted"),
                                ]
                            ),
                        ]
                    ),
                ],
                bordered=True,
                hover=True,
                responsive=True,
                className="mb-3",
            )
        )

        # Peak detection and heart rate analysis
        if "peaks" in analysis_options:
            try:
                from scipy.signal import find_peaks

                # Use adaptive threshold for peak detection
                mean_val = np.mean(raw_signal)
                std_val = np.std(raw_signal)
                threshold = mean_val + 2 * std_val

                # Find peaks with minimum distance constraint
                min_distance = int(
                    sampling_freq * 0.1
                )  # Minimum 0.1 seconds between peaks
                peaks, properties = find_peaks(
                    raw_signal,
                    height=threshold,
                    distance=min_distance,
                    prominence=0.1 * std_val,
                )

                if len(peaks) > 1:
                    # Calculate RR intervals
                    rr_intervals = np.diff(time_axis[peaks])
                    heart_rate = 60 / np.mean(rr_intervals)

                    results.append(html.Hr())
                    results.append(html.H6("❤️ Heart Rate Analysis", className="mb-2"))
                    results.append(
                        dbc.Table(
                            [
                                html.Thead(
                                    [
                                        html.Tr(
                                            [
                                                html.Th(
                                                    "Metric", className="text-center"
                                                ),
                                                html.Th(
                                                    "Value", className="text-center"
                                                ),
                                                html.Th(
                                                    "Unit", className="text-center"
                                                ),
                                            ]
                                        )
                                    ]
                                ),
                                html.Tbody(
                                    [
                                        html.Tr(
                                            [
                                                html.Td(
                                                    "Detected Peaks",
                                                    className="fw-bold",
                                                ),
                                                html.Td(
                                                    f"{len(peaks)}",
                                                    className="text-end",
                                                ),
                                                html.Td(
                                                    "Count", className="text-muted"
                                                ),
                                            ]
                                        ),
                                        html.Tr(
                                            [
                                                html.Td(
                                                    "Average RR Interval",
                                                    className="fw-bold",
                                                ),
                                                html.Td(
                                                    f"{np.mean(rr_intervals):.3f}",
                                                    className="text-end",
                                                ),
                                                html.Td(
                                                    "Seconds", className="text-muted"
                                                ),
                                            ]
                                        ),
                                        html.Tr(
                                            [
                                                html.Td(
                                                    "Heart Rate", className="fw-bold"
                                                ),
                                                html.Td(
                                                    f"{heart_rate:.1f}",
                                                    className="text-end",
                                                ),
                                                html.Td("BPM", className="text-muted"),
                                            ]
                                        ),
                                        html.Tr(
                                            [
                                                html.Td(
                                                    "RR Standard Deviation",
                                                    className="fw-bold",
                                                ),
                                                html.Td(
                                                    f"{np.std(rr_intervals):.3f}",
                                                    className="text-end",
                                                ),
                                                html.Td(
                                                    "Seconds", className="text-muted"
                                                ),
                                            ]
                                        ),
                                        html.Tr(
                                            [
                                                html.Td("pNN50", className="fw-bold"),
                                                html.Td(
                                                    f"{np.sum(np.abs(np.diff(rr_intervals)) > 0.05) / len(rr_intervals) * 100:.1f}",
                                                    className="text-end",
                                                ),
                                                html.Td("%", className="text-muted"),
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                            bordered=True,
                            hover=True,
                            responsive=True,
                            className="mb-3",
                        )
                    )
                else:
                    results.append(html.Hr())
                    results.append(html.H6("❤️ Heart Rate Analysis", className="mb-2"))
                    results.append(
                        html.P(
                            "Insufficient peaks detected for heart rate analysis",
                            className="text-muted",
                        )
                    )

            except Exception as e:
                logger.error(f"Peak detection failed: {e}")
                results.append(html.Hr())
                results.append(html.H6("❤️ Heart Rate Analysis", className="mb-2"))
                results.append(
                    html.P(
                        f"Error in peak detection: {str(e)}", className="text-danger"
                    )
                )

        # Signal quality assessment
        if "quality" in analysis_options:
            try:
                # Calculate SNR
                signal_power = np.mean(raw_signal**2)
                noise_power = np.var(raw_signal)
                snr_db = (
                    10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
                )

                # Artifact detection using IQR method
                q75, q25 = np.percentile(raw_signal, [75, 25])
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                artifact_count = np.sum(
                    (raw_signal < lower_bound) | (raw_signal > upper_bound)
                )
                artifact_percentage = (artifact_count / len(raw_signal)) * 100

                # Signal stability (coefficient of variation)
                stability_score = (
                    (np.std(raw_signal) / np.mean(raw_signal)) * 100
                    if np.mean(raw_signal) != 0
                    else 0
                )

                results.append(html.Hr())
                results.append(
                    html.H6("🎯 Signal Quality Assessment", className="mb-2")
                )
                results.append(
                    dbc.Table(
                        [
                            html.Thead(
                                [
                                    html.Tr(
                                        [
                                            html.Th("Metric", className="text-center"),
                                            html.Th("Value", className="text-center"),
                                            html.Th("Quality", className="text-center"),
                                        ]
                                    )
                                ]
                            ),
                            html.Tbody(
                                [
                                    html.Tr(
                                        [
                                            html.Td(
                                                "Signal-to-Noise Ratio",
                                                className="fw-bold",
                                            ),
                                            html.Td(
                                                f"{snr_db:.1f} dB", className="text-end"
                                            ),
                                            html.Td(
                                                html.Span(
                                                    (
                                                        "Excellent"
                                                        if snr_db > 20
                                                        else (
                                                            "Good"
                                                            if snr_db > 15
                                                            else (
                                                                "Fair"
                                                                if snr_db > 10
                                                                else "Poor"
                                                            )
                                                        )
                                                    ),
                                                    className=f"badge {'bg-success' if snr_db > 20 else 'bg-info' if snr_db > 15 else 'bg-warning' if snr_db > 10 else 'bg-danger'}",
                                                ),
                                                className="text-center",
                                            ),
                                        ]
                                    ),
                                    html.Tr(
                                        [
                                            html.Td(
                                                "Detected Artifacts",
                                                className="fw-bold",
                                            ),
                                            html.Td(
                                                f"{artifact_count} ({artifact_percentage:.1f}%)",
                                                className="text-end",
                                            ),
                                            html.Td(
                                                html.Span(
                                                    (
                                                        "Low"
                                                        if artifact_percentage < 5
                                                        else (
                                                            "Medium"
                                                            if artifact_percentage < 15
                                                            else "High"
                                                        )
                                                    ),
                                                    className=f"badge {'bg-success' if artifact_percentage < 5 else 'bg-warning' if artifact_percentage < 15 else 'bg-danger'}",
                                                ),
                                                className="text-center",
                                            ),
                                        ]
                                    ),
                                    html.Tr(
                                        [
                                            html.Td(
                                                "Signal Stability", className="fw-bold"
                                            ),
                                            html.Td(
                                                f"{stability_score:.1f}%",
                                                className="text-end",
                                            ),
                                            html.Td(
                                                html.Span(
                                                    (
                                                        "Excellent"
                                                        if stability_score < 10
                                                        else (
                                                            "Good"
                                                            if stability_score < 20
                                                            else (
                                                                "Fair"
                                                                if stability_score < 30
                                                                else "Poor"
                                                            )
                                                        )
                                                    ),
                                                    className=f"badge {'bg-success' if stability_score < 10 else 'bg-info' if stability_score < 20 else 'bg-warning' if stability_score < 30 else 'bg-danger'}",
                                                ),
                                                className="text-center",
                                            ),
                                        ]
                                    ),
                                    html.Tr(
                                        [
                                            html.Td(
                                                "Signal Range", className="fw-bold"
                                            ),
                                            html.Td(
                                                f"{np.max(raw_signal) - np.min(raw_signal):.2f}",
                                                className="text-end",
                                            ),
                                            html.Td(
                                                "Signal Units", className="text-muted"
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        bordered=True,
                        hover=True,
                        responsive=True,
                        className="mb-3",
                    )
                )

            except Exception as e:
                logger.error(f"Signal quality assessment failed: {e}")
                results.append(html.Hr())
                results.append(
                    html.H6("🎯 Signal Quality Assessment", className="mb-2")
                )
                results.append(
                    html.P(
                        f"Error in quality assessment: {str(e)}",
                        className="text-danger",
                    )
                )

        # Filtering results comparison
        if filtered_data is not None and not np.array_equal(
            raw_signal, filtered_signal
        ):
            try:
                results.append(html.Hr())
                results.append(html.H6("🔧 Filtering Results", className="mb-2"))

                # Calculate power reduction
                raw_power = np.mean(raw_signal**2)
                filtered_power = np.mean(filtered_signal**2)
                power_reduction = (
                    (raw_power - filtered_power) / raw_power * 100
                    if raw_power > 0
                    else 0
                )

                # Calculate frequency-specific improvements
                raw_fft = np.abs(np.fft.rfft(raw_signal))
                filtered_fft = np.abs(np.fft.rfft(filtered_signal))
                freqs = np.fft.rfftfreq(len(raw_signal), 1 / sampling_freq)

                # Low frequency improvement (0-5 Hz)
                low_freq_mask = freqs <= 5
                low_freq_improvement = np.mean(raw_fft[low_freq_mask]) - np.mean(
                    filtered_fft[low_freq_mask]
                )

                # High frequency improvement (40+ Hz)
                high_freq_mask = freqs >= 40
                high_freq_improvement = np.mean(raw_fft[high_freq_mask]) - np.mean(
                    filtered_fft[high_freq_mask]
                )

                results.append(
                    dbc.Table(
                        [
                            html.Thead(
                                [
                                    html.Tr(
                                        [
                                            html.Th("Metric", className="text-center"),
                                            html.Th("Value", className="text-center"),
                                            html.Th(
                                                "Description", className="text-center"
                                            ),
                                        ]
                                    )
                                ]
                            ),
                            html.Tbody(
                                [
                                    html.Tr(
                                        [
                                            html.Td(
                                                "Power Reduction", className="fw-bold"
                                            ),
                                            html.Td(
                                                f"{power_reduction:.1f}%",
                                                className="text-end",
                                            ),
                                            html.Td(
                                                "Overall signal power change",
                                                className="text-muted",
                                            ),
                                        ]
                                    ),
                                    html.Tr(
                                        [
                                            html.Td(
                                                "Low Freq Improvement",
                                                className="fw-bold",
                                            ),
                                            html.Td(
                                                f"{low_freq_improvement:.3f}",
                                                className="text-end",
                                            ),
                                            html.Td(
                                                "Low frequency noise reduction",
                                                className="text-muted",
                                            ),
                                        ]
                                    ),
                                    html.Tr(
                                        [
                                            html.Td(
                                                "High Freq Improvement",
                                                className="fw-bold",
                                            ),
                                            html.Td(
                                                f"{high_freq_improvement:.3f}",
                                                className="text-end",
                                            ),
                                            html.Td(
                                                "High frequency noise reduction",
                                                className="text-muted",
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        bordered=True,
                        hover=True,
                        responsive=True,
                        className="mb-3",
                    )
                )

            except Exception as e:
                logger.error(f"Filtering results analysis failed: {e}")
                results.append(html.Hr())
                results.append(html.H6("🔧 Filtering Results", className="mb-2"))
                results.append(
                    html.P(
                        f"Error in filtering analysis: {str(e)}",
                        className="text-danger",
                    )
                )

        return html.Div(results)

    except Exception as e:
        logger.error(f"Error generating analysis results: {e}")
        return f"Error in analysis: {str(e)}"


def create_peak_analysis_table(
    raw_data, filtered_data, time_axis, sampling_freq, analysis_options, column_mapping
):
    """Create comprehensive peak analysis table."""
    try:
        signal_col = column_mapping.get("signal")
        if not signal_col or signal_col not in raw_data.columns:
            return "Signal column not found in data"

        raw_signal = raw_data[signal_col].values

        if "peaks" not in analysis_options:
            return html.Div(
                [
                    html.H6("🔍 Peak Analysis", className="text-muted"),
                    html.P(
                        "Peak detection not enabled in analysis options",
                        className="text-muted",
                    ),
                ]
            )

        # Detect peaks with advanced parameters
        from scipy.signal import find_peaks

        mean_val = np.mean(raw_signal)
        std_val = np.std(raw_signal)
        threshold = mean_val + 2 * std_val
        min_distance = int(sampling_freq * 0.1)  # Minimum 0.1 seconds between peaks

        peaks, properties = find_peaks(
            raw_signal,
            height=threshold,
            distance=min_distance,
            prominence=0.1 * std_val,
            width=0.01 * sampling_freq,
        )

        if len(peaks) < 2:
            return html.Div(
                [
                    html.H6("🔍 Peak Analysis", className="text-muted"),
                    html.P(
                        "Insufficient peaks detected for analysis",
                        className="text-muted",
                    ),
                ]
            )

        # Calculate RR intervals and heart rate variability
        rr_intervals = np.diff(time_axis[peaks])
        heart_rates = 60 / rr_intervals

        # HRV metrics
        mean_rr = np.mean(rr_intervals)
        sdnn = np.std(rr_intervals)
        rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))

        # pNN50 calculation
        nn_intervals = rr_intervals
        nn_diff = np.abs(np.diff(nn_intervals))
        pnn50 = np.sum(nn_diff > 0.05) / len(nn_diff) * 100

        # Peak properties
        peak_amplitudes = raw_signal[peaks]
        peak_prominences = properties.get("prominences", np.zeros_like(peaks))
        peak_widths = properties.get("widths", np.zeros_like(peaks)) / sampling_freq

        return html.Div(
            [
                html.H6("🔍 Peak Analysis Summary", className="text-primary mb-3"),
                # Summary metrics
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4(
                                                    f"{len(peaks)}",
                                                    className="text-center text-primary mb-0",
                                                ),
                                                html.Small(
                                                    "Total Peaks",
                                                    className="text-center d-block text-muted",
                                                ),
                                            ]
                                        )
                                    ],
                                    className="text-center border-primary",
                                )
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4(
                                                    f"{np.mean(heart_rates):.1f}",
                                                    className="text-center text-success mb-0",
                                                ),
                                                html.Small(
                                                    "Mean HR (BPM)",
                                                    className="text-center d-block text-muted",
                                                ),
                                            ]
                                        )
                                    ],
                                    className="text-center border-success",
                                )
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4(
                                                    f"{sdnn:.2f}",
                                                    className="text-center text-warning mb-0",
                                                ),
                                                html.Small(
                                                    "SDNN (ms)",
                                                    className="text-center d-block text-muted",
                                                ),
                                            ]
                                        )
                                    ],
                                    className="text-center border-warning",
                                )
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4(
                                                    f"{pnn50:.1f}%",
                                                    className="text-center text-info mb-0",
                                                ),
                                                html.Small(
                                                    "pNN50",
                                                    className="text-center d-block text-muted",
                                                ),
                                            ]
                                        )
                                    ],
                                    className="text-center border-info",
                                )
                            ],
                            md=3,
                        ),
                    ],
                    className="mb-3",
                ),
                # HRV metrics table
                html.H6("Heart Rate Variability Metrics", className="mb-2"),
                dbc.Table(
                    [
                        html.Thead(
                            [
                                html.Tr(
                                    [
                                        html.Th("Metric", className="text-center"),
                                        html.Th("Value", className="text-center"),
                                        html.Th("Unit", className="text-center"),
                                        html.Th("Description", className="text-center"),
                                    ]
                                )
                            ]
                        ),
                        html.Tbody(
                            [
                                html.Tr(
                                    [
                                        html.Td(
                                            "Mean RR Interval", className="fw-bold"
                                        ),
                                        html.Td(f"{mean_rr:.3f}", className="text-end"),
                                        html.Td("Seconds", className="text-muted"),
                                        html.Td(
                                            "Average RR interval",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td("SDNN", className="fw-bold"),
                                        html.Td(f"{sdnn:.3f}", className="text-end"),
                                        html.Td("Seconds", className="text-muted"),
                                        html.Td(
                                            "Standard deviation of RR intervals",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td("RMSSD", className="fw-bold"),
                                        html.Td(f"{rmssd:.3f}", className="text-end"),
                                        html.Td("Seconds", className="text-muted"),
                                        html.Td(
                                            "Root mean square of RR differences",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td("pNN50", className="fw-bold"),
                                        html.Td(f"{pnn50:.1f}", className="text-end"),
                                        html.Td("%", className="text-muted"),
                                        html.Td(
                                            "Percentage of RR differences > 50ms",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(
                                            "Heart Rate Range", className="fw-bold"
                                        ),
                                        html.Td(
                                            f"{np.max(heart_rates):.1f} - {np.min(heart_rates):.1f}",
                                            className="text-end",
                                        ),
                                        html.Td("BPM", className="text-muted"),
                                        html.Td(
                                            "Min-Max heart rate", className="text-muted"
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ],
                    bordered=True,
                    hover=True,
                    responsive=True,
                    className="mb-3",
                ),
                # Peak properties table
                html.H6("Peak Properties", className="mb-2"),
                dbc.Table(
                    [
                        html.Thead(
                            [
                                html.Tr(
                                    [
                                        html.Th("Property", className="text-center"),
                                        html.Th("Mean", className="text-center"),
                                        html.Th("Std Dev", className="text-center"),
                                        html.Th("Min", className="text-center"),
                                        html.Th("Max", className="text-center"),
                                    ]
                                )
                            ]
                        ),
                        html.Tbody(
                            [
                                html.Tr(
                                    [
                                        html.Td("Peak Amplitude", className="fw-bold"),
                                        html.Td(
                                            f"{np.mean(peak_amplitudes):.3f}",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            f"{np.std(peak_amplitudes):.3f}",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            f"{np.min(peak_amplitudes):.3f}",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            f"{np.max(peak_amplitudes):.3f}",
                                            className="text-end",
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td("Peak Prominence", className="fw-bold"),
                                        html.Td(
                                            f"{np.mean(peak_prominences):.3f}",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            f"{np.std(peak_prominences):.3f}",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            f"{np.min(peak_prominences):.3f}",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            f"{np.max(peak_prominences):.3f}",
                                            className="text-end",
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td("Peak Width", className="fw-bold"),
                                        html.Td(
                                            f"{np.mean(peak_widths):.3f}",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            f"{np.std(peak_widths):.3f}",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            f"{np.min(peak_widths):.3f}",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            f"{np.max(peak_widths):.3f}",
                                            className="text-end",
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ],
                    bordered=True,
                    hover=True,
                    responsive=True,
                    size="sm",
                ),
            ]
        )

    except Exception as e:
        logger.error(f"Error creating peak analysis table: {e}")
        return f"Error in peak analysis: {str(e)}"


def create_signal_quality_table(
    raw_data, filtered_data, time_axis, sampling_freq, analysis_options, column_mapping
):
    """Create comprehensive signal quality assessment table."""
    try:
        signal_col = column_mapping.get("signal")
        if not signal_col or signal_col not in raw_data.columns:
            return "Signal column not found in data"

        raw_signal = raw_data[signal_col].values

        if "quality" not in analysis_options:
            return html.Div(
                [
                    html.H6("🎯 Signal Quality Assessment", className="text-muted"),
                    html.P(
                        "Signal quality assessment not enabled in analysis options",
                        className="text-muted",
                    ),
                ]
            )

        # Calculate comprehensive quality metrics
        # SNR calculation
        signal_power = np.mean(raw_signal**2)
        noise_power = np.var(raw_signal)
        snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0

        # Artifact detection using IQR method
        q75, q25 = np.percentile(raw_signal, [75, 25])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        artifact_count = np.sum((raw_signal < lower_bound) | (raw_signal > upper_bound))
        artifact_percentage = (artifact_count / len(raw_signal)) * 100

        # Baseline wander detection (low frequency drift)
        from scipy.signal import savgol_filter

        try:
            # Use Savitzky-Golay filter to estimate baseline
            window_length = min(51, len(raw_signal) // 10 * 2 + 1)  # Ensure odd number
            if window_length >= 3:
                baseline = savgol_filter(raw_signal, window_length, 1)
                baseline_wander = np.std(raw_signal - baseline)
                baseline_wander_percentage = (
                    baseline_wander / np.std(raw_signal)
                ) * 100
            else:
                baseline_wander = 0
                baseline_wander_percentage = 0
        except Exception:
            baseline_wander = 0
            baseline_wander_percentage = 0

        # Motion artifact detection (sudden amplitude changes)
        signal_diff = np.abs(np.diff(raw_signal))
        motion_threshold = np.mean(signal_diff) + 2 * np.std(signal_diff)
        motion_artifacts = np.sum(signal_diff > motion_threshold)
        motion_artifact_percentage = (motion_artifacts / len(signal_diff)) * 100

        # Signal stability (coefficient of variation)
        stability_score = (
            (np.std(raw_signal) / np.mean(raw_signal)) * 100
            if np.mean(raw_signal) != 0
            else 0
        )

        # Frequency content analysis
        fft_vals = np.abs(np.fft.rfft(raw_signal))
        freqs = np.fft.rfftfreq(len(raw_signal), 1 / sampling_freq)

        # Power in different frequency bands
        dc_power = np.sum(fft_vals[freqs <= 0.5])  # DC to 0.5 Hz
        low_freq_power = np.sum(fft_vals[(freqs > 0.5) & (freqs <= 5)])  # 0.5-5 Hz
        mid_freq_power = np.sum(fft_vals[(freqs > 5) & (freqs <= 40)])  # 5-40 Hz
        high_freq_power = np.sum(fft_vals[freqs > 40])  # >40 Hz

        total_power = np.sum(fft_vals)

        # Handle negative power values by using absolute values for percentage calculation
        # This ensures percentage is always meaningful regardless of power sign
        if total_power != 0:
            dc_percentage = (np.abs(dc_power) / np.abs(total_power)) * 100
            low_freq_percentage = (np.abs(low_freq_power) / np.abs(total_power)) * 100
            mid_freq_percentage = (np.abs(mid_freq_power) / np.abs(total_power)) * 100
            high_freq_percentage = (np.abs(high_freq_power) / np.abs(total_power)) * 100
        else:
            dc_percentage = 0
            low_freq_percentage = 0
            mid_freq_percentage = 0
            high_freq_percentage = 0

        # Peak detection quality
        from scipy.signal import find_peaks

        try:
            peaks, properties = find_peaks(
                raw_signal, prominence=0.1 * np.std(raw_signal)
            )
            # peak_quality = len(peaks) / (
            #     len(raw_signal) / sampling_freq
            # )  # peaks per second
        except Exception:
            pass

        # Signal continuity (gaps detection)
        signal_diff_norm = np.abs(np.diff(raw_signal)) / (
            np.max(raw_signal) - np.min(raw_signal)
        )
        continuity_score = np.sum(signal_diff_norm < 0.1) / len(signal_diff_norm) * 100

        # Outlier detection
        z_scores = np.abs((raw_signal - np.mean(raw_signal)) / np.std(raw_signal))
        outliers = np.sum(z_scores > 3)
        outlier_percentage = (outliers / len(raw_signal)) * 100

        return html.Div(
            [
                html.H6("🎯 Signal Quality Assessment", className="text-primary mb-3"),
                # Quality summary cards
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4(
                                                    f"{snr_db:.1f} dB",
                                                    className="text-center text-primary mb-0",
                                                ),
                                                html.Small(
                                                    "SNR",
                                                    className="text-center d-block text-muted",
                                                ),
                                            ]
                                        )
                                    ],
                                    className="text-center border-primary",
                                )
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4(
                                                    f"{artifact_percentage:.1f}%",
                                                    className="text-center text-success mb-0",
                                                ),
                                                html.Small(
                                                    "Artifacts",
                                                    className="text-center d-block text-muted",
                                                ),
                                            ]
                                        )
                                    ],
                                    className="text-center border-success",
                                )
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4(
                                                    f"{stability_score:.1f}%",
                                                    className="text-center text-warning mb-0",
                                                ),
                                                html.Small(
                                                    "Stability",
                                                    className="text-center d-block text-muted",
                                                ),
                                            ]
                                        )
                                    ],
                                    className="text-center border-warning",
                                )
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4(
                                                    f"{continuity_score:.1f}%",
                                                    className="text-center text-info mb-0",
                                                ),
                                                html.Small(
                                                    "Continuity",
                                                    className="text-center d-block text-muted",
                                                ),
                                            ]
                                        )
                                    ],
                                    className="text-center border-info",
                                )
                            ],
                            md=3,
                        ),
                    ],
                    className="mb-3",
                ),
                # Main quality metrics table
                html.H6("Quality Metrics", className="mb-2"),
                dbc.Table(
                    [
                        html.Thead(
                            [
                                html.Tr(
                                    [
                                        html.Th("Metric", className="text-center"),
                                        html.Th("Value", className="text-center"),
                                        html.Th("Quality", className="text-center"),
                                        html.Th("Description", className="text-center"),
                                    ]
                                )
                            ]
                        ),
                        html.Tbody(
                            [
                                html.Tr(
                                    [
                                        html.Td(
                                            "Signal-to-Noise Ratio", className="fw-bold"
                                        ),
                                        html.Td(
                                            f"{snr_db:.1f} dB", className="text-end"
                                        ),
                                        html.Td(
                                            html.Span(
                                                (
                                                    "Excellent"
                                                    if snr_db > 20
                                                    else (
                                                        "Good"
                                                        if snr_db > 15
                                                        else (
                                                            "Fair"
                                                            if snr_db > 10
                                                            else "Poor"
                                                        )
                                                    )
                                                ),
                                                className=f"badge {'bg-success' if snr_db > 20 else 'bg-info' if snr_db > 15 else 'bg-warning' if snr_db > 10 else 'bg-danger'}",
                                            ),
                                            className="text-center",
                                        ),
                                        html.Td(
                                            "Signal quality indicator",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(
                                            "Detected Artifacts", className="fw-bold"
                                        ),
                                        html.Td(
                                            f"{artifact_count} ({artifact_percentage:.1f}%)",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            html.Span(
                                                (
                                                    "Low"
                                                    if artifact_percentage < 5
                                                    else (
                                                        "Medium"
                                                        if artifact_percentage < 15
                                                        else "High"
                                                    )
                                                ),
                                                className=f"badge {'bg-success' if artifact_percentage < 5 else 'bg-warning' if artifact_percentage < 15 else 'bg-danger'}",
                                            ),
                                            className="text-center",
                                        ),
                                        html.Td(
                                            "Statistical outliers",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td("Baseline Wander", className="fw-bold"),
                                        html.Td(
                                            f"{baseline_wander_percentage:.1f}%",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            html.Span(
                                                (
                                                    "Low"
                                                    if baseline_wander_percentage < 10
                                                    else (
                                                        "Medium"
                                                        if baseline_wander_percentage
                                                        < 25
                                                        else "High"
                                                    )
                                                ),
                                                className=f"badge {'bg-success' if baseline_wander_percentage < 10 else 'bg-warning' if baseline_wander_percentage < 25 else 'bg-danger'}",
                                            ),
                                            className="text-center",
                                        ),
                                        html.Td(
                                            "Low frequency drift",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(
                                            "Motion Artifacts", className="fw-bold"
                                        ),
                                        html.Td(
                                            f"{motion_artifacts} ({motion_artifact_percentage:.1f}%)",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            html.Span(
                                                (
                                                    "Low"
                                                    if motion_artifact_percentage < 5
                                                    else (
                                                        "Medium"
                                                        if motion_artifact_percentage
                                                        < 15
                                                        else "High"
                                                    )
                                                ),
                                                className=f"badge {'bg-success' if motion_artifact_percentage < 5 else 'bg-warning' if motion_artifact_percentage < 15 else 'bg-danger'}",
                                            ),
                                            className="text-center",
                                        ),
                                        html.Td(
                                            "Sudden amplitude changes",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(
                                            "Signal Stability", className="fw-bold"
                                        ),
                                        html.Td(
                                            f"{stability_score:.1f}%",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            html.Span(
                                                (
                                                    "Excellent"
                                                    if stability_score < 10
                                                    else (
                                                        "Good"
                                                        if stability_score < 20
                                                        else (
                                                            "Fair"
                                                            if stability_score < 30
                                                            else "Poor"
                                                        )
                                                    )
                                                ),
                                                className=f"badge {'bg-success' if stability_score < 10 else 'bg-info' if stability_score < 20 else 'bg-warning' if stability_score < 30 else 'bg-danger'}",
                                            ),
                                            className="text-center",
                                        ),
                                        html.Td(
                                            "Coefficient of variation",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(
                                            "Signal Continuity", className="fw-bold"
                                        ),
                                        html.Td(
                                            f"{continuity_score:.1f}%",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            html.Span(
                                                (
                                                    "Excellent"
                                                    if continuity_score > 90
                                                    else (
                                                        "Good"
                                                        if continuity_score > 80
                                                        else (
                                                            "Fair"
                                                            if continuity_score > 70
                                                            else "Poor"
                                                        )
                                                    )
                                                ),
                                                className=f"badge {'bg-success' if continuity_score > 90 else 'bg-info' if continuity_score > 80 else 'bg-warning' if continuity_score > 70 else 'bg-danger'}",
                                            ),
                                            className="text-center",
                                        ),
                                        html.Td(
                                            "Smooth signal transitions",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(
                                            "Outlier Percentage", className="fw-bold"
                                        ),
                                        html.Td(
                                            f"{outlier_percentage:.1f}%",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            html.Span(
                                                (
                                                    "Low"
                                                    if outlier_percentage < 1
                                                    else (
                                                        "Medium"
                                                        if outlier_percentage < 5
                                                        else "High"
                                                    )
                                                ),
                                                className=f"badge {'bg-success' if outlier_percentage < 1 else 'bg-warning' if outlier_percentage < 5 else 'bg-danger'}",
                                            ),
                                            className="text-center",
                                        ),
                                        html.Td(
                                            "Z-score > 3 outliers",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ],
                    bordered=True,
                    hover=True,
                    responsive=True,
                    className="mb-3",
                ),
                # Frequency content analysis
                html.H6("Frequency Content Analysis", className="mb-2"),
                dbc.Table(
                    [
                        html.Thead(
                            [
                                html.Tr(
                                    [
                                        html.Th(
                                            "Frequency Band", className="text-center"
                                        ),
                                        html.Th("Power", className="text-center"),
                                        html.Th("Percentage", className="text-center"),
                                        html.Th("Description", className="text-center"),
                                    ]
                                )
                            ]
                        ),
                        html.Tbody(
                            [
                                html.Tr(
                                    [
                                        html.Td("DC (0-0.5 Hz)", className="fw-bold"),
                                        html.Td(
                                            f"{dc_power:.2e}", className="text-end"
                                        ),
                                        html.Td(
                                            f"{dc_percentage:.1f}%",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            "Baseline and drift", className="text-muted"
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td("Low (0.5-5 Hz)", className="fw-bold"),
                                        html.Td(
                                            f"{low_freq_power:.2e}",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            f"{low_freq_percentage:.1f}%",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            "Respiratory and slow variations",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td("Mid (5-40 Hz)", className="fw-bold"),
                                        html.Td(
                                            f"{mid_freq_power:.2e}",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            f"{mid_freq_percentage:.1f}%",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            "Cardiac and physiological",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td("High (>40 Hz)", className="fw-bold"),
                                        html.Td(
                                            f"{high_freq_power:.2e}",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            f"{high_freq_percentage:.1f}%",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            "Noise and artifacts",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ],
                    bordered=True,
                    hover=True,
                    responsive=True,
                    size="sm",
                ),
            ]
        )

    except Exception as e:
        logger.error(f"Error creating signal quality table: {e}")
        return f"Error in signal quality assessment: {str(e)}"


def create_filtering_results_table(
    raw_data, filtered_data, time_axis, sampling_freq, analysis_options, column_mapping
):
    """Create comprehensive filtering results table."""
    try:
        signal_col = column_mapping.get("signal")
        if not signal_col or signal_col not in raw_data.columns:
            return "Signal column not found in data"

        raw_signal = raw_data[signal_col].values

        # Check if we have filtered data to compare
        if filtered_data is None or np.array_equal(raw_signal, filtered_data):
            return html.Div(
                [
                    html.H6("🔧 Filtering Results", className="text-muted"),
                    html.P(
                        "No filtering applied or filtered data identical to raw data",
                        className="text-muted",
                    ),
                ]
            )

        # Calculate comprehensive filtering metrics
        # Power analysis
        raw_power = np.mean(raw_signal**2)
        filtered_power = np.mean(filtered_data**2)
        power_reduction = (
            (raw_power - filtered_power) / raw_power * 100 if raw_power > 0 else 0
        )

        # RMS analysis
        raw_rms = np.sqrt(np.mean(raw_signal**2))
        filtered_rms = np.sqrt(np.mean(filtered_data**2))
        rms_reduction = (raw_rms - filtered_rms) / raw_rms * 100 if raw_rms > 0 else 0

        # Frequency domain analysis
        raw_fft = np.abs(np.fft.rfft(raw_signal))
        filtered_fft = np.abs(np.fft.rfft(filtered_data))
        freqs = np.fft.rfftfreq(len(raw_signal), 1 / sampling_freq)

        # Power in different frequency bands
        # DC and very low frequency (0-0.5 Hz)
        dc_mask = freqs <= 0.5
        dc_reduction = np.mean(raw_fft[dc_mask]) - np.mean(filtered_fft[dc_mask])
        dc_reduction_percent = (
            (dc_reduction / np.mean(raw_fft[dc_mask])) * 100
            if np.mean(raw_fft[dc_mask]) > 0
            else 0
        )

        # Low frequency (0.5-5 Hz) - respiratory and slow variations
        low_freq_mask = (freqs > 0.5) & (freqs <= 5)
        low_freq_reduction = np.mean(raw_fft[low_freq_mask]) - np.mean(
            filtered_fft[low_freq_mask]
        )
        low_freq_reduction_percent = (
            (low_freq_reduction / np.mean(raw_fft[low_freq_mask])) * 100
            if np.mean(raw_fft[low_freq_mask]) > 0
            else 0
        )

        # Mid frequency (5-40 Hz) - cardiac and physiological
        mid_freq_mask = (freqs > 5) & (freqs <= 40)
        mid_freq_reduction = np.mean(raw_fft[mid_freq_mask]) - np.mean(
            filtered_fft[mid_freq_mask]
        )
        mid_freq_reduction_percent = (
            (mid_freq_reduction / np.mean(raw_fft[mid_freq_mask])) * 100
            if np.mean(raw_fft[mid_freq_mask]) > 0
            else 0
        )

        # High frequency (>40 Hz) - noise and artifacts
        high_freq_mask = freqs > 40
        high_freq_reduction = np.mean(raw_fft[high_freq_mask]) - np.mean(
            filtered_fft[high_freq_mask]
        )
        high_freq_reduction_percent = (
            (high_freq_reduction / np.mean(raw_fft[high_freq_mask])) * 100
            if np.mean(raw_fft[high_freq_mask]) > 0
            else 0
        )

        # Signal-to-noise ratio improvement
        raw_snr = (
            10 * np.log10(raw_power / np.var(raw_signal))
            if np.var(raw_signal) > 0
            else 0
        )
        filtered_snr = (
            10 * np.log10(filtered_power / np.var(filtered_data))
            if np.var(filtered_data) > 0
            else 0
        )
        snr_improvement = filtered_snr - raw_snr

        # Peak preservation analysis
        from scipy.signal import find_peaks

        try:
            # Detect peaks in both signals
            raw_peaks, _ = find_peaks(raw_signal, prominence=0.1 * np.std(raw_signal))
            filtered_peaks, _ = find_peaks(
                filtered_data, prominence=0.1 * np.std(filtered_data)
            )

            peak_preservation = (
                len(filtered_peaks) / len(raw_peaks) * 100 if len(raw_peaks) > 0 else 0
            )

            # Peak amplitude preservation
            if len(raw_peaks) > 0 and len(filtered_peaks) > 0:
                raw_peak_amps = raw_signal[raw_peaks]
                filtered_peak_amps = filtered_data[filtered_peaks]
                amplitude_preservation = (
                    np.mean(filtered_peak_amps) / np.mean(raw_peak_amps) * 100
                    if np.mean(raw_peak_amps) > 0
                    else 0
                )
            else:
                amplitude_preservation = 0
        except Exception:
            peak_preservation = 0
            amplitude_preservation = 0

        # Phase distortion analysis
        raw_phase = np.angle(np.fft.fft(raw_signal))
        filtered_phase = np.angle(np.fft.fft(filtered_data))
        phase_distortion = np.mean(np.abs(raw_phase - filtered_phase))

        # Group delay analysis (simplified)
        try:
            # Calculate group delay as derivative of phase
            raw_group_delay = np.gradient(raw_phase)
            filtered_group_delay = np.gradient(filtered_phase)
            group_delay_variation = np.std(filtered_group_delay - raw_group_delay)
        except Exception:
            group_delay_variation = 0

        return html.Div(
            [
                html.H6("🔧 Filtering Results Analysis", className="text-primary mb-3"),
                # Summary metrics cards
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4(
                                                    f"{power_reduction:.1f}%",
                                                    className="text-center text-primary mb-0",
                                                ),
                                                html.Small(
                                                    "Power Reduction",
                                                    className="text-center d-block text-muted",
                                                ),
                                            ]
                                        )
                                    ],
                                    className="text-center border-primary",
                                )
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4(
                                                    f"{snr_improvement:.1f} dB",
                                                    className="text-center text-success mb-0",
                                                ),
                                                html.Small(
                                                    "SNR Improvement",
                                                    className="text-center d-block text-muted",
                                                ),
                                            ]
                                        )
                                    ],
                                    className="text-center border-success",
                                )
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4(
                                                    f"{peak_preservation:.1f}%",
                                                    className="text-center text-warning mb-0",
                                                ),
                                                html.Small(
                                                    "Peak Preservation",
                                                    className="text-center d-block text-muted",
                                                ),
                                            ]
                                        )
                                    ],
                                    className="text-center border-warning",
                                )
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4(
                                                    f"{phase_distortion:.3f}",
                                                    className="text-center text-info mb-0",
                                                ),
                                                html.Small(
                                                    "Phase Distortion",
                                                    className="text-center d-block text-muted",
                                                ),
                                            ]
                                        )
                                    ],
                                    className="text-center border-info",
                                )
                            ],
                            md=3,
                        ),
                    ],
                    className="mb-3",
                ),
                # Overall filtering metrics
                html.H6("Overall Filtering Performance", className="mb-2"),
                dbc.Table(
                    [
                        html.Thead(
                            [
                                html.Tr(
                                    [
                                        html.Th("Metric", className="text-center"),
                                        html.Th("Raw Signal", className="text-center"),
                                        html.Th(
                                            "Filtered Signal", className="text-center"
                                        ),
                                        html.Th("Improvement", className="text-center"),
                                        html.Th("Description", className="text-center"),
                                    ]
                                )
                            ]
                        ),
                        html.Tbody(
                            [
                                html.Tr(
                                    [
                                        html.Td("Signal Power", className="fw-bold"),
                                        html.Td(
                                            f"{raw_power:.3f}", className="text-end"
                                        ),
                                        html.Td(
                                            f"{filtered_power:.3f}",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            f"{power_reduction:.1f}%",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            "Mean squared amplitude",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td("RMS Amplitude", className="fw-bold"),
                                        html.Td(f"{raw_rms:.3f}", className="text-end"),
                                        html.Td(
                                            f"{filtered_rms:.3f}", className="text-end"
                                        ),
                                        html.Td(
                                            f"{rms_reduction:.1f}%",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            "Root mean square amplitude",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td("Signal-to-Noise", className="fw-bold"),
                                        html.Td(
                                            f"{raw_snr:.1f} dB", className="text-end"
                                        ),
                                        html.Td(
                                            f"{filtered_snr:.1f} dB",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            f"{snr_improvement:.1f} dB",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            "Signal quality improvement",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td("Peak Count", className="fw-bold"),
                                        html.Td(
                                            f"{len(raw_peaks) if 'raw_peaks' in locals() else 'N/A'}",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            f"{len(filtered_peaks) if 'filtered_peaks' in locals() else 'N/A'}",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            f"{peak_preservation:.1f}%",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            "Peak detection preservation",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ],
                    bordered=True,
                    hover=True,
                    responsive=True,
                    className="mb-3",
                ),
                # Frequency-specific improvements
                html.H6("Frequency-Domain Improvements", className="mb-2"),
                dbc.Table(
                    [
                        html.Thead(
                            [
                                html.Tr(
                                    [
                                        html.Th(
                                            "Frequency Band", className="text-center"
                                        ),
                                        html.Th("Raw Power", className="text-center"),
                                        html.Th(
                                            "Filtered Power", className="text-center"
                                        ),
                                        html.Th("Reduction", className="text-center"),
                                        html.Th("Description", className="text-center"),
                                    ]
                                )
                            ]
                        ),
                        html.Tbody(
                            [
                                html.Tr(
                                    [
                                        html.Td("DC (0-0.5 Hz)", className="fw-bold"),
                                        html.Td(
                                            f"{np.mean(raw_fft[dc_mask]):.3f}",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            f"{np.mean(filtered_fft[dc_mask]):.3f}",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            f"{dc_reduction_percent:.1f}%",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            "Baseline and drift reduction",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td("Low (0.5-5 Hz)", className="fw-bold"),
                                        html.Td(
                                            f"{np.mean(raw_fft[low_freq_mask]):.3f}",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            f"{np.mean(filtered_fft[low_freq_mask]):.3f}",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            f"{low_freq_reduction_percent:.1f}%",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            "Respiratory noise reduction",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td("Mid (5-40 Hz)", className="fw-bold"),
                                        html.Td(
                                            f"{np.mean(raw_fft[mid_freq_mask]):.3f}",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            f"{np.mean(filtered_fft[mid_freq_mask]):.3f}",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            f"{mid_freq_reduction_percent:.1f}%",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            "Cardiac signal preservation",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td("High (>40 Hz)", className="fw-bold"),
                                        html.Td(
                                            f"{np.mean(raw_fft[high_freq_mask]):.3f}",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            f"{np.mean(filtered_fft[high_freq_mask]):.3f}",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            f"{high_freq_reduction_percent:.1f}%",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            "High frequency noise reduction",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ],
                    bordered=True,
                    hover=True,
                    responsive=True,
                    className="mb-3",
                ),
                # Signal integrity metrics
                html.H6("Signal Integrity Metrics", className="mb-2"),
                dbc.Table(
                    [
                        html.Thead(
                            [
                                html.Tr(
                                    [
                                        html.Th("Metric", className="text-center"),
                                        html.Th("Value", className="text-center"),
                                        html.Td("Quality", className="text-center"),
                                        html.Th("Description", className="text-center"),
                                    ]
                                )
                            ]
                        ),
                        html.Tbody(
                            [
                                html.Tr(
                                    [
                                        html.Td(
                                            "Peak Amplitude Preservation",
                                            className="fw-bold",
                                        ),
                                        html.Td(
                                            f"{amplitude_preservation:.1f}%",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            html.Span(
                                                (
                                                    "Excellent"
                                                    if amplitude_preservation > 95
                                                    else (
                                                        "Good"
                                                        if amplitude_preservation > 90
                                                        else (
                                                            "Fair"
                                                            if amplitude_preservation
                                                            > 80
                                                            else "Poor"
                                                        )
                                                    )
                                                ),
                                                className=f"badge {'bg-success' if amplitude_preservation > 95 else 'bg-info' if amplitude_preservation > 90 else 'bg-warning' if amplitude_preservation > 80 else 'bg-danger'}",
                                            ),
                                            className="text-center",
                                        ),
                                        html.Td(
                                            "Peak height preservation",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(
                                            "Phase Distortion", className="fw-bold"
                                        ),
                                        html.Td(
                                            f"{phase_distortion:.3f} rad",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            html.Span(
                                                (
                                                    "Excellent"
                                                    if phase_distortion < 0.1
                                                    else (
                                                        "Good"
                                                        if phase_distortion < 0.3
                                                        else (
                                                            "Fair"
                                                            if phase_distortion < 0.5
                                                            else "Poor"
                                                        )
                                                    )
                                                ),
                                                className=f"badge {'bg-success' if phase_distortion < 0.1 else 'bg-info' if phase_distortion < 0.3 else 'bg-warning' if phase_distortion < 0.5 else 'bg-danger'}",
                                            ),
                                            className="text-center",
                                        ),
                                        html.Td(
                                            "Phase response distortion",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(
                                            "Group Delay Variation", className="fw-bold"
                                        ),
                                        html.Td(
                                            f"{group_delay_variation:.3f}",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            html.Span(
                                                (
                                                    "Excellent"
                                                    if group_delay_variation < 0.1
                                                    else (
                                                        "Good"
                                                        if group_delay_variation < 0.3
                                                        else (
                                                            "Fair"
                                                            if group_delay_variation
                                                            < 0.5
                                                            else "Poor"
                                                        )
                                                    )
                                                ),
                                                className=f"badge {'bg-success' if group_delay_variation < 0.1 else 'bg-info' if group_delay_variation < 0.3 else 'bg-warning' if group_delay_variation < 0.5 else 'bg-danger'}",
                                            ),
                                            className="text-center",
                                        ),
                                        html.Td(
                                            "Group delay consistency",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ],
                    bordered=True,
                    hover=True,
                    responsive=True,
                    size="sm",
                ),
            ]
        )

    except Exception as e:
        logger.error(f"Error creating filtering results table: {e}")
        return f"Error in filtering analysis: {str(e)}"


def create_additional_metrics_table(
    raw_data, filtered_data, time_axis, sampling_freq, analysis_options, column_mapping
):
    """Create comprehensive additional metrics table."""
    try:
        signal_col = column_mapping.get("signal")
        if not signal_col or signal_col not in raw_data.columns:
            return "Signal column not found in data"

        raw_signal = raw_data[signal_col].values

        # Statistical measures
        signal_mean = np.mean(raw_signal)
        signal_std = np.std(raw_signal)
        signal_min = np.min(raw_signal)
        signal_max = np.max(raw_signal)
        signal_range = signal_max - signal_min
        signal_median = np.median(raw_signal)
        signal_skewness = float(pd.Series(raw_signal).skew())
        signal_kurtosis = float(pd.Series(raw_signal).kurtosis())

        # Entropy measures
        # Shannon entropy
        hist, bins = np.histogram(raw_signal, bins=50, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        shannon_entropy = -np.sum(hist * np.log2(hist))

        # Approximate entropy (simplified)
        try:
            from scipy.stats import entropy

            approx_entropy = entropy(hist) if len(hist) > 1 else 0
        except Exception:
            approx_entropy = 0

        # Fractal dimension (simplified Higuchi method)
        try:
            higuchi_fd = higuchi_fractal_dimension(raw_signal)
        except Exception:
            higuchi_fd = 0

        # Trend analysis
        x_trend = np.arange(len(raw_signal))
        trend_coeffs = np.polyfit(x_trend, raw_signal, 1)
        trend_slope = trend_coeffs[0]

        # Calculate R-squared for trend
        trend_line = np.polyval(trend_coeffs, x_trend)
        ss_res = np.sum((raw_signal - trend_line) ** 2)
        ss_tot = np.sum((raw_signal - np.mean(raw_signal)) ** 2)
        trend_r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Spectral features
        fft_vals = np.abs(np.fft.rfft(raw_signal))
        freqs = np.fft.rfftfreq(len(raw_signal), 1 / sampling_freq)

        # Dominant frequency
        dominant_freq_idx = np.argmax(fft_vals)
        dominant_freq = freqs[dominant_freq_idx]
        dominant_power = fft_vals[dominant_freq_idx]

        # Spectral centroid
        spectral_centroid = (
            np.sum(freqs * fft_vals) / np.sum(fft_vals) if np.sum(fft_vals) > 0 else 0
        )

        # Spectral bandwidth
        spectral_bandwidth = (
            np.sqrt(
                np.sum(((freqs - spectral_centroid) ** 2) * fft_vals) / np.sum(fft_vals)
            )
            if np.sum(fft_vals) > 0
            else 0
        )

        # Spectral rolloff (frequency below which 85% of energy is contained)
        cumulative_power = np.cumsum(fft_vals)
        total_power = cumulative_power[-1]
        rolloff_threshold = 0.85 * total_power
        rolloff_idx = np.where(cumulative_power >= rolloff_threshold)[0]
        spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0

        # Temporal features
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.signbit(raw_signal - signal_mean)))
        zero_crossing_rate = zero_crossings / len(raw_signal)

        # Peak-to-peak amplitude
        peak_to_peak = signal_max - signal_min

        # Crest factor
        crest_factor = (
            signal_max / np.sqrt(np.mean(raw_signal**2))
            if np.mean(raw_signal**2) > 0
            else 0
        )

        # Form factor
        form_factor = (
            np.sqrt(np.mean(raw_signal**2)) / np.mean(np.abs(raw_signal))
            if np.mean(np.abs(raw_signal)) > 0
            else 0
        )

        # Morphological features
        # Signal complexity (based on number of local extrema)
        from scipy.signal import argrelextrema

        try:
            local_maxima = argrelextrema(raw_signal, np.greater, order=3)[0]
            local_minima = argrelextrema(raw_signal, np.less, order=3)[0]
            complexity_score = (len(local_maxima) + len(local_minima)) / len(raw_signal)
        except Exception:
            complexity_score = 0

        # Signal regularity (inverse of complexity)
        regularity_score = 1 - complexity_score if complexity_score <= 1 else 0

        # Cross-correlation with filtered signal (if available)
        if filtered_data is not None and not np.array_equal(raw_signal, filtered_data):
            try:
                from scipy.signal import correlate

                correlation = correlate(raw_signal, filtered_data, mode="full")
                max_correlation = np.max(correlation)
                correlation_coefficient = max_correlation / (
                    np.std(raw_signal) * np.std(filtered_data) * len(raw_signal)
                )
            except Exception:
                correlation_coefficient = 0
        else:
            correlation_coefficient = 1.0  # Perfect correlation if no filtering

        return html.Div(
            [
                html.H6(
                    "📈 Additional Metrics & Advanced Features",
                    className="text-primary mb-3",
                ),
                # Summary metrics cards
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4(
                                                    f"{signal_skewness:.3f}",
                                                    className="text-center text-primary mb-0",
                                                ),
                                                html.Small(
                                                    "Skewness",
                                                    className="text-center d-block text-muted",
                                                ),
                                            ]
                                        )
                                    ],
                                    className="text-center border-primary",
                                )
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4(
                                                    f"{signal_kurtosis:.3f}",
                                                    className="text-center text-success mb-0",
                                                ),
                                                html.Small(
                                                    "Kurtosis",
                                                    className="text-center d-block text-muted",
                                                ),
                                            ]
                                        )
                                    ],
                                    className="text-center border-success",
                                )
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4(
                                                    f"{trend_slope:.3f}",
                                                    className="text-center text-warning mb-0",
                                                ),
                                                html.Small(
                                                    "Trend Slope",
                                                    className="text-center d-block text-muted",
                                                ),
                                            ]
                                        )
                                    ],
                                    className="text-center border-warning",
                                )
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4(
                                                    f"{dominant_freq:.2f} Hz",
                                                    className="text-center text-info mb-0",
                                                ),
                                                html.Small(
                                                    "Dominant Freq",
                                                    className="text-center d-block text-muted",
                                                ),
                                            ]
                                        )
                                    ],
                                    className="text-center border-info",
                                )
                            ],
                            md=3,
                        ),
                    ],
                    className="mb-3",
                ),
                # Statistical measures
                html.H6("Statistical Measures", className="mb-2"),
                dbc.Table(
                    [
                        html.Thead(
                            [
                                html.Tr(
                                    [
                                        html.Th("Metric", className="text-center"),
                                        html.Th("Value", className="text-center"),
                                        html.Th("Description", className="text-center"),
                                    ]
                                )
                            ]
                        ),
                        html.Tbody(
                            [
                                html.Tr(
                                    [
                                        html.Td("Mean", className="fw-bold"),
                                        html.Td(
                                            f"{signal_mean:.3f}", className="text-end"
                                        ),
                                        html.Td(
                                            "Central tendency", className="text-muted"
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td("Median", className="fw-bold"),
                                        html.Td(
                                            f"{signal_median:.3f}", className="text-end"
                                        ),
                                        html.Td("Middle value", className="text-muted"),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(
                                            "Standard Deviation", className="fw-bold"
                                        ),
                                        html.Td(
                                            f"{signal_std:.3f}", className="text-end"
                                        ),
                                        html.Td(
                                            "Variability measure",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td("Range", className="fw-bold"),
                                        html.Td(
                                            f"{signal_range:.3f}", className="text-end"
                                        ),
                                        html.Td(
                                            "Min to max spread", className="text-muted"
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td("Skewness", className="fw-bold"),
                                        html.Td(
                                            f"{signal_skewness:.3f}",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            "Distribution asymmetry",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td("Kurtosis", className="fw-bold"),
                                        html.Td(
                                            f"{signal_kurtosis:.3f}",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            "Distribution peakedness",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ],
                    bordered=True,
                    hover=True,
                    responsive=True,
                    className="mb-3",
                ),
                # Spectral features
                html.H6("Spectral Features", className="mb-2"),
                dbc.Table(
                    [
                        html.Thead(
                            [
                                html.Tr(
                                    [
                                        html.Th("Feature", className="text-center"),
                                        html.Th("Value", className="text-center"),
                                        html.Th("Description", className="text-center"),
                                    ]
                                )
                            ]
                        ),
                        html.Tbody(
                            [
                                html.Tr(
                                    [
                                        html.Td(
                                            "Dominant Frequency", className="fw-bold"
                                        ),
                                        html.Td(
                                            f"{dominant_freq:.2f} Hz",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            "Peak frequency component",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td("Dominant Power", className="fw-bold"),
                                        html.Td(
                                            f"{dominant_power:.2e}",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            "Power at dominant frequency",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(
                                            "Spectral Centroid", className="fw-bold"
                                        ),
                                        html.Td(
                                            f"{spectral_centroid:.2f} Hz",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            "Frequency center of mass",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(
                                            "Spectral Bandwidth", className="fw-bold"
                                        ),
                                        html.Td(
                                            f"{spectral_bandwidth:.2f} Hz",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            "Frequency spread", className="text-muted"
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(
                                            "Spectral Rolloff", className="fw-bold"
                                        ),
                                        html.Td(
                                            f"{spectral_rolloff:.2f} Hz",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            "85% energy frequency",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ],
                    bordered=True,
                    hover=True,
                    responsive=True,
                    className="mb-3",
                ),
                # Temporal and morphological features
                html.H6("Temporal & Morphological Features", className="mb-2"),
                dbc.Table(
                    [
                        html.Thead(
                            [
                                html.Tr(
                                    [
                                        html.Th("Feature", className="text-center"),
                                        html.Th("Value", className="text-center"),
                                        html.Th("Description", className="text-center"),
                                    ]
                                )
                            ]
                        ),
                        html.Tbody(
                            [
                                html.Tr(
                                    [
                                        html.Td(
                                            "Zero Crossing Rate", className="fw-bold"
                                        ),
                                        html.Td(
                                            f"{zero_crossing_rate:.3f}",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            "Signal oscillation frequency",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td("Peak-to-Peak", className="fw-bold"),
                                        html.Td(
                                            f"{peak_to_peak:.3f}", className="text-end"
                                        ),
                                        html.Td(
                                            "Amplitude range", className="text-muted"
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td("Crest Factor", className="fw-bold"),
                                        html.Td(
                                            f"{crest_factor:.3f}", className="text-end"
                                        ),
                                        html.Td(
                                            "Peak to RMS ratio", className="text-muted"
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td("Form Factor", className="fw-bold"),
                                        html.Td(
                                            f"{form_factor:.3f}", className="text-end"
                                        ),
                                        html.Td(
                                            "RMS to mean absolute ratio",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(
                                            "Complexity Score", className="fw-bold"
                                        ),
                                        html.Td(
                                            f"{complexity_score:.3f}",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            "Local extrema density",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(
                                            "Regularity Score", className="fw-bold"
                                        ),
                                        html.Td(
                                            f"{regularity_score:.3f}",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            "Signal smoothness", className="text-muted"
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ],
                    bordered=True,
                    hover=True,
                    responsive=True,
                    className="mb-3",
                ),
                # Advanced features
                html.H6("Advanced Features", className="mb-2"),
                dbc.Table(
                    [
                        html.Thead(
                            [
                                html.Tr(
                                    [
                                        html.Th("Feature", className="text-center"),
                                        html.Th("Value", className="text-center"),
                                        html.Th("Description", className="text-center"),
                                    ]
                                )
                            ]
                        ),
                        html.Tbody(
                            [
                                html.Tr(
                                    [
                                        html.Td("Shannon Entropy", className="fw-bold"),
                                        html.Td(
                                            f"{shannon_entropy:.3f}",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            "Information content",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(
                                            "Approximate Entropy", className="fw-bold"
                                        ),
                                        html.Td(
                                            f"{approx_entropy:.3f}",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            "Signal complexity measure",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(
                                            "Fractal Dimension", className="fw-bold"
                                        ),
                                        html.Td(
                                            f"{higuchi_fd:.3f}", className="text-end"
                                        ),
                                        html.Td(
                                            "Signal self-similarity",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td("Trend R²", className="fw-bold"),
                                        html.Td(
                                            f"{trend_r_squared:.3f}",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            "Linear trend fit quality",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(
                                            "Correlation Coefficient",
                                            className="fw-bold",
                                        ),
                                        html.Td(
                                            f"{correlation_coefficient:.3f}",
                                            className="text-end",
                                        ),
                                        html.Td(
                                            "Raw vs filtered correlation",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ],
                    bordered=True,
                    hover=True,
                    responsive=True,
                    size="sm",
                ),
            ]
        )

    except Exception as e:
        logger.error(f"Error creating additional metrics table: {e}")
        return f"Error in additional metrics: {str(e)}"


def generate_time_domain_stats(
    signal_data, time_axis, sampling_freq, peaks=None, filtered_signal=None
):
    """Generate comprehensive time domain statistics."""
    try:
        results = []

        # Basic statistics
        results.append(html.H5("Signal Statistics"))
        results.append(html.P(f"Duration: {time_axis[-1]:.2f} seconds"))
        results.append(html.P(f"Sampling Frequency: {sampling_freq} Hz"))
        results.append(html.P(f"Signal Length: {len(signal_data)} samples"))
        results.append(html.P(f"Mean Amplitude: {np.mean(signal_data):.4f}"))
        results.append(html.P(f"Std Amplitude: {np.std(signal_data):.4f}"))
        results.append(html.P(f"Min Amplitude: {np.min(signal_data):.4f}"))
        results.append(html.P(f"Max Amplitude: {np.max(signal_data):.4f}"))
        results.append(
            html.P(f"Peak-to-Peak: {np.max(signal_data) - np.min(signal_data):.4f}")
        )
        results.append(html.P(f"RMS: {np.sqrt(np.mean(signal_data**2)):.4f}"))

        # Peak analysis
        if peaks is not None and len(peaks) > 0:
            results.append(html.Hr())
            results.append(html.H6("Peak Analysis"))
            results.append(html.P(f"Number of Peaks: {len(peaks)}"))

            if len(peaks) > 1:
                intervals = np.diff(peaks) / sampling_freq
                results.append(
                    html.P(f"Mean Peak Interval: {np.mean(intervals):.3f} seconds")
                )
                results.append(
                    html.P(f"Peak Rate: {60/np.mean(intervals):.1f} peaks/minute")
                )
                results.append(
                    html.P(f"Peak Interval Std: {np.std(intervals):.3f} seconds")
                )

        # Filter information
        if filtered_signal is not None:
            results.append(html.Hr())
            results.append(html.H6("Filter Information"))
            results.append(html.P("Signal has been filtered"))
            results.append(
                html.P(
                    f"Filtered Signal RMS: {np.sqrt(np.mean(filtered_signal**2)):.4f}"
                )
            )

        return html.Div(results)

    except Exception as e:
        logger.error(f"Error generating time domain stats: {e}")
        return html.Div(
            [html.H5("Error"), html.P(f"Failed to generate stats: {str(e)}")]
        )


def apply_filter(
    signal_data,
    sampling_freq,
    filter_family,
    filter_response,
    low_freq,
    high_freq,
    filter_order,
):
    """Apply filter to the signal."""
    try:
        # Normalize cutoff frequencies
        nyquist = sampling_freq / 2
        low_freq_norm = low_freq / nyquist
        high_freq_norm = high_freq / nyquist

        # Ensure cutoff frequencies are within valid range
        low_freq_norm = max(0.001, min(low_freq_norm, 0.999))
        high_freq_norm = max(0.001, min(high_freq_norm, 0.999))

        # Determine filter type based on response
        if filter_response == "bandpass":
            btype = "band"
            cutoff = [low_freq_norm, high_freq_norm]
        elif filter_response == "bandstop":
            btype = "bandstop"
            cutoff = [low_freq_norm, high_freq_norm]
        elif filter_response == "lowpass":
            btype = "low"
            cutoff = high_freq_norm
        elif filter_response == "highpass":
            btype = "high"
            cutoff = low_freq_norm
        else:
            # Default to bandpass
            btype = "band"
            cutoff = [low_freq_norm, high_freq_norm]

        # Apply filter based on family
        if filter_family == "butter":
            b, a = signal.butter(filter_order, cutoff, btype=btype)
        elif filter_family == "cheby1":
            b, a = signal.cheby1(filter_order, 1, cutoff, btype=btype)
        elif filter_family == "cheby2":
            b, a = signal.cheby2(filter_order, 40, cutoff, btype=btype)
        elif filter_family == "ellip":
            b, a = signal.ellip(filter_order, 1, 40, cutoff, btype=btype)
        elif filter_family == "bessel":
            b, a = signal.bessel(filter_order, cutoff, btype=btype)
        else:
            # Default to Butterworth
            b, a = signal.butter(filter_order, cutoff, btype=btype)

        # Apply filter
        filtered_signal = signal.filtfilt(b, a, signal_data)
        return filtered_signal

    except Exception as e:
        logger.error(f"Error applying filter: {e}")
        return signal_data


def detect_peaks(signal_data, sampling_freq):
    """Detect peaks in the signal."""
    try:
        # Calculate adaptive threshold
        mean_val = np.mean(signal_data)
        std_val = np.std(signal_data)
        threshold = mean_val + 2 * std_val

        # Find peaks with minimum distance constraint
        min_distance = int(sampling_freq * 0.1)  # Minimum 0.1 seconds between peaks
        peaks, _ = signal.find_peaks(
            signal_data, height=threshold, distance=min_distance
        )

        return peaks

    except Exception as e:
        logger.error(f"Error detecting peaks: {e}")
        return np.array([])


def register_vitaldsp_callbacks(app):
    """Register all vitalDSP analysis callbacks."""

    @app.callback(
        [
            Output("main-signal-plot", "figure"),
            Output("filtered-signal-plot", "figure"),
            Output("analysis-results", "children"),
            Output("peak-analysis-table", "children"),
            Output("signal-quality-table", "children"),
            Output("filtering-results-table", "children"),
            Output("additional-metrics-table", "children"),
            Output("store-time-domain-data", "data"),
            Output("store-filtered-data", "data"),
        ],
        [
            Input("btn-update-analysis", "n_clicks"),
            Input("time-range-slider", "value"),
            Input("start-time", "value"),
            Input("end-time", "value"),
            Input("btn-nudge-m10", "n_clicks"),
            Input("btn-nudge-m1", "n_clicks"),
            Input("btn-nudge-p1", "n_clicks"),
            Input("btn-nudge-p10", "n_clicks"),
            Input("url", "pathname"),
        ],
        [
            State("filter-family", "value"),
            State("filter-response", "value"),
            State("filter-low-freq", "value"),
            State("filter-high-freq", "value"),
            State("filter-order", "value"),
            State("analysis-options", "value"),
            State("signal-type-select", "value"),
        ],
    )
    def analyze_time_domain(
        n_clicks,
        slider_value,
        start_time,
        end_time,
        nudge_m10,
        nudge_m1,
        nudge_p1,
        nudge_p10,
        pathname,
        filter_family,
        filter_response,
        filter_low_freq,
        filter_high_freq,
        filter_order,
        analysis_options,
        signal_type,
    ):
        """Main time domain analysis callback."""
        logger.info("=== TIME DOMAIN ANALYSIS CALLBACK ===")
        logger.info(
            f"Input values - start_time: {start_time}, end_time: {end_time}, slider_value: {slider_value}"
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
                create_empty_figure(),
                create_empty_figure(),
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
            from vitalDSP_webapp.services.data.data_service import get_data_service

            data_service = get_data_service()
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
                    create_empty_figure(),
                    create_empty_figure(),
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
                    create_empty_figure(),
                    create_empty_figure(),
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
            logger.info(f"Data frame info: {df.info() if df is not None else 'None'}")
            logger.info(f"Data frame head: {df.head() if df is not None else 'None'}")

            if df is None or df.empty:
                logger.warning("Data frame is empty")
                return (
                    create_empty_figure(),
                    create_empty_figure(),
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

            # Handle time window adjustments for nudge buttons and slider
            if trigger_id in [
                "btn-nudge-m10",
                "btn-nudge-m1",
                "btn-nudge-p1",
                "btn-nudge-p10",
            ]:
                if not start_time or not end_time:
                    start_time, end_time = 0, 10

                if trigger_id == "btn-nudge-m10":
                    start_time = max(0, start_time - 10)
                    end_time = max(10, end_time - 10)
                elif trigger_id == "btn-nudge-m1":
                    start_time = max(0, start_time - 1)
                    end_time = max(1, end_time - 1)
                elif trigger_id == "btn-nudge-p1":
                    start_time = start_time + 1
                    end_time = end_time + 1
                elif trigger_id == "btn-nudge-p10":
                    start_time = start_time + 10
                    end_time = end_time + 10

                logger.info(f"Time window adjusted: {start_time} to {end_time}")

            # Handle time range slider changes
            elif trigger_id == "time-range-slider" and slider_value:
                start_time = slider_value[0]
                end_time = slider_value[1]
                logger.info(f"Time range slider changed: {start_time} to {end_time}")

            # Handle manual time input changes
            elif trigger_id in ["start-time", "end-time"]:
                logger.info(
                    f"Manual time input changed - start_time: {start_time}, end_time: {end_time}"
                )

            # Set default time window if not specified
            if not start_time or not end_time:
                start_time, end_time = 0, 10
                logger.info(f"Using default time window: {start_time} to {end_time}")

            # Ensure time values are numbers
            try:
                start_time = float(start_time) if start_time is not None else 0
                end_time = float(end_time) if end_time is not None else 10
                logger.info(
                    f"Converted time values: start_time={start_time:.3f}, end_time={end_time:.3f}"
                )
            except (ValueError, TypeError):
                start_time, end_time = 0, 10
                logger.warning("Invalid time values, using defaults")

            # Apply time window
            start_sample = int(start_time * sampling_freq)
            end_sample = int(end_time * sampling_freq)
            logger.info(f"Sample range: {start_sample} to {end_sample}")
            logger.info(f"Data length: {len(df)}")
            logger.info(f"Time window: {start_time:.3f}s to {end_time:.3f}s")

            # Ensure we don't exceed data bounds
            if start_sample >= len(df):
                logger.warning(
                    f"Start sample {start_sample} >= data length {len(df)}, adjusting to 0"
                )
                start_sample = 0
                start_time = 0
            if end_sample > len(df):
                logger.warning(
                    f"End sample {end_sample} > data length {len(df)}, adjusting to {len(df)}"
                )
                end_sample = len(df)
                end_time = len(df) / sampling_freq

            windowed_data = df.iloc[start_sample:end_sample].copy()
            logger.info(f"Windowed data shape: {windowed_data.shape}")
            logger.info(f"Windowed data columns: {list(windowed_data.columns)}")
            logger.info(f"Windowed data head: {windowed_data.head()}")

            # Create time axis - use actual time values, not just windowed data length
            time_axis = np.linspace(start_time, end_time, len(windowed_data))
            logger.info(f"Time axis shape: {time_axis.shape}")
            logger.info(f"Time axis range: {time_axis[0]:.3f} to {time_axis[-1]:.3f}")
            logger.info(f"Requested time window: {start_time:.3f} to {end_time:.3f}")
            logger.info(
                f"Actual time axis range: {time_axis[0]:.3f} to {time_axis[-1]:.3f}"
            )

            # Verify that the time window is actually being applied
            logger.info(f"Original data length: {len(df)} samples")
            logger.info(f"Windowed data length: {len(windowed_data)} samples")
            logger.info(
                f"Expected samples for {end_time - start_time:.3f}s window: {(end_time - start_time) * sampling_freq:.0f}"
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
                        create_empty_figure(),
                        create_empty_figure(),
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
                    create_empty_figure(),
                    create_empty_figure(),
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

            # Apply filtering if requested
            filtered_signal = None
            if (
                filter_family
                and filter_response
                and filter_low_freq
                and filter_high_freq
                and filter_order
            ):
                logger.info("Applying filter to signal...")
                logger.info(
                    f"Filter params: {filter_family}, {filter_response}, {filter_low_freq}-{filter_high_freq} Hz, order {filter_order}"
                )
                filtered_signal = apply_filter(
                    signal_data,
                    sampling_freq,
                    filter_family,
                    filter_response,
                    filter_low_freq,
                    filter_high_freq,
                    filter_order,
                )
                logger.info("Filter applied successfully")
                logger.info(
                    f"Filtered signal shape: {filtered_signal.shape if filtered_signal is not None else 'None'}"
                )
                if filtered_signal is not None:
                    logger.info(
                        f"Filtered signal range: {np.min(filtered_signal):.3f} to {np.max(filtered_signal):.3f}"
                    )
            else:
                logger.info("No filter parameters provided, using raw signal")

            # Create filtered DataFrame with both raw and filtered signals for comparison
            if filtered_signal is not None:
                filtered_df = windowed_data.copy()
                filtered_df[f"{signal_column}_filtered"] = filtered_signal
            else:
                filtered_df = windowed_data.copy()

            # Detect peaks if requested
            peaks = None
            if analysis_options and "peaks" in analysis_options:
                logger.info("Detecting peaks in signal...")
                peaks = detect_peaks(signal_data, sampling_freq)
                logger.info(f"Detected {len(peaks)} peaks")
            else:
                logger.info("Peak detection not requested")

            # Create main signal plot
            logger.info("Creating main signal plot...")
            logger.info(
                f"Calling create_main_signal_plot with column_mapping: {column_mapping}"
            )
            main_plot = create_main_signal_plot(
                windowed_data,
                time_axis,
                sampling_freq,
                analysis_options or ["peaks", "hr", "quality"],
                column_mapping,
                signal_type or "PPG",
            )
            logger.info("Main signal plot created successfully")

            # Create filtered signal plot
            logger.info("Creating filtered signal plot...")
            filtered_plot = create_filtered_signal_plot(
                filtered_df,
                time_axis,
                sampling_freq,
                column_mapping,
                signal_type or "PPG",
                analysis_options,
            )
            logger.info("Filtered signal plot created successfully")

            # Store processed data
            time_domain_data = {
                "raw_data": windowed_data.to_dict("records"),
                "time_axis": time_axis.tolist(),
                "sampling_freq": sampling_freq,
                "window": [start_time, end_time],
            }

            filtered_data_store = {
                "filtered_data": (
                    windowed_data.to_dict("records")
                    if filtered_signal is not None
                    else None
                ),
                "filter_params": {
                    "family": filter_family,
                    "response": filter_response,
                    "low_freq": filter_low_freq,
                    "high_freq": filter_high_freq,
                    "order": filter_order,
                },
            }

            # Generate analysis results
            analysis_results = generate_analysis_results(
                windowed_data,
                filtered_signal,
                time_axis,
                sampling_freq,
                analysis_options or ["peaks", "hr", "quality"],
                column_mapping,
            )

            # Generate detailed table components
            peak_table = create_peak_analysis_table(
                windowed_data,
                filtered_signal,
                time_axis,
                sampling_freq,
                analysis_options or ["peaks", "hr", "quality"],
                column_mapping,
            )
            quality_table = create_signal_quality_table(
                windowed_data,
                filtered_signal,
                time_axis,
                sampling_freq,
                analysis_options or ["peaks", "hr", "quality"],
                column_mapping,
            )
            filtering_table = create_filtering_results_table(
                windowed_data,
                filtered_signal,
                time_axis,
                sampling_freq,
                analysis_options or ["peaks", "hr", "quality"],
                column_mapping,
            )
            additional_table = create_additional_metrics_table(
                windowed_data,
                filtered_signal,
                time_axis,
                sampling_freq,
                analysis_options or ["peaks", "hr", "quality"],
                column_mapping,
            )

            logger.info("Time domain analysis completed successfully")
            logger.info("Returning results:")
            logger.info(f"  - Main plot: {type(main_plot)}")
            logger.info(f"  - Filtered plot: {type(filtered_plot)}")
            logger.info(f"  - Analysis results: {type(analysis_results)}")
            logger.info(f"  - Peak table: {type(peak_table)}")
            logger.info(f"  - Quality table: {type(quality_table)}")
            logger.info(f"  - Filtering table: {type(filtering_table)}")
            logger.info(f"  - Additional table: {type(additional_table)}")
            logger.info(f"  - Data: {type(time_domain_data)}")
            logger.info(f"  - Filtered data: {type(filtered_data_store)}")

            return (
                main_plot,
                filtered_plot,
                analysis_results,
                peak_table,
                quality_table,
                filtering_table,
                additional_table,
                time_domain_data,
                filtered_data_store,
            )

        except Exception as e:
            logger.error(f"Error in time domain analysis callback: {e}")
            import traceback

            traceback.print_exc()
            error_msg = f"Error in analysis: {str(e)}"
            return (
                create_empty_figure(),
                create_empty_figure(),
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
            Output("time-range-slider", "min"),
            Output("time-range-slider", "max"),
            Output("time-range-slider", "value"),
        ],
        [Input("url", "pathname")],
    )
    def update_time_slider_range(pathname):
        """Update time slider range based on available data."""
        if pathname != "/time-domain":
            return no_update, no_update, no_update

        try:
            from vitalDSP_webapp.services.data.data_service import get_data_service

            data_service = get_data_service()
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

    @app.callback(
        [Output("start-time", "value"), Output("end-time", "value")],
        [Input("time-range-slider", "value")],
        prevent_initial_call=True,
        prevent_update=True,
    )
    def sync_time_inputs_with_slider(slider_value):
        """Sync time input fields with slider values to prevent circular updates."""
        if not slider_value or len(slider_value) != 2:
            return no_update, no_update

        logger.info(
            f"Syncing time inputs with slider: {slider_value[0]} to {slider_value[1]}"
        )
        return slider_value[0], slider_value[1]

    @app.callback(
        [
            Output("fft-params", "style"),
            Output("psd-params", "style"),
            Output("stft-params", "style"),
            Output("wavelet-params", "style"),
        ],
        [Input("freq-analysis-type", "value")],
    )
    def toggle_frequency_params(analysis_type):
        """Show/hide parameter sections based on selected analysis type."""
        # Default style (hidden)
        hidden_style = {"display": "none"}
        visible_style = {"display": "block"}

        # Initialize all as hidden
        fft_style = hidden_style
        psd_style = hidden_style
        stft_style = hidden_style
        wavelet_style = hidden_style

        # Show relevant section based on analysis type
        if analysis_type == "fft":
            fft_style = visible_style
        elif analysis_type == "psd":
            psd_style = visible_style
        elif analysis_type == "stft":
            stft_style = visible_style
        elif analysis_type == "wavelet":
            wavelet_style = visible_style

        return fft_style, psd_style, stft_style, wavelet_style


def create_fft_plot(
    signal_data, sampling_freq, window_type, n_points, freq_min, freq_max
):
    """Create FFT plot using vitalDSP FourierTransform."""
    try:
        from vitalDSP.transforms.fourier_transform import FourierTransform

        # Apply window if specified
        if window_type and window_type != "none":
            if window_type == "hann":
                windowed_signal = signal_data * np.hanning(len(signal_data))
            elif window_type == "hamming":
                windowed_signal = signal_data * np.hamming(len(signal_data))
            elif window_type == "blackman":
                windowed_signal = signal_data * np.blackman(len(signal_data))
            else:
                windowed_signal = signal_data
        else:
            windowed_signal = signal_data

        # Use vitalDSP FourierTransform
        ft = FourierTransform(windowed_signal)
        fft_result = ft.compute_dft()

        # Compute frequency axis
        fft_freq = np.fft.fftfreq(len(fft_result), 1 / sampling_freq)

        # Get positive frequencies only
        positive_freq_mask = fft_freq > 0
        fft_freq = fft_freq[positive_freq_mask]
        fft_magnitude = np.abs(fft_result[positive_freq_mask])

        # Apply frequency range filter if specified
        if freq_min is not None and freq_max is not None:
            freq_mask = (fft_freq >= freq_min) & (fft_freq <= freq_max)
            fft_freq = fft_freq[freq_mask]
            fft_magnitude = fft_magnitude[freq_mask]

        # Create plot
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=fft_freq,
                y=fft_magnitude,
                mode="lines",
                name="FFT Magnitude",
                line=dict(color="blue", width=2),
            )
        )

        fig.update_layout(
            title="Fast Fourier Transform (FFT)",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Magnitude",
            showlegend=True,
            plot_bgcolor="white",
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating FFT plot: {e}")
        return create_empty_figure()


def create_stft_plot(
    signal_data, sampling_freq, window_size, hop_size, freq_min, freq_max
):
    """Create STFT plot using vitalDSP STFT."""
    try:
        from vitalDSP.transforms.stft import STFT

        # Use vitalDSP STFT
        stft_obj = STFT(signal_data, window_size=window_size, hop_size=hop_size)
        stft_result = stft_obj.compute_stft()

        # Get time and frequency axes
        time_axis = np.arange(stft_result.shape[1]) * hop_size / sampling_freq
        freq_axis = np.fft.rfftfreq(window_size, 1 / sampling_freq)

        # Apply frequency range filter if specified
        if freq_min is not None and freq_max is not None:
            freq_mask = (freq_axis >= freq_min) & (freq_axis <= freq_max)
            freq_axis = freq_axis[freq_mask]
            stft_result = stft_result[freq_mask, :]

        # Create plot
        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=np.abs(stft_result),
                x=time_axis,
                y=freq_axis,
                colorscale="viridis",
                name="STFT Magnitude",
            )
        )

        fig.update_layout(
            title="Short-Time Fourier Transform (STFT)",
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
            showlegend=True,
            plot_bgcolor="white",
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating STFT plot: {e}")
        return create_empty_figure()


def create_wavelet_plot(
    signal_data, sampling_freq, wavelet_type, levels, freq_min, freq_max
):
    """Create wavelet plot using vitalDSP WaveletTransform."""
    try:
        from vitalDSP.transforms.wavelet_transform import WaveletTransform

        # Use vitalDSP WaveletTransform
        wt = WaveletTransform(signal_data, wavelet_name=wavelet_type)

        # Perform wavelet decomposition
        approximations = []
        details = []
        current_signal = signal_data.copy()

        for level in range(levels):
            approx, detail = wt._wavelet_decompose(current_signal)
            approximations.append(approx)
            details.append(detail)
            current_signal = approx

        # Create subplots for each level
        fig = make_subplots(
            rows=levels + 1,
            cols=1,
            subplot_titles=[f"Level {i+1} Detail" for i in range(levels)]
            + ["Final Approximation"],
            vertical_spacing=0.05,
        )

        # Add detail coefficients
        for i, detail in enumerate(details):
            time_axis = np.arange(len(detail)) / sampling_freq
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=detail,
                    mode="lines",
                    name=f"Level {i+1} Detail",
                    line=dict(width=1),
                ),
                row=i + 1,
                col=1,
            )

        # Add final approximation
        time_axis = np.arange(len(approximations[-1])) / sampling_freq
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=approximations[-1],
                mode="lines",
                name="Final Approximation",
                line=dict(color="red", width=2),
            ),
            row=levels + 1,
            col=1,
        )

        fig.update_layout(
            title=f"Wavelet Transform ({wavelet_type}) - {levels} Levels",
            height=300 * (levels + 1),
            showlegend=True,
            plot_bgcolor="white",
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating wavelet plot: {e}")
        return create_empty_figure()


def create_enhanced_psd_plot(
    signal_data,
    sampling_freq,
    window_sec,
    overlap,
    freq_max,
    log_scale,
    normalize,
    channel,
    column_mapping,
):
    """Create enhanced PSD plot using vitalDSP FrequencyDomainFeatures."""
    try:
        from vitalDSP.physiological_features.frequency_domain import (
            FrequencyDomainFeatures,
        )

        # Convert window from seconds to samples
        window_samples = int(window_sec * sampling_freq)

        # Use scipy.signal.welch for PSD computation
        from scipy.signal import welch

        # Apply window function
        windowed_signal = signal_data * np.hanning(len(signal_data))

        # Compute PSD
        freqs, psd = welch(
            windowed_signal,
            fs=sampling_freq,
            nperseg=min(window_samples, len(signal_data)),
            noverlap=int(min(window_samples, len(signal_data)) * overlap),
        )

        # Apply frequency range filter
        if freq_max:
            freq_mask = freqs <= freq_max
            freqs = freqs[freq_mask]
            psd = psd[freq_mask]

        # Apply log scale if requested
        if log_scale and "on" in log_scale:
            psd = 10 * np.log10(psd + 1e-10)  # Add small value to avoid log(0)
            y_label = "Power Spectral Density (dB/Hz)"
        else:
            y_label = "Power Spectral Density"

        # Normalize if requested
        if normalize and "on" in normalize:
            psd = psd / np.max(psd)
            y_label += " (Normalized)"

        # Create plot
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=freqs,
                y=psd,
                mode="lines",
                name="PSD",
                line=dict(color="green", width=2),
            )
        )

        fig.update_layout(
            title="Power Spectral Density (PSD)",
            xaxis_title="Frequency (Hz)",
            yaxis_title=y_label,
            showlegend=True,
            plot_bgcolor="white",
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating PSD plot: {e}")
        return create_empty_figure()


def create_enhanced_spectrogram_plot(
    signal_data,
    sampling_freq,
    window_size,
    overlap,
    freq_max,
    colormap,
    scaling,
    window_type,
):
    """Create enhanced spectrogram plot."""
    try:
        from scipy.signal import spectrogram

        # Apply window function
        if window_type == "hann":
            window_func = np.hanning(window_size)
        elif window_type == "hamming":
            window_func = np.hamming(window_size)
        elif window_type == "blackman":
            window_func = np.blackman(window_size)
        elif window_type == "kaiser":
            window_func = np.kaiser(window_size, beta=14)
        elif window_type == "gaussian":
            window_func = np.exp(
                -0.5
                * ((np.arange(window_size) - window_size / 2) / (window_size / 6)) ** 2
            )
        else:
            window_func = np.ones(window_size)

        # Compute spectrogram
        freqs, times, Sxx = spectrogram(
            signal_data,
            fs=sampling_freq,
            window=window_func,
            nperseg=window_size,
            noverlap=int(window_size * overlap),
        )

        # Apply frequency range filter
        if freq_max:
            freq_mask = freqs <= freq_max
            freqs = freqs[freq_mask]
            Sxx = Sxx[freq_mask, :]

        # Apply scaling
        if scaling == "density":
            Sxx = Sxx / sampling_freq
            title_suffix = " (Density)"
        else:
            title_suffix = " (Spectrum)"

        # Create plot
        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=10 * np.log10(Sxx + 1e-10),  # Convert to dB
                x=times,
                y=freqs,
                colorscale=colormap,
                name="Spectrogram",
            )
        )

        fig.update_layout(
            title=f"Time-Frequency Spectrogram{title_suffix}",
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
            showlegend=True,
            plot_bgcolor="white",
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating spectrogram plot: {e}")
        return create_empty_figure()


def generate_frequency_analysis_results(
    signal_data, sampling_freq, analysis_type, analysis_options
):
    """Generate frequency analysis results summary."""
    try:
        results = []

        # Basic signal information
        results.append(html.H5("Signal Information"))
        results.append(html.P(f"Signal Length: {len(signal_data)} samples"))
        results.append(
            html.P(f"Duration: {len(signal_data) / sampling_freq:.2f} seconds")
        )
        results.append(html.P(f"Sampling Frequency: {sampling_freq} Hz"))
        results.append(html.P(f"Nyquist Frequency: {sampling_freq / 2:.2f} Hz"))

        # Analysis type specific results
        results.append(html.H5(f"Analysis Type: {analysis_type.upper()}"))

        if analysis_type == "fft":
            # FFT specific analysis
            fft_result = np.fft.fft(signal_data)
            fft_freq = np.fft.fftfreq(len(signal_data), 1 / sampling_freq)

            # Find dominant frequency
            positive_freq_mask = fft_freq > 0
            fft_freq_pos = fft_freq[positive_freq_mask]
            fft_magnitude_pos = np.abs(fft_result[positive_freq_mask])

            if len(fft_magnitude_pos) > 0:
                dominant_freq_idx = np.argmax(fft_magnitude_pos)
                dominant_freq = fft_freq_pos[dominant_freq_idx]
                results.append(html.P(f"Dominant Frequency: {dominant_freq:.2f} Hz"))
                results.append(
                    html.P(
                        f"Frequency Resolution: {fft_freq_pos[1] - fft_freq_pos[0]:.4f} Hz"
                    )
                )

        elif analysis_type == "psd":
            # PSD specific analysis
            from scipy.signal import welch

            freqs, psd = welch(signal_data, fs=sampling_freq)

            # Find peak frequency
            peak_freq_idx = np.argmax(psd)
            peak_freq = freqs[peak_freq_idx]
            results.append(html.P(f"Peak Frequency: {peak_freq:.2f} Hz"))
            results.append(html.P(f"Total Power: {np.sum(psd):.2e}"))

        elif analysis_type == "stft":
            # STFT specific analysis
            results.append(html.P("STFT provides time-frequency representation"))
            results.append(html.P("Check the spectrogram plot for detailed analysis"))

        elif analysis_type == "wavelet":
            # Wavelet specific analysis
            results.append(
                html.P("Wavelet transform provides multi-resolution analysis")
            )
            results.append(html.P("Check the wavelet plot for decomposition levels"))

        return html.Div(results)

    except Exception as e:
        logger.error(f"Error generating frequency analysis results: {e}")
        return html.Div(
            [html.H5("Error"), html.P(f"Failed to generate results: {str(e)}")]
        )


def create_frequency_peak_analysis_table(
    signal_data, sampling_freq, analysis_type, analysis_options
):
    """Create frequency peak analysis table."""
    try:
        if "peak_detection" not in analysis_options:
            return html.Div([html.P("Peak detection not selected")])

        # Perform peak detection on frequency domain
        if analysis_type == "fft":
            fft_result = np.fft.fft(signal_data)
            fft_freq = np.fft.fftfreq(len(signal_data), 1 / sampling_freq)

            # Get positive frequencies
            positive_freq_mask = fft_freq > 0
            fft_freq_pos = fft_freq[positive_freq_mask]
            fft_magnitude_pos = np.abs(fft_result[positive_freq_mask])

            # Find peaks
            from scipy.signal import find_peaks

            peaks, properties = find_peaks(
                fft_magnitude_pos, height=np.max(fft_magnitude_pos) * 0.1
            )

            if len(peaks) > 0:
                # Create table
                table_data = []
                for i, peak_idx in enumerate(peaks[:10]):  # Limit to top 10 peaks
                    freq = fft_freq_pos[peak_idx]
                    magnitude = fft_magnitude_pos[peak_idx]
                    table_data.append(
                        [f"Peak {i+1}", f"{freq:.2f} Hz", f"{magnitude:.2e}"]
                    )

                return dbc.Table.from_dataframe(
                    pd.DataFrame(
                        table_data, columns=["Peak", "Frequency (Hz)", "Magnitude"]
                    ),
                    striped=True,
                    bordered=True,
                    hover=True,
                )
            else:
                return html.P("No significant peaks found")

        else:
            return html.P(f"Peak detection for {analysis_type} not implemented yet")

    except Exception as e:
        logger.error(f"Error creating peak analysis table: {e}")
        return html.Div(
            [html.H5("Error"), html.P(f"Failed to create peak table: {str(e)}")]
        )


def create_frequency_band_power_table(
    signal_data, sampling_freq, analysis_type, analysis_options
):
    """Create frequency band power analysis table."""
    try:
        if "band_power" not in analysis_options:
            return html.Div([html.P("Band power analysis not selected")])

        # Define frequency bands
        bands = {
            "Delta (0.5-4 Hz)": (0.5, 4),
            "Theta (4-8 Hz)": (4, 8),
            "Alpha (8-13 Hz)": (8, 13),
            "Beta (13-30 Hz)": (13, 30),
            "Gamma (30-100 Hz)": (30, 100),
        }

        # Compute PSD
        from scipy.signal import welch

        freqs, psd = welch(signal_data, fs=sampling_freq)

        # Calculate band powers
        band_powers = {}
        for band_name, (low_freq, high_freq) in bands.items():
            freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
            if np.any(freq_mask):
                power = np.trapz(psd[freq_mask], freqs[freq_mask])
                band_powers[band_name] = power

        # Create table
        table_data = []
        for band_name, power in band_powers.items():
            table_data.append([band_name, f"{power:.2e}"])

        return dbc.Table.from_dataframe(
            pd.DataFrame(table_data, columns=["Frequency Band", "Power"]),
            striped=True,
            bordered=True,
            hover=True,
        )

    except Exception as e:
        logger.error(f"Error creating band power table: {e}")
        return html.Div(
            [html.H5("Error"), html.P(f"Failed to create band power table: {str(e)}")]
        )


def create_frequency_stability_table(
    signal_data, sampling_freq, analysis_type, analysis_options
):
    """Create frequency stability analysis table."""
    try:
        if "stability" not in analysis_options:
            return html.Div([html.P("Stability analysis not selected")])

        # Basic stability metrics
        mean_value = np.mean(signal_data)
        std_value = np.std(signal_data)
        cv = std_value / abs(mean_value) if mean_value != 0 else float("inf")

        # Frequency stability (using FFT)
        fft_result = np.fft.fft(signal_data)
        fft_freq = np.fft.fftfreq(len(signal_data), 1 / sampling_freq)

        # Get positive frequencies
        positive_freq_mask = fft_freq > 0
        fft_freq_pos = fft_freq[positive_freq_mask]
        fft_magnitude_pos = np.abs(fft_result[positive_freq_mask])

        # Find dominant frequency
        if len(fft_magnitude_pos) > 0:
            dominant_freq_idx = np.argmax(fft_magnitude_pos)
            dominant_freq = fft_freq_pos[dominant_freq_idx]
        else:
            dominant_freq = 0

        # Create table
        table_data = [
            ["Mean Value", f"{mean_value:.4f}"],
            ["Standard Deviation", f"{std_value:.4f}"],
            ["Coefficient of Variation", f"{cv:.4f}"],
            ["Dominant Frequency", f"{dominant_freq:.2f} Hz"],
            ["Signal Length", f"{len(signal_data)} samples"],
        ]

        return dbc.Table.from_dataframe(
            pd.DataFrame(table_data, columns=["Metric", "Value"]),
            striped=True,
            bordered=True,
            hover=True,
        )

    except Exception as e:
        logger.error(f"Error creating stability table: {e}")
        return html.Div(
            [html.H5("Error"), html.P(f"Failed to create stability table: {str(e)}")]
        )


def create_frequency_harmonics_table(
    signal_data, sampling_freq, analysis_type, analysis_options
):
    """Create frequency harmonics analysis table."""
    try:
        if "harmonic_analysis" not in analysis_options:
            return html.Div([html.P("Harmonic analysis not selected")])

        # Perform FFT for harmonic analysis
        fft_result = np.fft.fft(signal_data)
        fft_freq = np.fft.fftfreq(len(signal_data), 1 / sampling_freq)

        # Get positive frequencies
        positive_freq_mask = fft_freq > 0
        fft_freq_pos = fft_freq[positive_freq_mask]
        fft_magnitude_pos = np.abs(fft_result[positive_freq_mask])

        if len(fft_magnitude_pos) > 0:
            # Find fundamental frequency (highest peak)
            from scipy.signal import find_peaks

            peaks, properties = find_peaks(
                fft_magnitude_pos, height=np.max(fft_magnitude_pos) * 0.1
            )

            if len(peaks) > 0:
                fundamental_freq = fft_freq_pos[peaks[0]]

                # Find harmonics (multiples of fundamental)
                harmonics = []
                for i in range(1, 6):  # Look for up to 5th harmonic
                    harmonic_freq = fundamental_freq * i

                    # Find closest frequency in our data
                    freq_idx = np.argmin(np.abs(fft_freq_pos - harmonic_freq))
                    if freq_idx < len(fft_magnitude_pos):
                        harmonic_magnitude = fft_magnitude_pos[freq_idx]
                        harmonics.append(
                            [
                                f"{i}st",
                                f"{harmonic_freq:.2f} Hz",
                                f"{harmonic_magnitude:.2e}",
                            ]
                        )

                # Create table
                table_data = [
                    [
                        "Fundamental",
                        f"{fundamental_freq:.2f} Hz",
                        f"{fft_magnitude_pos[peaks[0]]:.2e}",
                    ]
                ]
                table_data.extend(harmonics)

                return dbc.Table.from_dataframe(
                    pd.DataFrame(
                        table_data, columns=["Harmonic", "Frequency (Hz)", "Magnitude"]
                    ),
                    striped=True,
                    bordered=True,
                    hover=True,
                )
            else:
                return html.P("No significant peaks found for harmonic analysis")
        else:
            return html.P("Insufficient data for harmonic analysis")

    except Exception as e:
        logger.error(f"Error creating harmonics table: {e}")
        return html.Div(
            [html.H5("Error"), html.P(f"Failed to create harmonics table: {str(e)}")]
        )
