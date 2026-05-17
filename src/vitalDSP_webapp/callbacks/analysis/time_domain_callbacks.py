"""
Time Domain Analysis callbacks for vitalDSP webapp.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, callback_context, no_update, html, dcc
import dash_bootstrap_components as dbc
import logging

from vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks import (
    apply_traditional_filter,
)
from vitalDSP_webapp.callbacks.core.theme_callbacks import apply_plot_theme

logger = logging.getLogger(__name__)


def configure_plot_with_pan_zoom(fig, title="", height=400):
    """Compat helper — configure a plotly figure with pan/zoom defaults."""
    fig.update_layout(
        title=title,
        height=height,
        showlegend=True,
        template="plotly_white",
        dragmode="pan",
        xaxis=dict(rangeslider=dict(visible=False), type="linear"),
        modebar=dict(
            add=[
                "pan2d",
                "zoom2d",
                "zoomIn2d",
                "zoomOut2d",
                "autoScale2d",
                "resetScale2d",
            ]
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# Compat helpers (kept so tests / imports don't break)
# ─────────────────────────────────────────────────────────────


def format_large_number(value, precision=3, use_scientific=False):
    if value == 0:
        return "0"
    abs_value = abs(value)
    if use_scientific or abs_value >= 1e6:
        return f"{value:.{precision}e}"
    elif abs_value >= 1e3:
        return f"{value / 1e3:.{precision}f}k"
    elif abs_value >= 1:
        return f"{value:.{precision}f}"
    elif abs_value >= 1e-3:
        return f"{value * 1e3:.{precision}f}m"
    else:
        return f"{value:.{precision}e}"


def higuchi_fractal_dimension(signal, k_max=10):
    try:
        from vitalDSP.physiological_features.nonlinear import NonlinearFeatures

        nf = NonlinearFeatures(signal=signal)
        raw = float(nf.compute_fractal_dimension(kmax=int(k_max)))
        if np.isnan(raw) or np.isinf(raw):
            return 0.0
        return float(np.clip(raw, 0.0, 2.0))
    except Exception:
        sig = np.asarray(signal, dtype=float)
        if sig.size < 4:
            return 0.0
        diffs = np.abs(np.diff(sig))
        return float(np.clip(1.0 + diffs.mean() / (sig.std() + 1e-9), 0.0, 2.0))


def create_empty_figure(theme="light"):
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


def create_signal_comparison_plot(*args, **kwargs):
    return create_empty_figure()


def create_time_domain_plot(
    signal_data=None,
    time_axis=None,
    sampling_freq=None,
    peaks=None,
    filtered_signal=None,
    signal_type="PPG",
    theme="light",
    *args,
    **kwargs,
):
    """Compat wrapper — delegates to create_main_signal_plot."""
    if signal_data is None or time_axis is None:
        return create_empty_figure(theme)
    return create_main_signal_plot(
        signal_data=np.asarray(signal_data, dtype=float),
        time_axis=np.asarray(time_axis, dtype=float),
        sampling_freq=sampling_freq or 1000,
        signal_type=signal_type,
        theme=theme,
        filtered_overlay=filtered_signal,
    )


def create_peak_analysis_plot(
    signal_data=None, time_axis=None, peaks=None, sampling_freq=None, **kwargs
):
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
            peak_idx = peak_idx[(peak_idx >= 0) & (peak_idx < len(time_axis))]
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


def create_filtered_signal_plot(*args, **kwargs):
    fig = go.Figure()
    pos = list(args)
    df = pos[0] if pos else kwargs.get("df")
    time_axis = pos[1] if len(pos) > 1 else kwargs.get("time_axis")
    column_mapping = kwargs.get("column_mapping", None)
    if column_mapping is None:
        for cand in pos[3:6]:
            if isinstance(cand, dict):
                column_mapping = cand
                break
    column_mapping = column_mapping or {}
    signal_col = (
        column_mapping.get("signal") or column_mapping.get("amplitude") or "signal"
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


def create_signal_source_table(
    signal_source_info=None,
    filter_info=None,
    sampling_freq=None,
    signal_length=None,
    *args,
    **kwargs,
):
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
        html.Tr([html.Td("Sampling rate"), html.Td(f"{fs:.1f} Hz" if fs else "-")]),
        html.Tr(
            [
                html.Td("Samples"),
                html.Td(f"{int(signal_length):,}" if signal_length else "-"),
            ]
        ),
    ]
    if isinstance(filter_info, dict) and filter_info:
        rows.append(
            html.Tr(
                [html.Td("Filter"), html.Td(str(filter_info.get("filter_type", "-")))]
            )
        )
    return dbc.Table(
        [
            html.Thead(html.Tr([html.Th("Property"), html.Th("Value")])),
            html.Tbody(rows),
        ],
        bordered=False,
        striped=True,
        size="sm",
        className="mb-2",
    )


def create_filtering_results_table(*args, **kwargs):
    return html.Div()


def create_additional_metrics_table(*args, **kwargs):
    return html.Div()


def generate_time_domain_stats(
    signal=None,
    time_axis=None,
    sampling_freq=None,
    peaks=None,
    filtered_signal=None,
    **kwargs,
):
    if signal is None:
        return html.Div(
            [
                html.H6("Signal Statistics", className="mb-2"),
                html.P("Error: no signal provided.", className="small text-danger"),
            ]
        )
    children = [html.H6("Signal Statistics", className="mb-2")]
    try:
        fs = (
            float(np.asarray(sampling_freq).item())
            if sampling_freq is not None
            else 0.0
        )
    except (TypeError, ValueError):
        fs = 0.0
    try:
        n = len(signal)
    except TypeError:
        n = 0
    duration = n / fs if fs else 0.0
    try:
        mean_amp = float(np.mean(np.asarray(signal, dtype=float))) if n else 0.0
    except (TypeError, ValueError):
        mean_amp = 0.0
    children.append(
        dbc.Table(
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(
                                f"Sampling Frequency: {fs:.1f} Hz"
                                if fs
                                else "Sampling Frequency: -"
                            )
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td(
                                f"Duration: {duration:.2f} s" if fs else "Duration: -"
                            )
                        ]
                    ),
                    html.Tr([html.Td(f"Signal Length: {n:,} samples")]),
                    html.Tr([html.Td(f"Mean Amplitude: {mean_amp:.4f}")]),
                ]
            ),
            bordered=False,
            striped=True,
            size="sm",
            className="mb-2",
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


# ─────────────────────────────────────────────────────────────
# Main signal plot — shows ALL critical points for signal type
# ─────────────────────────────────────────────────────────────

# Alternating soft palette for beat/cycle bands
_CYCLE_COLORS = [
    "rgba(144, 238, 144, 0.18)",  # soft green
    "rgba(255, 223, 128, 0.18)",  # soft amber
]


def _add_cycle_bands(fig, signal_data, time_axis, sampling_freq, signal_type, wm=None):
    """Add alternating translucent background bands, one per beat/cycle.

    PPG: uses trough-to-trough sessions from detect_ppg_session.
    ECG: uses midpoints between consecutive R-peaks as cycle boundaries.
    Falls back to midpoints between systolic/r peaks if session detection fails.
    """
    try:
        boundaries = None  # list of (start_idx, end_idx) tuples

        if signal_type == "PPG":
            try:
                sessions = wm.detect_ppg_session()
                if sessions is not None and len(sessions) > 0:
                    boundaries = [(int(s[0]), int(s[1])) for s in sessions]
            except Exception as e:
                logger.debug(
                    "PPG session detection failed, falling back to peak midpoints: %s",
                    e,
                )

            if not boundaries:
                # Fallback: midpoints between systolic peaks
                _sp = getattr(wm, "systolic_peaks", None)
                peaks = np.asarray(_sp if _sp is not None else [], dtype=int)
                if peaks.size >= 2:
                    mids = np.round((peaks[:-1] + peaks[1:]) / 2).astype(int)
                    boundaries = [
                        (int(mids[i]), int(mids[i + 1])) for i in range(len(mids) - 1)
                    ]
                    # prepend first segment from 0 to first mid
                    boundaries = (
                        [(0, int(mids[0]))]
                        + boundaries
                        + [(int(mids[-1]), len(signal_data) - 1)]
                    )

        elif signal_type == "ECG":
            # For ECG use midpoints between R-peaks as natural beat boundaries
            _rp = getattr(wm, "r_peaks", None)
            r_peaks = np.asarray(_rp if _rp is not None else [], dtype=int)
            if r_peaks.size >= 2:
                mids = np.round((r_peaks[:-1] + r_peaks[1:]) / 2).astype(int)
                boundaries = [
                    (int(mids[i]), int(mids[i + 1])) for i in range(len(mids) - 1)
                ]
                boundaries = (
                    [(0, int(mids[0]))]
                    + boundaries
                    + [(int(mids[-1]), len(signal_data) - 1)]
                )

        if not boundaries:
            return

        n = len(signal_data)
        for i, (s, e) in enumerate(boundaries):
            s = max(0, min(s, n - 1))
            e = max(s + 1, min(e, n - 1))
            x0 = float(time_axis[s])
            x1 = float(time_axis[e])
            color = _CYCLE_COLORS[i % len(_CYCLE_COLORS)]
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor=color,
                layer="below",
                line_width=0,
            )

    except Exception as e:
        logger.debug("Cycle band shading failed: %s", e)


def _add_critical_points(
    fig, signal_data, time_axis, sampling_freq, signal_type, wm=None
):
    """Detect and add all critical points for the given signal type to fig."""
    try:
        from vitalDSP.physiological_features.waveform import WaveformMorphology

        if wm is None:
            wm = WaveformMorphology(
                waveform=signal_data,
                fs=sampling_freq,
                signal_type=signal_type,
                simple_mode=True,
            )

        def _safe_add(indices, name, color, symbol, size=8):
            if indices is None or len(indices) == 0:
                return
            idx = np.asarray(indices, dtype=int)
            idx = idx[(idx >= 0) & (idx < len(signal_data))]
            if idx.size == 0:
                return
            fig.add_trace(
                go.Scatter(
                    x=time_axis[idx],
                    y=signal_data[idx],
                    mode="markers",
                    name=name,
                    marker=dict(color=color, size=size, symbol=symbol),
                    hovertemplate=f"<b>{name}:</b> %{{y:.4f}}<extra></extra>",
                )
            )

        if signal_type == "PPG":
            _safe_add(
                getattr(wm, "systolic_peaks", None),
                "Systolic Peaks",
                "#e63946",
                "diamond",
                10,
            )
            try:
                _safe_add(
                    wm.detect_dicrotic_notches(),
                    "Dicrotic Notches",
                    "#f4a261",
                    "circle",
                    8,
                )
            except Exception as e:
                logger.warning("Dicrotic notch detection failed: %s", e)
            try:
                _safe_add(
                    wm.detect_diastolic_peak(),
                    "Diastolic Peaks",
                    "#2a9d8f",
                    "square",
                    8,
                )
            except Exception as e:
                logger.warning("Diastolic peak detection failed: %s", e)

        elif signal_type == "ECG":
            _safe_add(getattr(wm, "r_peaks", None), "R Peaks", "#e63946", "diamond", 10)
            try:
                _safe_add(wm.detect_p_peak(), "P Peaks", "#457b9d", "circle", 8)
            except Exception as e:
                logger.warning("P peak detection failed: %s", e)
            try:
                _safe_add(wm.detect_t_peak(), "T Peaks", "#2a9d8f", "square", 8)
            except Exception as e:
                logger.warning("T peak detection failed: %s", e)
            try:
                _safe_add(
                    wm.detect_q_valley(), "Q Valleys", "#f4a261", "triangle-down", 7
                )
            except Exception as e:
                logger.warning("Q valley detection failed: %s", e)
            try:
                _safe_add(
                    wm.detect_s_valley(), "S Valleys", "#c77dff", "triangle-down", 7
                )
            except Exception as e:
                logger.warning("S valley detection failed: %s", e)

    except Exception as e:
        logger.warning("Critical points detection failed: %s", e)


def create_main_signal_plot(*args, **kwargs):
    """
    Main signal plot with all critical points.

    New shape: create_main_signal_plot(signal_data, time_axis, fs,
        peaks=None, signal_type='PPG', theme='light',
        filtered_overlay=None, wm=None)

    Legacy shape (tests): 4th positional is a list → extract signal from df.
    """
    import pandas as pd

    if not args and not kwargs:
        return create_empty_figure()

    legacy = len(args) >= 4 and isinstance(args[3], (list, tuple))

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
            signal_data = np.asarray(df)
        peaks = None
        filtered_overlay = None
        wm = None
    else:
        signal_data = args[0] if args else kwargs.get("signal_data")
        time_axis = args[1] if len(args) > 1 else kwargs.get("time_axis")
        sampling_freq = args[2] if len(args) > 2 else kwargs.get("sampling_freq", 1000)
        peaks = (
            args[3] if len(args) > 3 else kwargs.get("peaks")
        )  # kept for compat only
        signal_type = args[4] if len(args) > 4 else kwargs.get("signal_type", "PPG")
        theme = args[5] if len(args) > 5 else kwargs.get("theme", "light")
        filtered_overlay = args[6] if len(args) > 6 else kwargs.get("filtered_overlay")
        wm = kwargs.get("wm")  # shared WaveformMorphology if available

    try:
        n_samples = len(signal_data) if signal_data is not None else 0
    except TypeError:
        n_samples = 0

    if signal_data is None or time_axis is None or n_samples == 0:
        return create_empty_figure(theme)

    try:
        fs_val = float(sampling_freq) if sampling_freq is not None else 0.0
    except (TypeError, ValueError):
        fs_val = 0.0

    # ── base signal trace ──────────────────────────────────────
    fig = go.Figure()

    if filtered_overlay is not None:
        # Show original faint, filtered bold
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=signal_data,
                mode="lines",
                name="Original",
                line=dict(color="rgba(100,100,100,0.4)", width=1),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=filtered_overlay,
                mode="lines",
                name="Filtered",
                line=dict(color="#1f77b4", width=1.6),
            )
        )
        plot_signal = filtered_overlay
    else:
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=signal_data,
                mode="lines",
                name="Signal",
                line=dict(color="#1f77b4", width=1.4),
            )
        )
        plot_signal = signal_data

    # ── cycle background bands ─────────────────────────────────
    if wm is not None:
        _add_cycle_bands(
            fig, plot_signal, time_axis, fs_val, signal_type or "PPG", wm=wm
        )

    # ── critical points ────────────────────────────────────────
    _add_critical_points(
        fig, plot_signal, time_axis, fs_val, signal_type or "PPG", wm=wm
    )

    duration_s = n_samples / max(fs_val, 1e-9)
    fig.update_layout(
        title=f"{signal_type or 'Signal'} — {duration_s:.1f} s window",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        showlegend=True,
        height=420,
        template="plotly_white",
        dragmode="pan",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return apply_plot_theme(fig, theme)


# ─────────────────────────────────────────────────────────────
# Signal summary banner
# ─────────────────────────────────────────────────────────────


def generate_analysis_results(*args, **kwargs):
    selected_signal = kwargs.get("selected_signal")
    time_axis = kwargs.get("time_axis")
    sampling_freq = kwargs.get("sampling_freq")
    analysis_options = kwargs.get("analysis_options")
    signal_source_info = kwargs.get("signal_source_info")
    signal_type = kwargs.get("signal_type")
    filter_info = kwargs.get("filter_info")

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

    def _fs():
        try:
            arr = np.asarray(sampling_freq)
            if arr.size == 1:
                return float(arr.item())
        except (TypeError, ValueError):
            pass
        return 0.0

    fs = _fs()
    try:
        n = len(selected_signal) if selected_signal is not None else 0
    except TypeError:
        n = 0

    badges = []
    if signal_source_info:
        badges.append(
            dbc.Badge(str(signal_source_info), color="primary", className="me-1")
        )
    if signal_type:
        badges.append(dbc.Badge(str(signal_type), color="info", className="me-1"))
    if fs:
        badges.append(dbc.Badge(f"{fs:.0f} Hz", color="secondary", className="me-1"))
    if n and fs:
        dur = n / fs
        badges.append(
            dbc.Badge(
                f"{dur:.2f} s / {n:,} samples",
                color="light",
                text_color="dark",
                className="me-1",
            )
        )

    rows = [
        html.Tr(
            [
                html.Td("Signal source", className="fw-semibold text-muted small"),
                html.Td(str(signal_source_info or "—")),
            ]
        ),
        html.Tr(
            [
                html.Td("Signal type", className="fw-semibold text-muted small"),
                html.Td(str(signal_type or "—")),
            ]
        ),
        html.Tr(
            [
                html.Td("Sampling rate", className="fw-semibold text-muted small"),
                html.Td(f"{fs:.1f} Hz" if fs else "—"),
            ]
        ),
        html.Tr(
            [
                html.Td("Window samples", className="fw-semibold text-muted small"),
                html.Td(f"{n:,}"),
            ]
        ),
        html.Tr(
            [
                html.Td("Window duration", className="fw-semibold text-muted small"),
                html.Td(f"{(n / fs):.2f} s" if fs else "—"),
            ]
        ),
    ]
    if isinstance(filter_info, dict) and filter_info:
        ftype = filter_info.get("filter_type", "unknown")
        params = filter_info.get("parameters", {}) or {}
        rows.append(
            html.Tr(
                [
                    html.Td("Filter", className="fw-semibold text-muted small"),
                    html.Td(
                        f"{ftype}"
                        + (
                            f" ({', '.join(f'{k}={v}' for k, v in params.items())})"
                            if params
                            else ""
                        )
                    ),
                ]
            )
        )

    return html.Div(
        [
            html.Div(badges, className="mb-2"),
            dbc.Table(
                [html.Tbody(rows)],
                bordered=False,
                size="sm",
                className="mb-0 table-borderless",
            ),
        ]
    )


# ─────────────────────────────────────────────────────────────
# Rich HRV / time-domain results panel
# ─────────────────────────────────────────────────────────────


def _safe_float(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _fmt(v, unit="", decimals=2):
    f = _safe_float(v)
    if f is None:
        return "—"
    return f"{f:.{decimals}f}{(' ' + unit) if unit else ''}"


def _metric_card(title, value, unit, description, color="primary"):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, className="text-muted small mb-1"),
                html.Div(
                    [
                        html.Span(value, className=f"fs-4 fw-bold text-{color}"),
                        html.Span(f" {unit}", className="text-muted small"),
                    ],
                ),
                html.Div(
                    description, className="text-muted", style={"fontSize": "0.72rem"}
                ),
            ]
        ),
        className="h-100 border-0 shadow-sm",
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
    """
    Rich time-domain feature panel:
    - Key metric cards (HR, SDNN, RMSSD, pNN50)
    - Full time-domain table with ALL vitalDSP features
    - NN-interval distribution sparkline
    - Successive-difference sparkline
    """
    try:
        sig = np.asarray(selected_signal, dtype=float)
        fs = float(sampling_freq) if sampling_freq else 100.0
        stype = signal_type or "PPG"

        # ── peak detection ─────────────────────────────────────
        if peaks is None:
            try:
                from vitalDSP.physiological_features.waveform import WaveformMorphology

                wm = WaveformMorphology(
                    waveform=sig, fs=fs, signal_type=stype, simple_mode=True
                )
                peaks = getattr(
                    wm, "systolic_peaks" if stype == "PPG" else "r_peaks", None
                )
            except Exception:
                peaks = None

        peaks = (
            np.asarray([], dtype=int) if peaks is None else np.asarray(peaks, dtype=int)
        )
        n_peaks = int(peaks.size)
        duration_s = len(sig) / max(fs, 1e-9)
        mean_hr = (60.0 * n_peaks / duration_s) if duration_s > 0 else 0.0

        nn_intervals = np.diff(peaks) / fs * 1000.0 if n_peaks >= 2 else np.array([])

        # ── HRV features via library ───────────────────────────
        hrv = {}
        if nn_intervals.size >= 2:
            try:
                from vitalDSP.physiological_features.hrv_analysis import HRVFeatures

                hrv = (
                    HRVFeatures(
                        nn_intervals=nn_intervals.tolist(),
                        signal=sig,
                        fs=fs,
                    ).compute_all_features(include_complex_methods=False)
                    or {}
                )
            except Exception:
                pass

            # Compute all TimeDomainFeatures individually and merge.
            # Note: compute_nn20 does not exist — derive it from the diffs directly.
            try:
                from vitalDSP.physiological_features.time_domain import (
                    TimeDomainFeatures,
                )

                tdf = TimeDomainFeatures(nn_intervals)
                diffs = np.abs(np.diff(nn_intervals))
                nn20_count = int(np.sum(diffs > 20))
                td_direct = {
                    "sdnn": tdf.compute_sdnn(),
                    "rmssd": tdf.compute_rmssd(),
                    "nn50": float(tdf.compute_nn50()),
                    "pnn50": tdf.compute_pnn50(),
                    "nn20": float(nn20_count),
                    "pnn20": tdf.compute_pnn20(),
                    "mean_nn": tdf.compute_mean_nn(),
                    "median_nn": tdf.compute_median_nn(),
                    "iqr_nn": tdf.compute_iqr_nn(),
                    "sdsd": tdf.compute_sdsd(),
                    "cvnn": tdf.compute_cvnn(),
                    "hrv_triangular_index": tdf.compute_hrv_triangular_index(),
                    "tinn": tdf.compute_tinn(),
                }
                # Direct computation wins over HRVFeatures aggregator
                hrv = {**hrv, **td_direct}
            except Exception as _e:
                logger.warning("TimeDomainFeatures computation failed: %s", _e)

        def g(key, fb=None):
            v = hrv.get(key, fb)
            return _safe_float(v) if v is not None else None

        sdnn = g("sdnn") or (float(np.std(nn_intervals)) if nn_intervals.size else 0.0)
        rmssd = g("rmssd") or (
            float(np.sqrt(np.mean(np.diff(nn_intervals) ** 2)))
            if nn_intervals.size >= 2
            else 0.0
        )
        pnn50 = g("pnn50") or 0.0
        mean_nn = g("mean_nn") or (
            float(np.mean(nn_intervals)) if nn_intervals.size else 0.0
        )

        # ── Key metric cards ───────────────────────────────────
        metric_cards = dbc.Row(
            [
                dbc.Col(
                    _metric_card(
                        "Heart Rate",
                        f"{mean_hr:.1f}",
                        "bpm",
                        f"{n_peaks} beats detected",
                        "danger",
                    ),
                    md=3,
                    className="mb-3",
                ),
                dbc.Col(
                    _metric_card(
                        "SDNN",
                        f"{sdnn:.2f}",
                        "ms",
                        "Overall HRV variability",
                        "primary",
                    ),
                    md=3,
                    className="mb-3",
                ),
                dbc.Col(
                    _metric_card(
                        "RMSSD",
                        f"{rmssd:.2f}",
                        "ms",
                        "Short-term variability",
                        "success",
                    ),
                    md=3,
                    className="mb-3",
                ),
                dbc.Col(
                    _metric_card(
                        "pNN50",
                        f"{pnn50:.1f}",
                        "%",
                        "Parasympathetic activity",
                        "warning",
                    ),
                    md=3,
                    className="mb-3",
                ),
            ],
            className="g-2",
        )

        # ── Full time-domain feature table ─────────────────────
        td_rows = [
            ("Mean NN", _fmt(g("mean_nn"), "ms"), "Mean RR / beat-to-beat interval"),
            ("Median NN", _fmt(g("median_nn"), "ms"), "Robust central tendency"),
            ("SDNN", _fmt(g("sdnn"), "ms"), "Std dev of NN intervals — global HRV"),
            (
                "RMSSD",
                _fmt(g("rmssd"), "ms"),
                "Root-mean-square successive diffs — vagal tone",
            ),
            ("SDSD", _fmt(g("sdsd"), "ms"), "Std dev of successive diffs"),
            ("NN50", _fmt(g("nn50"), decimals=0), "Intervals differing >50 ms"),
            ("pNN50", _fmt(g("pnn50"), "%"), "Proportion of NN50 pairs"),
            ("NN20", _fmt(g("nn20"), decimals=0), "Intervals differing >20 ms"),
            ("pNN20", _fmt(g("pnn20"), "%"), "Proportion of NN20 pairs"),
            ("IQR NN", _fmt(g("iqr_nn"), "ms"), "Interquartile range of NN intervals"),
            ("CVNN", _fmt(g("cvnn")), "Coefficient of variation (SDNN / mean NN)"),
            (
                "HRV Triangular Index",
                _fmt(g("hrv_triangular_index")),
                "Total NN / max histogram bin",
            ),
            (
                "TINN",
                _fmt(g("tinn"), "ms"),
                "Triangular interpolation of NN histogram baseline",
            ),
        ]

        feature_table = dbc.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Feature", className="small"),
                            html.Th("Value", className="small"),
                            html.Th("Interpretation", className="small text-muted"),
                        ]
                    )
                ),
                html.Tbody(
                    [
                        html.Tr(
                            [
                                html.Td(name, className="fw-semibold small"),
                                html.Td(val, className="small font-monospace"),
                                html.Td(desc, className="small text-muted"),
                            ]
                        )
                        for name, val, desc in td_rows
                    ]
                ),
            ],
            bordered=False,
            striped=True,
            hover=True,
            size="sm",
            className="mb-3",
        )

        # ── Visualisations ─────────────────────────────────────
        _CHART_H = 260
        _MARGIN = dict(l=48, r=24, t=44, b=36)
        _TPL = "plotly_white"
        charts = []

        if nn_intervals.size >= 4:
            beats = list(range(len(nn_intervals)))
            nn_list = nn_intervals.tolist()
            nn_mean = float(np.mean(nn_intervals))
            nn_sd = float(np.std(nn_intervals))

            # ── 1. NN tachogram with ±1 SD band + rolling RMSSD ──
            win = max(3, len(nn_intervals) // 5)
            roll_rmssd = [
                (
                    float(
                        np.sqrt(
                            np.mean(np.diff(nn_intervals[max(0, i - win) : i + 1]) ** 2)
                        )
                    )
                    if i >= 1
                    else 0.0
                )
                for i in range(len(nn_intervals))
            ]
            nn_fig = make_subplots(specs=[[{"secondary_y": True}]])
            # SD band
            nn_fig.add_trace(
                go.Scatter(
                    x=beats + beats[::-1],
                    y=[nn_mean + nn_sd] * len(beats) + [nn_mean - nn_sd] * len(beats),
                    fill="toself",
                    fillcolor="rgba(31,119,180,0.10)",
                    line=dict(width=0),
                    name="±1 SD",
                    showlegend=True,
                ),
                secondary_y=False,
            )
            # Mean line
            nn_fig.add_hline(
                y=nn_mean,
                line=dict(color="#1f77b4", dash="dash", width=1),
                annotation_text=f"mean {nn_mean:.0f} ms",
                annotation_font_size=10,
            )
            # NN line
            nn_fig.add_trace(
                go.Scatter(
                    x=beats,
                    y=nn_list,
                    mode="lines+markers",
                    name="NN (ms)",
                    line=dict(color="#1f77b4", width=1.8),
                    marker=dict(size=5, color="#1f77b4"),
                ),
                secondary_y=False,
            )
            # Rolling RMSSD on secondary axis
            nn_fig.add_trace(
                go.Scatter(
                    x=beats,
                    y=roll_rmssd,
                    mode="lines",
                    name=f"Rolling RMSSD (w={win})",
                    line=dict(color="#e63946", width=1.4, dash="dot"),
                ),
                secondary_y=True,
            )
            nn_fig.update_layout(
                title="NN Tachogram — beat-to-beat intervals",
                height=_CHART_H,
                template=_TPL,
                margin=_MARGIN,
                legend=dict(orientation="h", y=-0.25, x=0, font=dict(size=10)),
            )
            nn_fig.update_yaxes(title_text="NN (ms)", secondary_y=False)
            nn_fig.update_yaxes(
                title_text="RMSSD (ms)",
                secondary_y=True,
                showgrid=False,
                color="#e63946",
            )
            nn_fig.update_xaxes(title_text="Beat #")
            charts.append(
                dbc.Col(
                    dcc.Graph(figure=nn_fig, config={"displayModeBar": False}), md=6
                )
            )

            # ── 2. Instantaneous Heart Rate over beats ────────────
            hr_series = 60_000.0 / nn_intervals  # bpm per interval
            hr_mean = float(np.mean(hr_series))
            hr_fig = go.Figure()
            hr_fig.add_trace(
                go.Scatter(
                    x=beats,
                    y=hr_series.tolist(),
                    mode="lines+markers",
                    name="HR (bpm)",
                    line=dict(color="#2a9d8f", width=1.8),
                    marker=dict(size=5),
                    fill="tozeroy",
                    fillcolor="rgba(42,157,143,0.08)",
                    hovertemplate="Beat %{x}: %{y:.1f} bpm<extra></extra>",
                )
            )
            hr_fig.add_hline(
                y=hr_mean,
                line=dict(color="#2a9d8f", dash="dash", width=1),
                annotation_text=f"mean {hr_mean:.1f} bpm",
                annotation_font_size=10,
            )
            hr_fig.update_layout(
                title="Instantaneous Heart Rate",
                xaxis_title="Beat #",
                yaxis_title="HR (bpm)",
                height=_CHART_H,
                template=_TPL,
                margin=_MARGIN,
                showlegend=False,
            )
            charts.append(
                dbc.Col(
                    dcc.Graph(figure=hr_fig, config={"displayModeBar": False}), md=6
                )
            )

            # ── 3. NN distribution — histogram + KDE curve ────────
            n_bins = max(10, min(50, len(nn_intervals) // 3))
            hist_counts, hist_edges = np.histogram(nn_intervals, bins=n_bins)
            bin_centers = (hist_edges[:-1] + hist_edges[1:]) / 2.0
            # Gaussian KDE
            bw = 1.06 * nn_sd * (len(nn_intervals) ** -0.2)  # Silverman's rule
            x_kde = np.linspace(
                float(nn_intervals.min()), float(nn_intervals.max()), 200
            )
            kde_y = np.array(
                [
                    np.sum(np.exp(-0.5 * ((x - nn_intervals) / bw) ** 2))
                    / (len(nn_intervals) * bw * np.sqrt(2 * np.pi))
                    for x in x_kde
                ]
            )
            # Scale KDE to histogram counts
            bin_width = (
                float(hist_edges[1] - hist_edges[0]) if len(hist_edges) > 1 else 1.0
            )
            kde_scaled = kde_y * len(nn_intervals) * bin_width

            dist_fig = go.Figure()
            dist_fig.add_trace(
                go.Bar(
                    x=bin_centers.tolist(),
                    y=hist_counts.tolist(),
                    name="Count",
                    marker_color="rgba(42,157,143,0.55)",
                    width=bin_width * 0.85,
                )
            )
            dist_fig.add_trace(
                go.Scatter(
                    x=x_kde.tolist(),
                    y=kde_scaled.tolist(),
                    mode="lines",
                    name="KDE",
                    line=dict(color="#2a9d8f", width=2),
                )
            )
            # Mean ± SD references
            for xv, label, dash in [
                (nn_mean, f"μ={nn_mean:.0f}", "dash"),
                (nn_mean - nn_sd, f"μ−σ={nn_mean - nn_sd:.0f}", "dot"),
                (nn_mean + nn_sd, f"μ+σ={nn_mean + nn_sd:.0f}", "dot"),
            ]:
                dist_fig.add_vline(
                    x=xv,
                    line=dict(color="#457b9d", dash=dash, width=1.2),
                    annotation_text=label,
                    annotation_font_size=9,
                    annotation_position="top right",
                )
            dist_fig.update_layout(
                title="NN Interval Distribution",
                xaxis_title="NN (ms)",
                yaxis_title="Count",
                height=_CHART_H,
                template=_TPL,
                margin=_MARGIN,
                legend=dict(orientation="h", y=-0.25, x=0, font=dict(size=10)),
                barmode="overlay",
            )
            charts.append(
                dbc.Col(
                    dcc.Graph(figure=dist_fig, config={"displayModeBar": False}), md=6
                )
            )

            # ── 4. |ΔNN| violin + box + 20 ms / 50 ms thresholds ─
            succ_diffs = np.abs(np.diff(nn_intervals)).tolist()
            diff_fig = go.Figure()
            diff_fig.add_trace(
                go.Violin(
                    y=succ_diffs,
                    name="|ΔNN|",
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor="rgba(228,57,70,0.25)",
                    line_color="#e63946",
                    points="all",
                    marker=dict(size=4, opacity=0.5, color="#e63946"),
                )
            )
            diff_fig.add_hline(
                y=20,
                line=dict(color="#f4a261", dash="dash", width=1.2),
                annotation_text="NN20 (20 ms)",
                annotation_font_size=9,
                annotation_position="top left",
            )
            diff_fig.add_hline(
                y=50,
                line=dict(color="#e63946", dash="dash", width=1.2),
                annotation_text="NN50 (50 ms)",
                annotation_font_size=9,
                annotation_position="top right",
            )
            diff_fig.update_layout(
                title="Successive NN Differences",
                yaxis_title="Abs diff (ms)",
                xaxis=dict(visible=False),
                height=_CHART_H,
                template=_TPL,
                margin=_MARGIN,
                showlegend=False,
            )
            charts.append(
                dbc.Col(
                    dcc.Graph(figure=diff_fig, config={"displayModeBar": False}), md=6
                )
            )

        # ── 5. Poincaré with SD1/SD2 ellipse ──────────────────
        if nn_intervals.size >= 6:
            x_poi = nn_intervals[:-1]
            y_poi = nn_intervals[1:]
            # SD1 = std of perpendicular direction, SD2 = std of diagonal
            diff_xy = (y_poi - x_poi) / np.sqrt(2)
            sum_xy = (x_poi + y_poi) / np.sqrt(2)
            sd1 = float(np.std(diff_xy, ddof=1))
            sd2 = float(np.std(sum_xy, ddof=1))
            cx = float(np.mean(x_poi))
            cy = float(np.mean(y_poi))

            # Ellipse parametric
            theta = np.linspace(0, 2 * np.pi, 120)
            ellipse_x = (
                cx
                + sd2 * np.cos(theta) * np.cos(np.pi / 4)
                - sd1 * np.sin(theta) * np.sin(np.pi / 4)
            )
            ellipse_y = (
                cy
                + sd2 * np.cos(theta) * np.sin(np.pi / 4)
                + sd1 * np.sin(theta) * np.cos(np.pi / 4)
            )

            mn = float(min(x_poi.min(), y_poi.min()))
            mx = float(max(x_poi.max(), y_poi.max()))

            poi_fig = go.Figure()
            # Scatter points
            poi_fig.add_trace(
                go.Scatter(
                    x=x_poi.tolist(),
                    y=y_poi.tolist(),
                    mode="markers",
                    name="RRn / RRn+1",
                    marker=dict(color="#457b9d", size=6, opacity=0.55),
                    hovertemplate="RRn=%{x:.0f} ms, RRn+1=%{y:.0f} ms<extra></extra>",
                )
            )
            # SD1/SD2 ellipse
            poi_fig.add_trace(
                go.Scatter(
                    x=ellipse_x.tolist(),
                    y=ellipse_y.tolist(),
                    mode="lines",
                    name="SD1/SD2 ellipse",
                    line=dict(color="#e63946", width=1.6, dash="dot"),
                )
            )
            # Identity line
            poi_fig.add_trace(
                go.Scatter(
                    x=[mn, mx],
                    y=[mn, mx],
                    mode="lines",
                    name="Identity",
                    line=dict(color="rgba(100,100,100,0.5)", dash="dash", width=1),
                )
            )
            # SD1 and SD2 axis lines through centroid
            # SD1 axis: perpendicular to identity → direction (-1,1)/√2
            sd1_dx = sd1 * (-1 / np.sqrt(2))
            sd1_dy = sd1 * (1 / np.sqrt(2))
            poi_fig.add_trace(
                go.Scatter(
                    x=[cx - sd1_dx, cx + sd1_dx],
                    y=[cy - sd1_dy, cy + sd1_dy],
                    mode="lines",
                    name=f"SD1={sd1:.1f} ms",
                    line=dict(color="#2a9d8f", width=2),
                )
            )
            # SD2 axis: along identity → direction (1,1)/√2
            sd2_dx = sd2 * (1 / np.sqrt(2))
            sd2_dy = sd2 * (1 / np.sqrt(2))
            poi_fig.add_trace(
                go.Scatter(
                    x=[cx - sd2_dx, cx + sd2_dx],
                    y=[cy - sd2_dy, cy + sd2_dy],
                    mode="lines",
                    name=f"SD2={sd2:.1f} ms",
                    line=dict(color="#f4a261", width=2),
                )
            )
            poi_fig.update_layout(
                title=f"Poincaré  SD1={sd1:.1f} ms · SD2={sd2:.1f} ms · SD1/SD2={sd1/max(sd2,1e-9):.2f}",
                xaxis_title="RRn (ms)",
                yaxis_title="RRn+1 (ms)",
                height=_CHART_H,
                template=_TPL,
                margin=_MARGIN,
                legend=dict(orientation="h", y=-0.30, x=0, font=dict(size=10)),
            )
            charts.append(
                dbc.Col(
                    dcc.Graph(figure=poi_fig, config={"displayModeBar": False}), md=6
                )
            )

        # ── 6. HRV range bullet chart ──────────────────────────
        # Shows each metric's value as a dot against its typical
        # low / normal / high reference bands.
        if nn_intervals.size >= 4:
            # (label, value, low_norm, high_norm, unit, x_max)
            bullet_specs = [
                ("SDNN", g("sdnn") or 0.0, 20.0, 100.0, "ms", 160.0),
                ("RMSSD", g("rmssd") or 0.0, 15.0, 80.0, "ms", 120.0),
                ("pNN50", g("pnn50") or 0.0, 5.0, 40.0, "%", 60.0),
                ("CVNN", (g("cvnn") or 0.0) * 100, 3.0, 10.0, "%", 15.0),
                ("HRV-TI", g("hrv_triangular_index") or 0.0, 8.0, 30.0, "", 45.0),
            ]

            bullet_fig = go.Figure()
            n_rows = len(bullet_specs)
            row_h = 1.0 / n_rows  # fractional height per row in y-domain

            for i, (label, val, lo, hi, unit, x_max) in enumerate(bullet_specs):
                # y-domain for this row (bottom to top)
                y0 = i / n_rows
                y1 = (i + 1) / n_rows
                pad = row_h * 0.15
                cy = (y0 + y1) / 2  # vertical centre of this row

                val_clipped = min(float(val), x_max)

                # ── grey "max range" background bar
                bullet_fig.add_shape(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=0,
                    x1=x_max,
                    y0=y0 + pad,
                    y1=y1 - pad,
                    fillcolor="rgba(220,220,220,0.45)",
                    line_width=0,
                    layer="below",
                )
                # ── green "normal range" band
                bullet_fig.add_shape(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=lo,
                    x1=hi,
                    y0=y0 + pad,
                    y1=y1 - pad,
                    fillcolor="rgba(42,157,143,0.22)",
                    line_width=0,
                    layer="below",
                )
                # ── row label (left of chart)
                bullet_fig.add_annotation(
                    xref="paper",
                    yref="paper",
                    x=0,
                    y=cy,
                    xanchor="right",
                    text=label,
                    showarrow=False,
                    font=dict(size=11, color="#333"),
                )
                # ── value bar (solid)
                bar_col = (
                    "#2a9d8f"
                    if lo <= val <= hi
                    else ("#e63946" if val < lo else "#f4a261")
                )
                bullet_fig.add_shape(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=0,
                    x1=val_clipped,
                    y0=y0 + pad * 2.2,
                    y1=y1 - pad * 2.2,
                    fillcolor=bar_col,
                    line_width=0,
                )
                # ── value label (right end of bar)
                disp = f"{val:.1f} {unit}".strip()
                bullet_fig.add_annotation(
                    xref="x",
                    yref="paper",
                    x=val_clipped,
                    y=cy,
                    xanchor="left",
                    text=f"  {disp}",
                    showarrow=False,
                    font=dict(size=10, color=bar_col),
                )

            # Invisible scatter to give the figure a proper x-axis scale
            bullet_fig.add_trace(
                go.Scatter(
                    x=[0, max(s[5] for s in bullet_specs)],
                    y=[0.5, 0.5],
                    mode="markers",
                    marker=dict(opacity=0),
                    showlegend=False,
                )
            )
            # Phantom traces just for legend
            bullet_fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(
                        color="rgba(220,220,220,0.8)", size=12, symbol="square"
                    ),
                    name="Full range",
                )
            )
            bullet_fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(color="rgba(42,157,143,0.5)", size=12, symbol="square"),
                    name="Normal range",
                )
            )
            bullet_fig.update_layout(
                title="HRV Metrics vs Normal Ranges",
                xaxis=dict(title="Value", zeroline=True, zerolinecolor="#ccc"),
                yaxis=dict(visible=False, range=[0, 1]),
                height=max(_CHART_H, 50 * n_rows + 60),
                template=_TPL,
                margin=dict(l=80, r=40, t=44, b=36),
                legend=dict(
                    orientation="h",
                    y=-0.12,
                    x=0.5,
                    xanchor="center",
                    font=dict(size=10),
                ),
            )
            charts.append(
                dbc.Col(
                    dcc.Graph(figure=bullet_fig, config={"displayModeBar": False}), md=6
                )
            )

        # arrange in pairs of two columns
        chart_rows = []
        for i in range(0, len(charts), 2):
            row_cols = charts[i : i + 2]
            chart_rows.append(dbc.Row(row_cols, className="g-3 mb-3"))

        if chart_rows:
            charts_section = html.Div(
                [
                    dbc.Button(
                        [html.I(className="fas fa-chart-bar me-1"), " Analysis Plots"],
                        id="btn-collapse-charts",
                        color="outline-secondary",
                        size="sm",
                        className="mb-2",
                    ),
                    dbc.Collapse(
                        html.Div(chart_rows), id="collapse-charts", is_open=False
                    ),
                ]
            )
        else:
            charts_section = html.Div()

        # No data notice when not enough peaks
        if n_peaks < 2:
            feature_section = dbc.Alert(
                [
                    html.I(className="fas fa-info-circle me-2"),
                    f"Only {n_peaks} peak(s) detected — need ≥ 2 for HRV metrics. "
                    "Try a longer analysis window.",
                ],
                color="warning",
                className="mt-2",
            )
        else:
            # Collapsible feature table
            feature_section = html.Div(
                [
                    dbc.Button(
                        [html.I(className="fas fa-table me-1"), " Full Feature Table"],
                        id="btn-collapse-features",
                        color="outline-secondary",
                        className="mb-2",
                        size="sm",
                    ),
                    dbc.Collapse(feature_table, id="collapse-features", is_open=False),
                ]
            )

        return html.Div(
            [
                metric_cards,
                html.Hr(className="my-3"),
                feature_section,
                charts_section,
            ]
        )

    except Exception as exc:
        logger.exception("create_peak_analysis_table failed: %s", exc)
        return html.Div(
            [
                html.H6("Peak detection + HRV"),
                html.P(f"Could not compute: {exc}", className="small text-danger"),
            ]
        )


def create_signal_quality_table(
    selected_signal,
    time_axis,
    sampling_freq,
    analysis_options,
    signal_source_info,
    signal_type=None,
):
    """Kept for callback compat — hidden in layout."""
    return html.Div()


# ─────────────────────────────────────────────────────────────
# Callback registration
# ─────────────────────────────────────────────────────────────


def register_time_domain_callbacks(app):
    """Register all time domain analysis callbacks."""

    # ── collapse toggles ───────────────────────────────────────
    @app.callback(
        Output("collapse-summary", "is_open"),
        Input("btn-collapse-summary", "n_clicks"),
        State("collapse-summary", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_summary(n, is_open):
        return not is_open

    @app.callback(
        Output("collapse-features", "is_open"),
        Input("btn-collapse-features", "n_clicks"),
        State("collapse-features", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_features(n, is_open):
        return not is_open

    # ── main analysis ──────────────────────────────────────────
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
            Input("btn-nudge-m10", "n_clicks"),
            Input("btn-nudge-m5", "n_clicks"),
            Input("btn-nudge-p5", "n_clicks"),
            Input("btn-nudge-p10", "n_clicks"),
            Input("body", "data-theme"),
        ],
        [
            State("url", "pathname"),
            State("start-position-slider", "value"),
            State("duration-select", "value"),
            State("signal-source-select", "value"),
            State("analysis-options", "value"),
            State("signal-type-select", "value"),
            State("store-filtered-signal", "data"),
        ],
    )
    def analyze_time_domain(
        n_clicks,
        nudge_m10,
        nudge_m5,
        nudge_p5,
        nudge_p10,
        current_theme,
        pathname,
        start_position,
        duration,
        signal_source,
        analysis_options,
        signal_type,
        filtered_signal_data,
    ):
        logger.info("=== TIME DOMAIN ANALYSIS CALLBACK ===")

        ctx = callback_context
        trigger_id = (
            ctx.triggered[0]["prop_id"].split(".")[0]
            if ctx.triggered
            else "initial_load"
        )

        def _empty(msg):
            return (
                create_empty_figure(current_theme),
                create_empty_figure(current_theme),
                msg,
                msg,
                html.Div(),
                html.Div(),
                html.Div(),
                None,
                None,
            )

        if pathname != "/time-domain":
            return _empty("Navigate to Time Domain Analysis page")

        try:
            from vitalDSP_webapp.services.data.enhanced_data_service import (
                get_enhanced_data_service,
            )

            data_service = get_enhanced_data_service()
            all_data = data_service.get_all_data()

            if not all_data:
                return _empty(
                    "No data available. Please upload and process data first."
                )

            latest_data_id = list(all_data.keys())[-1]
            latest_data = all_data[latest_data_id]
            column_mapping = data_service.get_column_mapping(latest_data_id)

            if not column_mapping:
                return _empty(
                    "Please process your data on the Upload page first (configure column mapping)."
                )

            df = data_service.get_data(latest_data_id)
            if df is None or df.empty:
                return _empty("Data is empty or corrupted.")

            sampling_freq = latest_data.get("info", {}).get("sampling_freq", 1000)

            # ── nudge handling ─────────────────────────────────
            if start_position is None or duration is None:
                start_position, duration = 0, 60
            start_position = float(start_position)
            duration = float(duration) if duration else 60.0

            nudge_map = {
                "btn-nudge-m10": -10,
                "btn-nudge-m5": -5,
                "btn-nudge-p5": 5,
                "btn-nudge-p10": 10,
            }
            if trigger_id in nudge_map:
                start_position = max(
                    0.0, min(100.0, start_position + nudge_map[trigger_id])
                )

            # ── time window ────────────────────────────────────
            data_duration = len(df) / sampling_freq
            start_time_actual = (start_position / 100.0) * data_duration
            end_time = start_time_actual + duration
            start_sample = int(start_time_actual * sampling_freq)
            end_sample = min(int(end_time * sampling_freq), len(df))
            if start_sample >= len(df):
                start_sample = 0
                start_time_actual = 0.0
                end_sample = min(int(duration * sampling_freq), len(df))

            windowed_data = df.iloc[start_sample:end_sample].copy()
            time_axis = np.linspace(
                start_time_actual,
                start_time_actual + len(windowed_data) / sampling_freq,
                len(windowed_data),
            )

            # ── signal column ──────────────────────────────────
            signal_column = column_mapping.get("signal")
            if not signal_column or signal_column not in windowed_data.columns:
                for col in [
                    "waveform",
                    "pleth",
                    "pl",
                    "signal",
                    "ppg",
                    "ecg",
                    "red",
                    "ir",
                ]:
                    match = [c for c in windowed_data.columns if c.lower() == col]
                    if match:
                        signal_column = match[0]
                        break
            if not signal_column:
                return _empty("No suitable signal column found in data.")

            signal_data = windowed_data[signal_column].values
            if len(signal_data) == 0:
                return _empty("Signal data is empty after windowing.")

            # ── signal source (always filtered from chain) ─────
            selected_signal = signal_data
            signal_source_info = "Original Signal"
            filter_info = None

            # Always attempt to use the filtered signal from the filtering page
            filtered_full = None
            stored_filter_info = None

            # 1. Try data service
            svc_filtered = data_service.get_filtered_data(latest_data_id)
            svc_filter_info = data_service.get_filter_info(latest_data_id)
            if svc_filtered is not None:
                filtered_full = svc_filtered
                stored_filter_info = svc_filter_info

            # 2. Fall back to filtering-page store
            if (
                filtered_full is None
                and isinstance(filtered_signal_data, dict)
                and "signal" in filtered_signal_data
            ):
                try:
                    filtered_full = np.array(filtered_signal_data["signal"])
                    if "filter_params" in filtered_signal_data:
                        stored_filter_info = {
                            "filter_type": filtered_signal_data.get(
                                "filter_type", "unknown"
                            ),
                            "parameters": filtered_signal_data.get("filter_params", {}),
                        }
                except Exception as e:
                    logger.warning("Could not load filtered signal from store: %s", e)

            if filtered_full is not None:
                if len(filtered_full) >= end_sample:
                    selected_signal = filtered_full[start_sample:end_sample]
                    signal_source_info = "Filtered Signal"
                    filter_info = stored_filter_info
                elif len(filtered_full) == len(windowed_data):
                    selected_signal = filtered_full
                    signal_source_info = "Filtered Signal"
                    filter_info = stored_filter_info
                else:
                    # Re-apply filter from saved params
                    try:
                        saved_chain = (stored_filter_info or {}).get("chain") or []
                        if saved_chain:
                            from vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks import (
                                apply_filter_chain,
                            )

                            selected_signal = apply_filter_chain(
                                signal_data,
                                sampling_freq,
                                signal_type,
                                saved_chain,
                                logger=logger,
                            )
                            signal_source_info = (
                                f"Filtered Signal (chain x{len(saved_chain)})"
                            )
                            filter_info = stored_filter_info
                        elif (
                            stored_filter_info
                            and stored_filter_info.get("filter_type") == "traditional"
                        ):
                            p = stored_filter_info.get("parameters", {})
                            selected_signal = apply_traditional_filter(
                                signal_data,
                                sampling_freq,
                                p.get("filter_family", "butter"),
                                p.get("filter_response", "bandpass"),
                                p.get("low_freq", 0.5),
                                p.get("high_freq", 5),
                                p.get("filter_order", 4),
                            )
                            signal_source_info = "Filtered Signal (re-applied)"
                            filter_info = stored_filter_info
                    except Exception as e:
                        logger.warning("Re-applying filter failed: %s", e)

            # ── shared WaveformMorphology ───────────────────────
            stype = signal_type or "PPG"
            shared_wm = None
            shared_peaks = None
            try:
                from vitalDSP.physiological_features.waveform import WaveformMorphology

                shared_wm = WaveformMorphology(
                    waveform=selected_signal,
                    fs=sampling_freq,
                    signal_type=stype,
                    simple_mode=True,
                )
                attr = "systolic_peaks" if stype == "PPG" else "r_peaks"
                shared_peaks = getattr(shared_wm, attr, None)
                logger.info(
                    "Shared WaveformMorphology: %d peaks",
                    0 if shared_peaks is None else len(shared_peaks),
                )
            except Exception as exc:
                logger.warning("Shared WaveformMorphology failed: %s", exc)

            # ── main signal plot ───────────────────────────────
            if signal_source_info.startswith("Filtered"):
                # overlay: show original as underlay
                main_signal = signal_data
                filtered_overlay = selected_signal
            else:
                main_signal = selected_signal
                filtered_overlay = None

            main_plot = create_main_signal_plot(
                signal_data=main_signal,
                time_axis=time_axis,
                sampling_freq=sampling_freq,
                signal_type=stype,
                theme=current_theme,
                filtered_overlay=filtered_overlay,
                wm=shared_wm,
            )

            comparison_plot = create_empty_figure(current_theme)

            analysis_results = generate_analysis_results(
                selected_signal,
                time_axis,
                sampling_freq,
                analysis_options or [],
                signal_source_info,
                stype,
                filter_info,
            )

            peak_table = create_peak_analysis_table(
                selected_signal,
                time_axis,
                sampling_freq,
                analysis_options or [],
                signal_source_info,
                stype,
                peaks=shared_peaks,
            )

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
                html.Div(),
                html.Div(),
                html.Div(),
                time_domain_data,
                time_domain_data,
            )

        except Exception as e:
            logger.error("Error in time domain analysis callback: %s", e)
            import traceback

            traceback.print_exc()
            return _empty(f"Error in analysis: {str(e)}")

    # ── collapse toggles for charts panel ─────────────────────
    @app.callback(
        Output("collapse-charts", "is_open"),
        Input("btn-collapse-charts", "n_clicks"),
        State("collapse-charts", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_charts(n, is_open):
        return not is_open

    # ── slider range update ────────────────────────────────────
    @app.callback(
        [
            Output("start-position-slider", "min"),
            Output("start-position-slider", "max"),
            Output("start-position-slider", "value"),
        ],
        [Input("url", "pathname")],
        prevent_initial_call=True,
    )
    def update_time_slider_range(pathname):
        if pathname != "/time-domain":
            return no_update, no_update, no_update
        return 0, 100, 0
