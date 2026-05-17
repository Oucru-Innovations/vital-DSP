"""
Respiratory rate analysis callbacks for vitalDSP webapp.

Runs all 6 RR extraction methods via RespiratoryAnalysis and produces
per-method insight plots plus an ensemble summary.
"""

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, html
from dash.exceptions import PreventUpdate
from scipy import signal as scipy_signal
import dash_bootstrap_components as dbc
import logging

logger = logging.getLogger(__name__)

# ── vitalDSP imports ──────────────────────────────────────────────────────────
RespiratoryAnalysis = None
PreprocessConfig = None


def _import_vitaldsp_modules():
    global RespiratoryAnalysis, PreprocessConfig
    try:
        from vitalDSP.respiratory_analysis.respiratory_analysis import (
            RespiratoryAnalysis,
        )

        logger.info("RespiratoryAnalysis imported")
    except Exception as e:
        logger.warning(f"RespiratoryAnalysis unavailable: {e}")
    try:
        from vitalDSP.preprocess.preprocess_operations import PreprocessConfig

        logger.info("PreprocessConfig imported")
    except Exception as e:
        logger.warning(f"PreprocessConfig unavailable: {e}")


# ── signal loading ────────────────────────────────────────────────────────────


def _load_signal(start_pos, duration, filtered_signal_data):
    """
    Returns (time_axis, resp_signal, raw_signal, sampling_freq) for the requested window.
    resp_signal = bandpass-filtered 0.1–0.8 Hz version for RR extraction.
    raw_signal  = original (or pre-filtered from store) for display context.
    """
    from vitalDSP_webapp.services.data.enhanced_data_service import (
        get_enhanced_data_service,
    )

    ds = get_enhanced_data_service()
    all_data = ds.get_all_data()
    if not all_data:
        raise ValueError("No data available. Please upload data first.")

    lid = list(all_data.keys())[-1]
    info = all_data[lid].get("info", {})
    sf = float(info.get("sampling_freq", 1000))

    col_map = ds.get_column_mapping(lid)
    if not col_map:
        raise ValueError("Data not configured. Set column mapping on the Upload page.")

    df = ds.get_data(lid)
    if df is None or df.empty:
        raise ValueError("Data is empty.")

    # Total duration
    tc = col_map.get("time")
    if tc and tc in df.columns:
        td = df[tc].iloc[-1] - df[tc].iloc[0]
        total = td.total_seconds() if hasattr(td, "total_seconds") else float(td)
    else:
        total = len(df) / sf

    sp = float(start_pos or 0)
    dur = float(duration or 60)
    t0 = (sp / 100.0) * total
    t1 = min(t0 + dur, total)
    t0 = max(0.0, t1 - dur)

    s0, s1 = int(t0 * sf), int(t1 * sf)

    sig_col = col_map.get("signal")
    if not sig_col or sig_col not in df.columns:
        raise ValueError(f"Signal column '{sig_col}' not found.")

    raw = df[sig_col].values[s0:s1].astype(float)

    # Use filtered signal from store if available
    if (
        filtered_signal_data
        and isinstance(filtered_signal_data, dict)
        and "signal" in filtered_signal_data
    ):
        full = np.array(filtered_signal_data["signal"], dtype=float)
        if len(full) >= s1:
            raw = full[s0:s1]

    # Bandpass 0.1–0.8 Hz for RR extraction
    nyq = sf / 2.0
    lo = max(0.1 / nyq, 1e-4)
    hi = min(0.8 / nyq, 0.999)
    try:
        b, a = scipy_signal.butter(4, [lo, hi], btype="band")
        resp = scipy_signal.filtfilt(b, a, raw)
    except Exception:
        resp = raw.copy()

    time_axis = np.arange(len(resp)) / sf
    return time_axis, resp, raw, sf


# ── figure helpers ────────────────────────────────────────────────────────────

MARGIN = dict(l=45, r=15, t=15, b=35)
THEME = "plotly_white"
H = 180

COLORS = {
    "counting": "#2E86AB",
    "fft_based": "#457B9D",
    "freq_domain": "#1D3557",
    "time_domain": "#6A4C93",
    "peaks": "#E63946",
    "zero_crossing": "#2A9D8F",
    "signal": "#2E86AB",
    "marker": "#E63946",
    "vline": "#E63946",
}


def _empty_fig(msg="No data — upload data and click Run Analysis"):
    fig = go.Figure()
    fig.add_annotation(
        text=msg,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=13, color="gray"),
    )
    fig.update_layout(
        template=THEME,
        height=H,
        margin=MARGIN,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


def _rr_label(rr):
    return f"{rr:.1f} bpm" if rr is not None else "—"


def _shade_breath_cycles(fig, time_axis, peaks, resp, color="rgba(46,134,171,0.07)"):
    """Shade every other inter-peak region to visualise breath cycles."""
    for i in range(0, len(peaks) - 1, 2):
        fig.add_vrect(
            x0=time_axis[peaks[i]],
            x1=time_axis[peaks[i + 1]],
            fillcolor=color,
            layer="below",
            line_width=0,
        )


def _plot_counting(time_axis, resp, sf, min_bd):
    """Peak detection on smoothed signal + breath-cycle shading + interval annotations."""
    resp_s = _smooth_for_peak_methods(resp, sf)
    min_dist = max(1, int(1.25 * sf))
    prom = 0.3 * np.std(resp_s)
    peaks, _ = scipy_signal.find_peaks(resp_s, prominence=prom, distance=min_dist)

    fig = go.Figure()
    # Raw bandpassed signal in background (faint)
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=resp,
            mode="lines",
            line=dict(color=COLORS["counting"], width=1, dash="dot"),
            opacity=0.35,
            name="Bandpass",
            hovertemplate="t=%{x:.2f}s | raw=%{y:.3f}<extra></extra>",
        )
    )
    # Smoothed signal used for detection
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=resp_s,
            mode="lines",
            line=dict(color=COLORS["counting"], width=2),
            name="Smoothed",
            hovertemplate="t=%{x:.2f}s | smooth=%{y:.3f}<extra></extra>",
        )
    )
    if len(peaks) >= 2:
        _shade_breath_cycles(fig, time_axis, peaks, resp_s)
        fig.add_trace(
            go.Scatter(
                x=time_axis[peaks],
                y=resp_s[peaks],
                mode="markers",
                marker=dict(color=COLORS["marker"], size=8, symbol="diamond"),
                name="Peaks",
                showlegend=False,
                hovertemplate="peak @ t=%{x:.2f}s<extra></extra>",
            )
        )
        # Annotate inter-peak intervals
        ivs = np.diff(peaks) / sf
        for i, iv in enumerate(ivs):
            mid = (time_axis[peaks[i]] + time_axis[peaks[i + 1]]) / 2
            ypos = (resp_s[peaks[i]] + resp_s[peaks[i + 1]]) / 2
            fig.add_annotation(
                x=mid,
                y=ypos,
                text=f"{iv:.1f}s",
                showarrow=False,
                font=dict(size=9, color="#555"),
                bgcolor="rgba(255,255,255,0.6)",
                borderwidth=0,
            )
    elif len(peaks) > 0:
        fig.add_trace(
            go.Scatter(
                x=time_axis[peaks],
                y=resp_s[peaks],
                mode="markers",
                marker=dict(color=COLORS["marker"], size=8, symbol="diamond"),
                showlegend=False,
            )
        )

    fig.update_layout(
        template=THEME,
        height=H,
        margin=MARGIN,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        showlegend=False,
    )
    return fig


def _plot_fft_based(time_axis, resp, sf):
    """FFT spectrum in respiratory band — normal range band + harmonic markers."""
    N = len(resp)
    fft_mag = np.abs(np.fft.rfft(resp))
    freqs = np.fft.rfftfreq(N, 1.0 / sf)
    mask = (freqs >= 0.1) & (freqs <= 0.8)

    fig = go.Figure()
    if np.any(mask):
        x_bpm = freqs[mask] * 60
        y_mag = fft_mag[mask]
        # Normal adult range: 12–20 bpm
        fig.add_vrect(
            x0=12,
            x1=20,
            fillcolor="rgba(46,200,100,0.10)",
            layer="below",
            line_width=0,
            annotation_text="normal range",
            annotation_position="top left",
            annotation_font=dict(size=9, color="green"),
        )
        fig.add_trace(
            go.Scatter(
                x=x_bpm,
                y=y_mag,
                mode="lines",
                line=dict(color=COLORS["fft_based"], width=2),
                fill="tozeroy",
                fillcolor="rgba(69,123,157,0.15)",
                hovertemplate="RR=%{x:.1f} bpm | mag=%{y:.2f}<extra></extra>",
            )
        )
        peak_rr = x_bpm[np.argmax(y_mag)]
        fig.add_vline(
            x=peak_rr,
            line_color=COLORS["vline"],
            line_width=2,
            line_dash="dash",
            annotation_text=f"<b>{peak_rr:.1f} bpm</b>",
            annotation_position="top right",
            annotation_font=dict(color=COLORS["vline"]),
        )
        # Mark 2nd harmonic if in range
        harmonic = peak_rr * 2
        if harmonic <= 48:
            fig.add_vline(
                x=harmonic,
                line_color="orange",
                line_width=1,
                line_dash="dot",
                annotation_text=f"2×harmonic",
                annotation_position="top left",
                annotation_font=dict(size=9, color="orange"),
            )
    fig.update_layout(
        template=THEME,
        height=H,
        margin=MARGIN,
        xaxis_title="Rate (bpm)",
        yaxis_title="FFT Magnitude",
        showlegend=False,
    )
    return fig


def _plot_freq_domain(time_axis, resp, sf):
    """Welch PSD — normal range band + peak marker + bandwidth indicator."""
    try:
        f, psd = scipy_signal.welch(resp, fs=sf, nperseg=min(512, len(resp) // 4))
    except Exception:
        return _empty_fig("PSD computation failed")
    mask = (f >= 0.1) & (f <= 0.8)

    fig = go.Figure()
    if np.any(mask):
        x_bpm = f[mask] * 60
        y_psd = psd[mask]
        fig.add_vrect(
            x0=12,
            x1=20,
            fillcolor="rgba(46,200,100,0.10)",
            layer="below",
            line_width=0,
            annotation_text="normal range",
            annotation_position="top left",
            annotation_font=dict(size=9, color="green"),
        )
        fig.add_trace(
            go.Scatter(
                x=x_bpm,
                y=y_psd,
                mode="lines",
                line=dict(color=COLORS["freq_domain"], width=2),
                fill="tozeroy",
                fillcolor="rgba(29,53,87,0.12)",
                hovertemplate="RR=%{x:.1f} bpm | PSD=%{y:.3g}<extra></extra>",
            )
        )
        peak_rr = x_bpm[np.argmax(y_psd)]
        fig.add_vline(
            x=peak_rr,
            line_color=COLORS["vline"],
            line_width=2,
            line_dash="dash",
            annotation_text=f"<b>{peak_rr:.1f} bpm</b>",
            annotation_position="top right",
            annotation_font=dict(color=COLORS["vline"]),
        )
        # Half-power bandwidth: where PSD > max/2
        half_max = np.max(y_psd) / 2
        above = x_bpm[y_psd >= half_max]
        if len(above) >= 2:
            fig.add_vrect(
                x0=above[0],
                x1=above[-1],
                fillcolor="rgba(230,57,70,0.08)",
                layer="below",
                line_width=0,
            )
    fig.update_layout(
        template=THEME,
        height=H,
        margin=MARGIN,
        xaxis_title="Rate (bpm)",
        yaxis_title="Welch PSD",
        showlegend=False,
    )
    return fig


def _plot_time_domain(time_axis, resp, sf):
    """Autocorrelation — respiratory lag range shaded + breath-period markers."""
    centered = resp - np.mean(resp)
    corr = np.correlate(centered, centered, mode="full")
    corr = corr[len(corr) // 2 :]
    corr /= corr[0] + 1e-12

    lag_times = np.arange(len(corr)) / sf
    mask = (lag_times >= 1.0) & (lag_times <= 10.0)

    fig = go.Figure()
    # Shade the valid respiratory lag range
    fig.add_vrect(
        x0=1.25,
        x1=10.0,
        fillcolor="rgba(106,76,147,0.07)",
        layer="below",
        line_width=0,
        annotation_text="valid breath period",
        annotation_position="top left",
        annotation_font=dict(size=9, color="#6A4C93"),
    )
    fig.add_hline(
        y=0, line_color="rgba(150,150,150,0.5)", line_dash="dot", line_width=1
    )

    if np.any(mask):
        fig.add_trace(
            go.Scatter(
                x=lag_times[mask],
                y=corr[mask],
                mode="lines",
                line=dict(color=COLORS["time_domain"], width=2),
                hovertemplate="lag=%{x:.2f}s | r=%{y:.3f}<extra></extra>",
            )
        )
        peaks_c, props = scipy_signal.find_peaks(corr[mask], prominence=0.05)
        if len(peaks_c) > 0:
            # Mark all peaks, highlight the first (dominant breath period)
            lags_masked = lag_times[mask]
            best_lag = lags_masked[peaks_c[0]]
            best_rr = 60 / best_lag
            fig.add_vline(
                x=best_lag,
                line_color=COLORS["vline"],
                line_width=2,
                line_dash="dash",
                annotation_text=f"<b>{best_rr:.1f} bpm</b> ({best_lag:.2f}s)",
                annotation_position="top right",
                annotation_font=dict(color=COLORS["vline"]),
            )
            # Secondary peaks (possible harmonics)
            for pk in peaks_c[1:3]:
                lag_h = lags_masked[pk]
                fig.add_vline(
                    x=lag_h,
                    line_color="orange",
                    line_width=1,
                    line_dash="dot",
                    annotation_text=f"{60/lag_h:.1f}",
                    annotation_position="top left",
                    annotation_font=dict(size=9, color="orange"),
                )
    fig.update_layout(
        template=THEME,
        height=H,
        margin=MARGIN,
        xaxis_title="Lag (s)",
        yaxis_title="Autocorrelation",
        showlegend=False,
    )
    return fig


def _plot_peaks(time_axis, resp, sf, min_bd, max_bd):
    """Peak interval method — smoothed signal, peaks, cycle shading, interval labels."""
    BAND_MIN = 1.25
    BAND_MAX = 10.0
    resp_s = _smooth_for_peak_methods(resp, sf)
    min_dist = max(1, int(BAND_MIN * sf))
    peaks, _ = scipy_signal.find_peaks(
        resp_s, distance=min_dist, prominence=0.3 * np.std(resp_s)
    )
    intervals = np.diff(peaks) / sf if len(peaks) > 1 else np.array([])
    valid_mask = (
        (intervals >= BAND_MIN) & (intervals <= BAND_MAX)
        if len(intervals)
        else np.array([], dtype=bool)
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=resp,
            mode="lines",
            line=dict(color=COLORS["peaks"], width=1, dash="dot"),
            opacity=0.35,
            name="Bandpass",
            hovertemplate="t=%{x:.2f}s | raw=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=resp_s,
            mode="lines",
            line=dict(color=COLORS["peaks"], width=2),
            name="Smoothed",
            hovertemplate="t=%{x:.2f}s | smooth=%{y:.3f}<extra></extra>",
        )
    )

    if len(peaks) >= 2:
        # Shade valid cycles (green) and invalid cycles (red tint)
        for i in range(len(peaks) - 1):
            iv = intervals[i]
            fill = "rgba(42,157,143,0.10)" if valid_mask[i] else "rgba(230,57,70,0.08)"
            fig.add_vrect(
                x0=time_axis[peaks[i]],
                x1=time_axis[peaks[i + 1]],
                fillcolor=fill,
                layer="below",
                line_width=0,
            )
        fig.add_trace(
            go.Scatter(
                x=time_axis[peaks],
                y=resp_s[peaks],
                mode="markers",
                marker=dict(color=COLORS["marker"], size=8, symbol="triangle-up"),
                showlegend=False,
                hovertemplate="peak @ t=%{x:.2f}s<extra></extra>",
            )
        )
        # Interval labels
        for i, iv in enumerate(intervals):
            mid = (time_axis[peaks[i]] + time_axis[peaks[i + 1]]) / 2
            col = "#2A9D8F" if valid_mask[i] else "#E63946"
            fig.add_annotation(
                x=mid,
                y=np.min(resp_s),
                text=f"{iv:.1f}s",
                showarrow=False,
                font=dict(size=9, color=col),
                bgcolor="rgba(255,255,255,0.7)",
                borderwidth=0,
                yanchor="bottom",
            )

    fig.update_layout(
        template=THEME,
        height=H,
        margin=MARGIN,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        showlegend=False,
    )
    return fig


def _plot_zero_crossing(time_axis, resp, sf, min_bd, max_bd):
    """Zero-crossing — smoothed signal, positive crossings, breath-period shading."""
    BAND_MIN = 1.25
    BAND_MAX = 10.0
    resp_s = _smooth_for_peak_methods(resp, sf)
    centered = resp_s - np.mean(resp_s)
    signs = np.sign(centered)
    pos_cross = np.where(np.diff(signs) > 0)[0]  # positive-going crossings
    neg_cross = np.where(np.diff(signs) < 0)[0]  # negative-going (for context)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=resp - np.mean(resp),
            mode="lines",
            line=dict(color=COLORS["zero_crossing"], width=1, dash="dot"),
            opacity=0.35,
            name="Bandpass",
            hovertemplate="t=%{x:.2f}s | raw=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=centered,
            mode="lines",
            line=dict(color=COLORS["zero_crossing"], width=2),
            name="Smoothed",
            hovertemplate="t=%{x:.2f}s | amp=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=0, line_color="rgba(100,100,100,0.5)", line_dash="dot", line_width=1
    )

    # Shade inspiration (above zero) and expiration (below zero) phases
    y_max = np.max(np.abs(centered)) * 1.05
    all_cross = np.sort(np.concatenate([pos_cross, neg_cross]))
    for i in range(len(all_cross) - 1):
        t0, t1 = time_axis[all_cross[i]], time_axis[all_cross[i + 1]]
        mid_idx = (all_cross[i] + all_cross[i + 1]) // 2
        is_pos = centered[mid_idx] > 0
        fill = "rgba(46,134,171,0.08)" if is_pos else "rgba(230,57,70,0.06)"
        fig.add_vrect(x0=t0, x1=t1, fillcolor=fill, layer="below", line_width=0)

    # Mark positive crossings (= breath cycle starts)
    if len(pos_cross) > 0:
        fig.add_trace(
            go.Scatter(
                x=time_axis[pos_cross],
                y=np.zeros(len(pos_cross)),
                mode="markers",
                marker=dict(color=COLORS["marker"], size=7, symbol="circle"),
                name="Breath start",
                showlegend=False,
                hovertemplate="breath @ t=%{x:.2f}s<extra></extra>",
            )
        )
        # Annotate valid inter-crossing intervals
        if len(pos_cross) > 1:
            ivs = np.diff(pos_cross) / sf
            for i, iv in enumerate(ivs):
                if BAND_MIN <= iv <= BAND_MAX:
                    mid = (time_axis[pos_cross[i]] + time_axis[pos_cross[i + 1]]) / 2
                    fig.add_annotation(
                        x=mid,
                        y=y_max * 0.85,
                        text=f"{iv:.1f}s",
                        showarrow=False,
                        font=dict(size=9, color="#555"),
                        bgcolor="rgba(255,255,255,0.6)",
                        borderwidth=0,
                    )

    fig.update_layout(
        template=THEME,
        height=H,
        margin=MARGIN,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (centered)",
        showlegend=False,
    )
    return fig


def _main_plot(time_axis, resp):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=resp,
            mode="lines",
            line=dict(color=COLORS["signal"], width=1.5),
            hovertemplate="t=%{x:.2f}s | amp=%{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        template=THEME,
        height=200,
        margin=dict(l=45, r=15, t=10, b=35),
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        showlegend=False,
    )
    return fig


def _smooth_for_peak_methods(resp, sf):
    """
    Apply a light moving-average to remove residual cardiac / AM ripple before
    peak/zero-crossing counting.  Window = 1 respiratory cycle minimum (1.25 s).
    """
    win = max(3, int(1.25 * sf) | 1)  # odd number, >= 1.25 s
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win) / win
    return np.convolve(resp, kernel, mode="same")


def _run_all_methods(resp, sf, min_bd, max_bd):
    """
    Run all 6 RR methods. Returns (results_dict, ensemble_dict).

    Strategy:
    - fft_based / frequency_domain / time_domain / counting:
        use vitalDSP ensemble (all pass preprocess_config=no_preprocess).
    - peaks / zero_crossing:
        called individually via compute_respiratory_rate() so we can pass
        correct min/max breath durations (1.25–10 s from the band limits).
        These methods are sensitive to the duration bounds, which the ensemble
        path hard-codes at 0.5–6 s (too restrictive for PPG-derived resp).
    - All time-domain peak/ZC methods receive a lightly smoothed signal to
        suppress residual cardiac AM ripple that causes false peak detections.
    """
    BAND_MIN_BD = 1.0 / 0.8  # 1.25 s  (48 bpm max)
    BAND_MAX_BD = 1.0 / 0.1  # 10.0 s  (6 bpm min)

    ENSEMBLE_METHODS = ["counting", "fft_based", "frequency_domain", "time_domain"]
    ALL_METHODS = [
        "counting",
        "fft_based",
        "frequency_domain",
        "time_domain",
        "peaks",
        "zero_crossing",
    ]
    results = {}

    # Smoothed version for time-domain peak/ZC methods
    resp_smooth = _smooth_for_peak_methods(resp, sf)

    if RespiratoryAnalysis is not None:
        try:
            if PreprocessConfig is not None:
                no_preprocess = PreprocessConfig(
                    filter_type="ignore",
                    noise_reduction_method="ignore",
                )
            else:
                no_preprocess = None

            # ── Ensemble for the 4 frequency-robust methods ──────────────────
            ra = RespiratoryAnalysis(resp_smooth, fs=sf)
            ensemble = ra.compute_respiratory_rate_ensemble(
                methods=ENSEMBLE_METHODS,
                preprocess_config=no_preprocess,
            )
            results = {
                k: v for k, v in ensemble.get("individual_estimates", {}).items()
            }

            # ── peaks and zero_crossing with corrected duration bounds ────────
            for method in ("peaks", "zero_crossing"):
                try:
                    rr = ra.compute_respiratory_rate(
                        method=method,
                        min_breath_duration=BAND_MIN_BD,
                        max_breath_duration=BAND_MAX_BD,
                        preprocess_config=no_preprocess,
                    )
                    results[method] = float(rr) if rr and 6 <= rr <= 40 else None
                except Exception as e:
                    logger.warning(f"{method} direct call failed: {e}")
                    results[method] = None

            # Rebuild ensemble dict with all 6 methods
            valid_vals = [v for v in results.values() if v is not None]
            ensemble["individual_estimates"] = results
            if valid_vals:
                ensemble["respiratory_rate"] = float(np.median(valid_vals))
                ensemble["std"] = (
                    float(np.std(valid_vals)) if len(valid_vals) > 1 else 0.0
                )
                ensemble["n_methods"] = len(valid_vals)

            logger.info(f"All-method results: {results}")
            return results, ensemble

        except Exception as e:
            logger.warning(f"RespiratoryAnalysis failed: {e}, falling back to scipy")

    # ── Scipy fallback ────────────────────────────────────────────────────────
    for method in ALL_METHODS:
        try:
            sig = (
                resp_smooth
                if method in ("counting", "peaks", "zero_crossing")
                else resp
            )

            if method in ("counting", "peaks"):
                min_dist = max(1, int(BAND_MIN_BD * sf))
                pks, _ = scipy_signal.find_peaks(
                    sig, distance=min_dist, prominence=0.3 * np.std(sig)
                )
                if len(pks) > 1:
                    ivs = np.diff(pks) / sf
                    valid = ivs[(ivs >= BAND_MIN_BD) & (ivs <= BAND_MAX_BD)]
                    results[method] = (
                        float(60.0 / np.median(valid)) if len(valid) else None
                    )
                else:
                    results[method] = None

            elif method == "fft_based":
                N = len(resp)
                fft_mag = np.abs(np.fft.rfft(resp))
                freqs = np.fft.rfftfreq(N, 1.0 / sf)
                mask = (freqs >= 0.1) & (freqs <= 0.8)
                results[method] = (
                    float(freqs[mask][np.argmax(fft_mag[mask])] * 60)
                    if np.any(mask)
                    else None
                )

            elif method == "frequency_domain":
                f, psd = scipy_signal.welch(
                    resp, fs=sf, nperseg=min(512, len(resp) // 4)
                )
                mask = (f >= 0.1) & (f <= 0.8)
                results[method] = (
                    float(f[mask][np.argmax(psd[mask])] * 60) if np.any(mask) else None
                )

            elif method == "time_domain":
                centered = resp - np.mean(resp)
                corr = np.correlate(centered, centered, mode="full")
                corr = corr[len(corr) // 2 :]
                corr /= corr[0] + 1e-12
                lag_times = np.arange(len(corr)) / sf
                mask = (lag_times >= BAND_MIN_BD) & (lag_times <= BAND_MAX_BD)
                if np.any(mask):
                    pks, _ = scipy_signal.find_peaks(corr[mask], prominence=0.05)
                    results[method] = (
                        float(60.0 / lag_times[mask][pks[0]]) if len(pks) else None
                    )
                else:
                    results[method] = None

            elif method == "zero_crossing":
                centered = sig - np.mean(sig)
                pos_crossings = np.where(np.diff(np.sign(centered)) > 0)[0]
                if len(pos_crossings) > 1:
                    ivs = np.diff(pos_crossings) / sf
                    valid = ivs[(ivs >= BAND_MIN_BD) & (ivs <= BAND_MAX_BD)]
                    results[method] = (
                        float(60.0 / np.median(valid)) if len(valid) else None
                    )
                else:
                    results[method] = None

        except Exception as e:
            logger.warning(f"Fallback {method} failed: {e}")
            results[method] = None

    valid_vals = [v for v in results.values() if v is not None]
    ensemble = {
        "individual_estimates": results,
        "respiratory_rate": float(np.median(valid_vals)) if valid_vals else None,
        "std": float(np.std(valid_vals)) if len(valid_vals) > 1 else None,
        "confidence": None,
        "quality": None,
        "n_methods": len(valid_vals),
    }
    return results, ensemble


def _summary_table(results, ensemble):
    # Method metadata: label, category, principle
    METHODS = [
        (
            "counting",
            "Counting (Peak Detection RR)",
            "Time",
            "Counts peaks in respiratory signal; computes RR from inter-peak intervals",
        ),
        (
            "peaks",
            "Peak Interval Detection",
            "Time",
            "Detects breath cycles via peak-to-peak intervals with duration filtering",
        ),
        (
            "zero_crossing",
            "Zero-Crossing Detection",
            "Time",
            "Counts positive-going zero crossings as breath cycle markers",
        ),
        (
            "time_domain",
            "Time Domain (Autocorrelation)",
            "Time",
            "Finds dominant breath period from the autocorrelation lag peak",
        ),
        (
            "fft_based",
            "FFT-Based RR",
            "Frequency",
            "Identifies dominant frequency in respiratory band via FFT magnitude spectrum",
        ),
        (
            "frequency_domain",
            "Frequency Domain RR (Welch)",
            "Frequency",
            "Estimates dominant respiratory frequency from Welch power spectral density",
        ),
    ]

    CATEGORY_COLOR = {"Time": "info", "Frequency": "primary"}

    # Per-method rows
    method_rows = []
    for key, label, category, principle in METHODS:
        rr = results.get(key)
        rr_cell = html.Td(
            _rr_label(rr),
            className="fw-bold text-primary" if rr is not None else "text-muted",
        )
        method_rows.append(
            html.Tr(
                [
                    html.Td(
                        [
                            dbc.Badge(
                                category,
                                color=CATEGORY_COLOR[category],
                                className="me-2",
                            ),
                            label,
                        ]
                    ),
                    html.Td(principle, className="text-muted small"),
                    rr_cell,
                ]
            )
        )

    methods_table = dbc.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Method", style={"width": "25%"}),
                        html.Th("Estimation Principle", style={"width": "55%"}),
                        html.Th("RR (bpm)", style={"width": "20%"}),
                    ]
                )
            ),
            html.Tbody(method_rows),
        ],
        bordered=True,
        hover=True,
        responsive=True,
        size="sm",
        className="mb-0",
    )

    # Agreement stats
    ens_rr = ensemble.get("respiratory_rate")
    ens_mean = ensemble.get("mean_rate")
    ens_std = ensemble.get("std")
    ens_conf = ensemble.get("confidence")
    ens_qual = ensemble.get("quality", "")
    n_methods = ensemble.get("n_methods", 0)

    qual_color = {
        "high": "success",
        "medium": "warning",
        "low": "danger",
        "failed": "danger",
    }.get(ens_qual, "secondary")

    stats_cards = dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(
                                "Consensus (median)", className="text-muted small"
                            ),
                            html.Div(
                                _rr_label(ens_rr), className="fw-bold fs-5 text-success"
                            ),
                        ]
                    ),
                    className="text-center",
                ),
                md=3,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div("Mean", className="text-muted small"),
                            html.Div(_rr_label(ens_mean), className="fw-bold fs-5"),
                        ]
                    ),
                    className="text-center",
                ),
                md=3,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div("Std Dev", className="text-muted small"),
                            html.Div(
                                f"{ens_std:.1f} bpm" if ens_std is not None else "—",
                                className="fw-bold fs-5",
                            ),
                        ]
                    ),
                    className="text-center",
                ),
                md=3,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div("Confidence", className="text-muted small"),
                            html.Div(
                                f"{ens_conf:.0%}" if ens_conf is not None else "—",
                                className="fw-bold fs-5",
                            ),
                        ]
                    ),
                    className="text-center",
                ),
                md=2,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div("Quality", className="text-muted small"),
                            html.Div(
                                dbc.Badge(
                                    ens_qual.upper() if ens_qual else "—",
                                    color=qual_color,
                                    className="fs-6",
                                ),
                                className="mt-1",
                            ),
                        ]
                    ),
                    className="text-center",
                ),
                md=1,
            ),
        ],
        className="g-2 mb-3",
    )

    return html.Div(
        [
            stats_cards,
            methods_table,
        ]
    )


# ── callbacks ─────────────────────────────────────────────────────────────────


def register_respiratory_callbacks(app):
    logger.info("=== REGISTERING RESPIRATORY CALLBACKS ===")
    _import_vitaldsp_modules()

    @app.callback(
        [
            Output("resp-start-position-slider", "min"),
            Output("resp-start-position-slider", "max"),
            Output("resp-start-position-slider", "value"),
        ],
        Input("url", "pathname"),
        prevent_initial_call=True,
    )
    def update_slider_range(pathname):
        if pathname != "/respiratory":
            raise PreventUpdate
        return 0, 100, 0

    @app.callback(
        [
            Output("resp-main-plot", "figure"),
            Output("resp-plot-counting", "figure"),
            Output("resp-plot-fft-based", "figure"),
            Output("resp-plot-freq-domain", "figure"),
            Output("resp-plot-time-domain", "figure"),
            Output("resp-plot-peaks", "figure"),
            Output("resp-plot-zero-crossing", "figure"),
            Output("resp-result-counting", "children"),
            Output("resp-result-fft-based", "children"),
            Output("resp-result-freq-domain", "children"),
            Output("resp-result-time-domain", "children"),
            Output("resp-result-peaks", "children"),
            Output("resp-result-zero-crossing", "children"),
            Output("resp-analysis-results", "children"),
            Output("resp-data-store", "data"),
        ],
        [
            Input("url", "pathname"),
            Input("resp-analyze-btn", "n_clicks"),
            Input("resp-btn-nudge-m10", "n_clicks"),
            Input("resp-btn-nudge-m1", "n_clicks"),
            Input("resp-btn-nudge-p1", "n_clicks"),
            Input("resp-btn-nudge-p10", "n_clicks"),
        ],
        [
            State("resp-start-position-slider", "value"),
            State("resp-duration-select", "value"),
            State("resp-min-breath-duration", "value"),
            State("resp-max-breath-duration", "value"),
            State("store-filtered-signal", "data"),
        ],
        prevent_initial_call=True,
    )
    def run_analysis(
        pathname,
        n_clicks,
        nm10,
        nm1,
        np1,
        np10,
        start_pos,
        duration,
        min_bd,
        max_bd,
        filtered_data,
    ):

        N_OUTPUTS = 15
        EMPTY = tuple([_empty_fig()] * 7 + ["—"] * 6 + ["", None])

        if pathname != "/respiratory":
            raise PreventUpdate

        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate

        trigger = ctx.triggered[0]["prop_id"].split(".")[0]

        # Nudge
        sp = float(start_pos or 0)
        if trigger == "resp-btn-nudge-m10":
            sp = max(0.0, sp - 10)
        elif trigger == "resp-btn-nudge-m1":
            sp = max(0.0, sp - 5)
        elif trigger == "resp-btn-nudge-p1":
            sp = min(100.0, sp + 5)
        elif trigger == "resp-btn-nudge-p10":
            sp = min(100.0, sp + 10)

        min_bd = float(min_bd or 1.8)
        max_bd = float(max_bd or 6.0)

        try:
            time_axis, resp, raw, sf = _load_signal(sp, duration, filtered_data)
        except ValueError as e:
            err_fig = _empty_fig(str(e))
            alert = dbc.Alert(str(e), color="warning")
            return tuple([err_fig] * 7 + ["—"] * 6 + [alert, None])
        except Exception as e:
            logger.error(f"Signal load error: {e}")
            return EMPTY

        try:
            results, ensemble = _run_all_methods(resp, sf, min_bd, max_bd)

            return (
                _main_plot(time_axis, resp),
                _plot_counting(time_axis, resp, sf, min_bd),
                _plot_fft_based(time_axis, resp, sf),
                _plot_freq_domain(time_axis, resp, sf),
                _plot_time_domain(time_axis, resp, sf),
                _plot_peaks(time_axis, resp, sf, min_bd, max_bd),
                _plot_zero_crossing(time_axis, resp, sf, min_bd, max_bd),
                _rr_label(results.get("counting")),
                _rr_label(results.get("fft_based")),
                _rr_label(results.get("frequency_domain")),
                _rr_label(results.get("time_domain")),
                _rr_label(results.get("peaks")),
                _rr_label(results.get("zero_crossing")),
                _summary_table(results, ensemble),
                {
                    "respiratory_signal": resp.tolist(),
                    "time_axis": time_axis.tolist(),
                    "sampling_freq": sf,
                    "rr_results": {
                        k: (round(v, 2) if v is not None else None)
                        for k, v in results.items()
                    },
                    "ensemble_rr": ensemble.get("respiratory_rate"),
                },
            )

        except Exception as e:
            logger.error(f"Analysis error: {e}")
            import traceback

            traceback.print_exc()
            return EMPTY
