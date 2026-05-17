"""
Frequency Domain Analysis callbacks for vitalDSP webapp.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, callback_context, no_update, html, dcc
import dash_bootstrap_components as dbc
from scipy.fft import rfft, rfftfreq
from scipy.signal import welch, spectrogram as scipy_spectrogram
import logging

from vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks import apply_traditional_filter
from vitalDSP_webapp.callbacks.core.theme_callbacks import apply_plot_theme

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Compat stubs — kept so existing test imports don't break
# ─────────────────────────────────────────────────────────────

def configure_plot_with_pan_zoom(fig, title="", height=400):
    fig.update_layout(title=title, height=height, showlegend=True,
                      template="plotly_white", dragmode="pan")
    return fig


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
    return f"{value:.{precision}e}"


def create_empty_figure(theme="light"):
    fig = go.Figure()
    fig.add_annotation(text="No data available", xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="gray"))
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False),
                      plot_bgcolor="white")
    return apply_plot_theme(fig, theme)


# ─────────────────────────────────────────────────────────────
# Legacy compat stubs — kept so existing tests don't break
# ─────────────────────────────────────────────────────────────

def create_fft_plot(selected_signal=None, sampling_freq=100, window_type="hann",
                    n_points=None, freq_min=0, freq_max=25, **kwargs):
    if selected_signal is None or len(selected_signal) < 4:
        return create_empty_figure()
    sig = np.asarray(selected_signal, dtype=float)
    freqs, mag, _ = _compute_fft(sig, float(sampling_freq), window_type)
    fig = go.Figure(go.Scatter(x=freqs, y=mag, mode="lines", name="FFT"))
    fig.update_layout(template="plotly_white")
    return fig


def create_stft_plot(selected_signal=None, sampling_freq=100, window_size=256,
                     hop_size=128, window_type="hann", freq_max=25, **kwargs):
    if selected_signal is None or len(selected_signal) < 64:
        return create_empty_figure()
    try:
        sig = np.asarray(selected_signal, dtype=float)
        t, f, Sxx = _compute_spectrogram(sig, float(sampling_freq), window_type)
        return _build_spectrogram_fig(t, f, Sxx, float(freq_max), "light")
    except Exception:
        return create_empty_figure()


def create_wavelet_plot(selected_signal=None, sampling_freq=100, wavelet_type="db4",
                        levels=4, freq_min=0, freq_max=25, **kwargs):
    return create_empty_figure()


def perform_fft_analysis(time_data=None, selected_signal=None, sampling_freq=100,
                          window_type="hann", n_points=None, freq_min=0, freq_max=25,
                          analysis_options=None, **kwargs):
    ef = create_empty_figure()
    return ef, ef, ef, html.Div(), html.Div(), html.Div(), html.Div(), html.Div(), None, None


def perform_psd_analysis(time_data=None, selected_signal=None, sampling_freq=100,
                          psd_window=2.0, psd_overlap=50, psd_freq_max=25,
                          psd_log_scale=None, psd_normalize=None, psd_channel="signal",
                          freq_min=0, freq_max=25, analysis_options=None, **kwargs):
    ef = create_empty_figure()
    return ef, ef, ef, html.Div(), html.Div(), html.Div(), html.Div(), html.Div(), None, None


def perform_stft_analysis(time_data=None, selected_signal=None, sampling_freq=100,
                           stft_window_size=256, stft_hop_size=128, stft_window_type="hann",
                           stft_overlap=50, stft_scaling="density", stft_freq_max=25,
                           stft_colormap="viridis", freq_min=0, freq_max=25,
                           analysis_options=None, **kwargs):
    ef = create_empty_figure()
    return ef, ef, ef, html.Div(), html.Div(), html.Div(), html.Div(), html.Div(), None, None


def perform_wavelet_analysis(time_data=None, selected_signal=None, sampling_freq=100,
                              wavelet_type="db4", wavelet_levels=4, freq_min=0, freq_max=25,
                              analysis_options=None, **kwargs):
    ef = create_empty_figure()
    return ef, ef, ef, html.Div(), html.Div(), html.Div(), html.Div(), html.Div(), None, None


def generate_frequency_analysis_results(*args, **kwargs):
    return html.Div()


def generate_peak_analysis_table(*args, **kwargs):
    return html.Div()


def generate_band_power_table(*args, **kwargs):
    return html.Div()


def generate_stability_table(*args, **kwargs):
    return html.Div()


def generate_harmonics_table(*args, **kwargs):
    return html.Div()


# ─────────────────────────────────────────────────────────────
# Physiological band definitions per signal type
# ─────────────────────────────────────────────────────────────

_BANDS = {
    "PPG": {
        "VLF": (0.003, 0.04,  "rgba(147,112,219,0.15)", "#9370db"),
        "LF":  (0.04,  0.15,  "rgba(70,130,180,0.15)",  "#4682b4"),
        "HF":  (0.15,  0.40,  "rgba(42,157,143,0.15)",  "#2a9d8f"),
        "HR":  (0.67,  3.33,  "rgba(230,57,70,0.10)",   "#e63946"),
    },
    "ECG": {
        "VLF": (0.003, 0.04,  "rgba(147,112,219,0.15)", "#9370db"),
        "LF":  (0.04,  0.15,  "rgba(70,130,180,0.15)",  "#4682b4"),
        "HF":  (0.15,  0.40,  "rgba(42,157,143,0.15)",  "#2a9d8f"),
        "HR":  (0.67,  3.33,  "rgba(230,57,70,0.10)",   "#e63946"),
    },
    "EEG": {
        "Delta": (0.5,  4.0,  "rgba(147,112,219,0.15)", "#9370db"),
        "Theta": (4.0,  8.0,  "rgba(70,130,180,0.15)",  "#4682b4"),
        "Alpha": (8.0,  13.0, "rgba(42,157,143,0.15)",  "#2a9d8f"),
        "Beta":  (13.0, 30.0, "rgba(230,57,70,0.10)",   "#e63946"),
        "Gamma": (30.0, 80.0, "rgba(244,162,97,0.10)",  "#f4a261"),
    },
}


def _bands_for(signal_type):
    return _BANDS.get(signal_type, _BANDS["PPG"])


def _apply_window(sig, window_type):
    n = len(sig)
    if window_type == "hamming":
        return sig * np.hamming(n)
    elif window_type == "blackman":
        return sig * np.blackman(n)
    elif window_type == "hann":
        return sig * np.hanning(n)
    return sig  # rectangular



# ─────────────────────────────────────────────────────────────
# Metric card
# ─────────────────────────────────────────────────────────────

def _metric_card(title, value, unit, description, color="primary"):
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, className="text-muted small mb-1"),
            html.Div(
                [html.Span(value, className=f"fs-4 fw-bold text-{color}"),
                 html.Span(f" {unit}", className="text-muted small")],
            ),
            html.Div(description, className="text-muted", style={"fontSize": "0.72rem"}),
        ]),
        className="h-100 border-0 shadow-sm",
    )


# ─────────────────────────────────────────────────────────────
# Core spectrum computation
# ─────────────────────────────────────────────────────────────

def _compute_psd_welch(sig, fs, window_type="hann", nperseg_s=2.0):
    """Return (freqs, psd) via Welch's method."""
    nperseg = min(int(nperseg_s * fs), len(sig))
    nperseg = max(nperseg, 64)
    w = window_type if window_type != "none" else "boxcar"
    freqs, psd = welch(sig, fs=fs, window=w, nperseg=nperseg, noverlap=nperseg // 2)
    return freqs, psd


def _compute_fft(sig, fs, window_type="hann"):
    """Return (freqs, magnitude, power) via FFT."""
    windowed = _apply_window(sig, window_type)
    n = len(sig)
    fft_vals = rfft(windowed, n=n)
    freqs = rfftfreq(n, 1.0 / fs)
    magnitude = np.abs(fft_vals) * 2 / n  # single-sided, amplitude-normalised
    power = magnitude ** 2
    return freqs, magnitude, power


def _compute_spectrogram(sig, fs, window_type="hann", nperseg_s=2.0):
    """Return (times, freqs, Sxx_db) spectrogram."""
    nperseg = min(int(nperseg_s * fs), len(sig))
    nperseg = max(nperseg, 64)
    w = window_type if window_type != "none" else "boxcar"
    f, t, Sxx = scipy_spectrogram(sig, fs=fs, window=w, nperseg=nperseg,
                                   noverlap=nperseg // 2)
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    return t, f, Sxx_db


def _band_power(freqs, psd, lo, hi):
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


# ─────────────────────────────────────────────────────────────
# Main spectrum figure
# ─────────────────────────────────────────────────────────────

def _build_spectrum_fig(freqs, power_db, power_linear, analysis_type,
                        signal_type, scale, max_hz, theme, peaks=None):
    """Single-panel spectrum plot with band shading and peak markers."""
    bands = _bands_for(signal_type)
    use_db = (scale == "db")

    y = power_db if use_db else power_linear
    ylabel = "Power (dB)" if use_db else "Power"

    # Clip to max_hz
    mask = freqs <= max_hz
    xd, yd = freqs[mask], y[mask]

    fig = go.Figure()

    # Band shading
    for band_name, (lo, hi, fill, line_col) in bands.items():
        if lo >= max_hz:
            continue
        hi_clipped = min(hi, max_hz)
        fig.add_vrect(x0=lo, x1=hi_clipped, fillcolor=fill,
                      layer="below", line_width=0,
                      annotation_text=band_name,
                      annotation_position="top left",
                      annotation_font=dict(size=9, color=line_col))

    # Main spectrum
    method_label = {"psd": "PSD (Welch)", "fft": "FFT Amplitude"}.get(analysis_type, "Spectrum")
    fig.add_trace(go.Scatter(
        x=xd, y=yd, mode="lines", name=method_label,
        line=dict(color="#1f77b4", width=1.8),
        hovertemplate="<b>%{x:.3f} Hz</b><br>%{y:.2f}" + (" dB" if use_db else "") + "<extra></extra>",
    ))

    # Peak markers
    if peaks is not None and len(peaks) > 0:
        px_arr = np.array(peaks)
        valid = px_arr[(px_arr >= 0) & (px_arr < len(freqs[mask]))]
        if len(valid):
            fig.add_trace(go.Scatter(
                x=freqs[mask][valid], y=y[mask][valid],
                mode="markers+text", name="Peaks",
                marker=dict(color="#e63946", size=8, symbol="diamond"),
                text=[f"{freqs[mask][i]:.2f} Hz" for i in valid],
                textposition="top center",
                textfont=dict(size=9),
            ))

    fig.update_layout(
        xaxis_title="Frequency (Hz)",
        yaxis_title=ylabel,
        height=300,
        template="plotly_white",
        dragmode="pan",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=56, r=24, t=32, b=48),
    )
    return apply_plot_theme(fig, theme)


# ─────────────────────────────────────────────────────────────
# Spectrogram figure
# ─────────────────────────────────────────────────────────────

def _build_spectrogram_fig(t, f, Sxx_db, max_hz, theme):
    mask = f <= max_hz
    fig = go.Figure(go.Heatmap(
        x=t, y=f[mask], z=Sxx_db[mask],
        colorscale="Viridis",
        colorbar=dict(title="dB", thickness=14),
        hovertemplate="Time: %{x:.2f}s<br>Freq: %{y:.2f}Hz<br>%{z:.1f} dB<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        height=300,
        template="plotly_white",
        margin=dict(l=56, r=24, t=16, b=48),
        xaxis=dict(dtick=2, tickangle=0),
    )
    return apply_plot_theme(fig, theme)


# ─────────────────────────────────────────────────────────────
# Results panels
# ─────────────────────────────────────────────────────────────

def _build_peak_table(freqs, power, max_hz, n_peaks=8):
    """Top N spectral peaks table."""
    mask = freqs <= max_hz
    f_clip = freqs[mask]
    p_clip = power[mask]

    if len(f_clip) < 3:
        return html.P("Not enough data for peak detection.", className="small text-muted")

    # Simple local-max peak picking (min distance = 0.02 Hz)
    min_dist = max(1, int(0.02 / (f_clip[1] - f_clip[0]))) if len(f_clip) > 1 else 1
    peaks = []
    for i in range(1, len(f_clip) - 1):
        if p_clip[i] > p_clip[i - 1] and p_clip[i] > p_clip[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_dist:
                peaks.append(i)

    peaks = sorted(peaks, key=lambda i: p_clip[i], reverse=True)[:n_peaks]

    if not peaks:
        return html.P("No peaks detected.", className="small text-muted")

    rows = []
    for rank, idx in enumerate(peaks, 1):
        freq = f_clip[idx]
        power_val = p_clip[idx]
        power_db = 10 * np.log10(power_val + 1e-12)
        rows.append(html.Tr([
            html.Td(f"#{rank}", className="small text-muted"),
            html.Td(f"{freq:.3f} Hz", className="fw-semibold small font-monospace"),
            html.Td(f"{power_db:.1f} dB", className="small font-monospace"),
            html.Td(f"{power_val:.3e}", className="small font-monospace text-muted"),
        ]))

    return dbc.Table(
        [html.Thead(html.Tr([html.Th("Rank", className="small"), html.Th("Frequency", className="small"),
                              html.Th("Power (dB)", className="small"), html.Th("Power (lin)", className="small")])),
         html.Tbody(rows)],
        bordered=False, striped=True, hover=True, size="sm",
    )


def _build_band_power_panel(freqs, psd, signal_type, total_power):
    """Band power table."""
    bands = _bands_for(signal_type)
    rows = []

    for band_name, (lo, hi, fill, color) in bands.items():
        bp = _band_power(freqs, psd, lo, hi)
        pct = 100.0 * bp / total_power if total_power > 0 else 0.0
        rows.append(html.Tr([
            html.Td(html.Span(band_name, style={"color": color, "fontWeight": "600"}), className="small"),
            html.Td(f"{lo:.3f}–{hi:.3f} Hz", className="small font-monospace text-muted"),
            html.Td(f"{bp:.4f} ms²", className="small font-monospace"),
            html.Td(f"{pct:.1f}%", className="small font-monospace"),
        ]))

    return html.Div([
        dbc.Table(
            [html.Thead(html.Tr([
                html.Th("Band", className="small"), html.Th("Range", className="small"),
                html.Th("Power (ms²)", className="small"), html.Th("Share (%)", className="small"),
            ])),
             html.Tbody(rows)],
            bordered=False, striped=True, size="sm",
        ),
        html.P(f"Total power (all bands): {total_power:.4f} ms²",
               className="small text-muted text-end mt-1"),
    ])


def _build_harmonic_panel(freqs, power, fs, max_hz):
    """Fundamental frequency + harmonics analysis."""
    mask = freqs <= max_hz
    f_clip = freqs[mask]
    p_clip = power[mask]

    if len(f_clip) < 4:
        return html.P("Insufficient data.", className="small text-muted")

    # Find fundamental = highest-power peak above 0.05 Hz
    lo_mask = f_clip > 0.05
    if not np.any(lo_mask):
        return html.P("No fundamental detected above 0.05 Hz.", className="small text-muted")

    f_lo = f_clip[lo_mask]
    p_lo = p_clip[lo_mask]
    fund_rel = int(np.argmax(p_lo))
    fund_freq = float(f_lo[fund_rel])
    fund_power = float(p_lo[fund_rel])

    # Harmonics: 2nd … 5th
    harmonic_rows = [
        html.Tr([
            html.Td("Fundamental (1st)", className="small fw-semibold"),
            html.Td(f"{fund_freq:.3f} Hz", className="small font-monospace"),
            html.Td(f"{60 * fund_freq:.1f} bpm", className="small font-monospace text-muted"),
            html.Td(f"{10 * np.log10(fund_power + 1e-12):.1f} dB", className="small font-monospace"),
            html.Td("—", className="small text-muted"),
        ])
    ]
    for k in range(2, 6):
        hf = fund_freq * k
        if hf > max_hz:
            break
        # Find closest bin
        idx = int(np.argmin(np.abs(f_clip - hf)))
        actual_f = float(f_clip[idx])
        hp = float(p_clip[idx])
        ratio_db = 10 * np.log10(hp / (fund_power + 1e-12))
        harmonic_rows.append(html.Tr([
            html.Td(f"{k}{'nd' if k==2 else 'rd' if k==3 else 'th'} harmonic", className="small text-muted"),
            html.Td(f"{actual_f:.3f} Hz", className="small font-monospace"),
            html.Td(f"{60 * actual_f:.1f} bpm", className="small font-monospace text-muted"),
            html.Td(f"{10 * np.log10(hp + 1e-12):.1f} dB", className="small font-monospace"),
            html.Td(f"{ratio_db:+.1f} dB", className="small font-monospace text-muted"),
        ]))

    return dbc.Table(
        [html.Thead(html.Tr([html.Th("Component", className="small"), html.Th("Freq (Hz)", className="small"),
                              html.Th("Rate (bpm)", className="small"), html.Th("Power (dB)", className="small"),
                              html.Th("Rel. fund.", className="small")])),
         html.Tbody(harmonic_rows)],
        bordered=False, striped=True, size="sm",
    )


def _build_stability_panel(sig, fs, window_type, max_hz):
    """Spectral stability: compute PSD on 4 equal sub-windows, show CoV of band powers."""
    n = len(sig)
    if n < 256:
        return html.P("Signal too short for stability analysis (need ≥ 256 samples).",
                      className="small text-muted")

    n_segs = 4
    seg_len = n // n_segs
    band_powers = {k: [] for k in ["VLF", "LF", "HF", "HR"]}
    band_ranges = {"VLF": (0.003, 0.04), "LF": (0.04, 0.15), "HF": (0.15, 0.40), "HR": (0.67, 3.33)}

    for i in range(n_segs):
        seg = sig[i * seg_len: (i + 1) * seg_len]
        freqs_s, psd_s = _compute_psd_welch(seg, fs, window_type)
        for bname, (lo, hi) in band_ranges.items():
            band_powers[bname].append(_band_power(freqs_s, psd_s, lo, hi))

    rows = []
    for bname, vals in band_powers.items():
        vals_arr = np.array(vals)
        mean_v = float(np.mean(vals_arr))
        std_v = float(np.std(vals_arr))
        cov = std_v / mean_v if mean_v > 0 else 0.0
        stability = "High" if cov < 0.2 else ("Moderate" if cov < 0.5 else "Low")
        color = "success" if cov < 0.2 else ("warning" if cov < 0.5 else "danger")
        rows.append(html.Tr([
            html.Td(bname, className="small fw-semibold"),
            html.Td(f"{mean_v:.4f} ms²", className="small font-monospace"),
            html.Td(f"{std_v:.4f} ms²", className="small font-monospace text-muted"),
            html.Td(f"{cov:.2f}", className="small font-monospace"),
            html.Td(dbc.Badge(stability, color=color, className="small")),
        ]))

    note = html.P(
        f"Computed over {n_segs} equal segments. CoV < 0.2 = High, 0.2–0.5 = Moderate, > 0.5 = Low.",
        className="small text-muted mt-2",
    )

    return html.Div([
        dbc.Table(
            [html.Thead(html.Tr([html.Th("Band", className="small"), html.Th("Mean Power", className="small"),
                                  html.Th("Std Dev", className="small"), html.Th("CoV", className="small"),
                                  html.Th("Stability", className="small")])),
             html.Tbody(rows)],
            bordered=False, striped=True, size="sm",
        ),
        note,
    ])


def _build_metric_cards(freqs, psd, signal_type, fs, duration_s):
    """4 summary cards: dominant freq, LF/HF ratio, total power, spectral entropy."""
    bands = _bands_for(signal_type)

    # Dominant frequency (ignore DC)
    lo_mask = freqs > 0.05
    if np.any(lo_mask):
        dom_idx = np.argmax(psd[lo_mask])
        dom_freq = float(freqs[lo_mask][dom_idx])
    else:
        dom_freq = 0.0

    # LF/HF ratio (HRV bands)
    lf = _band_power(freqs, psd, 0.04, 0.15)
    hf = _band_power(freqs, psd, 0.15, 0.40)
    lf_hf = lf / hf if hf > 0 else 0.0

    # Total power
    total_power = _band_power(freqs, psd, 0.003, 0.40)

    # Spectral entropy
    psd_norm = psd / (psd.sum() + 1e-12)
    spectral_entropy = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))

    return dbc.Row(
        [
            dbc.Col(_metric_card("Dominant Freq", f"{dom_freq:.3f}", "Hz",
                                 f"{dom_freq * 60:.1f} bpm", "primary"), md=3, className="mb-3"),
            dbc.Col(_metric_card("LF/HF Ratio", f"{lf_hf:.2f}", "",
                                 "Autonomic balance (HRV bands)", "warning"), md=3, className="mb-3"),
            dbc.Col(_metric_card("Total Power", f"{total_power:.4f}", "ms²",
                                 "VLF + LF + HF (0.003–0.4 Hz)", "success"), md=3, className="mb-3"),
            dbc.Col(_metric_card("Spectral Entropy", f"{spectral_entropy:.2f}", "bits",
                                 "Signal complexity / randomness", "info"), md=3, className="mb-3"),
        ],
        className="g-2",
    )


def _build_signal_summary(signal_source_info, filter_info, fs, n_samples, signal_type, start_s, duration_s):
    rows = [
        html.Tr([html.Td("Signal source", className="fw-semibold text-muted small"), html.Td(signal_source_info or "—")]),
        html.Tr([html.Td("Signal type", className="fw-semibold text-muted small"), html.Td(signal_type or "—")]),
        html.Tr([html.Td("Sampling rate", className="fw-semibold text-muted small"), html.Td(f"{fs:.0f} Hz")]),
        html.Tr([html.Td("Window", className="fw-semibold text-muted small"), html.Td(f"{start_s:.1f}s – {start_s + duration_s:.1f}s ({duration_s:.1f}s)")]),
        html.Tr([html.Td("Samples", className="fw-semibold text-muted small"), html.Td(f"{n_samples:,}")]),
    ]

    if isinstance(filter_info, dict) and filter_info:
        ftype = filter_info.get("filter_type", "—")
        chain = filter_info.get("chain") or []
        if chain:
            # Render each chain step as its own row
            rows.append(html.Tr([
                html.Td("Filter chain", className="fw-semibold text-muted small",
                        rowSpan=len(chain) + 1),
                html.Td(html.Span(f"{len(chain)} stage{'s' if len(chain) != 1 else ''}",
                                  className="text-muted small")),
            ]))
            for i, step in enumerate(chain, 1):
                family = step.get("family", "—")
                params = step.get("params", {}) or {}
                param_str = ", ".join(f"{k}={v}" for k, v in params.items() if v is not None)
                label = f"  Step {i}: {family}"
                value = param_str or "—"
                rows.append(html.Tr([
                    html.Td(label, className="text-muted small ps-3 font-monospace"),
                    html.Td(value, className="small font-monospace"),
                ]))
        else:
            params = filter_info.get("parameters", {}) or {}
            param_str = ", ".join(f"{k}={v}" for k, v in params.items() if v is not None) if params else ""
            rows.append(html.Tr([
                html.Td("Filter", className="fw-semibold text-muted small"),
                html.Td(f"{ftype}" + (f" ({param_str})" if param_str else "")),
            ]))

    return dbc.Table([html.Tbody(rows)], bordered=False, size="sm", className="table-borderless")


# ─────────────────────────────────────────────────────────────
# Callback registration
# ─────────────────────────────────────────────────────────────

def register_frequency_filtering_callbacks(app):
    """Register all frequency domain analysis callbacks."""

    # ── collapse: spectrogram (index 0) ────────────────────────
    @app.callback(
        Output("collapse-spectrogram", "is_open"),
        Input("btn-collapse-spectrogram", "n_clicks"),
        State("collapse-spectrogram", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_spectrogram(n, is_open):
        return not is_open

    # ── main analysis callback (index 1) ────────────────────────
    @app.callback(
        [
            Output("freq-main-plot",          "figure"),
            Output("freq-spectrogram-plot",   "figure"),
            Output("freq-metric-cards",       "children"),
            Output("freq-peak-analysis-table","children"),
            Output("freq-band-power-table",   "children"),
            Output("freq-harmonics-table",    "children"),
            Output("freq-stability-table",    "children"),
            Output("freq-analysis-results",   "children"),
            Output("store-frequency-data",    "data"),
        ],
        [
            Input("freq-btn-update-analysis", "n_clicks"),
            Input("body", "data-theme"),
            Input("url", "pathname"),
        ],
        [
            State("freq-start-position-slider", "value"),
            State("freq-duration-select",       "value"),
            State("freq-analysis-type",         "value"),
            State("freq-window-type",           "value"),
            State("freq-max-hz",                "value"),
            State("freq-scale",                 "value"),
            State("freq-signal-type",           "value"),
            State("store-filtered-signal",      "data"),
        ],
    )
    def run_frequency_analysis(
        n_clicks, current_theme, pathname,
        start_position, duration,
        analysis_type, window_type, max_hz, scale, signal_type,
        filtered_signal_data,
    ):
        logger.info("=== FREQUENCY DOMAIN ANALYSIS ===")

        def _empty(msg):
            ef = create_empty_figure(current_theme)
            empty_div = html.P(msg, className="text-muted small p-3")
            return ef, ef, html.Div(), empty_div, empty_div, empty_div, empty_div, empty_div, None

        if pathname != "/frequency":
            return _empty("Navigate to the Frequency Domain Analysis page.")

        try:
            from vitalDSP_webapp.services.data.enhanced_data_service import get_enhanced_data_service
            data_service = get_enhanced_data_service()
            all_data = data_service.get_all_data()

            if not all_data:
                return _empty("No data available. Please upload and process data first.")

            latest_data_id = list(all_data.keys())[-1]
            latest_data = all_data[latest_data_id]
            column_mapping = data_service.get_column_mapping(latest_data_id)

            if not column_mapping:
                return _empty("Please configure column mapping on the Upload page first.")

            df = data_service.get_data(latest_data_id)
            if df is None or df.empty:
                return _empty("Data is empty or corrupted.")

            fs = float(latest_data.get("info", {}).get("sampling_freq", 100))

            # ── resolve signal column ──────────────────────────
            signal_col = column_mapping.get("signal")
            if not signal_col or signal_col not in df.columns:
                for cand in ["waveform", "pleth", "pl", "signal", "ppg", "ecg", "red", "ir"]:
                    matches = [c for c in df.columns if c.lower() == cand]
                    if matches:
                        signal_col = matches[0]
                        break
            if not signal_col:
                return _empty("No suitable signal column found.")

            # ── time window ─────────────────────────────────────
            start_position = float(start_position or 0)
            duration = float(duration or 60)
            data_duration = len(df) / fs
            start_s = (start_position / 100.0) * data_duration
            end_s = start_s + duration
            start_sample = int(start_s * fs)
            end_sample = min(int(end_s * fs), len(df))
            if start_sample >= len(df):
                start_sample, start_s = 0, 0.0
                end_sample = min(int(duration * fs), len(df))

            raw_signal = df[signal_col].values[start_sample:end_sample]
            if len(raw_signal) < 64:
                return _empty(f"Window too short ({len(raw_signal)} samples). Increase duration or move start position.")

            # ── always prefer filtered signal ────────────────────
            selected_signal = raw_signal
            signal_source_info = "Original Signal"
            filter_info = None

            svc_filtered = data_service.get_filtered_data(latest_data_id)
            svc_filter_info = data_service.get_filter_info(latest_data_id)

            filtered_full = None
            stored_filter_info = None

            if svc_filtered is not None:
                filtered_full = svc_filtered
                stored_filter_info = svc_filter_info
            elif isinstance(filtered_signal_data, dict) and "signal" in filtered_signal_data:
                try:
                    filtered_full = np.array(filtered_signal_data["signal"])
                    if "filter_params" in filtered_signal_data:
                        stored_filter_info = {
                            "filter_type": filtered_signal_data.get("filter_type", "unknown"),
                            "parameters": filtered_signal_data.get("filter_params", {}),
                        }
                except Exception as e:
                    logger.warning("Could not load filtered signal from store: %s", e)

            if filtered_full is not None:
                if len(filtered_full) >= end_sample:
                    selected_signal = filtered_full[start_sample:end_sample]
                    signal_source_info = "Filtered Signal"
                    filter_info = stored_filter_info
                elif len(filtered_full) == len(raw_signal):
                    selected_signal = filtered_full
                    signal_source_info = "Filtered Signal"
                    filter_info = stored_filter_info
                else:
                    saved_chain = (stored_filter_info or {}).get("chain") or []
                    if saved_chain:
                        try:
                            from vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks import apply_filter_chain
                            selected_signal = apply_filter_chain(
                                raw_signal, fs, signal_type, saved_chain, logger=logger)
                            signal_source_info = f"Filtered Signal (chain x{len(saved_chain)})"
                            filter_info = stored_filter_info
                        except Exception as e:
                            logger.warning("Re-applying filter chain failed: %s", e)
                    elif stored_filter_info and stored_filter_info.get("filter_type") == "traditional":
                        try:
                            p = stored_filter_info.get("parameters", {})
                            selected_signal = apply_traditional_filter(
                                raw_signal, fs,
                                p.get("filter_family", "butter"),
                                p.get("filter_response", "bandpass"),
                                p.get("low_freq", 0.5),
                                p.get("high_freq", 5),
                                p.get("filter_order", 4),
                            )
                            signal_source_info = "Filtered Signal (re-applied)"
                            filter_info = stored_filter_info
                        except Exception as e:
                            logger.warning("Re-applying traditional filter failed: %s", e)

            # ── zero-mean the signal before spectral analysis ────
            sig = np.asarray(selected_signal, dtype=float)
            sig = sig - np.mean(sig)

            # ── resolve parameters ───────────────────────────────
            stype = signal_type or "PPG"
            win = window_type or "hann"
            max_f = float(max_hz or 25)
            max_f = min(max_f, fs / 2 - 0.01)

            # ── compute PSD (always via Welch for results panels) ─
            freqs_w, psd_w = _compute_psd_welch(sig, fs, win)

            # ── compute the requested spectrum for main plot ──────
            if analysis_type == "fft":
                freqs_main, mag, power_lin = _compute_fft(sig, fs, win)
                power_db = 20 * np.log10(mag + 1e-12)
            else:  # psd or stft → use Welch for the spectrum
                freqs_main = freqs_w
                power_lin = psd_w
                power_db = 10 * np.log10(psd_w + 1e-12)

            # ── peak picking on main spectrum ────────────────────
            mask_main = freqs_main <= max_f
            p_clip = power_lin[mask_main]
            peak_indices = []
            if len(p_clip) > 3:
                min_dist = max(1, int(0.02 / (freqs_main[1] - freqs_main[0]))) if len(freqs_main) > 1 else 1
                for i in range(1, len(p_clip) - 1):
                    if p_clip[i] > p_clip[i - 1] and p_clip[i] > p_clip[i + 1]:
                        if not peak_indices or (i - peak_indices[-1]) >= min_dist:
                            peak_indices.append(i)
                peak_indices = sorted(peak_indices, key=lambda i: p_clip[i], reverse=True)[:5]

            # ── figures ──────────────────────────────────────────
            main_fig = _build_spectrum_fig(
                freqs_main, power_db, power_lin,
                analysis_type, stype, scale, max_f, current_theme,
                peaks=peak_indices,
            )

            # Spectrogram
            try:
                t_sg, f_sg, Sxx = _compute_spectrogram(sig, fs, win)
                spec_fig = _build_spectrogram_fig(t_sg, f_sg, Sxx, max_f, current_theme)
            except Exception as e:
                logger.warning("Spectrogram failed: %s", e)
                spec_fig = create_empty_figure(current_theme)

            # ── results panels ────────────────────────────────────
            # total_power = sum over ALL defined bands (not just VLF–HF)
            bands_for_type = _bands_for(stype)
            total_power = sum(
                _band_power(freqs_w, psd_w, lo, hi)
                for lo, hi, *_ in bands_for_type.values()
            )

            metric_cards = _build_metric_cards(freqs_w, psd_w, stype, fs, duration)
            peak_panel = _build_peak_table(freqs_main, power_lin, max_f)
            band_panel = _build_band_power_panel(freqs_w, psd_w, stype, total_power)
            harmonic_panel = _build_harmonic_panel(freqs_main, power_lin, fs, max_f)
            stability_panel = _build_stability_panel(sig, fs, win, max_f)
            summary = _build_signal_summary(
                signal_source_info, filter_info, fs,
                len(sig), stype, start_s, duration,
            )

            # ── unified Analysis Results panel ────────────────────
            unified_results = html.Div([
                # Config context strip
                html.Div(
                    dbc.Row([
                        dbc.Col(html.Small([html.B("Method: "), analysis_type.upper()], className="text-muted"), md="auto"),
                        dbc.Col(html.Small([html.B("Window fn: "), win], className="text-muted"), md="auto"),
                        dbc.Col(html.Small([html.B("Max freq: "), f"{max_f:.1f} Hz"], className="text-muted"), md="auto"),
                        dbc.Col(html.Small([html.B("Scale: "), scale], className="text-muted"), md="auto"),
                        dbc.Col(html.Small([html.B("Signal type: "), stype], className="text-muted"), md="auto"),
                        dbc.Col(html.Small([html.B("fs: "), f"{fs:.0f} Hz"], className="text-muted"), md="auto"),
                    ], className="g-3"),
                    className="p-2 mb-3 rounded",
                    style={"background": "var(--bs-light, #f8f9fa)"},
                ),
                dbc.Row([
                    dbc.Col([
                        html.H6("Peak Frequencies", className="fw-semibold mb-2 text-muted small text-uppercase"),
                        peak_panel,
                    ], md=6, className="mb-3"),
                    dbc.Col([
                        html.H6("Band Power", className="fw-semibold mb-2 text-muted small text-uppercase"),
                        band_panel,
                    ], md=6, className="mb-3"),
                ]),
                html.Hr(className="my-2"),
                dbc.Row([
                    dbc.Col([
                        html.H6("Harmonic Analysis", className="fw-semibold mb-2 text-muted small text-uppercase"),
                        harmonic_panel,
                    ], md=6, className="mb-3"),
                    dbc.Col([
                        html.H6("Spectral Stability", className="fw-semibold mb-2 text-muted small text-uppercase"),
                        stability_panel,
                    ], md=6, className="mb-3"),
                ]),
            ])

            freq_data = {
                "data_id": latest_data_id,
                "sampling_freq": fs,
                "window": [int(start_sample), int(end_sample)],
                "analysis_type": analysis_type,
            }

            logger.info("Frequency domain analysis completed.")
            return (
                main_fig, spec_fig, metric_cards,
                unified_results, band_panel, harmonic_panel,
                stability_panel, summary, freq_data,
            )

        except Exception as e:
            logger.exception("Frequency domain analysis failed: %s", e)
            return _empty(f"Analysis error: {e}")

    # ── collapse: analysis results ───────────────────────────────
    @app.callback(
        Output("freq-collapse-analysis-results", "is_open"),
        Input("freq-btn-collapse-analysis-results", "n_clicks"),
        State("freq-collapse-analysis-results", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_freq_analysis_results(n, is_open):
        return not is_open

    # ── collapse: signal summary ─────────────────────────────────
    @app.callback(
        Output("freq-collapse-summary", "is_open"),
        Input("freq-btn-collapse-summary", "n_clicks"),
        State("freq-collapse-summary", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_freq_summary(n, is_open):
        return not is_open

    # ── slider nudge ─────────────────────────────────────────────
    @app.callback(
        Output("freq-start-position-slider", "value"),
        [
            Input("freq-btn-nudge-m10", "n_clicks"),
            Input("freq-btn-nudge-m5",  "n_clicks"),
            Input("freq-btn-nudge-p5",  "n_clicks"),
            Input("freq-btn-nudge-p10", "n_clicks"),
        ],
        State("freq-start-position-slider", "value"),
        prevent_initial_call=True,
    )
    def nudge_freq_slider(m10, m5, p5, p10, current):
        ctx = callback_context
        if not ctx.triggered:
            return no_update
        trigger = ctx.triggered[0]["prop_id"].split(".")[0]
        delta = {"freq-btn-nudge-m10": -10, "freq-btn-nudge-m5": -5,
                 "freq-btn-nudge-p5": 5, "freq-btn-nudge-p10": 10}.get(trigger, 0)
        pos = float(current or 0)
        return max(0.0, min(100.0, pos + delta))

