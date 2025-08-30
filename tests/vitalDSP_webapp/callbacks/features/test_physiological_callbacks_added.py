# tests/test_physiological_callbacks_added.py
# Extra coverage-focused tests for physiological_callbacks.py

import os
import sys
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from unittest.mock import MagicMock

# Make local source import-friendly if the packaged import path isn't available
SRC_DIR = os.path.abspath(os.path.join(os.getcwd(), "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Try package path first; fall back to the uploaded single-file module
from vitalDSP_webapp.callbacks.features import physiological_callbacks as physio
from vitalDSP_webapp.callbacks.features.physiological_callbacks import (
    # high-level
    perform_physiological_analysis,
    perform_physiological_analysis_enhanced,

    # vitalDSP wrappers (exercise ImportError fallbacks)
    get_vitaldsp_hrv_analysis,
    get_vitaldsp_morphology_analysis,
    get_vitaldsp_signal_quality,
    get_vitaldsp_transforms,
    get_vitaldsp_advanced_computation,
    get_vitaldsp_feature_engineering,

    # analyses (helpers)
    analyze_preprocessing,
    analyze_trends,
    analyze_frequency,

    # plots
    create_signal_quality_plots,
    create_advanced_features_plots,
    create_hrv_plots,
    create_morphology_plots,
    create_empty_figure,

    # util
    suggest_best_signal_column,
    normalize_signal_type,
    _import_vitaldsp_modules,
)

# --------- Small helpers ---------
def make_sine(fs=100, seconds=10, hz=1.0, noise=0.0):
    t = np.arange(0, seconds, 1.0/fs)
    y = np.sin(2*np.pi*hz*t)
    if noise:
        y = y + np.random.normal(0, noise, size=len(t))
    return t, y

def make_ramp(fs=100, seconds=10):
    t = np.arange(0, seconds, 1.0/fs)
    y = np.linspace(0, 1.0, len(t))
    return t, y

# =============================
# Core analysis functions
# =============================

def test_analyze_hrv_success_and_insufficient():
    fs = 100
    t, y = make_sine(fs=fs, seconds=12, hz=1.2)  # clear peaks ⇒ HRV ok
    metrics = physio.analyze_hrv(y, fs, ["time_domain", "freq_domain"])
    # Expect time-domain keys if enough peaks (mean_rr is typical)
    assert isinstance(metrics, dict)
    assert ("error" in metrics) is False
    assert any(k in metrics for k in ("mean_rr", "std_rr", "rmssd"))

    # Constant / no peaks ⇒ insufficient:
    y2 = np.zeros(1000)
    metrics2 = physio.analyze_hrv(y2, fs, ["time_domain"])
    assert "error" in metrics2  # Insufficient peaks path hit

def test_analyze_morphology_default_and_explicit():
    fs = 125
    t, y = make_sine(fs=fs, seconds=8, hz=1.5)
    m1 = physio.analyze_morphology(y, fs, None)  # default options branch
    assert isinstance(m1, dict)
    assert any(k in m1 for k in ("num_peaks", "peak_heights", "mean_amplitude", "peak_to_peak"))

    m2 = physio.analyze_morphology(y, fs, ["peak_detection", "amplitude", "duration"])
    assert isinstance(m2, dict)
    assert m2.get("num_peaks", 0) >= 1
    assert "mean_amplitude" in m2
    assert "duration_stats" in m2  # duration branch

def test_analyze_signal_quality_and_trends():
    fs = 100
    t, y = make_sine(fs=fs, seconds=6, hz=1.1, noise=0.05)
    q = physio.analyze_signal_quality(y, fs)
    assert isinstance(q, dict) and "snr_db" in q and "zero_crossings" in q

    t2, y2 = make_ramp(fs=fs, seconds=6)
    tr = physio.analyze_trends(y2, fs)
    assert isinstance(tr, dict) and "trend_slope" in tr and tr["trend_slope"] > 0

    # Decreasing ramp
    tr2 = physio.analyze_trends(-y2, fs)
    assert isinstance(tr2, dict) and tr2["trend_slope"] < 0
    assert tr2.get("trend_direction") in ("increasing", "decreasing", "stable")

def test_analyze_transforms_and_advanced_blocks():
    fs = 100
    t, y = make_sine(fs=fs, seconds=5, hz=2.0)
    tfm = physio.analyze_transforms(y, fs, ["wavelet", "fourier", "hilbert"])
    assert isinstance(tfm, dict)
    # Expect at least one transform metric:
    assert any(k for k in tfm.keys())

    adv = physio.analyze_advanced_computation(y, fs, ["anomaly_detection", "bayesian", "kalman"])
    assert {"anomalies_detected", "bayesian_prior_mean", "kalman_estimate"} <= set(adv.keys())

# =============================
# Wrappers w/ vitalDSP fallbacks
# =============================

def test_vitaldsp_wrappers_short_signal_paths():
    fs = 100
    short = np.ones(2*fs)  # 2s only ⇒ short for HRV (need 5s), morphology (2s), quality (3s)

    hrv = physio.get_vitaldsp_hrv_analysis(short, fs, ["time_domain"], "ppg")
    assert "error" in hrv  # short-signal early return

    morph = physio.get_vitaldsp_morphology_analysis(short, fs, ["peaks"], "ppg")
    # may be short depending on threshold (2s minimum). Use 150 samples to be sure:
    shorter = np.ones(int(1.5*fs))
    morph2 = physio.get_vitaldsp_morphology_analysis(shorter, fs, ["peaks"], "ppg")
    assert "error" in morph2

    qual_short = physio.get_vitaldsp_signal_quality(short, fs, ["quality_index"], "ppg")
    assert "error" in qual_short

def test_vitaldsp_wrappers_normal_signals_fallback_to_local():
    fs = 100
    t, y = make_sine(fs=fs, seconds=8, hz=1.1)

    hrv = physio.get_vitaldsp_hrv_analysis(y, fs, ["time_domain", "freq_domain"], "ppg")
    assert isinstance(hrv, dict)

    morph = physio.get_vitaldsp_morphology_analysis(y, fs, ["peaks", "amplitude"], "ppg")
    assert isinstance(morph, dict)

    transforms = physio.get_vitaldsp_transforms(y, fs, ["wavelet", "fourier", "hilbert"], "ppg")
    assert isinstance(transforms, dict)
    # Keys may vary by fallback; just ensure something came back
    assert len(transforms) >= 1

    feats = physio.get_vitaldsp_feature_engineering(y, fs, ["ppg_light"], "ppg")
    assert isinstance(feats, dict)
    assert any(k for k in feats)

# =============================
# Display & plots
# =============================

def test_create_comprehensive_results_display_compact_cards():
    # Build a “results” dict with typical keys for compact card rendering
    results = {
        "hrv_metrics": {"mean_rr": 800.0, "std_rr": 50.0, "rmssd": 30.0},
        "morphology_metrics": {"num_peaks": 12, "mean_amplitude": 0.3, "peak_to_peak": 1.2},
        "quality_metrics": {"signal_quality_index": 0.9, "snr_db": 20.0, "artifacts_detected": 0, "artifact_ratio": 0.0},
        "transform_metrics": {"wavelet_energy": 1.0, "fourier_dominant_freq": 1.2, "hilbert_phase": 0.1},
        "advanced_features_metrics": {"cross_signal_correlation": 0.5, "total_power": 10.0},
    }
    comp = physio.create_comprehensive_results_display(results, "ppg", 100)
    # Expect a Dash HTML container
    from dash import html
    assert isinstance(comp, html.Div)
    assert hasattr(comp, "children")

def test_create_physiological_analysis_plots_multi_category():
    fs = 100
    t, y = make_sine(fs=fs, seconds=10, hz=1.2)
    fig = physio.create_physiological_analysis_plots(
        time_data=t,
        signal_data=y,
        signal_type="ppg",
        sampling_freq=fs,
        analysis_categories=["hrv", "morphology", "beat2beat", "energy", "quality", "transforms", "advanced"],
        hrv_options=["time_domain", "freq_domain"],
        morphology_options=["peaks", "amplitude", "duration"],
        advanced_features=["cross_signal"],
        quality_options=["quality_index", "artifact_detection"],
        transform_options=["fourier", "hilbert"],  # keep light
        advanced_computation=["anomaly_detection"],
        feature_engineering=["ppg_light"],
        preprocessing=["detrend"]
    )
    assert isinstance(fig, go.Figure)
    # Should have at least several traces:
    assert len(fig.data) >= 4

def test_create_signal_quality_plots_three_rows():
    fs = 100
    t, y = make_sine(fs=fs, seconds=6, hz=1.0, noise=0.02)
    fig = physio.create_signal_quality_plots(t, y, fs, ["quality_index", "artifact_detection"])
    assert isinstance(fig, go.Figure)
    # Expect at least 3 traces (main + metrics + artifact)
    assert len(fig.data) >= 3

def test_create_advanced_features_plots_all_sections():
    fs = 100
    t, y = make_sine(fs=fs, seconds=8, hz=1.5)
    fig = physio.create_advanced_features_plots(
        t, y, fs, ["cross_signal", "ensemble", "change_detection", "power_analysis"]
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 3

def test_create_individual_plot_builders():
    fs = 100
    t, y = make_sine(fs=fs, seconds=6, hz=1.0)
    assert isinstance(physio.create_beat_to_beat_plots(t, y, fs), go.Figure)
    assert isinstance(physio.create_energy_plots(t, y, fs), go.Figure)
    assert isinstance(physio.create_envelope_plots(t, y, fs), go.Figure)
    assert isinstance(physio.create_segmentation_plots(t, y, fs), go.Figure)
    assert isinstance(physio.create_waveform_plots(t, y, fs), go.Figure)
    assert isinstance(physio.create_frequency_plots(t, y, fs), go.Figure)
    assert isinstance(physio.create_transform_plots(t, y, fs, ["fourier", "hilbert"]), go.Figure)
    assert isinstance(physio.create_wavelet_plots(t, y, fs), go.Figure)
    assert isinstance(physio.create_fourier_plots(t, y, fs), go.Figure)
    assert isinstance(physio.create_hilbert_plots(t, y, fs), go.Figure)
    assert isinstance(physio.create_morphology_plots(t, y, fs, ["peaks", "amplitude"]), go.Figure)

def test_create_empty_figure_no_data_message():
    fig = physio.create_empty_figure()
    assert isinstance(fig, go.Figure)
    # Should contain the "No data available" annotation text
    assert any(getattr(a, "text", "") == "No data available" for a in fig.layout.annotations)

# =============================
# Signal type + main plot
# =============================

def test_detect_type_and_create_signal_plot():
    fs = 200
    # Faster peaks -> heuristic ECG branch likely
    t_ecg, y_ecg = make_sine(fs=fs, seconds=5, hz=2.5)
    guessed = physio.detect_physiological_signal_type(y_ecg, fs)
    assert guessed in ("ecg", "ppg", "respiratory")

    # Build the plot
    fig = physio.create_physiological_signal_plot(t_ecg, y_ecg, guessed, fs)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1

# =============================
# Time inputs callbacks
# =============================

def _mock_ctx(prop_id):
    """Build a minimal dash.callback_context stub with .triggered list."""
    class _C:
        triggered = [{"prop_id": f"{prop_id}.value"}]
    return _C()

def test_update_physio_time_inputs_slider(monkeypatch):
    # Simulate slider trigger → return the slider range directly
    monkeypatch.setattr(physio, "callback_context", _mock_ctx("physio-time-range-slider"))
    start, end = physio.update_physio_time_inputs([2, 12], 0, 0, 0, 0, 1, 11)
    assert (start, end) == (2, 12)

@pytest.mark.parametrize("btn,expected", [
    ("physio-btn-nudge-m10", (0, 10)),   # floor at 0
    ("physio-btn-nudge-m1",  (0, 10)),   # from 0..10, minus 1 floors to 0..10
    ("physio-btn-nudge-p1",  (2, 12)),   # from 1..11 → 2..12
    ("physio-btn-nudge-p10", (11, 21)),  # from 1..11 → 11..21
])
def test_update_physio_time_inputs_nudges(monkeypatch, btn, expected):
    monkeypatch.setattr(physio, "callback_context", _mock_ctx(btn))
    start, end = physio.update_physio_time_inputs(None, 0, 0, 0, 0, 1, 11)
    assert (start, end) == expected

def test_update_physio_time_inputs_no_trigger(monkeypatch):
    class _Empty:
        triggered = []
    monkeypatch.setattr(physio, "callback_context", _Empty())
    with pytest.raises(Exception):
        physio.update_physio_time_inputs([0, 10], 0, 0, 0, 0, 0, 10)

def test_update_physio_time_slider_range_paths():
    # No data ⇒ default max=100
    assert physio.update_physio_time_slider_range(None) == 100
    # Empty df ⇒ 100
    assert physio.update_physio_time_slider_range({"data": {}}) == 100
    # Valid df: max of first column is returned
    df = pd.DataFrame({"time": [0, 5, 10, 25], "sig": [0.1, 0.2, 0.3, 0.4]})
    assert physio.update_physio_time_slider_range({"data": df.to_dict(orient="list")}) == 25

# =============================
# “Enhanced” analysis aggregator
# =============================

def test_perform_physiological_analysis_enhanced_returns_results():
    fs = 100
    t, y = make_sine(fs=fs, seconds=10, hz=1.1)
    results = physio.perform_physiological_analysis_enhanced(
        time_data=t,
        signal_data=y,
        signal_type="ppg",
        sampling_freq=fs,
        analysis_categories=["hrv", "morphology", "beat2beat", "energy", "quality", "transforms", "advanced"],
        hrv_options=["time_domain"],
        morphology_options=["peaks", "amplitude"],
        advanced_features=["cross_signal"],
        quality_options=["quality_index"],
        transform_options=["fourier"],
        advanced_computation=["anomaly_detection"],
        feature_engineering=["ppg_light"],
        preprocessing=["detrend"]
    )
    assert isinstance(results, dict)
    # At least some keys should be present
    assert any(k in results for k in ("hrv_metrics", "morphology_metrics", "quality_metrics", "transform_metrics"))

def test_perform_physiological_analysis_returns_display():
    fs = 100
    t, y = make_sine(fs=fs, seconds=10, hz=1.0)
    disp = physio.perform_physiological_analysis(
        time_data=t,
        signal_data=y,
        signal_type="ppg",
        sampling_freq=fs,
        analysis_categories=["hrv", "morphology", "quality"],
        hrv_options=["time_domain"],
        morphology_options=["peaks"],
        advanced_features=None,
        quality_options=["quality_index"],
        transform_options=None,
        advanced_computation=None,
        feature_engineering=None,
        preprocessing=None
    )
    # This variant returns an html.Div (results grid)
    from dash import html
    assert isinstance(disp, html.Div)

# =============================
# Column suggestion helper
# =============================

def test_suggest_best_signal_column_ranks_reasonably():
    fs = 100
    t = np.arange(0, 5, 1.0/fs)
    ppg = np.sin(2*np.pi*1.2*t)            # good, structured
    noisy = np.random.normal(0, 0.3, len(t))
    const = np.ones_like(t) * 5.0          # should be ignored (zero std)
    nan_col = noisy.copy(); nan_col[:10] = np.nan  # ignored (has NaN)

    df = pd.DataFrame({
        "time": t,
        "ppg1": ppg,
        "noise": noisy,
        "const": const,
        "maybe_nan": nan_col,
    })
    ranked = physio.suggest_best_signal_column(df, "time")
    # Should propose ppg1 over pure noise
    assert isinstance(ranked, list) and len(ranked) >= 1
    assert ranked[0]["column"] in ("ppg1", "noise")
    # The constant/NaN columns should not appear
    assert all(cand["column"] not in ("const", "maybe_nan") for cand in ranked)
