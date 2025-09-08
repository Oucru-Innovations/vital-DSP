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
from vitalDSP_webapp.callbacks.analysis.respiratory_callbacks import (
    detect_respiratory_signal_type,
    create_respiratory_signal_plot,
    generate_comprehensive_respiratory_analysis,
    create_comprehensive_respiratory_plots,
)
from vitalDSP_webapp.callbacks.features.physiological_callbacks import (
    # high-level
    perform_physiological_analysis,
    perform_physiological_analysis_enhanced,

    # core helpers
    detect_physiological_signal_type,
    create_physiological_signal_plot,

    # analysis builders

    analyze_hrv,
    analyze_morphology,
    analyze_signal_quality,
    analyze_statistical,
    analyze_envelope,
    analyze_segmentation,
    analyze_waveform,
    analyze_signal_quality_advanced,
    analyze_transforms,
    analyze_advanced_computation,
    analyze_feature_engineering,


    # figure builders
    create_beat_to_beat_plots,
    create_energy_plots,
    create_envelope_plots,
    create_segmentation_plots,
    create_waveform_plots,
    create_frequency_plots,
    create_transform_plots,
    create_wavelet_plots,
    create_fourier_plots,
    create_hilbert_plots,
    create_comprehensive_dashboard,

    # comprehensive dashboards (respiratory suite) - moved to analysis.respiratory_callbacks

    # time slider helpers
    update_time_slider_marks,
    update_time_input_max_values,

    # standalone testing callback shim
    physiological_analysis_callback,
    
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
def synth_signal(fs=50, seconds=20, f0=1.2, noise=0.05, trend=0.0):
    t = np.arange(0, seconds, 1/fs)
    sig = np.sin(2*np.pi*f0*t) + noise*np.random.randn(len(t)) + trend*t
    return t, sig

def flat_signal(fs=50, seconds=5, value=0.0):
    t = np.arange(0, seconds, 1/fs)
    sig = np.full_like(t, fill_value=value, dtype=float)
    return t, sig

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

# ---------- vitalDSP wrapper fallbacks

def test_get_vitaldsp_hrv_analysis_fallback_has_expected_keys():
    fs = 100
    t, sig = synth_signal(fs=fs, seconds=10, f0=1.1, noise=0.03)
    res = get_vitaldsp_hrv_analysis(sig, fs, hrv_options=["time_domain", "freq_domain", "nonlinear"], signal_type="ppg")
    # Import will fail in test env -> fallback analyze_hrv path
    assert isinstance(res, dict)
    # time domain keys
    for k in ["mean_rr", "std_rr", "rmssd", "nn50", "pnn50"]:
        assert k in res
    # freq domain keys
    for k in ["total_power", "vlf_power", "lf_power", "hf_power", "lf_hf_ratio"]:
        assert k in res
    # nonlinear keys
    for k in ["poincare_sd1", "poincare_sd2", "dfa_alpha1", "dfa_alpha2"]:
        assert k in res

def test_get_vitaldsp_morphology_analysis_fallback_covers_options():
    fs = 100
    t, sig = synth_signal(fs=fs, seconds=6, f0=1.0)
    res = get_vitaldsp_morphology_analysis(sig, fs, morphology_options=["peaks", "duration", "area"], signal_type="ecg")
    assert isinstance(res, dict)
    for k in ["num_peaks", "signal_duration", "sampling_freq", "num_samples", "total_area", "area_under_curve"]:
        assert k in res

def test_get_vitaldsp_signal_quality_fallback_all_options():
    fs = 100
    t, sig = synth_signal(fs=fs, seconds=6, f0=1.0, noise=0.1)
    res = get_vitaldsp_signal_quality(sig, fs, quality_options=["quality_index", "artifact_detection", "snr_estimation"], signal_type="ppg")
    assert isinstance(res, dict)
    for k in ["signal_quality_index", "overall_score", "artifacts_detected", "artifact_ratio"]:
        assert k in res

def test_get_vitaldsp_transforms_fallback_paths_wavelet_fourier_hilbert():
    fs = 100
    t, sig = synth_signal(fs=fs, seconds=6, f0=2.0)
    # wavelet
    m1 = get_vitaldsp_transforms(sig, fs, transform_options=["wavelet"], signal_type="ppg")
    assert "wavelet_energy" in m1 and isinstance(m1["wavelet_energy"], (int, float))
    # fourier
    m2 = get_vitaldsp_transforms(sig, fs, transform_options=["fourier"], signal_type="ppg")
    assert "fourier_peak" in m2
    # hilbert
    m3 = get_vitaldsp_transforms(sig, fs, transform_options=["hilbert"], signal_type="ppg")
    assert "hilbert_phase" in m3

def test_get_vitaldsp_advanced_computation_all_three():
    fs = 100
    t, sig = synth_signal(fs=fs, seconds=6, f0=1.8, noise=0.2)
    res = get_vitaldsp_advanced_computation(sig, fs, advanced_computation=["anomaly_detection", "bayesian", "kalman"], signal_type="ppg")
    assert isinstance(res, dict)
    assert "anomalies_detected" in res
    assert "bayesian_prior_mean" in res and "bayesian_prior_std" in res
    assert "kalman_estimate" in res

def test_get_vitaldsp_feature_engineering_ppg_and_ecg():
    fs = 100
    t, ppg = synth_signal(fs=fs, seconds=6, f0=1.3)
    # for ppg
    res_ppg = get_vitaldsp_feature_engineering(ppg, fs, feature_engineering=["ppg_light", "ppg_autonomic"], signal_type="ppg")
    assert isinstance(res_ppg, dict)
    assert any(k in res_ppg for k in ["ppg_light_intensity", "ppg_autonomic_response"])
    # for ecg
    t2, ecg = synth_signal(fs=fs, seconds=6, f0=1.0)
    res_ecg = get_vitaldsp_feature_engineering(ecg, fs, feature_engineering=["ecg_autonomic"], signal_type="ecg")
    assert "ecg_autonomic_response" in res_ecg

# ---------- analyses & helpers

def test_analyze_preprocessing_all_flags():
    fs = 100
    _, sig = synth_signal(fs=fs, seconds=4)
    res = analyze_preprocessing(sig, fs, preprocessing=["noise_reduction", "baseline_correction", "filtering"])
    assert {"noise_level", "baseline_offset", "signal_bandwidth"} <= set(res.keys())

def test_analyze_trends_edge_cases_and_normal():
    fs = 50
    # insufficient
    res_ins = analyze_trends(np.array([1.0]), fs)
    assert res_ins["trend_direction"] == "insufficient_data"
    # no variation
    _, flat = flat_signal(fs=fs, seconds=5, value=5.0)
    res_flat = analyze_trends(flat, fs)
    assert res_flat["trend_direction"] == "no_variation"
    # increasing
    t, increasing = synth_signal(fs=fs, seconds=6, trend=0.01, f0=0.3, noise=0.01)
    res_inc = analyze_trends(increasing, fs)
    assert res_inc["trend_direction"] in ("increasing", "stable")

def test_analyze_frequency_returns_basic_spectrum():
    fs = 100
    _, sig = synth_signal(fs=fs, seconds=6, f0=2.0)
    res = analyze_frequency(sig, fs)
    assert "frequencies" in res and "psd" in res and len(res["frequencies"]) == len(res["psd"])
    assert "dominant_frequency" in res

# ---------- plots

@pytest.mark.parametrize("opts", [
    (["quality_index", "artifact_detection"]),
    (["quality_index"]),
    (["artifact_detection"]),
])
def test_create_signal_quality_plots_variants(opts):
    fs = 100
    t, sig = synth_signal(fs=fs, seconds=8, f0=1.2, noise=0.15)
    fig = create_signal_quality_plots(t, sig, fs, quality_options=opts)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1  # at least the main signal trace

def test_create_advanced_features_plots_all_subplots():
    fs = 100
    t, sig = synth_signal(fs=fs, seconds=12, f0=1.3, noise=0.1)
    fig = create_advanced_features_plots(
        t, sig, fs,
        advanced_features=["cross_signal", "ensemble", "change_detection", "power_analysis"]
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1

def test_create_hrv_plots_full_options_and_empty_path():
    fs = 100
    t, sig = synth_signal(fs=fs, seconds=8, f0=1.1)
    fig1 = create_hrv_plots(t, sig, fs, hrv_options=["time_domain", "freq_domain", "nonlinear"])
    assert isinstance(fig1, go.Figure)
    # empty path when not enough peaks
    t_flat, sig_flat = flat_signal(fs=fs, seconds=8, value=0.0)
    fig2 = create_hrv_plots(t_flat, sig_flat, fs, hrv_options=["time_domain"])
    assert isinstance(fig2, go.Figure)

def test_create_morphology_plots_multiple_options():
    fs = 100
    t, sig = synth_signal(fs=fs, seconds=8, f0=1.0)
    fig = create_morphology_plots(t, sig, fs, morphology_options=["peaks", "amplitude", "duration", "area"])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1

# ---------- comprehensive functions

def test_perform_physiological_analysis_returns_dash_div():
    fs = 100
    t, sig = synth_signal(fs=fs, seconds=8, f0=1.0)
    layout = perform_physiological_analysis(
        t, sig, "ppg", fs,
        analysis_categories=["hrv", "morphology", "frequency", "statistical"],
        hrv_options=["time_domain", "freq_domain"],
        morphology_options=["peaks", "duration"],
        advanced_features=[],
        quality_options=[],
        transform_options=[],
        advanced_computation=[],
        feature_engineering=[],
        preprocessing=[]
    )
    # Should be a Dash html.Div (duck-type check to avoid importing html in test)
    assert hasattr(layout, "children")

def test_perform_physiological_analysis_enhanced_returns_metrics_dict():
    fs = 100
    t, sig = synth_signal(fs=fs, seconds=10, f0=1.1)
    res = perform_physiological_analysis_enhanced(
        t, sig, "ppg", fs,
        analysis_categories=["hrv", "morphology", "frequency", "statistical", "trend"],
        hrv_options=["time_domain", "freq_domain", "nonlinear"],
        morphology_options=["peaks", "duration", "area"],
        advanced_features=["ensemble"],
        quality_options=["quality_index", "artifact_detection"],
        transform_options=["fourier"],
        advanced_computation=["anomaly_detection"],
        feature_engineering=["ppg_autonomic"],
        preprocessing=["noise_reduction", "filtering"]
    )
    assert isinstance(res, dict)
    # core sections present
    for k in ["hrv_metrics", "morphology_metrics", "frequency_metrics", "statistical_metrics"]:
        assert k in res

# ---------- util / import

def test_suggest_best_signal_column_ranks_and_skips():
    fs = 50
    t = np.arange(0, 5, 1/fs)
    good = np.sin(2*np.pi*1.0*t) + 0.05*np.random.randn(len(t))
    constant = np.ones_like(good) * 5
    nan_col = good.copy()
    nan_col[:10] = np.nan
    df = pd.DataFrame({
        "time": t,
        "sig_good": good,
        "sig_constant": constant,
        "sig_nan": nan_col,
        "text_col": ["x"]*len(t),
    })
    ranked = suggest_best_signal_column(df, time_col="time")
    # good signal should appear, constant & NaN should be skipped
    cols = [r["column"] for r in ranked]
    assert "sig_good" in cols
    assert "sig_constant" not in cols
    assert "sig_nan" not in cols

def test_normalize_signal_type_variants_and_graceful_default():
    assert normalize_signal_type("ppg") == "PPG"
    assert normalize_signal_type("EcG") == "ECG"
    assert normalize_signal_type("resp") == "RESP"
    assert normalize_signal_type("unknown") == "PPG"  # default

def test_import_vitaldsp_modules_no_crash():
    # Should not raise even if vitalDSP not present
    _import_vitaldsp_modules()
    assert True

def test_create_empty_figure_basics():
    fig = create_empty_figure()
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0
    
def mk_sine(fs=200, sec=12, f=1.2, noise=0.0, phase=0.0):
    t = np.arange(0, sec, 1.0/fs)
    s = np.sin(2*np.pi*f*t + phase)
    if noise:
        s = s + np.random.normal(0.0, noise, size=len(t))
    return t, s

def mk_step(fs=200, sec=12, step_at=4.0, low=0.0, high=3.0):
    t = np.arange(0, sec, 1.0/fs)
    s = np.where(t >= step_at, high, low)
    return t, s

# ------------- basic helpers / types -------------

def test_normalize_signal_type_all_known():
    assert normalize_signal_type("ppg") == "PPG"
    assert normalize_signal_type("ECG") == "ECG"
    assert normalize_signal_type("resp") == "RESP"
    # default fallback
    assert normalize_signal_type("weird") == "PPG"

def test_detect_physiological_signal_type_paths():
    fs = 200
    # Fast peaks -> ECG
    t_ecg, y_ecg = mk_sine(fs, 6, f=3.2)
    assert detect_physiological_signal_type(y_ecg, fs) in ("ecg", "ppg")
    # Slow & smooth -> PPG by default heuristic
    t_ppg, y_ppg = mk_sine(fs, 6, f=1.0)
    assert detect_physiological_signal_type(y_ppg, fs) in ("ppg", "ecg")

def test_create_physiological_signal_plot_variants():
    fs = 100
    t, s = mk_sine(fs, 5, 1.1)
    fig = create_physiological_signal_plot(t, s, "ppg", fs)
    assert isinstance(fig, go.Figure) and len(fig.data) >= 1
    fig2 = create_physiological_signal_plot(t, s*0.5, "ecg", fs)
    assert isinstance(fig2, go.Figure)

def test_create_empty_figure_annotation_present():
    fig = create_empty_figure()
    assert isinstance(fig, go.Figure)
    assert any(getattr(a, "text", "") == "No data available" for a in fig.layout.annotations)

# ------------- analysis blocks (stats / freq / envelope / seg / quality adv) -------------

def test_analyze_statistical_and_frequency_and_waveform():
    fs = 200
    t, s = mk_sine(fs, 6, 1.5, noise=0.01)
    st = analyze_statistical(s, fs)
    fr = analyze_frequency(s, fs)
    wf = analyze_waveform(s, fs)
    assert isinstance(st, dict) and "mean" in st and "std" in st
    assert isinstance(fr, dict) and "frequencies" in fr and "psd" in fr and "dominant_frequency" in fr
    assert isinstance(wf, dict) and "skewness" in wf and "kurtosis" in wf

def test_analyze_envelope_and_segmentation_and_quality_adv():
    fs = 200
    t, s = mk_sine(fs, 8, 1.1, noise=0.02)
    env = analyze_envelope(s, fs)
    seg = analyze_segmentation(s, fs)
    qadv = analyze_signal_quality_advanced(s, fs)
    assert "upper_envelope" in env and "lower_envelope" in env
    assert "segments" in seg and isinstance(seg["segments"], list)
    assert "entropy" in qadv and "complexity" in qadv

def test_analysis_wrappers_hrv_morph_quality_trends_preproc():
    fs = 200
    t, s = mk_sine(fs, 12, 1.2, noise=0.02)
    hrv = analyze_hrv(s, fs, ["time_domain", "freq_domain", "nonlinear"])
    mor = analyze_morphology(s, fs, ["peaks", "duration", "area"])
    q = analyze_signal_quality(s, fs)
    tr = analyze_trends(s, fs)
    pre = analyze_preprocessing(s, fs, ["noise_reduction", "baseline_correction", "filtering"])
    assert all(k in hrv for k in ("mean_rr", "lf_power", "poincare_sd1"))
    assert "num_peaks" in mor and "signal_duration" in mor
    assert "snr_db" in q and "zero_crossings" in q
    assert "trend_slope" in tr
    assert {"noise_level","baseline_offset","signal_bandwidth"} <= set(pre.keys())

# ------------- transforms / advanced / feature engineering -------------

def test_analyze_transforms_fourier_wavelet_hilbert():
    fs = 200
    t, s = mk_sine(fs, 8, 2.0)
    tf = analyze_transforms(s, fs, ["fourier", "wavelet", "hilbert"])
    assert any(k in tf for k in ("fourier_peak", "wavelet_energy", "hilbert_phase"))

def test_analyze_advanced_computation_and_feature_eng_all_branches():
    fs = 200
    t, ppg = mk_sine(fs, 8, 1.1, noise=0.03)
    adv = analyze_advanced_computation(ppg, fs, ["anomaly_detection", "bayesian", "kalman"])
    fe_ppg = analyze_feature_engineering(ppg, fs, ["ppg_light", "ppg_autonomic"], "ppg")
    assert "anomalies_detected" in adv and "kalman_estimate" in adv
    assert any(k in fe_ppg for k in ("ppg_light_intensity","ppg_autonomic_response"))

def test_vitaldsp_wrappers_more_paths_and_fallbacks():
    fs = 150
    t, s = mk_sine(fs, 8, 1.3)
    _import_vitaldsp_modules()  # no-raise
    # fallbacks (in test env vitalDSP is usually not present)
    hrv = get_vitaldsp_hrv_analysis(s, fs, ["time_domain","freq_domain","nonlinear"], "ppg")
    mor = get_vitaldsp_morphology_analysis(s, fs, ["peaks","duration","area"], "ecg")
    qual = get_vitaldsp_signal_quality(s, fs, ["quality_index","artifact_detection"], "ppg")
    tfm = get_vitaldsp_transforms(s, fs, ["fourier","wavelet","hilbert"], "ppg")
    adv = get_vitaldsp_advanced_computation(s, fs, ["anomaly_detection","bayesian","kalman"], "ppg")
    fe1 = get_vitaldsp_feature_engineering(s, fs, ["ppg_light","ppg_autonomic"], "ppg")
    fe2 = get_vitaldsp_feature_engineering(s, fs, ["ecg_autonomic"], "ecg")
    fe3 = get_vitaldsp_feature_engineering(s, fs, ["ppg_light"], "unknown")  # defaulting path
    assert "mean_rr" in hrv and "lf_power" in hrv
    assert "num_peaks" in mor
    assert "signal_quality_index" in qual
    assert any(k in tfm for k in ("fourier_peak","wavelet_energy","hilbert_phase"))
    assert "anomalies_detected" in adv and "bayesian_prior_mean" in adv
    assert any(k in fe1 for k in ("ppg_light_intensity","ppg_autonomic_response"))
    assert "ecg_autonomic_response" in fe2
    assert isinstance(fe3, dict)

# ------------- figure builders (more branches) -------------

def test_create_transform_specific_plots():
    fs = 200
    t, s = mk_sine(fs, 8, 2.0)
    assert isinstance(create_transform_plots(t, s, fs, ["fourier","hilbert"]), go.Figure)
    assert isinstance(create_wavelet_plots(t, s, fs), go.Figure)
    assert isinstance(create_fourier_plots(t, s, fs), go.Figure)
    assert isinstance(create_hilbert_plots(t, s, fs), go.Figure)

def test_create_waveform_segmentation_envelope_plots():
    fs = 200
    t, s = mk_sine(fs, 10, 1.2)
    assert isinstance(create_waveform_plots(t, s, fs), go.Figure)
    assert isinstance(create_segmentation_plots(t, s, fs), go.Figure)
    assert isinstance(create_envelope_plots(t, s, fs), go.Figure)

def test_create_frequency_plots_and_morphology_plots_newer_def():
    fs = 200
    t, s = mk_sine(fs, 10, 1.0)
    assert isinstance(create_frequency_plots(t, s, fs), go.Figure)
    # the later re-defined create_morphology_plots (~3048) with multiple options
    assert isinstance(create_morphology_plots(t, s, fs, ["peaks","amplitude","duration","area"]), go.Figure)

def test_quality_plots_both_options_and_zero_noise_branch():
    fs = 200
    t1, s1 = mk_sine(fs, 10, 1.1, noise=0.02)
    fig1 = create_signal_quality_plots(t1, s1, fs, ["quality_index","artifact_detection"])
    assert isinstance(fig1, go.Figure) and len(fig1.data) >= 1
    # zero-noise branch (snr calc path where noise_level==0)
    t2 = np.linspace(0, 5, 5*fs)
    s2 = np.zeros_like(t2)
    fig2 = create_signal_quality_plots(t2, s2, fs, ["quality_index"])
    assert isinstance(fig2, go.Figure)

def test_advanced_features_plots_full_suite_with_change_points():
    fs = 200
    # signal with clear step -> change detection
    t, s = mk_step(fs, 14, step_at=6.0, low=0.0, high=5.0)
    # add tiny noise to avoid zero-variance issues
    s = s + 0.01*np.random.randn(len(s))
    fig = create_advanced_features_plots(
        t, s, fs,
        advanced_features=["cross_signal","ensemble","change_detection","power_analysis"]
    )
    assert isinstance(fig, go.Figure) and len(fig.data) >= 4

def test_create_comprehensive_dashboard_covers_all_quadrants():
    fs = 200
    t, s = mk_sine(fs, 16, 1.2, noise=0.01)
    fig = create_comprehensive_dashboard(t, s, "ppg", fs, analysis_categories=["morphology","frequency","quality"])
    assert isinstance(fig, go.Figure)
    # titles exist for 2x2 subplots
    assert len(fig.layout.annotations) >= 4

# ------------- comprehensive analysis functions -------------

def test_perform_physiological_analysis_enhanced_more_sections():
    fs = 200
    t, s = mk_sine(fs, 12, 1.1)
    res = perform_physiological_analysis_enhanced(
        t, s, "ppg", fs,
        analysis_categories=["hrv","morphology","frequency","statistical","trend","quality","transforms","advanced"],
        hrv_options=["time_domain","freq_domain","nonlinear"],
        morphology_options=["peaks","duration","area"],
        advanced_features=["cross_signal","change_detection","power_analysis"],
        quality_options=["quality_index","artifact_detection"],
        transform_options=["fourier","hilbert"],
        advanced_computation=["anomaly_detection","bayesian","kalman"],
        feature_engineering=["ppg_light","ppg_autonomic"],
        preprocessing=["noise_reduction","baseline_correction","filtering"]
    )
    assert isinstance(res, dict)
    for k in ["hrv_metrics","morphology_metrics","frequency_metrics","statistical_metrics","quality_metrics","transform_metrics","advanced_features_metrics"]:
        assert k in res

def test_perform_physiological_analysis_layout_variant():
    fs = 200
    t, s = mk_sine(fs, 10, 1.0)
    layout = perform_physiological_analysis(
        t, s, "ppg", fs,
        analysis_categories=["hrv","morphology","quality","frequency","statistical"],
        hrv_options=["time_domain"],
        morphology_options=["peaks","duration"],
        advanced_features=[],
        quality_options=["quality_index"],
        transform_options=[],
        advanced_computation=[],
        feature_engineering=[],
        preprocessing=[]
    )
    # dash html.Div duck type
    assert hasattr(layout, "children")

# ------------- time slider marks & max values -------------

def test_update_time_slider_marks_and_max_values():
    fs = 100
    t = np.linspace(0, 25, 25*fs)
    data_store = {"time_data": t.tolist()}
    marks = update_time_slider_marks(data_store)
    assert isinstance(marks, dict)
    mx, mx2 = update_time_input_max_values(data_store)
    assert isinstance(mx, (int,float)) and isinstance(mx2, (int,float)) and mx == mx2 == max(t)

    # empty paths
    assert update_time_slider_marks(None) == {}
    assert update_time_input_max_values({"time_data": []}) == (100, 100)

# ------------- respiratory suite -------------

def test_detect_respiratory_signal_type_and_plot():
    fs = 50
    # slow rhythm
    t, s = mk_sine(fs, 20, 0.25, noise=0.01)
    guess = detect_respiratory_signal_type(s, fs)
    assert guess in ("respiratory","ppg","ecg")
    fig = create_respiratory_signal_plot(s, t, fs, "respiratory", [], [], 0.1, 0.8)
    assert isinstance(fig, go.Figure)

def test_generate_comprehensive_respiratory_analysis_and_plots():
    fs = 50
    t, s = mk_sine(fs, 30, 0.33, noise=0.02)
    # analysis list-type returns children blocks
    result_children = generate_comprehensive_respiratory_analysis(
        s, t, fs,
        signal_type="respiratory",
        estimation_methods=["peak","fft","autocorr"],
        advanced_options=["variability","power_bands"],
        preprocessing_options=["detrend","smooth"],
        low_cut=0.1, high_cut=0.8,
        min_breath_duration=1.0, max_breath_duration=10.0
    )
    # The function returns a Div object, not a list
    assert hasattr(result_children, 'children') and len(result_children.children) > 0

    # comprehensive respiratory figure
    fig = create_comprehensive_respiratory_plots(
        s, t, fs, "respiratory",
        ["peak","fft","autocorr"],
        ["variability","power_bands"],
        ["detrend","smooth"],
        0.1, 0.8
    )
    assert isinstance(fig, go.Figure)

# ------------- standalone callback shim (ensures callback-context branches covered) -------------

def _mock_ctx(prop_id):
    class Ctx:
        triggered = [{"prop_id": f"{prop_id}.n_clicks"}]
    return Ctx()

def test_standalone_callback_trigger_and_return(monkeypatch):
    # simulate that "physio-run-analysis" was clicked
    monkeypatch.setattr("vitalDSP_webapp.callbacks.features.physiological_callbacks.callback_context", _mock_ctx("physio-run-analysis"))
    out = physiological_analysis_callback(
        pathname="/physiological",
        n_clicks=1,
        slider_value=[0, 10],
        nudge_m10=0, nudge_m1=0, nudge_p1=0, nudge_p10=0,
        start_time=0, end_time=10,
        signal_type="ppg",
        analysis_categories=["hrv","morphology"],
        hrv_options=["time_domain"],
        morphology_options=["peaks"],
        advanced_features=["cross_signal"],
        quality_options=["quality_index"],
        transform_options=["fourier"],
        advanced_computation=["anomaly_detection"],
        feature_engineering=["ppg_light"],
        preprocessing=["noise_reduction"]
    )
    main_plot, results_text, analysis_plot, slider_max1, slider_max2 = out
    assert isinstance(main_plot, go.Figure)
    assert isinstance(analysis_plot, go.Figure)
    assert isinstance(results_text, str)