# tests/test_physiological_callbacks_register_paths.py
# Focused coverage for: early callback registration blocks, unified callback body branches,
# time-input/slider helpers, extra UI callbacks, respiratory callbacks, and the
# main-signal plot's peak-annotation path.

import os
import sys
import types
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
import importlib.util
from dash.exceptions import PreventUpdate


# --- Load the module even if package imports aren't available ---
# HERE = os.path.dirname(os.path.abspath(__file__))
# ALT_MODULE = "/mnt/data/physiological_callbacks.py"
# if not os.path.exists(ALT_MODULE):
#     ALT_MODULE = os.path.abspath(os.path.join(HERE, "..", "..", "physiological_callbacks.py"))
SRC_DIR = os.path.abspath(os.path.join(os.getcwd(), "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

spec = importlib.util.spec_from_file_location("physio_mod", SRC_DIR)
from vitalDSP_webapp.callbacks.features import physiological_callbacks as physio
# physio = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(physio)


# --- Dummy Dash app to capture registered callbacks without needing a real Dash server ---
class DummyApp:
    def __init__(self):
        self.callbacks = []  # (decorator_args, decorator_kwargs, function)

    def callback(self, *decorator_args, **decorator_kwargs):
        def _decorator(func):
            self.callbacks.append((decorator_args, decorator_kwargs, func))
            return func
        return _decorator


# --- Fake data-service plumbing so the module's internal import works (no external deps) ---
def install_fake_data_service(fake_service_obj):
    # Ensure the nested packages exist
    sys.modules.setdefault("vitalDSP_webapp", types.ModuleType("vitalDSP_webapp"))
    sys.modules.setdefault("vitalDSP_webapp.services", types.ModuleType("vitalDSP_webapp.services"))
    sys.modules.setdefault("vitalDSP_webapp.services.data", types.ModuleType("vitalDSP_webapp.services.data"))

    data_service_mod = types.ModuleType("vitalDSP_webapp.services.data.data_service")
    def _get_data_service():
        return fake_service_obj
    data_service_mod.get_data_service = _get_data_service

    sys.modules["vitalDSP_webapp.services.data.data_service"] = data_service_mod


# --- Helper signals ---
def sine(fs=200, sec=12, f=1.2, noise=0.0, phase=0.0):
    t = np.arange(0, sec, 1.0/fs)
    s = np.sin(2*np.pi*f*t + phase)
    if noise:
        s = s + np.random.normal(0.0, noise, size=len(t))
    return t, s

def sine_with_spikes(fs=200, sec=8, f=1.5):
    t = np.arange(0, sec, 1.0/fs)
    s = np.sin(2*np.pi*f*t)
    # add a few tall spikes to guarantee peak detection (height > mean + std)
    for center in [1.0, 3.0, 5.0, 7.0]:
        idx = int(center * fs)
        if 0 <= idx < len(s):
            s[idx] += 5.0
    return t, s


# -------- Registration: ensures decorator lines execute, and lets us call inner fns --------

@pytest.fixture(scope="module")
def registered_callbacks():
    app = DummyApp()
    # call register fn (this also calls "register_additional_*" inside)
    physio.register_physiological_callbacks(app)
    # collect functions by __name__ for convenience
    fn_by_name = {}
    for _args, _kwargs, f in app.callbacks:
        fn_by_name.setdefault(f.__name__, []).append(f)
    return fn_by_name


# -------- Main unified callback (physiological) branches --------

class FakeService:
    def __init__(self, all_data, df=None, mapping=None):
        self._all = all_data
        self._df = df
        self._map = mapping

    def get_all_data(self):
        return self._all

    def get_column_mapping(self, _id):
        return self._map

    def get_data(self, _id):
        return self._df


def _set_trigger(button_id):
    class Ctx:
        triggered = [{"prop_id": f"{button_id}.n_clicks"}]
    # patch the module global used by the inner callback
    physio.callback_context = Ctx()


def test_unified_callback_not_on_page(registered_callbacks):
    # Install any fake service; callback returns early before using it
    install_fake_data_service(FakeService(all_data={"X": {"info": {"sampling_freq": 100}}}))
    fn = registered_callbacks["physiological_analysis_callback"][0]

    _set_trigger("physio-btn-update-analysis")
    out = fn(
        pathname="/", n_clicks=1, slider_value=[0, 10],
        nudge_m10=0, nudge_m1=0, nudge_p1=0, nudge_p10=0,
        start_time=None, end_time=None,
        signal_type=None,
        analysis_categories=None, hrv_options=None, morphology_options=None,
        advanced_features=None, quality_options=None, transform_options=None,
        advanced_computation=None, feature_engineering=None, preprocessing=None
    )
    # Expect "navigate to page" message and empty figures
    assert isinstance(out[0], go.Figure) and isinstance(out[2], go.Figure)
    assert isinstance(out[1], str) or hasattr(out[1], "children")


def test_unified_callback_no_data(registered_callbacks):
    install_fake_data_service(FakeService(all_data={}))
    fn = registered_callbacks["physiological_analysis_callback"][0]
    _set_trigger("physio-btn-update-analysis")

    out = fn(
        pathname="/physiological", n_clicks=1, slider_value=[0, 10],
        nudge_m10=0, nudge_m1=0, nudge_p1=0, nudge_p10=0,
        start_time=None, end_time=None, signal_type=None,
        analysis_categories=None, hrv_options=None, morphology_options=None,
        advanced_features=None, quality_options=None, transform_options=None,
        advanced_computation=None, feature_engineering=None, preprocessing=None
    )
    assert isinstance(out[0], go.Figure) and isinstance(out[2], go.Figure) and out[3] is None and out[4] is None


def test_unified_callback_no_column_mapping(registered_callbacks):
    fs = 100
    t, s = sine(fs=fs, sec=5, f=1.0)
    df = pd.DataFrame({"time": t, "PLETH": s})
    all_data = {"S1": {"info": {"sampling_freq": fs}}}
    install_fake_data_service(FakeService(all_data=all_data, df=df, mapping=None))
    fn = registered_callbacks["physiological_analysis_callback"][0]
    _set_trigger("physio-btn-update-analysis")

    out = fn(
        pathname="/physiological", n_clicks=1, slider_value=[0, 2],
        nudge_m10=0, nudge_m1=0, nudge_p1=0, nudge_p10=0,
        start_time=None, end_time=None, signal_type=None,
        analysis_categories=None, hrv_options=None, morphology_options=None,
        advanced_features=None, quality_options=None, transform_options=None,
        advanced_computation=None, feature_engineering=None, preprocessing=None
    )
    assert isinstance(out[0], go.Figure) and isinstance(out[2], go.Figure)


def test_unified_callback_empty_dataframe(registered_callbacks):
    fs = 100
    df = pd.DataFrame({"time": [], "PLETH": []})
    all_data = {"S1": {"info": {"sampling_freq": fs}}}
    install_fake_data_service(FakeService(all_data=all_data, df=df, mapping={"time": "time", "signal": "PLETH"}))
    fn = registered_callbacks["physiological_analysis_callback"][0]
    _set_trigger("physio-btn-update-analysis")
    out = fn(
        pathname="/physiological", n_clicks=1, slider_value=[0, 1],
        nudge_m10=0, nudge_m1=0, nudge_p1=0, nudge_p10=0,
        start_time=None, end_time=None, signal_type=None,
        analysis_categories=None, hrv_options=None, morphology_options=None,
        advanced_features=None, quality_options=None, transform_options=None,
        advanced_computation=None, feature_engineering=None, preprocessing=None
    )
    assert isinstance(out[0], go.Figure)


def test_unified_callback_success_ms_to_sec_and_auto_switch_column(registered_callbacks, monkeypatch):
    # time in ms to hit the ms→s branch, start with a low-variance PLETH to trigger suggestions/auto-switch
    fs = 100
    t_ms = np.arange(0, 20000, 10)  # 0..20s in 10ms steps
    t_sec = t_ms / 1000.0
    pleth = np.ones_like(t_ms, dtype=float) * 0.01  # tiny variance
    ppg1 = np.sin(2*np.pi*1.1*t_sec) + 0.05*np.random.randn(len(t_sec))

    df = pd.DataFrame({"time_ms": t_ms, "PLETH": pleth, "ppg1": ppg1})
    all_data = {"RECENT": {"info": {"sampling_freq": fs}}}
    mapping = {"time": "time_ms", "signal": "PLETH"}
    install_fake_data_service(FakeService(all_data=all_data, df=df, mapping=mapping))

    # keep analysis light but still exercise the path
    fn = registered_callbacks["physiological_analysis_callback"][0]
    _set_trigger("physio-btn-update-analysis")

    out = fn(
        pathname="/physiological", n_clicks=1, slider_value=[0, 10],
        nudge_m10=0, nudge_m1=0, nudge_p1=0, nudge_p10=0,
        start_time=0, end_time=10, signal_type="auto",
        analysis_categories=["morphology", "frequency", "statistical"],
        hrv_options=[], morphology_options=["peaks", "duration"],
        advanced_features=[], quality_options=[], transform_options=[],
        advanced_computation=[], feature_engineering=[], preprocessing=[]
    )
    main_plot, analysis_children, analysis_plot, store_data, store_feats = out
    assert isinstance(main_plot, go.Figure) and isinstance(analysis_plot, go.Figure)
    assert isinstance(store_data, dict) and isinstance(store_feats, dict)


def test_unified_callback_error_route(registered_callbacks, monkeypatch):
    # Force an exception inside the callback to hit the except-block return
    fs = 100
    t, s = sine(fs=fs, sec=6, f=1.0)
    df = pd.DataFrame({"time": t, "sig": s})
    all_data = {"S1": {"info": {"sampling_freq": fs}}}
    mapping = {"time": "time", "signal": "sig"}
    install_fake_data_service(FakeService(all_data=all_data, df=df, mapping=mapping))

    # Make perform_physiological_analysis_enhanced blow up
    monkeypatch.setattr(physio, "perform_physiological_analysis_enhanced", lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))
    fn = registered_callbacks["physiological_analysis_callback"][0]
    _set_trigger("physio-btn-update-analysis")
    out = fn(
        pathname="/physiological", n_clicks=1, slider_value=[0, 5],
        nudge_m10=0, nudge_m1=0, nudge_p1=0, nudge_p10=0,
        start_time=0, end_time=5, signal_type="ppg",
        analysis_categories=["morphology"], hrv_options=[], morphology_options=["peaks"],
        advanced_features=[], quality_options=[], transform_options=[],
        advanced_computation=[], feature_engineering=[], preprocessing=[]
    )
    # Error path returns empty fig + error text
    assert isinstance(out[0], go.Figure) and isinstance(out[2], go.Figure)


# -------- Time-input + time-slider range (registered versions) --------

def test_update_physio_time_inputs_registered_slider_and_nudges(registered_callbacks):
    fn = registered_callbacks["update_physio_time_inputs"][0]

    # Slider triggered -> passthrough
    class Ctx1: triggered = [{"prop_id": "physio-time-range-slider.value"}]
    physio.callback_context = Ctx1()
    start, end = fn([3, 13], 0, 0, 0, 0, 1, 11)
    assert (start, end) == (3, 13)

    # Nudge +1
    class Ctx2: triggered = [{"prop_id": "physio-btn-nudge-p1.n_clicks"}]
    physio.callback_context = Ctx2()
    start, end = fn(None, 0, 0, 0, 0, 1, 11)
    assert (start, end) == (2, 12)

    # Nudge -10 floors at 0
    class Ctx3: triggered = [{"prop_id": "physio-btn-nudge-m10.n_clicks"}]
    physio.callback_context = Ctx3()
    start, end = fn(None, 0, 0, 0, 0, 0, 10)
    assert (start, end) == (0, 10)


def test_update_physio_time_slider_range_registered(registered_callbacks):
    fn = registered_callbacks["update_physio_time_slider_range"][0]
    # No data -> default 100
    assert fn(None) == 100
    # Empty -> 100
    assert fn({"data": {}}) == 100
    # With data dict -> max(time)
    df = pd.DataFrame({"time": [0, 5, 10, 25], "v": [0.1, 0.2, 0.3, 0.4]})
    assert fn({"data": df.to_dict(orient="list")}) == 25


# -------- Extra UI callbacks under "register_additional_physiological_callbacks" --------

def test_toggle_morphology_options_and_export_button(registered_callbacks):
    toggle_vis = registered_callbacks["toggle_morphology_options_visibility"][0]
    assert toggle_vis(["morphology"]) == {"display": "block"}
    assert toggle_vis(["hrv"]) == {"display": "none"}

    toggle_export = registered_callbacks["toggle_export_button"][0]
    assert toggle_export(None) is True
    assert toggle_export({"x": 1}) is False


def test_update_additional_analysis_section_empty_and_filled(registered_callbacks):
    update_extra = registered_callbacks["update_additional_analysis_section"][0]
    empty = update_extra(None)
    assert hasattr(empty, "children")

    sample_features = {
        "hrv_metrics": {"mean_rr": 800, "lf_power": 10},
        "morphology_metrics": {"num_peaks": 12, "area_under_curve": 1.2},
        "quality_metrics": {"signal_quality_index": 0.9},
        "transform_metrics": {"fourier_peak": 1.1},
        "advanced_features_metrics": {"total_power": 5.0},
    }
    filled = update_extra(sample_features)
    assert hasattr(filled, "children")


# -------- Respiratory page callbacks --------

def test_resp_time_inputs_and_range(registered_callbacks):
    # time inputs mirror slider
    upd_inputs = registered_callbacks["update_resp_time_inputs"][0]
    assert upd_inputs([2, 12]) == (2, 12)
    assert upd_inputs(None) == (physio.no_update, physio.no_update)

    # time range pulls from service -> returns (min,max,[range])
    fs = 50
    t, s = sine(fs=fs, sec=20, f=0.3)
    df = pd.DataFrame({"t": t, "resp": s})
    all_data = {"R1": {"info": {"sampling_freq": fs}}}

    install_fake_data_service(FakeService(all_data=all_data, df=df, mapping={"time": "t", "signal": "resp"}))
    upd_range = registered_callbacks["update_resp_time_slider_range"][0]
    # not on /respiratory -> default (0,100,[0,10])
    assert upd_range("/") == (0, 100, [0, 10])
    # respiratory route -> compute duration
    minv, maxv, rng = upd_range("/respiratory")
    assert minv == 0 and maxv > 0 and isinstance(rng, list)


def test_respiratory_analysis_callback_happy_path(registered_callbacks):
    fs = 50
    t, s = sine(fs=fs, sec=30, f=0.33, noise=0.02)
    df = pd.DataFrame({"t": t, "resp": s})
    all_data = {"R1": {"info": {"sampling_freq": fs}}}
    install_fake_data_service(FakeService(all_data=all_data, df=df, mapping={"time": "t", "signal": "resp"}))

    # Simulate click
    class Ctx: triggered = [{"prop_id": "resp-analyze-btn.n_clicks"}]
    physio.callback_context = Ctx()

    fn = registered_callbacks["respiratory_analysis_callback"][0]
    out = fn(
        pathname="/respiratory",
        n_clicks=1,
        slider_value=[0, 10],
        nudge_m10=0, nudge_m1=0, nudge_p1=0, nudge_p10=0,
        start_time=0, end_time=10,
        signal_type="auto",
        estimation_methods=["peak","fft","autocorr"],
        advanced_options=["variability","power_bands"],
        preprocessing_options=["detrend","smooth"],
        low_cut=None, high_cut=None, min_breath_duration=None, max_breath_duration=None
    )
    assert isinstance(out[0], go.Figure) and isinstance(out[2], go.Figure)


# -------- Peak-annotation branch in main signal plot (height/interval arrows + HR text) --------

def test_create_physiological_signal_plot_peak_annotations():
    fs = 200
    t, s = sine_with_spikes(fs=fs, sec=8, f=1.5)
    fig = physio.create_physiological_signal_plot(t, s, "ppg", fs)
    assert isinstance(fig, go.Figure)
    # Should have at least the main line trace and some peak markers or annotations
    assert len(fig.data) >= 1
    # Annotations typically include interval arrows and a HR text in the corner
    assert hasattr(fig.layout, "annotations")
