"""Tests for the segment-quality callbacks module.

Exercises:

* Pure helpers (`_nearest_length`, `_make_timeline_figure`,
  `_headline_text`, `_checklist_options_and_value`).
* The callbacks registered by ``register_segment_quality_callbacks``,
  captured via a MagicMock decorator and invoked directly.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

from vitalDSP_webapp.callbacks.analysis import segment_quality_callbacks as sqc
from vitalDSP_webapp.callbacks.analysis.segment_quality_callbacks import (
    _checklist_options_and_value,
    _headline_text,
    _make_timeline_figure,
    _nearest_length,
    register_segment_quality_callbacks,
)
from vitalDSP.signal_quality_assessment import load_rule_dict


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestNearestLength:
    def test_snaps_to_dropdown_options(self):
        # Window=60 → snap to 60s option exactly
        assert _nearest_length(60) == 60
        # Window=45 → equidistant between 30 and 60; min() ties pick 30
        assert _nearest_length(45) in (30, 60)
        # Window=7 → snap to 5 (closer than 10)
        assert _nearest_length(7) == 5
        # Window=12 → snap to 10
        assert _nearest_length(12) == 10

    def test_fallback_for_missing_or_invalid(self):
        assert _nearest_length(None) == 30
        assert _nearest_length(0) == 30
        assert _nearest_length(-5) == 30
        assert _nearest_length("nonsense") == 30


class TestTimelineFigure:
    def test_empty_decisions_renders_placeholder(self):
        fig = _make_timeline_figure([])
        assert isinstance(fig, go.Figure)
        assert fig.layout.annotations
        assert "No segments" in fig.layout.annotations[0].text

    def test_colours_by_decision(self):
        decisions = [
            {"decision": "accept"},
            {"decision": "reject"},
            {"decision": "accept"},
        ]
        fig = _make_timeline_figure(decisions)
        assert len(fig.data) == 1
        bar = fig.data[0]
        colors = list(bar.marker.color)
        assert colors[0] == sqc._ACCEPT_COLOR
        assert colors[1] == sqc._REJECT_COLOR
        assert colors[2] == sqc._ACCEPT_COLOR
        # Three cells
        assert list(bar.x) == [0, 1, 2]

    def test_unknown_decision_gets_grey(self):
        fig = _make_timeline_figure([{"decision": "weird"}])
        bar = fig.data[0]
        assert list(bar.marker.color)[0] == sqc._UNKNOWN_COLOR


class TestHeadlineText:
    def test_empty_returns_prompt(self):
        node = _headline_text([], 0, 30)
        # plain Span with text inside `children`
        assert "Apply Filter" in str(node.children)

    def test_counts_accepted(self):
        decisions = [
            {"decision": "accept"}, {"decision": "accept"},
            {"decision": "reject"}, {"decision": "accept"},
        ]
        node = _headline_text(decisions, 4, 30)
        text = str(node.children)
        assert "Accepted: 3/4" in text

    def test_neutral_decisions_no_accept_reject_count(self):
        decisions = [{"decision": "neutral"} for _ in range(5)]
        node = _headline_text(decisions, 5, 30)
        text = str(node.children)
        assert "5 segments" in text
        assert "Accepted" not in text  # no accept/reject count


class TestChecklistOptionsAndValue:
    @pytest.fixture
    def clean_sqi_df(self):
        rng = np.random.default_rng(0)
        n = 50
        return pd.DataFrame({
            "kurtosis_sqi": 2.5 + 0.1 * rng.standard_normal(n),
            "snr_sqi": 10.0 + 1.0 * rng.standard_normal(n),
            "perfusion_sqi": 1500.0 + 50.0 * rng.standard_normal(n),
            "skewness_sqi": 0.5 + 0.2 * rng.standard_normal(n),
            "entropy_sqi": 2.0 + 0.1 * rng.standard_normal(n),
        })

    def test_empty_df_yields_no_data(self):
        opts, val, skipped, summary = _checklist_options_and_value(
            pd.DataFrame(), {}, "PPG", None,
        )
        assert opts == []
        assert val == []
        assert summary == "no data"

    def test_defaults_used_when_no_current_value(self, clean_sqi_df):
        rule_dict = load_rule_dict("PPG")
        opts, val, _, summary = _checklist_options_and_value(
            clean_sqi_df, rule_dict, "PPG", None,
        )
        # value should be a non-empty subset of DEFAULT_SEGMENT_SQIS for PPG
        assert val
        from vitalDSP.signal_quality_assessment import DEFAULT_SEGMENT_SQIS
        for name in val:
            assert name in DEFAULT_SEGMENT_SQIS["PPG"]
        assert "active" in summary

    def test_current_value_preserved_if_still_usable(self, clean_sqi_df):
        rule_dict = load_rule_dict("PPG")
        # User had selected just kurtosis_sqi; it's usable → keep it
        opts, val, _, _ = _checklist_options_and_value(
            clean_sqi_df, rule_dict, "PPG", ["kurtosis_sqi"],
        )
        assert val == ["kurtosis_sqi"]

    def test_current_value_dropped_if_unusable(self, clean_sqi_df):
        rule_dict = load_rule_dict("PPG")
        # Force a column to be degenerate (constant) → unusable
        df = clean_sqi_df.copy()
        df["skewness_sqi"] = 0.5
        # User had only the unusable one → falls back to defaults
        opts, val, skipped, _ = _checklist_options_and_value(
            df, rule_dict, "PPG", ["skewness_sqi"],
        )
        assert "skewness_sqi" not in val
        # And the degenerate column is mentioned in `skipped`
        assert "skewness_sqi" in skipped


# ---------------------------------------------------------------------------
# Registered callbacks — capture via mock decorator and invoke directly
# ---------------------------------------------------------------------------


class _CallbackCapturer:
    """Records each ``@app.callback`` invocation and exposes its func."""

    def __init__(self):
        self.captured = []
        self.app = MagicMock()
        self.app.callback = self._decorator

    def _decorator(self, *args, **kwargs):
        def wrap(func):
            self.captured.append((args, kwargs, func))
            return func
        return wrap

    def by_name(self, name):
        for _, _, func in self.captured:
            if func.__name__ == name:
                return func
        raise KeyError(f"No callback named {name!r} captured.")


@pytest.fixture
def reg():
    cap = _CallbackCapturer()
    register_segment_quality_callbacks(cap.app)
    return cap


class TestRegisteredCallbacks:
    def test_toggle_slider_visibility(self, reg):
        toggle = reg.by_name("toggle_slider_visibility")
        tune_style, q_style = toggle("tune")
        assert tune_style == {"display": "block"}
        assert q_style == {"display": "none"}
        tune_style, q_style = toggle("quantile")
        assert tune_style == {"display": "none"}
        assert q_style == {"display": "block"}
        tune_style, q_style = toggle("manual")
        assert tune_style == {"display": "none"}
        assert q_style == {"display": "none"}

    def _patched_data_service(self, monkeypatch, *, signal, fs, chain=None, column="ppg"):
        """Install a stand-in enhanced_data_service for compute_sqis tests.

        The callback now reads the whole recording from the data
        service rather than from a Dash store, so tests need a service
        that returns a known DataFrame + filter_info.
        """
        from vitalDSP_webapp.services.data import enhanced_data_service as eds_mod

        df = pd.DataFrame({"time": np.arange(len(signal)) / fs, column: signal})

        svc = MagicMock()
        svc.get_all_data.return_value = {"data1": object()}
        svc.get_data.return_value = df
        svc.get_data_info.return_value = {
            "sampling_freq": fs,
            "signal_type": "PPG",
        }
        svc.get_column_mapping.return_value = {"signal": column, "time": "time"}
        svc.get_filter_info.return_value = {"chain": chain or []}
        svc.get_filtered_data.return_value = None
        monkeypatch.setattr(eds_mod, "get_enhanced_data_service", lambda: svc)
        return svc

    def test_compute_sqis_reads_full_signal_from_data_service(
        self, reg, monkeypatch,
    ):
        compute = reg.by_name("compute_sqis")
        sig = np.sin(np.linspace(0, 60, 6000)) + 1.0
        self._patched_data_service(monkeypatch, signal=sig, fs=100)
        # compute_sqis(pathname, _filter_apply_signal, segment_length, overlap_pct)
        records, milestones, filtered_payload = compute("/filtering", None, 30, 0, True)
        assert isinstance(records, list)
        # 60s / 30s = 2 segments
        assert len(records) == 2
        assert len(milestones) == 2
        for ms in milestones:
            assert {"start_idx", "end_idx", "t_start", "t_end"} <= ms.keys()
        assert filtered_payload["n_samples"] == 6000
        assert filtered_payload["sampling_freq"] == 100.0

    def test_compute_sqis_no_data_in_service_prevents(self, reg, monkeypatch):
        compute = reg.by_name("compute_sqis")
        from vitalDSP_webapp.services.data import enhanced_data_service as eds_mod
        svc = MagicMock()
        svc.get_all_data.return_value = {}
        monkeypatch.setattr(eds_mod, "get_enhanced_data_service", lambda: svc)
        with pytest.raises(PreventUpdate):
            compute("/filtering", None, 30, 0, True)

    def test_compute_sqis_scoring_off_skips_sqi_compute(self, reg, monkeypatch):
        """Scoring OFF: just produce milestones, no SQI table, no chain replay."""
        compute = reg.by_name("compute_sqis")
        sig = np.sin(np.linspace(0, 60, 6000)) + 1.0
        svc = self._patched_data_service(monkeypatch, signal=sig, fs=100)
        # patch apply_filter_chain so we can prove it was NOT called
        from vitalDSP_webapp.callbacks.analysis import signal_filtering_callbacks as sfc
        called = []
        monkeypatch.setattr(sfc, "apply_filter_chain", lambda *a, **k: called.append(1) or a[0])
        records, milestones, filtered = compute("/filtering", None, 30, 0, False)
        # No SQI table, no whole-signal filtered cache.
        assert records is None
        assert filtered is None
        # Milestones still produced: 60s / 30s = 2.
        assert len(milestones) == 2
        # apply_filter_chain was NOT invoked.
        assert not called

    def test_compute_sqis_replays_saved_chain(self, reg, monkeypatch):
        """When a chain is saved, compute_sqis calls apply_filter_chain
        with the saved stages and the full original signal."""
        compute = reg.by_name("compute_sqis")
        sig = np.sin(np.linspace(0, 60, 6000)) + 1.0
        chain = [{
            "family": "traditional",
            "iterations": 1,
            "params": {
                "filter_family": "butter",
                "filter_response": "lowpass",
                "low_freq": 0.5,
                "high_freq": 20,
                "filter_order": 4,
            },
        }]
        self._patched_data_service(monkeypatch, signal=sig, fs=100, chain=chain)

        # Patch apply_filter_chain to a sentinel that mutates the
        # signal in a way we can detect.  This avoids any flakiness
        # from real-filter numerics and conclusively proves the chain
        # branch executed.
        from vitalDSP_webapp.callbacks.analysis import signal_filtering_callbacks as sfc
        chain_calls = []
        def fake_apply_chain(signal_arr, fs, signal_type, chain_arg, logger=None):
            chain_calls.append({
                "fs": fs, "signal_type": signal_type,
                "chain_len": len(chain_arg), "n_samples": len(signal_arr),
            })
            return np.asarray(signal_arr, dtype=float) * 2.0  # detectable change
        monkeypatch.setattr(sfc, "apply_filter_chain", fake_apply_chain)

        records, milestones, filtered = compute("/filtering", None, 30, 0, True)
        assert len(records) == 2
        # apply_filter_chain was called exactly once with the right args.
        assert len(chain_calls) == 1
        assert chain_calls[0]["fs"] == 100.0
        assert chain_calls[0]["chain_len"] == 1
        assert chain_calls[0]["n_samples"] == 6000
        # The cached filtered signal is the 2x input from our fake.
        np.testing.assert_allclose(filtered["signal"], (sig * 2.0).tolist())

    def test_reclassify_basic(self, reg):
        reclass = reg.by_name("reclassify")
        # Synthetic clean batch with EVERY column the bundled PPG rule
        # dict knows about — otherwise manual-mode rules whose column
        # is missing produce NaN and reject the segment.
        rng = np.random.default_rng(0)
        n = 20
        df = pd.DataFrame({
            "kurtosis_sqi": 2.5 + 0.01 * rng.standard_normal(n),
            "snr_sqi": 10.0 + 0.5 * rng.standard_normal(n),
            "perfusion_sqi": 1500.0 + 50.0 * rng.standard_normal(n),
            "skewness_sqi": 0.5 + 0.05 * rng.standard_normal(n),
            "entropy_sqi": 2.0 + 0.05 * rng.standard_normal(n),
            "zero_crossing_sqi": 0.01 + 0.001 * rng.standard_normal(n),
            "peak_to_peak_amplitude_sqi": 50.0 + 5.0 * rng.standard_normal(n),
        })
        payload = df.to_dict("records")
        milestones = [{"start_idx": i * 100, "end_idx": (i + 1) * 100,
                       "t_start": i, "t_end": i + 1} for i in range(n)]
        decisions, fig, headline = reclass(
            payload, milestones, "manual", 0.90, 0.05, None, "PPG", 30,
        )
        assert isinstance(decisions, list)
        assert len(decisions) == n
        # Manual mode on clean batch → most accepted
        accepted = sum(1 for d in decisions if d["decision"] == "accept")
        assert accepted >= n - 2
        assert isinstance(fig, go.Figure)

    def test_reclassify_scoring_off_emits_neutral(self, reg):
        """When sqi_payload is None but milestones exist, emit neutral decisions."""
        reclass = reg.by_name("reclassify")
        milestones = [{"start_idx": 0, "end_idx": 100, "t_start": 0, "t_end": 1}] * 4
        decisions, fig, headline = reclass(
            None, milestones, "tune", 0.9, 0.05, None, "PPG", 30,
        )
        assert len(decisions) == 4
        assert all(d["decision"] == "neutral" for d in decisions)
        assert isinstance(fig, go.Figure)

    def test_reclassify_empty_no_milestones_returns_empty(self, reg):
        """Both None → empty timeline (no PreventUpdate; clean reset)."""
        reclass = reg.by_name("reclassify")
        decisions, fig, headline = reclass(
            None, None, "tune", 0.9, 0.05, None, "PPG", 30,
        )
        assert decisions == []

    def test_render_quality_timeline_empty(self, reg):
        render = reg.by_name("render_quality_timeline")
        fig, msg = render(None)
        assert isinstance(fig, go.Figure)
        assert "Apply a filter" in str(msg.children)

    def test_render_quality_timeline_with_decisions(self, reg):
        render = reg.by_name("render_quality_timeline")
        decisions = [{"decision": "accept"}] * 3 + [{"decision": "reject"}]
        fig, msg = render(decisions)
        assert isinstance(fig, go.Figure)
        assert "3/4" in str(msg.children)

    def test_populate_picker_filters_by_decision(self, reg):
        populate = reg.by_name("populate_picker")
        decisions = [
            {"decision": "accept"}, {"decision": "reject"},
            {"decision": "accept"}, {"decision": "reject"},
        ]
        opts, val = populate(decisions, "accept", None)
        # Only accepted indices appear
        assert {o["value"] for o in opts} == {0, 2}
        assert val in {0, 2}

    def test_populate_picker_preserves_current_when_visible(self, reg):
        populate = reg.by_name("populate_picker")
        decisions = [{"decision": "accept"}, {"decision": "accept"}]
        opts, val = populate(decisions, "all", 1)
        assert val == 1

    def test_click_to_pick(self, reg):
        click = reg.by_name("click_to_pick")
        decisions = [{"decision": "accept"}] * 5
        # Click on segment index 2
        assert click({"points": [{"x": 2}]}, decisions) == 2

    def test_click_to_pick_invalid(self, reg):
        click = reg.by_name("click_to_pick")
        with pytest.raises(PreventUpdate):
            click(None, [{"decision": "accept"}])
        with pytest.raises(PreventUpdate):
            click({"points": [{"x": 99}]}, [{"decision": "accept"}])

    def test_update_drop_button_no_decisions(self, reg):
        update = reg.by_name("update_drop_button")
        disabled, hint = update(None)
        assert disabled is True
        assert hint == ""

    def _build_decisions_with_outlier(self):
        """Build decisions where rule 'c' rejects ~10x more than peers,
        and peers have distinct non-zero counts so MAD > 0.

        4 rules (a/b/c/d).  Reject counts: a=3, b=4, c=50, d=5.
        median=4.5, MAD=median(|3-4.5|, |4-4.5|, |50-4.5|, |5-4.5|)
              =median(1.5, 0.5, 45.5, 0.5) = 1.0.
        threshold = 4.5 + 3*1.0 = 7.5 → only c (50) exceeds.
        """
        def reject_by(name):
            return {
                "decision": "reject",
                "trace": [
                    {"name": "a", "outcome": "reject" if name == "a" else "accept"},
                    {"name": "b", "outcome": "reject" if name == "b" else "accept"},
                    {"name": "c", "outcome": "reject" if name == "c" else "accept"},
                    {"name": "d", "outcome": "reject" if name == "d" else "accept"},
                ],
            }
        return (
            [reject_by("c")] * 50
            + [reject_by("a")] * 3
            + [reject_by("b")] * 4
            + [reject_by("d")] * 5
        )

    def test_update_drop_button_with_clear_outlier(self, reg):
        update = reg.by_name("update_drop_button")
        decisions = self._build_decisions_with_outlier()
        disabled, hint = update(decisions)
        assert disabled is False
        assert "c" in str(hint)

    def test_drop_strictest_removes_outlier(self, reg):
        drop = reg.by_name("drop_strictest")
        decisions = self._build_decisions_with_outlier()
        current = ["a", "b", "c", "d"]
        new_value = drop(1, decisions, current)
        assert "c" not in new_value
        assert set(new_value) == {"a", "b", "d"}

    def test_drop_strictest_no_clicks_prevents(self, reg):
        drop = reg.by_name("drop_strictest")
        with pytest.raises(PreventUpdate):
            drop(0, [{"decision": "reject"}], ["a"])

    # New callbacks introduced with the merged top-bar timeline -------

    def test_bridge_to_top_bar_empty(self, reg):
        bridge = reg.by_name("bridge_to_top_bar")
        fig, msg = bridge(None, 30)
        assert isinstance(fig, go.Figure)
        assert "Upload data" in str(msg.children)

    def test_bridge_to_top_bar_with_decisions(self, reg):
        bridge = reg.by_name("bridge_to_top_bar")
        decisions = [{"decision": "accept"}, {"decision": "reject"}]
        fig, msg = bridge(decisions, 30)
        assert isinstance(fig, go.Figure)
        assert "1/2" in str(msg.children)

    def test_timeline_click_drives_start_position(self, reg):
        cb = reg.by_name("timeline_click_to_position")
        milestones = [
            {"start_idx": 0,    "end_idx": 1000, "t_start": 0.0,  "t_end": 10.0},
            {"start_idx": 1000, "end_idx": 2000, "t_start": 10.0, "t_end": 20.0},
            {"start_idx": 2000, "end_idx": 3000, "t_start": 20.0, "t_end": 30.0},
        ]
        filtered_payload = {"n_samples": 3000, "sampling_freq": 100}
        # Click on segment 1 → start_idx=1000 / n_samples=3000 → pct ~ 33.3
        pct, picked = cb({"points": [{"x": 1}]}, milestones, filtered_payload)
        assert pct == pytest.approx(100.0 * 1000 / 3000)
        assert picked == 1
        # Boundary: segment 0 → 0%
        pct, picked = cb({"points": [{"x": 0}]}, milestones, filtered_payload)
        assert pct == 0.0 and picked == 0
        # Boundary: last segment → 66.7%
        pct, picked = cb({"points": [{"x": 2}]}, milestones, filtered_payload)
        assert pct == pytest.approx(100.0 * 2000 / 3000)
        assert picked == 2

    def test_timeline_click_invalid_index_prevents(self, reg):
        cb = reg.by_name("timeline_click_to_position")
        milestones = [{"start_idx": 0, "end_idx": 100, "t_start": 0.0, "t_end": 1.0}]
        filtered = {"n_samples": 100, "sampling_freq": 100}
        with pytest.raises(PreventUpdate):
            cb(None, milestones, filtered)
        with pytest.raises(PreventUpdate):
            cb({"points": [{"x": 99}]}, milestones, filtered)
        with pytest.raises(PreventUpdate):
            cb({"points": [{"x": 0}]}, None, filtered)

    def test_timeline_click_fallback_when_no_filtered_payload(self, reg):
        """Without the cached filtered payload, falls back to idx / count."""
        cb = reg.by_name("timeline_click_to_position")
        milestones = [{"start_idx": i * 100, "end_idx": (i + 1) * 100,
                       "t_start": i, "t_end": i + 1} for i in range(4)]
        # Click on segment 2 of 4 → 50%
        pct, picked = cb({"points": [{"x": 2}]}, milestones, None)
        assert pct == pytest.approx(100.0 * 2 / 4)
        assert picked == 2
