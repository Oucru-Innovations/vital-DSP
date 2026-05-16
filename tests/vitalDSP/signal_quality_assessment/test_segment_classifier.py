"""Unit tests for the segment-quality engine.

Covers:

* :class:`Rule` accept/reject boundaries and NaN handling.
* :class:`RuleSet` short-circuit and trace generation.
* :func:`quantile_band` degenerate-distribution guard.
* :func:`per_rule_quantile` joint-accept math.
* :func:`tuned_bands` two-pass survivor filtering.
* :func:`strictest_columns` MAD-based outlier detection.
* :func:`compute_segment_sqis` segmentation and overlap.
* :func:`classify_segments` end-to-end for each mode.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from vitalDSP.signal_quality_assessment import (
    AVAILABLE_SQIS,
    DEGENERATE_BAND_HALF_WIDTH,
    Rule,
    RuleSet,
    candidate_rule_columns,
    classify_segments,
    compute_segment_sqis,
    load_rule_dict,
    per_rule_quantile,
    per_rule_reject_counts,
    quantile_band,
    sanitize_sqi,
    strictest_columns,
    tuned_bands,
)


# ---------------------------------------------------------------------------
# Rule
# ---------------------------------------------------------------------------


class TestRule:
    def test_open_interval_boundaries(self):
        r = Rule("x_sqi", 0.5, 5.0)
        # Strict inside
        assert r.apply_rule(3.0) == "accept"
        # On the boundary → reject (open interval)
        assert r.apply_rule(0.5) == "reject"
        assert r.apply_rule(5.0) == "reject"
        # Outside
        assert r.apply_rule(0.49) == "reject"
        assert r.apply_rule(5.01) == "reject"

    def test_nan_and_inf_reject(self):
        r = Rule("x_sqi", 0.0, 1.0)
        assert r.apply_rule(float("nan")) == "reject"
        assert r.apply_rule(float("inf")) == "reject"
        assert r.apply_rule(float("-inf")) == "reject"

    def test_invalid_bounds_raise(self):
        with pytest.raises(ValueError):
            Rule("x_sqi", 1.0, 0.5)  # lower > upper
        with pytest.raises(ValueError):
            Rule("x_sqi", 1.0, 1.0)  # zero width
        with pytest.raises(ValueError):
            Rule("", 0.0, 1.0)  # empty name
        with pytest.raises(ValueError):
            Rule("x", float("inf"), 1.0)  # non-finite bound

    def test_from_rule_dict_entry_simplified(self):
        entry = {"lower": 0.5, "upper": 5.0, "desc": "test", "ref": "test"}
        r = Rule.from_rule_dict_entry("x_sqi", entry)
        assert r.lower == 0.5
        assert r.upper == 5.0
        assert r.desc == "test"
        assert r.ref == "test"

    def test_from_rule_dict_entry_vital_sqi_form(self):
        # vital-sqi's canonical four-row form
        entry = {
            "name": "x_sqi",
            "def": [
                {"op": ">", "value": "0.5", "label": "accept"},
                {"op": "<=", "value": "0.5", "label": "reject"},
                {"op": ">=", "value": "5.0", "label": "reject"},
                {"op": "<", "value": "5.0", "label": "accept"},
            ],
        }
        r = Rule.from_rule_dict_entry("x_sqi", entry)
        assert r.lower == 0.5
        assert r.upper == 5.0

    def test_from_rule_dict_entry_ignores_na_inf(self):
        # vital-sqi sometimes inserts NA/Inf catch-all rows
        entry = {
            "def": [
                {"op": ">", "value": "0.5", "label": "accept"},
                {"op": "<", "value": "5.0", "label": "accept"},
                {"op": "=", "value": "NA", "label": "accept"},
                {"op": "=", "value": "Inf", "label": "accept"},
            ],
        }
        r = Rule.from_rule_dict_entry("x_sqi", entry)
        assert r.lower == 0.5
        assert r.upper == 5.0


# ---------------------------------------------------------------------------
# RuleSet
# ---------------------------------------------------------------------------


class TestRuleSet:
    def test_execute_all_accept(self):
        rs = RuleSet([
            Rule("a", 0.0, 10.0),
            Rule("b", 0.0, 10.0),
        ])
        row = pd.DataFrame([{"a": 5.0, "b": 5.0}])
        assert rs.execute(row) == "accept"

    def test_execute_short_circuits_on_first_reject(self):
        # If the first rule rejects, the second rule's column doesn't even
        # need to be present.
        rs = RuleSet([
            Rule("a", 10.0, 20.0),
            Rule("missing", 0.0, 1.0),
        ])
        row = pd.DataFrame([{"a": 1.0}])  # 'missing' absent
        assert rs.execute(row) == "reject"

    def test_execute_trace_no_short_circuit(self):
        rs = RuleSet([
            Rule("a", 10.0, 20.0),  # will reject
            Rule("b", 0.0, 10.0),   # would accept
        ])
        row = pd.DataFrame([{"a": 1.0, "b": 5.0}])
        result = rs.execute_trace(row)
        assert result["decision"] == "reject"
        assert len(result["trace"]) == 2  # both evaluated
        assert result["trace"][0]["outcome"] == "reject"
        assert result["trace"][1]["outcome"] == "accept"

    def test_non_consecutive_keys_raise(self):
        with pytest.raises(ValueError):
            RuleSet({1: Rule("a", 0.0, 1.0), 3: Rule("b", 0.0, 1.0)})

    def test_invalid_row_shape_raises(self):
        rs = RuleSet([Rule("a", 0.0, 1.0)])
        with pytest.raises(TypeError):
            rs.execute([1.0])  # not a DataFrame
        with pytest.raises(ValueError):
            rs.execute(pd.DataFrame([{"a": 1.0}, {"a": 2.0}]))  # 2 rows


# ---------------------------------------------------------------------------
# quantile_band / degenerate guard
# ---------------------------------------------------------------------------


class TestQuantileBand:
    def test_basic_p5_p95(self):
        values = np.arange(100, dtype=float)
        b = quantile_band("x", values, lower_pct=0.05, upper_pct=0.95)
        assert b is not None
        # On 0-99 uniform, p5≈4.95, p95≈94.05
        assert b.lower == pytest.approx(4.95, abs=0.5)
        assert b.upper == pytest.approx(94.05, abs=0.5)

    def test_degenerate_collapses_to_none(self):
        values = np.full(50, 3.14)
        b = quantile_band("x", values)
        assert b is None

    def test_near_degenerate_under_guard(self):
        # Spread is smaller than DEGENERATE_BAND_HALF_WIDTH → None
        values = np.array([1.0] * 50 + [1.0 + DEGENERATE_BAND_HALF_WIDTH / 10])
        b = quantile_band("x", values)
        assert b is None

    def test_drops_nan_inf(self):
        values = np.concatenate([
            np.linspace(0, 10, 100),
            [np.nan, np.inf, -np.inf],
        ])
        b = quantile_band("x", values)
        assert b is not None
        assert b.lower < b.upper

    def test_too_few_finite_returns_none(self):
        values = np.array([np.nan, np.inf, 1.0])
        b = quantile_band("x", values)
        assert b is None

    def test_invalid_percentiles_raise(self):
        with pytest.raises(ValueError):
            quantile_band("x", [1, 2, 3], lower_pct=0.6, upper_pct=0.95)
        with pytest.raises(ValueError):
            quantile_band("x", [1, 2, 3], lower_pct=0.05, upper_pct=0.4)


# ---------------------------------------------------------------------------
# per_rule_quantile / tuned_bands
# ---------------------------------------------------------------------------


class TestPerRuleQuantile:
    def test_n_rules_one_returns_symmetric_split(self):
        # For n=1, trim = 1 - target; symmetric half on each side
        q = per_rule_quantile(0.90, n_rules=1)
        assert q == pytest.approx(0.05, rel=1e-6)

    def test_n_rules_five_target_85(self):
        # Exact: keep = 0.85^(1/5) ≈ 0.96802; trim ≈ 0.03198; q ≈ 0.01599.
        q = per_rule_quantile(0.85, n_rules=5)
        assert q == pytest.approx(0.01599, abs=1e-4)

    def test_n_rules_five_target_90(self):
        # Our default: target=0.90, n=5
        q = per_rule_quantile(0.90, n_rules=5)
        # keep = 0.90^0.2 ≈ 0.9791; trim ≈ 0.0209; q ≈ 0.01047
        assert q == pytest.approx(0.01047, abs=1e-4)

    def test_invalid_n_rules_raises(self):
        with pytest.raises(ValueError):
            per_rule_quantile(0.85, n_rules=0)

    def test_clipping(self):
        # Extreme target gets clipped — should not crash
        q = per_rule_quantile(0.0, n_rules=5)
        assert 0.0 <= q < 0.5
        q = per_rule_quantile(1.0, n_rules=5)
        assert 0.0 <= q < 0.5


class TestTunedBands:
    def test_basic_two_pass_filter(self):
        rng = np.random.default_rng(42)
        cols = {
            "good_a": rng.normal(0, 1, 500),
            "good_b": rng.normal(0, 1, 500),
            "flat": np.full(500, 7.0),  # degenerate, dropped by pass 1
        }
        bands = tuned_bands(cols, target_accept_rate=0.90)
        assert len(bands) == 2
        names = [b.column for b in bands]
        assert "flat" not in names
        assert "good_a" in names and "good_b" in names

    def test_empty_input(self):
        bands = tuned_bands({}, target_accept_rate=0.90)
        assert bands == []

    def test_quantile_chosen_from_survivors_only(self):
        # 2 good cols + 3 flat ones → n_rules should be 2, not 5
        rng = np.random.default_rng(0)
        cols = {
            "good_a": rng.normal(0, 1, 200),
            "good_b": rng.normal(0, 1, 200),
            "flat_a": np.full(200, 1.0),
            "flat_b": np.full(200, 2.0),
            "flat_c": np.full(200, 3.0),
        }
        bands = tuned_bands(cols, target_accept_rate=0.85)
        assert len(bands) == 2
        # n_rules = 2 → trim = (1 - 0.85^0.5)/2 ≈ 0.039
        expected_lo = per_rule_quantile(0.85, n_rules=2)
        for b in bands:
            assert b.quantile_lo == pytest.approx(expected_lo, rel=1e-6)


# ---------------------------------------------------------------------------
# strictest_columns
# ---------------------------------------------------------------------------


class TestStrictestColumns:
    def test_clear_outlier(self):
        counts = {"a": 5, "b": 6, "c": 5, "d": 4, "e": 50}
        flagged = strictest_columns(counts)
        assert flagged == ["e"]

    def test_no_outlier_returns_empty(self):
        counts = {"a": 5, "b": 6, "c": 5, "d": 4, "e": 5}
        assert strictest_columns(counts) == []

    def test_too_few_rules_returns_empty(self):
        # With only 2 rules, one is trivially the "outlier" → refuse to flag
        assert strictest_columns({"a": 1, "b": 100}) == []

    def test_all_equal_returns_empty(self):
        # MAD == 0 → no outlier
        assert strictest_columns({"a": 7, "b": 7, "c": 7, "d": 7}) == []

    def test_sorted_descending(self):
        counts = {"a": 5, "b": 6, "c": 5, "d": 50, "e": 100}
        flagged = strictest_columns(counts)
        # Worst (100) should be first
        assert flagged[0] == "e"


# ---------------------------------------------------------------------------
# sanitize_sqi
# ---------------------------------------------------------------------------


class TestSanitizeSqi:
    def test_inf_becomes_median(self):
        v = sanitize_sqi([1.0, 2.0, 3.0, np.inf, -np.inf, 4.0])
        assert np.all(np.isfinite(v))
        # The two inf positions get the median of the finite values = 2.5
        assert v[3] == pytest.approx(2.5)
        assert v[4] == pytest.approx(2.5)

    def test_all_non_finite_returns_zeros(self):
        v = sanitize_sqi([np.nan, np.inf, -np.inf])
        np.testing.assert_array_equal(v, np.zeros(3))


# ---------------------------------------------------------------------------
# compute_segment_sqis
# ---------------------------------------------------------------------------


class TestComputeSegmentSqis:
    def test_basic_segmentation(self):
        sig = np.sin(np.linspace(0, 60, 6000)) + 1.0
        df, ms = compute_segment_sqis(sig, sampling_freq=100, segment_seconds=10)
        assert len(df) == 6  # 60 s / 10 s = 6 non-overlapping segments
        assert len(ms) == 6
        assert set(df.columns).issubset(set(AVAILABLE_SQIS))
        # Milestones increase monotonically
        starts = [m["start_idx"] for m in ms]
        assert starts == sorted(starts)

    def test_overlap(self):
        sig = np.zeros(1000)
        df_0, _ = compute_segment_sqis(sig, sampling_freq=100, segment_seconds=2)
        df_50, _ = compute_segment_sqis(sig, sampling_freq=100, segment_seconds=2, overlap_pct=0.5)
        # 50% overlap roughly doubles the segment count
        assert len(df_50) > len(df_0)

    def test_short_signal_returns_empty(self):
        sig = np.zeros(50)
        df, ms = compute_segment_sqis(sig, sampling_freq=100, segment_seconds=10)
        assert df.empty
        assert ms == []

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            compute_segment_sqis([1, 2, 3], sampling_freq=1, overlap_pct=0.99)

    def test_invalid_fs_raises(self):
        with pytest.raises(ValueError):
            compute_segment_sqis([1, 2, 3], sampling_freq=0)

    def test_unknown_sqi_names_ignored(self):
        sig = np.zeros(1000)
        df, _ = compute_segment_sqis(
            sig, sampling_freq=100, segment_seconds=2,
            sqi_names=["kurtosis_sqi", "made_up_sqi"],
        )
        assert "kurtosis_sqi" in df.columns
        assert "made_up_sqi" not in df.columns


# ---------------------------------------------------------------------------
# classify_segments — end-to-end per mode
# ---------------------------------------------------------------------------


class TestClassifySegments:
    @pytest.fixture
    def clean_ppg_sqi_df(self):
        # 20 nearly identical "clean" rows
        rng = np.random.default_rng(0)
        n = 20
        return pd.DataFrame({
            "kurtosis_sqi": 2.5 + 0.01 * rng.standard_normal(n),
            "snr_sqi": 10.0 + 0.5 * rng.standard_normal(n),
            "perfusion_sqi": 1500.0 + 50.0 * rng.standard_normal(n),
            "skewness_sqi": 0.5 + 0.05 * rng.standard_normal(n),
            "entropy_sqi": 2.0 + 0.05 * rng.standard_normal(n),
            "zero_crossing_sqi": 0.01 + 0.001 * rng.standard_normal(n),
            "peak_to_peak_amplitude_sqi": 50.0 + 5.0 * rng.standard_normal(n),
        })

    def test_manual_mode_accepts_clean(self, clean_ppg_sqi_df):
        decs = classify_segments(clean_ppg_sqi_df, signal_type="PPG", mode="manual")
        accepted = sum(1 for d in decs if d["decision"] == "accept")
        # Manual bounds should accept the bulk of a deliberately-clean batch
        assert accepted >= 18

    def test_each_decision_has_decision_and_trace_keys(self, clean_ppg_sqi_df):
        decs = classify_segments(clean_ppg_sqi_df, signal_type="PPG", mode="manual")
        for d in decs:
            assert d["decision"] in ("accept", "reject")
            assert "trace" in d
            assert isinstance(d["trace"], list)

    def test_empty_df_returns_empty(self):
        decs = classify_segments(pd.DataFrame(), signal_type="PPG", mode="manual")
        assert decs == []

    def test_invalid_mode_raises(self, clean_ppg_sqi_df):
        with pytest.raises(ValueError):
            classify_segments(clean_ppg_sqi_df, mode="bogus")

    def test_selected_columns_filters(self, clean_ppg_sqi_df):
        # When we only enable one rule, every trace should be length 1
        decs = classify_segments(
            clean_ppg_sqi_df, signal_type="PPG", mode="manual",
            selected_columns=["kurtosis_sqi"],
        )
        for d in decs:
            assert len(d["trace"]) == 1
            assert d["trace"][0]["name"] == "kurtosis_sqi"

    def test_outlier_segment_gets_rejected_manual(self, clean_ppg_sqi_df):
        df = clean_ppg_sqi_df.copy()
        # Force one row to have an outlier kurtosis (below manual lower bound)
        df.loc[0, "kurtosis_sqi"] = 0.5
        decs = classify_segments(df, signal_type="PPG", mode="manual")
        assert decs[0]["decision"] == "reject"
        rejecting = [t["name"] for t in decs[0]["trace"] if t["outcome"] != "accept"]
        assert "kurtosis_sqi" in rejecting

    def test_missing_rule_dict_accepts_all(self, clean_ppg_sqi_df):
        # Empty rule_dict override → no rules → accept everyone
        decs = classify_segments(
            clean_ppg_sqi_df, signal_type="PPG", mode="manual",
            rule_dict={},
        )
        assert all(d["decision"] == "accept" for d in decs)
        assert all(d["trace"] == [] for d in decs)


# ---------------------------------------------------------------------------
# candidate_rule_columns + per_rule_reject_counts
# ---------------------------------------------------------------------------


class TestCandidateAndCounts:
    def test_candidate_marks_degenerate_unusable(self):
        df = pd.DataFrame({
            "kurtosis_sqi": np.linspace(2.0, 3.0, 50),
            "skewness_sqi": np.full(50, 0.5),  # degenerate
        })
        rule_dict = {
            "kurtosis_sqi": {"lower": 1.5, "upper": 5.0},
            "skewness_sqi": {"lower": -1.0, "upper": 1.0},
            "missing_sqi": {"lower": 0.0, "upper": 1.0},
        }
        out = candidate_rule_columns(df, rule_dict)
        by_name = {c["name"]: c for c in out}
        assert by_name["kurtosis_sqi"]["usable"] is True
        assert by_name["skewness_sqi"]["usable"] is False
        assert "degenerate" in by_name["skewness_sqi"]["reason"]
        assert by_name["missing_sqi"]["usable"] is False
        assert "not in SQI table" in by_name["missing_sqi"]["reason"]

    def test_per_rule_reject_counts(self):
        decisions = [
            {"decision": "accept", "trace": [
                {"name": "a", "value": 1.0, "outcome": "accept"},
                {"name": "b", "value": 1.0, "outcome": "accept"},
            ]},
            {"decision": "reject", "trace": [
                {"name": "a", "value": 1.0, "outcome": "accept"},
                {"name": "b", "value": 9.0, "outcome": "reject"},
            ]},
            {"decision": "reject", "trace": [
                {"name": "a", "value": 9.0, "outcome": "reject"},
                {"name": "b", "value": 9.0, "outcome": "reject"},
            ]},
        ]
        counts = per_rule_reject_counts(decisions)
        assert counts == {"a": 1, "b": 2}


# ---------------------------------------------------------------------------
# Bundled rule dicts present and parseable
# ---------------------------------------------------------------------------


class TestBundledRuleDicts:
    def test_ppg_dict_loads(self):
        d = load_rule_dict("PPG")
        assert "kurtosis_sqi" in d
        assert "snr_sqi" in d
        for name, entry in d.items():
            assert "lower" in entry and "upper" in entry
            assert entry["lower"] < entry["upper"]

    def test_ecg_dict_loads(self):
        d = load_rule_dict("ECG")
        assert "kurtosis_sqi" in d
        assert "snr_sqi" in d
        for name, entry in d.items():
            assert "lower" in entry and "upper" in entry
            assert entry["lower"] < entry["upper"]

    def test_unknown_signal_type_falls_back_to_ppg(self):
        d = load_rule_dict("EEG")
        # Falls back to PPG silently
        assert "kurtosis_sqi" in d
