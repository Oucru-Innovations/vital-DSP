"""
Tests targeting missing coverage in segment_classifier.py.

Missing lines: 64-69, 89-90, 105-141, 169, 246, 254-271, 290
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vitalDSP.signal_quality_assessment import (
    classify_segments,
    load_rule_dict,
)
from vitalDSP.signal_quality_assessment.segment_classifier import (
    _build_rules_auto,
    _build_rules_manual,
    candidate_rule_columns,
    per_rule_reject_counts,
)
from vitalDSP.signal_quality_assessment.rule import Rule


# ---------------------------------------------------------------------------
# _build_rules_manual  (lines 77-91) — warning on bad rule, whitelist
# ---------------------------------------------------------------------------


class TestBuildRulesManual:
    def test_skips_bad_entry_with_warning(self, caplog):
        """Lines 89-90: exception in Rule.from_rule_dict_entry → warning, skip."""
        import logging

        rule_dict = {
            "good_sqi": {"lower": 1.0, "upper": 5.0},
            "bad_sqi": {"lower": 5.0, "upper": 1.0},  # lower > upper → ValueError
        }
        with caplog.at_level(logging.WARNING):
            rules = _build_rules_manual(rule_dict, selected_columns=None)
        # Only the good rule should survive
        assert len(rules) == 1
        assert rules[0].name == "good_sqi"

    def test_whitelist_filters(self):
        """Lines 85-86: whitelist not None → skip if not in whitelist."""
        rule_dict = {
            "a_sqi": {"lower": 0.0, "upper": 10.0},
            "b_sqi": {"lower": 0.0, "upper": 10.0},
        }
        rules = _build_rules_manual(rule_dict, selected_columns=["a_sqi"])
        assert len(rules) == 1
        assert rules[0].name == "a_sqi"

    def test_no_whitelist_includes_all(self):
        rule_dict = {
            "a_sqi": {"lower": 0.0, "upper": 10.0},
            "b_sqi": {"lower": 0.0, "upper": 10.0},
        }
        rules = _build_rules_manual(rule_dict, selected_columns=None)
        assert len(rules) == 2


# ---------------------------------------------------------------------------
# _build_rules_auto  (lines 94-141) — quantile mode, tune mode, empty cols
# ---------------------------------------------------------------------------


class TestBuildRulesAuto:
    @pytest.fixture
    def sqi_df(self):
        rng = np.random.default_rng(42)
        n = 50
        return pd.DataFrame({
            "kurtosis_sqi": 2.5 + rng.standard_normal(n),
            "snr_sqi": 10.0 + rng.standard_normal(n),
            "perfusion_sqi": 1500.0 + 100.0 * rng.standard_normal(n),
        })

    @pytest.fixture
    def rule_dict(self):
        return {
            "kurtosis_sqi": {"lower": 1.5, "upper": 5.0, "desc": "", "ref": ""},
            "snr_sqi": {"lower": 5.0, "upper": 20.0, "desc": "", "ref": ""},
            "perfusion_sqi": {"lower": 500.0, "upper": 3000.0, "desc": "", "ref": ""},
        }

    def test_quantile_mode(self, sqi_df, rule_dict):
        """Lines 116-124: quantile mode."""
        rules = _build_rules_auto(
            sqi_df, rule_dict, selected_columns=None,
            mode="quantile", quantile_lo=0.05, quantile_hi=0.95,
            target_accept_rate=0.90,
        )
        assert len(rules) > 0
        for r in rules:
            assert isinstance(r, Rule)

    def test_tune_mode(self, sqi_df, rule_dict):
        """Lines 113-115: tune mode."""
        rules = _build_rules_auto(
            sqi_df, rule_dict, selected_columns=None,
            mode="tune", quantile_lo=0.05, quantile_hi=0.95,
            target_accept_rate=0.90,
        )
        assert isinstance(rules, list)

    def test_no_columns_returns_empty(self, sqi_df, rule_dict):
        """Lines 106-111: columns list is empty → return []."""
        rules = _build_rules_auto(
            sqi_df, rule_dict, selected_columns=["nonexistent_sqi"],
            mode="quantile", quantile_lo=0.05, quantile_hi=0.95,
            target_accept_rate=0.90,
        )
        assert rules == []

    def test_degenerate_col_skipped(self, rule_dict):
        """Lines 126-140: degenerate band → Rule skipped / warning."""
        import logging
        n = 50
        df = pd.DataFrame({
            "kurtosis_sqi": np.full(n, 2.5),  # degenerate
            "snr_sqi": np.random.default_rng(0).standard_normal(n) + 10,
        })
        rules = _build_rules_auto(
            df, rule_dict, selected_columns=None,
            mode="quantile", quantile_lo=0.05, quantile_hi=0.95,
            target_accept_rate=0.90,
        )
        # kurtosis degenerate → no rule for it; snr may be kept
        names = [r.name for r in rules]
        assert "kurtosis_sqi" not in names


# ---------------------------------------------------------------------------
# candidate_rule_columns  (lines 149-177) — fewer-than-2-finite branch
# ---------------------------------------------------------------------------


class TestCandidateRuleColumnsExtra:
    def test_fewer_than_2_finite_reason(self):
        """Line 169: fewer than 2 finite samples — or degenerate band."""
        df = pd.DataFrame({"x_sqi": [np.nan, np.inf, -np.inf]})
        rule_dict = {"x_sqi": {"lower": 0.0, "upper": 1.0}}
        out = candidate_rule_columns(df, rule_dict)
        assert out[0]["usable"] is False
        # Either "fewer than 2" or "degenerate" — both are valid unusable reasons
        assert "fewer than 2" in out[0]["reason"] or "degenerate" in out[0]["reason"]

    def test_degenerate_band_reason(self):
        """Line ~173: degenerate band p5/p95 reported."""
        df = pd.DataFrame({"x_sqi": np.full(50, 3.14)})
        rule_dict = {"x_sqi": {"lower": 0.0, "upper": 10.0}}
        out = candidate_rule_columns(df, rule_dict)
        assert out[0]["usable"] is False
        assert "degenerate" in out[0]["reason"]


# ---------------------------------------------------------------------------
# classify_segments — quantile mode, tune mode, missing-column handling
# ---------------------------------------------------------------------------


class TestClassifySegmentsExtraModes:
    @pytest.fixture
    def clean_df(self):
        rng = np.random.default_rng(0)
        n = 30
        return pd.DataFrame({
            "kurtosis_sqi": 2.5 + 0.01 * rng.standard_normal(n),
            "snr_sqi": 10.0 + 0.5 * rng.standard_normal(n),
            "perfusion_sqi": 1500.0 + 50.0 * rng.standard_normal(n),
            "skewness_sqi": 0.5 + 0.05 * rng.standard_normal(n),
            "entropy_sqi": 2.0 + 0.05 * rng.standard_normal(n),
        })

    def test_quantile_mode(self, clean_df):
        """Lines 245-251: quantile auto-mode."""
        decs = classify_segments(clean_df, signal_type="PPG", mode="quantile")
        assert len(decs) == len(clean_df)
        for d in decs:
            assert d["decision"] in ("accept", "reject")

    def test_tune_mode(self, clean_df):
        """Lines 245-251: tune auto-mode."""
        decs = classify_segments(clean_df, signal_type="PPG", mode="tune")
        assert len(decs) == len(clean_df)

    def test_missing_column_in_row_yields_reject(self, clean_df):
        """Lines 266-274: KeyError → reject entry with NaN value."""
        # Build a rule_dict that requires a column not in sqi_df
        rule_dict = {
            "missing_col": {"lower": 0.0, "upper": 1.0},
        }
        decs = classify_segments(
            clean_df, signal_type="PPG", mode="manual",
            rule_dict=rule_dict,
        )
        # Every segment should be reject since the column is absent
        for d in decs:
            assert d["decision"] == "reject"

    def test_no_usable_rules_warns_and_accepts_all(self, caplog):
        """Lines 253-258: no usable rules → accept all."""
        import logging

        # Use a DataFrame with all degenerate columns so no rules survive
        df = pd.DataFrame({
            "kurtosis_sqi": np.full(10, 2.5),
        })
        with caplog.at_level(logging.WARNING):
            decs = classify_segments(df, signal_type="PPG", mode="tune")
        assert all(d["decision"] == "accept" for d in decs)

    def test_ecg_mode(self, clean_df):
        decs = classify_segments(clean_df, signal_type="ECG", mode="manual")
        assert len(decs) == len(clean_df)


# ---------------------------------------------------------------------------
# per_rule_reject_counts  (line 278-292)
# ---------------------------------------------------------------------------


class TestPerRuleRejectCountsExtra:
    def test_empty_decisions(self):
        """Line 290: no decisions → empty dict."""
        assert per_rule_reject_counts([]) == {}

    def test_only_accept_decisions(self):
        decisions = [
            {"decision": "accept", "trace": [
                {"name": "a", "value": 1.0, "outcome": "accept"},
            ]}
        ]
        counts = per_rule_reject_counts(decisions)
        assert counts == {}

    def test_mixed_decisions(self):
        decisions = [
            {"decision": "reject", "trace": [
                {"name": "x", "value": 0.1, "outcome": "reject"},
                {"name": "y", "value": 9.0, "outcome": "reject"},
            ]},
            {"decision": "accept", "trace": [
                {"name": "x", "value": 5.0, "outcome": "accept"},
            ]},
            {"decision": "reject", "trace": [
                {"name": "x", "value": 0.2, "outcome": "reject"},
            ]},
        ]
        counts = per_rule_reject_counts(decisions)
        assert counts == {"x": 2, "y": 1}
