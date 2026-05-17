"""
Tests targeting missing coverage in rule.py.

Missing lines: 102-103, 110, 140, 153, 164, 204-205, 218, 222, 250, 264, 266, 274-275
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from vitalDSP.signal_quality_assessment.rule import Rule, RuleSet


# ---------------------------------------------------------------------------
# Rule.apply_rule — non-numeric value (line 102-103)
# ---------------------------------------------------------------------------


class TestRuleApplyRule:
    def test_non_numeric_rejects(self):
        r = Rule("x", 0.0, 10.0)
        # Passing a value that cannot be cast to float
        assert r.apply_rule("not_a_number") == "reject"   # line 102-103
        assert r.apply_rule(None) == "reject"              # line 102-103
        assert r.apply_rule([1, 2]) == "reject"            # line 102-103

    def test_boundary_rejects(self):
        r = Rule("x", 2.0, 5.0)
        assert r.apply_rule(2.0) == "reject"  # equal to lower → reject
        assert r.apply_rule(5.0) == "reject"  # equal to upper → reject

    def test_inside_accepts(self):
        r = Rule("x", 2.0, 5.0)
        assert r.apply_rule(3.5) == "accept"


# ---------------------------------------------------------------------------
# Rule.write_rule  (line 110)
# ---------------------------------------------------------------------------


class TestRuleWriteRule:
    def test_write_rule_format(self):
        r = Rule("test_sqi", 1.5, 10.0)
        s = r.write_rule()
        assert "1.5" in s
        assert "10" in s
        assert "<" in s


# ---------------------------------------------------------------------------
# Rule.from_rule_dict_entry — missing 'def' key (line 140)
# ---------------------------------------------------------------------------


class TestFromRuleDictEntry:
    def test_missing_lower_upper_and_no_def_raises(self):
        entry = {"desc": "bad entry"}  # neither 'lower'/'upper' nor 'def'
        with pytest.raises(ValueError, match="neither"):
            Rule.from_rule_dict_entry("sqi", entry)   # line 140

    def test_def_list_with_fewer_than_two_thresholds_raises(self):
        # Only one distinct numeric value
        entry = {
            "def": [
                {"op": ">", "value": "0.5", "label": "accept"},
                {"op": "<=", "value": "0.5", "label": "reject"},
            ]
        }
        with pytest.raises(ValueError, match="two distinct"):
            Rule.from_rule_dict_entry("sqi", entry)   # line 164

    def test_def_list_all_none_values_raises(self):
        entry = {
            "def": [
                {"op": ">", "value": None},
                {"op": "<", "value": None},
            ]
        }
        with pytest.raises(ValueError):
            Rule.from_rule_dict_entry("sqi", entry)

    def test_def_list_all_nan_inf_raises(self):
        entry = {
            "def": [
                {"op": "=", "value": "NA"},
                {"op": "=", "value": "Inf"},
            ]
        }
        with pytest.raises(ValueError):
            Rule.from_rule_dict_entry("sqi", entry)

    def test_from_rule_dict_with_desc_and_ref(self):
        entry = {"lower": 1.0, "upper": 5.0, "desc": "My desc", "ref": "Ref 1"}
        r = Rule.from_rule_dict_entry("my_sqi", entry)
        assert r.desc == "My desc"
        assert r.ref == "Ref 1"


# ---------------------------------------------------------------------------
# RuleSet constructor — non-Rule values, non-integer keys (lines 204-222)
# ---------------------------------------------------------------------------


class TestRuleSetConstructor:
    def test_dict_with_non_rule_value_raises(self):
        with pytest.raises(ValueError, match="Rule instances"):
            RuleSet({1: "not_a_rule"})   # line 218

    def test_dict_with_non_integer_keys_raises(self):
        with pytest.raises(ValueError):
            RuleSet({"a": Rule("x", 0.0, 1.0)})  # line 204-205

    def test_consecutive_keys_required(self):
        with pytest.raises(ValueError, match="consecutive"):
            RuleSet({1: Rule("x", 0.0, 1.0), 3: Rule("y", 0.0, 1.0)})  # line 213

    def test_dict_with_valid_integer_keys(self):
        rs = RuleSet({1: Rule("a", 0.0, 10.0), 2: Rule("b", 0.0, 10.0)})
        assert len(rs) == 2

    def test_len_and_iter(self):
        rules = [Rule("a", 0.0, 5.0), Rule("b", 0.0, 5.0)]
        rs = RuleSet(rules)
        assert len(rs) == 2
        names = [r.name for r in rs]
        assert names == ["a", "b"]


# ---------------------------------------------------------------------------
# RuleSet.execute — missing column raises KeyError (line 250)
# ---------------------------------------------------------------------------


class TestRuleSetExecute:
    def test_missing_column_raises_key_error(self):
        rs = RuleSet([Rule("present_col", 0.0, 10.0), Rule("missing_col", 0.0, 10.0)])
        row = pd.DataFrame([{"present_col": 5.0}])  # 'missing_col' absent
        with pytest.raises(KeyError):
            rs.execute(row)  # line 250

    def test_execute_returns_accept_for_all_pass(self):
        rs = RuleSet([Rule("a", 0.0, 10.0)])
        row = pd.DataFrame([{"a": 5.0}])
        assert rs.execute(row) == "accept"

    def test_execute_type_error_non_df(self):
        rs = RuleSet([Rule("a", 0.0, 10.0)])
        with pytest.raises(TypeError):
            rs.execute({"a": 5.0})  # dict, not DataFrame

    def test_execute_value_error_multi_row(self):
        rs = RuleSet([Rule("a", 0.0, 10.0)])
        with pytest.raises(ValueError):
            rs.execute(pd.DataFrame({"a": [1.0, 2.0]}))


# ---------------------------------------------------------------------------
# RuleSet.execute_trace — value is None (lines 264, 266, 274-275)
# ---------------------------------------------------------------------------


class TestRuleSetExecuteTrace:
    def test_missing_column_gets_reject(self):
        rs = RuleSet([Rule("a", 0.0, 10.0), Rule("b", 0.0, 10.0)])
        row = pd.DataFrame([{"a": 5.0}])  # 'b' absent
        result = rs.execute_trace(row)
        # 'b' missing → outcome "reject"
        b_trace = [t for t in result["trace"] if t["name"] == "b"][0]
        assert b_trace["outcome"] == "reject"
        assert result["decision"] == "reject"

    def test_value_is_none_becomes_nan(self):
        rs = RuleSet([Rule("a", 0.0, 10.0)])
        # Build a row where 'a' is present but is None
        row = pd.DataFrame([{"a": None}])
        result = rs.execute_trace(row)
        # None → can't float → NaN stored, outcome "reject"
        trace = result["trace"][0]
        # value should be NaN
        import math
        assert math.isnan(trace["value"])
        assert trace["outcome"] == "reject"

    def test_execute_trace_non_df_raises(self):
        rs = RuleSet([Rule("a", 0.0, 10.0)])
        with pytest.raises(TypeError):
            rs.execute_trace([{"a": 1.0}])

    def test_execute_trace_multi_row_raises(self):
        rs = RuleSet([Rule("a", 0.0, 10.0)])
        with pytest.raises(ValueError):
            rs.execute_trace(pd.DataFrame({"a": [1.0, 2.0]}))

    def test_execute_trace_all_accept(self):
        rs = RuleSet([Rule("a", 0.0, 10.0), Rule("b", 0.0, 10.0)])
        row = pd.DataFrame([{"a": 3.0, "b": 4.0}])
        result = rs.execute_trace(row)
        assert result["decision"] == "accept"
        assert all(t["outcome"] == "accept" for t in result["trace"])
