"""Segment classifier — turn per-segment SQI values into accept/reject decisions.

Single entry point :func:`classify_segments`.  Three threshold modes:

* ``"manual"`` — use the bundled rule_dict's stored bounds verbatim.
* ``"quantile"`` — replace each rule's bounds with the empirical
  ``(lower_pct, upper_pct)`` quantiles of this recording's own SQI
  values.  Simple but with many rules the joint accept rate compounds
  low.
* ``"tune"`` — derive a per-rule quantile so the *joint* accept rate
  targets ``target_accept_rate`` (independence approximation).

Loading rule dicts: :func:`load_rule_dict` reads either of the bundled
``rule_dict_ppg.json`` / ``rule_dict_ecg.json`` based on ``signal_type``.
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .auto_threshold import (
    DEGENERATE_BAND_HALF_WIDTH,
    Band,
    quantile_band,
    sanitize_sqi,
    tuned_bands,
)
from .rule import Rule, RuleSet

logger = logging.getLogger(__name__)


_BUNDLED_DIR = os.path.dirname(__file__)
_RULE_DICT_FILES = {
    "PPG": os.path.join(_BUNDLED_DIR, "rule_dict_ppg.json"),
    "ECG": os.path.join(_BUNDLED_DIR, "rule_dict_ecg.json"),
}


# ---------------------------------------------------------------------------
# Rule-dict loading
# ---------------------------------------------------------------------------


def load_rule_dict(signal_type: str) -> Dict[str, dict]:
    """Load the bundled rule_dict for the given signal type.

    Returns an empty dict (with a warning) when the file is absent —
    callers should treat that as "no rules applicable" and accept all
    segments.
    """
    key = (signal_type or "PPG").upper()
    path = _RULE_DICT_FILES.get(key, _RULE_DICT_FILES["PPG"])
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        logger.warning("Bundled rule_dict not found for %s at %s", key, path)
        return {}
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse %s: %s", path, exc)
        return {}


# ---------------------------------------------------------------------------
# Rule construction by mode
# ---------------------------------------------------------------------------


def _build_rules_manual(
    rule_dict: Dict[str, dict],
    selected_columns: Optional[Iterable[str]],
) -> List[Rule]:
    """Build Rules from the rule_dict's stored bounds, no adaptation."""
    whitelist = set(selected_columns) if selected_columns else None
    rules: List[Rule] = []
    for name, entry in rule_dict.items():
        if whitelist is not None and name not in whitelist:
            continue
        try:
            rules.append(Rule.from_rule_dict_entry(name, entry))
        except Exception as exc:
            logger.warning("Skipping manual rule %r: %s", name, exc)
    return rules


def _build_rules_auto(
    sqi_df: pd.DataFrame,
    rule_dict: Dict[str, dict],
    selected_columns: Optional[Iterable[str]],
    *,
    mode: str,
    quantile_lo: float,
    quantile_hi: float,
    target_accept_rate: float,
) -> List[Rule]:
    """Build Rules whose bounds are derived from this recording's SQI distribution."""
    whitelist = set(selected_columns) if selected_columns else None
    columns = [
        c
        for c in sqi_df.columns
        if c in rule_dict and (whitelist is None or c in whitelist)
    ]
    if not columns:
        return []

    if mode == "tune":
        col_values = {c: sanitize_sqi(sqi_df[c].values) for c in columns}
        bands = tuned_bands(col_values, target_accept_rate=target_accept_rate)
    else:  # "quantile"
        bands = []
        for column in columns:
            band = quantile_band(
                column,
                sanitize_sqi(sqi_df[column].values),
                lower_pct=quantile_lo,
                upper_pct=quantile_hi,
            )
            if band is not None:
                bands.append(band)

    rules: List[Rule] = []
    for band in bands:
        try:
            entry = rule_dict.get(band.column, {})
            rules.append(
                Rule(
                    band.column,
                    band.lower,
                    band.upper,
                    desc=entry.get("desc", ""),
                    ref=entry.get("ref", ""),
                )
            )
        except Exception as exc:
            logger.warning("Could not build Rule for %r: %s", band.column, exc)
    return rules


# ---------------------------------------------------------------------------
# Candidate listing for the UI
# ---------------------------------------------------------------------------


def candidate_rule_columns(
    sqi_df: pd.DataFrame,
    rule_dict: Dict[str, dict],
) -> List[Dict[str, object]]:
    """For each column in *rule_dict*, report whether it's usable on this recording.

    Returns ``[{name, usable, reason}, ...]`` in rule-dict order.  Used by
    the UI to populate the checklist with disabled-with-reason entries.
    A column is "usable" iff its p5/p95 band is non-degenerate.
    """
    out: List[Dict[str, object]] = []
    for column in rule_dict:
        if column not in sqi_df.columns:
            out.append({"name": column, "usable": False, "reason": "not in SQI table"})
            continue
        values = sanitize_sqi(sqi_df[column].values)
        band = quantile_band(column, values, lower_pct=0.05, upper_pct=0.95)
        if band is None:
            finite = values[np.isfinite(values)] if values is not None else np.array([])
            if finite.size < 2:
                reason = "fewer than 2 finite samples"
            else:
                lo = float(np.quantile(finite, 0.05))
                hi = float(np.quantile(finite, 0.95))
                reason = f"degenerate band p5={lo:.4g}, p95={hi:.4g}"
            out.append({"name": column, "usable": False, "reason": reason})
        else:
            out.append({"name": column, "usable": True, "reason": ""})
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def classify_segments(
    sqi_df: pd.DataFrame,
    *,
    signal_type: str = "PPG",
    selected_columns: Optional[Iterable[str]] = None,
    mode: str = "tune",
    quantile_lo: float = 0.05,
    quantile_hi: float = 0.95,
    target_accept_rate: float = 0.90,
    rule_dict: Optional[Dict[str, dict]] = None,
) -> List[dict]:
    """Return ``[{decision, trace}, ...]`` — one entry per row of *sqi_df*.

    Parameters
    ----------
    sqi_df
        One row per segment, one column per SQI.  Output of
        :func:`compute_segment_sqis`.
    signal_type
        ``"PPG"`` or ``"ECG"`` — selects the bundled rule_dict.
        Ignored when *rule_dict* is supplied explicitly.
    selected_columns
        Whitelist of SQI names that participate.  ``None`` ⇒ every
        column present in both *sqi_df* and the rule_dict.
    mode
        ``"manual"``, ``"quantile"``, or ``"tune"``.
    quantile_lo, quantile_hi
        Used when ``mode == "quantile"``.
    target_accept_rate
        Used when ``mode == "tune"``.
    rule_dict
        Override the bundled dict (mainly for tests).

    Returns
    -------
    list of {decision, trace}
        ``decision`` is ``"accept"`` or ``"reject"``.  ``trace`` is a
        list of ``{name, value, outcome}`` dicts in rule-evaluation
        order.  Empty *sqi_df* yields ``[]``.  When no usable rules
        exist every segment gets ``decision="accept"`` and an empty
        trace, plus a warning is logged.
    """
    if mode not in ("manual", "quantile", "tune"):
        raise ValueError(f"mode must be 'manual', 'quantile', or 'tune'; got {mode!r}.")
    if sqi_df is None or sqi_df.empty:
        return []

    if rule_dict is None:
        rule_dict = load_rule_dict(signal_type)

    if not rule_dict:
        logger.warning(
            "No rule_dict available for %s; accepting all segments.", signal_type
        )
        return [{"decision": "accept", "trace": []} for _ in range(len(sqi_df))]

    if mode == "manual":
        rules = _build_rules_manual(rule_dict, selected_columns)
    else:
        rules = _build_rules_auto(
            sqi_df,
            rule_dict,
            selected_columns,
            mode=mode,
            quantile_lo=quantile_lo,
            quantile_hi=quantile_hi,
            target_accept_rate=target_accept_rate,
        )

    if not rules:
        logger.warning(
            "No usable rules for %s under mode=%r; accepting all segments.",
            signal_type,
            mode,
        )
        return [{"decision": "accept", "trace": []} for _ in range(len(sqi_df))]

    ruleset = RuleSet(rules)
    decisions: List[dict] = []
    for idx in range(len(sqi_df)):
        row = sqi_df.iloc[[idx]]
        try:
            decisions.append(ruleset.execute_trace(row))
        except KeyError as exc:
            # A rule references a column missing from sqi_df.  Don't
            # blow up the whole classification — emit a reject with
            # an explanatory trace entry.
            logger.warning("Missing SQI column at segment %d: %s", idx, exc)
            decisions.append(
                {
                    "decision": "reject",
                    "trace": [
                        {"name": str(exc), "value": float("nan"), "outcome": "reject"}
                    ],
                }
            )
    return decisions


def per_rule_reject_counts(decisions: List[dict]) -> Dict[str, int]:
    """Tally how many segments each rule rejected.

    Used by the "Drop strictest rule" feature.
    """
    counts: Dict[str, int] = {}
    for d in decisions:
        if d.get("decision") != "reject":
            continue
        for entry in d.get("trace", []):
            if entry.get("outcome") != "accept":
                name = entry.get("name")
                if name:
                    counts[name] = counts.get(name, 0) + 1
    return counts
