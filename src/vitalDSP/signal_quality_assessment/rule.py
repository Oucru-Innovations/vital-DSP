"""Rule and RuleSet — threshold-based accept/reject for SQI values.

Ported from ``vital_sqi.rule.rule_class`` and ``vital_sqi.rule.ruleset_class``
with the JSON op-list parser collapsed into a direct ``(lower, upper)``
constructor.  The canonical four-entry rule form from vital-sqi's
``rule_dict.json``::

    [{"op": ">",  "value": "0.5", "label": "accept"},
     {"op": "<=", "value": "0.5", "label": "reject"},
     {"op": ">=", "value": "5.0", "label": "reject"},
     {"op": "<",  "value": "5.0", "label": "accept"}]

is exactly equivalent to "accept if ``0.5 < x < 5.0``, else reject" — and
that's the only form we use.  We expose:

* :class:`Rule` — one SQI's accept band ``(lower, upper)``.
* :class:`RuleSet` — ordered collection of rules; ``execute`` linearly
  scans and short-circuits on the first ``reject``.

Both classes are deliberately UI-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Outcome:
    """Internal record for a per-rule decision, used by RuleSet.execute_trace."""

    name: str
    value: float
    outcome: str


class Rule:
    """A single SQI's accept band.

    A value ``x`` is accepted iff ``lower < x < upper`` (open interval).
    NaN / Inf always reject.  The ``(lower, upper)`` form matches the
    canonical four-entry rule shipped in vital-sqi's ``rule_dict.json``.

    Parameters
    ----------
    name
        SQI column name; must match the column in the SQI DataFrame.
    lower, upper
        Accept-band bounds (exclusive).  Stored as floats; if you pass
        a number-like string we coerce.

    Raises
    ------
    ValueError
        If ``name`` is empty or ``lower >= upper``.
    """

    __slots__ = ("name", "lower", "upper", "desc", "ref")

    def __init__(
        self,
        name: str,
        lower: float,
        upper: float,
        *,
        desc: str = "",
        ref: str = "",
    ) -> None:
        if not name:
            raise ValueError("Rule name must be non-empty.")
        lo = float(lower)
        hi = float(upper)
        if not math.isfinite(lo) or not math.isfinite(hi):
            raise ValueError(f"Rule bounds must be finite; got ({lo}, {hi}).")
        if lo >= hi:
            raise ValueError(
                f"Rule {name!r}: lower ({lo}) must be < upper ({hi})."
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "lower", lo)
        object.__setattr__(self, "upper", hi)
        object.__setattr__(self, "desc", desc)
        object.__setattr__(self, "ref", ref)

    def apply_rule(self, x: float) -> str:
        """Return ``"accept"`` iff ``lower < x < upper``; else ``"reject"``.

        NaN / Inf always reject.
        """
        try:
            v = float(x)
        except (TypeError, ValueError):
            return "reject"
        if not math.isfinite(v):
            return "reject"
        return "accept" if (self.lower < v < self.upper) else "reject"

    def write_rule(self) -> str:
        """One-line human-readable summary."""
        return f"{self.lower:g} < x < {self.upper:g}"

    @classmethod
    def from_rule_dict_entry(cls, name: str, entry: dict) -> "Rule":
        """Build a Rule from a vital-sqi-style rule_dict entry.

        Expects either:

        * ``{"lower": .., "upper": ..}`` — the simplified form we write
          in our own bundled dicts, OR
        * ``{"def": [{"op": ..., "value": ..., "label": ...}, ...]}``
          — the original vital-sqi four-row form, from which we
          extract the (lower, upper) pair.

        Any ``"desc"`` / ``"ref"`` fields are passed through.
        """
        desc = entry.get("desc", "") if isinstance(entry, dict) else ""
        ref = entry.get("ref", "") if isinstance(entry, dict) else ""

        if "lower" in entry and "upper" in entry:
            return cls(
                name,
                float(entry["lower"]),
                float(entry["upper"]),
                desc=desc,
                ref=ref,
            )

        def_list = entry.get("def")
        if not def_list:
            raise ValueError(
                f"Rule entry for {name!r} has neither 'lower'/'upper' "
                "nor a 'def' list."
            )

        # Extract the (lower, upper) from the canonical four-row form.
        # We look for the two distinct numeric values appearing with
        # both ">" and "<" / "<=" / ">=" operators.  Discard NA/Inf
        # entries that vital-sqi sometimes includes as catch-alls.
        numeric_values: List[float] = []
        for row in def_list:
            v = row.get("value")
            if v is None:
                continue
            try:
                num = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(num):
                continue
            if num not in numeric_values:
                numeric_values.append(num)

        if len(numeric_values) < 2:
            raise ValueError(
                f"Rule entry for {name!r}: could not extract two distinct "
                f"finite thresholds from def list."
            )
        # The smaller value is the lower bound, the larger is the upper.
        lo = min(numeric_values)
        hi = max(numeric_values)
        return cls(name, lo, hi, desc=desc, ref=ref)


# ---------------------------------------------------------------------------
# RuleSet
# ---------------------------------------------------------------------------


class RuleSet:
    """Ordered collection of :class:`Rule` instances.

    Rules are stored in a dict keyed by integer order (1..N) — lower keys
    are evaluated first.  ``execute`` short-circuits on the first reject.

    Parameters
    ----------
    rules
        Either ``{order: Rule}`` (keys 1..N, consecutive) or an iterable
        of Rules in evaluation order.

    Raises
    ------
    ValueError
        If keys are not consecutive starting from 1, or any value is
        not a Rule instance.
    """

    __slots__ = ("rules",)

    def __init__(self, rules) -> None:
        if isinstance(rules, dict):
            try:
                normalised: Dict[int, Rule] = {int(k): v for k, v in rules.items()}
            except (TypeError, ValueError) as exc:
                raise ValueError(f"RuleSet keys must be integers: {exc}") from exc
        else:
            # Treat as ordered iterable
            normalised = {i + 1: r for i, r in enumerate(rules)}

        order = sorted(normalised.keys())
        if order != list(range(1, len(order) + 1)):
            raise ValueError(
                "Rule keys must be consecutive integers starting from 1; "
                f"got {order}."
            )
        for r in normalised.values():
            if not isinstance(r, Rule):
                raise ValueError(f"RuleSet values must be Rule instances; got {type(r)}.")
        self.rules = normalised

    def __len__(self) -> int:
        return len(self.rules)

    def __iter__(self):
        """Iterate Rules in evaluation order."""
        for k in sorted(self.rules):
            yield self.rules[k]

    def execute(self, row: pd.DataFrame) -> str:
        """Return ``"accept"`` if every rule accepts, ``"reject"`` otherwise.

        ``row`` is a single-row DataFrame whose columns include every
        ``rule.name``.  Short-circuits on the first reject.

        Raises
        ------
        TypeError
            If *row* is not a DataFrame.
        ValueError
            If *row* doesn't have exactly one row.
        KeyError
            If a rule's SQI is absent from *row*.
        """
        if not isinstance(row, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(row).__name__}")
        if len(row) != 1:
            raise ValueError(f"Expected single-row DataFrame, got {len(row)} rows.")
        for rule in self:
            if rule.name not in row.columns:
                raise KeyError(f"SQI {rule.name!r} not found in input row.")
            if rule.apply_rule(row.iloc[0][rule.name]) != "accept":
                return "reject"
        return "accept"

    def execute_trace(self, row: pd.DataFrame) -> dict:
        """Like :meth:`execute` but returns ``{decision, trace}``.

        ``trace`` is a list of ``{name, value, outcome}`` dicts for every
        rule.  Used by the UI's per-segment rule-trace panel.  Does NOT
        short-circuit — every rule is evaluated so the user can see which
        ones agreed.
        """
        if not isinstance(row, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(row).__name__}")
        if len(row) != 1:
            raise ValueError(f"Expected single-row DataFrame, got {len(row)} rows.")
        trace: List[dict] = []
        overall = "accept"
        for rule in self:
            value = row.iloc[0].get(rule.name)
            outcome = rule.apply_rule(value) if rule.name in row.columns else "reject"
            try:
                val_float = float(value) if value is not None else float("nan")
            except (TypeError, ValueError):
                val_float = float("nan")
            trace.append({"name": rule.name, "value": val_float, "outcome": outcome})
            if outcome != "accept":
                overall = "reject"
        return {"decision": overall, "trace": trace}
