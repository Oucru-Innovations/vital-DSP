"""Auto-threshold helpers — derive accept bands from SQI distributions.

Ported from ``vital_sqi.rule.auto_threshold``.  Pure-numpy, UI-free.

Two band-derivation policies:

* :func:`quantile_band` — fixed quantile window per SQI.  The user picks
  the per-rule trim (e.g. p5/p95).
* :func:`tuned_bands` — derive per-rule quantile from a target *joint*
  accept rate, assuming independence between rules.  Solves
  ``target = keep ** n_rules`` for each rule's keep rate, then splits
  the trim symmetrically across both tails.

Plus :func:`strictest_columns` — modified-Z-score detector to flag a
rule whose reject count is an upward outlier among its peers.

The degenerate-band guard (:data:`DEGENERATE_BAND_HALF_WIDTH`) returns
``None`` when an SQI's distribution collapses to a single value, so the
caller can drop it from the rule set instead of producing a "reject
everything" rule.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


#: Bands narrower than this collapse to "reject everything" under
#: percentile-based auto-mode.  Same threshold vital-sqi uses.
DEGENERATE_BAND_HALF_WIDTH = 1e-6


@dataclass(frozen=True)
class Band:
    """An accept band ``(lower, upper)`` plus diagnostic provenance."""

    column: str
    lower: float
    upper: float
    quantile_lo: float
    quantile_hi: float
    note: str = ""

    @property
    def width(self) -> float:
        return self.upper - self.lower


# ---------------------------------------------------------------------------
# Sanitiser — used by both policies and by the classifier
# ---------------------------------------------------------------------------


def sanitize_sqi(values) -> np.ndarray:
    """Replace inf/-inf with NaN, then fill NaN with the column median.

    Mirrors vital-sqi's ``common.utils.sanitize_sqi``.  Returns a float
    array of the same length with no inf/NaN, or all-zeros when the
    input is entirely non-finite.
    """
    v = np.asarray(values, dtype=float)
    v = np.where(np.isfinite(v), v, np.nan)
    if np.all(np.isnan(v)):
        return np.zeros(v.shape, dtype=float)
    med = float(np.nanmedian(v))
    v = np.where(np.isnan(v), med, v)
    return v


# ---------------------------------------------------------------------------
# Policy 1 — fixed quantile window
# ---------------------------------------------------------------------------


def quantile_band(
    column: str,
    values: Sequence[float],
    *,
    lower_pct: float = 0.05,
    upper_pct: float = 0.95,
) -> Optional[Band]:
    """Compute an accept band from empirical lower/upper quantiles.

    Parameters
    ----------
    column
        SQI name; carried into the returned Band for diagnostics.
    values
        Observed SQI values; NaN / inf are dropped before quantile
        computation.
    lower_pct
        Lower quantile in ``[0, 0.5)``.
    upper_pct
        Upper quantile in ``(0.5, 1]``.

    Returns
    -------
    Band or None
        ``None`` when fewer than 2 finite values are available, or when
        the resulting band width is below :data:`DEGENERATE_BAND_HALF_WIDTH`.
    """
    if not (0.0 <= lower_pct < 0.5 < upper_pct <= 1.0):
        raise ValueError(
            f"Need 0 <= lower_pct ({lower_pct}) < 0.5 < upper_pct "
            f"({upper_pct}) <= 1."
        )
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size < 2:
        return None
    lo = float(np.quantile(finite, lower_pct))
    hi = float(np.quantile(finite, upper_pct))
    if (hi - lo) < DEGENERATE_BAND_HALF_WIDTH:
        return None
    return Band(
        column=column,
        lower=lo,
        upper=hi,
        quantile_lo=lower_pct,
        quantile_hi=upper_pct,
    )


# ---------------------------------------------------------------------------
# Policy 2 — joint-accept-rate auto-tune
# ---------------------------------------------------------------------------


def per_rule_quantile(target_accept_rate: float, n_rules: int) -> float:
    """Return symmetric per-rule trim that yields *target_accept_rate* jointly.

    Under the independence approximation, joint accept = product of
    per-rule accept rates::

        target = keep ** n_rules
        keep   = target ** (1 / n_rules)
        trim   = 1 - keep
        lower_pct = trim / 2  # symmetric split

    For ``target=0.90, n_rules=5``: keep ~0.979, trim ~0.021,
    lower_pct ~0.0105 → bands at p1.05/p98.95.  Much wider than the
    plain p5/p95 band (which on 5 rules expects ~59% joint accept).
    """
    if n_rules < 1:
        raise ValueError("n_rules must be >= 1.")
    p = float(np.clip(target_accept_rate, 1e-3, 0.999))
    keep = p ** (1.0 / n_rules)
    trim = max(0.0, 1.0 - keep)
    return float(trim / 2.0)


def tuned_bands(
    column_values: "Dict[str, Sequence[float]]",
    *,
    target_accept_rate: float = 0.90,
) -> List[Band]:
    """Per-column accept bands sized to hit *target_accept_rate* jointly.

    Two-pass algorithm:

    1. Pre-filter: drop columns whose p5/p95 band is already degenerate
       (they won't survive any tighter trim either).
    2. Compute the per-rule quantile from the survivor count, then
       compute each surviving column's actual band at that quantile.

    A column whose p5/p95 was non-degenerate but whose *tighter* band
    collapses is dropped in pass 2 — never produces a 0-width band.
    """
    survivors: List[Tuple[str, np.ndarray]] = []
    for column, values in column_values.items():
        arr = np.asarray(values, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size < 2:
            continue
        lo = float(np.quantile(finite, 0.05))
        hi = float(np.quantile(finite, 0.95))
        if (hi - lo) < DEGENERATE_BAND_HALF_WIDTH:
            continue
        survivors.append((column, finite))

    if not survivors:
        return []

    lower_pct = per_rule_quantile(target_accept_rate, n_rules=len(survivors))
    upper_pct = 1.0 - lower_pct

    bands: List[Band] = []
    for column, finite in survivors:
        lo = float(np.quantile(finite, lower_pct))
        hi = float(np.quantile(finite, upper_pct))
        if (hi - lo) < DEGENERATE_BAND_HALF_WIDTH:
            continue
        bands.append(
            Band(
                column=column,
                lower=lo,
                upper=hi,
                quantile_lo=lower_pct,
                quantile_hi=upper_pct,
                note=f"auto-tuned for joint accept ~{target_accept_rate:.0%}",
            )
        )
    return bands


# ---------------------------------------------------------------------------
# Strict-rule detector
# ---------------------------------------------------------------------------


def strictest_columns(
    per_rule_rejects: "Dict[str, int]",
    *,
    mad_multiplier: float = 3.0,
) -> List[str]:
    """Return rule names whose rejection count is an upward outlier.

    Uses the **modified Z-score** (median + ``mad_multiplier`` × MAD)
    instead of mean + k·std, because the latter is inflated by the
    very outlier we're trying to detect (Iglewicz & Hoaglin 1993).

    Returns an empty list when fewer than 3 rules are supplied (with 2,
    one is trivially the "outlier") or when every rule rejects the same
    number of segments (MAD == 0).

    Output is sorted by reject count, descending — caller can pop ``[0]``
    to get the worst offender.
    """
    if len(per_rule_rejects) < 3:
        return []
    counts = np.fromiter(per_rule_rejects.values(), dtype=float)
    med = float(np.median(counts))
    mad = float(np.median(np.abs(counts - med)))
    if mad < 1e-9:
        return []
    threshold = med + mad_multiplier * mad
    flagged = [name for name, count in per_rule_rejects.items() if count > threshold]
    flagged.sort(key=lambda n: per_rule_rejects[n], reverse=True)
    return flagged
