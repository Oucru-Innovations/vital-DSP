"""Column introspection for the upload page's signal-column dropdown.

Given a CSV (or other tabular) file the user just uploaded, list the
columns that could plausibly hold a signal, marking the recommended
choice based on signal type (PPG / ECG / ...).  Each option gets a
human-readable preview suffix (``array of 100 samples/row`` or
``numeric, 50 rows``) so the user can see at a glance how big each
candidate column is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from vitalDSP.utils.data_processing.oucru_detect import (
    ALL_SIGNAL_COLUMNS,
    ECG_SIGNAL_COLUMNS,
    PPG_SIGNAL_COLUMNS,
    parse_array_cell,
)

logger = logging.getLogger(__name__)


@dataclass
class ColumnCandidate:
    """A column that could plausibly hold the signal."""

    name: str
    kind: str  # "array" (OUCRU row-per-second) or "numeric"
    sample_size: int  # array: elements per row; numeric: row count
    preferred: bool  # True if this matches the signal type's default name


def introspect_columns(
    file_path: str,
    signal_type: Optional[str] = None,
    n_peek: int = 5,
) -> List[ColumnCandidate]:
    """Return the columns that could hold a signal, recommended-first.

    Reads only the first ``n_peek`` rows of the file.  Returns an empty
    list on any error or for non-CSV files — the caller should fall
    back to an unfiltered "all columns" dropdown in that case.
    """
    try:
        df = pd.read_csv(file_path, nrows=n_peek)
    except Exception as exc:
        logger.debug("Column introspection failed for %s: %s", file_path, exc)
        return []
    if df.empty:
        return []
    return _build_candidates(df, signal_type)


def _build_candidates(
    df: pd.DataFrame, signal_type: Optional[str]
) -> List[ColumnCandidate]:
    preferred_names = _preferred_names_for(signal_type)
    rank_by_lower = {name.lower(): i for i, name in enumerate(preferred_names)}

    candidates: List[ColumnCandidate] = []
    for column in df.columns:
        series = df[column].dropna()
        if series.empty:
            continue
        first = series.iloc[0]
        # OUCRU-style array cell?
        if isinstance(first, str) and (
            first.lstrip().startswith("[")
            or ("," in first and len(parse_array_cell(first)) >= 2)
        ):
            arr = parse_array_cell(first)
            if arr.size < 1:
                continue
            candidates.append(
                ColumnCandidate(
                    name=column,
                    kind="array",
                    sample_size=int(arr.size),
                    preferred=column.lower() in rank_by_lower,
                )
            )
        elif pd.api.types.is_numeric_dtype(df[column]):
            candidates.append(
                ColumnCandidate(
                    name=column,
                    kind="numeric",
                    sample_size=int(len(df)),
                    preferred=False,
                )
            )

    candidates.sort(key=lambda c: _sort_key(c, rank_by_lower))
    return candidates


def _preferred_names_for(signal_type: Optional[str]) -> tuple:
    if signal_type is None:
        return ALL_SIGNAL_COLUMNS
    hint = signal_type.lower()
    if hint in ("ppg", "photoplethysmography"):
        return PPG_SIGNAL_COLUMNS
    if hint in ("ecg", "electrocardiogram", "ekg"):
        return ECG_SIGNAL_COLUMNS
    return ALL_SIGNAL_COLUMNS


def _sort_key(c: ColumnCandidate, rank_by_lower: dict) -> tuple:
    rank = rank_by_lower.get(c.name.lower())
    if rank is not None:
        return (0, rank, c.name.lower())
    return (1, 0, c.name.lower())


def candidate_to_option(c: ColumnCandidate) -> dict:
    """Turn a ColumnCandidate into a Dash dropdown option."""
    if c.kind == "array":
        preview = f"array of {c.sample_size} samples/row"
    else:
        preview = f"numeric, {c.sample_size} rows"
    suffix = "  (recommended)" if c.preferred else ""
    return {
        "label": f"{c.name}  -  {preview}{suffix}",
        "value": c.name,
    }
