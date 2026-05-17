"""OUCRU CSV format auto-detection.

OUCRU exports store one row per second, with the signal column holding a
JSON-array (or bracket-less comma-separated) string of samples for that
second.  This module sniffs the first few rows of a CSV to recognise the
shape without loading the whole file, so callers can route to the right
loader without forcing the user to pick a format manually.
"""

from __future__ import annotations

import ast
import io
import json
import logging
import re
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_NP_FLOAT64_RE = re.compile(r"np\.float64\(([^)]+)\)")


PPG_SIGNAL_COLUMNS: tuple = ("pleth", "ppg", "red", "ir")
ECG_SIGNAL_COLUMNS: tuple = ("ecg", "ecg_signal", "ecg_data")
ALL_SIGNAL_COLUMNS: tuple = PPG_SIGNAL_COLUMNS + ECG_SIGNAL_COLUMNS

DEFAULT_PPG_RATE: float = 100.0
DEFAULT_ECG_RATE: float = 128.0


class OucruDetection(dict):
    """Result of :func:`detect_oucru_csv`.

    A plain dict subclass so it's still JSON-serialisable, with attribute
    access for ergonomics in calling code.
    """

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def parse_array_column(series: pd.Series) -> "tuple[np.ndarray, int]":
    """Vectorise OUCRU row-by-row parsing into one 2-D array.

    Each cell becomes a 1-D float array via :func:`parse_array_cell`.
    The function picks the sample-count from the first non-empty row
    and pads short / trims long rows to that length so the output is a
    clean ``(n_rows, n_samples_per_row)`` block ready for ``.ravel()``.
    Short rows are padded with their last value (edge mode), matching
    the legacy ``_load_oucru_csv`` behaviour - including the
    ``UserWarning`` emitted on length mismatch.

    Returns ``(block, n_samples_per_row)``.  Raises ``ValueError`` when
    a cell fails to parse or no rows are parseable.
    """
    import warnings as _warnings

    # Parse each cell.  Strings that look like array cells but contain
    # tokens that can't be floated (e.g. ``"[1.0, 1.1, invalid]"``)
    # raise here so the caller can produce a "Failed to parse signal
    # array" error matching the historic contract.  Plain numeric
    # strings become single-element arrays; non-numeric strings raise a
    # "Cannot convert" error matching the legacy single-value path.
    parsed: list = []
    for idx, cell in enumerate(series):
        if isinstance(cell, str) and (cell.lstrip().startswith("[") or "," in cell):
            arr = parse_array_cell(cell)
            if arr.size == 0:
                raise ValueError(
                    f"Failed to parse signal array at row {idx}: " f"{cell!r}"
                )
            parsed.append(arr)
        elif isinstance(cell, (list, tuple, np.ndarray)):
            parsed.append(np.asarray(cell, dtype=float))
        elif isinstance(cell, (int, float)) and not (
            isinstance(cell, float) and np.isnan(cell)
        ):
            parsed.append(np.asarray([float(cell)]))
        elif cell is None or (isinstance(cell, float) and np.isnan(cell)):
            parsed.append(np.array([], dtype=float))
        elif isinstance(cell, str):
            # Bare scalar string: legacy behaviour was to coerce via float()
            # and raise "Cannot convert '<value>' to float" on failure.
            try:
                parsed.append(np.asarray([float(cell)]))
            except ValueError:
                raise ValueError(f"Cannot convert {cell!r} to float")
        else:
            raise ValueError(
                f"Failed to parse signal array at row {idx}: "
                f"unexpected type {type(cell).__name__} ({cell!r})"
            )

    # First non-empty row determines the expected width.
    n_samples = 0
    for arr in parsed:
        if arr.size > 0:
            n_samples = int(arr.size)
            break
    if n_samples == 0:
        raise ValueError("OUCRU column has no parseable array rows.")

    rows = len(parsed)
    block = np.empty((rows, n_samples), dtype=float)
    for i, arr in enumerate(parsed):
        if arr.size == n_samples:
            block[i] = arr
        elif arr.size > n_samples:
            _warnings.warn(
                f"Inconsistent array length at row {i}: expected "
                f"{n_samples}, got {arr.size}. Padding/truncating to match."
            )
            block[i] = arr[:n_samples]
        elif arr.size > 0:
            _warnings.warn(
                f"Inconsistent array length at row {i}: expected "
                f"{n_samples}, got {arr.size}. Padding/truncating to match."
            )
            # Pad short rows with their last value (edge mode).
            block[i, : arr.size] = arr
            block[i, arr.size :] = arr[-1]
        else:
            # Empty row: fill with NaN so downstream interpolation can fix.
            block[i] = np.nan
    return block, n_samples


def parse_array_cell(cell: object) -> np.ndarray:
    """Parse a single OUCRU array cell into a 1-D float array.

    Accepts both the canonical JSON form ``"[1, 2, 3]"`` and the
    bracket-less comma-separated form ``"-439,-446,-446"`` seen in the ECG
    file's ``acc_x``/``acc_y``/``acc_z`` columns.  Returns an empty array
    for missing or unparseable values rather than raising — the caller
    decides whether that's an error.
    """
    if cell is None:
        return np.array([], dtype=float)
    if isinstance(cell, float) and np.isnan(cell):
        return np.array([], dtype=float)
    if isinstance(cell, (list, tuple, np.ndarray)):
        return np.asarray(cell, dtype=float)
    text = str(cell).strip()
    if not text:
        return np.array([], dtype=float)
    try:
        if text.startswith("["):
            # Strip ``np.float64(...)`` wrappers if present so json.loads
            # can parse the array literal.  Fall back to ast.literal_eval
            # for any weirder Python-repr cells.
            if "np.float64(" in text:
                cleaned = _NP_FLOAT64_RE.sub(r"\1", text)
                try:
                    return np.asarray(json.loads(cleaned), dtype=float)
                except (ValueError, json.JSONDecodeError):
                    return np.asarray(ast.literal_eval(cleaned), dtype=float)
            try:
                return np.asarray(json.loads(text), dtype=float)
            except (ValueError, json.JSONDecodeError):
                return np.asarray(ast.literal_eval(text), dtype=float)
        # Bracket-less form: split on commas, ignore empty pieces
        pieces = [p for p in text.split(",") if p.strip()]
        if not pieces:
            return np.array([], dtype=float)
        return np.asarray([float(p) for p in pieces], dtype=float)
    except (ValueError, SyntaxError, json.JSONDecodeError) as exc:
        logger.debug("Could not parse array cell %r: %s", text[:40], exc)
        return np.array([], dtype=float)


def _looks_like_array_cell(value: object) -> Optional[int]:
    """Return the parsed array length if `value` looks like an OUCRU cell.

    Returns None otherwise.  The threshold of ``>= 2`` elements prevents
    false positives on multi-word string columns (e.g. ``"male,en"``).
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.startswith("["):
        arr = parse_array_cell(text)
        return int(arr.size) if arr.size >= 1 else None
    if "," in text:
        arr = parse_array_cell(text)
        return int(arr.size) if arr.size >= 2 else None
    return None


def pick_signal_column(
    columns: Sequence[str],
    signal_type_hint: Optional[str] = None,
) -> Optional[str]:
    """Pick the most likely OUCRU signal column by name.

    Returns the first match against the preferred list for the given
    signal type (PPG / ECG), then falls back to any known OUCRU column
    name.  Case-insensitive.  Returns ``None`` if no preferred column
    name is present — the caller should then fall back to shape-based
    detection.
    """
    lookup = {c.lower(): c for c in columns}
    if signal_type_hint is not None:
        hint = signal_type_hint.lower()
        if hint in ("ppg", "photoplethysmography"):
            preferred = PPG_SIGNAL_COLUMNS
        elif hint in ("ecg", "electrocardiogram", "ekg"):
            preferred = ECG_SIGNAL_COLUMNS
        else:
            preferred = ALL_SIGNAL_COLUMNS
    else:
        preferred = ALL_SIGNAL_COLUMNS
    for name in preferred:
        if name in lookup:
            return lookup[name]
    return None


def detect_oucru_csv(
    source: Union[str, Path, pd.DataFrame, bytes, io.IOBase],
    signal_type_hint: Optional[str] = None,
    n_peek: int = 5,
) -> Optional[OucruDetection]:
    """Detect whether `source` is an OUCRU row-per-second CSV.

    Reads only the first ``n_peek`` rows from a file path / bytes / open
    handle, or examines a pre-loaded DataFrame.  Returns an
    :class:`OucruDetection` describing the detected shape if the file
    matches, else ``None``.

    Detection algorithm
    -------------------
    1. Try to pick a signal column by name (``pleth``/``ppg``/``ecg``…),
       biased by ``signal_type_hint`` if provided.
    2. If a named column is found and its first non-null cell parses as
       an array of >= 1 element (or >= 2 for bracket-less), report it.
    3. If no named column matches, scan all columns for one whose first
       non-null cell looks like an array — useful for non-canonical
       column names.
    """
    df = _peek_dataframe(source, n_peek=n_peek)
    if df is None or df.empty:
        return None

    # First pass: pick by name and confirm shape
    by_name = pick_signal_column(df.columns, signal_type_hint=signal_type_hint)
    if by_name is not None:
        result = _confirm_shape(df, by_name)
        if result is not None:
            return result

    # Second pass: scan columns for any array-shaped cell
    for col in df.columns:
        if col == by_name:
            continue
        result = _confirm_shape(df, col)
        if result is not None:
            return result

    return None


def _confirm_shape(df: pd.DataFrame, column: str) -> Optional[OucruDetection]:
    """Return detection result if `column` holds OUCRU array cells."""
    series = df[column].dropna()
    if series.empty:
        return None
    first_value = series.iloc[0]
    size = _looks_like_array_cell(first_value)
    if size is None or size < 1:
        return None
    text = str(first_value).strip()
    bracket_style = "json" if text.startswith("[") else "bare"
    return OucruDetection(
        is_oucru=True,
        signal_column=column,
        samples_per_row=int(size),
        bracket_style=bracket_style,
        columns=list(df.columns),
    )


def _peek_dataframe(
    source: Union[str, Path, pd.DataFrame, bytes, io.IOBase],
    n_peek: int,
) -> Optional[pd.DataFrame]:
    """Read the first ``n_peek`` rows of `source` into a DataFrame.

    Accepts a path, a raw bytes payload, an open file-like, or an
    already-loaded DataFrame.  Returns ``None`` on read failure rather
    than raising — detection should never crash a caller.
    """
    if isinstance(source, pd.DataFrame):
        return source.head(n_peek) if n_peek else source
    try:
        if isinstance(source, (bytes, bytearray)):
            return pd.read_csv(io.BytesIO(source), nrows=n_peek)
        if isinstance(source, io.IOBase):
            return pd.read_csv(source, nrows=n_peek)
        return pd.read_csv(source, nrows=n_peek)
    except Exception as exc:
        logger.debug("OUCRU peek failed for %r: %s", source, exc)
        return None


def default_rate_for_signal_type(signal_type_hint: Optional[str]) -> Optional[float]:
    """Return the conventional default sampling rate for PPG / ECG.

    ``None`` is returned for unknown / missing hints so the caller falls
    back to its own inference (e.g. array-length).
    """
    if signal_type_hint is None:
        return None
    hint = signal_type_hint.lower()
    if hint in ("ppg", "photoplethysmography"):
        return DEFAULT_PPG_RATE
    if hint in ("ecg", "electrocardiogram", "ekg"):
        return DEFAULT_ECG_RATE
    return None
