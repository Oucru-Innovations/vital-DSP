"""Segment-quality callbacks.

Wires the Segment-quality card on the filtering page AND the new
accordion section on the Quality page.  Both surfaces share the same
``store-segment-decisions`` / ``store-segment-sqis`` /
``store-segment-milestones`` stores.

Pipeline on Apply (filtering page) or any control change:

1. Read the full filtered signal from ``store-filtered-signal``.
2. ``compute_segment_sqis`` on the chosen segment length + overlap.
3. ``classify_segments`` with the chosen mode + slider + checklist.
4. Write the three shared stores; render timeline + headline.

Slider / mode / checklist edits *do not* re-run the filter — they just
re-classify against the cached SQI DataFrame stored in
``store-segment-sqis``.  That keeps the panel feel live without
triggering the big Apply pipeline.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, html, dcc, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from vitalDSP.signal_quality_assessment import (
    AVAILABLE_SQIS,
    DEFAULT_SEGMENT_SQIS,
    candidate_rule_columns,
    classify_segments,
    compute_segment_sqis,
    load_rule_dict,
    per_rule_reject_counts,
    strictest_columns,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ACCEPT_COLOR = "#198754"   # bootstrap success green
_REJECT_COLOR = "#dc3545"   # bootstrap danger red
_NEUTRAL_COLOR = "#0d6efd"  # bootstrap primary blue — scoring off
_UNKNOWN_COLOR = "#adb5bd"  # bootstrap secondary grey

#: Segment-length options shown in the dropdown.  Auto-sync snaps the
#: window-duration value (seconds) to the nearest entry.
_LENGTH_OPTIONS = (5, 10, 15, 30, 60)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _nearest_length(window_duration: Optional[float]) -> int:
    """Snap a window duration (seconds) to the nearest segment-length option."""
    if window_duration is None:
        return 30
    try:
        d = float(window_duration)
    except (TypeError, ValueError):
        return 30
    if d <= 0:
        return 30
    return min(_LENGTH_OPTIONS, key=lambda v: abs(v - d))


def _make_timeline_figure(decisions: List[dict]) -> go.Figure:
    """One Bar trace, one cell per segment, coloured by decision."""
    if not decisions:
        return go.Figure(
            layout={
                "annotations": [{
                    "text": "No segments yet",
                    "showarrow": False,
                    "xref": "paper", "yref": "paper",
                    "x": 0.5, "y": 0.5,
                }],
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
                "margin": {"l": 0, "r": 0, "t": 4, "b": 4},
                "height": 80,
            }
        )
    n = len(decisions)
    def _color_for(d):
        decision = d.get("decision")
        if decision == "accept":
            return _ACCEPT_COLOR
        if decision == "reject":
            return _REJECT_COLOR
        if decision == "neutral":
            return _NEUTRAL_COLOR
        return _UNKNOWN_COLOR
    colors = [_color_for(d) for d in decisions]
    hover = [d.get("decision", "?") for d in decisions]
    fig = go.Figure(
        go.Bar(
            x=list(range(n)),
            y=[1] * n,
            marker={"color": colors},
            hovertemplate="<b>Segment %{x}</b><br>%{customdata}<extra></extra>",
            customdata=hover,
            width=1.0,
        )
    )
    fig.update_layout(
        showlegend=False,
        margin={"l": 0, "r": 0, "t": 4, "b": 20},
        bargap=0,
        plot_bgcolor="#ffffff",
        xaxis={
            "title": "Segment index",
            "showgrid": False,
            "zeroline": False,
            "fixedrange": True,
        },
        yaxis={
            "showgrid": False,
            "showticklabels": False,
            "zeroline": False,
            "range": [0, 1],
            "fixedrange": True,
        },
        height=80,
    )
    return fig


def _headline_text(decisions: List[dict], n_total: int, segment_seconds: int) -> Any:
    if not decisions:
        return html.Span(
            "Apply Filter to compute segment quality on the full recording.",
            className="text-muted",
        )
    # Neutral (scoring off): just report the segmentation, no accept/reject.
    if all(d.get("decision") == "neutral" for d in decisions):
        return html.Span(
            f"{n_total} segments of {segment_seconds}s   click to filter",
            className="small",
        )
    accepted = sum(1 for d in decisions if d.get("decision") == "accept")
    pct = (100.0 * accepted / len(decisions)) if decisions else 0.0
    return html.Span(
        f"{n_total} segments of {segment_seconds}s   "
        f"Accepted: {accepted}/{len(decisions)} ({pct:.0f}%)",
        className="small",
    )


def _checklist_options_and_value(
    sqi_df: pd.DataFrame,
    rule_dict: Dict[str, dict],
    signal_type: str,
    current_value: Optional[List[str]],
) -> Tuple[List[dict], List[str], str, str]:
    """Build (options, value, skipped_text, summary_text) for the checklist."""
    if sqi_df is None or sqi_df.empty or not rule_dict:
        return [], [], "", "no data"

    candidates = candidate_rule_columns(sqi_df, rule_dict)
    options = [
        {
            "label": (f" {c['name']}" if c["usable"] else f" {c['name']}  ✗ {c['reason']}"),
            "value": c["name"],
            "disabled": not c["usable"],
        }
        for c in candidates
    ]
    usable_names = [c["name"] for c in candidates if c["usable"]]

    defaults = [
        n for n in DEFAULT_SEGMENT_SQIS.get(signal_type.upper(), ())
        if n in usable_names
    ]

    # Preserve current selection where still usable; else fall back to defaults
    if current_value:
        retained = [c for c in current_value if c in usable_names]
        value = retained or defaults
    else:
        value = defaults

    skipped = [c for c in candidates if not c["usable"]]
    if skipped:
        skipped_text = "Auto-skipped: " + ", ".join(c["name"] for c in skipped)
    else:
        skipped_text = ""
    summary_text = (
        f"{len(usable_names)} usable, {len(value)} active"
    )
    return options, value, skipped_text, summary_text


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def register_segment_quality_callbacks(app):
    """Register all segment-quality callbacks on the Dash *app*."""

    # ------------------------------------------------------------------
    # 1) Slider visibility tied to mode
    # ------------------------------------------------------------------
    @app.callback(
        [
            Output("filter-segment-tune-row", "style"),
            Output("filter-segment-quantile-row", "style"),
        ],
        Input("filter-segment-mode", "value"),
        prevent_initial_call=False,
    )
    def toggle_slider_visibility(mode):
        hidden = {"display": "none"}
        visible = {"display": "block"}
        if mode == "tune":
            return visible, hidden
        if mode == "quantile":
            return hidden, visible
        return hidden, hidden  # manual

    # ------------------------------------------------------------------
    # 3) Heavy path: recompute SQIs over the WHOLE recording with the
    #    saved filter chain replayed end-to-end.
    #
    # We do NOT use ``store-filtered-signal`` here because that store
    # may contain only the displayed window slice.  Instead:
    #   • read the full original signal from the data service,
    #   • read the saved filter chain (``filter_info["chain"]``),
    #   • replay the chain on the full signal,
    #   • segment the result.
    # ------------------------------------------------------------------
    @app.callback(
        [
            Output("store-segment-sqis", "data"),
            Output("store-segment-milestones", "data"),
            Output("store-segment-filtered-signal", "data"),
        ],
        [
            # Recompute when:
            #   • the user lands on the filtering page (URL change) so the
            #     timeline + comparison plot show something before Apply,
            #   • the apply callback finishes (new chain),
            #   • the user changes segment length / overlap / scoring toggle.
            Input("url", "pathname"),
            Input("store-filtered-signal", "data"),
            Input("filter-segment-length", "value"),
            Input("filter-segment-overlap", "value"),
            Input("filter-segment-scoring-enabled", "value"),
        ],
        prevent_initial_call=False,
    )
    def compute_sqis(pathname, _filter_apply_signal, segment_length, overlap_pct, scoring_enabled):
        # The URL Input fires on every page change; only run when we're
        # on the filtering page (the only place these IDs exist).
        if pathname and pathname not in ("/filtering", "/"):
            raise PreventUpdate
        try:
            from vitalDSP_webapp.services.data.enhanced_data_service import (
                get_enhanced_data_service,
            )
            from vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks import (
                apply_filter_chain,
            )
        except Exception as exc:
            logger.warning("Cannot import data_service / chain helper: %s", exc)
            raise PreventUpdate

        data_service = get_enhanced_data_service()
        all_data = data_service.get_all_data() if hasattr(data_service, "get_all_data") else {}
        if not all_data:
            raise PreventUpdate
        # Latest data id == last key inserted (mirrors how other pages do it).
        latest_id = list(all_data.keys())[-1]
        df = data_service.get_data(latest_id)
        data_info = data_service.get_data_info(latest_id) or {}
        column_mapping = data_service.get_column_mapping(latest_id) or {}
        if df is None or df.empty:
            raise PreventUpdate

        signal_column = (
            column_mapping.get("signal")
            or column_mapping.get("amplitude")
            or column_mapping.get("value")
        )
        if not signal_column and len(df.columns) >= 2:
            signal_column = df.columns[1]
        elif not signal_column and len(df.columns) == 1:
            signal_column = df.columns[0]
        if signal_column is None or signal_column not in df.columns:
            logger.warning("compute_sqis: could not resolve signal column.")
            raise PreventUpdate

        full_signal = np.asarray(df[signal_column].values, dtype=float)
        sampling_freq = (
            data_info.get("sampling_freq")
            or data_info.get("sampling_rate")
            or 1000
        )
        try:
            sampling_freq = float(sampling_freq)
        except (TypeError, ValueError):
            sampling_freq = 1000.0
        if sampling_freq <= 0:
            sampling_freq = 1000.0

        # FAST PATH — scoring disabled: just compute segment milestones
        # so the timeline has something to display.  No SQI compute, no
        # chain replay on the whole signal.  The per-segment filtering
        # happens on demand when the user clicks a cell (see
        # ``apply_filter_to_picked_segment`` below).
        seg_s = int(segment_length or 30)
        overlap = float(overlap_pct or 0) / 100.0
        if not scoring_enabled:
            from vitalDSP.signal_quality_assessment.segment_sqi import _segment_indices
            window_samples = max(1, int(round(seg_s * sampling_freq)))
            step_samples = max(1, int(round(window_samples * (1.0 - overlap))))
            spans = _segment_indices(full_signal.size, window_samples, step_samples)
            milestones = [
                {
                    "start_idx": int(s), "end_idx": int(e),
                    "t_start": float(s / sampling_freq),
                    "t_end": float(e / sampling_freq),
                }
                for s, e in spans
            ]
            logger.info(
                "Scoring OFF: %d neutral segments (seg=%ds, overlap=%d%%, fs=%g).",
                len(milestones), seg_s, int(overlap * 100), sampling_freq,
            )
            # No SQI table, no whole-signal filtered cache; just
            # milestones.  ``store-segment-sqis`` set to ``None`` is the
            # signal to the reclassify callback that scoring is off.
            return None, milestones, None

        # FULL PATH — scoring enabled: replay the chain on the full
        # signal, compute SQIs, downstream reclassify produces the
        # green/red decisions.
        filter_info = data_service.get_filter_info(latest_id) or {}
        saved_chain = filter_info.get("chain") or []
        signal_type = data_info.get("signal_type", "PPG")
        filtered_full = None
        try:
            if saved_chain:
                filtered_full = apply_filter_chain(
                    full_signal, sampling_freq, signal_type, saved_chain,
                    logger=logger,
                )
                logger.info(
                    "compute_sqis: replayed %d-stage chain on full %d-sample signal.",
                    len(saved_chain), full_signal.size,
                )
            else:
                cached = data_service.get_filtered_data(latest_id)
                if cached is not None and len(cached) == len(full_signal):
                    filtered_full = np.asarray(cached, dtype=float)
                    logger.info(
                        "compute_sqis: no saved chain; using cached filtered signal."
                    )
                else:
                    filtered_full = full_signal
                    logger.info(
                        "compute_sqis: no saved chain or cached full signal; "
                        "using original signal."
                    )
        except Exception as exc:
            logger.exception("compute_sqis: chain replay failed: %s", exc)
            filtered_full = full_signal

        try:
            sqi_df, milestones = compute_segment_sqis(
                filtered_full, sampling_freq=sampling_freq,
                segment_seconds=seg_s, overlap_pct=overlap,
            )
        except Exception as exc:
            logger.exception("compute_segment_sqis failed: %s", exc)
            return None, None, None

        logger.info(
            "Segment SQIs: %d segments x %d SQIs (seg=%ds, overlap=%d%%, fs=%g).",
            len(sqi_df), sqi_df.shape[1], seg_s, int(overlap * 100), sampling_freq,
        )

        # Cache the full filtered signal for the Quality page's
        # per-segment waveform inspector.
        filtered_payload = {
            "signal": filtered_full.tolist(),
            "sampling_freq": sampling_freq,
            "n_samples": int(filtered_full.size),
        }
        return sqi_df.to_dict("records"), milestones, filtered_payload

    # ------------------------------------------------------------------
    # 4) Populate the SQI checklist when the SQI DataFrame changes
    # ------------------------------------------------------------------
    @app.callback(
        [
            Output("filter-segment-rules-checklist", "options"),
            Output("filter-segment-rules-checklist", "value"),
            Output("filter-segment-rules-skipped", "children"),
            Output("filter-segment-rules-summary", "children"),
        ],
        [
            # Fires on initial page load (URL) so the checklist is
            # populated even when scoring is off and sqi_payload is None.
            Input("url", "pathname"),
            Input("store-segment-sqis", "data"),
            Input("filter-signal-type-select", "value"),
        ],
        State("filter-segment-rules-checklist", "value"),
        prevent_initial_call=False,
    )
    def populate_checklist(pathname, sqi_payload, signal_type, current_value):
        if pathname and pathname not in ("/filtering", "/"):
            raise PreventUpdate
        rule_dict = load_rule_dict((signal_type or "PPG").upper())
        if sqi_payload:
            # Scoring on — use real SQI distribution to mark degenerate
            # columns as disabled with a reason.
            df = pd.DataFrame(sqi_payload)
            options, value, skipped, summary = _checklist_options_and_value(
                df, rule_dict, signal_type or "PPG", current_value,
            )
            return options, value, skipped, summary
        # Scoring off — show every SQI the rule dict knows about as a
        # plain enabled option (no usability check, since we haven't
        # computed values to inspect).
        options = [
            {"label": f" {name}", "value": name, "disabled": False}
            for name in rule_dict.keys()
        ]
        usable_names = list(rule_dict.keys())
        defaults = [
            n for n in DEFAULT_SEGMENT_SQIS.get(
                (signal_type or "PPG").upper(), ()
            ) if n in usable_names
        ]
        if current_value:
            retained = [c for c in current_value if c in usable_names]
            value = retained or defaults
        else:
            value = defaults
        summary = f"{len(usable_names)} available, {len(value)} active"
        return options, value, "", summary

    # ------------------------------------------------------------------
    # 5) Light path: re-classify when the user nudges any control
    # ------------------------------------------------------------------
    @app.callback(
        [
            Output("store-segment-decisions", "data"),
            Output("filter-segment-timeline", "figure"),
            Output("filter-segment-headline", "children"),
        ],
        [
            Input("store-segment-sqis", "data"),
            Input("store-segment-milestones", "data"),
            Input("filter-segment-mode", "value"),
            Input("filter-segment-tune-slider", "value"),
            Input("filter-segment-quantile-slider", "value"),
            Input("filter-segment-rules-checklist", "value"),
        ],
        [
            State("filter-signal-type-select", "value"),
            State("filter-segment-length", "value"),
        ],
        prevent_initial_call=False,
    )
    def reclassify(
        sqi_payload, milestones,
        mode, tune_target, quantile_trim, selected_sqis,
        signal_type, segment_length,
    ):
        # Scoring OFF: SQIs are not computed.  Emit a neutral
        # decisions list driven purely by the segment count from
        # milestones so the timeline still draws something.
        if not sqi_payload:
            if not milestones:
                return [], _make_timeline_figure([]), _headline_text(
                    [], 0, int(segment_length or 30),
                )
            decisions = [{"decision": "neutral", "trace": []} for _ in milestones]
            fig = _make_timeline_figure(decisions)
            headline = _headline_text(
                decisions, len(milestones), int(segment_length or 30),
            )
            return decisions, fig, headline

        df = pd.DataFrame(sqi_payload)
        if df.empty:
            return [], _make_timeline_figure([]), _headline_text([], 0, int(segment_length or 30))

        try:
            q_trim = float(np.clip(quantile_trim or 0.05, 0.0, 0.49))
            tune = float(np.clip(tune_target or 0.90, 0.5, 0.99))
            decisions = classify_segments(
                df,
                signal_type=(signal_type or "PPG").upper(),
                selected_columns=selected_sqis or None,
                mode=mode or "tune",
                quantile_lo=q_trim,
                quantile_hi=1.0 - q_trim,
                target_accept_rate=tune,
            )
        except Exception as exc:
            logger.exception("classify_segments failed: %s", exc)
            return no_update, no_update, html.Span(
                f"Classification failed: {exc}", className="text-danger small",
            )

        fig = _make_timeline_figure(decisions)
        headline = _headline_text(decisions, len(df), int(segment_length or 30))
        return decisions, fig, headline

    # ------------------------------------------------------------------
    # 7) Quality-page accordion: timeline mirror + segment detail
    # ------------------------------------------------------------------
    @app.callback(
        [
            Output("quality-segment-timeline", "figure"),
            Output("quality-segment-summary", "children"),
        ],
        Input("store-segment-decisions", "data"),
        prevent_initial_call=False,
    )
    def render_quality_timeline(decisions):
        if not decisions:
            return _make_timeline_figure([]), html.Em(
                "Apply a filter on the Filtering page to populate segment quality.",
                className="text-muted",
            )
        accepted = sum(1 for d in decisions if d.get("decision") == "accept")
        n = len(decisions)
        pct = 100.0 * accepted / n if n else 0.0
        return _make_timeline_figure(decisions), html.Span(
            f"{accepted}/{n} accepted ({pct:.0f}%)  •  click a cell to inspect",
            className="small text-muted",
        )

    @app.callback(
        Output("quality-segment-picker", "options"),
        Output("quality-segment-picker", "value"),
        [
            Input("store-segment-decisions", "data"),
            Input("quality-segment-filter", "value"),
        ],
        State("quality-segment-picker", "value"),
        prevent_initial_call=True,
    )
    def populate_picker(decisions, filter_mode, current_value):
        if not decisions:
            return [], None
        indices = [
            i for i, d in enumerate(decisions)
            if filter_mode == "all" or d.get("decision") == filter_mode
        ]
        options = [
            {
                "label": f"#{i:04d}  ({decisions[i].get('decision', '?')})",
                "value": i,
            }
            for i in indices
        ]
        if current_value in indices:
            value = current_value
        else:
            value = indices[0] if indices else None
        return options, value

    @app.callback(
        Output("quality-segment-picker", "value", allow_duplicate=True),
        Input("quality-segment-timeline", "clickData"),
        State("store-segment-decisions", "data"),
        prevent_initial_call=True,
    )
    def click_to_pick(click_data, decisions):
        if not click_data or not decisions:
            raise PreventUpdate
        try:
            idx = int(click_data["points"][0]["x"])
        except (KeyError, IndexError, TypeError, ValueError):
            raise PreventUpdate
        if 0 <= idx < len(decisions):
            return idx
        raise PreventUpdate

    @app.callback(
        [
            Output("quality-segment-waveform", "figure"),
            Output("quality-segment-sqi-table", "children"),
            Output("quality-segment-rule-trace", "children"),
        ],
        Input("quality-segment-picker", "value"),
        [
            State("store-segment-decisions", "data"),
            State("store-segment-sqis", "data"),
            State("store-segment-milestones", "data"),
            # Use the segment-quality compute's own whole-recording
            # filtered cache, not the windowed ``store-filtered-signal``.
            State("store-segment-filtered-signal", "data"),
        ],
        prevent_initial_call=True,
    )
    def render_segment_detail(seg_idx, decisions, sqi_payload, milestones, filtered_payload):
        if seg_idx is None or not decisions:
            return go.Figure(), html.Em("No segment selected."), html.Em("")
        seg_idx = int(seg_idx)
        if seg_idx < 0 or seg_idx >= len(decisions):
            return go.Figure(), html.Em("Out of range."), html.Em("")

        decision_entry = decisions[seg_idx]
        decision = decision_entry.get("decision", "unknown")
        trace = decision_entry.get("trace", [])

        # Waveform — slice the full filtered signal by the segment milestones
        waveform_fig = go.Figure()
        if milestones and filtered_payload and seg_idx < len(milestones):
            try:
                signal = (
                    filtered_payload.get("filtered_signal")
                    or filtered_payload.get("signal")
                    or filtered_payload.get("data")
                )
                fs = (
                    filtered_payload.get("sampling_freq")
                    or filtered_payload.get("sampling_rate")
                    or 1.0
                )
                if signal is not None:
                    arr = np.asarray(signal, dtype=float)
                    ms = milestones[seg_idx]
                    s, e = int(ms["start_idx"]), int(ms["end_idx"])
                    s = max(0, min(s, arr.size))
                    e = max(s, min(e, arr.size))
                    samples = arr[s:e]
                    t = np.arange(samples.size) / float(fs) + (s / float(fs))
                    waveform_fig = go.Figure(
                        go.Scattergl(x=t, y=samples, mode="lines", name="segment")
                    )
                    waveform_fig.update_layout(
                        title=f"Segment {seg_idx} - {decision}",
                        xaxis_title="Time (s)",
                        margin={"l": 40, "r": 10, "t": 30, "b": 30},
                        height=280,
                    )
            except Exception as exc:
                logger.warning("Could not draw segment waveform: %s", exc)
                waveform_fig = go.Figure(layout={"title": "Waveform unavailable"})

        # SQI value table — one row per SQI
        sqi_table_children = html.Em("No SQI values cached.")
        if sqi_payload and seg_idx < len(sqi_payload):
            sqi_row = sqi_payload[seg_idx]
            sqi_table_children = dbc.Table(
                [
                    html.Thead(html.Tr([html.Th("SQI"), html.Th("Value", style={"textAlign": "right"})])),
                    html.Tbody([
                        html.Tr([
                            html.Td(name),
                            html.Td(
                                f"{float(val):.4f}" if isinstance(val, (int, float)) and val == val else str(val),
                                style={"textAlign": "right"},
                            ),
                        ])
                        for name, val in sqi_row.items()
                    ]),
                ],
                striped=True, bordered=False, hover=True, size="sm",
            )

        # Rule trace — first reject marked as decisive
        if not trace:
            trace_node = html.Em(
                "No rules applied to this recording.",
            )
        else:
            decisive_marked = False
            rows = []
            for entry in trace:
                outcome = entry.get("outcome", "?")
                is_decisive = outcome != "accept" and not decisive_marked
                if is_decisive:
                    decisive_marked = True
                cls = "table-danger" if is_decisive else (
                    "" if outcome == "accept" else "text-muted"
                )
                val = entry.get("value")
                try:
                    val_str = f"{float(val):.4f}"
                except (TypeError, ValueError):
                    val_str = str(val)
                rows.append(html.Tr(
                    [
                        html.Td(entry.get("name", "?")),
                        html.Td(val_str),
                        html.Td(outcome.upper(), style={"fontWeight": "bold" if is_decisive else "normal"}),
                        html.Td("← decisive" if is_decisive else "", className="text-muted small"),
                    ],
                    className=cls,
                ))
            trace_node = dbc.Table(
                [
                    html.Thead(html.Tr([
                        html.Th("Rule"), html.Th("Value"),
                        html.Th("Outcome"), html.Th(""),
                    ])),
                    html.Tbody(rows),
                ],
                striped=True, bordered=False, hover=True, size="sm",
            )

        return waveform_fig, sqi_table_children, trace_node

    # ------------------------------------------------------------------
    # 8) Drop-strictest button — Quality page only
    # ------------------------------------------------------------------
    @app.callback(
        [
            Output("quality-segment-drop-strictest", "disabled"),
            Output("quality-segment-strictest-hint", "children"),
        ],
        Input("store-segment-decisions", "data"),
        prevent_initial_call=False,
    )
    def update_drop_button(decisions):
        if not decisions:
            return True, ""
        counts = per_rule_reject_counts(decisions)
        if not counts:
            return True, ""
        flagged = strictest_columns(counts)
        if not flagged:
            return True, "no clear outlier"
        worst = flagged[0]
        return False, f"would drop: {worst} (rejected {counts[worst]} segments)"

    @app.callback(
        Output("filter-segment-rules-checklist", "value", allow_duplicate=True),
        Input("quality-segment-drop-strictest", "n_clicks"),
        State("store-segment-decisions", "data"),
        State("filter-segment-rules-checklist", "value"),
        prevent_initial_call=True,
    )
    def drop_strictest(n_clicks, decisions, current):
        if not n_clicks or not decisions or not current:
            raise PreventUpdate
        counts = per_rule_reject_counts(decisions)
        flagged = strictest_columns(counts)
        if not flagged:
            raise PreventUpdate
        worst = flagged[0]
        return [c for c in current if c != worst]

    # ------------------------------------------------------------------
    # 9) Bridge — copy the hidden timeline / headline to the top-bar
    #    widgets that actually face the user.
    # ------------------------------------------------------------------
    @app.callback(
        [
            Output("filter-segment-timeline-top", "figure"),
            Output("filter-segment-headline-top", "children"),
        ],
        Input("store-segment-decisions", "data"),
        State("filter-segment-length", "value"),
        prevent_initial_call=False,
    )
    def bridge_to_top_bar(decisions, segment_length):
        if not decisions:
            return _make_timeline_figure([]), html.Span(
                "Upload data to see segment quality across the recording.",
                className="text-muted",
            )
        fig = _make_timeline_figure(decisions)
        headline = _headline_text(decisions, len(decisions), int(segment_length or 30))
        return fig, headline

    # ------------------------------------------------------------------
    # 10) Top-bar timeline click → drive the hidden start-position-slider
    #     so the existing apply callback's start_time math just works,
    #     and remember the picked segment index for the preview below.
    # ------------------------------------------------------------------
    @app.callback(
        [
            Output("start-position-slider", "value", allow_duplicate=True),
            Output("store-picked-segment", "data"),
        ],
        Input("filter-segment-timeline-top", "clickData"),
        [
            State("store-segment-milestones", "data"),
            State("store-segment-filtered-signal", "data"),
        ],
        prevent_initial_call=True,
    )
    def timeline_click_to_position(click_data, milestones, filtered_payload):
        if not click_data or not milestones:
            raise PreventUpdate
        try:
            idx = int(click_data["points"][0]["x"])
        except (KeyError, IndexError, TypeError, ValueError):
            raise PreventUpdate
        if idx < 0 or idx >= len(milestones):
            raise PreventUpdate
        start_idx = float(milestones[idx].get("start_idx", 0))
        # Fraction of the recording the segment starts at.  Fall back to
        # segment-index/segment-count when we don't know the recording
        # length yet (no filtered cache).
        if filtered_payload and filtered_payload.get("n_samples"):
            n_samples = float(filtered_payload["n_samples"]) or 1.0
            pct = 100.0 * start_idx / n_samples
        else:
            pct = 100.0 * idx / max(1, len(milestones))
        pct = max(0.0, min(100.0, pct))
        return pct, idx

    # ------------------------------------------------------------------
    # 11) Comparison-plot preview — fires on timeline click or picked-
    #     segment change.  Re-runs the saved filter chain on just that
    #     ONE segment of the original recording, then renders the
    #     before/after into ``filter-comparison-plot``.  This is cheap
    #     (one segment) so the user gets instant feedback when clicking
    #     around the timeline, without touching the full-signal Apply
    #     pipeline.
    # ------------------------------------------------------------------
    @app.callback(
        Output("filter-comparison-plot", "figure", allow_duplicate=True),
        [
            Input("store-picked-segment", "data"),
            Input("store-segment-milestones", "data"),
        ],
        State("filter-signal-type-select", "value"),
        prevent_initial_call=True,
    )
    def preview_segment_in_comparison_plot(seg_idx, milestones, signal_type):
        if seg_idx is None or not milestones:
            raise PreventUpdate
        try:
            seg_idx = int(seg_idx)
        except (TypeError, ValueError):
            raise PreventUpdate
        if seg_idx < 0 or seg_idx >= len(milestones):
            raise PreventUpdate

        try:
            from vitalDSP_webapp.services.data.enhanced_data_service import (
                get_enhanced_data_service,
            )
            from vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks import (
                apply_filter_chain,
                create_filter_comparison_plot,
            )
        except Exception as exc:
            logger.warning("preview: cannot import helpers: %s", exc)
            raise PreventUpdate

        ds = get_enhanced_data_service()
        all_data = ds.get_all_data() if hasattr(ds, "get_all_data") else {}
        if not all_data:
            raise PreventUpdate
        latest_id = list(all_data.keys())[-1]
        df = ds.get_data(latest_id)
        data_info = ds.get_data_info(latest_id) or {}
        column_mapping = ds.get_column_mapping(latest_id) or {}
        if df is None or df.empty:
            raise PreventUpdate

        signal_column = (
            column_mapping.get("signal")
            or column_mapping.get("amplitude")
            or column_mapping.get("value")
        )
        if not signal_column and len(df.columns) >= 2:
            signal_column = df.columns[1]
        elif not signal_column and len(df.columns) == 1:
            signal_column = df.columns[0]
        if signal_column is None or signal_column not in df.columns:
            raise PreventUpdate

        sampling_freq = (
            data_info.get("sampling_freq")
            or data_info.get("sampling_rate")
            or 1000
        )
        try:
            sampling_freq = float(sampling_freq)
        except (TypeError, ValueError):
            sampling_freq = 1000.0
        if sampling_freq <= 0:
            sampling_freq = 1000.0

        ms = milestones[seg_idx]
        s = int(ms.get("start_idx", 0))
        e = int(ms.get("end_idx", 0))
        full_signal = np.asarray(df[signal_column].values, dtype=float)
        s = max(0, min(s, full_signal.size))
        e = max(s, min(e, full_signal.size))
        original_segment = full_signal[s:e]
        if original_segment.size < 2:
            raise PreventUpdate

        filter_info = ds.get_filter_info(latest_id) or {}
        saved_chain = filter_info.get("chain") or []
        sig_type = (signal_type or data_info.get("signal_type") or "PPG").upper()
        try:
            if saved_chain:
                filtered_segment = apply_filter_chain(
                    original_segment, sampling_freq, sig_type, saved_chain,
                    logger=logger,
                )
            else:
                # No saved chain — show the raw segment in both traces.
                # Better than an empty plot; the user knows nothing has
                # been applied yet.
                filtered_segment = original_segment
        except Exception as exc:
            logger.warning("preview: chain replay failed on segment %d: %s", seg_idx, exc)
            filtered_segment = original_segment

        time_axis = np.arange(original_segment.size) / sampling_freq + (s / sampling_freq)

        # Critical points overlay — only when a real filter chain has
        # produced ``filtered_segment``.  Running WaveformMorphology on
        # the raw original signal (no chain saved) gives spurious peaks
        # because the detector assumes baseline-corrected, band-limited
        # input.  Show no markers in that case instead of misleading
        # ones — the user knows to press Apply.
        peaks = None
        notches = None
        if saved_chain:
            try:
                from vitalDSP.physiological_features.waveform import (
                    WaveformMorphology,
                )
                wm = WaveformMorphology(
                    waveform=np.asarray(filtered_segment, dtype=float),
                    fs=sampling_freq,
                    signal_type=sig_type,
                    simple_mode=True,
                )
                peaks = getattr(wm, "systolic_peaks", None)
                notches = getattr(wm, "dicrotic_notches", None)
            except Exception as exc:
                logger.debug("preview: WaveformMorphology skipped: %s", exc)

        try:
            fig = create_filter_comparison_plot(
                time_axis=time_axis,
                original_signal=original_segment,
                filtered_signal=filtered_segment,
                sampling_freq=sampling_freq,
                signal_type=sig_type,
                peaks=peaks,
                notches=notches,
            )
        except Exception as exc:
            logger.warning("preview: create_filter_comparison_plot failed: %s", exc)
            raise PreventUpdate
        return fig
