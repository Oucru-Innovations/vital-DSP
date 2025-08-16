"""
Window and slider management callbacks.
"""

import dash
from dash import Input, Output, State, callback_context, no_update


def register_window_callbacks(app):
    """Register window and slider-related callbacks with the Dash app."""

    @app.callback(
        Output("store_window", "data", allow_duplicate=True),
        Output("row_slider", "value", allow_duplicate=True),
        Output("row_slider", "min", allow_duplicate=True),
        Output("row_slider", "max", allow_duplicate=True),
        Input("btn_apply_window", "n_clicks"),
        Input("nudge_m10k", "n_clicks"),
        Input("nudge_m1k", "n_clicks"),
        Input("nudge_p1k", "n_clicks"),
        Input("nudge_p10k", "n_clicks"),
        State("store_window", "data"),
        State("start_row", "value"),
        State("end_row", "value"),
        State("store_total_rows", "data"),
        prevent_initial_call=True,
    )
    def window_controls(n_apply, nm10, nm1, np1, np10, window, start_in, end_in, total_rows):
        """Handle window controls and update slider accordingly."""
        ctx = callback_context
        trig = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
        total = int(total_rows or 0)

        if not window:
            window = {"start": 0, "end": 9999}

        start, end = int(window["start"]), int(window["end"])

        if trig == "btn_apply_window.n_clicks":
            start = int(start_in or 0)
            end = int(end_in or start)
        elif trig == "nudge_m10k.n_clicks":
            start -= 10_000
            end -= 10_000
        elif trig == "nudge_m1k.n_clicks":
            start -= 1_000
            end -= 1_000
        elif trig == "nudge_p1k.n_clicks":
            start += 1_000
            end += 1_000
        elif trig == "nudge_p10k.n_clicks":
            start += 10_000
            end += 10_000
        else:
            return no_update, no_update, no_update, no_update

        if total > 0:
            start = max(0, min(start, total - 1))
            end = max(start, min(end, total - 1))

        # Update slider bounds and value in the same callback
        slider_min = 0
        slider_max = max(1, total - 1) if total > 0 else 10000
        slider_value = [start, end]

        return {"start": start, "end": end}, slider_value, slider_min, slider_max

    @app.callback(
        Output("window_badge", "children"),
        Output("start_row", "value"),
        Output("end_row", "value"),
        Output("store_prev_total_rows", "data"),
        Input("store_window", "data"),
        State("store_total_rows", "data"),
        State("store_prev_total_rows", "data"),
        prevent_initial_call=False,
    )
    def reflect_window(window, total_rows, prev_total):
        """Reflect window changes in the UI."""
        total = int(total_rows or 0)
        if not window:
            return "Rows: 0–0", 0, 0, total

        start = int(window.get("start", 0))
        end = int(window.get("end", start))

        if total > 0:
            start = max(0, min(start, total - 1))
            end = max(start, min(end, total - 1))

        badge = f"Rows: {start:,}–{end:,}"
        return badge, start, end, total

    @app.callback(
        Output("store_window", "data", allow_duplicate=True),
        Input("row_slider", "value"),
        State("store_window", "data"),
        prevent_initial_call=True,
    )
    def handle_slider_change(slider_value, current_window):
        """Handle slider changes from user interaction."""
        if not slider_value or len(slider_value) != 2:
            return no_update

        start, end = int(slider_value[0]), int(slider_value[1])

        # Only update if the slider value actually changed the window
        if (
            current_window
            and current_window.get("start") == start
            and current_window.get("end") == end
        ):
            return no_update

        return {"start": start, "end": end}
