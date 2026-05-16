"""
Signal filtering callbacks for vitalDSP webapp.
Handles traditional filtering, advanced filtering, artifact removal, neural filtering, and ensemble filtering.
Uses actual vitalDSP functions for all computations.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, callback_context, no_update, html
from dash.exceptions import PreventUpdate
from vitalDSP.filtering.signal_filtering import SignalFiltering
import dash_bootstrap_components as dbc
import logging

# Import plot utilities for performance optimization
try:
    from vitalDSP_webapp.utils.plot_utils import limit_plot_data, check_plot_data_size
except ImportError:
    # Fallback if plot_utils not available
    def limit_plot_data(
        time_axis, signal_data, max_duration=300, max_points=10000, start_time=None
    ):
        """Fallback implementation of limit_plot_data"""
        return time_axis, signal_data

    def check_plot_data_size(time_axis, signal_data, max_points=10000):
        """Fallback implementation of check_plot_data_size"""
        return True


logger = logging.getLogger(__name__)


def safe_log_range(logger, data, name="data", precision=4):
    """
    Safely log min/max range of data, handling empty arrays.

    Args:
        logger: Logger instance
        data: Array-like data
        name: Name for logging (default: "data")
        precision: Decimal precision for formatting (default: 4)
    """
    if len(data) > 0:
        logger.info(
            f"{name} range: {np.min(data):.{precision}f} to {np.max(data):.{precision}f}"
        )
    else:
        logger.warning(f"{name} is empty - cannot compute range")


def configure_plot_with_pan_zoom(fig, title="", height=400):
    """
    Configure plotly figure with pan/zoom tools and consistent styling.

    Args:
        fig: Plotly figure object
        title: Plot title
        height: Plot height in pixels

    Returns:
        Configured plotly figure
    """
    fig.update_layout(
        title=title,
        height=height,
        showlegend=True,
        template="plotly_white",
        # Enable pan and zoom - use simpler configuration
        dragmode="pan",  # Default to pan mode
        # Show the modebar with all tools
        modebar=dict(
            orientation="v",  # Vertical orientation
            bgcolor="rgba(255,255,255,0.8)",  # Semi-transparent background
            color="rgba(0,0,0,0.5)",  # Dark icons
            activecolor="rgba(0,0,0,0.8)",  # Darker when active
        ),
    )
    return fig


def _render_chain_list(chain):
    """Render the staged filter chain as a numbered list for the UI."""
    if not chain:
        return html.Small(
            "Chain is empty. Pressing Apply runs the single filter "
            "configured below.  Click “Add as stage” to queue "
            "additional filters and have them applied in sequence.",
            className="text-muted",
        )
    items = []
    for i, stage in enumerate(chain, start=1):
        family = stage.get("family", "?")
        params = stage.get("params", {}) or {}
        # One-line summary tailored to the family.
        if family == "traditional":
            summary = (
                f"{params.get('filter_family', 'butter')} "
                f"{params.get('filter_response', 'bandpass')} "
                f"({params.get('low_freq')}-{params.get('high_freq')} Hz, "
                f"order {params.get('filter_order', 4)})"
            )
        elif family == "advanced":
            summary = params.get("advanced_method", "?")
        elif family == "smoothing":
            method = params.get("smoothing_method", "savgol")
            if method == "savgol":
                summary = (
                    f"savgol (win {params.get('savgol_window', 11)}, "
                    f"poly {params.get('savgol_polyorder', 2)})"
                )
            elif method == "moving_avg":
                summary = f"moving avg (win {params.get('moving_avg_window', 5)})"
            elif method == "gaussian":
                summary = f"gaussian (sigma {params.get('gaussian_sigma', 1.0)})"
            else:
                summary = method
        elif family == "artifact":
            summary = (
                f"{params.get('artifact_type', '?')} "
                f"(strength {params.get('artifact_strength', '?')})"
            )
        elif family == "neural":
            summary = (
                f"{params.get('neural_type', '?')} "
                f"({params.get('neural_complexity', '?')})"
            )
        elif family == "ensemble":
            summary = (
                f"{params.get('ensemble_method', '?')} "
                f"x{params.get('ensemble_n_filters', '?')}"
            )
        else:
            summary = "(no params)"
        iters = stage.get("iterations", 1)
        iter_txt = f" x{iters}" if iters and int(iters) > 1 else ""
        items.append(
            html.Li(
                [
                    html.Strong(f"Stage {i}: ", className="me-1"),
                    html.Span(f"{family} - {summary}{iter_txt}"),
                ],
                className="small mb-1",
            )
        )
    return html.Ol(items, className="mb-1 ps-3")


def _apply_chain_stage(signal, sampling_freq, signal_type, stage, logger=None):
    """Apply ONE stage of a filter chain to ``signal``.

    Routes by ``stage['family']`` to the existing webapp filter
    wrappers (which in turn call the relevant vitalDSP helpers).
    Returns the filtered signal; on error, returns the input
    unchanged and logs the failure.
    """
    import numpy as np

    family = stage.get("family", "traditional")
    params = stage.get("params", {}) or {}
    iters = max(1, min(int(stage.get("iterations", 1) or 1), 10))

    out = np.asarray(signal, dtype=float)
    for _ in range(iters):
        try:
            if family == "traditional":
                out = apply_traditional_filter(
                    out,
                    sampling_freq,
                    params.get("filter_family", "butter"),
                    params.get("filter_response", "bandpass"),
                    params.get("low_freq"),
                    params.get("high_freq"),
                    int(params.get("filter_order", 4)),
                )
            elif family == "advanced":
                out = apply_advanced_filter(
                    out,
                    params.get("advanced_method", "kalman"),
                )
            elif family == "smoothing":
                out = apply_smoothing_filter(
                    out,
                    params.get("smoothing_method", "savgol"),
                    params.get("savgol_window"),
                    params.get("savgol_polyorder"),
                    params.get("moving_avg_window"),
                    params.get("gaussian_sigma"),
                )
            elif family == "artifact":
                out = apply_enhanced_artifact_removal(
                    out,
                    sampling_freq,
                    params.get("artifact_type", "baseline"),
                    params.get("artifact_strength", 0.5),
                    None, None, None, None, None, None, None, None, None,
                )
            elif family == "neural":
                out = apply_neural_filter(
                    out,
                    params.get("neural_type", "lstm"),
                    params.get("neural_complexity", "medium"),
                )
            elif family == "ensemble":
                out = apply_enhanced_ensemble_filter(
                    out,
                    params.get("ensemble_method", "voting"),
                    int(params.get("ensemble_n_filters", 3)),
                    None, None, None, None, None, None,
                )
        except Exception as exc:
            if logger:
                logger.warning(
                    "Chain stage %r failed: %s; passing input through.",
                    family, exc,
                )
            # Keep ``out`` unchanged so subsequent stages still run.
    return out


def apply_filter_chain(signal, sampling_freq, signal_type, chain, logger=None):
    """Apply a full filter chain (list of stages) to ``signal``.

    Used by the filtering page on Apply AND by downstream pages
    (time-domain, frequency, features) when they re-apply the user's
    saved preprocessing chain to a new window.  ``chain=None`` or an
    empty list returns the signal unchanged.
    """
    import numpy as np

    if not chain:
        return np.asarray(signal, dtype=float)
    out = np.asarray(signal, dtype=float)
    for stage in chain:
        out = _apply_chain_stage(out, sampling_freq, signal_type, stage, logger=logger)
    return out


def _panel_filter_as_stage(
    family,
    filter_family,
    filter_response,
    low_freq,
    high_freq,
    filter_order,
    advanced_method,
    artifact_type,
    artifact_strength,
    neural_type,
    neural_complexity,
    ensemble_method,
    ensemble_n_filters,
    smoothing_method=None,
    savgol_window=None,
    savgol_polyorder=None,
    moving_avg_window=None,
    gaussian_sigma=None,
):
    """Snapshot the panel filter inputs into a chain stage dict.

    Used by the Apply callback to make the panel filter the implicit
    final stage of the chain.  Returns ``None`` when the panel has no
    useful config (e.g. unknown family) so the caller can skip it.
    """
    if not family:
        return None
    stage = {"family": family, "iterations": 1}
    if family == "traditional":
        stage["params"] = {
            "filter_family": filter_family or "butter",
            "filter_response": filter_response or "bandpass",
            "low_freq": low_freq,
            "high_freq": high_freq,
            "filter_order": filter_order or 4,
        }
    elif family == "advanced":
        stage["params"] = {"advanced_method": advanced_method or "kalman"}
    elif family == "smoothing":
        stage["params"] = {
            "smoothing_method": smoothing_method or "savgol",
            "savgol_window": savgol_window,
            "savgol_polyorder": savgol_polyorder,
            "moving_avg_window": moving_avg_window,
            "gaussian_sigma": gaussian_sigma,
        }
    elif family == "artifact":
        stage["params"] = {
            "artifact_type": artifact_type or "baseline",
            "artifact_strength": artifact_strength or 0.5,
        }
    elif family == "neural":
        stage["params"] = {
            "neural_type": neural_type or "lstm",
            "neural_complexity": neural_complexity or "medium",
        }
    elif family == "ensemble":
        stage["params"] = {
            "ensemble_method": ensemble_method or "voting",
            "ensemble_n_filters": ensemble_n_filters or 3,
        }
    else:
        return None
    return stage


def apply_smoothing_filter(signal_data, method, savgol_window, savgol_polyorder, moving_avg_window, gaussian_sigma):
    """Apply a single smoothing method to ``signal_data``.

    ``method`` is one of ``"savgol"``, ``"moving_avg"``, ``"gaussian"``.
    Returns the smoothed signal; on failure returns the input unchanged.
    """
    out = np.asarray(signal_data, dtype=float)
    try:
        sf = SignalFiltering(out)
        if method == "savgol":
            window = int(savgol_window or 11)
            if window % 2 == 0:
                window += 1
            window = max(3, window)
            polyorder = max(1, int(savgol_polyorder or 2))
            polyorder = min(polyorder, window - 2)
            return sf.savgol_filter(out, window_length=window, polyorder=polyorder)
        if method == "moving_avg":
            window = max(2, int(moving_avg_window or 5))
            return sf.moving_average(window_size=window, iterations=1, method="edge")
        if method == "gaussian":
            sigma = float(gaussian_sigma or 1.0)
            return sf.gaussian(sigma=sigma, iterations=1)
    except Exception as exc:
        logger.warning("Smoothing %r failed: %s", method, exc)
    return out


def register_signal_filtering_callbacks(app):
    """Register all signal filtering callbacks."""

    # Auto-select signal type and set defaults based on uploaded data
    @app.callback(
        [
            Output("filter-signal-type-select", "value"),
            Output("filter-type-select", "value"),
            Output("advanced-filter-method", "value"),
        ],
        [Input("url", "pathname")],
        [
            State("filter-signal-type-select", "value"),
            State("filter-type-select", "value"),
            State("advanced-filter-method", "value"),
        ],
        prevent_initial_call=True,
    )
    def auto_select_signal_type_and_defaults(
        pathname, current_signal_type, current_filter_type, current_advanced_method
    ):
        """Auto-select signal type and set appropriate defaults based on uploaded data."""
        logger.info("=== AUTO-SELECT SIGNAL TYPE CALLBACK TRIGGERED ===")
        logger.info(f"Pathname: {pathname}")
        logger.info(
            f"Current user selections - Signal: {current_signal_type}, Filter: {current_filter_type}, Method: {current_advanced_method}"
        )

        if pathname != "/filtering":
            logger.info("Not on filtering page, preventing update")
            raise PreventUpdate

        # If user has already made selections, don't override them
        # Check if this is NOT the first time on the page (current values are not None)
        if (
            current_filter_type is not None
            or current_advanced_method is not None
            or current_signal_type is not None
        ):
            logger.info(
                f"User has existing selections - Signal: {current_signal_type}, Filter: {current_filter_type}, Method: {current_advanced_method}"
            )
            logger.info("Preserving user selections, not auto-selecting")
            raise PreventUpdate

        try:
            from vitalDSP_webapp.services.data.enhanced_data_service import (
                get_enhanced_data_service,
            )

            data_service = get_enhanced_data_service()
            if not data_service:
                logger.warning("Data service not available")
                return "PPG", "traditional", "convolution"

            # Get the latest data
            all_data = data_service.get_all_data()
            if not all_data:
                logger.info("No data available, using defaults")
                return "PPG", "traditional", "convolution"

            # Get the most recent data
            latest_data_id = max(
                all_data.keys(), key=lambda x: int(x.split("_")[1]) if "_" in x else 0
            )
            data_info = data_service.get_data_info(latest_data_id)

            if not data_info:
                logger.info("No data info available, using defaults")
                return "PPG", "traditional", "convolution"

            # Debug: Log the data_info to see what's stored
            logger.info(
                f"Data info keys: {list(data_info.keys()) if data_info else 'None'}"
            )
            logger.info(f"Full data info: {data_info}")

            # First, check if signal type is stored in data info
            stored_signal_type = data_info.get("signal_type", None)
            logger.info(f"Stored signal type: {stored_signal_type}")

            if stored_signal_type and stored_signal_type.lower() != "auto":
                # Convert stored value to match filtering screen dropdown format
                if stored_signal_type.lower() == "ecg":
                    signal_type = "ECG"
                elif stored_signal_type.lower() == "ppg":
                    signal_type = "PPG"
                elif stored_signal_type.lower() == "other":
                    signal_type = "Other"
                else:
                    signal_type = stored_signal_type.upper()
                logger.info(f"Using stored signal type: {signal_type}")
            else:
                # Auto-detect signal type based on data characteristics
                signal_type = "PPG"  # Default
                logger.info("Auto-detecting signal type from data characteristics")

            filter_type = "traditional"  # Default
            advanced_method = "convolution"  # Default

            # Try to detect signal type from column names or data characteristics if not stored
            if (
                stored_signal_type
                and stored_signal_type.lower() == "auto"
                or not stored_signal_type
            ):
                df = data_service.get_data(latest_data_id)
                if df is not None and not df.empty:
                    column_mapping = data_service.get_column_mapping(latest_data_id)
                    signal_column = column_mapping.get("signal", "")

                    # Check column names for signal type hints
                    if any(
                        keyword in signal_column.lower()
                        for keyword in ["ecg", "electrocardio"]
                    ):
                        signal_type = "ECG"
                        filter_type = "advanced"  # Set ECG default to advanced filters
                        advanced_method = (
                            "convolution"  # Set convolution as default method
                        )
                        logger.info("Auto-detected ECG signal type from column name")
                    elif any(
                        keyword in signal_column.lower()
                        for keyword in ["ppg", "pleth", "photopleth"]
                    ):
                        signal_type = "PPG"
                        logger.info("Auto-detected PPG signal type from column name")
                    else:
                        # Try to detect from data characteristics
                        try:
                            signal_data = (
                                df[signal_column].values
                                if signal_column
                                else df.iloc[:, 1].values
                            )
                            sampling_freq = data_info.get("sampling_freq", 1000)

                            # Simple heuristic: ECG typically has higher frequency content
                            from vitalDSP.transforms.fourier_transform import (
                                FourierTransform,
                            )

                            ft = FourierTransform(
                                signal_data
                            )  # FourierTransform takes only signal, not fs
                            f, psd = ft.compute_psd()
                            dominant_freq = f[np.argmax(psd)]

                            if (
                                dominant_freq > 1.0
                            ):  # Higher frequency content suggests ECG
                                signal_type = "ECG"
                                filter_type = "advanced"
                                advanced_method = "convolution"
                                logger.info(
                                    "Auto-detected ECG signal type from frequency analysis"
                                )
                            else:
                                signal_type = "PPG"
                                logger.info(
                                    "Auto-detected PPG signal type from frequency analysis"
                                )
                        except Exception as e:
                            logger.warning(
                                f"Could not analyze signal characteristics: {e}"
                            )
                            signal_type = "PPG"

            logger.info(
                f"Auto-selected signal type: {signal_type}, filter type: {filter_type}, method: {advanced_method}"
            )
            return signal_type, filter_type, advanced_method

        except Exception as e:
            logger.error(f"Error in auto-selection: {e}")
            return "PPG", "traditional", "convolution"

    # Filter Type Selection Callback
    @app.callback(
        [
            Output("traditional-filter-params", "style", allow_duplicate=True),
            Output("advanced-filter-params", "style", allow_duplicate=True),
            Output("smoothing-filter-params", "style", allow_duplicate=True),
            Output("artifact-removal-params", "style", allow_duplicate=True),
            Output("neural-network-params", "style", allow_duplicate=True),
            Output("ensemble-params", "style", allow_duplicate=True),
        ],
        [Input("filter-type-select", "value")],
        prevent_initial_call=True,
    )
    def update_filter_parameter_visibility(filter_type):
        """Show/hide parameter sections based on selected filter type."""
        hidden_style = {"display": "none"}
        visible_style = {"display": "block"}

        styles = {
            "traditional": hidden_style,
            "advanced": hidden_style,
            "smoothing": hidden_style,
            "artifact": hidden_style,
            "neural": hidden_style,
            "ensemble": hidden_style,
        }
        if filter_type in styles:
            styles[filter_type] = visible_style

        return (
            styles["traditional"],
            styles["advanced"],
            styles["smoothing"],
            styles["artifact"],
            styles["neural"],
            styles["ensemble"],
        )

    # Smoothing method sub-toggle
    @app.callback(
        [
            Output("smoothing-savgol-params", "style"),
            Output("smoothing-movavg-params", "style"),
            Output("smoothing-gaussian-params", "style"),
        ],
        [Input("smoothing-method-select", "value")],
        prevent_initial_call=True,
    )
    def update_smoothing_method_visibility(method):
        hidden = {"display": "none"}
        visible = {"display": "block"}
        return (
            visible if method == "savgol" else hidden,
            visible if method == "moving_avg" else hidden,
            visible if method == "gaussian" else hidden,
        )

    # Advanced Filter Method Selection Callback
    @app.callback(
        [
            Output("kalman-params", "style"),
            Output("optimization-params", "style"),
            Output("gradient-params", "style"),
            Output("convolution-params", "style"),
            Output("attention-params", "style"),
            Output("adaptive-params", "style"),
        ],
        [Input("advanced-filter-method", "value")],
        prevent_initial_call=True,
    )
    def update_advanced_filter_method_visibility(method):
        """Show/hide advanced filter method parameters based on selected method."""
        hidden_style = {"display": "none"}
        visible_style = {"display": "block"}

        # Initialize all as hidden
        kalman_style = hidden_style
        optimization_style = hidden_style
        gradient_style = hidden_style
        convolution_style = hidden_style
        attention_style = hidden_style
        adaptive_style = hidden_style

        # Show the appropriate section based on method
        if method == "kalman":
            kalman_style = visible_style
        elif method == "optimization":
            optimization_style = visible_style
        elif method == "gradient_descent":
            gradient_style = visible_style
        elif method == "convolution":
            convolution_style = visible_style
        elif method == "attention":
            attention_style = visible_style
        elif method == "adaptive":
            adaptive_style = visible_style

        return (
            kalman_style,
            optimization_style,
            gradient_style,
            convolution_style,
            attention_style,
            adaptive_style,
        )

    # Attention Type Selection Callback
    @app.callback(
        [
            Output("attention-gaussian-params", "style"),
            Output("attention-linear-params", "style"),
            Output("attention-exponential-params", "style"),
        ],
        [Input("attention-type", "value")],
        prevent_initial_call=True,
    )
    def update_attention_type_visibility(attention_type):
        """Show/hide attention-specific parameters based on selected attention type."""
        hidden_style = {"display": "none"}
        visible_style = {"display": "block"}

        # Initialize all as hidden
        gaussian_style = hidden_style
        linear_style = hidden_style
        exponential_style = hidden_style

        # Show the appropriate section based on attention type
        if attention_type == "gaussian":
            gaussian_style = visible_style
        elif attention_type == "linear":
            linear_style = visible_style
        elif attention_type == "exponential":
            linear_style = visible_style  # Linear params include direction
            exponential_style = visible_style  # Also show base parameter

        return (
            gaussian_style,
            linear_style,
            exponential_style,
        )

    # ------------------------------------------------------------------
    # Filter chain callbacks
    # ------------------------------------------------------------------
    # The chain lets the user stage several filters in sequence on the
    # same segment - e.g. detrend -> Butterworth bandpass -> median
    # smoothing.  Each "Add as stage" press snapshots the current
    # filter-config inputs and appends the entry to the chain Store.
    # Apply Filter runs the whole chain in order; when the chain is
    # empty it falls back to the single configured filter.

    @app.callback(
        [
            Output("filter-chain-store", "data", allow_duplicate=True),
            Output("filter-chain-list", "children", allow_duplicate=True),
        ],
        Input("filter-chain-add", "n_clicks"),
        [
            State("filter-chain-store", "data"),
            State("filter-type-select", "value"),
            State("filter-family-advanced", "value"),
            State("filter-response-advanced", "value"),
            State("filter-low-freq-advanced", "value"),
            State("filter-high-freq-advanced", "value"),
            State("filter-order-advanced", "value"),
            State("advanced-filter-method", "value"),
            State("artifact-type", "value"),
            State("artifact-removal-strength", "value"),
            State("neural-network-type", "value"),
            State("neural-model-complexity", "value"),
            State("ensemble-method", "value"),
            State("ensemble-n-filters", "value"),
            State("filter-application-count", "value"),
            State("smoothing-method-select", "value"),
            State("savgol-window", "value"),
            State("savgol-polyorder", "value"),
            State("moving-avg-window", "value"),
            State("gaussian-sigma", "value"),
        ],
        prevent_initial_call=True,
    )
    def add_filter_to_chain(
        n_clicks,
        chain,
        family,
        filter_family,
        filter_response,
        low_freq,
        high_freq,
        filter_order,
        advanced_method,
        artifact_type,
        artifact_strength,
        neural_type,
        neural_complexity,
        ensemble_method,
        ensemble_n_filters,
        iterations,
        smoothing_method,
        savgol_window,
        savgol_polyorder,
        moving_avg_window,
        gaussian_sigma,
    ):
        """Snapshot the current filter config and append it to the chain."""
        if not n_clicks:
            raise PreventUpdate

        stage = {
            "family": family or "traditional",
            "iterations": int(iterations or 1),
        }
        if family == "traditional":
            stage["params"] = {
                "filter_family": filter_family or "butter",
                "filter_response": filter_response or "bandpass",
                "low_freq": low_freq,
                "high_freq": high_freq,
                "filter_order": filter_order or 4,
            }
        elif family == "advanced":
            stage["params"] = {"advanced_method": advanced_method or "kalman"}
        elif family == "smoothing":
            stage["params"] = {
                "smoothing_method": smoothing_method or "savgol",
                "savgol_window": savgol_window,
                "savgol_polyorder": savgol_polyorder,
                "moving_avg_window": moving_avg_window,
                "gaussian_sigma": gaussian_sigma,
            }
        elif family == "artifact":
            stage["params"] = {
                "artifact_type": artifact_type or "baseline",
                "artifact_strength": artifact_strength or 0.5,
            }
        elif family == "neural":
            stage["params"] = {
                "neural_type": neural_type or "lstm",
                "neural_complexity": neural_complexity or "medium",
            }
        elif family == "ensemble":
            stage["params"] = {
                "ensemble_method": ensemble_method or "voting",
                "ensemble_n_filters": ensemble_n_filters or 3,
            }
        else:
            stage["params"] = {}

        new_chain = list(chain or []) + [stage]
        return new_chain, _render_chain_list(new_chain)

    @app.callback(
        [
            Output("filter-chain-store", "data", allow_duplicate=True),
            Output("filter-chain-list", "children", allow_duplicate=True),
        ],
        Input("filter-chain-clear", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_filter_chain(n_clicks):
        """Wipe the chain back to empty."""
        if not n_clicks:
            raise PreventUpdate
        return [], _render_chain_list([])

    @app.callback(
        Output("filter-chain-panel-preview", "children"),
        [
            Input("filter-type-select", "value"),
            Input("filter-family-advanced", "value"),
            Input("filter-response-advanced", "value"),
            Input("filter-low-freq-advanced", "value"),
            Input("filter-high-freq-advanced", "value"),
            Input("filter-order-advanced", "value"),
            Input("advanced-filter-method", "value"),
            Input("artifact-type", "value"),
            Input("smoothing-method-select", "value"),
            Input("filter-chain-store", "data"),
        ],
        prevent_initial_call=False,
    )
    def update_chain_panel_preview(
        family,
        filter_family,
        filter_response,
        low_freq,
        high_freq,
        filter_order,
        advanced_method,
        artifact_type,
        smoothing_method,
        chain,
    ):
        """Render a 'next stage (preview)' hint matching the current panel.

        Helps the user see what Apply will actually run on top of the
        queued chain.  Tied to a small subset of panel inputs (the ones
        most commonly tweaked); editing any other panel control still
        works at Apply time, just isn't reflected in this hint.
        """
        if family == "traditional":
            summary = (
                f"{filter_family or 'butter'} "
                f"{filter_response or 'bandpass'} "
                f"({low_freq}-{high_freq} Hz, order {filter_order or 4})"
            )
        elif family == "advanced":
            summary = f"advanced - {advanced_method or '?'}"
        elif family == "smoothing":
            summary = f"smoothing - {smoothing_method or 'savgol'}"
        elif family == "artifact":
            summary = f"artifact removal - {artifact_type or 'baseline'}"
        elif family == "neural":
            summary = "neural"
        elif family == "ensemble":
            summary = "ensemble"
        else:
            summary = "(no panel filter)"
        prefix = (
            "Apply will run the chain, then this panel filter as a "
            "final stage:"
            if chain
            else "Apply will run this panel filter (chain is empty):"
        )
        return [
            html.Span(prefix, className="me-1"),
            html.Strong(summary),
        ]

    # Advanced Filtering Callback
    @app.callback(
        [
            Output("filter-original-plot", "figure"),
            Output("filter-filtered-plot", "figure"),
            Output("filter-comparison-plot", "figure"),
            Output("filter-quality-metrics", "children"),
            Output("filter-quality-plots", "figure"),
            Output("store-filtering-data", "data"),
            Output("store-filter-comparison", "data"),
            Output("store-filter-quality-metrics", "data"),
            Output("store-filtered-signal", "data"),
            Output("filter-saved-banner", "children"),
        ],
        [
            Input("filter-btn-apply", "n_clicks"),
            Input("btn-nudge-m10", "n_clicks"),
            Input("btn-center", "n_clicks"),
            Input("btn-nudge-p10", "n_clicks"),
        ],
        [
            # Was Input - moved to State so a route change to /filtering
            # no longer fires this expensive callback with no user intent.
            State("url", "pathname"),
            State("start-position-slider", "value"),
            State("duration-select", "value"),
            # The top bar's "Duration" dropdown is now a view-zoom in
            # *segments* (1/3/5).  Multiply by the segment-length below
            # so downstream math (start_time, end_time) keeps the same
            # seconds-based interpretation it always had.
            State("filter-segment-length", "value"),
            State("filter-type-select", "value"),
            State("filter-signal-source", "value"),  # NEW: Signal source selector
            State("filter-application-count", "value"),  # NEW: Filter application count
            State(
                "store-filtered-signal", "data"
            ),  # NEW: Access to current filtered signal
            State("filter-family-advanced", "value"),
            State("filter-response-advanced", "value"),
            State("filter-low-freq-advanced", "value"),
            State("filter-high-freq-advanced", "value"),
            State("filter-order-advanced", "value"),
            State("advanced-filter-method", "value"),
            # Kalman filter parameters
            State("kalman-r", "value"),
            State("kalman-q", "value"),
            # Optimization parameters
            State("optimization-loss-type", "value"),
            State("optimization-initial-guess", "value"),
            State("optimization-learning-rate", "value"),
            State("optimization-iterations", "value"),
            # Gradient descent parameters
            State("gradient-learning-rate", "value"),
            State("gradient-iterations", "value"),
            # Convolution parameters
            State("convolution-kernel-type", "value"),
            State("convolution-kernel-size", "value"),
            # Attention parameters
            State("attention-type", "value"),
            State("attention-size", "value"),
            State("attention-sigma", "value"),
            State("attention-ascending", "value"),
            State("attention-base", "value"),
            # Adaptive filter parameters
            State("adaptive-mu", "value"),
            State("adaptive-order", "value"),
            State("artifact-type", "value"),
            State("artifact-removal-strength", "value"),
            State("neural-network-type", "value"),
            State("neural-model-complexity", "value"),
            State("ensemble-method", "value"),
            State("ensemble-n-filters", "value"),
            State("filter-quality-options", "value"),
            State("detrend-option", "value"),
            State("filter-signal-type-select", "value"),
            # Smoothing family parameters (also reused as legacy Additional Filters
            # State for back-compat)
            State("smoothing-method-select", "value"),
            State("savgol-window", "value"),
            State("savgol-polyorder", "value"),
            State("moving-avg-window", "value"),
            State("gaussian-sigma", "value"),
            # Artifact removal parameters
            State("wavelet-type", "value"),
            State("wavelet-level", "value"),
            State("threshold-type", "value"),
            State("threshold-value", "value"),
            # Multi-modal parameters
            State("reference-signal", "value"),
            State("fusion-method", "value"),
            # Filter chain (list of staged filters to apply in order
            # before / instead of the single configured filter).
            State("filter-chain-store", "data"),
        ],
        prevent_initial_call=True,
    )
    def advanced_filtering_callback(
        n_clicks,
        nudge_m10,
        center_click,
        nudge_p10,
        pathname,
        start_position,
        duration,
        segment_length_seconds,
        filter_type,
        signal_source,  # NEW: Signal source selector
        filter_count,  # NEW: Filter application count
        current_filtered_signal,  # NEW: Current filtered signal data
        filter_family,
        filter_response,
        low_freq,
        high_freq,
        filter_order,
        advanced_method,
        # Kalman filter parameters
        kalman_r,
        kalman_q,
        # Optimization parameters
        optimization_loss_type,
        optimization_initial_guess,
        optimization_learning_rate,
        optimization_iterations,
        # Gradient descent parameters
        gradient_learning_rate,
        gradient_iterations,
        # Convolution parameters
        convolution_kernel_type,
        convolution_kernel_size,
        # Attention parameters
        attention_type,
        attention_size,
        attention_sigma,
        attention_ascending,
        attention_base,
        # Adaptive filter parameters
        adaptive_mu,
        adaptive_order,
        artifact_type,
        artifact_strength,
        neural_type,
        neural_complexity,
        ensemble_method,
        ensemble_n_filters,
        quality_options,
        detrend_option,
        signal_type,
        # Smoothing family parameters (also serve as legacy Additional Filters State)
        smoothing_method,
        savgol_window,
        savgol_polyorder,
        moving_avg_window,
        gaussian_sigma,
        # Artifact removal parameters
        wavelet_type,
        wavelet_level,
        threshold_type,
        threshold_value,
        # Multi-modal parameters
        reference_signal,
        fusion_method,
        # Optional chain (list of staged filters); when non-empty it
        # overrides the single-filter execution below.  Defaults to
        # ``None`` so legacy tests that invoke the callback positionally
        # without the new arg still pass.
        filter_chain=None,
    ):

        ctx = callback_context
        logger.info("=== ADVANCED FILTERING CALLBACK TRIGGERED ===")

        # ``prevent_initial_call=True`` and the four button Inputs mean this
        # callback only fires on a real user action.  Path is now a State
        # only so route changes don't trigger it.  We still bail if the
        # user landed here without uploading + processing data first.
        if pathname and pathname != "/filtering":
            logger.info("Pathname %s != /filtering; skipping update", pathname)
            raise PreventUpdate

        if ctx.triggered:
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
            logger.info(f"Trigger ID: {trigger_id}")

        try:
            # Get data from the data service
            from vitalDSP_webapp.services.data.enhanced_data_service import (
                get_enhanced_data_service,
            )

            data_service = get_enhanced_data_service()

            # Get all stored data and find the latest
            all_data = data_service.get_all_data()
            logger.info(f"Data service returned: {type(all_data)}")
            logger.info(
                f"All data keys: {list(all_data.keys()) if all_data else 'None'}"
            )
            # logger.info(f"All data content: {all_data}")  # Too verbose - disabled

            if not all_data:
                logger.warning("No data available for filtering")
                return (
                    create_empty_figure(),
                    create_empty_figure(),
                    create_empty_figure(),
                    dbc.Alert(
                        [
                            html.Strong("No processed data found. "),
                            "Go to the Upload page, drop a recording, pick a "
                            "signal column, and press Process data first.",
                        ],
                        color="warning",
                        className="mb-0",
                    ),
                    create_empty_figure(),
                    None,
                    None,
                    None,
                    None,
                    no_update,  # banner left unchanged
                )

            # Get the most recent data entry
            latest_data_id = list(all_data.keys())[-1]
            latest_data = all_data[latest_data_id]
            logger.info(f"Latest data ID: {latest_data_id}")
            # logger.info(f"Latest data content: {latest_data}")  # Too verbose - disabled

            # Log the callback parameters
            logger.info("=== CALLBACK PARAMETERS ===")
            logger.info(
                f"Start position: {start_position}% (type: {type(start_position)})"
            )
            logger.info(f"Duration: {duration}s (type: {type(duration)})")
            logger.info(f"Filter type: {filter_type} (type: {type(filter_type)})")
            logger.info(f"Filter family: {filter_family} (type: {type(filter_family)})")
            logger.info(
                f"Filter response: {filter_response} (type: {type(filter_response)})"
            )
            logger.info(f"Low frequency: {low_freq} (type: {type(low_freq)})")
            logger.info(f"High frequency: {high_freq} (type: {type(high_freq)})")
            logger.info(f"Filter order: {filter_order} (type: {type(filter_order)})")
            logger.info(
                f"Advanced method: {advanced_method} (type: {type(advanced_method)})"
            )
            # Log method-specific parameters based on selected method
            if advanced_method == "kalman":
                logger.info(f"Kalman R: {kalman_r}, Q: {kalman_q}")
            elif advanced_method == "optimization":
                logger.info(
                    f"Optimization: loss={optimization_loss_type}, lr={optimization_learning_rate}, iterations={optimization_iterations}"
                )
            elif advanced_method == "gradient_descent":
                logger.info(
                    f"Gradient descent: lr={gradient_learning_rate}, iterations={gradient_iterations}"
                )
            elif advanced_method == "convolution":
                logger.info(
                    f"Convolution: kernel_type={convolution_kernel_type}, kernel_size={convolution_kernel_size}"
                )
            elif advanced_method == "attention":
                logger.info(
                    f"Attention: type={attention_type}, size={attention_size}, sigma={attention_sigma}"
                )
            elif advanced_method == "adaptive":
                logger.info(f"Adaptive: mu={adaptive_mu}, order={adaptive_order}")
            logger.info(f"Artifact type: {artifact_type} (type: {type(artifact_type)})")
            logger.info(
                f"Artifact strength: {artifact_strength} (type: {type(artifact_strength)})"
            )
            logger.info(f"Neural type: {neural_type} (type: {type(neural_type)})")
            logger.info(
                f"Neural complexity: {neural_complexity} (type: {type(neural_complexity)})"
            )
            logger.info(
                f"Ensemble method: {ensemble_method} (type: {type(ensemble_method)})"
            )
            logger.info(
                f"Ensemble n filters: {ensemble_n_filters} (type: {type(ensemble_n_filters)})"
            )
            logger.info(
                f"Quality options: {quality_options} (type: {type(quality_options)})"
            )
            logger.info(
                f"Detrend option: {detrend_option} (type: {type(detrend_option)})"
            )

            # Get the data and info using the data ID
            df = data_service.get_data(latest_data_id)
            data_info = data_service.get_data_info(latest_data_id)
            column_mapping = data_service.get_column_mapping(latest_data_id)

            # Calculate time range from start position and duration
            # IMPORTANT: Convert types - duration comes as STRING from dropdown!
            if start_position is None:
                start_position = 0
            else:
                start_position = float(start_position)  # Ensure numeric

            # ``duration`` now arrives as a view-zoom in *segments*
            # (1/3/5).  Translate to seconds by multiplying by the
            # configured segment length.  Defaults: 1 segment * 30 s
            # = 30 s window if neither value is set.
            view_zoom = duration  # capture pre-translation value for logs
            try:
                view_zoom = int(view_zoom) if view_zoom is not None else 1
            except (TypeError, ValueError):
                view_zoom = 1
            try:
                seg_len_s = float(segment_length_seconds) if segment_length_seconds else 30.0
            except (TypeError, ValueError):
                seg_len_s = 30.0
            duration = float(view_zoom) * seg_len_s
            logger.info(
                "view_zoom=%d x segment_length=%gs → duration=%gs",
                view_zoom, seg_len_s, duration,
            )

            if False:  # legacy branches kept for shape; never executed
                # Duration comes as STRING from dropdown - must convert!
                try:
                    duration = float(duration)
                    logger.info(f"Duration converted to float: {duration}")
                except (ValueError, TypeError) as e:
                    logger.error(
                        f"Failed to convert duration '{duration}' to float: {e}"
                    )
                    duration = 60  # Fallback to default

            # Get sampling frequency - CRITICAL for proper time calculations.
            # ``.get(key, default)`` returns ``None`` when the key exists with
            # value None (which the upload pipeline could produce when the
            # user left the field blank and the loader couldn't infer it),
            # so use ``or`` to coerce both missing-key and None-value to the
            # default.
            sampling_freq = (
                data_info.get("sampling_freq")
                or data_info.get("sampling_rate")
                or 1000
            )
            logger.info(f"Sampling frequency from data_info: {sampling_freq} Hz")

            # Get data duration to calculate actual time range
            data_duration = data_info.get("duration", 0)
            if data_duration == 0:
                # Calculate duration from sampling frequency and data length
                data_duration = len(df) / sampling_freq
                logger.info(
                    f"Calculated data duration: {data_duration:.2f} seconds ({len(df)} samples / {sampling_freq} Hz)"
                )

            # Calculate start time based on percentage (start_position is 0-100)
            start_time = (start_position / 100.0) * data_duration

            # Calculate end time based on duration
            end_time = start_time + duration

            logger.info(f"Time window calculation:")
            logger.info(
                f"  start_position: {start_position}% → start_time: {start_time:.2f}s"
            )
            logger.info(f"  duration: {duration}s → end_time: {end_time:.2f}s")
            logger.info(f"  data_duration: {data_duration:.2f}s")

            # Ensure end time doesn't exceed data duration
            if end_time > data_duration:
                end_time = data_duration
                start_time = max(0, end_time - duration)

            # Handle nudge button adjustments
            ctx = callback_context
            if ctx.triggered:
                trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

                # Only adjust position for actual nudge buttons, not for Apply Filter
                if trigger_id == "btn-nudge-m10":
                    start_position = max(0, start_position - 10)  # Adjust percentage
                    # Recalculate time range with adjusted position
                    start_time = (start_position / 100.0) * data_duration
                    end_time = start_time + duration
                    # Ensure end time doesn't exceed data duration
                    if end_time > data_duration:
                        end_time = data_duration
                        start_position = max(
                            0, (end_time - duration) / data_duration * 100
                        )
                    logger.info(f"Nudge button triggered: {trigger_id}")
                    logger.info(f"Adjusted start position: {start_position}%")
                    logger.info(
                        f"Adjusted time range: {start_time:.2f} to {end_time:.2f} seconds"
                    )
                elif trigger_id == "btn-center":
                    start_position = 50  # Center at 50%
                    # Recalculate time range with adjusted position
                    start_time = (start_position / 100.0) * data_duration
                    end_time = start_time + duration
                    # Ensure end time doesn't exceed data duration
                    if end_time > data_duration:
                        end_time = data_duration
                        start_position = max(0, end_time - duration)
                    logger.info(f"Nudge button triggered: {trigger_id}")
                    logger.info(f"Adjusted start position: {start_position}%")
                    logger.info(
                        f"Adjusted time range: {start_time:.2f} to {end_time:.2f} seconds"
                    )
                elif trigger_id == "btn-nudge-p10":
                    start_position = min(100, start_position + 10)
                    # Recalculate time range with adjusted position
                    start_time = (start_position / 100.0) * data_duration
                    end_time = start_time + duration
                    # Ensure end time doesn't exceed data duration
                    if end_time > data_duration:
                        end_time = data_duration
                        start_position = max(0, end_time - duration)
                    logger.info(f"Nudge button triggered: {trigger_id}")
                    logger.info(f"Adjusted start position: {start_position}%")
                    logger.info(
                        f"Adjusted time range: {start_time:.2f} to {end_time:.2f} seconds"
                    )
                elif trigger_id == "filter-btn-apply":
                    # Apply Filter button - use the current start_position and duration as-is
                    logger.info(
                        f"Apply Filter button triggered - using current parameters"
                    )
                    logger.info(f"Start position: {start_position}%")
                    logger.info(
                        f"Time range: {start_time:.2f} to {end_time:.2f} seconds"
                    )

            logger.info("Data service method results:")
            logger.info(f"  get_data returned: {type(df)}")
            logger.info(f"  get_data_info returned: {type(data_info)}")
            logger.info(f"  get_column_mapping returned: {type(column_mapping)}")

            # Reduced logging - only essentials
            logger.info(f"DataFrame shape: {df.shape}")
            # logger.info(f"DataFrame columns: {list(df.columns)}")  # Too verbose
            # logger.info(f"Column mapping: {column_mapping}")  # Too verbose
            # logger.info(f"Data info: {data_info}")  # Too verbose

            if df is None or df.empty:
                logger.warning("No data available for filtering")
                return (
                    create_empty_figure(),
                    create_empty_figure(),
                    create_empty_figure(),
                    "No data available",
                    create_empty_figure(),
                    None,
                    None,
                    None,
                    None,
                    no_update,
                )

            # Determine which column to use for signal data
            signal_column = None
            time_column = None

            if column_mapping:
                # Use column mapping if available
                signal_column = column_mapping.get(
                    "signal",
                    column_mapping.get("amplitude", column_mapping.get("value")),
                )
                time_column = column_mapping.get(
                    "time", column_mapping.get("timestamp", column_mapping.get("index"))
                )
                logger.info(
                    f"Using mapped columns - Signal: {signal_column}, Time: {time_column}"
                )
                logger.info(f"Column mapping details: {column_mapping}")
            else:
                # Fallback: try to auto-detect columns
                if len(df.columns) >= 2:
                    # Assume first column is time, second is signal
                    time_column = df.columns[0]
                    signal_column = df.columns[1]
                    logger.info(
                        f"Auto-detected columns - Signal: {signal_column}, Time: {time_column}"
                    )
                    logger.info(f"Available columns: {list(df.columns)}")
                else:
                    # Single column - use it as signal
                    signal_column = df.columns[0]
                    time_column = None
                    logger.info(f"Single column detected - Signal: {signal_column}")
                    logger.info(f"Available columns: {list(df.columns)}")

            if not signal_column:
                logger.error("Could not determine signal column")
                logger.error(f"Available columns: {list(df.columns)}")
                logger.error(f"Column mapping: {column_mapping}")
                return (
                    create_empty_figure(),
                    create_empty_figure(),
                    create_empty_figure(),
                    "Could not determine signal column",
                    create_empty_figure(),
                    None,
                    None,
                    None,
                    None,
                    no_update,
                )

            # Extract FULL signal BEFORE any windowing (for filtering entire signal)
            # This allows us to store the full filtered signal for use in other screens
            full_signal_data = df[signal_column].values
            logger.info(
                f"Extracted full signal for filtering: {len(full_signal_data)} samples"
            )

            # Log the actual data sample
            logger.info(f"Signal column: {signal_column}")
            logger.info(
                f"Signal column data sample: {df[signal_column].head(10).values}"
            )

            # Check if signal column contains numeric data
            try:
                signal_data_numeric = pd.to_numeric(df[signal_column], errors="coerce")
                if signal_data_numeric.isna().any():
                    logger.warning(
                        "Signal column contains non-numeric data. Converting to numeric with coercion."
                    )
                    df[signal_column] = signal_data_numeric

                logger.info(
                    f"Signal column data info: min={df[signal_column].min()}, max={df[signal_column].max()}, mean={df[signal_column].mean():.4f}"
                )
            except Exception as e:
                logger.error(f"Error processing signal column data: {e}")
                logger.info(f"Signal column data type: {df[signal_column].dtype}")
                logger.info(
                    f"Signal column data range: {df[signal_column].min():.3f} to {df[signal_column].max():.3f}"
                )
                return (
                    create_empty_figure(),
                    create_empty_figure(),
                    create_empty_figure(),
                    "Error: Signal column contains non-numeric data that cannot be processed",
                    create_empty_figure(),
                    None,
                    None,
                    None,
                    None,
                    no_update,
                )

            # Initialize start_idx and end_idx
            start_idx = 0
            end_idx = len(df)

            # Handle time range selection
            if time_column and time_column in df.columns:
                # Use actual time column if available
                time_data = df[time_column].values
                logger.info(f"Using actual time column: {time_column}")

                # Check if time column contains datetime data
                try:
                    # First try to convert to datetime if it's not already
                    if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
                        logger.info("Converting time column to datetime")
                        df[time_column] = pd.to_datetime(
                            df[time_column], errors="coerce"
                        )

                    # Convert datetime to numeric (seconds since first timestamp)
                    if pd.api.types.is_datetime64_any_dtype(df[time_column]):
                        logger.info("Converting datetime to numeric seconds")
                        first_timestamp = df[time_column].iloc[0]
                        time_data = (
                            (df[time_column] - first_timestamp)
                            .dt.total_seconds()
                            .values
                        )
                        logger.info(f"First timestamp: {first_timestamp}")
                        logger.info(
                            "Time data converted to seconds from first timestamp"
                        )
                        logger.info(
                            f"Converted time data range: {np.min(time_data):.4f} to {np.max(time_data):.4f}"
                        )
                    else:
                        # Try numeric conversion
                        time_data_numeric = pd.to_numeric(time_data, errors="coerce")
                        if np.isnan(time_data_numeric).any():
                            logger.warning(
                                "Time column contains non-numeric data. Converting to numeric with coercion."
                            )
                            time_data = time_data_numeric

                    logger.info(
                        f"Full time data range: {np.min(time_data):.4f} to {np.max(time_data):.4f}"
                    )
                    logger.info(f"Time column data type: {type(time_data[0])}")
                    logger.info(
                        f"Time column data range: min={np.min(time_data):.4f}, max={np.max(time_data):.4f}, mean={np.mean(time_data):.4f}"
                    )
                except Exception as e:
                    logger.error(f"Error processing time column data: {e}")
                    logger.info(f"Time column data type: {type(time_data[0])}")
                    logger.info(
                        f"Time column data range: {np.min(time_data):.4f} to {np.max(time_data):.4f}"
                    )
                    # Fall back to index-based time axis
                    time_column = None
                    logger.info(
                        "Falling back to index-based time axis due to non-numeric time data"
                    )
                    # Generate time axis based on sampling frequency
                    sampling_freq = data_info.get(
                        "sampling_freq", 1000
                    )  # Use consistent key name
                    time_data = np.arange(len(df)) / sampling_freq
                    logger.info(
                        f"Generated index-based time axis: {np.min(time_data):.4f} to {np.max(time_data):.4f} seconds"
                    )

                # Calculate sample indices based on duration and sampling frequency
                if start_time is not None and end_time is not None:
                    logger.info(f"Looking for time range: {start_time} to {end_time}")
                    logger.info(
                        f"Time data type: {type(time_data[0])}, Start time type: {type(start_time)}"
                    )
                    logger.info(
                        f"Time data range: {np.min(time_data):.4f} to {np.max(time_data):.4f}"
                    )

                # Get sampling frequency from data_info (consistent key name)
                sampling_freq = data_info.get(
                    "sampling_freq", 1000
                )  # Use consistent default
                logger.info(f"Using sampling frequency: {sampling_freq} Hz")

                # Calculate number of samples needed: duration * sampling_frequency
                duration_samples = int(duration * sampling_freq)
                logger.info(
                    f"Duration {duration}s requires {duration_samples} samples at {sampling_freq} Hz"
                )

                # Calculate start sample index based on start_time
                start_sample_idx = int(start_time * sampling_freq)
                end_sample_idx = start_sample_idx + duration_samples

                logger.info(
                    f"Sample range: {start_sample_idx} to {end_sample_idx} ({duration_samples} samples)"
                )

                # Ensure we don't exceed data bounds
                if end_sample_idx > len(df):
                    logger.warning(
                        f"End sample {end_sample_idx} exceeds data length {len(df)}, adjusting"
                    )
                    end_sample_idx = len(df)
                    start_sample_idx = max(0, end_sample_idx - duration_samples)
                    logger.info(
                        f"Adjusted sample range: {start_sample_idx} to {end_sample_idx}"
                    )

                # Use sample-based indexing instead of time-based masking
                start_idx = start_sample_idx
                end_idx = end_sample_idx

                logger.info(
                    f"Final sample indices: {start_idx} to {end_idx} ({end_idx - start_idx} samples)"
                )

                # Verify we have enough data points for filtering
                min_points = 100  # Minimum points needed for filtering
                if (end_idx - start_idx) < min_points:
                    logger.warning(
                        f"Only {end_idx - start_idx} samples selected, expanding range to get at least {min_points} samples"
                    )
                    # Expand the range to get more points
                    center_idx = (start_idx + end_idx) // 2
                    half_range = min_points // 2
                    start_idx = max(0, center_idx - half_range)
                    end_idx = min(len(df), center_idx + half_range)
                    logger.info(
                        f"Expanded range: indices {start_idx} to {end_idx} ({end_idx - start_idx} samples)"
                    )

                # If still not enough points, use a larger range
                if (end_idx - start_idx) < min_points:
                    logger.warning("Still not enough points, using larger range")
                    # Use a larger range around the center
                    half_range = min(
                        min_points, len(df) // 4
                    )  # Use 1/4 of data or min_points
                    center_idx = len(df) // 2  # Use center of full dataset
                    start_idx = max(0, center_idx - half_range)
                    end_idx = min(len(df), center_idx + half_range)
                    logger.info(
                        f"Using larger range: indices {start_idx} to {end_idx} ({end_idx - start_idx} samples)"
                    )

                # If full range is too large, use a reasonable subset
                max_points = 10000  # Maximum points to process
                if (end_idx - start_idx) > max_points:
                    logger.info(
                        f"Full range too large ({end_idx - start_idx} points), using subset of {max_points} points"
                    )
                    center_idx = len(time_data) // 2
                    half_range = max_points // 2
                    start_idx = max(0, center_idx - half_range)
                    end_idx = min(len(time_data), center_idx + half_range)
                    logger.info(
                        f"Using subset: indices {start_idx} to {end_idx} ({end_idx - start_idx} points)"
                    )

                # Extract data for the selected range (for display)
                signal_data = df[signal_column].iloc[start_idx:end_idx].values

                # NEW: Check if we should use filtered signal instead of original (CASCADING FILTERS)
                if signal_source == "filtered":
                    logger.info("=== CASCADING FILTER MODE (time column path) ===")
                    logger.info(
                        "Attempting to retrieve FULL filtered signal from Enhanced Data Service..."
                    )

                    try:
                        # PRIORITY 1: Get FULL filtered signal from Enhanced Data Service
                        full_filtered_from_service = data_service.get_filtered_data(
                            latest_data_id
                        )

                        if (
                            full_filtered_from_service is not None
                            and len(full_filtered_from_service) > 0
                        ):
                            logger.info(
                                f" Retrieved FULL filtered signal from Data Service: {len(full_filtered_from_service)} samples"
                            )

                            # Use full filtered signal as the base for cascading
                            full_signal_data = full_filtered_from_service

                            # Extract window for display
                            signal_data = full_filtered_from_service[start_idx:end_idx]
                            logger.info(
                                f"Extracted window for display: {len(signal_data)} samples (from {start_idx} to {end_idx})"
                            )
                        else:
                            # PRIORITY 2: Fallback to Dash Store (windowed, less ideal)
                            logger.warning(
                                " No full filtered signal in Data Service, falling back to Dash Store (windowed)"
                            )
                            if current_filtered_signal:
                                filtered_signal_array = np.array(
                                    current_filtered_signal.get("filtered_signal", [])
                                )
                                if len(filtered_signal_array) > 0:
                                    # Use the filtered signal for the selected range
                                    filtered_start_idx = max(
                                        0,
                                        min(start_idx, len(filtered_signal_array) - 1),
                                    )
                                    filtered_end_idx = max(
                                        filtered_start_idx + 1,
                                        min(end_idx, len(filtered_signal_array)),
                                    )
                                    signal_data = filtered_signal_array[
                                        filtered_start_idx:filtered_end_idx
                                    ]
                                    # WARN: This is windowed data, not full signal
                                    full_signal_data = filtered_signal_array
                                    logger.warning(
                                        f" Using WINDOWED filtered signal from Dash Store: {len(full_signal_data)} samples (not ideal for cascading)"
                                    )
                                else:
                                    logger.warning(
                                        "Filtered signal from Dash Store is empty, using original signal"
                                    )
                            else:
                                logger.warning(
                                    "No current filtered signal in Dash Store, using original signal"
                                )
                    except Exception as e:
                        logger.error(
                            f"Error accessing filtered signal: {e}, using original signal",
                            exc_info=True,
                        )

                # Generate time axis in seconds
                # Use the converted time_data (already in seconds from datetime conversion)
                time_axis = time_data[start_idx:end_idx]
                logger.info(
                    f"Time axis in seconds: {np.min(time_axis):.4f} to {np.max(time_axis):.4f}"
                )

            else:
                # Generate time axis based on sampling frequency
                sampling_freq = data_info.get(
                    "sampling_freq", 1000
                )  # Use consistent key name
                logger.info(f"Using sampling frequency: {sampling_freq} Hz")
                logger.info(
                    f"Data info keys: {list(data_info.keys()) if data_info else 'None'}"
                )
                logger.info(
                    f"Sampling frequency from data_info: {data_info.get('sampling_frequency') if data_info else 'None'}"
                )

                if start_time is not None and end_time is not None:
                    start_idx = int(start_time * sampling_freq)
                    end_idx = int(end_time * sampling_freq)

                    # Ensure valid indices
                    start_idx = max(0, min(start_idx, len(df) - 1))
                    end_idx = max(start_idx + 1, min(end_idx, len(df)))
                    logger.info(
                        f"Time range {start_time} to {end_time} maps to indices {start_idx} to {end_idx}"
                    )
                else:
                    # Use full range if no time selection
                    start_idx = 0
                    end_idx = len(df)
                    logger.info("No time range selected, using full data")

                # Extract data for the selected range (for display)
                signal_data = df[signal_column].iloc[start_idx:end_idx].values

                # NEW: Check if we should use filtered signal instead of original (CASCADING FILTERS)
                if signal_source == "filtered":
                    logger.info("=== CASCADING FILTER MODE (no time column path) ===")
                    logger.info(
                        "Attempting to retrieve FULL filtered signal from Enhanced Data Service..."
                    )

                    try:
                        # PRIORITY 1: Get FULL filtered signal from Enhanced Data Service
                        full_filtered_from_service = data_service.get_filtered_data(
                            latest_data_id
                        )

                        if (
                            full_filtered_from_service is not None
                            and len(full_filtered_from_service) > 0
                        ):
                            logger.info(
                                f" Retrieved FULL filtered signal from Data Service: {len(full_filtered_from_service)} samples"
                            )

                            # Use full filtered signal as the base for cascading
                            full_signal_data = full_filtered_from_service

                            # Extract window for display
                            signal_data = full_filtered_from_service[start_idx:end_idx]
                            logger.info(
                                f"Extracted window for display: {len(signal_data)} samples (from {start_idx} to {end_idx})"
                            )
                        else:
                            # PRIORITY 2: Fallback to Dash Store (windowed, less ideal)
                            logger.warning(
                                " No full filtered signal in Data Service, falling back to Dash Store (windowed)"
                            )
                            if current_filtered_signal:
                                filtered_signal_array = np.array(
                                    current_filtered_signal.get("filtered_signal", [])
                                )
                                if len(filtered_signal_array) > 0:
                                    # Use the filtered signal for the selected range
                                    filtered_start_idx = max(
                                        0,
                                        min(start_idx, len(filtered_signal_array) - 1),
                                    )
                                    filtered_end_idx = max(
                                        filtered_start_idx + 1,
                                        min(end_idx, len(filtered_signal_array)),
                                    )
                                    signal_data = filtered_signal_array[
                                        filtered_start_idx:filtered_end_idx
                                    ]
                                    # WARN: This is windowed data, not full signal
                                    full_signal_data = filtered_signal_array
                                    logger.warning(
                                        f" Using WINDOWED filtered signal from Dash Store: {len(full_signal_data)} samples (not ideal for cascading)"
                                    )
                                else:
                                    logger.warning(
                                        "Filtered signal is empty, using original signal"
                                    )
                            else:
                                logger.warning(
                                    "No current filtered signal in Dash Store, using original signal"
                                )
                    except Exception as e:
                        logger.error(
                            f"Error accessing filtered signal: {e}, using original signal",
                            exc_info=True,
                        )

                # Create time axis that matches the signal data length
                # Start from the actual time of start_idx to maintain correct time reference
                time_axis = (np.arange(len(signal_data)) + start_idx) / sampling_freq
                logger.info(
                    f"Created time axis: length={len(time_axis)}, range={time_axis[0]:.3f}s to {time_axis[-1]:.3f}s"
                )

            # Ensure signal data is numeric
            try:
                signal_data = pd.to_numeric(signal_data, errors="coerce")
                if np.isnan(signal_data).any():
                    logger.warning(
                        "Signal data contains non-numeric values. Converting with coercion."
                    )
                    # Remove NaN values
                    valid_mask = ~np.isnan(signal_data)
                    signal_data = signal_data[valid_mask]
                    time_axis = time_axis[valid_mask]
                    logger.info(
                        f"Removed {np.sum(~valid_mask)} non-numeric values from signal data"
                    )
            except Exception as e:
                logger.error(f"Error converting signal data to numeric: {e}")
                return (
                    create_empty_figure(),
                    create_empty_figure(),
                    create_empty_figure(),
                    "Error: Signal data contains non-numeric values that cannot be processed",
                    create_empty_figure(),
                    None,
                    None,
                    None,
                    None,
                    no_update,
                )

            # Ensure time axis is valid for plotting
            if np.isnan(time_axis).any() or np.isinf(time_axis).any():
                logger.warning(
                    "Time axis contains invalid values (NaN or Inf). Generating index-based time axis."
                )
                time_axis = np.arange(len(signal_data)) / sampling_freq
                logger.info(
                    f"Generated index-based time axis: {np.min(time_axis):.4f} to {np.max(time_axis):.4f} seconds"
                )

            logger.info(f"Signal data shape: {signal_data.shape}")
            logger.info(f"Time axis shape: {time_axis.shape}")

            # CRITICAL CHECK: Ensure time_axis and signal_data have same length
            if len(time_axis) != len(signal_data):
                logger.error(
                    f"LENGTH MISMATCH: time_axis={len(time_axis)}, signal_data={len(signal_data)}"
                )
                logger.error(
                    "This will cause empty plots! Fixing by regenerating time_axis..."
                )
                time_axis = np.arange(len(signal_data)) / sampling_freq
                logger.info(f"Regenerated time_axis with length {len(time_axis)}")

            safe_log_range(logger, signal_data, "Signal data")
            if len(signal_data) > 0:
                logger.info(
                    f"Signal data sample: {signal_data[:5] if len(signal_data) >= 5 else signal_data}"
                )
                logger.info(
                    f"Time axis sample: {time_axis[:5] if len(time_axis) >= 5 else time_axis}"
                )
            else:
                logger.warning("Signal data is empty - cannot compute range or sample")

            logger.info("Final data extraction:")
            logger.info(f"  Start index: {start_idx}")
            logger.info(f"  End index: {end_idx}")
            logger.info(f"  Signal data length: {len(signal_data)}")
            logger.info(f"  Time axis length: {len(time_axis)}")
            logger.info(f"  Signal data shape: {signal_data.shape}")
            logger.info(f"  Time axis shape: {time_axis.shape}")

            safe_log_range(logger, signal_data, "  Signal data")
            safe_log_range(logger, time_axis, "  Time axis")

            if len(signal_data) > 0:
                logger.info(
                    f"  Signal data sample: {signal_data[:5] if len(signal_data) >= 5 else signal_data}"
                )
            if len(time_axis) > 0:
                logger.info(
                    f"  Time axis sample: {time_axis[:5] if len(time_axis) >= 5 else time_axis}"
                )

            # Check if we have valid data
            if len(signal_data) == 0:
                logger.error("No signal data extracted")
                return (
                    create_empty_figure(),
                    create_empty_figure(),
                    create_empty_figure(),
                    "No signal data extracted",
                    create_empty_figure(),
                    None,
                    None,
                    None,
                    None,
                    no_update,
                )

            # Store raw signal for plotting (before any preprocessing)
            raw_signal_for_plotting = signal_data.copy()
            logger.info(
                f"Raw signal stored for plotting - mean: {np.mean(raw_signal_for_plotting):.4f}, range: {np.min(raw_signal_for_plotting):.4f} to {np.max(raw_signal_for_plotting):.4f}"
            )

            # Store original signal for comparison (before any preprocessing)
            original_signal = signal_data.copy()

            # Get sampling frequency from data_info (use consistent key name)
            sampling_freq = data_info.get(
                "sampling_freq", 1000
            )  # Use consistent default
            logger.info(f"Sampling frequency: {sampling_freq} Hz")

            # Process FULL signal for filtering entire signal (to store in data service)
            # This allows other screens to extract any window from the filtered signal
            full_signal_for_filtering = None
            if full_signal_data is not None and len(full_signal_data) > 0:
                logger.info(
                    f"Processing full signal for filtering: {len(full_signal_data)} samples"
                )
                # Ensure full signal is numeric
                try:
                    full_signal_for_filtering = pd.to_numeric(
                        full_signal_data, errors="coerce"
                    )
                    if np.isnan(full_signal_for_filtering).any():
                        valid_mask = ~np.isnan(full_signal_for_filtering)
                        full_signal_for_filtering = full_signal_for_filtering[
                            valid_mask
                        ]
                        logger.info(
                            f"Removed {np.sum(~valid_mask)} non-numeric values from full signal"
                        )
                except Exception as e:
                    logger.warning(
                        f"Error converting full signal to numeric: {e}, skipping full signal filtering"
                    )
                    full_signal_for_filtering = None
            else:
                logger.info(
                    "Full signal data not available, will filter only windowed signal"
                )

            # Detrending (optional, off by default).
            #
            # Was: ``ArtifactRemoval(signal).baseline_correction(cutoff=0.5,
            # fs=...)`` followed by a redundant mean subtraction.
            # ``baseline_correction`` is internally a 0.5 Hz Butterworth
            # high-pass - i.e. ANOTHER filter applied silently before the
            # user's filter, with several bugs:
            #   * For respiratory signals (0.1-0.5 Hz) it wipes the
            #     entire signal band.
            #   * Combined with the user's bandpass it became a stacked
            #     filter with doubled order + ringing.
            #   * The label said "Detrending" but the implementation was
            #     a high-pass filter, not polynomial / linear detrending.
            #
            # vitalDSP has a proper detrender at
            # ``VitalTransformation.apply_detrending`` (linear / constant
            # / polynomial).  We use that here.  The trace we save as
            # ``raw_signal_for_plotting`` is also overwritten with the
            # detrended version so the comparison plot reflects what the
            # filter actually saw - the previous code plotted the raw
            # signal as "Original" but ran the filter on a different
            # signal, which gave results different from a direct
            # function call.
            if detrend_option and "detrend" in detrend_option:
                logger.info(
                    "Applying linear detrending via VitalTransformation"
                )
                try:
                    from vitalDSP.transforms.vital_transformation import (
                        VitalTransformation,
                    )

                    transformer = VitalTransformation(
                        signal_data, fs=sampling_freq, signal_type=signal_type
                    )
                    transformer.apply_detrending(
                        options={"detrend_type": "linear"}
                    )
                    signal_data = np.asarray(transformer.signal, dtype=float)
                    # Keep the displayed "Original" trace consistent with
                    # what the filter operates on, so the user's direct-
                    # function-call check now matches the plot.
                    raw_signal_for_plotting = signal_data.copy()
                    logger.info(
                        "Detrend applied: signal_mean=%.4f, range=[%.4f, %.4f]",
                        float(np.mean(signal_data)),
                        float(np.min(signal_data)),
                        float(np.max(signal_data)),
                    )
                except Exception as exc:
                    logger.warning(
                        "Detrend failed; using raw signal.  Cause: %s", exc
                    )

                # Apply the same detrend to the full signal so the
                # stored "full filtered signal" downstream is computed
                # from a consistent preprocessing pipeline.
                if full_signal_for_filtering is not None:
                    try:
                        from vitalDSP.transforms.vital_transformation import (
                            VitalTransformation,
                        )

                        full_transformer = VitalTransformation(
                            full_signal_for_filtering,
                            fs=sampling_freq,
                            signal_type=signal_type,
                        )
                        full_transformer.apply_detrending(
                            options={"detrend_type": "linear"}
                        )
                        full_signal_for_filtering = np.asarray(
                            full_transformer.signal, dtype=float
                        )
                    except Exception as exc:
                        logger.warning(
                            "Full-signal detrend failed; continuing with "
                            "un-detrended full signal.  Cause: %s",
                            exc,
                        )
            else:
                logger.info("Detrending not selected, using original signal baseline")
                logger.info(
                    f"Signal baseline - mean: {np.mean(signal_data):.4f}, range: {np.min(signal_data):.4f} to {np.max(signal_data):.4f}"
                )

            # Check if we have enough data points for filtering
            if len(signal_data) < 50:
                logger.warning(
                    f"Only {len(signal_data)} data points available, this may not be enough for effective filtering"
                )
                # Use vitalDSP convolution-based smoothing instead of numpy convolution
                logger.info(
                    "Using vitalDSP convolution-based smoothing instead of numpy convolution"
                )
                from vitalDSP.filtering.advanced_signal_filtering import (
                    AdvancedSignalFiltering,
                )

                af = AdvancedSignalFiltering(signal_data)
                filtered_data = af.convolution_based_filter(
                    kernel_type="smoothing", kernel_size=3
                )
            else:
                logger.info(
                    f"Sufficient data points ({len(signal_data)}) for filtering"
                )

            # Get sampling frequency for filtering
            sampling_freq = data_info.get(
                "sampling_freq", 100
            )  # Default to 100 Hz based on the data info
            logger.info(f"Using sampling frequency for filtering: {sampling_freq} Hz")

            # Filter chain (queued stages) runs BEFORE the panel filter
            # below.  We overwrite both ``signal_data`` (windowed input
            # used for the comparison plot) and ``full_signal_for_filtering``
            # (full-recording input used for the saved-as-preprocessing
            # filtered signal) so that downstream pages and the live
            # plot see the SAME ordering: detrend -> chain -> panel.
            chain_in_use = bool(filter_chain)
            if chain_in_use:
                logger.info(
                    "Filter chain has %d stage(s); running chain then panel filter.",
                    len(filter_chain),
                )
                signal_data = apply_filter_chain(
                    signal_data,
                    sampling_freq,
                    signal_type,
                    filter_chain,
                    logger=logger,
                )
                if full_signal_for_filtering is not None:
                    full_signal_for_filtering = apply_filter_chain(
                        full_signal_for_filtering,
                        sampling_freq,
                        signal_type,
                        filter_chain,
                        logger=logger,
                    )

            # FILTER FULL SIGNAL FIRST (for storage in data service)
            # This allows other screens to extract any window from the filtered signal
            full_filtered_signal = None
            if (
                full_signal_for_filtering is not None
                and len(full_signal_for_filtering) > 0
            ):
                logger.info(f"=== FILTERING FULL SIGNAL FOR STORAGE ===")
                logger.info(
                    f"Full signal length: {len(full_signal_for_filtering)} samples"
                )

                try:
                    # Apply the same filtering logic to the full signal
                    full_filtered_temp = full_signal_for_filtering.copy()

                    # Apply filter n times (iterative filtering)
                    filter_count_full = max(1, min(filter_count or 1, 10))

                    for iteration in range(filter_count_full):
                        current_full_input = full_filtered_temp.copy()
                        logger.info(
                            f"=== Full Signal Filter Iteration {iteration + 1}/{filter_count_full} ==="
                        )
                        logger.info(f"Filter type for FULL signal: '{filter_type}'")
                        logger.info(f"Advanced method: '{advanced_method}'")

                        if len(current_full_input) >= 50:
                            if filter_type == "traditional":
                                logger.info(
                                    "Applying TRADITIONAL filter to FULL signal"
                                )
                                filter_family = filter_family or "butter"
                                filter_response = filter_response or "low"
                                low_freq = low_freq or 10
                                high_freq = high_freq or 50
                                filter_order = filter_order or 4

                                full_filtered_temp = apply_traditional_filter(
                                    current_full_input,
                                    sampling_freq,
                                    filter_family,
                                    filter_response,
                                    low_freq,
                                    high_freq,
                                    filter_order,
                                )
                            elif filter_type == "smoothing":
                                logger.info("Applying SMOOTHING filter to FULL signal")
                                full_filtered_temp = apply_smoothing_filter(
                                    current_full_input,
                                    smoothing_method,
                                    savgol_window,
                                    savgol_polyorder,
                                    moving_avg_window,
                                    gaussian_sigma,
                                )
                            elif filter_type == "advanced":
                                logger.info("Applying ADVANCED filter to FULL signal")
                                full_filtered_temp = apply_advanced_filter(
                                    current_full_input,
                                    advanced_method,
                                    kalman_r=kalman_r,
                                    kalman_q=kalman_q,
                                    optimization_loss_type=optimization_loss_type,
                                    optimization_initial_guess=optimization_initial_guess,
                                    optimization_learning_rate=optimization_learning_rate,
                                    optimization_iterations=optimization_iterations,
                                    gradient_learning_rate=gradient_learning_rate,
                                    gradient_iterations=gradient_iterations,
                                    convolution_kernel_type=convolution_kernel_type,
                                    convolution_kernel_size=convolution_kernel_size,
                                    attention_type=attention_type,
                                    attention_size=attention_size,
                                    attention_sigma=attention_sigma,
                                    attention_ascending=attention_ascending,
                                    attention_base=attention_base,
                                    adaptive_mu=adaptive_mu,
                                    adaptive_order=adaptive_order,
                                )
                            elif filter_type == "artifact":
                                full_filtered_temp = apply_enhanced_artifact_removal(
                                    current_full_input,
                                    sampling_freq,
                                    artifact_type,
                                    artifact_strength,
                                    None,
                                    None,
                                    None,
                                    None,
                                    None,
                                    None,
                                    None,
                                    None,
                                    None,
                                )
                            elif filter_type == "neural":
                                full_filtered_temp = apply_neural_filter(
                                    current_full_input, neural_type, neural_complexity
                                )
                            elif filter_type == "ensemble":
                                full_filtered_temp = apply_enhanced_ensemble_filter(
                                    current_full_input,
                                    ensemble_method,
                                    ensemble_n_filters,
                                    None,
                                    None,
                                    None,
                                    None,
                                    None,
                                    None,
                                )
                            else:
                                full_filtered_temp = current_full_input
                        else:
                            # Simple smoothing for small signals
                            from vitalDSP.filtering.advanced_signal_filtering import (
                                AdvancedSignalFiltering,
                            )

                            af_full = AdvancedSignalFiltering(current_full_input)
                            full_filtered_temp = af_full.convolution_based_filter(
                                kernel_type="smoothing", kernel_size=3
                            )

                    full_filtered_signal = full_filtered_temp
                    logger.info(
                        f" Full signal filtered successfully: {len(full_filtered_signal)} samples"
                    )
                except Exception as e:
                    logger.warning(
                        f"Error filtering full signal: {e}, will store windowed filtered signal only",
                        exc_info=True,
                    )
                    full_filtered_signal = None

            # Panel filter: runs on the signal that emerges from
            # detrend + (optional) chain stages above.  When the chain
            # is empty, this is effectively the "single-filter" path
            # the page started with.  When the chain has stages, this
            # acts as the implicit final stage on top of the chain
            # output - matching the "chain + panel" Apply semantics.
            if True:  # kept as ``if True`` to preserve the indent of the legacy block below
                logger.info(
                    "=== APPLYING PANEL FILTER (on top of chain if any) ==="
                )
                logger.info(f"Filter type: {filter_type}")
                logger.info(f"Input signal shape: {signal_data.shape}")
                current_input = signal_data.copy()
                if len(current_input) < 50:
                    logger.info(
                        "Less than 50 samples; using vitalDSP smoothing."
                    )
                    from vitalDSP.filtering.advanced_signal_filtering import (
                        AdvancedSignalFiltering,
                    )
                    af = AdvancedSignalFiltering(current_input)
                    filtered_data = af.convolution_based_filter(
                        kernel_type="smoothing", kernel_size=3
                    )
                elif filter_type == "traditional":
                    filter_family = filter_family or "butter"
                    filter_response = filter_response or "low"
                    filtered_data = apply_traditional_filter(
                        current_input,
                        sampling_freq,
                        filter_family,
                        filter_response,
                        low_freq or 10,
                        high_freq or 50,
                        int(filter_order or 4),
                    )
                elif filter_type == "smoothing":
                    filtered_data = apply_smoothing_filter(
                        current_input,
                        smoothing_method,
                        savgol_window,
                        savgol_polyorder,
                        moving_avg_window,
                        gaussian_sigma,
                    )
                elif filter_type == "advanced":
                    filtered_data = apply_advanced_filter(
                        current_input,
                        advanced_method,
                        kalman_r=kalman_r,
                        kalman_q=kalman_q,
                        optimization_loss_type=optimization_loss_type,
                        optimization_initial_guess=optimization_initial_guess,
                        optimization_learning_rate=optimization_learning_rate,
                        optimization_iterations=optimization_iterations,
                        gradient_learning_rate=gradient_learning_rate,
                        gradient_iterations=gradient_iterations,
                        convolution_kernel_type=convolution_kernel_type,
                        convolution_kernel_size=convolution_kernel_size,
                        attention_type=attention_type,
                        attention_size=attention_size,
                        attention_sigma=attention_sigma,
                        attention_ascending=attention_ascending,
                        attention_base=attention_base,
                        adaptive_mu=adaptive_mu,
                        adaptive_order=adaptive_order,
                    )
                elif filter_type == "artifact":
                    filtered_data = apply_enhanced_artifact_removal(
                        current_input,
                        sampling_freq,
                        artifact_type,
                        artifact_strength,
                        None, None, None, None, None, None, None, None, None,
                    )
                elif filter_type == "neural":
                    filtered_data = apply_neural_filter(
                        current_input, neural_type, neural_complexity,
                    )
                elif filter_type == "ensemble":
                    filtered_data = apply_enhanced_ensemble_filter(
                        current_input,
                        ensemble_method,
                        ensemble_n_filters,
                        None, None, None, None, None, None,
                    )
                else:
                    filtered_data = current_input
                logger.info(
                    "Filtered range: [%.4f, %.4f]",
                    float(np.min(filtered_data)), float(np.max(filtered_data)),
                )

            # Apply multi-modal filtering if reference signal is specified
            # Note: These parameters would need to be added to the callback inputs
            if False:  # Placeholder for multi-modal filtering
                filtered_data = apply_multi_modal_filtering(
                    filtered_data, None, None, None, None
                )

            # Create visualizations
            logger.info("=== CREATING PLOTS ===")
            logger.info(f"Raw signal data shape: {raw_signal_for_plotting.shape}")
            logger.info(f"Processed signal data shape: {signal_data.shape}")
            logger.info(f"Filtered signal data shape: {filtered_data.shape}")
            logger.info(f"Time axis shape: {time_axis.shape}")

            # CRITICAL: Verify lengths match before plotting
            if len(time_axis) != len(raw_signal_for_plotting):
                logger.error(
                    f"MISMATCH BEFORE PLOT: time_axis={len(time_axis)}, raw_signal={len(raw_signal_for_plotting)}"
                )
                logger.error("Adjusting time_axis to match raw_signal length...")
                time_axis = np.arange(len(raw_signal_for_plotting)) / sampling_freq
                logger.info(f"Adjusted time_axis to length {len(time_axis)}")

            if len(filtered_data) != len(raw_signal_for_plotting):
                logger.warning(
                    f"Filtered data length ({len(filtered_data)}) != raw signal length ({len(raw_signal_for_plotting)})"
                )

            logger.info(
                f"Raw signal range: {np.min(raw_signal_for_plotting):.4f} to {np.max(raw_signal_for_plotting):.4f}"
            )
            logger.info(
                f"Processed signal range: {np.min(signal_data):.4f} to {np.max(signal_data):.4f}"
            )
            logger.info(
                f"Filtered signal range: {np.min(filtered_data):.4f} to {np.max(filtered_data):.4f}"
            )
            logger.info(f"Time axis range: {time_axis[0]:.4f}s to {time_axis[-1]:.4f}s")

            # Shared WaveformMorphology pass: detect peaks ONCE on the
            # filtered signal and reuse them for the comparison plot.
            # Previously the comparison-plot / original-plot / filtered-
            # plot builders all ran their own peak detection.
            comparison_peaks = None
            comparison_notches = None
            try:
                from vitalDSP.physiological_features.waveform import (
                    WaveformMorphology,
                )

                wm_filt = WaveformMorphology(
                    waveform=np.asarray(filtered_data, dtype=float),
                    fs=sampling_freq,
                    signal_type=(signal_type or "PPG"),
                    simple_mode=True,
                )
                comparison_peaks = getattr(wm_filt, "systolic_peaks", None)
                comparison_notches = getattr(wm_filt, "dicrotic_notches", None)
            except Exception as exc:
                logger.warning(
                    "Shared WaveformMorphology failed on filtered signal: %s",
                    exc,
                )

            # Slim layout: ``filter-original-plot`` /
            # ``filter-filtered-plot`` / ``filter-quality-metrics`` /
            # ``filter-quality-plots`` are hidden carriers (the visible
            # surfaces are the comparison plot + segment-quality card).
            # Skip the heavy figure builders for them — saves several
            # hundred ms per Apply on long recordings.  Comparison plot
            # is the only one with a visible Output, so build that.
            original_plot = no_update
            filtered_plot = no_update
            comparison_plot = create_filter_comparison_plot(
                time_axis,
                raw_signal_for_plotting,
                filtered_data,
                sampling_freq,
                signal_type,
                peaks=comparison_peaks,
                notches=comparison_notches,
            )
            quality_metrics = no_update
            quality_plots = no_update

            # Compact descriptor of the filter that was just run.  The
            # large arrays (raw_signal / filtered_signal / time_axis)
            # used to live here, but nothing downstream reads
            # ``store-filtering-data``; we only need the parameters
            # block for building filter_info below.
            stored_data = {
                "filter_type": filter_type,
                "detrending_applied": detrend_option and "detrend" in detrend_option,
                "signal_source": signal_source,
                "filter_count": filter_count,
                "parameters": {
                    "filter_family": filter_family,
                    "filter_response": filter_response,
                    "low_freq": low_freq,
                    "high_freq": high_freq,
                    "filter_order": filter_order,
                    "advanced_method": advanced_method,
                    "artifact_type": artifact_type,
                    "ensemble_method": ensemble_method,
                    "signal_source": signal_source,
                    "filter_count": filter_count,
                },
            }

            # Store filtered data in data service for use in other screens
            try:
                from vitalDSP_webapp.services.data.enhanced_data_service import (
                    get_enhanced_data_service,
                )

                data_service = get_enhanced_data_service()

                # Create filter info for storage.  The "effective chain"
                # we save is the queued stages plus the panel filter as
                # an implicit final stage - that matches what just ran
                # on Apply, so downstream pages re-apply the same
                # sequence and produce identical output to the comparison
                # plot the user just saw.
                panel_stage = _panel_filter_as_stage(
                    filter_type,
                    filter_family,
                    filter_response,
                    low_freq,
                    high_freq,
                    filter_order,
                    advanced_method,
                    artifact_type,
                    artifact_strength,
                    neural_type,
                    neural_complexity,
                    ensemble_method,
                    ensemble_n_filters,
                    smoothing_method=smoothing_method,
                    savgol_window=savgol_window,
                    savgol_polyorder=savgol_polyorder,
                    moving_avg_window=moving_avg_window,
                    gaussian_sigma=gaussian_sigma,
                )
                effective_chain = list(filter_chain or []) + (
                    [panel_stage] if panel_stage else []
                )
                filter_info = {
                    "filter_type": "chain" if effective_chain else filter_type,
                    "parameters": stored_data["parameters"],
                    "detrending_applied": stored_data["detrending_applied"],
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "chain": effective_chain or None,
                }

                # Store the FULL filtered signal data (not windowed) so other screens can extract any window
                signal_to_store = (
                    full_filtered_signal
                    if full_filtered_signal is not None
                    else filtered_data
                )
                logger.info(
                    f"Storing {'FULL' if full_filtered_signal is not None else 'WINDOWED'} filtered signal: {len(signal_to_store)} samples"
                )

                # Log signal statistics for debugging
                logger.info(
                    f" STORED Filtered signal stats - min: {np.min(signal_to_store):.4f}, max: {np.max(signal_to_store):.4f}, mean: {np.mean(signal_to_store):.4f}"
                )

                success = data_service.store_filtered_data(
                    latest_data_id, signal_to_store, filter_info
                )

                if success:
                    logger.info(
                        f"Filtered data stored successfully for data ID: {latest_data_id}"
                    )
                else:
                    logger.warning(
                        f"Failed to store filtered data for data ID: {latest_data_id}"
                    )

            except Exception as e:
                logger.error(f"Error storing filtered data: {e}")

            logger.info("Advanced filtering completed successfully")
            logger.info(
                f"Returning plots - original_plot type: {type(original_plot)}, has {len(original_plot.data) if hasattr(original_plot, 'data') else 'N/A'} traces"
            )
            logger.info(
                f"Returning plots - filtered_plot type: {type(filtered_plot)}, has {len(filtered_plot.data) if hasattr(filtered_plot, 'data') else 'N/A'} traces"
            )
            logger.info(
                f"Returning plots - comparison_plot type: {type(comparison_plot)}, has {len(comparison_plot.data) if hasattr(comparison_plot, 'data') else 'N/A'} traces"
            )
            # ``store-filter-comparison`` and ``store-filter-quality-metrics``
            # are no longer read by any callback.  Skip building their
            # payloads and emit ``no_update`` so the apply path doesn't
            # waste a full signal serialisation per click.
            comparison_data = no_update
            quality_metrics_data = no_update

            filtered_signal_data = {
                "signal": filtered_data.tolist() if filtered_data is not None else [],
                "filtered_signal": (
                    filtered_data.tolist() if filtered_data is not None else []
                ),  # NEW: Also store with this key for compatibility
                "time": time_axis.tolist() if time_axis is not None else [],
                "sampling_freq": sampling_freq,
                "filter_type": filter_type,
                "signal_source": signal_source,  # NEW: Store signal source
                "filter_count": filter_count,  # NEW: Store filter application count
                "filter_params": {
                    "filter_family": filter_family,
                    "filter_response": filter_response,
                    "low_freq": low_freq,
                    "high_freq": high_freq,
                    "filter_order": filter_order,
                    "signal_source": signal_source,  # NEW
                    "filter_count": filter_count,  # NEW
                },
                "signal_type": signal_type,
                "timestamp": pd.Timestamp.now().isoformat(),
            }

            # The saved-as-preprocessing banner used to live here.  It was
            # information noise — the cross-page preprocessing behaviour is
            # already documented and visible (other pages' "Filtered" mode
            # toggles).  Output kept as ``no_update`` so the Output list
            # arity is preserved without writing anything user-visible.
            # ``store-filtering-data`` is also unread by downstream
            # callbacks now; emit no_update so we don't serialise a
            # large dict on every Apply.
            return (
                original_plot,
                filtered_plot,
                comparison_plot,
                quality_metrics,
                quality_plots,
                no_update,           # store-filtering-data (unread)
                comparison_data,     # already no_update
                quality_metrics_data,  # already no_update
                filtered_signal_data,
                no_update,           # filter-saved-banner
            )

        except Exception as e:
            logger.error(f"Error in advanced filtering callback: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")

            # Handle string formatting errors specifically
            if "format code" in str(e) and "object of type 'str'" in str(e):
                error_msg = "Error: Data contains non-numeric values that cannot be processed. Please check your data format."
            else:
                error_msg = f"Error during filtering: {str(e)}"

            return (
                create_empty_figure(),
                create_empty_figure(),
                create_empty_figure(),
                error_msg,
                create_empty_figure(),
                None,
                None,
                None,
                None,
                no_update,
            )

    # Removed update_start_position_slider callback - duplicate with time domain callback


# Helper functions for signal filtering
def create_original_signal_plot(time_axis, signal_data, sampling_freq, signal_type):
    """Create plot for original signal with critical points detection."""
    try:
        logger.info("=" * 80)
        logger.info("CREATING ORIGINAL SIGNAL PLOT")
        logger.info(
            f"  Time axis type: {type(time_axis)}, shape: {time_axis.shape if hasattr(time_axis, 'shape') else 'N/A'}"
        )
        logger.info(
            f"  Signal data type: {type(signal_data)}, shape: {signal_data.shape if hasattr(signal_data, 'shape') else 'N/A'}"
        )
        logger.info(f"  Time axis length: {len(time_axis)}")
        logger.info(f"  Signal data length: {len(signal_data)}")
        logger.info(
            f"  Time axis range: {np.min(time_axis):.4f} to {np.max(time_axis):.4f}"
        )
        logger.info(
            f"  Signal data range: {np.min(signal_data):.4f} to {np.max(signal_data):.4f}"
        )
        logger.info(
            f"  Time axis sample: {time_axis[:5] if len(time_axis) >= 5 else time_axis}"
        )
        logger.info(
            f"  Signal data sample: {signal_data[:5] if len(signal_data) >= 5 else signal_data}"
        )

        # PERFORMANCE OPTIMIZATION: Limit plot data to max 5 minutes and 10K points
        time_axis_plot, signal_data_plot = limit_plot_data(
            time_axis,
            signal_data,
            max_duration=300,  # 5 minutes max
            max_points=10000,  # 10K points max
        )

        logger.info(
            f"Plot data limited: {len(signal_data)} → {len(signal_data_plot)} points"
        )
        logger.info(f"Time axis plot length: {len(time_axis_plot)}")
        logger.info(f"Signal data plot length: {len(signal_data_plot)}")

        fig = go.Figure()

        # Add main signal (using limited data)
        logger.info(
            f"Adding trace with x length: {len(time_axis_plot)}, y length: {len(signal_data_plot)}"
        )
        fig.add_trace(
            go.Scatter(
                x=time_axis_plot,
                y=signal_data_plot,
                mode="lines",
                name="Original Signal",
                line=dict(color="blue", width=2),
            )
        )
        logger.info(f"Figure now has {len(fig.data)} traces")

        # Add critical points detection using vitalDSP waveform module
        # NOTE: Use limited data for peak detection to match the plot
        try:
            from vitalDSP.physiological_features.waveform import WaveformMorphology

            # Create waveform morphology object (use FULL data for accurate peak detection)
            wm = WaveformMorphology(
                waveform=signal_data,  # Use FULL data for accurate peak detection
                fs=sampling_freq,  # Use actual sampling frequency
                signal_type=signal_type,  # Use signal type from UI
                simple_mode=True,
            )

            # Detect critical points based on signal type
            if signal_type == "PPG":
                # For PPG: systolic peaks, dicrotic notches, diastolic peaks
                if hasattr(wm, "systolic_peaks") and wm.systolic_peaks is not None:
                    # Filter peaks to only show those within the plot time range
                    plot_start_time = np.min(time_axis_plot)
                    plot_end_time = np.max(time_axis_plot)
                    plot_start_idx = int(plot_start_time * sampling_freq)
                    plot_end_idx = int(plot_end_time * sampling_freq)

                    # Filter peaks to only those within the plot range
                    valid_peaks = wm.systolic_peaks[
                        (wm.systolic_peaks >= plot_start_idx)
                        & (wm.systolic_peaks < plot_end_idx)
                    ]

                    if len(valid_peaks) > 0:
                        # Convert peak indices to plot time coordinates
                        peak_times = valid_peaks / sampling_freq
                        peak_values = signal_data[valid_peaks]

                        fig.add_trace(
                            go.Scatter(
                                x=peak_times,
                                y=peak_values,
                                mode="markers",
                                name="Systolic Peaks",
                                marker=dict(color="red", size=10, symbol="diamond"),
                                hovertemplate="<b>Systolic Peak:</b> %{y}<br><b>Time:</b> %{x:.3f}s<extra></extra>",
                            )
                        )

                # Detect and plot dicrotic notches
                try:
                    dicrotic_notches = wm.detect_dicrotic_notches()
                    if dicrotic_notches is not None and len(dicrotic_notches) > 0:
                        # Filter dicrotic notches to only show those within the plot time range
                        plot_start_time = np.min(time_axis_plot)
                        plot_end_time = np.max(time_axis_plot)
                        plot_start_idx = int(plot_start_time * sampling_freq)
                        plot_end_idx = int(plot_end_time * sampling_freq)

                        # Filter dicrotic notches to only those within the plot range
                        valid_notches = dicrotic_notches[
                            (dicrotic_notches >= plot_start_idx)
                            & (dicrotic_notches < plot_end_idx)
                        ]

                        if len(valid_notches) > 0:
                            # Convert notch indices to plot time coordinates
                            notch_times = valid_notches / sampling_freq
                            notch_values = signal_data[valid_notches]

                            fig.add_trace(
                                go.Scatter(
                                    x=notch_times,
                                    y=notch_values,
                                    mode="markers",
                                    name="Dicrotic Notches",
                                    marker=dict(
                                        color="orange", size=8, symbol="circle"
                                    ),
                                    hovertemplate="<b>Dicrotic Notch:</b> %{y}<br><b>Time:</b> %{x:.3f}s<extra></extra>",
                                )
                            )
                except Exception as e:
                    logger.warning(f"Dicrotic notch detection failed: {e}")

                # Detect and plot diastolic peaks
                try:
                    diastolic_peaks = wm.detect_diastolic_peak()
                    if diastolic_peaks is not None and len(diastolic_peaks) > 0:
                        # Filter diastolic peaks to only show those within the plot time range
                        plot_start_time = np.min(time_axis_plot)
                        plot_end_time = np.max(time_axis_plot)
                        plot_start_idx = int(plot_start_time * sampling_freq)
                        plot_end_idx = int(plot_end_time * sampling_freq)

                        # Filter diastolic peaks to only those within the plot range
                        valid_peaks = diastolic_peaks[
                            (diastolic_peaks >= plot_start_idx)
                            & (diastolic_peaks < plot_end_idx)
                        ]

                        if len(valid_peaks) > 0:
                            # Convert peak indices to plot time coordinates
                            peak_times = valid_peaks / sampling_freq
                            peak_values = signal_data[valid_peaks]

                            fig.add_trace(
                                go.Scatter(
                                    x=peak_times,
                                    y=peak_values,
                                    mode="markers",
                                    name="Diastolic Peaks",
                                    marker=dict(color="green", size=8, symbol="square"),
                                    hovertemplate="<b>Diastolic Peak:</b> %{y}<br><b>Time:</b> %{x:.3f}s<extra></extra>",
                                )
                            )
                except Exception as e:
                    logger.warning(f"Diastolic peak detection failed: {e}")

            elif signal_type == "ECG":
                # For ECG: R peaks, P peaks, T peaks
                if hasattr(wm, "r_peaks") and wm.r_peaks is not None:
                    # Filter R peaks to only show those within the plot time range
                    plot_start_time = np.min(time_axis_plot)
                    plot_end_time = np.max(time_axis_plot)
                    plot_start_idx = int(plot_start_time * sampling_freq)
                    plot_end_idx = int(plot_end_time * sampling_freq)

                    # Filter R peaks to only those within the plot range
                    valid_peaks = wm.r_peaks[
                        (wm.r_peaks >= plot_start_idx) & (wm.r_peaks < plot_end_idx)
                    ]

                    if len(valid_peaks) > 0:
                        # Convert peak indices to plot time coordinates
                        peak_times = valid_peaks / sampling_freq
                        peak_values = signal_data[valid_peaks]

                        fig.add_trace(
                            go.Scatter(
                                x=peak_times,
                                y=peak_values,
                                mode="markers",
                                name="R Peaks",
                                marker=dict(color="red", size=10, symbol="diamond"),
                                hovertemplate="<b>R Peak:</b> %{y}<br><b>Time:</b> %{x:.3f}s<extra></extra>",
                            )
                        )

                # Detect and plot P peaks
                try:
                    p_peaks = wm.detect_p_peak()
                    if p_peaks is not None and len(p_peaks) > 0:
                        # Filter P peaks to only show those within the plot time range
                        plot_start_time = np.min(time_axis_plot)
                        plot_end_time = np.max(time_axis_plot)
                        plot_start_idx = int(plot_start_time * sampling_freq)
                        plot_end_idx = int(plot_end_time * sampling_freq)

                        # Filter P peaks to only those within the plot range
                        valid_peaks = p_peaks[
                            (p_peaks >= plot_start_idx) & (p_peaks < plot_end_idx)
                        ]

                        if len(valid_peaks) > 0:
                            # Convert peak indices to plot time coordinates
                            peak_times = valid_peaks / sampling_freq
                            peak_values = signal_data[valid_peaks]

                            fig.add_trace(
                                go.Scatter(
                                    x=peak_times,
                                    y=peak_values,
                                    mode="markers",
                                    name="P Peaks",
                                    marker=dict(color="blue", size=8, symbol="circle"),
                                    hovertemplate="<b>P Peak:</b> %{y}<br><b>Time:</b> %{x:.3f}s<extra></extra>",
                                )
                            )
                except Exception as e:
                    logger.warning(f"P peak detection failed: {e}")

                # Detect and plot T peaks
                try:
                    t_peaks = wm.detect_t_peak()
                    if t_peaks is not None and len(t_peaks) > 0:
                        # Filter T peaks to only show those within the plot time range
                        plot_start_time = np.min(time_axis_plot)
                        plot_end_time = np.max(time_axis_plot)
                        plot_start_idx = int(plot_start_time * sampling_freq)
                        plot_end_idx = int(plot_end_time * sampling_freq)

                        # Filter T peaks to only those within the plot range
                        valid_peaks = t_peaks[
                            (t_peaks >= plot_start_idx) & (t_peaks < plot_end_idx)
                        ]

                        if len(valid_peaks) > 0:
                            # Convert peak indices to plot time coordinates
                            peak_times = valid_peaks / sampling_freq
                            peak_values = signal_data[valid_peaks]

                            fig.add_trace(
                                go.Scatter(
                                    x=peak_times,
                                    y=peak_values,
                                    mode="markers",
                                    name="T Peaks",
                                    marker=dict(color="green", size=8, symbol="square"),
                                    hovertemplate="<b>T Peak:</b> %{y}<br><b>Time:</b> %{x:.3f}s<extra></extra>",
                                )
                            )
                except Exception as e:
                    logger.warning(f"T peak detection failed: {e}")

            else:
                # For other signal types, use basic peak detection
                logger.info(
                    f"Using basic peak detection for signal type: {signal_type}"
                )
                try:
                    from vitalDSP.physiological_features.peak_detection import (
                        detect_peaks,
                    )

                    peaks = detect_peaks(signal_data, sampling_freq)
                    if len(peaks) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=time_axis[peaks],
                                y=signal_data[peaks],
                                mode="markers",
                                name="Detected Peaks",
                                marker=dict(color="red", size=8, symbol="diamond"),
                                hovertemplate="<b>Peak:</b> %{y}<extra></extra>",
                            )
                        )
                except Exception as e:
                    logger.warning(f"Basic peak detection failed: {e}")

        except Exception as e:
            logger.warning(f"Critical points detection failed: {e}")

        fig.update_layout(
            title="Original Signal with Critical Points",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            height=400,
            showlegend=True,
            template="plotly_white",
            # Enable pan and zoom
            dragmode="pan",
            modebar=dict(
                orientation="v",
                bgcolor="rgba(255,255,255,0.8)",
                color="rgba(0,0,0,0.5)",
                activecolor="rgba(0,0,0,0.8)",
            ),
        )

        logger.info(
            f" Original signal plot created successfully with {len(fig.data)} traces"
        )
        logger.info("=" * 80)
        return fig
    except Exception as e:
        logger.error(f" Error creating original signal plot: {e}")
        logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        logger.info("=" * 80)
        return create_empty_figure()


def create_filtered_signal_plot(time_axis, filtered_data, sampling_freq, signal_type):
    """Create plot for filtered signal with critical points detection."""
    try:
        logger.info("Creating filtered signal plot:")
        logger.info(f"  Time axis shape: {time_axis.shape}")
        logger.info(f"  Filtered data shape: {filtered_data.shape}")
        logger.info(
            f"  Time axis range: {np.min(time_axis):.4f} to {np.max(time_axis):.4f}"
        )
        logger.info(
            f"  Filtered data range: {np.min(filtered_data):.4f} to {np.max(filtered_data):.4f}"
        )
        logger.info(
            f"  Time axis sample: {time_axis[:5] if len(time_axis) >= 5 else time_axis}"
        )
        logger.info(
            f"  Filtered data sample: {filtered_data[:5] if len(filtered_data) >= 5 else filtered_data}"
        )

        # PERFORMANCE OPTIMIZATION: Limit plot data to max 5 minutes and 10K points
        time_axis_plot, filtered_data_plot = limit_plot_data(
            time_axis,
            filtered_data,
            max_duration=300,  # 5 minutes max
            max_points=10000,  # 10K points max
        )

        logger.info(
            f"Plot data limited: {len(filtered_data)} → {len(filtered_data_plot)} points"
        )

        fig = go.Figure()

        # Add main filtered signal (using limited data)
        fig.add_trace(
            go.Scatter(
                x=time_axis_plot,
                y=filtered_data_plot,
                mode="lines",
                name="Filtered Signal",
                line=dict(color="red", width=2),
            )
        )

        # Add critical points detection using vitalDSP waveform module
        # NOTE: Use limited data for peak detection to match the plot
        try:
            from vitalDSP.physiological_features.waveform import WaveformMorphology

            # Create waveform morphology object for filtered signal (use FULL data for accurate peak detection)
            wm = WaveformMorphology(
                waveform=filtered_data,  # Use FULL filtered data for accurate peak detection
                fs=sampling_freq,  # Use actual sampling frequency
                signal_type=signal_type,  # Use signal type from UI
                simple_mode=True,
            )

            # Detect critical points based on signal type
            if signal_type == "PPG":
                # For PPG: systolic peaks, dicrotic notches, diastolic peaks
                if hasattr(wm, "systolic_peaks") and wm.systolic_peaks is not None:
                    # Filter peaks to only show those within the plot time range
                    plot_start_time = np.min(time_axis_plot)
                    plot_end_time = np.max(time_axis_plot)
                    plot_start_idx = int(plot_start_time * sampling_freq)
                    plot_end_idx = int(plot_end_time * sampling_freq)

                    # Filter peaks to only those within the plot range
                    valid_peaks = wm.systolic_peaks[
                        (wm.systolic_peaks >= plot_start_idx)
                        & (wm.systolic_peaks < plot_end_idx)
                    ]

                    if len(valid_peaks) > 0:
                        # Convert peak indices to plot time coordinates
                        peak_times = valid_peaks / sampling_freq
                        peak_values = filtered_data[valid_peaks]

                        fig.add_trace(
                            go.Scatter(
                                x=peak_times,
                                y=peak_values,
                                mode="markers",
                                name="Systolic Peaks (Filtered)",
                                marker=dict(color="darkred", size=10, symbol="diamond"),
                                hovertemplate="<b>Systolic Peak (Filtered):</b> %{y}<br><b>Time:</b> %{x:.3f}s<extra></extra>",
                            )
                        )

                # Detect and plot dicrotic notches
                try:
                    dicrotic_notches = wm.detect_dicrotic_notches()
                    if dicrotic_notches is not None and len(dicrotic_notches) > 0:
                        # Filter dicrotic notches to only show those within the plot time range
                        plot_start_time = np.min(time_axis_plot)
                        plot_end_time = np.max(time_axis_plot)
                        plot_start_idx = int(plot_start_time * sampling_freq)
                        plot_end_idx = int(plot_end_time * sampling_freq)

                        # Filter dicrotic notches to only those within the plot range
                        valid_notches = dicrotic_notches[
                            (dicrotic_notches >= plot_start_idx)
                            & (dicrotic_notches < plot_end_idx)
                        ]

                        if len(valid_notches) > 0:
                            # Convert notch indices to plot time coordinates
                            notch_times = valid_notches / sampling_freq
                            notch_values = filtered_data[valid_notches]

                            fig.add_trace(
                                go.Scatter(
                                    x=notch_times,
                                    y=notch_values,
                                    mode="markers",
                                    name="Dicrotic Notches (Filtered)",
                                    marker=dict(
                                        color="darkorange", size=8, symbol="circle"
                                    ),
                                    hovertemplate="<b>Dicrotic Notch (Filtered):</b> %{y}<br><b>Time:</b> %{x:.3f}s<extra></extra>",
                                )
                            )
                except Exception as e:
                    logger.warning(f"Dicrotic notch detection failed: {e}")

                # Detect and plot diastolic peaks
                try:
                    diastolic_peaks = wm.detect_diastolic_peak()
                    if diastolic_peaks is not None and len(diastolic_peaks) > 0:
                        # Filter diastolic peaks to only show those within the plot time range
                        plot_start_time = np.min(time_axis_plot)
                        plot_end_time = np.max(time_axis_plot)
                        plot_start_idx = int(plot_start_time * sampling_freq)
                        plot_end_idx = int(plot_end_time * sampling_freq)

                        # Filter diastolic peaks to only those within the plot range
                        valid_peaks = diastolic_peaks[
                            (diastolic_peaks >= plot_start_idx)
                            & (diastolic_peaks < plot_end_idx)
                        ]

                        if len(valid_peaks) > 0:
                            # Convert peak indices to plot time coordinates
                            peak_times = valid_peaks / sampling_freq
                            peak_values = filtered_data[valid_peaks]

                            fig.add_trace(
                                go.Scatter(
                                    x=peak_times,
                                    y=peak_values,
                                    mode="markers",
                                    name="Diastolic Peaks (Filtered)",
                                    marker=dict(
                                        color="darkgreen", size=8, symbol="square"
                                    ),
                                    hovertemplate="<b>Diastolic Peak (Filtered):</b> %{y}<br><b>Time:</b> %{x:.3f}s<extra></extra>",
                                )
                            )
                except Exception as e:
                    logger.warning(f"Diastolic peak detection failed: {e}")

            elif signal_type == "ECG":
                # For ECG: R peaks, P peaks, T peaks
                if hasattr(wm, "r_peaks") and wm.r_peaks is not None:
                    # Plot R peaks
                    fig.add_trace(
                        go.Scatter(
                            x=time_axis[wm.r_peaks],
                            y=filtered_data[wm.r_peaks],
                            mode="markers",
                            name="R Peaks (Filtered)",
                            marker=dict(color="darkred", size=10, symbol="diamond"),
                            hovertemplate="<b>R Peak (Filtered):</b> %{y}<extra></extra>",
                        )
                    )

                # Detect and plot P peaks
                try:
                    p_peaks = wm.detect_p_peak()
                    if p_peaks is not None and len(p_peaks) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=time_axis[p_peaks],
                                y=filtered_data[p_peaks],
                                mode="markers",
                                name="P Peaks (Filtered)",
                                marker=dict(color="darkblue", size=8, symbol="circle"),
                                hovertemplate="<b>P Peak (Filtered):</b> %{y}<extra></extra>",
                            )
                        )
                except Exception as e:
                    logger.warning(f"P peak detection failed: {e}")

                # Detect and plot T peaks
                try:
                    t_peaks = wm.detect_t_peak()
                    if t_peaks is not None and len(t_peaks) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=time_axis[t_peaks],
                                y=filtered_data[t_peaks],
                                mode="markers",
                                name="T Peaks (Filtered)",
                                marker=dict(color="darkgreen", size=8, symbol="square"),
                                hovertemplate="<b>T Peak (Filtered):</b> %{y}<extra></extra>",
                            )
                        )
                except Exception as e:
                    logger.warning(f"T peak detection failed: {e}")

            else:
                # For other signal types, use basic peak detection
                logger.info(
                    f"Using basic peak detection for signal type: {signal_type}"
                )
                try:
                    from vitalDSP.physiological_features.peak_detection import (
                        detect_peaks,
                    )

                    peaks = detect_peaks(filtered_data, sampling_freq)
                    if len(peaks) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=time_axis[peaks],
                                y=filtered_data[peaks],
                                mode="markers",
                                name="Detected Peaks (Filtered)",
                                marker=dict(color="darkred", size=8, symbol="diamond"),
                                hovertemplate="<b>Peak (Filtered):</b> %{y}<extra></extra>",
                            )
                        )
                except Exception as e:
                    logger.warning(f"Basic peak detection failed: {e}")

        except Exception as e:
            logger.warning(f"Critical points detection failed: {e}")

        fig.update_layout(
            title="Filtered Signal with Critical Points",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            height=400,
            showlegend=True,
            template="plotly_white",
            # Enable pan and zoom
            dragmode="pan",
            modebar=dict(
                orientation="v",
                bgcolor="rgba(255,255,255,0.8)",
                color="rgba(0,0,0,0.5)",
                activecolor="rgba(0,0,0,0.8)",
            ),
        )

        logger.info("Filtered signal plot with critical points created successfully")
        return fig
    except Exception as e:
        logger.error(f"Error creating filtered signal plot: {e}")
        return create_empty_figure()


def create_filter_comparison_plot(
    time_axis,
    original_signal,
    filtered_signal,
    sampling_freq,
    signal_type,
    peaks=None,
    notches=None,
):
    """Overlay the original signal and the user-filtered signal.

    Optional ``peaks`` (systolic peak indices) and ``notches``
    (dicrotic notch indices) are rendered as markers on the FILTERED
    trace so the user can see what survived the filter.  Pre-computed
    by the caller from a SINGLE shared ``WaveformMorphology`` pass so
    we don't repeat the heavy peak detection here.
    """
    import numpy as np

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=original_signal,
            mode="lines",
            name="Original",
            line=dict(color="rgba(60, 60, 60, 0.6)", width=1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=filtered_signal,
            mode="lines",
            name="Filtered",
            line=dict(color="#1f77b4", width=1.5),
        )
    )

    if peaks is not None:
        try:
            idx = np.asarray(peaks, dtype=int)
            idx = idx[(idx >= 0) & (idx < len(time_axis))]
            if idx.size:
                fig.add_trace(
                    go.Scatter(
                        x=np.asarray(time_axis)[idx],
                        y=np.asarray(filtered_signal)[idx],
                        mode="markers",
                        name="Systolic peaks",
                        marker=dict(color="#d62728", size=7, symbol="diamond"),
                    )
                )
        except Exception:
            pass

    if notches is not None:
        try:
            idx = np.asarray(notches, dtype=int)
            idx = idx[(idx >= 0) & (idx < len(time_axis))]
            if idx.size:
                fig.add_trace(
                    go.Scatter(
                        x=np.asarray(time_axis)[idx],
                        y=np.asarray(filtered_signal)[idx],
                        mode="markers",
                        name="Dicrotic notches",
                        marker=dict(color="#9467bd", size=6, symbol="x"),
                    )
                )
        except Exception:
            pass

    fig.update_layout(
        title="Original vs Filtered (same window)",
        xaxis_title="Time (s)",
        yaxis_title=f"{signal_type} amplitude",
        height=500,
        showlegend=True,
        margin=dict(l=50, r=20, t=50, b=40),
        hovermode="x unified",
    )
    return fig


def create_empty_figure():
    """Create an empty figure for error cases."""
    fig = go.Figure()
    fig.add_annotation(
        text="No data available",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
    )
    fig.update_layout(height=400)
    return fig


# Removed unused create_signal_plot function


def apply_traditional_filter(
    signal_data,
    sampling_freq,
    filter_family,
    filter_response,
    low_freq,
    high_freq,
    filter_order,
):
    """Apply traditional filtering using vitalDSP functions."""
    try:
        logger.info("=== TRADITIONAL FILTER DEBUG ===")
        logger.info(f"Input signal shape: {signal_data.shape}")
        logger.info(
            f"Input signal range: {np.min(signal_data):.4f} to {np.max(signal_data):.4f}"
        )
        logger.info(f"Filter family: {filter_family}")
        logger.info(f"Filter response: {filter_response}")
        logger.info(f"Low frequency: {low_freq}")
        logger.info(f"High frequency: {high_freq}")
        logger.info(f"Filter order: {filter_order}")
        logger.info(f"Sampling frequency: {sampling_freq}")

        # Normalise response aliases so callers can pass either ``low``/
        # ``high`` (legacy) or ``lowpass``/``highpass`` (UI radio).
        response_aliases = {
            "lowpass": "low",
            "highpass": "high",
            "low_pass": "low",
            "high_pass": "high",
        }
        filter_response = response_aliases.get(filter_response, filter_response)

        # Import vitalDSP traditional filtering functions
        from vitalDSP.filtering.signal_filtering import SignalFiltering

        # Create filter instance
        logger.info("SignalFiltering instance created successfully")

        # Apply filter based on type using vitalDSP.  IMPORTANT:
        # ``SignalFiltering`` filter methods take PHYSICAL Hz for
        # ``cutoff`` (and normalise using the supplied ``fs``).  The
        # wrapper used to pre-normalise by dividing by Nyquist, which
        # produced double-normalised cutoffs and blow-up coefficients
        # (~1e19 outputs - the "no trace shown" bug).  Note the library
        # exposes ``chebyshev`` (== type-1) and ``chebyshev2``; there
        # is NO ``chebyshev1`` attribute, so the dispatch must use the
        # right name or the whole function aborts with AttributeError.
        sf = SignalFiltering(signal_data)
        family_dispatch = {
            "butter": sf.butterworth,
            "cheby1": sf.chebyshev,
            "cheby2": sf.chebyshev2,
            "ellip": sf.elliptic,
        }
        family_fn = family_dispatch.get(filter_family, sf.butterworth)

        # ``SignalFiltering.butterworth`` carries ``adaptive=True`` by
        # default, which routes the call through
        # ``optimize_filtering_parameters`` and silently overrides the
        # user's cutoff / order based on the signal's dominant
        # frequency (clamped to 10 Hz for PPG, 40 Hz for ECG).  That
        # made the webapp's filter output diverge from raw-library
        # output for the same parameters.  Pass ``adaptive=False``
        # when the family accepts it so the user gets exactly what
        # they configured.  ``chebyshev``, ``chebyshev2``, ``elliptic``
        # don't have the kwarg, so the call site is conditional.
        adaptive_kwargs = {"adaptive": False} if filter_family == "butter" else {}

        # UI convention:
        #   Low Freq  = lower edge of the passband = high-pass cutoff
        #   High Freq = upper edge of the passband = low-pass cutoff
        # A low-pass filter keeps frequencies BELOW its cutoff, so the
        # cutoff is High Freq (the top of what passes through).  A
        # high-pass filter keeps frequencies ABOVE its cutoff, so the
        # cutoff is Low Freq.  Earlier the wiring was swapped — a
        # "low-pass at high_freq=40 Hz" was running a filter at 0.5 Hz
        # (the low_freq value), which silently destroyed the signal.
        if filter_response == "low":
            cutoff = high_freq if high_freq is not None else low_freq
            logger.info(f"Applying low-pass filter with cutoff: {cutoff} Hz")
            filtered_signal = family_fn(
                cutoff=cutoff,
                fs=sampling_freq,
                order=filter_order,
                btype="low",
                **adaptive_kwargs,
            )
            logger.info(f"Low-pass filter applied using vitalDSP {filter_family}")

        elif filter_response == "high":
            cutoff = low_freq if low_freq is not None else high_freq
            logger.info(f"Applying high-pass filter with cutoff: {cutoff} Hz")
            filtered_signal = family_fn(
                cutoff=cutoff,
                fs=sampling_freq,
                order=filter_order,
                btype="high",
                **adaptive_kwargs,
            )
            logger.info(f"High-pass filter applied using vitalDSP {filter_family}")

        elif filter_response == "bandpass":
            logger.info(
                f"Applying bandpass filter with range: {low_freq} - {high_freq} Hz"
            )
            # CRITICAL: ``SignalFiltering.bandpass`` expects PHYSICAL Hz
            # for ``lowcut`` and ``highcut`` and normalises internally
            # using the supplied ``fs``.  An earlier version of this
            # wrapper pre-normalised by dividing by Nyquist (and passed
            # the same ``fs``), which double-normalised the cutoffs to
            # near-zero, blew up the filter coefficients, and produced
            # signals on the order of 1e19 - the "no trace shown" bug.
            # NOTE: ``SignalFiltering.bandpass`` only recognises
            # ``"butter"``, ``"cheby"`` (type-1), and ``"elliptic"``;
            # other names fall through to a ValueError.  We map the
            # webapp's family identifiers to whichever value the
            # library actually accepts, and use butter as the safe
            # default.
            sf = SignalFiltering(signal_data)
            family_map = {
                "butter": "butter",
                "cheby1": "cheby",
                "cheby2": "cheby",   # library has no cheby-2 bandpass
                "ellip": "elliptic",
            }
            filter_type_lib = family_map.get(filter_family, "butter")
            filtered_signal = sf.bandpass(
                lowcut=low_freq,
                highcut=high_freq,
                fs=sampling_freq,
                order=filter_order,
                filter_type=filter_type_lib,
            )

            logger.info(f"Bandpass filter applied using vitalDSP {filter_family}")

        elif filter_response == "bandstop":
            logger.info(
                f"Applying bandstop filter with range: {low_freq} - {high_freq}"
            )
            # Apply bandstop filter using vitalDSP
            nyquist = sampling_freq / 2
            low_freq_norm = low_freq / nyquist
            high_freq_norm = high_freq / nyquist

            # Ensure cutoff frequencies are within valid range
            low_freq_norm = max(0.001, min(low_freq_norm, 0.999))
            high_freq_norm = max(0.001, min(high_freq_norm, 0.999))

            # Apply bandstop filter using vitalDSP
            sf = SignalFiltering(signal_data)

            if filter_family == "butter":
                filtered_signal = sf.bandstop(
                    lowcut=low_freq_norm,
                    highcut=high_freq_norm,
                    fs=sampling_freq,
                    order=filter_order,
                    filter_type="butter",
                )
            elif filter_family == "cheby1":
                filtered_signal = sf.bandstop(
                    lowcut=low_freq_norm,
                    highcut=high_freq_norm,
                    fs=sampling_freq,
                    order=filter_order,
                    filter_type="cheby1",
                )
            elif filter_family == "cheby2":
                filtered_signal = sf.bandstop(
                    lowcut=low_freq_norm,
                    highcut=high_freq_norm,
                    fs=sampling_freq,
                    order=filter_order,
                    filter_type="cheby2",
                )
            elif filter_family == "ellip":
                filtered_signal = sf.bandstop(
                    lowcut=low_freq_norm,
                    highcut=high_freq_norm,
                    fs=sampling_freq,
                    order=filter_order,
                    filter_type="ellip",
                )
            else:
                filtered_signal = sf.bandstop(
                    lowcut=low_freq_norm,
                    highcut=high_freq_norm,
                    fs=sampling_freq,
                    order=filter_order,
                    filter_type="butter",
                )

            logger.info(f"Bandstop filter applied using vitalDSP {filter_family}")

        elif filter_response == "default":
            # Default to low pass
            logger.info(f"Defaulting to low-pass filter with cutoff: {low_freq}")
            # Use vitalDSP for consistent filtering
            nyquist = sampling_freq / 2
            low_freq_norm = low_freq / nyquist

            # Ensure cutoff frequency is within valid range
            low_freq_norm = max(0.001, min(low_freq_norm, 0.999))

            # Apply default lowpass filter using vitalDSP.  ``adaptive=False``
            # keeps the user-supplied cutoff / order intact (see comment in
            # the low/high branches above).
            sf = SignalFiltering(signal_data)
            filtered_signal = sf.butterworth(
                cutoff=low_freq_norm,
                fs=sampling_freq,
                order=filter_order,
                btype="low",
                adaptive=False,
            )
            logger.info("Default low-pass filter applied using vitalDSP")

        # Safety net: if none of the response branches above set
        # ``filtered_signal`` (e.g. unknown response value), return the
        # input unchanged rather than raising UnboundLocalError.
        if "filtered_signal" not in locals():
            logger.warning(
                "apply_traditional_filter: unknown response %r; "
                "returning input unchanged.",
                filter_response,
            )
            filtered_signal = signal_data

        logger.info(
            f"Traditional filter applied successfully: {filter_family} {filter_response}"
        )
        logger.info(f"Output signal shape: {filtered_signal.shape}")
        logger.info(
            f"Output signal range: {np.min(filtered_signal):.4f} to {np.max(filtered_signal):.4f}"
        )
        return filtered_signal

    except Exception as e:
        logger.error(f"Error applying traditional filter: {e}")
        logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
        # Fallback to basic filtering
        logger.info("Falling back to basic filtering implementation")
        try:
            # Normalize cutoff frequencies
            nyquist = sampling_freq / 2
            low_freq_norm = low_freq / nyquist
            high_freq_norm = high_freq / nyquist

            # Ensure cutoff frequencies are within valid range
            low_freq_norm = max(0.001, min(low_freq_norm, 0.999))
            high_freq_norm = max(0.001, min(high_freq_norm, 0.999))

            # Determine filter type based on response
            if filter_response == "bandpass":
                btype = "band"
                cutoff = [low_freq_norm, high_freq_norm]
            elif filter_response == "bandstop":
                btype = "bandstop"
                cutoff = [low_freq_norm, high_freq_norm]
            elif filter_response == "lowpass":
                btype = "low"
                cutoff = high_freq_norm
            elif filter_response == "highpass":
                btype = "high"
                cutoff = low_freq_norm
            else:
                btype = "low"
                cutoff = high_freq_norm

            # Apply filter using vitalDSP
            sf = SignalFiltering(signal_data)
            filtered_signal = sf.butterworth(
                cutoff=cutoff, fs=sampling_freq, order=filter_order, btype=btype
            )

            logger.info("vitalDSP filter applied successfully")
            return filtered_signal

        except Exception as fallback_error:
            logger.error(f"Scipy fallback also failed: {fallback_error}")
            return signal_data


def apply_additional_traditional_filters(
    signal_data,
    savgol_window,
    savgol_polyorder,
    moving_avg_window,
    moving_avg_iterations,
    gaussian_sigma,
    gaussian_iterations,
):
    """Apply additional traditional filters like Savitzky-Golay, moving average, and Gaussian."""
    try:
        logger.info("Applying additional traditional filters")

        from vitalDSP.filtering.signal_filtering import SignalFiltering

        signal_filter = SignalFiltering(signal_data)
        filtered_signal = signal_data.copy()

        # Apply Savitzky-Golay filter if parameters are provided
        if savgol_window and savgol_polyorder:
            try:
                filtered_signal = signal_filter.savgol_filter(
                    filtered_signal,
                    window_length=savgol_window,
                    polyorder=savgol_polyorder,
                )
                logger.info(
                    f"Savitzky-Golay filter applied: window={savgol_window}, polyorder={savgol_polyorder}"
                )
            except Exception as e:
                logger.warning(f"Savitzky-Golay filter failed: {e}")

        # Apply moving average filter if parameters are provided
        if moving_avg_window:
            try:
                filtered_signal = signal_filter.moving_average(
                    window_size=moving_avg_window,
                    iterations=moving_avg_iterations or 1,
                    method="edge",
                )
                logger.info(
                    f"Moving average filter applied: window={moving_avg_window}, iterations={moving_avg_iterations}"
                )
            except Exception as e:
                logger.warning(f"Moving average filter failed: {e}")

        # Apply Gaussian filter if parameters are provided
        if gaussian_sigma:
            try:
                filtered_signal = signal_filter.gaussian(
                    sigma=gaussian_sigma, iterations=gaussian_iterations or 1
                )
                logger.info(
                    f"Gaussian filter applied: sigma={gaussian_sigma}, iterations={gaussian_iterations}"
                )
            except Exception as e:
                logger.warning(f"Gaussian filter failed: {e}")

        return filtered_signal

    except Exception as e:
        logger.error(f"Error applying additional traditional filters: {e}")
        return signal_data


def apply_enhanced_artifact_removal(
    signal_data,
    sampling_freq,
    artifact_type,
    artifact_strength,
    wavelet_type,
    wavelet_level,
    threshold_type,
    threshold_value,
    powerline_freq,
    notch_q_factor,
    pca_components,
    ica_components,
):
    """Apply enhanced artifact removal using vitalDSP functions with configurable parameters."""
    try:
        logger.info(f"Applying enhanced artifact removal: {artifact_type}")

        from vitalDSP.filtering.artifact_removal import ArtifactRemoval

        # Set default values
        artifact_type = artifact_type or "baseline"
        artifact_strength = artifact_strength or 0.5

        # Create artifact removal instance
        artifact_remover = ArtifactRemoval(signal_data)

        # Apply artifact removal based on type with enhanced parameters
        if artifact_type == "baseline":
            filtered_signal = artifact_remover.baseline_correction(
                cutoff=artifact_strength, fs=sampling_freq
            )
        elif artifact_type == "spike":
            filtered_signal = artifact_remover.median_filter_removal()
        elif artifact_type == "noise":
            # Enhanced wavelet denoising with configurable parameters
            filtered_signal = artifact_remover.wavelet_denoising(
                wavelet_type=wavelet_type or "db4",
                level=wavelet_level or 3,
                threshold_type=threshold_type or "soft",
                threshold_value=threshold_value or 0.1,
            )
        elif artifact_type == "powerline":
            filtered_signal = artifact_remover.notch_filter(
                freq=powerline_freq or 50, fs=sampling_freq, Q=notch_q_factor or 30
            )
        elif artifact_type == "pca":
            filtered_signal = artifact_remover.pca_artifact_removal(
                num_components=pca_components or 1
            )
        elif artifact_type == "ica":
            filtered_signal = artifact_remover.ica_artifact_removal(
                num_components=ica_components or 2
            )
        else:
            # Default to baseline correction
            filtered_signal = artifact_remover.baseline_correction(
                cutoff=artifact_strength, fs=sampling_freq
            )

        logger.info(f"Enhanced artifact removal applied successfully: {artifact_type}")
        return filtered_signal

    except Exception as e:
        logger.error(f"Error applying enhanced artifact removal: {e}")
        # Return original signal if artifact removal fails
        return signal_data


def apply_advanced_filter(
    signal_data,
    advanced_method,
    # Kalman parameters
    kalman_r=None,
    kalman_q=None,
    # Optimization parameters
    optimization_loss_type=None,
    optimization_initial_guess=None,
    optimization_learning_rate=None,
    optimization_iterations=None,
    # Gradient descent parameters
    gradient_learning_rate=None,
    gradient_iterations=None,
    # Convolution parameters
    convolution_kernel_type=None,
    convolution_kernel_size=None,
    # Attention parameters
    attention_type=None,
    attention_size=None,
    attention_sigma=None,
    attention_ascending=None,
    attention_base=None,
    # Adaptive parameters
    adaptive_mu=None,
    adaptive_order=None,
):
    """Apply advanced filtering using vitalDSP functions with method-specific parameters."""
    try:
        logger.info(f"Applying advanced filter: {advanced_method}")

        # Import vitalDSP advanced filtering functions
        from vitalDSP.filtering.advanced_signal_filtering import AdvancedSignalFiltering

        # Set default values for method
        advanced_method = advanced_method or "kalman"

        # Create advanced filter instance
        advanced_filter = AdvancedSignalFiltering(signal_data)

        # Apply filter based on method with specific parameters
        if advanced_method == "kalman":
            R = kalman_r if kalman_r is not None else 1.0
            Q = kalman_q if kalman_q is not None else 1.0
            logger.info(f"Kalman filter parameters: R={R}, Q={Q}")
            filtered_signal = advanced_filter.kalman_filter(R=R, Q=Q)

        elif advanced_method == "optimization":
            loss_type = optimization_loss_type or "mse"
            initial_guess = (
                optimization_initial_guess
                if optimization_initial_guess is not None
                else 0.0
            )
            lr = (
                optimization_learning_rate
                if optimization_learning_rate is not None
                else 0.01
            )
            iters = (
                optimization_iterations if optimization_iterations is not None else 100
            )
            logger.info(
                f"Optimization parameters: loss={loss_type}, lr={lr}, iterations={iters}"
            )
            filtered_signal = advanced_filter.optimization_based_filtering(
                target=signal_data,
                loss_type=loss_type,
                initial_guess=initial_guess,
                learning_rate=lr,
                iterations=iters,
            )

        elif advanced_method == "gradient_descent":
            lr = gradient_learning_rate if gradient_learning_rate is not None else 0.01
            iters = gradient_iterations if gradient_iterations is not None else 100
            logger.info(f"Gradient descent parameters: lr={lr}, iterations={iters}")
            filtered_signal = advanced_filter.gradient_descent_filter(
                target=signal_data, learning_rate=lr, iterations=iters
            )

        elif advanced_method == "convolution":
            kernel_type = convolution_kernel_type or "smoothing"
            kernel_size = (
                convolution_kernel_size if convolution_kernel_size is not None else 3
            )
            logger.info(
                f"Convolution parameters: kernel_type={kernel_type}, kernel_size={kernel_size}"
            )
            filtered_signal = advanced_filter.convolution_based_filter(
                kernel_type=kernel_type, kernel_size=kernel_size
            )

        elif advanced_method == "attention":
            att_type = attention_type or "uniform"
            att_size = attention_size if attention_size is not None else 5
            kwargs = {}
            if att_type == "gaussian" and attention_sigma is not None:
                kwargs["sigma"] = attention_sigma
            if (
                att_type in ["linear", "exponential"]
                and attention_ascending is not None
            ):
                kwargs["ascending"] = attention_ascending == "true"
            if att_type == "exponential" and attention_base is not None:
                kwargs["base"] = attention_base
            logger.info(
                f"Attention parameters: type={att_type}, size={att_size}, kwargs={kwargs}"
            )
            filtered_signal = advanced_filter.attention_based_filter(
                attention_type=att_type, size=att_size, **kwargs
            )

        elif advanced_method == "adaptive":
            mu = adaptive_mu if adaptive_mu is not None else 0.01
            order = adaptive_order if adaptive_order is not None else 4
            logger.info(f"Adaptive filter parameters: mu={mu}, order={order}")
            # Adaptive filter needs a desired signal - use smoothed version as target
            from scipy.signal import savgol_filter

            desired = savgol_filter(
                signal_data,
                window_length=min(51, len(signal_data) // 2 * 2 + 1),
                polyorder=3,
            )
            filtered_signal = advanced_filter.adaptive_filtering(
                desired_signal=desired, mu=mu, filter_order=order
            )

        else:
            # Default to Kalman filter
            logger.warning(
                f"Unknown advanced method '{advanced_method}', defaulting to Kalman"
            )
            filtered_signal = advanced_filter.kalman_filter(R=1.0, Q=1.0)

        logger.info(f"Advanced filter applied successfully: {advanced_method}")
        return filtered_signal

    except Exception as e:
        logger.error(f"Error applying advanced filter: {e}", exc_info=True)
        # Return original signal if advanced filtering fails
        return signal_data


def apply_neural_filter(signal_data, neural_type, neural_complexity):
    """Apply neural network filtering using vitalDSP functions."""
    try:
        logger.info(f"Applying neural filter: {neural_type}")

        # Import vitalDSP neural filtering functions
        from vitalDSP.advanced_computation.neural_network_filtering import (
            NeuralNetworkFiltering,
        )

        # Set default values
        neural_type = neural_type or "feedforward"
        neural_complexity = neural_complexity or "medium"

        # Set complexity-based parameters
        if neural_complexity == "low":
            hidden_layers = [32]
            epochs = 50
        elif neural_complexity == "high":
            hidden_layers = [128, 64, 32]
            epochs = 200
        else:  # medium
            hidden_layers = [64, 32]
            epochs = 100

        # Create neural filter instance
        neural_filter = NeuralNetworkFiltering(
            signal_data,
            network_type=neural_type,
            hidden_layers=hidden_layers,
            epochs=epochs,
        )

        # Train the neural network first
        neural_filter.train()

        # Apply the trained filter
        filtered_signal = neural_filter.apply_filter()

        logger.info(f"Neural filter applied successfully: {neural_type}")
        return filtered_signal

    except Exception as e:
        logger.error(f"Error applying neural filter: {e}")
        # Return original signal if neural filtering fails
        return signal_data


def apply_ensemble_filter(signal_data, ensemble_method, ensemble_n_filters):
    """Apply ensemble filtering using vitalDSP functions."""
    try:
        logger.info(f"Applying ensemble filter: {ensemble_method}")

        # Import vitalDSP ensemble filtering functions
        from vitalDSP.filtering.advanced_signal_filtering import AdvancedSignalFiltering

        # Set default values
        ensemble_method = ensemble_method or "mean"
        ensemble_n_filters = ensemble_n_filters or 3

        # Create advanced filter instance
        advanced_filter = AdvancedSignalFiltering(signal_data)

        # Create multiple filters for ensemble
        filters = []
        for i in range(ensemble_n_filters):
            # Create different filter configurations
            if i == 0:
                # Kalman filter
                filtered = advanced_filter.kalman_filter(R=0.1, Q=1)
            elif i == 1:
                # Convolution filter
                filtered = advanced_filter.convolution_based_filter(
                    kernel_type="smoothing", kernel_size=3
                )
            else:
                # Optimization filter
                filtered = advanced_filter.optimization_based_filtering(
                    target=signal_data,
                    loss_type="mse",
                    learning_rate=0.01,
                    iterations=50,
                )
            filters.append(filtered)

        # Apply ensemble method
        if ensemble_method == "mean":
            filtered_signal = np.mean(filters, axis=0)
        elif ensemble_method == "median":
            filtered_signal = np.median(filters, axis=0)
        elif ensemble_method == "weighted":
            # Weight by filter performance (inverse of variance)
            weights = 1 / (np.var(filters, axis=0) + 1e-10)
            weights = weights / np.sum(weights)
            filtered_signal = np.average(filters, axis=0, weights=weights)
        else:
            # Default to mean
            filtered_signal = np.mean(filters, axis=0)

        logger.info(
            f"Ensemble filter applied successfully: {ensemble_method} with {ensemble_n_filters} filters"
        )
        return filtered_signal

    except Exception as e:
        logger.error(f"Error applying ensemble filter: {e}")
        # Return original signal if ensemble filtering fails
        return signal_data


def apply_enhanced_ensemble_filter(
    signal_data,
    ensemble_method,
    ensemble_n_filters,
    ensemble_learning_rate,
    ensemble_iterations,
    adaptive_filter_order,
    adaptive_step_size,
    forgetting_factor,
    regularization_param,
):
    """Apply enhanced ensemble filtering with real-time capabilities."""
    try:
        logger.info(f"Applying enhanced ensemble filter: {ensemble_method}")

        from vitalDSP.filtering.advanced_signal_filtering import AdvancedSignalFiltering

        # Set default values
        ensemble_method = ensemble_method or "mean"
        ensemble_n_filters = ensemble_n_filters or 3

        # Create advanced filter instance
        advanced_filter = AdvancedSignalFiltering(signal_data)

        # Create multiple filters for ensemble
        filters = []
        for i in range(ensemble_n_filters):
            # Create different filter configurations
            if i == 0:
                # Kalman filter
                filtered = advanced_filter.kalman_filter(R=0.1, Q=1)
            elif i == 1:
                # Convolution filter
                filtered = advanced_filter.convolution_based_filter(
                    kernel_type="smoothing", kernel_size=3
                )
            elif i == 2:
                # Optimization filter
                filtered = advanced_filter.optimization_based_filtering(
                    target=signal_data,
                    loss_type="mse",
                    learning_rate=ensemble_learning_rate or 0.01,
                    iterations=ensemble_iterations or 50,
                )
            elif i == 3:
                # Adaptive filtering
                filtered = advanced_filter.adaptive_filtering(
                    desired_signal=signal_data,
                    mu=adaptive_step_size or 0.5,
                    filter_order=adaptive_filter_order or 4,
                )
            else:
                # Additional filter types
                filtered = advanced_filter.gradient_descent_filter(
                    target=signal_data,
                    learning_rate=ensemble_learning_rate or 0.01,
                    iterations=ensemble_iterations or 50,
                )
            filters.append(filtered)

        # Apply enhanced ensemble method
        if ensemble_method == "mean":
            filtered_signal = np.mean(filters, axis=0)
        elif ensemble_method == "median":
            filtered_signal = np.median(filters, axis=0)
        elif ensemble_method == "weighted":
            # Weight by filter performance (inverse of variance)
            weights = 1 / (np.var(filters, axis=0) + 1e-10)
            weights = weights / np.sum(weights)
            filtered_signal = np.average(filters, axis=0, weights=weights)
        elif ensemble_method == "bagging":
            # Bootstrap aggregating
            n_samples = len(signal_data)
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            filtered_signal = np.mean(
                [filters[i % len(filters)][indices] for i in range(len(filters))],
                axis=0,
            )
        elif ensemble_method == "boosting":
            # Adaptive boosting
            weights = np.ones(len(filters))
            filtered_signal = np.zeros_like(signal_data)
            for i, (filter_output, weight) in enumerate(zip(filters, weights)):
                filtered_signal += weight * filter_output
                # Update weights based on performance
                error = np.mean((signal_data - filter_output) ** 2)
                weights[i] = 0.5 * np.log((1 - error) / (error + 1e-10))
        elif ensemble_method == "stacking":
            # Stacking with meta-learner
            # Use the first filter as meta-learner
            filtered_signal = filters[0]
        else:
            # Default to mean
            filtered_signal = np.mean(filters, axis=0)

        logger.info(
            f"Enhanced ensemble filter applied successfully: {ensemble_method} with {ensemble_n_filters} filters"
        )
        return filtered_signal

    except Exception as e:
        logger.error(f"Error applying enhanced ensemble filter: {e}")
        # Return original signal if ensemble filtering fails
        return signal_data


def apply_multi_modal_filtering(
    signal_data, reference_signal, fusion_method, update_rate, performance_window
):
    """Apply multi-modal filtering using vitalDSP functions."""
    try:
        logger.info(f"Applying multi-modal filtering: {fusion_method}")

        # Import multi-modal fusion functions
        from vitalDSP.respiratory_analysis.fusion.multimodal_analysis import (
            MultiModalAnalysis,
        )

        # Create multi-modal analysis instance
        multimodal_analyzer = MultiModalAnalysis(signal_data)

        # Apply fusion based on method
        if fusion_method == "weighted":
            filtered_signal = multimodal_analyzer.weighted_fusion(
                reference_signal=(
                    reference_signal if reference_signal != "none" else None
                )
            )
        elif fusion_method == "kalman":
            filtered_signal = multimodal_analyzer.kalman_fusion(
                reference_signal=(
                    reference_signal if reference_signal != "none" else None
                )
            )
        elif fusion_method == "bayesian":
            filtered_signal = multimodal_analyzer.bayesian_fusion(
                reference_signal=(
                    reference_signal if reference_signal != "none" else None
                )
            )
        elif fusion_method == "deep_learning":
            filtered_signal = multimodal_analyzer.deep_learning_fusion(
                reference_signal=(
                    reference_signal if reference_signal != "none" else None
                )
            )
        else:
            # Default to weighted fusion
            filtered_signal = multimodal_analyzer.weighted_fusion(
                reference_signal=(
                    reference_signal if reference_signal != "none" else None
                )
            )

        logger.info(f"Multi-modal filtering applied successfully: {fusion_method}")
        return filtered_signal

    except Exception as e:
        logger.error(f"Error applying multi-modal filtering: {e}")
        # Return original signal if multi-modal filtering fails
        return signal_data


def generate_filter_quality_metrics(
    original_signal, filtered_signal, sampling_freq, quality_options, signal_type="ECG"
):
    """Render the headline filter-quality metrics as a compact table.

    Uses ``vitalDSP.signal_quality_assessment.FilteringQualityAssessment.assess_quality()``
    as the single source of truth.  The previous version called the same
    library helper internally but then layered ~1800 lines of per-domain
    breakdowns (statistical, temporal, morphological, frequency, ...)
    that duplicated ``calculate_*`` helpers also defined here.  Most of
    that breakdown was unread by users and made the page heavy.  The
    table below shows the five metrics that actually drive the overall
    quality verdict, plus the recommendation string.
    """
    try:
        from vitalDSP.signal_quality_assessment.filtering_quality_assessment import (
            FilteringQualityAssessment,
        )

        fqa = FilteringQualityAssessment(
            original_signal=original_signal,
            filtered_signal=filtered_signal,
            fs=sampling_freq,
            signal_type=signal_type,
        )
        result = fqa.assess_quality()
        metrics = result.get("metrics", {})
        overall = result.get("overall_quality", "n/a")
        recommendation = result.get("recommendation", "")

        def _row(label: str, key: str, value_key: str = "score") -> html.Tr:
            entry = metrics.get(key, {}) or {}
            raw = entry.get(value_key)
            try:
                shown = f"{float(raw):.3f}" if raw is not None else "-"
            except (TypeError, ValueError):
                shown = str(raw)
            verdict = entry.get("verdict") or entry.get("status") or ""
            return html.Tr(
                [
                    html.Td(label),
                    html.Td(shown, className="text-end"),
                    html.Td(html.Small(verdict, className="text-muted")),
                ]
            )

        table = dbc.Table(
            [
                html.Thead(html.Tr([html.Th("Metric"), html.Th("Value"), html.Th("")])),
                html.Tbody(
                    [
                        _row("SNR (dB)", "snr_db", value_key="value"),
                        _row("Shape similarity", "shape_similarity"),
                        _row("Smoothness improvement", "smoothness_improvement"),
                        _row("Noise reduction", "noise_reduction"),
                        _row("Peak preservation", "peak_preservation"),
                    ]
                ),
            ],
            bordered=False,
            striped=True,
            size="sm",
            className="mb-2",
        )

        return html.Div(
            [
                dbc.Alert(
                    [
                        html.Strong(f"Overall quality: {overall}"),
                        html.Br() if recommendation else None,
                        html.Small(recommendation, className="text-muted")
                        if recommendation
                        else None,
                    ],
                    color="info",
                    className="py-2 mb-2",
                ),
                table,
            ]
        )

    except Exception as exc:
        logger.exception("generate_filter_quality_metrics failed: %s", exc)
        return html.Div(
            [
                html.H6("Filter quality metrics"),
                html.P(f"Could not compute metrics: {exc}", className="text-danger small"),
            ]
        )


def create_filter_quality_plots(
    original_signal, filtered_signal, sampling_freq, quality_options, signal_type="ECG"
):
    """Two focused panels: amplitude overlay + frequency response.

    Replaces the old 6-panel mega-plot (which mixed signal comparison,
    frequency response, statistical distribution, temporal features,
    error analysis, performance metrics, plus PPG-hardcoded critical-
    point overlays).  Most of that information is better presented in
    the text Quality Metrics panel below; here we keep only the two
    plots a filter user actually reads.
    """
    try:
        if len(original_signal) == 0 or len(filtered_signal) == 0:
            return create_empty_figure()

        n = min(len(original_signal), len(filtered_signal))
        original_signal = np.asarray(original_signal[:n], dtype=float)
        filtered_signal = np.asarray(filtered_signal[:n], dtype=float)
        time_axis = np.arange(n) / max(sampling_freq, 1e-9)

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Amplitude overlay", "Frequency response (FFT magnitude)"),
            horizontal_spacing=0.12,
        )

        # Panel 1: amplitude overlay (same x-range)
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=original_signal,
                mode="lines",
                name="Original",
                line=dict(color="rgba(60, 60, 60, 0.6)", width=1),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=filtered_signal,
                mode="lines",
                name="Filtered",
                line=dict(color="#1f77b4", width=1.5),
            ),
            row=1,
            col=1,
        )

        # Panel 2: FFT magnitude (positive frequencies up to Nyquist)
        # Pad to next power of 2 for a slightly cleaner spectrum.
        nfft = 1 << (n - 1).bit_length() if n > 1 else 1
        freqs = np.fft.rfftfreq(nfft, d=1.0 / max(sampling_freq, 1e-9))
        mag_orig = np.abs(np.fft.rfft(original_signal, n=nfft))
        mag_filt = np.abs(np.fft.rfft(filtered_signal, n=nfft))
        fig.add_trace(
            go.Scatter(
                x=freqs,
                y=mag_orig,
                mode="lines",
                name="Original (spectrum)",
                line=dict(color="rgba(60, 60, 60, 0.6)", width=1),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=freqs,
                y=mag_filt,
                mode="lines",
                name="Filtered (spectrum)",
                line=dict(color="#1f77b4", width=1.5),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_yaxes(title_text=f"{signal_type} amplitude", row=1, col=1)
        fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)
        fig.update_yaxes(title_text="Magnitude", type="log", row=1, col=2)

        fig.update_layout(
            height=350,
            showlegend=True,
            margin=dict(l=50, r=20, t=50, b=40),
            hovermode="x",
        )
        return fig

    except Exception as exc:
        # Keep this defensive - quality plots failing should never crash
        # the filtering callback.
        logger = __import__("logging").getLogger(__name__)
        logger.exception("create_filter_quality_plots failed: %s", exc)
        return create_empty_figure()


def calculate_mse(original_signal, filtered_signal):
    """Calculate Mean Square Error between original and filtered signals."""
    try:
        return np.mean((original_signal - filtered_signal) ** 2)
    except Exception as e:
        logger.error(f"Error calculating MSE: {e}")
        return 0

