"""
Export callbacks for vitalDSP webapp pages.
"""

import logging
from dash import Input, Output, State
from dash.exceptions import PreventUpdate

try:
    from vitalDSP_webapp.utils.export_utils import (
        export_filtered_signal_csv,
        export_filtered_signal_json,
        export_respiratory_analysis_csv,
        export_respiratory_analysis_json,
        export_time_domain_analysis_csv,
        export_time_domain_analysis_json,
        export_frequency_domain_analysis_csv,
        export_frequency_domain_analysis_json,
    )
except ImportError:
    def export_filtered_signal_csv(*args, **kwargs): return "CSV Export not available"
    def export_filtered_signal_json(*args, **kwargs): return "{}"
    def export_respiratory_analysis_csv(*args, **kwargs): return "CSV Export not available"
    def export_respiratory_analysis_json(*args, **kwargs): return "{}"
    def export_time_domain_analysis_csv(*args, **kwargs): return "CSV Export not available"
    def export_time_domain_analysis_json(*args, **kwargs): return "{}"
    def export_frequency_domain_analysis_csv(*args, **kwargs): return "CSV Export not available"
    def export_frequency_domain_analysis_json(*args, **kwargs): return "{}"

logger = logging.getLogger(__name__)


def register_filtering_export_callbacks(app):
    """No-op stub — filtering export buttons were removed."""
    return None


def register_time_domain_export_callbacks(app):
    @app.callback(
        Output("download-time-domain-csv", "data"),
        Input("btn-export-time-domain-csv", "n_clicks"),
        State("store-time-domain-features", "data"),
        prevent_initial_call=True,
    )
    def export_time_domain_csv(n_clicks, features_data):
        if not n_clicks or not features_data:
            raise PreventUpdate
        try:
            return {"content": export_time_domain_analysis_csv(features_data), "filename": "time_domain_features.csv"}
        except Exception as e:
            logger.error(f"Error exporting time domain CSV: {e}")
            raise PreventUpdate

    @app.callback(
        Output("download-time-domain-json", "data"),
        Input("btn-export-time-domain-json", "n_clicks"),
        State("store-time-domain-features", "data"),
        prevent_initial_call=True,
    )
    def export_time_domain_json(n_clicks, features_data):
        if not n_clicks or not features_data:
            raise PreventUpdate
        try:
            return {"content": export_time_domain_analysis_json(features_data), "filename": "time_domain_features.json"}
        except Exception as e:
            logger.error(f"Error exporting time domain JSON: {e}")
            raise PreventUpdate


def register_frequency_domain_export_callbacks(app):
    @app.callback(
        Output("download-frequency-domain-csv", "data"),
        Input("btn-export-frequency-domain-csv", "n_clicks"),
        State("store-frequency-domain-features", "data"),
        prevent_initial_call=True,
    )
    def export_frequency_domain_csv(n_clicks, features_data):
        if not n_clicks or not features_data:
            raise PreventUpdate
        try:
            return {"content": export_frequency_domain_analysis_csv(features_data), "filename": "frequency_domain_features.csv"}
        except Exception as e:
            logger.error(f"Error exporting frequency domain CSV: {e}")
            raise PreventUpdate

    @app.callback(
        Output("download-frequency-domain-json", "data"),
        Input("btn-export-frequency-domain-json", "n_clicks"),
        State("store-frequency-domain-features", "data"),
        prevent_initial_call=True,
    )
    def export_frequency_domain_json(n_clicks, features_data):
        if not n_clicks or not features_data:
            raise PreventUpdate
        try:
            return {"content": export_frequency_domain_analysis_json(features_data), "filename": "frequency_domain_features.json"}
        except Exception as e:
            logger.error(f"Error exporting frequency domain JSON: {e}")
            raise PreventUpdate


def register_respiratory_export_callbacks(app):
    @app.callback(
        Output("download-respiratory-csv", "data"),
        Input("btn-export-respiratory-csv", "n_clicks"),
        State("store-respiratory-results", "data"),
        prevent_initial_call=True,
    )
    def export_respiratory_csv(n_clicks, respiratory_data):
        if not n_clicks or not respiratory_data:
            raise PreventUpdate
        try:
            return {"content": export_respiratory_analysis_csv(respiratory_data), "filename": "respiratory_analysis.csv"}
        except Exception as e:
            logger.error(f"Error exporting respiratory CSV: {e}")
            raise PreventUpdate

    @app.callback(
        Output("download-respiratory-json", "data"),
        Input("btn-export-respiratory-json", "n_clicks"),
        State("store-respiratory-results", "data"),
        prevent_initial_call=True,
    )
    def export_respiratory_json(n_clicks, respiratory_data):
        if not n_clicks or not respiratory_data:
            raise PreventUpdate
        try:
            return {"content": export_respiratory_analysis_json(respiratory_data), "filename": "respiratory_analysis.json"}
        except Exception as e:
            logger.error(f"Error exporting respiratory JSON: {e}")
            raise PreventUpdate


def register_all_export_callbacks(app):
    logger.info("Registering all export callbacks...")
    try:
        register_filtering_export_callbacks(app)
        logger.info(" Filtering export callbacks registered")
    except Exception as e:
        logger.warning(f"Could not register filtering export callbacks: {e}")
    try:
        register_time_domain_export_callbacks(app)
        logger.info(" Time domain export callbacks registered")
    except Exception as e:
        logger.warning(f"Could not register time domain export callbacks: {e}")
    try:
        register_frequency_domain_export_callbacks(app)
        logger.info(" Frequency domain export callbacks registered")
    except Exception as e:
        logger.warning(f"Could not register frequency domain export callbacks: {e}")
    try:
        register_respiratory_export_callbacks(app)
        logger.info(" Respiratory export callbacks registered")
    except Exception as e:
        logger.warning(f"Could not register respiratory export callbacks: {e}")
    logger.info("All export callbacks registered successfully")
