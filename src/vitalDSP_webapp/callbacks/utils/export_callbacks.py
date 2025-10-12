"""
Export callbacks for all vitalDSP webapp pages.

This module provides export functionality for:
- Filtering page: filtered signal with timestamps
- Feature extraction pages: extracted features
- Quality assessment: quality metrics
- Respiratory analysis: RR estimates
- Transform pages: transform results
"""

import logging
from dash import Input, Output, State, callback_context, no_update
from dash.exceptions import PreventUpdate
import numpy as np

try:
    from vitalDSP_webapp.utils.export_utils import (
        export_filtered_signal_csv,
        export_filtered_signal_json,
        export_features_csv,
        export_features_json,
        export_quality_metrics_csv,
        export_quality_metrics_json,
        export_respiratory_analysis_csv,
        export_respiratory_analysis_json,
        export_transform_results_csv,
        export_transform_results_json,
        export_time_domain_analysis_csv,
        export_time_domain_analysis_json,
        export_frequency_domain_analysis_csv,
        export_frequency_domain_analysis_json,
    )
except ImportError:
    # Fallback for testing
    def export_filtered_signal_csv(*args, **kwargs):
        return "CSV Export not available"

    def export_filtered_signal_json(*args, **kwargs):
        return "{}"

    # Add other fallbacks...

logger = logging.getLogger(__name__)


def register_filtering_export_callbacks(app):
    """Register export callbacks for filtering page."""

    @app.callback(
        Output("download-filtered-csv", "data"),
        Input("btn-export-filtered-csv", "n_clicks"),
        [
            State("store-filtered-signal", "data"),
            State("sampling-freq", "value"),
        ],
        prevent_initial_call=True,
    )
    def export_filtered_csv(n_clicks, filtered_data, sampling_freq):
        """Export filtered signal to CSV."""
        if not n_clicks or not filtered_data:
            raise PreventUpdate

        try:
            signal = np.array(filtered_data.get('signal', []))
            time = np.array(filtered_data.get('time', []))

            if len(signal) == 0:
                raise PreventUpdate

            # If no time array, generate one
            if len(time) == 0:
                fs = sampling_freq or filtered_data.get('sampling_freq', 100)
                time = np.arange(len(signal)) / fs

            metadata = {
                'filter_type': filtered_data.get('filter_type', 'Unknown'),
                'filter_params': filtered_data.get('filter_params', {}),
                'signal_type': filtered_data.get('signal_type', 'Unknown'),
            }

            csv_content = export_filtered_signal_csv(
                signal=signal,
                time=time,
                sampling_freq=sampling_freq or 100,
                metadata=metadata
            )

            return {
                'content': csv_content,
                'filename': f'filtered_signal_{filtered_data.get("filter_type", "unknown")}.csv'
            }

        except Exception as e:
            logger.error(f"Error exporting filtered signal to CSV: {e}")
            raise PreventUpdate

    @app.callback(
        Output("download-filtered-json", "data"),
        Input("btn-export-filtered-json", "n_clicks"),
        [
            State("store-filtered-signal", "data"),
            State("sampling-freq", "value"),
        ],
        prevent_initial_call=True,
    )
    def export_filtered_json(n_clicks, filtered_data, sampling_freq):
        """Export filtered signal to JSON."""
        if not n_clicks or not filtered_data:
            raise PreventUpdate

        try:
            signal = np.array(filtered_data.get('signal', []))
            time = np.array(filtered_data.get('time', []))

            if len(signal) == 0:
                raise PreventUpdate

            # If no time array, generate one
            if len(time) == 0:
                fs = sampling_freq or filtered_data.get('sampling_freq', 100)
                time = np.arange(len(signal)) / fs

            metadata = {
                'filter_type': filtered_data.get('filter_type', 'Unknown'),
                'filter_params': filtered_data.get('filter_params', {}),
                'signal_type': filtered_data.get('signal_type', 'Unknown'),
            }

            json_content = export_filtered_signal_json(
                signal=signal,
                time=time,
                sampling_freq=sampling_freq or 100,
                metadata=metadata
            )

            return {
                'content': json_content,
                'filename': f'filtered_signal_{filtered_data.get("filter_type", "unknown")}.json'
            }

        except Exception as e:
            logger.error(f"Error exporting filtered signal to JSON: {e}")
            raise PreventUpdate


def register_time_domain_export_callbacks(app):
    """Register export callbacks for time domain analysis page."""

    @app.callback(
        Output("download-time-domain-csv", "data"),
        Input("btn-export-time-domain-csv", "n_clicks"),
        State("store-time-domain-features", "data"),
        prevent_initial_call=True,
    )
    def export_time_domain_csv(n_clicks, features_data):
        """Export time domain features to CSV."""
        if not n_clicks or not features_data:
            raise PreventUpdate

        try:
            csv_content = export_time_domain_analysis_csv(features_data)

            return {
                'content': csv_content,
                'filename': 'time_domain_features.csv'
            }

        except Exception as e:
            logger.error(f"Error exporting time domain features to CSV: {e}")
            raise PreventUpdate

    @app.callback(
        Output("download-time-domain-json", "data"),
        Input("btn-export-time-domain-json", "n_clicks"),
        State("store-time-domain-features", "data"),
        prevent_initial_call=True,
    )
    def export_time_domain_json(n_clicks, features_data):
        """Export time domain features to JSON."""
        if not n_clicks or not features_data:
            raise PreventUpdate

        try:
            json_content = export_time_domain_analysis_json(features_data)

            return {
                'content': json_content,
                'filename': 'time_domain_features.json'
            }

        except Exception as e:
            logger.error(f"Error exporting time domain features to JSON: {e}")
            raise PreventUpdate


def register_frequency_domain_export_callbacks(app):
    """Register export callbacks for frequency domain analysis page."""

    @app.callback(
        Output("download-frequency-domain-csv", "data"),
        Input("btn-export-frequency-domain-csv", "n_clicks"),
        State("store-frequency-domain-features", "data"),
        prevent_initial_call=True,
    )
    def export_frequency_domain_csv(n_clicks, features_data):
        """Export frequency domain features to CSV."""
        if not n_clicks or not features_data:
            raise PreventUpdate

        try:
            csv_content = export_frequency_domain_analysis_csv(features_data)

            return {
                'content': csv_content,
                'filename': 'frequency_domain_features.csv'
            }

        except Exception as e:
            logger.error(f"Error exporting frequency domain features to CSV: {e}")
            raise PreventUpdate

    @app.callback(
        Output("download-frequency-domain-json", "data"),
        Input("btn-export-frequency-domain-json", "n_clicks"),
        State("store-frequency-domain-features", "data"),
        prevent_initial_call=True,
    )
    def export_frequency_domain_json(n_clicks, features_data):
        """Export frequency domain features to JSON."""
        if not n_clicks or not features_data:
            raise PreventUpdate

        try:
            json_content = export_frequency_domain_analysis_json(features_data)

            return {
                'content': json_content,
                'filename': 'frequency_domain_features.json'
            }

        except Exception as e:
            logger.error(f"Error exporting frequency domain features to JSON: {e}")
            raise PreventUpdate


def register_physiological_export_callbacks(app):
    """Register export callbacks for physiological features page."""

    @app.callback(
        Output("download-physio-csv", "data"),
        Input("btn-export-physio-csv", "n_clicks"),
        State("store-physio-features", "data"),
        prevent_initial_call=True,
    )
    def export_physio_csv(n_clicks, features_data):
        """Export physiological features to CSV."""
        if not n_clicks or not features_data:
            raise PreventUpdate

        try:
            signal_type = features_data.get('signal_type', 'Physiological')
            csv_content = export_features_csv(features_data, signal_type=signal_type)

            return {
                'content': csv_content,
                'filename': f'physiological_features_{signal_type.lower()}.csv'
            }

        except Exception as e:
            logger.error(f"Error exporting physiological features to CSV: {e}")
            raise PreventUpdate

    @app.callback(
        Output("download-physio-json", "data"),
        Input("btn-export-physio-json", "n_clicks"),
        State("store-physio-features", "data"),
        prevent_initial_call=True,
    )
    def export_physio_json(n_clicks, features_data):
        """Export physiological features to JSON."""
        if not n_clicks or not features_data:
            raise PreventUpdate

        try:
            signal_type = features_data.get('signal_type', 'Physiological')
            json_content = export_features_json(features_data, signal_type=signal_type)

            return {
                'content': json_content,
                'filename': f'physiological_features_{signal_type.lower()}.json'
            }

        except Exception as e:
            logger.error(f"Error exporting physiological features to JSON: {e}")
            raise PreventUpdate


def register_quality_export_callbacks(app):
    """Register export callbacks for quality assessment page."""

    @app.callback(
        Output("download-quality-csv", "data"),
        Input("btn-export-quality-csv", "n_clicks"),
        State("store-quality-results", "data"),
        prevent_initial_call=True,
    )
    def export_quality_csv(n_clicks, quality_data):
        """Export quality metrics to CSV."""
        if not n_clicks or not quality_data:
            raise PreventUpdate

        try:
            csv_content = export_quality_metrics_csv(quality_data)

            return {
                'content': csv_content,
                'filename': 'signal_quality_metrics.csv'
            }

        except Exception as e:
            logger.error(f"Error exporting quality metrics to CSV: {e}")
            raise PreventUpdate

    @app.callback(
        Output("download-quality-json", "data"),
        Input("btn-export-quality-json", "n_clicks"),
        State("store-quality-results", "data"),
        prevent_initial_call=True,
    )
    def export_quality_json(n_clicks, quality_data):
        """Export quality metrics to JSON."""
        if not n_clicks or not quality_data:
            raise PreventUpdate

        try:
            json_content = export_quality_metrics_json(quality_data)

            return {
                'content': json_content,
                'filename': 'signal_quality_metrics.json'
            }

        except Exception as e:
            logger.error(f"Error exporting quality metrics to JSON: {e}")
            raise PreventUpdate


def register_respiratory_export_callbacks(app):
    """Register export callbacks for respiratory analysis page."""

    @app.callback(
        Output("download-respiratory-csv", "data"),
        Input("btn-export-respiratory-csv", "n_clicks"),
        State("store-respiratory-results", "data"),
        prevent_initial_call=True,
    )
    def export_respiratory_csv(n_clicks, respiratory_data):
        """Export respiratory analysis to CSV."""
        if not n_clicks or not respiratory_data:
            raise PreventUpdate

        try:
            csv_content = export_respiratory_analysis_csv(respiratory_data)

            return {
                'content': csv_content,
                'filename': 'respiratory_analysis.csv'
            }

        except Exception as e:
            logger.error(f"Error exporting respiratory analysis to CSV: {e}")
            raise PreventUpdate

    @app.callback(
        Output("download-respiratory-json", "data"),
        Input("btn-export-respiratory-json", "n_clicks"),
        State("store-respiratory-results", "data"),
        prevent_initial_call=True,
    )
    def export_respiratory_json(n_clicks, respiratory_data):
        """Export respiratory analysis to JSON."""
        if not n_clicks or not respiratory_data:
            raise PreventUpdate

        try:
            json_content = export_respiratory_analysis_json(respiratory_data)

            return {
                'content': json_content,
                'filename': 'respiratory_analysis.json'
            }

        except Exception as e:
            logger.error(f"Error exporting respiratory analysis to JSON: {e}")
            raise PreventUpdate


def register_transforms_export_callbacks(app):
    """Register export callbacks for transforms page."""

    @app.callback(
        Output("download-transforms-csv", "data"),
        Input("btn-export-transforms-csv", "n_clicks"),
        [
            State("store-transform-results", "data"),
            State("transform-type-selector", "value"),
        ],
        prevent_initial_call=True,
    )
    def export_transforms_csv(n_clicks, transform_data, transform_type):
        """Export transform results to CSV."""
        if not n_clicks or not transform_data:
            raise PreventUpdate

        try:
            csv_content = export_transform_results_csv(
                transform_data,
                transform_type=transform_type or "Unknown"
            )

            return {
                'content': csv_content,
                'filename': f'transform_{transform_type or "results"}.csv'
            }

        except Exception as e:
            logger.error(f"Error exporting transform results to CSV: {e}")
            raise PreventUpdate

    @app.callback(
        Output("download-transforms-json", "data"),
        Input("btn-export-transforms-json", "n_clicks"),
        [
            State("store-transform-results", "data"),
            State("transform-type-selector", "value"),
        ],
        prevent_initial_call=True,
    )
    def export_transforms_json(n_clicks, transform_data, transform_type):
        """Export transform results to JSON."""
        if not n_clicks or not transform_data:
            raise PreventUpdate

        try:
            json_content = export_transform_results_json(
                transform_data,
                transform_type=transform_type or "Unknown"
            )

            return {
                'content': json_content,
                'filename': f'transform_{transform_type or "results"}.json'
            }

        except Exception as e:
            logger.error(f"Error exporting transform results to JSON: {e}")
            raise PreventUpdate


def register_all_export_callbacks(app):
    """Register all export callbacks for all pages."""
    logger.info("Registering all export callbacks...")

    try:
        register_filtering_export_callbacks(app)
        logger.info("✓ Filtering export callbacks registered")
    except Exception as e:
        logger.warning(f"Could not register filtering export callbacks: {e}")

    try:
        register_time_domain_export_callbacks(app)
        logger.info("✓ Time domain export callbacks registered")
    except Exception as e:
        logger.warning(f"Could not register time domain export callbacks: {e}")

    try:
        register_frequency_domain_export_callbacks(app)
        logger.info("✓ Frequency domain export callbacks registered")
    except Exception as e:
        logger.warning(f"Could not register frequency domain export callbacks: {e}")

    try:
        register_physiological_export_callbacks(app)
        logger.info("✓ Physiological export callbacks registered")
    except Exception as e:
        logger.warning(f"Could not register physiological export callbacks: {e}")

    try:
        register_quality_export_callbacks(app)
        logger.info("✓ Quality export callbacks registered")
    except Exception as e:
        logger.warning(f"Could not register quality export callbacks: {e}")

    try:
        register_respiratory_export_callbacks(app)
        logger.info("✓ Respiratory export callbacks registered")
    except Exception as e:
        logger.warning(f"Could not register respiratory export callbacks: {e}")

    try:
        register_transforms_export_callbacks(app)
        logger.info("✓ Transforms export callbacks registered")
    except Exception as e:
        logger.warning(f"Could not register transforms export callbacks: {e}")

    logger.info("All export callbacks registered successfully")
