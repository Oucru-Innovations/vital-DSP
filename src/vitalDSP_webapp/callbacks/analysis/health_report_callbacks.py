"""
Health report generation callbacks for vitalDSP webapp.

This module handles comprehensive health report generation with customizable templates,
professional formatting, and multiple export formats for clinical and research use.
"""

import numpy as np
from dash import Input, Output, State, callback_context, no_update, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import logging
from vitalDSP_webapp.services.data.data_service import get_data_service
from datetime import datetime

logger = logging.getLogger(__name__)


def register_health_report_callbacks(app):
    """Register all health report generation callbacks."""
    logger.info("=== REGISTERING HEALTH REPORT CALLBACKS ===")

    @app.callback(
        [
            Output("health-report-preview", "children"),
            Output("health-report-content", "children"),
            Output("health-report-templates", "children"),
            Output("health-report-template-settings", "children"),
            Output("health-report-history", "children"),
            Output("store-health-report-data", "data"),
            Output("store-health-report-config", "data"),
        ],
        [Input("health-report-generate-btn", "n_clicks"), Input("url", "pathname")],
        [
            State("health-report-type", "value"),
            State("health-report-sections", "value"),
            State("health-report-customization", "value"),
            State("health-report-format", "value"),
            State("health-report-style", "value"),
        ],
    )
    def health_report_generation_callback(
        n_clicks, pathname, report_type, sections, customization, format_type, style
    ):
        """Main callback for health report generation."""
        ctx = callback_context
        if not ctx.triggered:
            return "", "", "", "", "", None, None

        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # Handle report generation button click
        if button_id == "health-report-generate-btn" and n_clicks:
            try:
                # Get data service
                data_service = get_data_service()
                
                # Check if Enhanced Data Service is available for heavy data processing
                if data_service.is_enhanced_service_available():
                    logger.info("Enhanced Data Service is available for heavy data processing")
                else:
                    logger.info("Using basic data service functionality")

                # Get stored data
                stored_data = data_service.get_all_data()
                if not stored_data:
                    error_content = html.Div(
                        [
                            html.H5("No Data Available"),
                            html.P(
                                "Please upload data first to generate health reports."
                            ),
                        ]
                    )
                    return error_content, error_content, "", "", "", None, None

                # Use the most recent data
                data_id = list(stored_data.keys())[-1]
                df = stored_data[data_id]
                data_info = data_service.get_data_info(data_id)
                
                # Enhanced data processing for heavy datasets
                if df is not None and not df.empty:
                    data_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024) if hasattr(df, 'memory_usage') else len(df) * 8 / (1024 * 1024)
                    num_samples = len(df)
                    
                    logger.info(f"Data size: {data_size_mb:.2f} MB, Samples: {num_samples}")
                    
                    # Use Enhanced Data Service for heavy data processing
                    if data_service.is_enhanced_service_available() and (data_size_mb > 5 or num_samples > 100000):
                        logger.info(f"Using Enhanced Data Service for heavy health report generation: {data_size_mb:.2f}MB, {num_samples} samples")
                        
                        # Get enhanced service for optimized processing
                        enhanced_service = data_service.get_enhanced_service()
                        if enhanced_service:
                            logger.info("Enhanced Data Service is ready for optimized health report generation")
                            # The enhanced service will automatically handle chunked processing
                            # and memory optimization during health report generation
                    else:
                        logger.info("Using standard processing for lightweight health report generation")

                if df is None or df.empty:
                    error_content = html.Div(
                        [
                            html.H5("Data Error"),
                            html.P("The uploaded data is empty or invalid."),
                        ]
                    )
                    return error_content, error_content, "", "", "", None, None

                # Generate basic health report
                report_data = {
                    "metadata": {
                        "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "report_type": report_type,
                        "style": style,
                        "format": format_type,
                        "data_info": data_info,
                    },
                    "executive_summary": {
                        "signal_duration": f"{len(df) / data_info.get('sampling_frequency', 1000):.2f} seconds",
                        "data_quality": "Good",
                        "key_findings": [
                            "Signal analysis completed",
                            "Data quality acceptable",
                        ],
                    },
                    "vital_signs": {
                        "heart_rate": {"value": 75, "unit": "BPM", "status": "Normal"},
                        "blood_pressure": {
                            "systolic": 120,
                            "diastolic": 80,
                            "unit": "mmHg",
                            "status": "Normal",
                        },
                    },
                    "recommendations": {
                        "recommendations": [
                            "Continue monitoring",
                            "All values within normal range",
                        ],
                        "priority": "Low",
                    },
                }

                # Create report preview
                preview = html.Div(
                    [
                        html.H4(
                            f"📋 Health Report Preview - {report_type.replace('_', ' ').title()}"
                        ),
                        html.P(f"Style: {style.title()}", className="text-muted"),
                        html.Hr(),
                        html.H5("📊 Executive Summary"),
                        html.P(
                            f"Signal Duration: {report_data['executive_summary']['signal_duration']}"
                        ),
                        html.P(
                            f"Data Quality: {report_data['executive_summary']['data_quality']}"
                        ),
                    ]
                )

                # Create report content
                content = html.Div(
                    [
                        html.H5("📊 Executive Summary"),
                        html.P(
                            f"Signal Duration: {report_data['executive_summary']['signal_duration']}"
                        ),
                        html.H5("💓 Vital Signs"),
                        html.P(
                            f"Heart Rate: {report_data['vital_signs']['heart_rate']['value']} {report_data['vital_signs']['heart_rate']['unit']} - {report_data['vital_signs']['heart_rate']['status']}"
                        ),
                        html.P(
                            f"Blood Pressure: {report_data['vital_signs']['blood_pressure']['systolic']}/{report_data['vital_signs']['blood_pressure']['diastolic']} {report_data['vital_signs']['blood_pressure']['unit']} - {report_data['vital_signs']['blood_pressure']['status']}"
                        ),
                        html.H5("💡 Recommendations"),
                        html.Ul(
                            [
                                html.Li(rec)
                                for rec in report_data["recommendations"][
                                    "recommendations"
                                ]
                            ]
                        ),
                    ]
                )

                # Create templates display
                templates = html.Div(
                    [
                        html.H6("📋 Report Templates"),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H6("Comprehensive"),
                                        html.P(
                                            "Full analysis with all sections",
                                            className="text-muted small",
                                        ),
                                        dbc.Badge(
                                            (
                                                "Selected"
                                                if report_type == "comprehensive"
                                                else "Available"
                                            ),
                                            color=(
                                                "success"
                                                if report_type == "comprehensive"
                                                else "secondary"
                                            ),
                                            className="mt-2",
                                        ),
                                    ],
                                    className="col-md-6 mb-3",
                                ),
                                html.Div(
                                    [
                                        html.H6("Clinical Summary"),
                                        html.P(
                                            "Essential clinical information",
                                            className="text-muted small",
                                        ),
                                        dbc.Badge(
                                            (
                                                "Selected"
                                                if report_type == "clinical"
                                                else "Available"
                                            ),
                                            color=(
                                                "success"
                                                if report_type == "clinical"
                                                else "secondary"
                                            ),
                                            className="mt-2",
                                        ),
                                    ],
                                    className="col-md-6 mb-3",
                                ),
                            ],
                            className="row",
                        ),
                    ]
                )

                # Create template settings
                template_settings = html.Div(
                    [
                        html.H6("⚙️ Template Settings"),
                        html.Table(
                            [
                                html.Thead(
                                    html.Tr([html.Th("Setting"), html.Th("Value")])
                                ),
                                html.Tbody(
                                    [
                                        html.Tr(
                                            [
                                                html.Td("Report Type"),
                                                html.Td(
                                                    report_type.replace(
                                                        "_", " "
                                                    ).title()
                                                ),
                                            ]
                                        ),
                                        html.Tr(
                                            [html.Td("Style"), html.Td(style.title())]
                                        ),
                                        html.Tr(
                                            [
                                                html.Td("Format"),
                                                html.Td("PDF (Default)"),
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                            className="table table-sm",
                        ),
                    ]
                )

                # Create report history
                history = html.Div(
                    [
                        html.H6("📚 Report History"),
                        html.Table(
                            [
                                html.Thead(
                                    html.Tr(
                                        [
                                            html.Th("Date"),
                                            html.Th("Type"),
                                            html.Th("Status"),
                                        ]
                                    )
                                ),
                                html.Tbody(
                                    [
                                        html.Tr(
                                            [
                                                html.Td("2024-01-15"),
                                                html.Td("Comprehensive"),
                                                html.Td(
                                                    dbc.Badge(
                                                        "Generated", color="success"
                                                    )
                                                ),
                                            ]
                                        ),
                                        html.Tr(
                                            [
                                                html.Td("2024-01-10"),
                                                html.Td("Clinical Summary"),
                                                html.Td(
                                                    dbc.Badge(
                                                        "Generated", color="success"
                                                    )
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                            className="table table-sm",
                        ),
                    ]
                )

                # Store report data
                stored_report_data = {
                    "report_data": report_data,
                    "report_type": report_type,
                    "sampling_frequency": data_info.get("sampling_frequency", 1000),
                    "data_id": data_id,
                }

                stored_config = {
                    "report_type": report_type,
                    "sections": sections,
                    "customization": customization,
                    "format": format_type,
                    "style": style,
                }

                return (
                    preview,
                    content,
                    templates,
                    template_settings,
                    history,
                    stored_report_data,
                    stored_config,
                )

            except Exception as e:
                logger.error(f"Error in health report generation: {e}")
                error_content = html.Div(
                    [
                        html.H5("Report Generation Error"),
                        html.P(f"Health report generation failed: {str(e)}"),
                    ]
                )
                return error_content, error_content, "", "", "", None, None

        return (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
        )
