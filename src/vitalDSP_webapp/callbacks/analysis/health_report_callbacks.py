"""
Health report generation callbacks for vitalDSP webapp.

This module handles comprehensive health report generation with customizable templates,
professional formatting, and multiple export formats for clinical and research use.

This implementation uses:
- Real feature extraction from physiological signals
- vitalDSP's HealthReportGenerator for professional report generation
- Multiple data sources (uploaded signals, pipeline results, stored features)
"""

import numpy as np
from dash import Input, Output, State, callback_context, no_update, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import logging
from vitalDSP_webapp.services.data.enhanced_data_service import (
    get_enhanced_data_service,
)
from datetime import datetime
import traceback

# Import feature extraction
from vitalDSP_webapp.callbacks.analysis.health_report_feature_extractor import (
    extract_health_features_from_data,
)

# Import vitalDSP health report generator
try:
    from vitalDSP.health_analysis.health_report_generator import HealthReportGenerator

    VITALDSP_AVAILABLE = True
except ImportError as e:
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning(f"vitalDSP HealthReportGenerator not available: {e}")
    VITALDSP_AVAILABLE = False

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
            State("health-report-data-selection", "value"),
        ],
    )
    def health_report_generation_callback(
        n_clicks,
        pathname,
        report_type,
        sections,
        customization,
        format_type,
        style,
        data_selection,
    ):
        """
        Main callback for health report generation.

        This callback:
        1. Retrieves signal data from data service
        2. Extracts comprehensive health features
        3. Generates professional health report using vitalDSP
        4. Returns preview and content for display
        """
        ctx = callback_context

        # Log callback trigger
        logger.info(f"=== HEALTH REPORT CALLBACK TRIGGERED ===")
        logger.info(f"Context triggered: {ctx.triggered}")
        logger.info(f"n_clicks: {n_clicks}")
        logger.info(f"pathname: {pathname}")

        if not ctx.triggered:
            # Initial page load - show instructions
            logger.info("No trigger context - showing initial instructions")
            return _generate_initial_content()

        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        logger.info(f"Button ID: {button_id}")

        # On initial page load via URL, show instructions
        if button_id == "url":
            logger.info("URL change detected - showing initial instructions")
            return _generate_initial_content()

        # Only process on button click
        if button_id != "health-report-generate-btn" or not n_clicks:
            logger.info(f"Not the generate button or no clicks - returning no_update")
            return (
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
            )

        try:
            logger.info(
                f"Generating health report: type={report_type}, style={style}, data={data_selection}"
            )

            # Step 1: Get signal data
            signal_data, fs, signal_type, data_info = _get_signal_data(data_selection)

            if signal_data is None:
                return _generate_error_content(
                    "No Data Available",
                    "Please upload signal data first or run the pipeline to generate analysis results.",
                )

            # Step 2: Extract health features
            logger.info(f"Extracting health features from {signal_type} signal...")
            extracted_features = extract_health_features_from_data(
                signal_data=signal_data, sampling_frequency=fs, signal_type=signal_type
            )

            if not extracted_features or len(extracted_features) < 3:
                return _generate_error_content(
                    "Feature Extraction Failed",
                    f"Could not extract sufficient features from signal. Extracted: {list(extracted_features.keys())}",
                )

            logger.info(
                f"Successfully extracted {len(extracted_features)} health features"
            )
            logger.debug(f"Features: {list(extracted_features.keys())}")

            # Step 3: Generate health report using vitalDSP
            if not VITALDSP_AVAILABLE:
                return _generate_error_content(
                    "vitalDSP Not Available",
                    "vitalDSP HealthReportGenerator module is not installed or not available.",
                )

            logger.info(
                "Generating health report with vitalDSP HealthReportGenerator..."
            )

            try:
                # Determine segment duration based on signal length
                signal_duration_minutes = len(signal_data) / fs / 60
                segment_duration = "5_min" if signal_duration_minutes >= 5 else "1_min"

                logger.debug(
                    f"Signal duration: {signal_duration_minutes:.2f} min, using {segment_duration} segments"
                )

                # Create health report generator
                generator = HealthReportGenerator(
                    feature_data=extracted_features,
                    segment_duration=segment_duration,
                    max_workers=4,  # Parallel processing
                )

                # Generate report HTML
                report_html = generator.generate()

                logger.info(
                    f"Health report generated successfully ({len(report_html)} chars)"
                )

            except Exception as e:
                logger.error(f"vitalDSP report generation failed: {e}", exc_info=True)
                return _generate_error_content(
                    "Report Generation Error",
                    f"vitalDSP HealthReportGenerator failed: {str(e)}\n\nExtracted features: {list(extracted_features.keys())}",
                )

            # Step 4: Create preview
            preview = _create_report_preview(
                report_html=report_html,
                report_type=report_type,
                style=style,
                features_count=len(extracted_features),
                signal_type=signal_type,
                duration=len(signal_data) / fs,
            )

            # Step 5: Create content display
            content = _create_report_content(report_html)

            # Step 6: Create templates display
            templates = _create_templates_display(report_type)

            # Step 7: Create template settings display
            template_settings = _create_template_settings(
                report_type, style, format_type, sections, customization
            )

            # Step 8: Create report history
            history = _create_report_history()

            # Step 9: Store report data
            stored_report_data = {
                "report_html": report_html,
                "features": extracted_features,
                "report_type": report_type,
                "signal_type": signal_type,
                "sampling_frequency": fs,
                "signal_duration": len(signal_data) / fs,
                "generation_date": datetime.now().isoformat(),
                "data_info": data_info,
            }

            stored_config = {
                "report_type": report_type,
                "sections": sections or [],
                "customization": customization or [],
                "format": format_type,
                "style": style,
                "data_selection": data_selection,
            }

            logger.info("Health report generation completed successfully")

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
            logger.error(
                f"Unexpected error in health report generation: {e}", exc_info=True
            )
            error_details = traceback.format_exc()
            return _generate_error_content(
                "Unexpected Error",
                f"An unexpected error occurred during report generation:\n\n{str(e)}\n\nDetails:\n{error_details}",
            )


def _get_signal_data(data_selection: str):
    """
    Retrieve signal data from available sources.

    Args:
        data_selection: "all", "recent", "time_range", or "selected"

    Returns:
        Tuple of (signal_data, sampling_frequency, signal_type, data_info)
    """
    try:
        # Get data service
        data_service = get_enhanced_data_service()
        stored_data = data_service.get_all_data()

        if not stored_data:
            logger.warning("No data available in data service")
            return None, None, None, None

        # For now, always use most recent data
        # TODO: Implement time_range and selected options
        data_id = list(stored_data.keys())[-1]
        df = stored_data[data_id]
        data_info = data_service.get_data_info(data_id)

        if df is None or df.empty:
            logger.warning(f"Data {data_id} is empty or invalid")
            return None, None, None, None

        # Extract signal (use first column)
        signal_data = df.iloc[:, 0].values

        # Get metadata
        fs = data_info.get("sampling_frequency", 1000)
        signal_type = data_info.get("signal_type", "ECG")

        logger.info(
            f"Retrieved signal: {signal_type}, {len(signal_data)} samples, {fs} Hz"
        )

        return signal_data, fs, signal_type, data_info

    except Exception as e:
        logger.error(f"Error retrieving signal data: {e}", exc_info=True)
        return None, None, None, None


def _create_report_preview(
    report_html, report_type, style, features_count, signal_type, duration
):
    """Create report preview display."""
    return html.Div(
        [
            html.H4(f"📋 Health Report Preview", className="mb-3"),
            dbc.Alert(
                [
                    html.Strong("✅ Report Generated Successfully!"),
                    html.Br(),
                    html.Br(),
                    html.Div(
                        [
                            html.Strong("Report Type: "),
                            f"{report_type.replace('_', ' ').title()}",
                            html.Br(),
                            html.Strong("Style: "),
                            f"{style.title()}",
                            html.Br(),
                            html.Strong("Signal: "),
                            f"{signal_type}, {duration:.1f} seconds",
                            html.Br(),
                            html.Strong("Features Extracted: "),
                            f"{features_count}",
                        ]
                    ),
                ],
                color="success",
                className="mb-3",
            ),
            html.P(
                "Scroll down to view the complete report in the 'Report Content' section.",
                className="text-muted",
            ),
            html.Hr(),
            html.H6("Report Preview (First Section):"),
            html.Div(
                html.Iframe(
                    srcDoc=report_html,
                    style={
                        "width": "100%",
                        "height": "400px",
                        "border": "1px solid #ddd",
                        "borderRadius": "4px",
                    },
                )
            ),
        ],
        className="p-3",
    )


def _create_report_content(report_html):
    """Create full report content display."""
    return html.Div(
        [
            html.Div(
                [
                    html.H5("📊 Complete Health Report", className="mb-3"),
                    dbc.Badge(
                        "Generated with vitalDSP", color="primary", className="mb-3"
                    ),
                ],
                className="d-flex justify-content-between align-items-center",
            ),
            html.Hr(),
            html.Iframe(
                srcDoc=report_html,
                style={
                    "width": "100%",
                    "height": "1000px",
                    "border": "1px solid #ddd",
                    "borderRadius": "4px",
                    "backgroundColor": "white",
                },
            ),
            html.Div(
                [
                    html.P(
                        "💡 Tip: Use the 'Save Report' button in the Report Actions section to download this report.",
                        className="text-muted small mt-3",
                    )
                ]
            ),
        ],
        className="p-3",
    )


def _create_templates_display(selected_template):
    """Create templates display."""
    templates_info = {
        "comprehensive": {
            "name": "Comprehensive Health Assessment",
            "description": "Complete analysis with all sections and detailed interpretations",
            "sections": ["All sections included"],
        },
        "cardiovascular": {
            "name": "Cardiovascular Health Report",
            "description": "Focus on heart-related metrics and HRV analysis",
            "sections": ["HRV features", "Heart rate analysis", "ECG/PPG features"],
        },
        "respiratory": {
            "name": "Respiratory Health Report",
            "description": "Focus on breathing patterns and respiratory rate",
            "sections": [
                "Respiratory rate",
                "Breathing regularity",
                "Respiratory features",
            ],
        },
        "wellness": {
            "name": "General Wellness Report",
            "description": "Overview of key health indicators",
            "sections": ["Summary statistics", "Key findings", "Recommendations"],
        },
        "research": {
            "name": "Research Summary Report",
            "description": "Technical details and comprehensive feature analysis",
            "sections": ["All features", "Statistical analysis", "Methodology"],
        },
        "clinical": {
            "name": "Clinical Assessment Report",
            "description": "Medical-grade assessment for clinical use",
            "sections": ["Clinical metrics", "Diagnostic features", "Risk assessment"],
        },
        "fitness": {
            "name": "Fitness & Performance Report",
            "description": "Performance metrics and fitness indicators",
            "sections": ["Performance metrics", "Training zones", "Recovery analysis"],
        },
        "custom": {
            "name": "Custom Report",
            "description": "User-defined sections and customization",
            "sections": ["User-selected sections"],
        },
    }

    template_cards = []
    for template_key, template_info in templates_info.items():
        is_selected = template_key == selected_template

        card = dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H6(template_info["name"], className="card-title"),
                        html.P(
                            template_info["description"],
                            className="card-text small text-muted",
                        ),
                        html.Ul(
                            [
                                html.Li(section, className="small")
                                for section in template_info["sections"][:3]
                            ],
                            className="mb-2",
                        ),
                        dbc.Badge(
                            "✓ Selected" if is_selected else "Available",
                            color="success" if is_selected else "secondary",
                            className="mt-2",
                        ),
                    ]
                )
            ],
            className="mb-3",
            color="primary" if is_selected else "light",
            outline=not is_selected,
        )
        template_cards.append(dbc.Col(card, md=6))

    return html.Div(
        [
            html.H6("📋 Available Report Templates", className="mb-3"),
            dbc.Row(template_cards),
        ]
    )


def _create_template_settings(report_type, style, format_type, sections, customization):
    """Create template settings display."""
    return html.Div(
        [
            html.H6("⚙️ Current Configuration", className="mb-3"),
            dbc.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th("Setting", style={"width": "40%"}),
                                html.Th("Value"),
                            ]
                        )
                    ),
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td(html.Strong("Report Type")),
                                    html.Td(report_type.replace("_", " ").title()),
                                ]
                            ),
                            html.Tr(
                                [html.Td(html.Strong("Style")), html.Td(style.title())]
                            ),
                            html.Tr(
                                [
                                    html.Td(html.Strong("Format")),
                                    html.Td(format_type.upper()),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td(html.Strong("Sections")),
                                    html.Td(f"{len(sections or [])} selected"),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td(html.Strong("Customization")),
                                    html.Td(
                                        f"{len(customization or [])} options enabled"
                                    ),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td(html.Strong("Generated")),
                                    html.Td(
                                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    ),
                                ]
                            ),
                        ]
                    ),
                ],
                bordered=True,
                hover=True,
                responsive=True,
                size="sm",
                className="mb-0",
            ),
        ]
    )


def _create_report_history():
    """Create report history display."""
    # TODO: Implement actual history tracking
    # For now, show placeholder
    return html.Div(
        [
            html.H6("📚 Recent Reports", className="mb-3"),
            dbc.Alert(
                [
                    html.I(className="bi bi-info-circle me-2"),
                    "Report history tracking will be available in a future update. ",
                    "For now, save important reports using the 'Save Report' button.",
                ],
                color="info",
                className="mb-0",
            ),
        ]
    )


def _generate_initial_content():
    """Generate initial content when page loads."""
    instructions = html.Div(
        [
            html.H5("👋 Welcome to Health Report Generator", className="mb-3"),
            html.P(
                "Generate comprehensive health reports from your physiological signal data with professional "
                "formatting and detailed analysis.",
                className="mb-3",
            ),
            html.H6("📋 How to Use:", className="mb-2"),
            html.Ol(
                [
                    html.Li(
                        "Upload signal data (ECG, PPG, or respiratory signals) using the Upload page"
                    ),
                    html.Li("Configure your report settings in the left panel"),
                    html.Li("Click 'Generate Health Report' to create your report"),
                    html.Li(
                        "Review the generated report and use 'Save Report' to download"
                    ),
                ],
                className="mb-3",
            ),
            html.H6("✨ Features:", className="mb-2"),
            html.Ul(
                [
                    html.Li("Automatic feature extraction from signals"),
                    html.Li("HRV (Heart Rate Variability) analysis"),
                    html.Li("Respiratory rate estimation"),
                    html.Li("Statistical and frequency domain features"),
                    html.Li("Professional HTML reports with visualizations"),
                    html.Li("Multiple report templates and styles"),
                ],
                className="mb-3",
            ),
            dbc.Alert(
                [
                    html.Strong("💡 Note: "),
                    "Make sure you have uploaded signal data before generating a report.",
                ],
                color="info",
            ),
        ],
        className="p-4",
    )

    return (
        instructions,  # preview
        instructions,  # content
        "",  # templates
        "",  # template_settings
        "",  # history
        None,  # stored_report_data
        None,  # stored_config
    )


def _generate_error_content(error_title, error_message):
    """Generate error content display."""
    error_display = html.Div(
        [
            html.H5(f"❌ {error_title}", className="text-danger mb-3"),
            dbc.Alert(
                [
                    html.Pre(
                        error_message,
                        style={"whiteSpace": "pre-wrap", "fontSize": "0.9em"},
                    )
                ],
                color="danger",
            ),
            html.H6("🔧 Troubleshooting:", className="mt-4 mb-2"),
            html.Ul(
                [
                    html.Li("Ensure you have uploaded signal data via the Upload page"),
                    html.Li(
                        "Check that the signal data is valid (not empty or corrupted)"
                    ),
                    html.Li("Verify that vitalDSP modules are properly installed"),
                    html.Li("Try uploading a different signal file"),
                    html.Li("Check the browser console for additional error details"),
                ]
            ),
        ],
        className="p-4",
    )

    return (
        error_display,  # preview
        error_display,  # content
        "",  # templates
        "",  # template_settings
        "",  # history
        None,  # stored_report_data
        None,  # stored_config
    )
