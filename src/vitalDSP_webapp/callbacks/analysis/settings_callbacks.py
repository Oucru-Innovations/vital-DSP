"""
Application settings callbacks for vitalDSP webapp.

This module handles comprehensive application configuration including general,
analysis, data, system, and security settings with full functionality.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, callback_context, no_update, html, dcc
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import logging
import json
from datetime import datetime
import base64
import io

# Import our settings services
from vitalDSP_webapp.services.settings_service import (
    get_settings_service,
    get_current_settings,
    update_settings,
)
from vitalDSP_webapp.utils.settings_utils import (
    ThemeManager,
    SystemMonitor,
    SettingsValidator,
    SettingsExporter,
    get_system_recommendations,
)

logger = logging.getLogger(__name__)


def register_settings_callbacks(app):
    """Register all application settings callbacks."""
    logger.info("=== REGISTERING COMPREHENSIVE SETTINGS CALLBACKS ===")

    # Main settings management callback
    @app.callback(
        [
            Output("settings-status", "children"),
            Output("store-settings-data", "data"),
            Output("store-settings-config", "data"),
        ],
        [
            Input("settings-save-btn", "n_clicks"),
            Input("settings-reset-btn", "n_clicks"),
            Input("settings-export-btn", "n_clicks"),
            Input("settings-import-btn", "n_clicks"),
            Input("url", "pathname"),
        ],
        [
            State("settings-theme", "value"),
            State("settings-timezone", "value"),
            State("settings-page-size", "value"),
            State("settings-auto-refresh", "value"),
            State("settings-display-options", "value"),
            State("settings-default-sampling-freq", "value"),
            State("settings-default-fft-points", "value"),
            State("settings-default-window", "value"),
            State("settings-peak-threshold", "value"),
            State("settings-quality-threshold", "value"),
            State("settings-analysis-options", "value"),
            State("settings-max-file-size", "value"),
            State("settings-auto-save", "value"),
            State("settings-data-retention", "value"),
            State("settings-export-format", "value"),
            State("settings-image-format", "value"),
            State("settings-export-options", "value"),
            State("settings-cpu-usage", "value"),
            State("settings-memory-limit", "value"),
            State("settings-parallel-threads", "value"),
            State("settings-session-timeout", "value"),
            State("settings-security-options", "value"),
        ],
    )
    def settings_management_callback(
        save_clicks,
        reset_clicks,
        export_clicks,
        import_clicks,
        pathname,
        theme,
        timezone,
        page_size,
        auto_refresh,
        display_options,
        sampling_freq,
        fft_points,
        window_type,
        peak_threshold,
        quality_threshold,
        analysis_options,
        max_file_size,
        auto_save,
        data_retention,
        export_format,
        image_format,
        export_options,
        cpu_usage,
        memory_limit,
        parallel_threads,
        session_timeout,
        security_options,
    ):
        """Main callback for comprehensive settings management."""
        ctx = callback_context
        if not ctx.triggered:
            return create_settings_status_display(), {}, {}

        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        try:
            # Handle save button click
            if button_id == "settings-save-btn" and save_clicks:
                status, settings_data, config = handle_save_settings(
                    theme,
                    timezone,
                    page_size,
                    auto_refresh,
                    display_options,
                    sampling_freq,
                    fft_points,
                    window_type,
                    peak_threshold,
                    quality_threshold,
                    analysis_options,
                    max_file_size,
                    auto_save,
                    data_retention,
                    export_format,
                    image_format,
                    export_options,
                    cpu_usage,
                    memory_limit,
                    parallel_threads,
                    session_timeout,
                    security_options,
                )
                return status, settings_data, config

            # Handle reset button click
            elif button_id == "settings-reset-btn" and reset_clicks:
                status, settings_data, config = handle_reset_settings()
                return status, settings_data, config

            # Handle export button click
            elif button_id == "settings-export-btn" and export_clicks:
                status, settings_data, config = handle_export_settings()
                return status, settings_data, config

            # Handle import button click
            elif button_id == "settings-import-btn" and import_clicks:
                status, settings_data, config = handle_import_settings()
                return status, settings_data, config

            # Default: show current status
            return create_settings_status_display(), {}, {}

        except Exception as e:
            logger.error(f"Error in settings management: {e}")
            error_content = create_error_display("Settings Error", str(e))
            return error_content, {}, {}

    # Real-time system monitoring callback
    @app.callback(
        Output("system-monitor-display", "children"),
        [Input("system-monitor-interval", "n_intervals")],
        prevent_initial_call=True,
    )
    def system_monitoring_callback(n_intervals):
        """Real-time system monitoring callback."""
        try:
            return create_system_monitor_display()
        except Exception as e:
            logger.error(f"Error in system monitoring: {e}")
            return create_error_display("System Monitor Error", str(e))

    # Settings validation callback
    @app.callback(
        Output("settings-validation-display", "children"),
        [Input("settings-validate-btn", "n_clicks")],
        prevent_initial_call=True,
    )
    def settings_validation_callback(n_clicks):
        """Settings validation callback."""
        try:
            if n_clicks:
                return create_settings_validation_display()
            return ""
        except Exception as e:
            logger.error(f"Error in settings validation: {e}")
            return create_error_display("Validation Error", str(e))

    # Theme preview callback
    @app.callback(
        Output("theme-preview-display", "children"),
        [Input("settings-theme", "value")],
        prevent_initial_call=True,
    )
    def theme_preview_callback(theme):
        """Theme preview callback."""
        try:
            if theme:
                return create_theme_preview_display(theme)
            return ""
        except Exception as e:
            logger.error(f"Error in theme preview: {e}")
            return create_error_display("Theme Preview Error", str(e))

    # Settings recommendations callback
    @app.callback(
        Output("settings-recommendations-display", "children"),
        [Input("settings-recommendations-btn", "n_clicks")],
        prevent_initial_call=True,
    )
    def settings_recommendations_callback(n_clicks):
        """Settings recommendations callback."""
        try:
            if n_clicks:
                return create_settings_recommendations_display()
            return ""
        except Exception as e:
            logger.error(f"Error in settings recommendations: {e}")
            return create_error_display("Recommendations Error", str(e))


def handle_save_settings(
    theme,
    timezone,
    page_size,
    auto_refresh,
    display_options,
    sampling_freq,
    fft_points,
    window_type,
    peak_threshold,
    quality_threshold,
    analysis_options,
    max_file_size,
    auto_save,
    data_retention,
    export_format,
    image_format,
    export_options,
    cpu_usage,
    memory_limit,
    parallel_threads,
    session_timeout,
    security_options,
):
    """Handle saving all settings."""
    try:
        service = get_settings_service()

        # Update general settings
        service.update_general_settings(
            theme=theme or "light",
            timezone=timezone or "UTC",
            page_size=page_size or 25,
            auto_refresh_interval=auto_refresh or 30,
            display_options=display_options or ["tooltips", "loading"],
        )

        # Update analysis settings
        service.update_analysis_settings(
            default_sampling_freq=sampling_freq or 1000,
            default_fft_points=fft_points or 1024,
            default_window_type=window_type or "hann",
            peak_threshold=peak_threshold or 0.5,
            quality_threshold=quality_threshold or 0.7,
            analysis_options=analysis_options or ["auto_detect", "advanced_features"],
        )

        # Update data settings
        service.update_data_settings(
            max_file_size=max_file_size or 100,
            auto_save_interval=auto_save or 5,
            data_retention_days=data_retention or 30,
            export_format=export_format or "csv",
            image_format=image_format or "png",
            export_options=export_options or ["metadata", "high_quality"],
        )

        # Update system settings
        service.update_system_settings(
            max_cpu_usage=cpu_usage or 80,
            memory_limit_gb=memory_limit or 4,
            parallel_threads=parallel_threads or 4,
            session_timeout_minutes=session_timeout or 60,
            security_options=security_options or ["https", "encryption"],
        )

        # Get updated settings for storage
        current_settings = service.get_all_settings()
        settings_summary = service.get_settings_summary()

        # Create success status
        status_display = create_success_display(
            "Settings Saved Successfully", "All settings have been updated and saved."
        )

        return status_display, settings_summary, current_settings

    except Exception as e:
        logger.error(f"Error saving settings: {e}")
        error_display = create_error_display(
            "Save Error", f"Failed to save settings: {str(e)}"
        )
        return error_display, {}, {}


def handle_reset_settings():
    """Handle resetting settings to defaults."""
    try:
        service = get_settings_service()
        success = service.reset_to_defaults()

        if success:
            current_settings = service.get_all_settings()
            settings_summary = service.get_settings_summary()

            status_display = create_success_display(
                "Settings Reset", "All settings have been reset to default values."
            )

            return status_display, settings_summary, current_settings
        else:
            error_display = create_error_display(
                "Reset Error", "Failed to reset settings."
            )
            return error_display, {}, {}

    except Exception as e:
        logger.error(f"Error resetting settings: {e}")
        error_display = create_error_display(
            "Reset Error", f"Failed to reset settings: {str(e)}"
        )
        return error_display, {}, {}


def handle_export_settings():
    """Handle exporting settings."""
    try:
        service = get_settings_service()
        settings_data = service.get_all_settings()

        # Convert to dictionary for export
        settings_dict = {
            "general": {
                "theme": settings_data.general.theme,
                "timezone": settings_data.general.timezone,
                "page_size": settings_data.general.page_size,
                "auto_refresh_interval": settings_data.general.auto_refresh_interval,
                "display_options": settings_data.general.display_options,
            },
            "analysis": {
                "default_sampling_freq": settings_data.analysis.default_sampling_freq,
                "default_fft_points": settings_data.analysis.default_fft_points,
                "default_window_type": settings_data.analysis.default_window_type,
                "peak_threshold": settings_data.analysis.peak_threshold,
                "quality_threshold": settings_data.analysis.quality_threshold,
                "analysis_options": settings_data.analysis.analysis_options,
            },
            "data": {
                "max_file_size": settings_data.data.max_file_size,
                "auto_save_interval": settings_data.data.auto_save_interval,
                "data_retention_days": settings_data.data.data_retention_days,
                "export_format": settings_data.data.export_format,
                "image_format": settings_data.data.image_format,
                "export_options": settings_data.data.export_options,
            },
            "system": {
                "max_cpu_usage": settings_data.system.max_cpu_usage,
                "memory_limit_gb": settings_data.system.memory_limit_gb,
                "parallel_threads": settings_data.system.parallel_threads,
                "session_timeout_minutes": settings_data.system.session_timeout_minutes,
                "security_options": settings_data.system.security_options,
            },
        }

        # Export to JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vitaldsp_settings_{timestamp}.json"

        try:
            SettingsExporter.export_settings_json(settings_dict, filename)
            status_display = create_success_display(
                "Settings Exported", f"Settings exported to {filename}"
            )
        except Exception as export_error:
            logger.error(f"Export error: {export_error}")
            status_display = create_error_display(
                "Export Error", f"Failed to export: {str(export_error)}"
            )

        return status_display, settings_dict, {}

    except Exception as e:
        logger.error(f"Error handling export: {e}")
        error_display = create_error_display(
            "Export Error", f"Failed to handle export: {str(e)}"
        )
        return error_display, {}, {}


def handle_import_settings():
    """Handle importing settings."""
    try:
        # This would typically involve file upload handling
        # For now, we'll return a message about the feature
        status_display = create_info_display(
            "Import Feature",
            "Settings import functionality requires file upload integration.",
        )
        return status_display, {}, {}

    except Exception as e:
        logger.error(f"Error handling import: {e}")
        error_display = create_error_display(
            "Import Error", f"Failed to handle import: {str(e)}"
        )
        return error_display, {}, {}


def create_settings_status_display():
    """Create comprehensive settings status display."""
    try:
        service = get_settings_service()
        summary = service.get_settings_summary()

        # Get system health
        system_health = SystemMonitor.get_system_health()

        return html.Div(
            [
                html.H6("📊 Current Settings Status", className="text-primary mb-3"),
                # General Status
                html.Div(
                    [
                        html.H6("🌐 General", className="text-info"),
                        html.P(f"Theme: {summary['general']['theme'].title()}"),
                        html.P(f"Page Size: {summary['general']['page_size']} items"),
                        html.P(f"Auto-refresh: {summary['general']['auto_refresh']}s"),
                    ],
                    className="mb-3",
                ),
                # Analysis Status
                html.Div(
                    [
                        html.H6("📊 Analysis", className="text-success"),
                        html.P(
                            f"Sampling Freq: {summary['analysis']['sampling_freq']} Hz"
                        ),
                        html.P(f"FFT Points: {summary['analysis']['fft_points']}"),
                        html.P(
                            f"Window Type: {summary['analysis']['window_type'].title()}"
                        ),
                    ],
                    className="mb-3",
                ),
                # System Health
                html.Div(
                    [
                        html.H6(
                            "🖥️ System Health",
                            className=f"text-{system_health.get('color', 'info')}",
                        ),
                        html.P(
                            f"Overall Score: {system_health.get('overall_score', 'N/A')}/100"
                        ),
                        html.P(f"Status: {system_health.get('status', 'N/A').title()}"),
                        html.P(
                            f"Memory: {system_health.get('memory_score', 'N/A')}/100"
                        ),
                        html.P(f"CPU: {system_health.get('cpu_score', 'N/A')}/100"),
                    ],
                    className="mb-3",
                ),
                html.Small(
                    f"Last Updated: {summary['last_updated']}", className="text-muted"
                ),
            ]
        )

    except Exception as e:
        logger.error(f"Error creating status display: {e}")
        return create_error_display("Status Error", str(e))


def create_system_monitor_display():
    """Create real-time system monitoring display."""
    try:
        # Get system information
        system_info = SystemMonitor.get_system_info()
        memory_info = SystemMonitor.get_memory_info()
        cpu_info = SystemMonitor.get_cpu_info()
        disk_info = SystemMonitor.get_disk_info()
        system_health = SystemMonitor.get_system_health()

        return html.Div(
            [
                html.H6("🖥️ Real-time System Monitor", className="text-primary mb-3"),
                # System Information
                html.Div(
                    [
                        html.H6("System Info", className="text-info"),
                        html.P(f"Platform: {system_info.get('platform', 'N/A')}"),
                        html.P(
                            f"Architecture: {system_info.get('architecture', 'N/A')}"
                        ),
                        html.P(f"Python: {system_info.get('python_version', 'N/A')}"),
                    ],
                    className="mb-3",
                ),
                # Memory Usage
                html.Div(
                    [
                        html.H6("Memory Usage", className="text-warning"),
                        html.P(f"Total: {memory_info.get('total_gb', 0)} GB"),
                        html.P(
                            f"Used: {memory_info.get('used_gb', 0)} GB ({memory_info.get('percent_used', 0)}%)"
                        ),
                        html.P(f"Available: {memory_info.get('available_gb', 0)} GB"),
                    ],
                    className="mb-3",
                ),
                # CPU Usage
                html.Div(
                    [
                        html.H6("CPU Usage", className="text-danger"),
                        html.P(f"Usage: {cpu_info.get('usage_percent', 0)}%"),
                        html.P(f"Cores: {cpu_info.get('core_count', 0)}"),
                        html.P(f"Frequency: {cpu_info.get('frequency_mhz', 0)} MHz"),
                    ],
                    className="mb-3",
                ),
                # Disk Usage
                html.Div(
                    [
                        html.H6("Disk Usage", className="text-success"),
                        html.P(f"Total: {disk_info.get('total_gb', 0)} GB"),
                        html.P(
                            f"Used: {disk_info.get('used_gb', 0)} GB ({disk_info.get('percent_used', 0)}%)"
                        ),
                        html.P(f"Free: {disk_info.get('free_gb', 0)} GB"),
                    ],
                    className="mb-3",
                ),
                # System Health
                html.Div(
                    [
                        html.H6(
                            f"System Health: {system_health.get('status', 'N/A').title()}",
                            className=f"text-{system_health.get('color', 'info')}",
                        ),
                        html.P(
                            f"Overall Score: {system_health.get('overall_score', 'N/A')}/100"
                        ),
                        html.Small(
                            f"Updated: {system_health.get('timestamp', 'N/A')}",
                            className="text-muted",
                        ),
                    ]
                ),
            ]
        )

    except Exception as e:
        logger.error(f"Error creating system monitor display: {e}")
        return create_error_display("System Monitor Error", str(e))


def create_settings_validation_display():
    """Create settings validation display."""
    try:
        service = get_settings_service()
        settings = service.get_all_settings()

        # Validate all settings
        general_validation = SettingsValidator.validate_general_settings(
            {
                "theme": settings.general.theme,
                "page_size": settings.general.page_size,
                "auto_refresh_interval": settings.general.auto_refresh_interval,
            }
        )

        analysis_validation = SettingsValidator.validate_analysis_settings(
            {
                "default_sampling_freq": settings.analysis.default_sampling_freq,
                "default_fft_points": settings.analysis.default_fft_points,
                "peak_threshold": settings.analysis.peak_threshold,
            }
        )

        data_validation = SettingsValidator.validate_data_settings(
            {
                "max_file_size": settings.data.max_file_size,
                "auto_save_interval": settings.data.auto_save_interval,
                "data_retention_days": settings.data.data_retention_days,
            }
        )

        system_validation = SettingsValidator.validate_system_settings(
            {
                "max_cpu_usage": settings.system.max_cpu_usage,
                "memory_limit_gb": settings.system.memory_limit_gb,
                "parallel_threads": settings.system.parallel_threads,
            }
        )

        return html.Div(
            [
                html.H6(
                    "✅ Settings Validation Results", className="text-primary mb-3"
                ),
                # General Settings Validation
                html.Div(
                    [
                        html.H6(
                            "🌐 General Settings",
                            className=f"text-{'success' if general_validation['valid'] else 'danger'}",
                        ),
                        html.Div(
                            [
                                html.P(
                                    f"Status: {'✅ Valid' if general_validation['valid'] else '❌ Invalid'}"
                                ),
                                *[
                                    html.P(f"❌ Error: {error}")
                                    for error in general_validation.get("errors", [])
                                ],
                                *[
                                    html.P(f"⚠️ Warning: {warning}")
                                    for warning in general_validation.get(
                                        "warnings", []
                                    )
                                ],
                                *[
                                    html.P(f"💡 Recommendation: {rec}")
                                    for rec in general_validation.get(
                                        "recommendations", []
                                    )
                                ],
                            ]
                        ),
                    ],
                    className="mb-3",
                ),
                # Analysis Settings Validation
                html.Div(
                    [
                        html.H6(
                            "📊 Analysis Settings",
                            className=f"text-{'success' if analysis_validation['valid'] else 'danger'}",
                        ),
                        html.Div(
                            [
                                html.P(
                                    f"Status: {'✅ Valid' if analysis_validation['valid'] else '❌ Invalid'}"
                                ),
                                *[
                                    html.P(f"❌ Error: {error}")
                                    for error in analysis_validation.get("errors", [])
                                ],
                                *[
                                    html.P(f"⚠️ Warning: {warning}")
                                    for warning in analysis_validation.get(
                                        "warnings", []
                                    )
                                ],
                                *[
                                    html.P(f"💡 Recommendation: {rec}")
                                    for rec in analysis_validation.get(
                                        "recommendations", []
                                    )
                                ],
                            ]
                        ),
                    ],
                    className="mb-3",
                ),
                # Data Settings Validation
                html.Div(
                    [
                        html.H6(
                            "💾 Data Settings",
                            className=f"text-{'success' if data_validation['valid'] else 'danger'}",
                        ),
                        html.Div(
                            [
                                html.P(
                                    f"Status: {'✅ Valid' if data_validation['valid'] else '❌ Invalid'}"
                                ),
                                *[
                                    html.P(f"❌ Error: {error}")
                                    for error in data_validation.get("errors", [])
                                ],
                                *[
                                    html.P(f"⚠️ Warning: {warning}")
                                    for warning in data_validation.get("warnings", [])
                                ],
                                *[
                                    html.P(f"💡 Recommendation: {rec}")
                                    for rec in data_validation.get(
                                        "recommendations", []
                                    )
                                ],
                            ]
                        ),
                    ],
                    className="mb-3",
                ),
                # System Settings Validation
                html.Div(
                    [
                        html.H6(
                            "🖥️ System Settings",
                            className=f"text-{'success' if system_validation['valid'] else 'danger'}",
                        ),
                        html.Div(
                            [
                                html.P(
                                    f"Status: {'✅ Valid' if system_validation['valid'] else '❌ Invalid'}"
                                ),
                                *[
                                    html.P(f"❌ Error: {error}")
                                    for error in system_validation.get("errors", [])
                                ],
                                *[
                                    html.P(f"⚠️ Warning: {warning}")
                                    for warning in system_validation.get("warnings", [])
                                ],
                                *[
                                    html.P(f"💡 Recommendation: {rec}")
                                    for rec in system_validation.get(
                                        "recommendations", []
                                    )
                                ],
                            ]
                        ),
                    ]
                ),
            ]
        )

    except Exception as e:
        logger.error(f"Error creating validation display: {e}")
        return create_error_display("Validation Error", str(e))


def create_theme_preview_display(theme):
    """Create theme preview display."""
    try:
        theme_info = ThemeManager.get_theme_preview(theme)
        colors = ThemeManager.get_theme_colors(theme)

        return html.Div(
            [
                html.H6(
                    f"🎨 {theme.title()} Theme Preview", className="text-primary mb-3"
                ),
                # Color preview
                html.Div(
                    [
                        html.H6("Color Scheme", className="text-info"),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            style={
                                                "width": "20px",
                                                "height": "20px",
                                                "backgroundColor": color,
                                                "border": "1px solid #ccc",
                                                "display": "inline-block",
                                                "marginRight": "10px",
                                            }
                                        ),
                                        html.Span(
                                            f"{key.replace('_', ' ').title()}: {color}"
                                        ),
                                    ],
                                    className="mb-2",
                                )
                                for key, color in colors.items()
                            ]
                        ),
                    ],
                    className="mb-3",
                ),
                html.P(
                    f"Description: {theme_info['description']}", className="text-muted"
                ),
            ]
        )

    except Exception as e:
        logger.error(f"Error creating theme preview: {e}")
        return create_error_display("Theme Preview Error", str(e))


def create_settings_recommendations_display():
    """Create settings recommendations display."""
    try:
        # Get system recommendations
        system_recs = get_system_recommendations()

        # Get current settings for context
        service = get_settings_service()
        current_settings = service.get_all_settings()

        recommendations = []

        # General recommendations
        if current_settings.general.auto_refresh_interval == 0:
            recommendations.append(
                "Consider enabling auto-refresh for better user experience"
            )

        if current_settings.general.page_size > 50:
            recommendations.append(
                "Large page sizes may impact performance on slower devices"
            )

        # Analysis recommendations
        if current_settings.analysis.default_fft_points < 1024:
            recommendations.append(
                "Consider increasing FFT points for better frequency resolution"
            )

        if current_settings.analysis.default_sampling_freq < 500:
            recommendations.append("Low sampling frequency may limit analysis accuracy")

        # System recommendations
        if current_settings.system.parallel_threads > 8:
            recommendations.append(
                "High parallel processing may impact system responsiveness"
            )

        if current_settings.system.memory_limit_gb > 8:
            recommendations.append("High memory limits may impact system stability")

        # Add system-specific recommendations
        for category, recs in system_recs.items():
            recommendations.extend(recs)

        return html.Div(
            [
                html.H6("💡 Settings Recommendations", className="text-primary mb-3"),
                html.Div(
                    [
                        html.H6("General Recommendations", className="text-info"),
                        html.Ul(
                            [
                                html.Li(rec)
                                for rec in recommendations[:5]  # Show first 5
                            ]
                        ),
                    ],
                    className="mb-3",
                ),
                html.Div(
                    [
                        html.H6("Performance Tips", className="text-success"),
                        html.Ul(
                            [
                                html.Li(
                                    "Use appropriate FFT sizes for your signal characteristics"
                                ),
                                html.Li(
                                    "Balance parallel processing with system resources"
                                ),
                                html.Li(
                                    "Consider data retention policies for storage optimization"
                                ),
                                html.Li("Monitor system health regularly"),
                            ]
                        ),
                    ],
                    className="mb-3",
                ),
                html.Small(
                    "Recommendations are based on current system capabilities and best practices",
                    className="text-muted",
                ),
            ]
        )

    except Exception as e:
        logger.error(f"Error creating recommendations: {e}")
        return create_error_display("Recommendations Error", str(e))


def create_success_display(title, message):
    """Create success message display."""
    return html.Div(
        [
            html.H6(f"✅ {title}", className="text-success"),
            html.P(message, className="text-success"),
            html.Small("Settings updated successfully", className="text-muted"),
        ]
    )


def create_error_display(title, message):
    """Create error message display."""
    return html.Div(
        [
            html.H6(f"❌ {title}", className="text-danger"),
            html.P(message, className="text-danger"),
            html.Small(
                "Please check the settings and try again", className="text-muted"
            ),
        ]
    )


def create_info_display(title, message):
    """Create info message display."""
    return html.Div(
        [
            html.H6(f"ℹ️ {title}", className="text-info"),
            html.P(message, className="text-info"),
            html.Small("Information message", className="text-muted"),
        ]
    )
