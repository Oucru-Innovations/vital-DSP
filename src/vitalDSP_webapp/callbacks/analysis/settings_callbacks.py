"""
Application settings callbacks for vitalDSP webapp.

This module handles application configuration including general, data, display, system, and security/privacy settings.
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

logger = logging.getLogger(__name__)

def register_settings_callbacks(app):
    """Register all application settings callbacks."""
    logger.info("=== REGISTERING SETTINGS CALLBACKS ===")
    
    @app.callback(
        [Output("settings-general-config", "children"),
         Output("settings-data-config", "children"),
         Output("settings-display-config", "children"),
         Output("settings-system-config", "children"),
         Output("settings-security-config", "children"),
         Output("store-settings-data", "data"),
         Output("store-settings-config", "data")],
        [Input("settings-save-btn", "n_clicks"),
         Input("settings-reset-btn", "n_clicks"),
         Input("url", "pathname")],
        [State("settings-general-enabled", "value"),
         State("settings-data-enabled", "value"),
         State("settings-display-enabled", "value"),
         State("settings-system-enabled", "value"),
         State("settings-security-enabled", "value"),
         State("settings-auto-save", "value"),
         State("settings-backup-frequency", "value"),
         State("settings-theme", "value"),
         State("settings-language", "value"),
         State("settings-timezone", "value"),
         State("settings-data-retention", "value"),
         State("settings-privacy-level", "value")]
    )
    def settings_management_callback(save_clicks, reset_clicks, pathname,
                                   general_enabled, data_enabled, display_enabled,
                                   system_enabled, security_enabled, auto_save,
                                   backup_frequency, theme, language, timezone,
                                   data_retention, privacy_level):
        """Main callback for settings management."""
        ctx = callback_context
        if not ctx.triggered:
            return "", "", "", "", "", None, None
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Handle save button click
        if button_id == "settings-save-btn" and save_clicks:
            try:
                # Create settings configuration
                settings_config = {
                    "general": {
                        "enabled": general_enabled or [],
                        "auto_save": auto_save or [],
                        "backup_frequency": backup_frequency or "daily"
                    },
                    "data": {
                        "enabled": data_enabled or [],
                        "retention": data_retention or "30_days"
                    },
                    "display": {
                        "enabled": display_enabled or [],
                        "theme": theme or "light",
                        "language": language or "en",
                        "timezone": timezone or "UTC"
                    },
                    "system": {
                        "enabled": system_enabled or []
                    },
                    "security": {
                        "enabled": security_enabled or [],
                        "privacy_level": privacy_level or "standard"
                    },
                    "last_updated": datetime.now().isoformat()
                }
                
                # Create configuration displays
                general_config = create_general_settings_display(settings_config["general"])
                data_config = create_data_settings_display(settings_config["data"])
                display_config = create_display_settings_display(settings_config["display"])
                system_config = create_system_settings_display(settings_config["system"])
                security_config = create_security_settings_display(settings_config["security"])
                
                # Store settings
                stored_settings = {
                    "settings_config": settings_config,
                    "last_updated": datetime.now().isoformat()
                }
                
                stored_config = {
                    "auto_save": "enabled" if auto_save and "auto_save" in auto_save else "disabled",
                    "backup_frequency": backup_frequency or "daily",
                    "theme": theme or "light",
                    "language": language or "en"
                }
                
                return (general_config, data_config, display_config, system_config, security_config,
                        stored_settings, stored_config)
                
            except Exception as e:
                logger.error(f"Error in settings management: {e}")
                error_content = html.Div([
                    html.H5("Settings Error"),
                    html.P(f"Settings management failed: {str(e)}")
                ])
                return error_content, error_content, error_content, error_content, error_content, None, None
        
        # Handle reset button click
        elif button_id == "settings-reset-btn" and reset_clicks:
            try:
                # Reset to default settings
                default_settings = {
                    "general": {
                        "enabled": ["auto_save", "notifications"],
                        "auto_save": ["auto_save"],
                        "backup_frequency": "daily"
                    },
                    "data": {
                        "enabled": ["data_retention", "backup"],
                        "retention": "30_days"
                    },
                    "display": {
                        "enabled": ["theme", "language"],
                        "theme": "light",
                        "language": "en",
                        "timezone": "UTC"
                    },
                    "system": {
                        "enabled": ["performance", "logging"]
                    },
                    "security": {
                        "enabled": ["privacy", "encryption"],
                        "privacy_level": "standard"
                    }
                }
                
                # Create default configuration displays
                general_config = create_general_settings_display(default_settings["general"])
                data_config = create_data_settings_display(default_settings["data"])
                display_config = create_display_settings_display(default_settings["display"])
                system_config = create_system_settings_display(default_settings["system"])
                security_config = create_security_settings_display(default_settings["security"])
                
                # Store default settings
                stored_settings = {
                    "settings_config": default_settings,
                    "last_updated": datetime.now().isoformat(),
                    "reset": True
                }
                
                stored_config = {
                    "auto_save": "enabled",
                    "backup_frequency": "daily",
                    "theme": "light",
                    "language": "en"
                }
                
                return (general_config, data_config, display_config, system_config, security_config,
                        stored_settings, stored_config)
                
            except Exception as e:
                logger.error(f"Error in settings reset: {e}")
                error_content = html.Div([
                    html.H5("Settings Reset Error"),
                    html.P(f"Settings reset failed: {str(e)}")
                ])
                return error_content, error_content, error_content, error_content, error_content, None, None
        
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update


# Helper functions for settings display
def create_general_settings_display(general_config):
    """Create general settings display."""
    try:
        enabled_features = general_config.get("enabled", [])
        auto_save_enabled = "auto_save" in general_config.get("auto_save", [])
        backup_frequency = general_config.get("backup_frequency", "daily")
        
        return html.Div([
            html.H6("⚙️ General Settings"),
            html.Div([
                html.P(f"Auto-save: {'✅ Enabled' if auto_save_enabled else '❌ Disabled'}"),
                html.P(f"Backup Frequency: {backup_frequency.replace('_', ' ').title()}"),
                html.P(f"Enabled Features: {', '.join(enabled_features) if enabled_features else 'None'}")
            ])
        ])
        
    except Exception as e:
        logger.error(f"Error creating general settings display: {e}")
        return html.Div([
            html.H6("Error"),
            html.P(f"Failed to create general settings display: {str(e)}")
        ])


def create_data_settings_display(data_config):
    """Create data settings display."""
    try:
        enabled_features = data_config.get("enabled", [])
        retention = data_config.get("retention", "30_days")
        
        return html.Div([
            html.H6("💾 Data Settings"),
            html.Div([
                html.P(f"Data Retention: {retention.replace('_', ' ').title()}"),
                html.P(f"Enabled Features: {', '.join(enabled_features) if enabled_features else 'None'}")
            ])
        ])
        
    except Exception as e:
        logger.error(f"Error creating data settings display: {e}")
        return html.Div([
            html.H6("Error"),
            html.P(f"Failed to create data settings display: {str(e)}")
        ])


def create_display_settings_display(display_config):
    """Create display settings display."""
    try:
        enabled_features = display_config.get("enabled", [])
        theme = display_config.get("theme", "light")
        language = display_config.get("language", "en")
        timezone = display_config.get("timezone", "UTC")
        
        return html.Div([
            html.H6("🎨 Display Settings"),
            html.Div([
                html.P(f"Theme: {theme.title()}"),
                html.P(f"Language: {language.upper()}"),
                html.P(f"Timezone: {timezone}"),
                html.P(f"Enabled Features: {', '.join(enabled_features) if enabled_features else 'None'}")
            ])
        ])
        
    except Exception as e:
        logger.error(f"Error creating display settings display: {e}")
        return html.Div([
            html.H6("Error"),
            html.P(f"Failed to create display settings display: {str(e)}")
        ])


def create_system_settings_display(system_config):
    """Create system settings display."""
    try:
        enabled_features = system_config.get("enabled", [])
        
        return html.Div([
            html.H6("🖥️ System Settings"),
            html.Div([
                html.P(f"Enabled Features: {', '.join(enabled_features) if enabled_features else 'None'}"),
                html.P("Performance monitoring and logging configuration")
            ])
        ])
        
    except Exception as e:
        logger.error(f"Error creating system settings display: {e}")
        return html.Div([
            html.H6("Error"),
            html.P(f"Failed to create system settings display: {str(e)}")
        ])


def create_security_settings_display(security_config):
    """Create security settings display."""
    try:
        enabled_features = security_config.get("enabled", [])
        privacy_level = security_config.get("privacy_level", "standard")
        
        return html.Div([
            html.H6("🔒 Security & Privacy Settings"),
            html.Div([
                html.P(f"Privacy Level: {privacy_level.replace('_', ' ').title()}"),
                html.P(f"Enabled Features: {', '.join(enabled_features) if enabled_features else 'None'}"),
                html.P("Data encryption and privacy protection")
            ])
        ])
        
    except Exception as e:
        logger.error(f"Error creating security settings display: {e}")
        return html.Div([
            html.H6("Error"),
            html.P(f"Failed to create security settings display: {str(e)}")
        ])
