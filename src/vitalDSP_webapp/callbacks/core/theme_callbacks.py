from dash import Input, Output, State, html, callback_context, no_update
from dash.exceptions import PreventUpdate
import logging
import json

logger = logging.getLogger(__name__)

def apply_plot_theme(fig, theme):
    """Applies a theme to a Plotly figure."""
    if theme == "dark":
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="#1a1a1a",
            paper_bgcolor="#1a1a1a",
            font=dict(color="#ffffff"),
            xaxis=dict(gridcolor="#404040"),
            yaxis=dict(gridcolor="#404040"),
        )
    elif theme == "light":
        fig.update_layout(
            template="plotly_white",
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font=dict(color="#2c3e50"),
            xaxis=dict(gridcolor="#e0e0e0"),
            yaxis=dict(gridcolor="#e0e0e0"),
        )
    else: # auto or default
        fig.update_layout(
            template="plotly_white",
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font=dict(color="#2c3e50"),
            xaxis=dict(gridcolor="#e0e0e0"),
            yaxis=dict(gridcolor="#e0e0e0"),
        )
    return fig

def get_theme_from_settings():
    """Get theme from settings service."""
    try:
        from vitalDSP_webapp.services.settings_service import SettingsService
        service = SettingsService()
        settings = service.get_general_settings()
        theme = settings.theme if settings and hasattr(settings, 'theme') else "light"
        logger.info(f"Loaded theme from settings: {theme}")
        return theme
    except Exception as e:
        logger.error(f"Error loading theme from settings: {e}")
        return "light"

def register_theme_callbacks(app):
    """Register all theme switching callbacks."""
    logger.info("=== REGISTERING THEME CALLBACKS ===")

    # Main theme callback - handles initialization, navigation, and sidebar toggling
    @app.callback(
        [
            Output("body", "data-theme"),
            Output("store_theme", "data"),
            Output("theme-icon", "className"),
            Output("theme-text", "children"),
            Output("theme-icon-collapsed", "className"),
        ],
        [
            Input("url", "pathname"),
            Input("theme-toggle", "n_clicks"),
            Input("theme-toggle-collapsed", "n_clicks"),
        ],
        [
            State("store_theme", "data"),
        ],
        prevent_initial_call=False,
    )
    def main_theme_callback(pathname, expanded_clicks, collapsed_clicks, current_theme):
        """Main callback for theme operations."""
        ctx = callback_context
        if not ctx.triggered:
            # Initial load - get theme from settings
            settings_theme = get_theme_from_settings()
            theme = settings_theme if settings_theme else (current_theme if current_theme else "light")
            logger.info(f"Initial theme load: {theme}")
        else:
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
            
            if trigger_id in ["theme-toggle", "theme-toggle-collapsed"]:
                # Theme toggle button clicked
                current_theme = current_theme if current_theme else "light"
                theme = "dark" if current_theme == "light" else "light"
                logger.info(f"Theme toggled from {current_theme} to {theme}")
                
                # Save theme to settings
                try:
                    from vitalDSP_webapp.services.settings_service import SettingsService
                    service = SettingsService()
                    service.update_general_settings(theme=theme)
                    logger.info(f"Theme saved to settings: {theme}")
                except Exception as e:
                    logger.error(f"Error saving theme to settings: {e}")
            
            elif trigger_id == "url":
                # Page navigation - maintain current theme
                theme = current_theme if current_theme else "light"
                logger.info(f"Page navigation, maintaining theme: {theme}")
            
            else:
                # Default fallback
                theme = current_theme if current_theme else "light"
        
        # Update button states
        if theme == "dark":
            icon_class = "fas fa-moon me-2"
            text = "Dark"
        else:
            icon_class = "fas fa-sun me-2"
            text = "Light"
        
        return theme, theme, icon_class, text, icon_class.replace(" me-2", "")

    # Settings save callback (only when on settings page)
    @app.callback(
        [
            Output("body", "data-theme", allow_duplicate=True),
            Output("store_theme", "data", allow_duplicate=True),
            Output("theme-icon", "className", allow_duplicate=True),
            Output("theme-text", "children", allow_duplicate=True),
            Output("theme-icon-collapsed", "className", allow_duplicate=True),
        ],
        [
            Input("settings-save-btn", "n_clicks"),
        ],
        [
            State("settings-theme", "value"),
        ],
        prevent_initial_call=True,
    )
    def settings_save_callback(save_clicks, theme_value):
        """Handle theme changes when settings are saved."""
        if save_clicks and theme_value:
            logger.info(f"Theme saved from settings: {theme_value}")
            
            # Update button states
            if theme_value == "dark":
                icon_class = "fas fa-moon me-2"
                text = "Dark"
            else:
                icon_class = "fas fa-sun me-2"
                text = "Light"
            
            return theme_value, theme_value, icon_class, text, icon_class.replace(" me-2", "")
        return no_update, no_update, no_update, no_update, no_update

    # Client-side callback for immediate theme change
    app.clientside_callback(
        """
        function(theme_value) {
            if (theme_value) {
                console.log('Client-side theme change:', theme_value);
                document.body.setAttribute('data-theme', theme_value);
                // Force immediate style update
                if (theme_value === 'dark') {
                    document.body.style.backgroundColor = '#1a1a1a';
                    document.body.style.color = '#ffffff';
                } else {
                    document.body.style.backgroundColor = '#ffffff';
                    document.body.style.color = '#2c3e50';
                }
                return theme_value;
            }
            return dash_clientside.no_update;
        }
        """,
        Output("body", "data-theme", allow_duplicate=True),
        Input("store_theme", "data"),
        prevent_initial_call=True,
    )

    # Sync settings dropdown with sidebar theme
    @app.callback(
        Output("settings-theme", "value", allow_duplicate=True),
        [
            Input("store_theme", "data"),
        ],
        prevent_initial_call=True,
    )
    def sync_settings_dropdown(store_theme):
        """Sync settings dropdown with sidebar theme."""
        if store_theme:
            logger.info(f"Syncing settings dropdown to: {store_theme}")
            return store_theme
        return no_update

    # Debug callback
    @app.callback(
        Output("theme-debug", "children"),
        [
            Input("body", "data-theme"),
            Input("store_theme", "data"),
        ],
        prevent_initial_call=True,
    )
    def theme_debug_callback(body_theme, store_theme):
        """Debug callback to show all theme data."""
        debug_info = f"Body: {body_theme} | Store: {store_theme}"
        logger.info(f"Theme Debug: {debug_info}")
        return debug_info