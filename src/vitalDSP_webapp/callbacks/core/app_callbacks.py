"""
Core application callbacks for vitalDSP webapp.

This module handles basic application functionality like sidebar toggling.
"""

from dash.dependencies import Input, Output


def register_sidebar_callbacks(app):
    """
    Registers the callback to toggle the sidebar between expanded and collapsed states.
    Also toggles the icon between the hamburger icon and a right-pointing arrow.
    """

    @app.callback(
        [
            Output("sidebar", "className"),  # Toggle sidebar class
            Output("sidebar-toggle-icon", "className"),  # Toggle icon class
        ],
        [Input("sidebar-toggle", "n_clicks")],
    )
    def toggle_sidebar(n_clicks):
        """
        Toggles the sidebar's width and the toggle button icon.
        """
        if n_clicks is None:
            n_clicks = 0
            
        print(f"Sidebar toggle clicked {n_clicks} times")
        
        if n_clicks % 2 == 0:
            # Expanded state
            print("Setting sidebar to expanded state")
            return (
                "sidebar sidebar-expanded",
                "fas fa-bars"
            )
        else:
            # Collapsed state
            print("Setting sidebar to collapsed state")
            return (
                "sidebar sidebar-collapsed",
                "fas fa-arrow-right"
            )

    @app.callback(
        Output("page-content", "style"),
        [Input("sidebar", "className")]
    )
    def adjust_page_content_position(sidebar_class):
        """
        Adjusts the page content position based on sidebar state.
        """
        from vitalDSP_webapp.config.settings import app_config
        
        if "sidebar-collapsed" in sidebar_class:
            # Sidebar is collapsed, use collapsed width
            left_position = f"{app_config.SIDEBAR_COLLAPSED_WIDTH}px"
        else:
            # Sidebar is expanded, use full width
            left_position = f"{app_config.SIDEBAR_WIDTH}px"
        
        return {
            "position": "absolute",
            "top": f"{app_config.HEADER_HEIGHT}px",
            "left": left_position,
            "right": "0",
            "padding": "2rem",
            "backgroundColor": "#ffffff",
            "minHeight": f"calc(100vh - {app_config.HEADER_HEIGHT}px)",
            "zIndex": 100,
            "transition": "left 0.3s cubic-bezier(0.4, 0, 0.2, 1)"
        }
