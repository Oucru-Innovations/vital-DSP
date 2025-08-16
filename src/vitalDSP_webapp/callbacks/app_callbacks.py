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
            
        if n_clicks % 2 == 0:
            # Expanded state
            return (
                "sidebar sidebar-expanded",
                "fas fa-bars"
            )
        else:
            # Collapsed state
            return (
                "sidebar sidebar-collapsed",
                "fas fa-arrow-right"
            )
