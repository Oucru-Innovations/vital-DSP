from dash.dependencies import Input, Output


def register_sidebar_callbacks(app):
    """
    Registers the callback to toggle the sidebar between expanded and collapsed states.
    Also toggles the icon between the hamburger icon and a right-pointing arrow.
    """

    @app.callback(
        [
            Output("sidebar", "className"),  # Toggle sidebar class
            Output("toggle-icon", "className"),  # Toggle icon class
            Output("page-content", "style"),
        ],  # Adjust content margin based on sidebar state
        [Input("sidebar-toggle", "n_clicks")],
    )
    def toggle_sidebar(n_clicks):
        """
        Toggles the sidebar's width and the toggle button icon.
        """
        if n_clicks % 2 == 0:
            return (
                "sidebar-expanded",
                "fas fa-bars",
                {"margin-left": "250px", "padding": "2rem 1rem"},
            )  # Expanded
        else:
            return (
                "sidebar-collapsed",
                "fas fa-arrow-right",
                {"margin-left": "80px", "padding": "2rem 1rem"},
            )  # Collapsed
