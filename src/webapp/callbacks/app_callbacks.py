from dash.dependencies import Input, Output


def register_callbacks(app):
    """
    Registers the callback functions to the app instance.

    Parameters
    ----------
    app : Dash
        The Dash app instance.
    """

    @app.callback(
        Output("sidebar-collapse", "is_open"),
        Input("sidebar-toggle", "n_clicks"),
        prevent_initial_call=True,
    )
    def toggle_sidebar(n_clicks):
        """
        Toggles the sidebar collapse when the button is clicked.

        Parameters
        ----------
        n_clicks : int
            The number of clicks on the toggle button.

        Returns
        -------
        bool
            The new state of the collapse (True for open, False for collapsed).
        """
        return not n_clicks % 2 if n_clicks else True
