from dash.dependencies import Input, Output
from dash import html
from vitalDSP_webapp.layout.upload_section import upload_layout


def register_page_routing_callbacks(app):
    """
    Registers the callback for page routing based on the URL path.

    Parameters
    ----------
    app : Dash
        The Dash app object where the callback is being registered.

    Example
    -------
    >>> register_page_routing_callbacks(app)
    """

    @app.callback(Output("page-content", "children"), [Input("url", "pathname")])
    def display_page(pathname: str) -> html.Div:
        """
        Callback function to dynamically update the content of the main page based on the current URL path.

        Parameters
        ----------
        pathname : str
            The current URL path to determine which page to display.

        Returns
        -------
        html.Div
            The HTML content for the selected page.

        Example
        -------
        If the URL path is '/upload', it returns the upload layout:

        >>> display_page('/upload')
        html.Div([html.H3('Upload Page')])
        """
        if pathname == "/upload":
            return upload_layout
        elif pathname == "/visualize":
            return html.Div([html.H3("Data Visualization Page")])
        elif pathname == "/settings":
            return html.Div([html.H3("Settings Page")])
        else:
            return html.Div([html.H3("Welcome to vitalDSP Dashboard")])
