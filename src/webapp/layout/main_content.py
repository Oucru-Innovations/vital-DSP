from dash import html
from webapp.layout.upload_section import upload_layout  # Import your upload layout


def MainContent():
    """
    Generates the main content area of the Dash web application.

    Returns
    -------
    html.Div
        A Dash HTML Div component containing the upload section and more if needed.

    Example
    -------
    >>> main_content = MainContent()
    """
    main_content = html.Div(
        [
            upload_layout,  # The layout section for file uploads
        ],
        style={
            "margin-left": "18rem",  # Adjust for sidebar width
            "margin-right": "2rem",
            "padding": "2rem 1rem",
        },
    )
    return main_content
