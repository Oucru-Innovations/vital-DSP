from dash import html
from webapp.layout.upload_section import upload_layout  # Import your upload layout


def MainContent():
    main_content = html.Div(
        [
            upload_layout,
            # You can add more sections or use Dash's `dcc.Location` for multi-page apps
        ],
        style={
            "margin-left": "18rem",  # Width of the sidebar + some padding
            "margin-right": "2rem",
            "padding": "2rem 1rem",
        },
    )
    return main_content
