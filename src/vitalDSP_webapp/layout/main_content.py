from dash import html, dcc
import dash_bootstrap_components as dbc
from vitalDSP_webapp.layout.upload_section import upload_layout
from vitalDSP_webapp.layout.vitaldsp_layouts import (
    time_domain_layout,
    frequency_layout,
    filtering_layout,
    physiological_layout,
    respiratory_layout,
    features_layout,
    transforms_layout,
    quality_layout,
    advanced_layout,
    health_report_layout,
    settings_layout
)


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


def create_comprehensive_layout():
    """Create a comprehensive layout for all vitalDSP features."""
    return html.Div([
        # Data Upload and Management Section
        html.Div([
            html.H2("Data Management", className="section-title"),
            upload_layout,
            html.Hr(),
            
            # Data Controls
            html.Div([
                html.H3("Data Controls"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Sampling Frequency (Hz)"),
                        dcc.Input(
                            id="fs_input",
                            type="number",
                            value=1000,
                            step=0.1,
                            min=1,
                            style={"width": "120px"}
                        ),
                    ], width=3),
                    dbc.Col([
                        html.Label("Theme"),
                        dcc.Dropdown(
                            id="theme_dropdown",
                            options=[
                                {"label": "Light", "value": "light"},
                                {"label": "Dark", "value": "dark"}
                            ],
                            value="light",
                            clearable=False,
                            style={"width": "120px"}
                        ),
                    ], width=3),
                    dbc.Col([
                        html.Label("Window Size (seconds)"),
                        dcc.Input(
                            id="window_size",
                            type="number",
                            value=10.0,
                            step=0.5,
                            min=1.0,
                            style={"width": "120px"}
                        ),
                    ], width=3),
                    dbc.Col([
                        html.Button(
                            "Process Data",
                            id="process_data_btn",
                            className="btn btn-primary",
                            n_clicks=0
                        ),
                    ], width=3),
                ]),
            ], className="control-section"),
            
            html.Hr(),
            
            # Main Analysis Tabs
            html.Div([
                dcc.Tabs([
                    dcc.Tab(label="Time Domain Analysis", value="time_domain", children=time_domain_layout()),
                    dcc.Tab(label="Frequency Analysis", value="frequency", children=frequency_layout()),
                    dcc.Tab(label="Signal Filtering", value="filtering", children=filtering_layout()),
                    dcc.Tab(label="Physiological Features", value="physiological", children=physiological_layout()),
                    dcc.Tab(label="Respiratory Analysis", value="respiratory", children=respiratory_layout()),
                    dcc.Tab(label="Feature Engineering", value="features", children=features_layout()),
                    dcc.Tab(label="Signal Transforms", value="transforms", children=transforms_layout()),
                    dcc.Tab(label="Quality Assessment", value="quality", children=quality_layout()),
                    dcc.Tab(label="Advanced Analysis", value="advanced", children=advanced_layout()),
                    dcc.Tab(label="Health Report", value="health_report", children=health_report_layout()),
                ], id="main_tabs", value="time_domain"),
            ]),
            
        ], className="main-container")
    ])
