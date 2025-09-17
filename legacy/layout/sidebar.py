from dash import html
import dash_bootstrap_components as dbc


def Sidebar():
    """
    Generates the sidebar component for navigation and filtering in the Dash web application.
    Features expandable/collapsible functionality with smooth animations.

    Returns
    -------
    html.Div
        A Dash HTML Div component containing navigation links and filters in the sidebar.
    """
    sidebar = html.Div(
        [
            # Header section with toggle button and title
            html.Div(
                [
                    # Toggle button
                    dbc.Button(
                        html.I(className="fas fa-bars", id="sidebar-toggle-icon"),
                        id="sidebar-toggle",
                        n_clicks=0,
                        className="sidebar-toggle-btn",
                        color="light",
                        outline=True,
                        size="sm"
                    ),
                    # Title (hidden when collapsed)
                    html.Div(
                        "VitalDSP",
                        id="sidebar-title",
                        className="sidebar-title"
                    )
                ],
                className="sidebar-header"
            ),
            
            # Navigation items
            html.Div(
                [
                    # Home
                    dbc.NavLink(
                        [
                            html.I(className="fas fa-home sidebar-icon"),
                            html.Span("Home", className="sidebar-text")
                        ],
                        href="/",
                        active="exact",
                        className="sidebar-nav-item",
                    ),
                    # Upload
                    dbc.NavLink(
                        [
                            html.I(className="fas fa-upload sidebar-icon"),
                            html.Span("Upload", className="sidebar-text")
                        ],
                        href="/upload",
                        active="exact",
                        className="sidebar-nav-item",
                    ),
<<<<<<< HEAD
                    # Filtering
                    dbc.NavLink(
                        [
                            html.I(className="fas fa-filter sidebar-icon"),
                            html.Span("Filtering", className="sidebar-text")
                        ],
                        href="/filtering",
                        active="exact",
                        className="sidebar-nav-item",
                    ),
=======
>>>>>>> eccdf5be7a36e20173cd38c06df45f532d4e73b2
                    # Time Domain
                    dbc.NavLink(
                        [
                            html.I(className="fas fa-chart-line sidebar-icon"),
                            html.Span("Time Domain", className="sidebar-text")
                        ],
                        href="/time-domain",
                        active="exact",
                        className="sidebar-nav-item",
                    ),
                    # Frequency
                    dbc.NavLink(
                        [
                            html.I(className="fas fa-wave-square sidebar-icon"),
                            html.Span("Frequency", className="sidebar-text")
                        ],
                        href="/frequency",
                        active="exact",
                        className="sidebar-nav-item",
                    ),
<<<<<<< HEAD
                    
=======
                    # Filtering
                    dbc.NavLink(
                        [
                            html.I(className="fas fa-filter sidebar-icon"),
                            html.Span("Filtering", className="sidebar-text")
                        ],
                        href="/filtering",
                        active="exact",
                        className="sidebar-nav-item",
                    ),
>>>>>>> eccdf5be7a36e20173cd38c06df45f532d4e73b2
                    # Physiological
                    dbc.NavLink(
                        [
                            html.I(className="fas fa-heartbeat sidebar-icon"),
                            html.Span("Physiological", className="sidebar-text")
                        ],
                        href="/physiological",
                        active="exact",
                        className="sidebar-nav-item",
                    ),
                    # Respiratory
                    dbc.NavLink(
                        [
                            html.I(className="fas fa-lungs sidebar-icon"),
                            html.Span("Respiratory", className="sidebar-text")
                        ],
                        href="/respiratory",
                        active="exact",
                        className="sidebar-nav-item",
                    ),
                    # Features
                    dbc.NavLink(
                        [
                            html.I(className="fas fa-chart-bar sidebar-icon"),
                            html.Span("Features", className="sidebar-text")
                        ],
                        href="/features",
                        active="exact",
                        className="sidebar-nav-item",
                    ),
                    # Transforms
                    dbc.NavLink(
                        [
                            html.I(className="fas fa-magic sidebar-icon"),
                            html.Span("Transforms", className="sidebar-text")
                        ],
                        href="/transforms",
                        active="exact",
                        className="sidebar-nav-item",
                    ),
                    # Quality
                    dbc.NavLink(
                        [
                            html.I(className="fas fa-shield-alt sidebar-icon"),
                            html.Span("Quality", className="sidebar-text")
                        ],
                        href="/quality",
                        active="exact",
                        className="sidebar-nav-item",
                    ),
                    # Advanced
                    dbc.NavLink(
                        [
                            html.I(className="fas fa-brain sidebar-icon"),
                            html.Span("Advanced", className="sidebar-text")
                        ],
                        href="/advanced",
                        active="exact",
                        className="sidebar-nav-item",
                    ),
                    # Health Report
                    dbc.NavLink(
                        [
                            html.I(className="fas fa-file-medical sidebar-icon"),
                            html.Span("Health Report", className="sidebar-text")
                        ],
                        href="/health-report",
                        active="exact",
                        className="sidebar-nav-item",
                    ),
                    # Settings
                    dbc.NavLink(
                        [
                            html.I(className="fas fa-cog sidebar-icon"),
                            html.Span("Settings", className="sidebar-text")
                        ],
                        href="/settings",
                        active="exact",
                        className="sidebar-nav-item",
                    ),
                ],
                className="sidebar-nav"
            ),
        ],
        id="sidebar",
        className="sidebar sidebar-expanded"
    )
    return sidebar
