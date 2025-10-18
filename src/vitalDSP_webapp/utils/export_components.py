"""
Export UI components for vitalDSP webapp pages.

This module provides reusable export button groups and download components.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


def create_export_buttons(page_id: str, button_text: str = "Export Results") -> dbc.Row:
    """
    Create export buttons (CSV and JSON) with download components.

    Args:
        page_id: Unique identifier for the page (e.g., 'filtered', 'time-domain')
        button_text: Text to display on the button group

    Returns:
        dbc.Row: Row containing export buttons and download components
    """
    return dbc.Row(
        [
            dbc.Col(
                [
                    dbc.ButtonGroup(
                        [
                            dbc.Button(
                                [
                                    html.I(className="fas fa-file-csv me-2"),
                                    "CSV",
                                ],
                                id=f"btn-export-{page_id}-csv",
                                color="success",
                                outline=True,
                                size="sm",
                            ),
                            dbc.Button(
                                [
                                    html.I(className="fas fa-file-code me-2"),
                                    "JSON",
                                ],
                                id=f"btn-export-{page_id}-json",
                                color="info",
                                outline=True,
                                size="sm",
                            ),
                        ],
                        size="sm",
                    ),
                    # Download components
                    dcc.Download(id=f"download-{page_id}-csv"),
                    dcc.Download(id=f"download-{page_id}-json"),
                ],
                width="auto",
            ),
        ],
        className="mb-3",
    )


def create_export_card(page_id: str, title: str = "Export Results") -> dbc.Card:
    """
    Create an export card with CSV and JSON export options.

    Args:
        page_id: Unique identifier for the page
        title: Title for the export card

    Returns:
        dbc.Card: Card containing export options
    """
    return dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.I(className="fas fa-download me-2 text-success"),
                    html.Span(title, className="fw-bold"),
                ],
                className="bg-light border-0",
            ),
            dbc.CardBody(
                [
                    html.P(
                        "Export your analysis results in your preferred format:",
                        className="text-muted mb-3",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Button(
                                        [
                                            html.I(
                                                className="fas fa-file-csv fa-2x mb-2"
                                            ),
                                            html.Br(),
                                            "Export CSV",
                                        ],
                                        id=f"btn-export-{page_id}-csv",
                                        color="success",
                                        className="w-100 py-3",
                                    ),
                                    html.Small(
                                        "Comma-separated values",
                                        className="text-muted d-block mt-2 text-center",
                                    ),
                                ],
                                md=6,
                                className="mb-3 mb-md-0",
                            ),
                            dbc.Col(
                                [
                                    dbc.Button(
                                        [
                                            html.I(
                                                className="fas fa-file-code fa-2x mb-2"
                                            ),
                                            html.Br(),
                                            "Export JSON",
                                        ],
                                        id=f"btn-export-{page_id}-json",
                                        color="info",
                                        className="w-100 py-3",
                                    ),
                                    html.Small(
                                        "JavaScript Object Notation",
                                        className="text-muted d-block mt-2 text-center",
                                    ),
                                ],
                                md=6,
                            ),
                        ]
                    ),
                    # Download components
                    dcc.Download(id=f"download-{page_id}-csv"),
                    dcc.Download(id=f"download-{page_id}-json"),
                ],
            ),
        ],
        className="shadow-sm border-0 mb-4",
    )


def create_inline_export_buttons(page_id: str) -> html.Div:
    """
    Create inline export buttons suitable for placing in action bars.

    Args:
        page_id: Unique identifier for the page

    Returns:
        html.Div: Div containing export buttons and download components
    """
    return html.Div(
        [
            dbc.ButtonGroup(
                [
                    dbc.Button(
                        [
                            html.I(className="fas fa-file-csv me-1"),
                            "CSV",
                        ],
                        id=f"btn-export-{page_id}-csv",
                        color="success",
                        outline=True,
                        size="sm",
                    ),
                    dbc.Button(
                        [
                            html.I(className="fas fa-file-code me-1"),
                            "JSON",
                        ],
                        id=f"btn-export-{page_id}-json",
                        color="info",
                        outline=True,
                        size="sm",
                    ),
                ],
                size="sm",
                className="me-2",
            ),
            dcc.Download(id=f"download-{page_id}-csv"),
            dcc.Download(id=f"download-{page_id}-json"),
        ],
        className="d-inline-block",
    )


def create_dropdown_export_menu(page_id: str) -> dbc.DropdownMenu:
    """
    Create a dropdown menu for export options.

    Args:
        page_id: Unique identifier for the page

    Returns:
        dbc.DropdownMenu: Dropdown menu with export options
    """
    return html.Div(
        [
            dbc.DropdownMenu(
                [
                    dbc.DropdownMenuItem(
                        [
                            html.I(className="fas fa-file-csv me-2"),
                            "Export as CSV",
                        ],
                        id=f"btn-export-{page_id}-csv",
                    ),
                    dbc.DropdownMenuItem(
                        [
                            html.I(className="fas fa-file-code me-2"),
                            "Export as JSON",
                        ],
                        id=f"btn-export-{page_id}-json",
                    ),
                ],
                label="Export",
                color="success",
                size="sm",
                className="me-2",
            ),
            dcc.Download(id=f"download-{page_id}-csv"),
            dcc.Download(id=f"download-{page_id}-json"),
        ],
        className="d-inline-block",
    )


def create_export_section_with_preview(
    page_id: str, title: str = "Export & Download"
) -> dbc.Card:
    """
    Create an export section with preview and download options.

    Args:
        page_id: Unique identifier for the page
        title: Title for the section

    Returns:
        dbc.Card: Card with export section
    """
    return dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.I(className="fas fa-download me-2 text-primary"),
                    html.Span(title, className="fw-bold"),
                ],
                className="bg-light border-0",
            ),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Div(
                                        id=f"{page_id}-export-preview",
                                        className="border rounded p-3 mb-3",
                                        style={
                                            "max-height": "200px",
                                            "overflow-y": "auto",
                                        },
                                    ),
                                ],
                                md=12,
                            ),
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label(
                                        "Export Format:", className="fw-semibold mb-2"
                                    ),
                                    dbc.RadioItems(
                                        id=f"{page_id}-export-format",
                                        options=[
                                            {
                                                "label": " CSV (Spreadsheet)",
                                                "value": "csv",
                                            },
                                            {
                                                "label": " JSON (Structured Data)",
                                                "value": "json",
                                            },
                                        ],
                                        value="csv",
                                        inline=True,
                                        className="mb-3",
                                    ),
                                ],
                                md=6,
                            ),
                            dbc.Col(
                                [
                                    html.Label("​", className="d-block mb-2"),  # Spacer
                                    dbc.Button(
                                        [
                                            html.I(className="fas fa-download me-2"),
                                            "Download",
                                        ],
                                        id=f"btn-export-{page_id}-download",
                                        color="primary",
                                        className="w-100",
                                    ),
                                ],
                                md=6,
                            ),
                        ]
                    ),
                    dcc.Download(id=f"download-{page_id}-csv"),
                    dcc.Download(id=f"download-{page_id}-json"),
                ],
            ),
        ],
        className="shadow-sm border-0 mb-4",
    )
