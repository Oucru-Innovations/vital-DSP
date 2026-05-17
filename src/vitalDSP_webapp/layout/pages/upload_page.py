"""Upload page layout for vitalDSP webapp.

Simplified one-card flow: drop a file, pick PPG/ECG, optionally override
the sampling rate, pick the signal column from the populated dropdown,
press Process Data.  The format is auto-detected from the extension and,
for CSV, from the cell shape (OUCRU vs flat).

Visual style matches the filtering page (``filtering_page.css``):
section-label typography, 12px card radii, pill-style toggles, gradient
primary button.  All page-scoped polish lives in
``assets/upload_page.css``.
"""

from dash import dcc, html
import dash_bootstrap_components as dbc

from vitalDSP_webapp.layout.common.progress_components import (
    create_progress_interval,
    create_progress_store,
)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _page_header() -> html.Div:
    """One-line title + tight subtitle, matches filtering-page style."""
    return html.Div(
        [
            html.Div(
                [
                    html.I(className="fas fa-cloud-upload-alt me-2 text-primary"),
                    html.Span("Data Upload", className="fw-semibold"),
                ],
                className="h4 mb-0 d-flex align-items-center",
            ),
            html.Small(
                "Drop a physiological recording, pick the signal type, then press "
                "Process data.  Format and signal column are auto-detected.",
                className="text-muted",
            ),
        ],
        className="mb-3",
    )


def _file_dropzone() -> dcc.Upload:
    """The drop target.  CSS lives in upload_page.css under .upload-dropzone."""
    return dcc.Upload(
        id="upload-data",
        children=html.Div(
            [
                html.I(className="fas fa-file-import dropzone-icon"),
                html.Div(
                    [
                        html.Strong("Drag & drop"),
                        html.Span(" or ", className="text-muted"),
                        html.A("browse for a recording", className="text-primary"),
                    ]
                ),
                html.Small(
                    "CSV (flat or OUCRU row-per-second), Excel, HDF5, "
                    "Parquet, JSON, WFDB, EDF, MATLAB",
                    className="dropzone-hint",
                ),
            ]
        ),
        # The default ``style`` argument is gone — CSS handles it via the
        # ``upload-dropzone`` class on the inner div Dash renders.  Setting
        # ``className`` here applies to the wrapper Dash builds.
        className="upload-dropzone",
        multiple=False,
    )


def _quick_actions_card() -> dbc.Card:
    """Side card: synthetic-data generators (PPG / ECG).

    Replaces the older "load from file path" path — uploading is
    already covered by the drop-zone, so the side panel is now just a
    fast way to try the page out without a real recording.
    """
    return dbc.Card(
        dbc.CardBody(
            [
                html.H6(
                    [
                        html.I(className="fas fa-bolt me-2 text-warning"),
                        "Try with synthetic data",
                    ],
                    className="card-title mb-2",
                ),
                html.Small(
                    "No recording handy?  Generate a clean sample at the "
                    "current sampling rate and explore the rest of the app.",
                    className="text-muted d-block mb-3",
                ),
                dbc.Button(
                    [
                        html.I(className="fas fa-heartbeat me-1"),
                        "Synthetic PPG",
                    ],
                    id="btn-load-sample-ppg",
                    color="success",
                    outline=True,
                    size="sm",
                    className="w-100 mb-2",
                ),
                dbc.Button(
                    [
                        html.I(className="fas fa-wave-square me-1"),
                        "Synthetic ECG",
                    ],
                    id="btn-load-sample-ecg",
                    color="info",
                    outline=True,
                    size="sm",
                    className="w-100",
                ),
                # Hidden legacy carriers for IDs that older callbacks or
                # tests may still reference.  Inert — no callback writes
                # to them anymore.
                html.Div(
                    [
                        dbc.Input(id="file-path-input", type="text", value=""),
                        html.Button(id="btn-load-path"),
                        html.Button(id="btn-load-sample"),
                        html.Div(id="file-path-loading"),
                    ],
                    style={"display": "none"},
                ),
            ]
        ),
        className="h-100 shadow-sm border-0 quick-actions-card",
    )


def _config_row() -> dbc.Row:
    """Inline config: signal type (pill toggle) + sampling rate + status chip."""
    return dbc.Row(
        [
            dbc.Col(
                [
                    html.Label(
                        "Signal type",
                        className="form-label section-label mb-1",
                    ),
                    # Pill-style toggle, same idiom as filter-family on
                    # the filtering page.
                    dbc.RadioItems(
                        id="data-type",
                        options=[
                            {"label": "PPG", "value": "ppg"},
                            {"label": "ECG", "value": "ecg"},
                            {"label": "EEG", "value": "eeg"},
                            {"label": "Respiratory", "value": "resp"},
                            {"label": "Other", "value": "other"},
                        ],
                        value="ppg",
                        inline=True,
                        className="signal-type-pills",
                        inputClassName="btn-check",
                        labelClassName="btn",
                        labelCheckedClassName="active",
                    ),
                ],
                md=6,
            ),
            dbc.Col(
                [
                    html.Label(
                        "Sampling rate (Hz)",
                        className="form-label section-label mb-1",
                    ),
                    dbc.Input(
                        id="sampling-freq",
                        type="number",
                        min=1,
                        step=1,
                        placeholder="auto · 1000 for synthetic",
                        size="sm",
                    ),
                ],
                md=3,
            ),
            dbc.Col(
                [
                    html.Label(
                        "Status",
                        className="form-label section-label mb-1",
                    ),
                    html.Div(
                        id="upload-status",
                        className="small text-muted py-1",
                    ),
                ],
                md=3,
            ),
        ],
        className="mb-3 g-3",
    )


def _signal_column_row() -> dbc.Row:
    """Hidden until a file is staged; then shows one dropdown with rich labels."""
    return dbc.Row(
        [
            dbc.Col(
                [
                    html.Label(
                        "Signal column",
                        className="form-label section-label mb-1",
                    ),
                    dcc.Dropdown(
                        id="signal-column",
                        placeholder="Drop a file first",
                        clearable=False,
                    ),
                    html.Small(
                        "Pick which column to process.  The recommended choice is "
                        "selected automatically based on the signal type.",
                        className="text-muted",
                    ),
                ],
                md=12,
            ),
        ],
        id="signal-column-row",
        style={"display": "none"},
        className="mb-3",
    )


def _action_row() -> html.Div:
    return html.Div(
        [
            dbc.Button(
                [html.I(className="fas fa-play me-1"), "Process data"],
                id="btn-process-data",
                color="primary",
                disabled=True,
                className="process-btn",
            ),
            html.Span(
                id="processing-status",
                className="ms-3 text-muted small",
            ),
            # Hidden carrier for the legacy ``btn-cancel-process`` ID.
            # The background-callback path that used it was removed, but
            # we keep the element so any orphan State / Input reads
            # still resolve at registration time.
            html.Button(
                id="btn-cancel-process",
                style={"display": "none"},
            ),
        ],
        className="d-flex align-items-center mb-0",
    )


def _data_preview_card() -> dbc.Card:
    return dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.Span(
                        "Data preview",
                        className="fw-semibold small text-uppercase text-muted",
                    ),
                ],
                className="bg-white border-0 pb-1",
            ),
            dbc.CardBody(
                dcc.Loading(
                    id="data-preview-loading",
                    type="default",
                    children=html.Div(
                        id="data-preview-section",
                        className="small text-muted",
                        children="Process a file to see the preview here.",
                    ),
                )
            ),
        ],
        className="mb-4 shadow-sm border-0",
    )


# ---------------------------------------------------------------------------
# Page composition
# ---------------------------------------------------------------------------


def upload_layout() -> html.Div:
    return html.Div(
        [
            _page_header(),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    _file_dropzone(),
                                    _config_row(),
                                    _signal_column_row(),
                                    _action_row(),
                                ]
                            ),
                            className="shadow-sm border-0",
                        ),
                        md=8,
                    ),
                    dbc.Col(_quick_actions_card(), md=4),
                ],
                className="mb-3",
            ),
            # Progress placeholders written by the upload + processing
            # callbacks.  Kept hidden by default; the callbacks toggle
            # ``style.display`` when they have something to render.
            html.Div(
                id="upload-progress-section",
                className="mb-3",
                style={"display": "none"},
            ),
            html.Div(
                id="processing-progress-section",
                className="mb-3",
                style={"display": "none"},
            ),
            _data_preview_card(),
            # Shared stores — read by analysis pages.
            dcc.Store(id="store-uploaded-data"),
            dcc.Store(id="store-data-config"),
            dcc.Store(id="store-preview-window", data={"start": 0, "end": 1000}),
            dcc.Store(
                id="store-loading-states",
                data={
                    "uploading": False,
                    "processing": False,
                    "upload_progress": 0,
                    "processing_progress": 0,
                },
            ),
            # Progress-tracker plumbing retained from the previous design.
            create_progress_store(
                store_id="upload-progress-store",
                initial_data={
                    "task_id": None,
                    "progress_percent": 0,
                    "bytes_processed": 0,
                    "total_bytes": 0,
                    "chunks_processed": 0,
                    "total_chunks": 0,
                    "elapsed_time": 0,
                    "estimated_remaining": 0,
                    "status": "idle",
                    "message": "Ready to upload",
                },
            ),
            create_progress_interval(
                interval_id="upload-progress-interval",
                interval_ms=500,
                disabled=True,
            ),
        ],
        className="container-fluid px-4 py-3 upload-page",
    )
