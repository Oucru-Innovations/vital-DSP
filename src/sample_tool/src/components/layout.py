"""
Layout components for the PPG analysis tool.
"""

import dash
from dash import dcc, html

from ..config.settings import (
    APP_SUBTITLE,
    APP_TITLE,
    DEFAULT_DECIM_USER,
    DEFAULT_FS,
    DEFAULT_HR_MAX,
    DEFAULT_HR_MIN,
    DEFAULT_PEAK_PROM_FACTOR,
    DEFAULT_SPEC_OVERLAP,
    DEFAULT_SPEC_WIN_SEC,
    DEFAULT_WINDOW_END,
    DEFAULT_WINDOW_START,
)


def create_layout():
    """Create the main app layout."""
    return html.Div(
        [
            # Stores
            dcc.Store(id="store_file_path"),
            dcc.Store(id="store_total_rows"),
            dcc.Store(id="store_window"),  # {"start": int, "end": int}
            dcc.Store(id="store_theme", data="dark"),
            dcc.Store(id="store_prev_total_rows"),
            dcc.Store(id="store_initial_load", data=True),  # Triggers initial data loading
            html.Div(
                className="header",
                children=[
                    html.Div(className="title", children=APP_TITLE),
                    html.Div(className="subtle", children=APP_SUBTITLE),
                ],
            ),
            html.Div(
                className="grid",
                children=[
                    # ---------------- Left: Data & Filter ----------------
                    _create_left_panel(),
                    # ---------------- Middle: Charts (wider/taller) ----------------
                    _create_middle_panel(),
                    # ---------------- Right: Insights & Export ----------------
                    _create_right_panel(),
                ],
            ),
        ]
    )


def _create_left_panel():
    """Create the left control panel."""
    return html.Div(
        className="card",
        children=[
            html.Div(className="section-title", children="Data"),
            html.Div(
                className="row",
                children=[
                    html.Div(
                        children=[
                            html.Label("Load from local path"),
                            dcc.Input(
                                id="file_path",
                                type="text",
                                placeholder="e.g., PPG.csv",
                                value="",
                                style={"width": "220px"},
                            ),
                        ]
                    ),
                    html.Button("Load", id="btn_load_path", className="btn", n_clicks=0),
                ],
            ),
            html.Div(style={"height": "8px"}),
            dcc.Upload(
                id="upload_csv",
                children=html.Div(["⬆️ ", html.Span("Drag & drop or click to upload CSV")]),
                multiple=False,
                className="upload",
            ),
            html.Div(id="file_status", className="hint", style={"marginTop": "6px"}),
            html.Div(style={"height": "10px"}),
            html.Label("Column mapping"),
            html.Div(
                className="row",
                children=[
                    html.Div(
                        children=[
                            "RED",
                            dcc.Dropdown(
                                id="red_col", options=[], value=None, style={"width": "280px"}
                            ),
                        ]
                    ),
                ],
            ),
            html.Div(
                className="row",
                style={"marginTop": "6px"},
                children=[
                    html.Div(
                        children=[
                            "IR",
                            dcc.Dropdown(
                                id="ir_col", options=[], value=None, style={"width": "280px"}
                            ),
                        ]
                    ),
                ],
            ),
            html.Div(
                className="row",
                style={"marginTop": "6px"},
                children=[
                    html.Div(
                        children=[
                            "Waveform",
                            dcc.Dropdown(
                                id="waveform_col",
                                options=[],
                                value=None,
                                placeholder="PLETH",
                                style={"width": "280px"},
                            ),
                        ]
                    ),
                ],
            ),
            html.Div(style={"height": "10px"}),
            html.Label("Sampling frequency (Hz)"),
            dcc.Input(
                id="fs", type="number", value=DEFAULT_FS, step=0.5, min=1, style={"width": "120px"}
            ),
            html.Div(style={"height": "10px"}),
            html.Label("Theme"),
            dcc.Dropdown(
                id="theme",
                options=[{"label": "Dark", "value": "dark"}, {"label": "Light", "value": "light"}],
                value="dark",
                clearable=False,
                style={"width": "160px"},
            ),
            html.Div(style={"height": "16px"}),
            html.Div(className="section-title", children="Window (rows)"),
            html.Div(
                className="row",
                children=[
                    html.Div(
                        children=[
                            html.Div("start_row"),
                            dcc.Input(
                                id="start_row",
                                type="number",
                                value=DEFAULT_WINDOW_START,
                                min=0,
                                step=100,
                            ),
                        ]
                    ),
                    html.Div(
                        children=[
                            html.Div("end_row"),
                            dcc.Input(
                                id="end_row",
                                type="number",
                                value=DEFAULT_WINDOW_END,
                                min=0,
                                step=100,
                            ),
                        ]
                    ),
                    html.Button("Apply", id="btn_apply_window", className="btn", n_clicks=0),
                ],
            ),
            html.Div(
                className="row",
                style={"marginTop": "6px"},
                children=[
                    html.Button("−10k", id="nudge_m10k", className="btn secondary", n_clicks=0),
                    html.Button("−1k", id="nudge_m1k", className="btn secondary", n_clicks=0),
                    html.Button("+1k", id="nudge_p1k", className="btn secondary", n_clicks=0),
                    html.Button("+10k", id="nudge_p10k", className="btn secondary", n_clicks=0),
                    html.Div(id="window_badge", className="pill", children="Rows: 0–0"),
                ],
            ),
            html.Div(style={"height": "8px"}),
            dcc.RangeSlider(
                id="row_slider",
                min=0,
                max=10000,
                step=1,
                value=[0, 10000],
                allowCross=False,
                pushable=100,
                updatemode="mouseup",
            ),
            html.Div(style={"height": "16px"}),
            html.Div(className="section-title", children="Filter"),
            html.Div(
                className="row",
                children=[
                    dcc.Dropdown(
                        id="family",
                        value="butter",
                        options=[
                            {"label": "Butterworth", "value": "butter"},
                            {"label": "Chebyshev I", "value": "cheby1"},
                            {"label": "Chebyshev II", "value": "cheby2"},
                            {"label": "Elliptic", "value": "ellip"},
                            {"label": "Bessel", "value": "bessel"},
                        ],
                        clearable=False,
                        style={"width": "160px"},
                    ),
                    dcc.Dropdown(
                        id="resp",
                        value="bandpass",
                        options=[
                            {"label": "Bandpass", "value": "bandpass"},
                            {"label": "Bandstop (Notch)", "value": "bandstop"},
                            {"label": "Lowpass", "value": "lowpass"},
                            {"label": "Highpass", "value": "highpass"},
                        ],
                        clearable=False,
                        style={"width": "170px"},
                    ),
                ],
            ),
            html.Div(className="row"),  # (Note: keep syntax valid if you edit)
        ],
    )


def _create_middle_panel():
    """Create the middle charts panel."""
    return html.Div(
        children=[
            dcc.Tabs(
                id="main_charts_tabs",
                value="time_domain",
                children=[
                    dcc.Tab(
                        label="Time-domain",
                        value="time_domain",
                        children=[
                            html.Div(
                                className="card",
                                children=[
                                    html.Div(
                                        className="section-title", children="Time-domain (bigger)"
                                    ),
                                    dcc.Loading(
                                        dcc.Graph(id="fig_raw", style={"height": "600px"}),
                                        type="default",
                                    ),
                                    dcc.Loading(
                                        dcc.Graph(id="fig_ac", style={"height": "600px"}),
                                        type="default",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="Frequency",
                        value="frequency",
                        children=[
                            html.Div(
                                className="card",
                                children=[
                                    html.Div(className="section-title", children="Frequency"),
                                    dcc.Loading(
                                        dcc.Graph(id="fig_psd", style={"height": "360px"}),
                                        type="default",
                                    ),
                                    html.Div(
                                        className="row",
                                        children=[
                                            html.Div(
                                                children=[
                                                    html.Div("Spectrogram win (s)"),
                                                    dcc.Input(
                                                        id="spec_win_sec",
                                                        type="number",
                                                        value=DEFAULT_SPEC_WIN_SEC,
                                                        min=0.5,
                                                        step=0.5,
                                                        style={"width": "100px"},
                                                    ),
                                                ]
                                            ),
                                            html.Div(
                                                children=[
                                                    html.Div("Overlap (0–0.95)"),
                                                    dcc.Input(
                                                        id="spec_overlap",
                                                        type="number",
                                                        value=DEFAULT_SPEC_OVERLAP,
                                                        min=0.0,
                                                        max=0.95,
                                                        step=0.05,
                                                        style={"width": "100px"},
                                                    ),
                                                ]
                                            ),
                                            html.Div(
                                                children=[
                                                    html.Div("Show spectrogram"),
                                                    dcc.Checklist(
                                                        id="show_spec",
                                                        options=[
                                                            {"label": "Enable", "value": "on"}
                                                        ],
                                                        value=["on"],
                                                    ),
                                                ]
                                            ),
                                        ],
                                        style={"margin": "6px 0"},
                                    ),
                                    dcc.Loading(
                                        dcc.Graph(id="fig_spec", style={"height": "380px"}),
                                        type="default",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="Dual-source analytics",
                        value="dual_source",
                        children=[
                            html.Div(
                                className="card",
                                children=[
                                    html.Div(
                                        className="section-title", children="Dual-source analytics"
                                    ),
                                    dcc.Tabs(
                                        id="dual_source_tabs",
                                        value="rtrend",
                                        children=[
                                            dcc.Tab(
                                                label="R-trend & SpO₂",
                                                value="rtrend",
                                                children=[
                                                    dcc.Loading(
                                                        dcc.Graph(
                                                            id="fig_rtrend",
                                                            style={"height": "320px"},
                                                        ),
                                                        type="default",
                                                    )
                                                ],
                                            ),
                                            dcc.Tab(
                                                label="Coherence",
                                                value="coh",
                                                children=[
                                                    dcc.Loading(
                                                        dcc.Graph(
                                                            id="fig_coh", style={"height": "300px"}
                                                        ),
                                                        type="default",
                                                    )
                                                ],
                                            ),
                                            dcc.Tab(
                                                label="Lissajous",
                                                value="liss",
                                                children=[
                                                    dcc.Loading(
                                                        dcc.Graph(
                                                            id="fig_liss", style={"height": "300px"}
                                                        ),
                                                        type="default",
                                                    )
                                                ],
                                            ),
                                            dcc.Tab(
                                                label="Average Beat",
                                                value="avgbeat",
                                                children=[
                                                    dcc.Loading(
                                                        dcc.Graph(
                                                            id="fig_avgbeat",
                                                            style={"height": "300px"},
                                                        ),
                                                        type="default",
                                                    )
                                                ],
                                            ),
                                            dcc.Tab(
                                                label="SDPPG",
                                                value="sdppg",
                                                children=[
                                                    dcc.Loading(
                                                        dcc.Graph(
                                                            id="fig_sdppg",
                                                            style={"height": "300px"},
                                                        ),
                                                        type="default",
                                                    )
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="Waveform Analysis",
                        value="waveform",
                        children=[
                            html.Div(
                                className="card",
                                children=[
                                    html.Div(
                                        className="section-title", children="Waveform Analysis"
                                    ),
                                    html.Div(
                                        className="row",
                                        children=[
                                            html.Div(
                                                children=[
                                                    html.Div("Waveform Type"),
                                                    dcc.Dropdown(
                                                        id="waveform_type",
                                                        options=[
                                                            {"label": "Raw Signal", "value": "raw"},
                                                            {
                                                                "label": "Filtered Signal",
                                                                "value": "filtered",
                                                            },
                                                            {
                                                                "label": "Normalized",
                                                                "value": "normalized",
                                                            },
                                                            {
                                                                "label": "Derivative",
                                                                "value": "derivative",
                                                            },
                                                        ],
                                                        value="raw",
                                                        clearable=False,
                                                        style={"width": "160px"},
                                                    ),
                                                ]
                                            ),
                                            html.Div(
                                                children=[
                                                    html.Div("Window Size (s)"),
                                                    dcc.Input(
                                                        id="waveform_window",
                                                        type="number",
                                                        value=5.0,
                                                        min=1.0,
                                                        max=30.0,
                                                        step=0.5,
                                                        style={"width": "100px"},
                                                    ),
                                                ]
                                            ),
                                            html.Div(
                                                children=[
                                                    html.Div("Show Annotations"),
                                                    dcc.Checklist(
                                                        id="show_waveform_annotations",
                                                        options=[
                                                            {"label": "Peaks", "value": "peaks"},
                                                            {
                                                                "label": "Valleys",
                                                                "value": "valleys",
                                                            },
                                                            {
                                                                "label": "Zero Crossings",
                                                                "value": "zero_crossings",
                                                            },
                                                        ],
                                                        value=["peaks"],
                                                        style={"margin": "6px 0"},
                                                    ),
                                                ]
                                            ),
                                        ],
                                        style={"margin": "6px 0"},
                                    ),
                                    dcc.Loading(
                                        dcc.Graph(id="fig_waveform", style={"height": "400px"}),
                                        type="default",
                                    ),
                                    dcc.Loading(
                                        dcc.Graph(
                                            id="fig_waveform_stats", style={"height": "300px"}
                                        ),
                                        type="default",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="Dynamics (HR/IBI)",
                        value="dynamics",
                        children=[
                            html.Div(
                                className="card",
                                children=[
                                    html.Div(
                                        className="section-title", children="Dynamics (HR/IBI)"
                                    ),
                                    html.Div(
                                        className="row",
                                        children=[
                                            html.Div(
                                                children=[
                                                    html.Div("HR source"),
                                                    dcc.Dropdown(
                                                        id="hr_source",
                                                        options=[
                                                            {
                                                                "label": "IR (default)",
                                                                "value": "ir",
                                                            },
                                                            {"label": "RED", "value": "red"},
                                                        ],
                                                        value="ir",
                                                        clearable=False,
                                                        style={"width": "140px"},
                                                    ),
                                                ]
                                            ),
                                            html.Div(
                                                children=[
                                                    html.Div("HR min/max (bpm)"),
                                                    dcc.Input(
                                                        id="hr_min",
                                                        type="number",
                                                        value=DEFAULT_HR_MIN,
                                                        min=20,
                                                        step=5,
                                                        style={"width": "80px"},
                                                    ),
                                                    dcc.Input(
                                                        id="hr_max",
                                                        type="number",
                                                        value=DEFAULT_HR_MAX,
                                                        min=60,
                                                        step=5,
                                                        style={"width": "80px"},
                                                    ),
                                                ]
                                            ),
                                            html.Div(
                                                children=[
                                                    html.Div("Peak prom ×std"),
                                                    dcc.Input(
                                                        id="peak_prom",
                                                        type="number",
                                                        value=DEFAULT_PEAK_PROM_FACTOR,
                                                        min=0.1,
                                                        step=0.1,
                                                        style={"width": "80px"},
                                                    ),
                                                ]
                                            ),
                                            html.Div(
                                                children=[
                                                    html.Div("Show advanced"),
                                                    dcc.Checklist(
                                                        id="show_adv",
                                                        options=[
                                                            {"label": "HR trend", "value": "hr"},
                                                            {"label": "IBI hist", "value": "hist"},
                                                            {"label": "Poincaré", "value": "poi"},
                                                            {
                                                                "label": "Cross-corr",
                                                                "value": "xcorr",
                                                            },
                                                        ],
                                                        value=["hr", "hist", "poi", "xcorr"],
                                                    ),
                                                ]
                                            ),
                                        ],
                                    ),
                                    dcc.Tabs(
                                        id="dynamics_tabs",
                                        value="hr",
                                        children=[
                                            dcc.Tab(
                                                label="HR Trend",
                                                value="hr",
                                                children=[
                                                    dcc.Loading(
                                                        dcc.Graph(
                                                            id="fig_hr_trend",
                                                            style={"height": "320px"},
                                                        ),
                                                        type="default",
                                                    )
                                                ],
                                            ),
                                            dcc.Tab(
                                                label="IBI Histogram",
                                                value="hist",
                                                children=[
                                                    dcc.Loading(
                                                        dcc.Graph(
                                                            id="fig_ibi_hist",
                                                            style={"height": "280px"},
                                                        ),
                                                        type="default",
                                                    )
                                                ],
                                            ),
                                            dcc.Tab(
                                                label="Poincaré",
                                                value="poi",
                                                children=[
                                                    dcc.Loading(
                                                        dcc.Graph(
                                                            id="fig_poincare",
                                                            style={"height": "280px"},
                                                        ),
                                                        type="default",
                                                    )
                                                ],
                                            ),
                                            dcc.Tab(
                                                label="Cross-correlation",
                                                value="xcorr",
                                                children=[
                                                    dcc.Loading(
                                                        dcc.Graph(
                                                            id="fig_xcorr",
                                                            style={"height": "280px"},
                                                        ),
                                                        type="default",
                                                    )
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ]
    )


def _create_right_panel():
    """Create the right control and info panel."""
    return html.Div(
        children=[
            html.Div(
                className="card",
                children=[
                    html.Div(className="section-title", children="Filter Controls (cont'd)"),
                    html.Div(
                        className="row",
                        children=[
                            html.Div(
                                children=[
                                    html.Div("Low (Hz)"),
                                    dcc.Input(
                                        id="low_hz", type="number", value=0.5, step=0.1, min=0
                                    ),
                                ]
                            ),
                            html.Div(
                                children=[
                                    html.Div("High (Hz)"),
                                    dcc.Input(
                                        id="high_hz", type="number", value=5.0, step=0.1, min=0.1
                                    ),
                                ]
                            ),
                        ],
                    ),
                    html.Div(style={"height": "6px"}),
                    html.Div(
                        children=[
                            html.Div("Order"),
                            dcc.Slider(
                                id="order",
                                min=1,
                                max=10,
                                step=1,
                                value=2,
                                marks={i: str(i) for i in range(1, 11)},
                            ),
                        ]
                    ),
                    html.Div(style={"height": "10px"}),
                    html.Label("Ripple (Cheby/Ellip)"),
                    html.Div(
                        className="row",
                        children=[
                            html.Div(
                                children=[
                                    html.Div("rp (dB)"),
                                    dcc.Input(
                                        id="rp",
                                        type="number",
                                        value=1.0,
                                        step=0.1,
                                        min=0.01,
                                        style={"width": "90px"},
                                    ),
                                ]
                            ),
                            html.Div(
                                children=[
                                    html.Div("rs (dB)"),
                                    dcc.Input(
                                        id="rs",
                                        type="number",
                                        value=40.0,
                                        step=1,
                                        min=10,
                                        style={"width": "90px"},
                                    ),
                                ]
                            ),
                        ],
                    ),
                    html.Div(style={"height": "10px"}),
                    html.Label("Line-noise notch"),
                    html.Div(
                        className="row",
                        children=[
                            dcc.Checklist(
                                id="notch_enable",
                                options=[{"label": "Enable", "value": "on"}],
                                value=[],
                            ),
                            dcc.Dropdown(
                                id="notch_hz",
                                value=50.0,
                                options=[
                                    {"label": "50 Hz", "value": 50.0},
                                    {"label": "60 Hz", "value": 60.0},
                                ],
                                clearable=False,
                                style={"width": "110px"},
                            ),
                            html.Div(
                                children=[
                                    html.Div("Q"),
                                    dcc.Input(
                                        id="notch_q",
                                        type="number",
                                        value=30.0,
                                        step=1,
                                        min=5,
                                        style={"width": "90px"},
                                    ),
                                ]
                            ),
                        ],
                    ),
                    html.Div(style={"height": "10px"}),
                    html.Label("Display decimation (auto-ups if needed)"),
                    dcc.Input(
                        id="decim",
                        type="number",
                        value=DEFAULT_DECIM_USER,
                        min=1,
                        step=1,
                        style={"width": "120px"},
                    ),
                    html.Div(style={"height": "10px"}),
                    dcc.Checklist(
                        id="flags",
                        options=[
                            {"label": "Detrend (remove mean)", "value": "detrend"},
                            {"label": "Invert filtered AC (pulse up)", "value": "invert"},
                        ],
                        value=["invert"],
                    ),
                ],
            ),
            html.Div(style={"height": "12px"}),
            html.Div(
                className="card",
                children=[
                    html.Div(className="section-title", children="Insights"),
                    html.Div(
                        id="insights", className="row", style={"flexWrap": "wrap", "gap": "10px"}
                    ),
                    html.Div(id="notes", className="hint", style={"marginTop": "10px"}),
                ],
            ),
            html.Div(style={"height": "12px"}),
            html.Div(
                className="card",
                children=[
                    html.Div(className="section-title", children="File / Window / Export"),
                    html.Div(
                        id="file_info", className="row", style={"flexWrap": "wrap", "gap": "10px"}
                    ),
                    html.Div(style={"height": "8px"}),
                    html.Button(
                        "Download current window (CSV)",
                        id="btn_dl_csv",
                        className="btn",
                        n_clicks=0,
                    ),
                    dcc.Download(id="dl_csv"),
                ],
            ),
        ]
    )
