"""
Advanced Analysis page layout for vitalDSP webapp.

This module provides the layout for the advanced analysis page including
machine learning, deep learning, and advanced signal processing methods.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


def advanced_layout():
    """Create the comprehensive advanced analysis layout."""
    return html.Div(
        [
            # Page Header
            html.Div(
                [
                    html.H1("🧠 Advanced Analysis", className="text-center mb-4"),
                    html.P(
                        [
                            "Advanced signal processing methods including machine learning, deep learning, ",
                            "ensemble methods, and cutting-edge analysis techniques for research applications.",
                        ],
                        className="text-center text-muted mb-5",
                    ),
                ],
                className="mb-4",
            ),
            # Main Analysis Section
            dbc.Row(
                [
                    # Left Panel - Controls & Parameters
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H4(
                                                "🎛️ Advanced Analysis Configuration",
                                                className="mb-0",
                                            ),
                                            html.Small(
                                                "Configure advanced analysis parameters",
                                                className="text-muted",
                                            ),
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            # Signal Type Selection
                                            html.H6("Signal Type", className="mb-3"),
                                            dcc.Dropdown(
                                                id="advanced-signal-type",
                                                options=[
                                                    {
                                                        "label": "Auto-detect",
                                                        "value": "auto",
                                                    },
                                                    {
                                                        "label": "PPG (Photoplethysmography)",
                                                        "value": "ppg",
                                                    },
                                                    {
                                                        "label": "ECG (Electrocardiography)",
                                                        "value": "ecg",
                                                    },
                                                    {
                                                        "label": "Respiratory",
                                                        "value": "respiratory",
                                                    },
                                                    {
                                                        "label": "General Signal",
                                                        "value": "general",
                                                    },
                                                    {
                                                        "label": "Multi-modal",
                                                        "value": "multimodal",
                                                    },
                                                ],
                                                value="auto",
                                                className="mb-3",
                                            ),
                                            # Signal Source Selection
                                            html.H6("Signal Source", className="mb-3"),
                                            dbc.Select(
                                                id="advanced-signal-source",
                                                options=[
                                                    {"label": "Original Signal", "value": "original"},
                                                    {"label": "Filtered Signal", "value": "filtered"},
                                                ],
                                                value="filtered",
                                                className="mb-3",
                                            ),
                                            # Time Window Configuration
                                            html.H6("Time Window", className="mb-3"),
                                            # Position Slider (0-100%)
                                            html.Label(
                                                "Start Position (%)", className="form-label"
                                            ),
                                            dcc.Slider(
                                                id="advanced-start-position",
                                                min=0,
                                                max=100,
                                                step=1,
                                                value=0,
                                                marks={
                                                    0: "0%",
                                                    25: "25%",
                                                    50: "50%",
                                                    75: "75%",
                                                    100: "100%",
                                                },
                                                tooltip={"placement": "bottom", "always_visible": True},
                                                className="mb-3",
                                            ),
                                            # Duration Dropdown
                                            html.Label("Duration", className="form-label"),
                                            dbc.Select(
                                                id="advanced-duration",
                                                options=[
                                                    {"label": "30 seconds", "value": 30},
                                                    {"label": "1 minute", "value": 60},
                                                    {"label": "2 minutes", "value": 120},
                                                    {"label": "5 minutes", "value": 300},
                                                    {"label": "10 minutes", "value": 600},
                                                ],
                                                value=60,
                                                className="mb-3",
                                            ),
                                            # Quick Window Navigation
                                            html.Label("Quick Navigation", className="form-label"),
                                            html.Div(
                                                [
                                                    dbc.Button(
                                                        "⏪ -10%",
                                                        id="advanced-btn-nudge-m10",
                                                        color="secondary",
                                                        size="sm",
                                                        className="me-1",
                                                    ),
                                                    dbc.Button(
                                                        "⏪ -1%",
                                                        id="advanced-btn-nudge-m1",
                                                        color="secondary",
                                                        size="sm",
                                                        className="me-1",
                                                    ),
                                                    dbc.Button(
                                                        "+1% ⏩",
                                                        id="advanced-btn-nudge-p1",
                                                        color="secondary",
                                                        size="sm",
                                                        className="me-1",
                                                    ),
                                                    dbc.Button(
                                                        "+10% ⏩",
                                                        id="advanced-btn-nudge-p10",
                                                        color="secondary",
                                                        size="sm",
                                                    ),
                                                ],
                                                className="mb-4",
                                            ),
                                            # Analysis Categories
                                            html.H6(
                                                "Analysis Categories", className="mb-3"
                                            ),
                                            dcc.Checklist(
                                                id="advanced-analysis-categories",
                                                options=[
                                                    {
                                                        "label": "Machine Learning Analysis",
                                                        "value": "ml_analysis",
                                                    },
                                                    {
                                                        "label": "Deep Learning Models",
                                                        "value": "deep_learning",
                                                    },
                                                    {
                                                        "label": "Ensemble Methods",
                                                        "value": "ensemble",
                                                    },
                                                    {
                                                        "label": "Pattern Recognition",
                                                        "value": "pattern_recognition",
                                                    },
                                                    {
                                                        "label": "Anomaly Detection",
                                                        "value": "anomaly_detection",
                                                    },
                                                    {
                                                        "label": "Classification",
                                                        "value": "classification",
                                                    },
                                                    {
                                                        "label": "Regression Analysis",
                                                        "value": "regression",
                                                    },
                                                    {
                                                        "label": "Clustering",
                                                        "value": "clustering",
                                                    },
                                                    {
                                                        "label": "Dimensionality Reduction",
                                                        "value": "dimensionality_reduction",
                                                    },
                                                    {
                                                        "label": "Time Series Forecasting",
                                                        "value": "forecasting",
                                                    },
                                                ],
                                                value=[
                                                    "ml_analysis",
                                                    "pattern_recognition",
                                                ],
                                                className="mb-3",
                                            ),
                                            # Machine Learning Methods (Accordion)
                                            html.H6("🤖 Machine Learning Methods", className="mb-3"),
                                            dbc.Accordion(
                                                [
                                                    # SVM
                                                    dbc.AccordionItem(
                                                        [
                                                            dbc.Checkbox(
                                                                id="advanced-svm-enable",
                                                                label="Enable SVM Analysis",
                                                                value=True,
                                                                className="mb-3",
                                                            ),
                                                            html.Label("Kernel Type", className="form-label small"),
                                                            dbc.Select(
                                                                id="advanced-svm-kernel",
                                                                options=[
                                                                    {"label": "RBF (Radial Basis Function)", "value": "rbf"},
                                                                    {"label": "Linear", "value": "linear"},
                                                                    {"label": "Polynomial", "value": "poly"},
                                                                    {"label": "Sigmoid", "value": "sigmoid"},
                                                                ],
                                                                value="rbf",
                                                                size="sm",
                                                                className="mb-2",
                                                            ),
                                                            html.Label("C (Regularization)", className="form-label small"),
                                                            dbc.Input(
                                                                id="advanced-svm-c",
                                                                type="number",
                                                                value=1.0,
                                                                min=0.001,
                                                                step=0.1,
                                                                size="sm",
                                                                className="mb-2",
                                                            ),
                                                            html.Label("Gamma", className="form-label small"),
                                                            dbc.Select(
                                                                id="advanced-svm-gamma",
                                                                options=[
                                                                    {"label": "Auto", "value": "auto"},
                                                                    {"label": "Scale", "value": "scale"},
                                                                ],
                                                                value="scale",
                                                                size="sm",
                                                                className="mb-2",
                                                            ),
                                                            html.Small("ℹ️ SVM finds optimal hyperplane for classification", className="text-muted"),
                                                        ],
                                                        title="Support Vector Machine (SVM)",
                                                    ),
                                                    # Random Forest
                                                    dbc.AccordionItem(
                                                        [
                                                            dbc.Checkbox(
                                                                id="advanced-rf-enable",
                                                                label="Enable Random Forest",
                                                                value=True,
                                                                className="mb-3",
                                                            ),
                                                            html.Label("Number of Trees", className="form-label small"),
                                                            dbc.Input(
                                                                id="advanced-rf-n-estimators",
                                                                type="number",
                                                                value=100,
                                                                min=10,
                                                                max=1000,
                                                                step=10,
                                                                size="sm",
                                                                className="mb-2",
                                                            ),
                                                            html.Label("Max Depth", className="form-label small"),
                                                            dbc.Input(
                                                                id="advanced-rf-max-depth",
                                                                type="number",
                                                                value=10,
                                                                min=1,
                                                                max=50,
                                                                step=1,
                                                                size="sm",
                                                                className="mb-2",
                                                            ),
                                                            html.Label("Min Samples Split", className="form-label small"),
                                                            dbc.Input(
                                                                id="advanced-rf-min-samples-split",
                                                                type="number",
                                                                value=2,
                                                                min=2,
                                                                max=20,
                                                                step=1,
                                                                size="sm",
                                                                className="mb-2",
                                                            ),
                                                            html.Small("ℹ️ Ensemble of decision trees for robust predictions", className="text-muted"),
                                                        ],
                                                        title="Random Forest",
                                                    ),
                                                    # Neural Network
                                                    dbc.AccordionItem(
                                                        [
                                                            dbc.Checkbox(
                                                                id="advanced-nn-enable",
                                                                label="Enable Neural Network",
                                                                value=False,
                                                                className="mb-3",
                                                            ),
                                                            html.Label("Hidden Layer Sizes", className="form-label small"),
                                                            dbc.Input(
                                                                id="advanced-nn-hidden-layers",
                                                                type="text",
                                                                value="100,50",
                                                                placeholder="e.g., 100,50",
                                                                size="sm",
                                                                className="mb-2",
                                                            ),
                                                            html.Label("Activation Function", className="form-label small"),
                                                            dbc.Select(
                                                                id="advanced-nn-activation",
                                                                options=[
                                                                    {"label": "ReLU", "value": "relu"},
                                                                    {"label": "Tanh", "value": "tanh"},
                                                                    {"label": "Logistic", "value": "logistic"},
                                                                ],
                                                                value="relu",
                                                                size="sm",
                                                                className="mb-2",
                                                            ),
                                                            html.Label("Learning Rate", className="form-label small"),
                                                            dbc.Input(
                                                                id="advanced-nn-learning-rate",
                                                                type="number",
                                                                value=0.001,
                                                                min=0.0001,
                                                                max=1.0,
                                                                step=0.0001,
                                                                size="sm",
                                                                className="mb-2",
                                                            ),
                                                            html.Small("ℹ️ Multi-layer perceptron for complex patterns", className="text-muted"),
                                                        ],
                                                        title="Neural Network (MLP)",
                                                    ),
                                                ],
                                                id="advanced-ml-accordion",
                                                start_collapsed=False,
                                                className="mb-3",
                                            ),
                                            # Hidden checklist for backward compatibility
                                            dcc.Checklist(
                                                id="advanced-ml-options",
                                                options=[
                                                    {"label": "SVM", "value": "svm"},
                                                    {"label": "Random Forest", "value": "random_forest"},
                                                    {"label": "Neural Networks", "value": "neural_networks"},
                                                ],
                                                value=["svm", "random_forest"],
                                                style={"display": "none"},
                                            ),
                                            # Deep Learning Methods (Accordion)
                                            html.H6("🧠 Deep Learning Methods", className="mb-3"),
                                            dbc.Accordion(
                                                [
                                                    # LSTM
                                                    dbc.AccordionItem(
                                                        [
                                                            dbc.Checkbox(
                                                                id="advanced-lstm-enable",
                                                                label="Enable LSTM Analysis",
                                                                value=True,
                                                                className="mb-3",
                                                            ),
                                                            html.Label("Hidden Units", className="form-label small"),
                                                            dbc.Input(
                                                                id="advanced-lstm-units",
                                                                type="number",
                                                                value=128,
                                                                min=16,
                                                                max=512,
                                                                step=16,
                                                                size="sm",
                                                                className="mb-2",
                                                            ),
                                                            html.Label("Number of Layers", className="form-label small"),
                                                            dbc.Input(
                                                                id="advanced-lstm-layers",
                                                                type="number",
                                                                value=2,
                                                                min=1,
                                                                max=5,
                                                                step=1,
                                                                size="sm",
                                                                className="mb-2",
                                                            ),
                                                            html.Label("Dropout Rate", className="form-label small"),
                                                            dbc.Input(
                                                                id="advanced-lstm-dropout",
                                                                type="number",
                                                                value=0.2,
                                                                min=0.0,
                                                                max=0.9,
                                                                step=0.1,
                                                                size="sm",
                                                                className="mb-2",
                                                            ),
                                                            html.Label("Learning Rate", className="form-label small"),
                                                            dbc.Input(
                                                                id="advanced-lstm-lr",
                                                                type="number",
                                                                value=0.001,
                                                                min=0.0001,
                                                                max=0.1,
                                                                step=0.0001,
                                                                size="sm",
                                                                className="mb-2",
                                                            ),
                                                            html.Small("ℹ️ LSTM networks excel at time series analysis", className="text-muted"),
                                                        ],
                                                        title="LSTM (Long Short-Term Memory)",
                                                    ),
                                                    # CNN
                                                    dbc.AccordionItem(
                                                        [
                                                            dbc.Checkbox(
                                                                id="advanced-cnn-enable",
                                                                label="Enable CNN Analysis",
                                                                value=True,
                                                                className="mb-3",
                                                            ),
                                                            html.Label("Number of Filters", className="form-label small"),
                                                            dbc.Input(
                                                                id="advanced-cnn-filters",
                                                                type="text",
                                                                value="64,128,256",
                                                                placeholder="e.g., 64,128,256",
                                                                size="sm",
                                                                className="mb-2",
                                                            ),
                                                            html.Label("Kernel Size", className="form-label small"),
                                                            dbc.Input(
                                                                id="advanced-cnn-kernel",
                                                                type="number",
                                                                value=3,
                                                                min=2,
                                                                max=11,
                                                                step=1,
                                                                size="sm",
                                                                className="mb-2",
                                                            ),
                                                            html.Label("Pool Size", className="form-label small"),
                                                            dbc.Input(
                                                                id="advanced-cnn-pool",
                                                                type="number",
                                                                value=2,
                                                                min=2,
                                                                max=5,
                                                                step=1,
                                                                size="sm",
                                                                className="mb-2",
                                                            ),
                                                            html.Label("Dropout Rate", className="form-label small"),
                                                            dbc.Input(
                                                                id="advanced-cnn-dropout",
                                                                type="number",
                                                                value=0.3,
                                                                min=0.0,
                                                                max=0.9,
                                                                step=0.1,
                                                                size="sm",
                                                                className="mb-2",
                                                            ),
                                                            html.Small("ℹ️ CNN extracts local patterns from signal data", className="text-muted"),
                                                        ],
                                                        title="CNN (Convolutional Neural Network)",
                                                    ),
                                                    # Transformer
                                                    dbc.AccordionItem(
                                                        [
                                                            dbc.Checkbox(
                                                                id="advanced-transformer-enable",
                                                                label="Enable Transformer",
                                                                value=False,
                                                                className="mb-3",
                                                            ),
                                                            html.Label("Attention Heads", className="form-label small"),
                                                            dbc.Input(
                                                                id="advanced-transformer-heads",
                                                                type="number",
                                                                value=8,
                                                                min=1,
                                                                max=16,
                                                                step=1,
                                                                size="sm",
                                                                className="mb-2",
                                                            ),
                                                            html.Label("Number of Layers", className="form-label small"),
                                                            dbc.Input(
                                                                id="advanced-transformer-layers",
                                                                type="number",
                                                                value=4,
                                                                min=1,
                                                                max=12,
                                                                step=1,
                                                                size="sm",
                                                                className="mb-2",
                                                            ),
                                                            html.Label("Model Dimension", className="form-label small"),
                                                            dbc.Input(
                                                                id="advanced-transformer-dim",
                                                                type="number",
                                                                value=512,
                                                                min=128,
                                                                max=1024,
                                                                step=128,
                                                                size="sm",
                                                                className="mb-2",
                                                            ),
                                                            html.Small("ℹ️ Transformer uses self-attention mechanisms", className="text-muted"),
                                                        ],
                                                        title="Transformer Model",
                                                    ),
                                                ],
                                                id="advanced-dl-accordion",
                                                start_collapsed=False,
                                                className="mb-3",
                                            ),
                                            # Hidden checklist for backward compatibility
                                            dcc.Checklist(
                                                id="advanced-deep-learning-options",
                                                options=[
                                                    {"label": "LSTM", "value": "lstm"},
                                                    {"label": "CNN", "value": "cnn"},
                                                    {"label": "Transformer", "value": "transformer"},
                                                ],
                                                value=["lstm", "cnn"],
                                                style={"display": "none"},
                                            ),
                                            # Advanced Parameters
                                            html.H6(
                                                "Advanced Parameters", className="mb-3"
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "Cross-validation Folds",
                                                                className="form-label",
                                                            ),
                                                            dbc.Input(
                                                                id="advanced-cv-folds",
                                                                type="number",
                                                                value=5,
                                                                min=2,
                                                                max=10,
                                                                step=1,
                                                                placeholder="5",
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "Random State",
                                                                className="form-label",
                                                            ),
                                                            dbc.Input(
                                                                id="advanced-random-state",
                                                                type="number",
                                                                value=42,
                                                                min=0,
                                                                step=1,
                                                                placeholder="42",
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            # Model Configuration
                                            html.H6(
                                                "Model Configuration", className="mb-3"
                                            ),
                                            dcc.Checklist(
                                                id="advanced-model-config",
                                                options=[
                                                    {
                                                        "label": "Hyperparameter Tuning",
                                                        "value": "hyperparameter_tuning",
                                                    },
                                                    {
                                                        "label": "Feature Selection",
                                                        "value": "feature_selection",
                                                    },
                                                    {
                                                        "label": "Model Interpretability",
                                                        "value": "interpretability",
                                                    },
                                                    {
                                                        "label": "Model Validation",
                                                        "value": "validation",
                                                    },
                                                    {
                                                        "label": "Performance Metrics",
                                                        "value": "performance_metrics",
                                                    },
                                                    {
                                                        "label": "Confusion Matrix",
                                                        "value": "confusion_matrix",
                                                    },
                                                    {
                                                        "label": "ROC Analysis",
                                                        "value": "roc_analysis",
                                                    },
                                                    {
                                                        "label": "Learning Curves",
                                                        "value": "learning_curves",
                                                    },
                                                ],
                                                value=[
                                                    "hyperparameter_tuning",
                                                    "feature_selection",
                                                ],
                                                className="mb-4",
                                            ),
                                            # Main Analysis Button
                                            dbc.Button(
                                                "🧠 Run Advanced Analysis",
                                                id="advanced-analyze-btn",
                                                color="primary",
                                                size="lg",
                                                className="w-100 mb-3",
                                            ),
                                        ]
                                    ),
                                ],
                                className="h-100",
                            )
                        ],
                        md=4,
                    ),
                    # Right Panel - Results & Plots
                    dbc.Col(
                        [
                            # Main Analysis Plot
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H5(
                                                "📈 Advanced Analysis Results",
                                                className="mb-0",
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="advanced-main-plot",
                                                style={"height": "400px"},
                                                config={
                                                    "displayModeBar": True,
                                                    "displaylogo": False,
                                                },
                                            )
                                        ]
                                    ),
                                ],
                                className="mb-4",
                            ),
                            # Model Performance Visualization
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H5(
                                                "📊 Model Performance & Metrics",
                                                className="mb-0",
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="advanced-performance-plot",
                                                style={"height": "400px"},
                                                config={
                                                    "displayModeBar": True,
                                                    "displaylogo": False,
                                                },
                                            )
                                        ]
                                    ),
                                ],
                                className="mb-4",
                            ),
                        ],
                        md=8,
                    ),
                ]
            ),
            # Analysis Results Section
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H5(
                                                "📋 Analysis Summary", className="mb-0"
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [html.Div(id="advanced-analysis-summary")]
                                    ),
                                ]
                            )
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [html.H5("🔍 Model Details", className="mb-0")]
                                    ),
                                    dbc.CardBody(
                                        [html.Div(id="advanced-model-details")]
                                    ),
                                ]
                            )
                        ],
                        md=6,
                    ),
                ],
                className="mb-4",
            ),
            # Performance Metrics
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H5(
                                                "📊 Performance Metrics",
                                                className="mb-0",
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [html.Div(id="advanced-performance-metrics")]
                                    ),
                                ]
                            )
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H5(
                                                "🎯 Feature Importance",
                                                className="mb-0",
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [html.Div(id="advanced-feature-importance")]
                                    ),
                                ]
                            )
                        ],
                        md=6,
                    ),
                ],
                className="mb-4",
            ),
            # Advanced Visualizations
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H5(
                                                "🌊 Advanced Visualizations",
                                                className="mb-0",
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="advanced-visualizations",
                                                style={"height": "800px"},
                                                config={
                                                    "displayModeBar": True,
                                                    "displaylogo": False,
                                                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                                                },
                                            )
                                        ]
                                    ),
                                ]
                            )
                        ],
                        md=12,
                    )
                ],
                className="mb-4",
            ),
            # Detailed Analysis Report Section
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H5(
                                                "📝 Detailed Analysis Report",
                                                className="mb-0",
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [html.Div(id="advanced-detailed-report")]
                                    ),
                                ],
                                style={"height": "100%"}
                            )
                        ],
                        md=12,
                    )
                ],
                className="mb-4",
            ),
            # Data Storage
            dcc.Store(id="store-advanced-data"),
            dcc.Store(id="store-advanced-results"),
            dcc.Store(id="store-filtered-signal"),  # For filtered signal from filtering page
            # Hidden components for compatibility with filtering page callbacks
            # These are needed because filtering page callbacks may reference these IDs
            html.Div(id="filter-btn-apply", style={"display": "none"}),
            dcc.Dropdown(id="filter-type-select", style={"display": "none"}),
            dcc.Dropdown(id="filter-signal-type-select", style={"display": "none"}),
            dcc.Checklist(id="filter-quality-options", style={"display": "none"}),
            dcc.Checklist(id="detrend-option", style={"display": "none"}),
            html.Div(id="filter-type-callback", style={"display": "none"}),
            # Traditional filter parameters
            dcc.Dropdown(id="filter-family-advanced", style={"display": "none"}),
            dcc.Dropdown(id="filter-response-advanced", style={"display": "none"}),
            dbc.Input(id="filter-low-freq-advanced", style={"display": "none"}),
            dbc.Input(id="filter-high-freq-advanced", style={"display": "none"}),
            dbc.Input(id="filter-order-advanced", style={"display": "none"}),
            # Savitzky-Golay parameters
            dbc.Input(id="savgol-window", style={"display": "none"}),
            dbc.Input(id="savgol-polyorder", style={"display": "none"}),
            # Moving average and Gaussian parameters
            dbc.Input(id="moving-avg-window", style={"display": "none"}),
            dbc.Input(id="gaussian-sigma", style={"display": "none"}),
            html.Div(id="traditional-filter-params", style={"display": "none"}),
            # Advanced filter parameters
            dcc.Dropdown(id="advanced-filter-method", style={"display": "none"}),
            # Kalman filter parameters
            dbc.Input(id="kalman-r", style={"display": "none"}),
            dbc.Input(id="kalman-q", style={"display": "none"}),
            html.Div(id="kalman-params", style={"display": "none"}),
            # Optimization-based filter parameters
            dcc.Dropdown(id="optimization-loss-type", style={"display": "none"}),
            dbc.Input(id="optimization-initial-guess", style={"display": "none"}),
            dbc.Input(id="optimization-learning-rate", style={"display": "none"}),
            dbc.Input(id="optimization-iterations", style={"display": "none"}),
            html.Div(id="optimization-params", style={"display": "none"}),
            # Gradient-based filter parameters
            dbc.Input(id="gradient-learning-rate", style={"display": "none"}),
            dbc.Input(id="gradient-iterations", style={"display": "none"}),
            html.Div(id="gradient-params", style={"display": "none"}),
            # Convolution filter parameters
            dcc.Dropdown(id="convolution-kernel-type", style={"display": "none"}),
            dbc.Input(id="convolution-kernel-size", style={"display": "none"}),
            html.Div(id="convolution-params", style={"display": "none"}),
            # Attention mechanism parameters
            dcc.Dropdown(id="attention-type", style={"display": "none"}),
            dbc.Input(id="attention-size", style={"display": "none"}),
            dbc.Input(id="attention-sigma", style={"display": "none"}),
            html.Div(id="attention-gaussian-params", style={"display": "none"}),
            dcc.Checklist(id="attention-ascending", style={"display": "none"}),
            html.Div(id="attention-linear-params", style={"display": "none"}),
            dbc.Input(id="attention-base", style={"display": "none"}),
            html.Div(id="attention-exponential-params", style={"display": "none"}),
            html.Div(id="attention-params", style={"display": "none"}),
            # Adaptive filter parameters
            dbc.Input(id="adaptive-mu", style={"display": "none"}),
            dbc.Input(id="adaptive-order", style={"display": "none"}),
            html.Div(id="adaptive-params", style={"display": "none"}),
            html.Div(id="advanced-filter-params", style={"display": "none"}),
            # Artifact removal parameters
            dcc.Dropdown(id="artifact-type", style={"display": "none"}),
            dbc.Input(id="artifact-removal-strength", style={"display": "none"}),
            dcc.Dropdown(id="wavelet-type", style={"display": "none"}),
            dbc.Input(id="wavelet-level", style={"display": "none"}),
            dcc.Dropdown(id="threshold-type", style={"display": "none"}),
            dbc.Input(id="threshold-value", style={"display": "none"}),
            html.Div(id="artifact-removal-params", style={"display": "none"}),
            # Neural network parameters
            dcc.Dropdown(id="neural-network-type", style={"display": "none"}),
            dbc.Input(id="neural-model-complexity", style={"display": "none"}),
            html.Div(id="neural-network-params", style={"display": "none"}),
            # Ensemble parameters
            dcc.Dropdown(id="ensemble-method", style={"display": "none"}),
            dbc.Input(id="ensemble-n-filters", style={"display": "none"}),
            html.Div(id="ensemble-params", style={"display": "none"}),
            # Reference signal and fusion parameters
            dcc.Dropdown(id="reference-signal", style={"display": "none"}),
            dcc.Dropdown(id="fusion-method", style={"display": "none"}),
            dcc.Store(id="store-filtering-data"),
            dcc.Store(id="store-filter-comparison"),
            dcc.Store(id="store-filter-quality-metrics"),
            dcc.Graph(id="filter-original-plot", style={"display": "none"}),
            dcc.Graph(id="filter-filtered-plot", style={"display": "none"}),
            dcc.Graph(id="filter-comparison-plot", style={"display": "none"}),
            dcc.Graph(id="filter-quality-plots", style={"display": "none"}),
            html.Div(id="filter-quality-metrics", style={"display": "none"}),
            dcc.Graph(id="frequency-filtered-signal-plot", style={"display": "none"}),
            dcc.Graph(id="filter-response-plot", style={"display": "none"}),
            html.Div(id="filter-stats", style={"display": "none"}),
            # Hidden navigation buttons for signal_filtering_callbacks compatibility
            html.Button(id="btn-nudge-m10", style={"display": "none"}),
            html.Button(id="btn-center", style={"display": "none"}),
            html.Button(id="btn-nudge-p10", style={"display": "none"}),
            # Hidden time window controls for signal_filtering_callbacks compatibility
            html.Div(
                children=[dcc.Slider(id="start-position-slider", min=0, max=100, value=0)],
                style={"display": "none"}
            ),
            dcc.Dropdown(id="duration-select", style={"display": "none"}),
        ]
    )