# VitalDSP Webapp Structure & Architecture Guide

## 📋 Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Core Architecture](#core-architecture)
4. [Layout System](#layout-system)
5. [Callback System](#callback-system)
6. [Data Flow](#data-flow)
7. [Configuration](#configuration)
8. [Services](#services)
9. [API Integration](#api-integration)
10. [Development Guidelines](#development-guidelines)

---

## 🎯 Overview

The VitalDSP Webapp is a comprehensive digital signal processing dashboard built with **Dash** (Python web framework) and **FastAPI** for backend services. It provides interactive analysis tools for PPG/ECG signals with real-time visualization and processing capabilities.

### Key Features:
- **Multi-page Analysis**: Time domain, frequency domain, filtering, physiological analysis
- **Interactive Visualizations**: Real-time plots with Plotly
- **Advanced Signal Processing**: Traditional and neural network filtering
- **Data Management**: Upload, process, and export signal data
- **Modular Architecture**: Clean separation of concerns

---

## 🏗️ Project Structure

```
src/vitalDSP_webapp/
├── 📁 api/                    # FastAPI endpoints
│   ├── __init__.py
│   └── endpoints.py           # API route definitions
├── 📁 assets/                 # Static assets
│   ├── custom_plots.css       # Custom plot styling
│   ├── logo.png              # Application logo
│   └── styles.css            # Global styles
├── 📁 callbacks/              # Dash callback functions
│   ├── 📁 analysis/           # Analysis-specific callbacks
│   │   ├── vitaldsp_callbacks.py          # Time domain analysis
│   │   ├── frequency_filtering_callbacks.py # Frequency analysis
│   │   ├── signal_filtering_callbacks.py   # Signal filtering
│   │   ├── respiratory_callbacks.py        # Respiratory analysis
│   │   ├── physiological_callbacks.py      # Physiological features
│   │   ├── quality_callbacks.py           # Signal quality
│   │   ├── advanced_callbacks.py          # Advanced computations
│   │   ├── health_report_callbacks.py     # Health reporting
│   │   └── settings_callbacks.py          # Settings management
│   ├── 📁 core/               # Core application callbacks
│   │   ├── app_callbacks.py               # Sidebar, theme, global
│   │   ├── page_routing_callbacks.py      # Page navigation
│   │   └── upload_callbacks.py            # File upload handling
│   ├── 📁 features/           # Feature-specific callbacks
│   │   ├── features_callbacks.py          # Feature engineering
│   │   ├── physiological_callbacks.py     # Physiological analysis
│   │   └── preview_callbacks.py           # Data preview
│   └── 📁 utils/              # Callback utilities
├── 📁 config/                 # Configuration management
│   ├── settings.py            # App configuration
│   └── logging_config.py      # Logging setup
├── 📁 layout/                 # UI layout components
│   ├── 📁 common/             # Shared layout components
│   │   ├── header.py          # Application header
│   │   ├── sidebar.py         # Navigation sidebar
│   │   └── footer.py          # Application footer
│   └── 📁 pages/              # Page-specific layouts
│       ├── analysis_pages.py  # All analysis page layouts
│       └── upload_page.py     # File upload page
├── 📁 models/                 # Data models
│   └── signal_processing.py   # Signal processing models
├── 📁 services/               # Business logic services
│   ├── 📁 data/
│   │   └── data_service.py    # Data management service
│   └── settings_service.py    # Settings management
├── 📁 utils/                  # Utility functions
│   ├── data_processor.py      # Data processing utilities
│   ├── error_handler.py       # Error handling
│   └── settings_utils.py      # Settings utilities
├── app.py                     # Main application setup
├── run_webapp.py             # Application runner
└── requirements.txt          # Dependencies
```

---

## 🏛️ Core Architecture

### Application Stack
```
┌─────────────────────────────────────────┐
│              FastAPI Server             │
│  ┌─────────────────────────────────────┐ │
│  │         Dash Application            │ │
│  │  ┌─────────────────────────────────┐│ │
│  │  │        Layout System            ││ │
│  │  │  ┌─────────────────────────────┐││ │
│  │  │  │      Callback System        │││ │
│  │  │  │  ┌─────────────────────────┐│││ │
│  │  │  │  │    Data Services        ││││ │
│  │  │  │  └─────────────────────────┘│││ │
│  │  │  └─────────────────────────────┘││ │
│  │  └─────────────────────────────────┘│ │
│  └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

### Key Components:

1. **FastAPI Server**: Main server handling HTTP requests and API endpoints
2. **Dash Application**: Interactive web interface with real-time updates
3. **Layout System**: Modular UI components and page layouts
4. **Callback System**: Event-driven logic for user interactions
5. **Data Services**: Centralized data management and processing

---

## 🎨 Layout System

### Layout Hierarchy

```
app.layout
├── Global Stores (dcc.Store components)
├── Location (dcc.Location for routing)
├── Header Component
├── Sidebar Component
└── Page Content (dynamically updated)
    ├── Time Domain Analysis
    ├── Frequency Domain Analysis
    ├── Filtering Interface
    ├── Physiological Analysis
    ├── Respiratory Analysis
    ├── Feature Engineering
    └── Settings
```

### Layout Components

#### 1. **Common Components** (`layout/common/`)

**Header** (`header.py`):
- Application title and branding
- Theme toggle functionality
- User information display

**Sidebar** (`sidebar.py`):
- Navigation menu with icons
- Collapsible design
- Page routing links
- Organized by analysis categories

**Footer** (`footer.py`):
- Copyright information
- Version details
- Additional links

#### 2. **Page Layouts** (`layout/pages/`)

**Analysis Pages** (`analysis_pages.py`):
- `time_domain_layout()`: Time domain analysis interface
- `frequency_layout()`: Frequency domain analysis interface
- `filtering_layout()`: Advanced filtering interface
- `physiological_layout()`: Physiological feature extraction
- `respiratory_layout()`: Respiratory analysis tools
- `features_layout()`: Feature engineering interface
- `quality_layout()`: Signal quality assessment
- `advanced_layout()`: Advanced computational methods
- `health_report_layout()`: Health report generation
- `settings_layout()`: Application settings

**Upload Page** (`upload_page.py`):
- File upload interface
- Data configuration options
- Preview functionality

### Layout Features

- **Responsive Design**: Bootstrap-based responsive grid system
- **Interactive Controls**: Real-time parameter adjustment
- **Data Visualization**: Plotly-based interactive plots
- **Modular Structure**: Reusable components across pages
- **Consistent Styling**: Centralized CSS and theme management

---

## ⚡ Callback System

### Callback Architecture

The callback system follows a modular approach with clear separation of concerns and extensive integration with vitalDSP functions:

```
Callback Registration (app.py)
├── Core Callbacks
│   ├── Sidebar callbacks (app_callbacks.py)
│   ├── Page routing (page_routing_callbacks.py)
│   └── Upload handling (upload_callbacks.py)
├── Analysis Callbacks
│   ├── Time domain analysis (vitaldsp_callbacks.py)
│   ├── Frequency analysis (frequency_filtering_callbacks.py)
│   ├── Signal filtering (signal_filtering_callbacks.py)
│   ├── Respiratory analysis (respiratory_callbacks.py)
│   ├── Physiological features (physiological_callbacks.py)
│   ├── Signal quality (quality_callbacks.py)
│   ├── Advanced computations (advanced_callbacks.py)
│   └── Health reporting (health_report_callbacks.py)
└── Feature Callbacks
    ├── Feature engineering (features_callbacks.py)
    └── Data preview (preview_callbacks.py)
```

---

## 🔬 **Detailed Callback Analysis**

### **1. Time Domain Analysis Callbacks** (`vitaldsp_callbacks.py`)

#### **Main Callback: `analyze_time_domain()`**

**Purpose**: Comprehensive time domain signal analysis with real-time visualization and vitalDSP integration.

**File Location**: `src/vitalDSP_webapp/callbacks/analysis/vitaldsp_callbacks.py`

**Callback Registration** (Lines 3451-3486):
```python
def register_vitaldsp_callbacks(app):
    """Register all vitalDSP analysis callbacks."""

    @app.callback(
        [
            Output("main-signal-plot", "figure"),           # Line 3456: Main signal visualization
            Output("filtered-signal-plot", "figure"),       # Line 3457: Filtered signal plot
            Output("analysis-results", "children"),         # Line 3458: Analysis results text
            Output("peak-analysis-table", "children"),      # Line 3459: Peak analysis table
            Output("signal-quality-table", "children"),     # Line 3460: Signal quality metrics
            Output("filtering-results-table", "children"),  # Line 3461: Filtering results
            Output("additional-metrics-table", "children"), # Line 3462: Additional metrics
            Output("store-time-domain-data", "data"),       # Line 3463: Data store for persistence
            Output("store-filtered-data", "data"),          # Line 3464: Filtered data store
        ],
        [
            Input("btn-update-analysis", "n_clicks"),       # Line 3467: Update button trigger
            Input("time-range-slider", "value"),            # Line 3468: Time range slider
            Input("start-time", "value"),                   # Line 3469: Manual start time input
            Input("end-time", "value"),                     # Line 3470: Manual end time input
            Input("btn-nudge-m10", "n_clicks"),             # Line 3471: Nudge -10s button
            Input("btn-nudge-m1", "n_clicks"),              # Line 3472: Nudge -1s button
            Input("btn-nudge-p1", "n_clicks"),              # Line 3473: Nudge +1s button
            Input("btn-nudge-p10", "n_clicks"),             # Line 3474: Nudge +10s button
            Input("url", "pathname"),                       # Line 3475: URL pathname for routing
        ],
        [
            State("filter-family", "value"),                # Line 3478: Filter family selection
            State("filter-response", "value"),              # Line 3479: Filter response type
            State("filter-low-freq", "value"),              # Line 3480: Low frequency cutoff
            State("filter-high-freq", "value"),             # Line 3481: High frequency cutoff
            State("filter-order", "value"),                 # Line 3482: Filter order
            State("analysis-options", "value"),             # Line 3483: Analysis options
            State("signal-type-select", "value"),           # Line 3484: Signal type selection
        ],
    )
```

**Function Definition** (Lines 3487-3504):
```python
def analyze_time_domain(
    n_clicks,           # Button click count
    slider_value,       # Time range slider value [start, end]
    start_time,         # Manual start time input
    end_time,          # Manual end time input
    nudge_m10,         # Nudge -10s button clicks
    nudge_m1,          # Nudge -1s button clicks
    nudge_p1,          # Nudge +1s button clicks
    nudge_p10,         # Nudge +10s button clicks
    pathname,          # Current URL pathname
    filter_family,     # Filter family (Butterworth, Chebyshev, etc.)
    filter_response,   # Filter response (lowpass, highpass, etc.)
    filter_low_freq,   # Low frequency cutoff
    filter_high_freq,  # High frequency cutoff
    filter_order,      # Filter order
    analysis_options,  # Analysis options list
    signal_type,       # Signal type (PPG, ECG, etc.)
):
```

**Detailed Line-by-Line Logic Analysis**:

#### **Phase 1: Initialization and Trigger Detection** (Lines 3505-3534)

**Lines 3505-3509**: Logging and Input Validation
```python
logger.info("=== TIME DOMAIN ANALYSIS CALLBACK ===")
logger.info(f"Input values - start_time: {start_time}, end_time: {end_time}, slider_value: {slider_value}")
```
- **Purpose**: Initialize logging and log input parameters for debugging
- **Logic**: Captures all input values to track callback execution

**Lines 3511-3516**: Trigger Context Analysis
```python
ctx = callback_context
if not ctx.triggered:
    trigger_id = "initial_load"
else:
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
```
- **Purpose**: Determine what triggered the callback execution
- **Logic**: 
  - If no trigger context exists, assume initial page load
  - Otherwise, extract the component ID that triggered the callback
  - This enables different behavior based on user interaction

**Lines 3518-3519**: Logging Trigger Information
```python
logger.info(f"Trigger ID: {trigger_id}")
logger.info(f"Pathname: {pathname}")
```
- **Purpose**: Debug information about callback trigger source

**Lines 3521-3534**: Page Route Validation
```python
if pathname != "/time-domain":
    logger.info("Not on time domain page, returning empty figures")
    return (
        create_empty_figure(),
        create_empty_figure(),
        "Navigate to Time Domain Analysis page",
        # ... more empty outputs
    )
```
- **Purpose**: Ensure callback only runs on the correct page
- **Logic**: If not on time-domain page, return empty figures and navigation message
- **Performance**: Prevents unnecessary processing when on other pages

#### **Phase 2: Data Service Integration** (Lines 3540-3594)

**Lines 3540-3546**: Data Service Initialization
```python
try:
    logger.info("Attempting to get data service...")
    from vitalDSP_webapp.services.data.data_service import get_data_service
    data_service = get_data_service()
    logger.info("Data service retrieved successfully")
```
- **Purpose**: Initialize connection to data service
- **vitalDSP Integration**: Uses vitalDSP webapp data service for data management
- **Error Handling**: Wrapped in try-catch for graceful failure

**Lines 3548-3567**: Data Retrieval and Validation
```python
all_data = data_service.get_all_data()
logger.info(f"All data keys: {list(all_data.keys()) if all_data else 'None'}")

if not all_data:
    logger.warning("No data found in service")
    return (
        create_empty_figure(),
        # ... error message outputs
    )
```
- **Purpose**: Retrieve all stored data and validate availability
- **Logic**: 
  - Get all data from service
  - Check if data exists
  - Return error state if no data available
- **User Experience**: Provides clear feedback when no data is loaded

**Lines 3569-3594**: Latest Data Selection and Column Mapping
```python
latest_data_id = list(all_data.keys())[-1]
latest_data = all_data[latest_data_id]
column_mapping = data_service.get_column_mapping(latest_data_id)

if not column_mapping:
    logger.warning("Data has not been processed yet - no column mapping found")
    return (
        # ... error message for unprocessed data
    )
```
- **Purpose**: Select most recent data and validate processing status
- **Logic**:
  - Get the last (most recent) data entry
  - Retrieve column mapping for data interpretation
  - Validate that data has been processed (has column mapping)
- **Error Handling**: Graceful failure with informative messages

#### **Phase 3: Data Frame Processing** (Lines 3596-3618)

**Lines 3596-3604**: Data Frame Retrieval and Validation
```python
df = data_service.get_data(latest_data_id)
logger.info(f"Data frame shape: {df.shape if df is not None else 'None'}")
logger.info(f"Data frame columns: {list(df.columns) if df is not None else 'None'}")
logger.info(f"Data frame info: {df.info() if df is not None else 'None'}")
logger.info(f"Data frame head: {df.head() if df is not None else 'None'}")

if df is None or df.empty:
    logger.warning("Data frame is empty")
    return (
        # ... error message for empty data
    )
```
- **Purpose**: Retrieve actual data frame and validate content
- **Logic**:
  - Get data frame from service
  - Log comprehensive data information for debugging
  - Validate data frame is not None or empty
- **Debugging**: Extensive logging for troubleshooting data issues

**Lines 3620-3622**: Sampling Frequency Extraction
```python
sampling_freq = latest_data.get("info", {}).get("sampling_freq", 1000)
logger.info(f"Sampling frequency: {sampling_freq}")
```
- **Purpose**: Extract sampling frequency for signal processing
- **Logic**: Get from data info with default fallback to 1000 Hz
- **vitalDSP Integration**: Critical for proper signal analysis

#### **Phase 4: Time Window Management** (Lines 3624-3699)

**Lines 3624-3647**: Nudge Button Logic
```python
if trigger_id in ["btn-nudge-m10", "btn-nudge-m1", "btn-nudge-p1", "btn-nudge-p10"]:
    if not start_time or not end_time:
        start_time, end_time = 0, 10

    if trigger_id == "btn-nudge-m10":
        start_time = max(0, start_time - 10)
        end_time = max(10, end_time - 10)
    elif trigger_id == "btn-nudge-m1":
        start_time = max(0, start_time - 1)
        end_time = max(1, end_time - 1)
    elif trigger_id == "btn-nudge-p1":
        start_time = start_time + 1
        end_time = end_time + 1
    elif trigger_id == "btn-nudge-p10":
        start_time = start_time + 10
        end_time = end_time + 10
```
- **Purpose**: Handle time window adjustments via nudge buttons
- **Logic**:
  - Check if trigger is a nudge button
  - Set default time window if not specified
  - Adjust time window based on button pressed
  - Ensure minimum window size and non-negative start time
- **User Experience**: Intuitive time navigation controls

**Lines 3649-3659**: Slider and Manual Input Handling
```python
elif trigger_id == "time-range-slider" and slider_value:
    start_time = slider_value[0]
    end_time = slider_value[1]
    logger.info(f"Time range slider changed: {start_time} to {end_time}")

elif trigger_id in ["start-time", "end-time"]:
    logger.info(f"Manual time input changed - start_time: {start_time}, end_time: {end_time}")
```
- **Purpose**: Handle different time input methods
- **Logic**:
  - Slider provides [start, end] array
  - Manual inputs provide individual values
  - Log changes for debugging

**Lines 3661-3675**: Time Value Validation and Conversion
```python
if not start_time or not end_time:
    start_time, end_time = 0, 10
    logger.info(f"Using default time window: {start_time} to {end_time}")

try:
    start_time = float(start_time) if start_time is not None else 0
    end_time = float(end_time) if end_time is not None else 10
    logger.info(f"Converted time values: start_time={start_time:.3f}, end_time={end_time:.3f}")
except (ValueError, TypeError):
    start_time, end_time = 0, 10
    logger.warning("Invalid time values, using defaults")
```
- **Purpose**: Ensure time values are valid numbers
- **Logic**:
  - Set defaults if values are None or empty
  - Convert to float with error handling
  - Fallback to defaults on conversion error
- **Robustness**: Handles various input formats and edge cases

**Lines 3677-3699**: Sample Index Calculation and Bounds Checking
```python
start_sample = int(start_time * sampling_freq)
end_sample = int(end_time * sampling_freq)
logger.info(f"Sample range: {start_sample} to {end_sample}")

if start_sample >= len(df):
    logger.warning(f"Start sample {start_sample} >= data length {len(df)}, adjusting to 0")
    start_sample = 0
    start_time = 0
if end_sample > len(df):
    logger.warning(f"End sample {end_sample} > data length {len(df)}, adjusting to {len(df)}")
    end_sample = len(df)
    end_time = len(df) / sampling_freq

windowed_data = df.iloc[start_sample:end_sample].copy()
```
- **Purpose**: Convert time to sample indices and apply windowing
- **Logic**:
  - Convert time to sample indices using sampling frequency
  - Check bounds and adjust if necessary
  - Extract windowed data using pandas iloc
- **Data Integrity**: Ensures indices are within data bounds

#### **Phase 5: Signal Processing and Analysis** (Lines 3700+)

**Lines 3703-3718**: Time Axis Creation and Validation
```python
time_axis = np.linspace(start_time, end_time, len(windowed_data))
logger.info(f"Time axis shape: {time_axis.shape}")
logger.info(f"Time axis range: {time_axis[0]:.3f} to {time_axis[-1]:.3f}")

logger.info(f"Original data length: {len(df)} samples")
logger.info(f"Windowed data length: {len(windowed_data)} samples")
logger.info(f"Expected samples for {end_time - start_time:.3f}s window: {(end_time - start_time) * sampling_freq:.0f}")
```
- **Purpose**: Create time axis for plotting and validate windowing
- **Logic**:
  - Create linear time axis from start to end time
  - Log comprehensive information for debugging
  - Validate expected vs actual sample counts
- **Debugging**: Extensive logging for troubleshooting time windowing issues

**Lines 3720-3749**: Signal Column Detection and Validation
```python
signal_column = column_mapping.get("signal")
logger.info(f"Signal column from mapping: {signal_column}")

if not signal_column or signal_column not in windowed_data.columns:
    logger.warning(f"Signal column {signal_column} not found in data")
    potential_signal_cols = ["waveform", "pleth", "pl", "signal", "ppg", "ecg", "red", "ir"]
    for col in potential_signal_cols:
        if col in [c.lower() for c in windowed_data.columns]:
            signal_column = [c for c in windowed_data.columns if c.lower() == col][0]
            logger.info(f"Found alternative signal column: {signal_column}")
            break
```
- **Purpose**: Identify the correct signal column for analysis
- **Logic**:
  - First try column mapping
  - If not found, search for common signal column names
  - Case-insensitive matching for robustness
- **Flexibility**: Handles various data formats and column naming conventions

**vitalDSP Integration Points**:
- **Data Service**: `vitalDSP_webapp.services.data.data_service` (Line 3543)
- **Signal Processing**: Integration with vitalDSP filtering functions
- **Quality Assessment**: Uses vitalDSP signal quality modules
- **Preprocessing**: Leverages vitalDSP preprocessing capabilities

**Key Features**:
- **Real-time Updates**: Interactive time range selection with nudge controls
- **Multi-plot Visualization**: Original signal, filtered signal, and peak overlays
- **Comprehensive Tables**: Detailed analysis results in tabular format
- **Error Handling**: Graceful degradation with informative error messages
- **Robust Data Handling**: Extensive validation and fallback mechanisms

---

### **2. Signal Filtering Callbacks** (`signal_filtering_callbacks.py`)

#### **Main Callback: `advanced_filtering_callback()`**

**Purpose**: Advanced signal filtering with multiple filter types and vitalDSP integration.

**File Location**: `src/vitalDSP_webapp/callbacks/analysis/signal_filtering_callbacks.py`

**Callback Registration** (Lines 69-109):
```python
@app.callback(
    [
        Output("filter-original-plot", "figure"),           # Line 71: Original signal plot
        Output("filter-filtered-plot", "figure"),           # Line 72: Filtered signal plot
        Output("filter-comparison-plot", "figure"),         # Line 73: Comparison plot
        Output("filter-quality-metrics", "children"),       # Line 74: Quality metrics display
        Output("filter-quality-plots", "figure"),           # Line 75: Quality plots
        Output("store-filtering-data", "data"),             # Line 76: Filtering data store
    ],
    [
        Input("url", "pathname"),                           # Line 79: URL routing
        Input("filter-btn-apply", "n_clicks"),              # Line 80: Apply filter button
        Input("filter-time-range-slider", "value"),         # Line 81: Time range slider
        Input("filter-btn-nudge-m10", "n_clicks"),          # Line 82: Nudge -10s button
        Input("filter-btn-nudge-m1", "n_clicks"),           # Line 83: Nudge -1s button
        Input("filter-btn-nudge-p1", "n_clicks"),           # Line 84: Nudge +1s button
        Input("filter-btn-nudge-p10", "n_clicks"),          # Line 85: Nudge +10s button
    ],
    [
        State("filter-start-time", "value"),                # Line 88: Start time input
        State("filter-end-time", "value"),                  # Line 89: End time input
        State("filter-type-select", "value"),               # Line 90: Filter type selection
        State("filter-family-advanced", "value"),           # Line 91: Filter family
        State("filter-response-advanced", "value"),         # Line 92: Filter response type
        State("filter-low-freq-advanced", "value"),         # Line 93: Low frequency cutoff
        State("filter-high-freq-advanced", "value"),        # Line 94: High frequency cutoff
        State("filter-order-advanced", "value"),            # Line 95: Filter order
        State("advanced-filter-method", "value"),           # Line 96: Advanced method
        State("advanced-noise-level", "value"),             # Line 97: Noise level
        State("advanced-iterations", "value"),              # Line 98: Iterations
        State("advanced-learning-rate", "value"),           # Line 99: Learning rate
        State("artifact-type", "value"),                    # Line 100: Artifact type
        State("artifact-removal-strength", "value"),        # Line 101: Artifact strength
        State("neural-network-type", "value"),              # Line 102: Neural network type
        State("neural-model-complexity", "value"),          # Line 103: Model complexity
        State("ensemble-method", "value"),                  # Line 104: Ensemble method
        State("ensemble-n-filters", "value"),               # Line 105: Number of filters
        State("filter-quality-options", "value"),           # Line 106: Quality options
        State("detrend-option", "value"),                   # Line 107: Detrend option
    ],
)
```

**Detailed Line-by-Line Logic Analysis**:

#### **Phase 1: Callback Initialization** (Lines 140-164)

**Lines 140-143**: Context and Trigger Analysis
```python
ctx = callback_context
if not ctx.triggered:
    raise PreventUpdate
trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
```
- **Purpose**: Determine what triggered the callback and validate trigger context
- **Logic**: Get callback context, prevent update if no trigger, extract component ID

**Lines 150-160**: Page Route Validation
```python
if pathname != "/filtering":
    logger.info("Not on filtering page, returning empty figures")
    return (create_empty_figure(), create_empty_figure(), ...)
```
- **Purpose**: Ensure callback only runs on filtering page
- **Performance**: Prevents unnecessary processing on other pages

#### **Phase 2: Data Service Integration** (Lines 166-288)

**Lines 166-170**: Data Service Initialization
```python
from vitalDSP_webapp.services.data.data_service import get_data_service
data_service = get_data_service()
```
- **vitalDSP Integration**: Uses vitalDSP webapp data service
- **Error Handling**: Wrapped in try-catch for graceful failure

**Lines 172-189**: Data Retrieval and Validation
```python
all_data = data_service.get_all_data()
if not all_data:
    logger.warning("No data available for filtering")
    return (create_empty_figure(), ...)
```
- **Purpose**: Retrieve and validate data availability
- **User Experience**: Clear feedback when no data is loaded

#### **Phase 3: Filter Type Processing** (Lines 500+)

**Traditional Filters** (Lines 500-600):
```python
if filter_type == "traditional":
    if filter_family == "butterworth":
        b, a = signal.butter(filter_order, [low_freq, high_freq], btype=filter_response, fs=sampling_freq)
    elif filter_family == "chebyshev1":
        b, a = signal.cheby1(filter_order, 1, [low_freq, high_freq], btype=filter_response, fs=sampling_freq)
    filtered_signal = signal.filtfilt(b, a, signal_data)
```
- **Purpose**: Apply traditional digital filters (Butterworth, Chebyshev, Elliptic)
- **Logic**: Design filter coefficients and apply zero-phase filtering

**Advanced Filters** (Lines 600-700):
```python
elif filter_type == "advanced":
    if advanced_method == "kalman":
        from vitalDSP.filtering.kalman_filter import KalmanFilter
        kf = KalmanFilter(initial_state=np.mean(signal_data))
        filtered_signal = kf.filter(signal_data)
    elif advanced_method == "adaptive":
        from vitalDSP.filtering.adaptive_filter import AdaptiveFilter
        af = AdaptiveFilter(learning_rate=learning_rate, iterations=iterations)
        filtered_signal = af.filter(signal_data, noise_level=noise_level)
```
- **vitalDSP Integration**: Uses vitalDSP filtering modules
- **Purpose**: Apply advanced filtering techniques (Kalman, Adaptive, Wiener)

**Artifact Removal** (Lines 700-800):
```python
elif filter_type == "artifact":
    if artifact_type == "amplitude":
        from vitalDSP.signal_quality_assessment.artifact_detection_removal import ArtifactDetectionRemoval
        adr = ArtifactDetectionRemoval()
        filtered_signal, artifact_mask = adr.remove_amplitude_artifacts(signal_data, strength=artifact_strength)
```
- **vitalDSP Integration**: Uses vitalDSP artifact detection and preprocessing modules
- **Purpose**: Remove various types of artifacts (amplitude, statistical, baseline)

**Neural Network Filtering** (Lines 800-900):
```python
elif filter_type == "neural":
    from vitalDSP.advanced_computation.neural_filtering import NeuralFilter
    nf = NeuralFilter(network_type=neural_type, complexity=neural_complexity, noise_level=noise_level)
    filtered_signal = nf.filter(signal_data)
```
- **vitalDSP Integration**: Uses vitalDSP advanced computation modules
- **Purpose**: Apply neural network-based filtering

**Ensemble Methods** (Lines 900-1000):
```python
elif filter_type == "ensemble":
    from vitalDSP.filtering.ensemble_filtering import EnsembleFilter
    ef = EnsembleFilter(method=ensemble_method, n_filters=ensemble_n_filters, base_filters=["butterworth", "chebyshev", "elliptic"])
    filtered_signal = ef.filter(signal_data)
```
- **vitalDSP Integration**: Uses vitalDSP ensemble filtering modules
- **Purpose**: Apply ensemble filtering methods

**vitalDSP Integration Points**:
- **Filtering Functions**: `vitalDSP.filtering.*` modules (Lines 600-1000)
- **Quality Assessment**: `vitalDSP.signal_quality_assessment.*` modules (Lines 700-800)
- **Preprocessing**: `vitalDSP.preprocess.*` modules (Lines 700-800)
- **Advanced Computation**: `vitalDSP.advanced_computation.*` modules (Lines 800-900)

**Callback Signature**:
```python
@app.callback(
    [
        Output("filter-original-plot", "figure"),
        Output("filter-filtered-plot", "figure"),
        Output("filter-comparison-plot", "figure"),
        Output("filter-quality-metrics", "children"),
        Output("filter-quality-plots", "figure"),
        Output("store-filtering-data", "data"),
    ],
    [
        Input("url", "pathname"),
        Input("filter-btn-apply", "n_clicks"),
        Input("filter-time-range-slider", "value"),
        Input("filter-btn-nudge-m10", "n_clicks"),
        Input("filter-btn-nudge-m1", "n_clicks"),
        Input("filter-btn-nudge-p1", "n_clicks"),
        Input("filter-btn-nudge-p10", "n_clicks"),
    ],
    [
        State("filter-start-time", "value"),
        State("filter-end-time", "value"),
        State("filter-type-select", "value"),
        State("filter-family-advanced", "value"),
        State("filter-response-advanced", "value"),
        State("filter-low-freq-advanced", "value"),
        State("filter-high-freq-advanced", "value"),
        State("filter-order-advanced", "value"),
        State("advanced-filter-method", "value"),
        State("advanced-noise-level", "value"),
        State("advanced-iterations", "value"),
        State("advanced-learning-rate", "value"),
        State("artifact-type", "value"),
        State("artifact-removal-strength", "value"),
        State("neural-network-type", "value"),
        State("neural-model-complexity", "value"),
        State("ensemble-method", "value"),
        State("ensemble-n-filters", "value"),
        State("filter-quality-options", "value"),
        State("detrend-option", "value"),
    ],
)
```

**Filter Types and Logic**:

1. **Traditional Filters**:
   - Butterworth, Chebyshev, Elliptic filters
   - Low-pass, high-pass, band-pass, band-stop
   - Configurable order and frequency parameters

2. **Advanced Filters**:
   - Kalman filtering for noise reduction
   - Adaptive filtering with learning rate control
   - Wiener filtering for optimal noise reduction

3. **Artifact Removal**:
   - Amplitude-based artifact detection
   - Statistical outlier removal
   - Baseline wander correction

4. **Neural Network Filtering**:
   - Deep learning-based denoising
   - Configurable model complexity
   - Training with noise level parameters

5. **Ensemble Methods**:
   - Multiple filter combination
   - Weighted averaging
   - Voting-based selection

**vitalDSP Integration**:
- **Filtering Functions**: Uses vitalDSP filtering modules for advanced techniques
- **Quality Assessment**: Integrates with vitalDSP signal quality assessment
- **Preprocessing**: Leverages vitalDSP preprocessing functions

**Complex Logic**:
- **Dynamic Parameter Visibility**: Shows/hides parameters based on filter type
- **Multi-stage Processing**: Combines multiple filtering techniques
- **Quality Metrics**: Computes comprehensive filtering performance metrics
- **Real-time Visualization**: Updates plots with filtering results

---

### **3. Frequency Domain Analysis Callbacks** (`frequency_filtering_callbacks.py`)

#### **Main Callback: `frequency_domain_callback()`**

**Purpose**: Comprehensive frequency domain analysis with FFT, STFT, and wavelet transforms.

**Callback Signature**:
```python
@app.callback(
    [
        Output("freq-main-plot", "figure"),
        Output("freq-psd-plot", "figure"),
        Output("freq-spectrogram-plot", "figure"),
        Output("freq-analysis-results", "children"),
        Output("freq-peak-analysis-table", "children"),
        Output("freq-band-power-table", "children"),
        Output("freq-stability-table", "children"),
        Output("freq-harmonics-table", "children"),
        Output("store-frequency-data", "data"),
        Output("store-time-freq-data", "data"),
    ],
    [
        Input("url", "pathname"),
        Input("freq-btn-update-analysis", "n_clicks"),
        Input("freq-time-range-slider", "value"),
        Input("freq-btn-nudge-m10", "n_clicks"),
        Input("freq-btn-nudge-m1", "n_clicks"),
        Input("freq-btn-nudge-p1", "n_clicks"),
        Input("freq-btn-nudge-p10", "n_clicks"),
    ],
    [
        State("freq-start-time", "value"),
        State("freq-end-time", "value"),
        State("freq-analysis-type", "value"),
        State("fft-window-type", "value"),
        State("fft-n-points", "value"),
        # PSD Parameters
        State("psd-window", "value"),
        State("psd-overlap", "value"),
        State("psd-freq-max", "value"),
        State("psd-log-scale", "value"),
        State("psd-normalize", "value"),
        State("psd-channel", "value"),
        # STFT Parameters
        State("stft-window-size", "value"),
        State("stft-hop-size", "value"),
        State("stft-window-type", "value"),
        State("stft-overlap", "value"),
        State("stft-scaling", "value"),
        State("stft-freq-max", "value"),
        State("stft-colormap", "value"),
        State("wavelet-type", "value"),
        State("wavelet-levels", "value"),
        State("freq-min", "value"),
        State("freq-max", "value"),
        State("freq-analysis-options", "value"),
    ],
)
```

**Analysis Types**:

1. **FFT Analysis**:
   - Fast Fourier Transform with configurable window functions
   - Frequency domain representation
   - Peak detection in frequency domain

2. **Power Spectral Density (PSD)**:
   - Welch's method for PSD estimation
   - Configurable window size and overlap
   - Log scale and normalization options

3. **Short-Time Fourier Transform (STFT)**:
   - Time-frequency analysis
   - Configurable window size and hop length
   - Spectrogram visualization

4. **Wavelet Transforms**:
   - Continuous and discrete wavelet transforms
   - Multiple wavelet types (Haar, Daubechies, etc.)
   - Multi-resolution analysis

**vitalDSP Integration**:
- **Transform Functions**: Uses vitalDSP transform modules
- **Frequency Analysis**: Integrates with vitalDSP frequency domain analysis
- **Wavelet Processing**: Leverages vitalDSP wavelet transform functions

**Complex Logic**:
- **Multi-parameter Processing**: Handles numerous analysis parameters
- **Dynamic Visualization**: Updates multiple plots based on analysis type
- **Band Power Analysis**: Computes power in specific frequency bands
- **Harmonic Analysis**: Identifies and analyzes harmonic components

---

### **4. Respiratory Analysis Callbacks** (`respiratory_callbacks.py`)

#### **Main Callback: `respiratory_analysis_callback()`**

**Purpose**: Comprehensive respiratory rate estimation and breathing pattern analysis.

**vitalDSP Integration**:
```python
# Extensive vitalDSP module imports
from vitalDSP.respiratory_analysis.respiratory_analysis import RespiratoryAnalysis
from vitalDSP.respiratory_analysis.estimate_rr.peak_detection_rr import peak_detection_rr
from vitalDSP.respiratory_analysis.estimate_rr.fft_based_rr import fft_based_rr
from vitalDSP.respiratory_analysis.estimate_rr.frequency_domain_rr import frequency_domain_rr
from vitalDSP.respiratory_analysis.estimate_rr.time_domain_rr import time_domain_rr
from vitalDSP.respiratory_analysis.sleep_apnea_detection.amplitude_threshold import detect_apnea_amplitude
from vitalDSP.respiratory_analysis.sleep_apnea_detection.pause_detection import detect_apnea_pauses
from vitalDSP.respiratory_analysis.fusion.multimodal_analysis import multimodal_analysis
from vitalDSP.respiratory_analysis.fusion.ppg_ecg_fusion import ppg_ecg_fusion
from vitalDSP.respiratory_analysis.fusion.respiratory_cardiac_fusion import respiratory_cardiac_fusion
from vitalDSP.preprocess.preprocess_operations import PreprocessConfig, preprocess_signal
```

**Analysis Methods**:
1. **Peak Detection RR**: Time-domain respiratory rate estimation
2. **FFT-based RR**: Frequency-domain respiratory rate estimation
3. **Frequency Domain RR**: Advanced frequency analysis
4. **Sleep Apnea Detection**: Amplitude and pause-based detection
5. **Multimodal Analysis**: Fusion of multiple signal sources
6. **PPG-ECG Fusion**: Combined PPG and ECG analysis
7. **Respiratory-Cardiac Fusion**: Integration of respiratory and cardiac signals

**Complex Logic**:
- **Multi-method Integration**: Combines multiple vitalDSP respiratory analysis methods
- **Error Handling**: Graceful fallback when vitalDSP modules are unavailable
- **Advanced Options**: Sleep apnea detection, multimodal analysis, fusion techniques
- **Real-time Processing**: Updates respiratory rate estimates in real-time

---

### **5. Physiological Features Callbacks** (`physiological_callbacks.py`)

#### **Main Callback: `physiological_analysis_callback()`**

**Purpose**: Comprehensive physiological feature extraction and analysis.

**vitalDSP Integration**:
```python
# Extensive vitalDSP physiological features imports
from vitalDSP.physiological_features import hrv_analysis, time_domain, frequency_domain
from vitalDSP.feature_engineering import ppg_light_features, ppg_autonomic_features, ecg_autonomic_features
from vitalDSP.signal_quality_assessment import signal_quality_index, artifact_detection_removal
from vitalDSP.transforms import wavelet_transform, fourier_transform, hilbert_transform
from vitalDSP.advanced_computation import anomaly_detection, bayesian_analysis, kalman_filter
```

**Feature Categories**:

1. **HRV Analysis**:
   - Time domain features (RMSSD, SDNN, pNN50)
   - Frequency domain features (LF, HF, LF/HF ratio)
   - Non-linear features (SD1, SD2, entropy)

2. **PPG Features**:
   - Light source features (red, IR, ratio)
   - Autonomic features (sympathetic, parasympathetic)
   - Morphological features (peaks, valleys, slopes)

3. **ECG Features**:
   - R-R interval analysis
   - P-QRS-T complex features
   - Autonomic nervous system indicators

4. **Signal Quality**:
   - Signal-to-noise ratio (SNR)
   - Artifact detection and removal
   - Baseline wander assessment

5. **Advanced Transformations**:
   - Wavelet transforms for multi-resolution analysis
   - Fourier transforms for frequency analysis
   - Hilbert transforms for instantaneous features

6. **Advanced Computation**:
   - Anomaly detection using machine learning
   - Bayesian analysis for uncertainty quantification
   - Kalman filtering for state estimation

**Complex Logic**:
- **Multi-signal Support**: Handles both PPG and ECG signals
- **Feature Engineering**: Extracts comprehensive feature sets
- **Quality Assessment**: Ensures signal quality before analysis
- **Advanced Processing**: Integrates machine learning and statistical methods

---

### **6. Signal Quality Assessment Callbacks** (`quality_callbacks.py`)

#### **Main Callback: `quality_assessment_callback()`**

**Purpose**: Comprehensive signal quality assessment and validation.

**vitalDSP Integration**:
```python
from vitalDSP.signal_quality_assessment.signal_quality_index import SignalQualityIndex
from vitalDSP.signal_quality_assessment.artifact_detection_removal import ArtifactDetectionRemoval
```

**Quality Metrics**:
1. **Signal-to-Noise Ratio (SNR)**:
   - Power-based SNR calculation
   - Frequency-domain SNR analysis
   - Adaptive SNR estimation

2. **Artifact Detection**:
   - Amplitude-based artifact detection
   - Statistical outlier detection
   - Machine learning-based artifact classification

3. **Baseline Wander**:
   - Low-frequency drift detection
   - Baseline correction algorithms
   - Trend analysis

4. **Signal Stability**:
   - Variance analysis
   - Stationarity tests
   - Consistency metrics

5. **Advanced Quality Assessment**:
   - Multi-dimensional quality scoring
   - Composite quality indices
   - Real-time quality monitoring

**Complex Logic**:
- **Multi-metric Analysis**: Combines multiple quality assessment methods
- **Threshold-based Classification**: Categorizes signal quality levels
- **Recommendation System**: Provides improvement suggestions
- **Real-time Monitoring**: Updates quality metrics continuously

---

### **7. Advanced Computation Callbacks** (`advanced_callbacks.py`)

**Purpose**: Advanced computational methods and machine learning integration.

**vitalDSP Integration**:
```python
from vitalDSP.advanced_computation.anomaly_detection import AnomalyDetection
from vitalDSP.advanced_computation.bayesian_analysis import GaussianProcess
from vitalDSP.advanced_computation.kalman_filter import KalmanFilter
```

**Advanced Methods**:
1. **Anomaly Detection**:
   - Statistical anomaly detection
   - Machine learning-based detection
   - Real-time anomaly scoring

2. **Bayesian Analysis**:
   - Gaussian process regression
   - Uncertainty quantification
   - Probabilistic modeling

3. **Kalman Filtering**:
   - State estimation
   - Noise reduction
   - Prediction and smoothing

---

### **Callback Registration**

Callbacks are registered in `app.py`:

```python
# Register callbacks AFTER app is created
register_sidebar_callbacks(app)
register_page_routing_callbacks(app)
register_upload_callbacks(app)
register_vitaldsp_callbacks(app)  # Time domain analysis
register_frequency_filtering_callbacks(app)  # Frequency analysis
register_signal_filtering_callbacks(app)  # Signal filtering
register_respiratory_callbacks(app)  # Respiratory analysis
register_physiological_callbacks(app)  # Physiological features
register_features_callbacks(app)  # Feature engineering
register_preview_callbacks(app)  # Data preview
```

### **Common Callback Patterns**

1. **Error Handling**: All callbacks include comprehensive error handling with graceful degradation
2. **Data Validation**: Extensive input validation and data integrity checks
3. **Real-time Updates**: Interactive parameter adjustment with immediate feedback
4. **vitalDSP Integration**: Deep integration with vitalDSP functions and modules
5. **Performance Optimization**: Efficient data processing and visualization updates
6. **User Experience**: Intuitive controls and informative feedback messages

---

## 🖥️ **Screen Descriptions and User Interface**

### **Screen Architecture Overview**

The vitalDSP webapp consists of multiple interconnected screens, each designed for specific analysis tasks. All screens share a common layout structure:

```
┌─────────────────────────────────────────────────────────┐
│                    Header (Fixed)                       │
├─────────────┬───────────────────────────────────────────┤
│   Sidebar   │              Main Content Area            │
│ (Collapsible)│         (Dynamic Content)                │
│             │                                           │
│             │                                           │
├─────────────┴───────────────────────────────────────────┤
│                    Footer (Fixed)                       │
└─────────────────────────────────────────────────────────┘
```

### **Common Screen Components**

#### **Header Component** (`layout/common/header.py`)
- **Application Title**: "Vital-DSP Comprehensive Dashboard"
- **Theme Toggle**: Light/Dark mode switching
- **User Information**: Display current user context
- **Navigation Breadcrumbs**: Show current page location

#### **Sidebar Component** (`layout/common/sidebar.py`)
- **Collapsible Design**: Expandable/collapsible navigation
- **Icon Navigation**: Compact view with tooltips
- **Page Categories**:
  - **Analysis**: Time Domain, Frequency Domain, Filtering
  - **Features**: Physiological, Respiratory, Advanced Features
  - **Other**: Preview, Settings
- **Active Page Highlighting**: Visual indication of current page

#### **Footer Component** (`layout/common/footer.py`)
- **Version Information**: Application version and build info
- **Copyright Notice**: Legal and attribution information
- **Additional Links**: Documentation, support, etc.

---

## 📱 **Individual Screen Descriptions**

### **1. Upload Screen** (`/upload`)

**Purpose**: Data ingestion and initial configuration for signal analysis.

**File Location**: `src/vitalDSP_webapp/layout/pages/upload_page.py`

**Layout Structure**:
```
┌─────────────────────────────────────────────────────────┐
│                    Page Header                          │
│              "📊 Data Upload"                          │
├─────────────────┬───────────────────────────────────────┤
│   File Upload   │        Data Configuration            │
│     Section     │           Section                    │
│                 │                                       │
│  - Drag & Drop  │  - Sampling Frequency                │
│  - File Browser │  - Time Unit Selection               │
│  - Progress Bar │  - Column Mapping                    │
│  - Status Info  │  - Data Preview                      │
│                 │  - Processing Options                │
├─────────────────┴───────────────────────────────────────┤
│                Data Preview Section                     │
│              (Table with sample data)                   │
└─────────────────────────────────────────────────────────┘
```

**Key Components**:

1. **File Upload Area** (Lines 45-83):
   - **Drag & Drop Interface**: Modern file upload with visual feedback
   - **Supported Formats**: CSV, TXT, MAT files
   - **File Size Limit**: 50MB maximum
   - **Upload Progress**: Real-time progress indication
   - **Status Messages**: Success/error feedback

2. **Data Configuration Panel** (Lines 100-200):
   - **Sampling Frequency Input**: Configure data sampling rate
   - **Time Unit Selection**: Seconds, milliseconds, minutes
   - **Column Mapping**: Auto-detect or manual column assignment
   - **Data Preview**: Table showing first few rows of uploaded data
   - **Processing Options**: Data cleaning and preprocessing settings

3. **Data Preview Table** (Lines 200-300):
   - **Sample Data Display**: First 10-20 rows of uploaded data
   - **Column Information**: Data types and statistics
   - **Validation Status**: Data quality indicators
   - **Edit Capabilities**: Modify column mappings if needed

**User Workflow**:
1. **Upload File**: Drag & drop or browse for data file
2. **Configure Parameters**: Set sampling frequency and time units
3. **Map Columns**: Assign time and signal columns
4. **Preview Data**: Review uploaded data in table format
5. **Process Data**: Apply preprocessing and validation
6. **Navigate to Analysis**: Proceed to analysis screens

**vitalDSP Integration**:
- **Data Service**: `vitalDSP_webapp.services.data.data_service`
- **Data Processing**: `vitalDSP_webapp.utils.data_processor`
- **Column Detection**: Automatic signal/time column identification
- **Data Validation**: Quality checks and format validation

#### **Callbacks Used by Upload Screen**

**File Location**: `src/vitalDSP_webapp/callbacks/core/upload_callbacks.py`

**Main Callback: `handle_all_uploads()`** (Lines 100-1048)
```python
@app.callback(
    [
        Output("upload-status", "children"),
        Output("store-uploaded-data", "data"),
        Output("store-data-config", "data"),
        Output("data-preview-section", "children"),
        Output("time-column", "options"),
        Output("signal-column", "options"),
        # ... more outputs
    ],
    [
        Input("upload-data", "contents"),
        Input("btn-load-path", "n_clicks"),
        Input("btn-load-sample", "n_clicks"),
    ],
    [
        State("upload-data", "filename"),
        State("file-path-input", "value"),
        State("sampling-freq", "value"),
        State("time-unit", "value"),
    ],
)
```

**Purpose**: Handles all data upload methods (file upload, path loading, sample data)

**Key Logic**:
1. **File Upload Processing** (Lines 150-300):
   - Decode base64 file content
   - Validate file format (CSV, TXT, MAT)
   - Parse file using appropriate parser
   - Store data in data service

2. **Path Loading** (Lines 300-500):
   - Load data from file path
   - Validate file existence and format
   - Process and store data

3. **Sample Data Loading** (Lines 500-700):
   - Load predefined sample datasets
   - Configure sample data parameters
   - Store in data service

4. **Data Processing** (Lines 700-900):
   - Auto-detect column types
   - Generate column options for dropdowns
   - Create data preview table
   - Store configuration data

**Maintenance Notes**:
- **File Format Support**: Add new file formats in the parsing section (Lines 200-250)
- **Column Detection**: Modify auto-detection logic in `_detect_columns()` function
- **Data Validation**: Update validation rules in `_validate_data()` function
- **Error Handling**: Add new error types in the exception handling blocks

**Dependencies**:
- `vitalDSP_webapp.services.data.data_service`
- `vitalDSP_webapp.utils.data_processor`
- `pandas`, `numpy`, `scipy`

**Common Issues & Fixes**:
- **Memory Issues**: Large files may cause memory problems - implement chunked processing
- **Column Detection**: May fail with unusual column names - add more detection patterns
- **File Format**: New formats may not be supported - add format-specific parsers

---

### **2. Time Domain Analysis Screen** (`/time-domain`)

**Purpose**: Comprehensive time domain signal analysis with real-time visualization.

**File Location**: `src/vitalDSP_webapp/layout/pages/analysis_pages.py` (Lines 11-1583)

**Layout Structure**:
```
┌─────────────────────────────────────────────────────────┐
│                    Page Header                          │
│           "⏱️ Time Domain Analysis"                    │
├─────────────────────────────────────────────────────────┤
│                Action Buttons Row                       │
│  [🔄 Update Analysis] [📊 Export Results] [🎯 Dashboard] │
├─────────────┬───────────────────────────────────────────┤
│   Controls  │              Visualization Area           │
│   Panel     │                                           │
│             │  ┌─────────────────────────────────────┐  │
│  - Data     │  │        Main Signal Plot             │  │
│    Selection│  │                                     │  │
│  - Signal   │  └─────────────────────────────────────┘  │
│    Type     │  ┌─────────────────────────────────────┐  │
│  - Time     │  │      Filtered Signal Plot           │  │
│    Window   │  │                                     │  │
│  - Filter   │  └─────────────────────────────────────┘  │
│    Settings │                                           │
│  - Analysis │                Results Tables            │
│    Options  │  ┌─────────┬─────────┬─────────┬───────┐ │
│             │  │  Peak   │ Quality │Filtering│Metrics│ │
│             │  │Analysis │ Metrics │ Results │       │ │
│             │  └─────────┴─────────┴─────────┴───────┘ │
└─────────────┴───────────────────────────────────────────┘
```

**Key Components**:

1. **Action Buttons Row** (Lines 30-89):
   - **Update Analysis Button**: Trigger analysis computation
   - **Export Results Button**: Export analysis results
   - **Comprehensive Dashboard Button**: Generate detailed report

2. **Controls Panel** (Lines 94-500):
   - **Data Selection**: Choose between uploaded or sample data
   - **Signal Type Selection**: PPG or ECG signal type
   - **Time Window Controls**: Start/end time inputs with nudge buttons
   - **Signal Source Selection**: Choose between original or filtered signal
   - **Analysis Options**: Selectable analysis features

3. **Visualization Area** (Lines 500-800):
   - **Main Signal Plot**: Selected signal (original or filtered) with peak detection
   - **Signal Comparison Plot**: Side-by-side comparison of original vs filtered (if available)
   - **Interactive Controls**: Zoom, pan, time range selection
   - **Real-time Updates**: Dynamic plot updates based on parameters

4. **Results Tables** (Lines 800-1000):
   - **Peak Analysis Table**: Peak detection results and statistics
   - **Signal Quality Table**: SNR, artifact ratio, quality metrics
   - **Signal Source Table**: Information about signal source and processing
   - **Additional Metrics Table**: Advanced analysis results

**Interactive Features**:
- **Time Range Slider**: Interactive time window selection
- **Nudge Buttons**: Quick time window adjustment (-10s, -1s, +1s, +10s)
- **Signal Source Toggle**: Switch between original and filtered signal
- **Peak Detection**: Automatic and manual peak identification
- **Export Capabilities**: Save plots and data in multiple formats

**vitalDSP Integration**:
- **Peak Detection**: Custom algorithms with vitalDSP functions
- **Quality Assessment**: `vitalDSP.signal_quality_assessment.*`
- **Data Management**: `vitalDSP_webapp.services.data.data_service`
- **Filtered Data Loading**: Load pre-filtered data from filtering screen

#### **Callbacks Used by Time Domain Analysis Screen**

**File Location**: `src/vitalDSP_webapp/callbacks/analysis/vitaldsp_callbacks.py`

**Main Callback: `analyze_time_domain()`** (Lines 3487-4720)
```python
@app.callback(
    [
        Output("main-signal-plot", "figure"),
        Output("signal-comparison-plot", "figure"),
        Output("analysis-results", "children"),
        Output("peak-analysis-table", "children"),
        Output("signal-quality-table", "children"),
        Output("signal-source-table", "children"),
        Output("additional-metrics-table", "children"),
        Output("store-time-domain-data", "data"),
        Output("store-analysis-results", "data"),
    ],
    [
        Input("btn-update-analysis", "n_clicks"),
        Input("time-range-slider", "value"),
        Input("start-time", "value"),
        Input("end-time", "value"),
        Input("btn-nudge-m10", "n_clicks"),
        Input("btn-nudge-m1", "n_clicks"),
        Input("btn-nudge-p1", "n_clicks"),
        Input("btn-nudge-p10", "n_clicks"),
        Input("url", "pathname"),
    ],
    [
        State("signal-source-select", "value"),
        State("analysis-options", "value"),
        State("signal-type-select", "value"),
    ],
)
```

**Purpose**: Comprehensive time domain analysis with real-time visualization

**Key Logic**:
1. **Data Retrieval** (Lines 3540-3594):
   - Get data from DataService
   - Validate data availability and format
   - Extract column mapping and metadata

2. **Signal Source Selection** (Lines 3595-3650):
   - Check for filtered data from filtering screen
   - Load filtered signal if available
   - Fallback to original signal if no filtering performed
   - Update signal source information

3. **Time Window Management** (Lines 3651-3720):
   - Handle nudge button interactions
   - Process slider and manual input changes
   - Apply time window to selected signal

4. **Analysis Computation** (Lines 3721-4500):
   - Perform peak detection algorithms
   - Calculate signal statistics and quality metrics
   - Generate analysis results tables
   - Create visualization plots

**Maintenance Notes**:
- **Signal Source Loading**: Modify filtered data loading logic in the signal source section (Lines 3595-3650)
- **Peak Detection**: Update peak detection algorithms in the peak analysis section (Lines 3900-4000)
- **Quality Metrics**: Add new quality metrics in the quality assessment section (Lines 4000-4100)
- **Visualization**: Update plot configurations in the plotting section (Lines 4100-4500)

**Dependencies**:
- `vitalDSP_webapp.services.data.data_service`
- `plotly` for visualization
- `numpy`, `pandas` for data processing

**Common Issues & Fixes**:
- **Memory Issues**: Large datasets may cause memory problems - implement data chunking
- **Peak Detection**: May miss peaks with unusual signal characteristics - adjust detection parameters
- **Filtered Data Loading**: May fail to load filtered data - add error handling and fallback
- **Performance**: Slow processing with large datasets - optimize algorithms or add progress indicators

**Secondary Callbacks**:

**`update_time_range_slider()`** (Lines 4721-4800)
```python
@app.callback(
    Output("time-range-slider", "value"),
    [Input("start-time", "value"), Input("end-time", "value")],
    prevent_initial_call=True,
)
```
- **Purpose**: Synchronize time range slider with manual inputs
- **Maintenance**: Update slider bounds and step size as needed

**`update_signal_source_options()`** (Lines 4801-4900)
```python
@app.callback(
    Output("signal-source-select", "options"),
    [Input("url", "pathname")],
)
```
- **Purpose**: Update signal source options based on available filtered data
- **Maintenance**: Add new signal source types and their corresponding options

---

### **3. Frequency Domain Analysis Screen** (`/frequency`)

**Purpose**: Frequency domain analysis with FFT, STFT, and wavelet transforms.

**File Location**: `src/vitalDSP_webapp/layout/pages/analysis_pages.py` (Lines 2000-3000)

**Layout Structure**:
```
┌─────────────────────────────────────────────────────────┐
│                    Page Header                          │
│         "🌊 Frequency Domain Analysis"                 │
├─────────────────────────────────────────────────────────┤
│                Analysis Controls                        │
│  [Analysis Type] [Parameters] [Time Range] [Update]    │
├─────────────┬───────────────────────────────────────────┤
│   Settings  │              Visualization Area           │
│   Panel     │                                           │
│             │  ┌─────────────────────────────────────┐  │
│  - Analysis │  │        Main Frequency Plot          │  │
│    Type     │  │         (FFT/PSD/STFT)             │  │
│  - Window   │  └─────────────────────────────────────┘  │
│    Settings │  ┌─────────────────────────────────────┐  │
│  - Frequency│  │        Power Spectral Density       │  │
│    Range    │  │                                     │  │
│  - Advanced │  └─────────────────────────────────────┘  │
│    Options  │  ┌─────────────────────────────────────┐  │
│             │  │         Spectrogram/STFT            │  │
│             │  │                                     │  │
│             │  └─────────────────────────────────────┘  │
│             │                Analysis Tables            │
│             │  ┌─────────┬─────────┬─────────┬───────┐ │
│             │  │  Peak   │  Band   │Stability│Harmon.│ │
│             │  │Analysis │  Power  │ Metrics │       │ │
│             │  └─────────┴─────────┴─────────┴───────┘ │
└─────────────┴───────────────────────────────────────────┘
```

**Key Components**:

1. **Analysis Type Selection**:
   - **FFT Analysis**: Fast Fourier Transform
   - **PSD Analysis**: Power Spectral Density
   - **STFT Analysis**: Short-Time Fourier Transform
   - **Wavelet Analysis**: Wavelet transforms

2. **Parameter Controls**:
   - **Window Functions**: Hann, Hamming, Blackman, etc.
   - **Frequency Range**: Min/max frequency selection
   - **Window Size**: FFT window size configuration
   - **Overlap Settings**: STFT overlap parameters

3. **Visualization Plots**:
   - **Main Frequency Plot**: FFT magnitude spectrum
   - **PSD Plot**: Power spectral density with Welch's method
   - **Spectrogram**: Time-frequency representation
   - **Interactive Controls**: Frequency range selection, zoom

4. **Analysis Results**:
   - **Peak Analysis Table**: Frequency peaks and harmonics
   - **Band Power Table**: Power in specific frequency bands
   - **Stability Metrics**: Frequency stability analysis
   - **Harmonics Table**: Harmonic analysis results

**vitalDSP Integration**:
- **Transform Functions**: `vitalDSP.transforms.*` modules
- **Frequency Analysis**: `vitalDSP.physiological_features.frequency_domain`
- **Wavelet Processing**: `vitalDSP.transforms.wavelet_transform`

#### **Callbacks Used by Frequency Domain Analysis Screen**

**File Location**: `src/vitalDSP_webapp/callbacks/analysis/frequency_filtering_callbacks.py`

**Main Callback: `frequency_domain_callback()`** (Lines 106-2000)
```python
@app.callback(
    [
        Output("freq-main-plot", "figure"),
        Output("freq-psd-plot", "figure"),
        Output("freq-spectrogram-plot", "figure"),
        Output("freq-analysis-results", "children"),
        Output("freq-peak-analysis-table", "children"),
        Output("freq-band-power-table", "children"),
        Output("freq-stability-table", "children"),
        Output("freq-harmonics-table", "children"),
        Output("store-frequency-data", "data"),
        Output("store-time-freq-data", "data"),
    ],
    [
        Input("url", "pathname"),
        Input("freq-btn-update-analysis", "n_clicks"),
        Input("freq-time-range-slider", "value"),
        Input("freq-btn-nudge-m10", "n_clicks"),
        Input("freq-btn-nudge-m1", "n_clicks"),
        Input("freq-btn-nudge-p1", "n_clicks"),
        Input("freq-btn-nudge-p10", "n_clicks"),
    ],
    [
        State("freq-start-time", "value"),
        State("freq-end-time", "value"),
        State("freq-analysis-type", "value"),
        State("fft-window-type", "value"),
        State("fft-n-points", "value"),
        # ... more states
    ],
)
```

**Purpose**: Comprehensive frequency domain analysis with multiple transform methods

**Key Logic**:
1. **Analysis Type Selection** (Lines 300-500):
   - FFT Analysis: Fast Fourier Transform
   - PSD Analysis: Power Spectral Density
   - STFT Analysis: Short-Time Fourier Transform
   - Wavelet Analysis: Wavelet transforms

2. **Parameter Processing** (Lines 500-800):
   - Window function selection
   - Frequency range configuration
   - Window size and overlap settings
   - Advanced parameter handling

3. **Transform Computation** (Lines 800-1200):
   - Apply selected transform method
   - Compute frequency domain features
   - Generate time-frequency representations
   - Calculate spectral characteristics

4. **Results Generation** (Lines 1200-1500):
   - Create visualization plots
   - Generate analysis tables
   - Compute statistical metrics
   - Store results for further analysis

**Maintenance Notes**:
- **New Transform Methods**: Add new analysis types in the analysis type selection section (Lines 300-400)
- **Window Functions**: Add new window functions in the window processing section (Lines 500-600)
- **Visualization**: Update plot configurations in the plotting section (Lines 1200-1400)
- **Metrics**: Add new frequency domain metrics in the metrics calculation section (Lines 1400-1500)

**Dependencies**:
- `vitalDSP.transforms.*` modules
- `scipy.fft` for FFT operations
- `scipy.signal` for PSD and STFT
- `plotly` for visualization

**Common Issues & Fixes**:
- **Memory Issues**: Large FFT sizes may cause memory problems - implement chunked processing
- **Window Effects**: Window function selection may affect results - add window effect visualization
- **Frequency Resolution**: Trade-off between time and frequency resolution - add parameter guidance
- **Artifacts**: Spectral leakage may occur - add windowing options and explanations

**Secondary Callbacks**:

**`update_frequency_parameters()`** (Lines 2001-2100)
```python
@app.callback(
    [
        Output("fft-parameters-container", "style"),
        Output("psd-parameters-container", "style"),
        Output("stft-parameters-container", "style"),
        Output("wavelet-parameters-container", "style"),
    ],
    [Input("freq-analysis-type", "value")],
)
```
- **Purpose**: Show/hide parameter controls based on analysis type
- **Maintenance**: Add new analysis types and their parameter containers

**`update_frequency_range_slider()`** (Lines 2101-2200)
```python
@app.callback(
    Output("freq-time-range-slider", "value"),
    [Input("freq-start-time", "value"), Input("freq-end-time", "value")],
    prevent_initial_call=True,
)
```
- **Purpose**: Synchronize frequency range slider with manual inputs
- **Maintenance**: Update slider configuration for different analysis types

---

### **4. Filtering Screen** (`/filtering`)

**Purpose**: Advanced signal filtering with multiple filter types and real-time comparison.

**File Location**: `src/vitalDSP_webapp/layout/pages/analysis_pages.py` (Lines 1584-2798)

**Layout Structure**:
```
┌─────────────────────────────────────────────────────────┐
│                    Page Header                          │
│         "🔧 Advanced Signal Filtering"                 │
├─────────────────────────────────────────────────────────┤
│                Filter Type Selection                    │
│  [Traditional] [Advanced] [Artifact] [Neural] [Ensemble]│
├─────────────────────────────────────────────────────────┤
│                Filter Parameters                        │
│  (Dynamic based on filter type selection)               │
├─────────────┬───────────────────────────────────────────┤
│   Settings  │              Visualization Area           │
│   Panel     │                                           │
│             │  ┌─────────────────────────────────────┐  │
│  - Filter   │  │        Original Signal Plot         │  │
│    Type     │  │                                     │  │
│  - Filter   │  └─────────────────────────────────────┘  │
│    Family   │  ┌─────────────────────────────────────┐  │
│  - Frequency│  │        Filtered Signal Plot         │  │
│    Cutoffs  │  │                                     │  │
│  - Order    │  └─────────────────────────────────────┘  │
│  - Advanced │  ┌─────────────────────────────────────┐  │
│    Options  │  │        Comparison Plot              │  │
│             │  │     (Original vs Filtered)          │  │
│             │  └─────────────────────────────────────┘  │
│             │                Quality Metrics            │
│             │  ┌─────────────────────────────────────┐  │
│             │  │      Filter Performance Metrics     │  │
│             │  │                                     │  │
│             │  └─────────────────────────────────────┘  │
└─────────────┴───────────────────────────────────────────┘
```

**Key Components**:

1. **Filter Type Selection** (Lines 1618-1644):
   - **Traditional Filters**: Butterworth, Chebyshev, Elliptic
   - **Advanced Filters**: Kalman, Adaptive, Wiener
   - **Artifact Removal**: Amplitude, statistical, baseline
   - **Neural Network**: Deep learning-based filtering
   - **Ensemble Methods**: Multiple filter combination

2. **Dynamic Parameter Panels**:
   - **Traditional Parameters**: Filter family, response, frequency cutoffs, order
   - **Advanced Parameters**: Method selection, noise level, iterations, learning rate
   - **Artifact Parameters**: Artifact type, removal strength
   - **Neural Parameters**: Network type, model complexity
   - **Ensemble Parameters**: Method, number of filters

3. **Visualization Area**:
   - **Original Signal Plot**: Unfiltered signal display
   - **Filtered Signal Plot**: Processed signal with applied filter
   - **Comparison Plot**: Side-by-side or overlaid comparison
   - **Quality Plots**: Filter performance visualization

4. **Quality Metrics**:
   - **Filter Performance**: SNR improvement, noise reduction
   - **Signal Quality**: Before/after quality assessment
   - **Filter Characteristics**: Frequency response, phase response
   - **Statistical Metrics**: Mean, variance, correlation

**Interactive Features**:
- **Real-time Filtering**: Live filter parameter adjustment
- **Filter Comparison**: Multiple filter results comparison
- **Parameter Optimization**: Automatic parameter tuning
- **Quality Assessment**: Real-time quality metrics

**vitalDSP Integration**:
- **Filtering Functions**: `vitalDSP.filtering.*` modules
- **Quality Assessment**: `vitalDSP.signal_quality_assessment.*`
- **Preprocessing**: `vitalDSP.preprocess.*` modules
- **Advanced Computation**: `vitalDSP.advanced_computation.*`

#### **Callbacks Used by Filtering Screen**

**File Location**: `src/vitalDSP_webapp/callbacks/analysis/signal_filtering_callbacks.py`

**Main Callback: `advanced_filtering_callback()`** (Lines 110-2000)
```python
@app.callback(
    [
        Output("filter-original-plot", "figure"),
        Output("filter-filtered-plot", "figure"),
        Output("filter-comparison-plot", "figure"),
        Output("filter-quality-metrics", "children"),
        Output("filter-quality-plots", "figure"),
        Output("store-filtering-data", "data"),
    ],
    [
        Input("url", "pathname"),
        Input("filter-btn-apply", "n_clicks"),
        Input("filter-time-range-slider", "value"),
        Input("filter-btn-nudge-m10", "n_clicks"),
        Input("filter-btn-nudge-m1", "n_clicks"),
        Input("filter-btn-nudge-p1", "n_clicks"),
        Input("filter-btn-nudge-p10", "n_clicks"),
    ],
    [
        State("filter-start-time", "value"),
        State("filter-end-time", "value"),
        State("filter-type-select", "value"),
        State("filter-family-advanced", "value"),
        State("filter-response-advanced", "value"),
        # ... more states
    ],
)
```

**Purpose**: Advanced signal filtering with multiple filter types and real-time comparison

**Key Logic**:
1. **Filter Type Selection** (Lines 500-800):
   - Traditional Filters: Butterworth, Chebyshev, Elliptic
   - Advanced Filters: Kalman, Adaptive, Wiener
   - Artifact Removal: Amplitude, statistical, baseline
   - Neural Network: Deep learning-based filtering
   - Ensemble Methods: Multiple filter combination

2. **Parameter Processing** (Lines 800-1200):
   - Dynamic parameter panel updates
   - Filter-specific parameter validation
   - Advanced parameter configuration
   - Real-time parameter adjustment

3. **Filter Application** (Lines 1200-1600):
   - Apply selected filter type
   - Process signal with filter parameters
   - Generate filtered signal output
   - Calculate filter performance metrics

4. **Results Generation** (Lines 1600-2000):
   - Create comparison visualizations
   - Generate quality assessment plots
   - Compute performance metrics
   - Store filtering results

**Maintenance Notes**:
- **New Filter Types**: Add new filter types in the filter type selection section (Lines 500-600)
- **Filter Parameters**: Add new parameters for existing filters in the parameter processing section (Lines 800-1000)
- **Filter Algorithms**: Implement new filtering algorithms in the filter application section (Lines 1200-1400)
- **Quality Metrics**: Add new quality assessment metrics in the results generation section (Lines 1600-1800)

**Dependencies**:
- `vitalDSP.filtering.*` modules
- `vitalDSP.signal_quality_assessment.*` modules
- `vitalDSP.preprocess.*` modules
- `vitalDSP.advanced_computation.*` modules
- `scipy.signal` for traditional filters

**Common Issues & Fixes**:
- **Filter Instability**: High-order filters may be unstable - add stability checks
- **Parameter Validation**: Invalid parameters may cause errors - add comprehensive validation
- **Performance**: Complex filters may be slow - add progress indicators and optimization
- **Memory Issues**: Large datasets may cause memory problems - implement chunked processing

**Secondary Callbacks**:

**`update_filter_parameters_visibility()`** (Lines 2001-2100)
```python
@app.callback(
    [
        Output("traditional-parameters", "style"),
        Output("advanced-parameters", "style"),
        Output("artifact-parameters", "style"),
        Output("neural-parameters", "style"),
        Output("ensemble-parameters", "style"),
    ],
    [Input("filter-type-select", "value")],
)
```
- **Purpose**: Show/hide parameter panels based on filter type selection
- **Maintenance**: Add new filter types and their parameter containers

**`update_filter_quality_options()`** (Lines 2101-2200)
```python
@app.callback(
    Output("filter-quality-options", "options"),
    [Input("filter-type-select", "value")],
)
```
- **Purpose**: Update quality assessment options based on filter type
- **Maintenance**: Add new quality metrics for different filter types

**`update_filter_time_range_slider()`** (Lines 2201-2300)
```python
@app.callback(
    Output("filter-time-range-slider", "value"),
    [Input("filter-start-time", "value"), Input("filter-end-time", "value")],
    prevent_initial_call=True,
)
```
- **Purpose**: Synchronize filter time range slider with manual inputs
- **Maintenance**: Update slider configuration for different filter types

---

### **5. Physiological Features Screen** (`/physiological`)

**Purpose**: Comprehensive physiological feature extraction and analysis.

**File Location**: `src/vitalDSP_webapp/layout/pages/analysis_pages.py` (Lines 3000-4000)

**Layout Structure**:
```
┌─────────────────────────────────────────────────────────┐
│                    Page Header                          │
│        "❤️ Physiological Features Analysis"            │
├─────────────────────────────────────────────────────────┤
│                Feature Categories                       │
│  [HRV] [PPG Features] [ECG Features] [Quality] [Advanced]│
├─────────────┬───────────────────────────────────────────┤
│   Settings  │              Visualization Area           │
│   Panel     │                                           │
│             │  ┌─────────────────────────────────────┐  │
│  - Signal   │  │        Signal Visualization         │  │
│    Type     │  │      (PPG/ECG with annotations)     │  │
│  - Feature  │  └─────────────────────────────────────┘  │
│    Selection│  ┌─────────────────────────────────────┐  │
│  - Analysis │  │        Feature Extraction           │  │
│    Options  │  │         Results Display             │  │
│  - Quality  │  └─────────────────────────────────────┘  │
│    Settings │                                           │
│  - Advanced │                Feature Tables             │
│    Options  │  ┌─────────┬─────────┬─────────┬───────┐ │
│             │  │   HRV   │   PPG   │   ECG   │Quality│ │
│             │  │Features │Features │Features │Metrics│ │
│             │  └─────────┴─────────┴─────────┴───────┘ │
└─────────────┴───────────────────────────────────────────┘
```

**Key Components**:

1. **Feature Categories**:
   - **HRV Analysis**: Heart rate variability features
   - **PPG Features**: Photoplethysmography features
   - **ECG Features**: Electrocardiography features
   - **Signal Quality**: Quality assessment metrics
   - **Advanced Features**: Machine learning features

2. **Feature Extraction**:
   - **Time Domain Features**: RMSSD, SDNN, pNN50
   - **Frequency Domain Features**: LF, HF, LF/HF ratio
   - **Non-linear Features**: SD1, SD2, entropy
   - **Morphological Features**: Peaks, valleys, slopes
   - **Autonomic Features**: Sympathetic/parasympathetic indicators

3. **Visualization**:
   - **Signal Display**: Annotated signal with feature markers
   - **Feature Plots**: Time series of extracted features
   - **Quality Indicators**: Signal quality visualization
   - **Statistical Plots**: Distribution and correlation plots

4. **Results Tables**:
   - **HRV Features Table**: Comprehensive HRV metrics
   - **PPG Features Table**: PPG-specific features
   - **ECG Features Table**: ECG-specific features
   - **Quality Metrics Table**: Signal quality assessment

**vitalDSP Integration**:
- **Physiological Features**: `vitalDSP.physiological_features.*`
- **Feature Engineering**: `vitalDSP.feature_engineering.*`
- **Quality Assessment**: `vitalDSP.signal_quality_assessment.*`
- **Transforms**: `vitalDSP.transforms.*`

#### **Callbacks Used by Physiological Features Screen**

**File Location**: `src/vitalDSP_webapp/callbacks/features/physiological_callbacks.py`

**Main Callback: `physiological_features_callback()`** (Lines 50-1500)
```python
@app.callback(
    [
        Output("physio-signal-plot", "figure"),
        Output("physio-features-plot", "figure"),
        Output("physio-hrv-table", "children"),
        Output("physio-ppg-table", "children"),
        Output("physio-ecg-table", "children"),
        Output("physio-quality-table", "children"),
        Output("store-physiological-data", "data"),
        Output("store-features-data", "data"),
    ],
    [
        Input("physio-analyze-btn", "n_clicks"),
        Input("url", "pathname"),
        Input("physio-time-range-slider", "value"),
        Input("physio-btn-nudge-m10", "n_clicks"),
        Input("physio-btn-nudge-m1", "n_clicks"),
        Input("physio-btn-nudge-p1", "n_clicks"),
        Input("physio-btn-nudge-p10", "n_clicks"),
    ],
    [
        State("physio-start-time", "value"),
        State("physio-end-time", "value"),
        State("physio-signal-type", "value"),
        State("physio-feature-categories", "value"),
        State("physio-analysis-options", "value"),
        # ... more states
    ],
)
```

**Purpose**: Comprehensive physiological feature extraction and analysis

**Key Logic**:
1. **Feature Category Selection** (Lines 200-400):
   - HRV Analysis: Heart rate variability features
   - PPG Features: Photoplethysmography features
   - ECG Features: Electrocardiography features
   - Signal Quality: Quality assessment metrics
   - Advanced Features: Machine learning features

2. **Feature Extraction** (Lines 400-800):
   - Time Domain Features: RMSSD, SDNN, pNN50
   - Frequency Domain Features: LF, HF, LF/HF ratio
   - Non-linear Features: SD1, SD2, entropy
   - Morphological Features: Peaks, valleys, slopes
   - Autonomic Features: Sympathetic/parasympathetic indicators

3. **Signal Processing** (Lines 800-1200):
   - Peak detection and validation
   - Signal preprocessing and cleaning
   - Feature calculation algorithms
   - Quality assessment metrics

4. **Results Generation** (Lines 1200-1500):
   - Create feature visualization plots
   - Generate comprehensive feature tables
   - Compute statistical summaries
   - Store extracted features

**Maintenance Notes**:
- **New Feature Types**: Add new feature categories in the feature selection section (Lines 200-300)
- **Feature Algorithms**: Implement new feature extraction algorithms in the feature extraction section (Lines 400-600)
- **Quality Metrics**: Add new quality assessment metrics in the quality section (Lines 600-800)
- **Visualization**: Update plot configurations in the visualization section (Lines 1200-1400)

**Dependencies**:
- `vitalDSP.physiological_features.*` modules
- `vitalDSP.feature_engineering.*` modules
- `vitalDSP.signal_quality_assessment.*` modules
- `vitalDSP.transforms.*` modules
- `scipy.signal` for signal processing

**Common Issues & Fixes**:
- **Peak Detection**: May miss peaks with unusual characteristics - adjust detection parameters
- **Feature Calculation**: Some features may fail with poor quality signals - add quality checks
- **Memory Issues**: Large datasets may cause memory problems - implement chunked processing
- **Performance**: Complex feature calculations may be slow - add progress indicators

**Secondary Callbacks**:

**`update_feature_categories_visibility()`** (Lines 1501-1600)
```python
@app.callback(
    [
        Output("hrv-parameters", "style"),
        Output("ppg-parameters", "style"),
        Output("ecg-parameters", "style"),
        Output("quality-parameters", "style"),
        Output("advanced-parameters", "style"),
    ],
    [Input("physio-feature-categories", "value")],
)
```
- **Purpose**: Show/hide parameter panels based on feature category selection
- **Maintenance**: Add new feature categories and their parameter containers

**`update_physiological_time_range_slider()`** (Lines 1601-1700)
```python
@app.callback(
    Output("physio-time-range-slider", "value"),
    [Input("physio-start-time", "value"), Input("physio-end-time", "value")],
    prevent_initial_call=True,
)
```
- **Purpose**: Synchronize physiological time range slider with manual inputs
- **Maintenance**: Update slider configuration for different feature types

---

### **6. Respiratory Analysis Screen** (`/respiratory`)

**Purpose**: Respiratory rate estimation and breathing pattern analysis.

**File Location**: `src/vitalDSP_webapp/layout/pages/analysis_pages.py` (Lines 4000-5000)

**Layout Structure**:
```
┌─────────────────────────────────────────────────────────┐
│                    Page Header                          │
│         "🫁 Respiratory Analysis"                     │
├─────────────────────────────────────────────────────────┤
│                Analysis Methods                         │
│  [Peak Detection] [FFT] [Frequency] [Sleep Apnea] [Fusion]│
├─────────────┬───────────────────────────────────────────┤
│   Settings  │              Visualization Area           │
│   Panel     │                                           │
│             │  ┌─────────────────────────────────────┐  │
│  - Method   │  │        Respiratory Signal           │  │
│    Selection│  │      (PPG/ECG derived)             │  │
│  - Time     │  └─────────────────────────────────────┘  │
│    Window   │  ┌─────────────────────────────────────┐  │
│  - Advanced │  │        Respiratory Rate             │  │
│    Options  │  │         Estimation Plot             │  │
│  - Fusion   │  └─────────────────────────────────────┘  │
│    Settings │  ┌─────────────────────────────────────┐  │
│             │  │        Breathing Pattern            │  │
│             │  │         Analysis Plot               │  │
│             │  └─────────────────────────────────────┘  │
│             │                Results Tables             │
│             │  ┌─────────┬─────────┬─────────┬───────┐ │
│             │  │   RR    │Patterns │Apnea    │Fusion │ │
│             │  │Estimate │Analysis │Detection│Results│ │
│             │  └─────────┴─────────┴─────────┴───────┘ │
└─────────────┴───────────────────────────────────────────┘
```

**Key Components**:

1. **Analysis Methods**:
   - **Peak Detection RR**: Time-domain respiratory rate estimation
   - **FFT-based RR**: Frequency-domain respiratory rate estimation
   - **Frequency Domain RR**: Advanced frequency analysis
   - **Sleep Apnea Detection**: Apnea detection algorithms
   - **Multimodal Fusion**: PPG-ECG fusion analysis

2. **Visualization**:
   - **Respiratory Signal**: Derived respiratory signal from PPG/ECG
   - **Respiratory Rate Plot**: Time series of estimated respiratory rate
   - **Breathing Pattern Plot**: Breathing pattern analysis
   - **Apnea Detection Plot**: Apnea events visualization

3. **Results Tables**:
   - **RR Estimation Table**: Respiratory rate estimates and confidence
   - **Pattern Analysis Table**: Breathing pattern characteristics
   - **Apnea Detection Table**: Apnea events and severity
   - **Fusion Results Table**: Multimodal analysis results

**vitalDSP Integration**:
- **Respiratory Analysis**: `vitalDSP.respiratory_analysis.*`
- **Sleep Apnea Detection**: `vitalDSP.respiratory_analysis.sleep_apnea_detection.*`
- **Multimodal Fusion**: `vitalDSP.respiratory_analysis.fusion.*`
- **Preprocessing**: `vitalDSP.preprocess.*`

#### **Callbacks Used by Respiratory Analysis Screen**

**File Location**: `src/vitalDSP_webapp/callbacks/analysis/respiratory_callbacks.py`

**Main Callback: `respiratory_analysis_callback()`** (Lines 50-1200)
```python
@app.callback(
    [
        Output("resp-signal-plot", "figure"),
        Output("resp-rate-plot", "figure"),
        Output("resp-pattern-plot", "figure"),
        Output("resp-apnea-plot", "figure"),
        Output("resp-rr-table", "children"),
        Output("resp-pattern-table", "children"),
        Output("resp-apnea-table", "children"),
        Output("resp-fusion-table", "children"),
        Output("store-respiratory-data", "data"),
    ],
    [
        Input("resp-analyze-btn", "n_clicks"),
        Input("url", "pathname"),
        Input("resp-time-range-slider", "value"),
        Input("resp-btn-nudge-m10", "n_clicks"),
        Input("resp-btn-nudge-m1", "n_clicks"),
        Input("resp-btn-nudge-p1", "n_clicks"),
        Input("resp-btn-nudge-p10", "n_clicks"),
    ],
    [
        State("resp-start-time", "value"),
        State("resp-end-time", "value"),
        State("resp-method", "value"),
        State("resp-signal-type", "value"),
        State("resp-analysis-options", "value"),
        # ... more states
    ],
)
```

**Purpose**: Respiratory rate estimation and breathing pattern analysis

**Key Logic**:
1. **Method Selection** (Lines 200-400):
   - Peak Detection RR: Time-domain respiratory rate estimation
   - FFT-based RR: Frequency-domain respiratory rate estimation
   - Frequency Domain RR: Advanced frequency analysis
   - Sleep Apnea Detection: Apnea detection algorithms
   - Multimodal Fusion: PPG-ECG fusion analysis

2. **Signal Processing** (Lines 400-800):
   - Derive respiratory signal from PPG/ECG
   - Apply preprocessing and filtering
   - Extract breathing patterns
   - Detect respiratory events

3. **Analysis Computation** (Lines 800-1000):
   - Calculate respiratory rate estimates
   - Analyze breathing patterns
   - Detect apnea events
   - Perform multimodal fusion

4. **Results Generation** (Lines 1000-1200):
   - Create visualization plots
   - Generate analysis tables
   - Compute confidence metrics
   - Store respiratory data

**Maintenance Notes**:
- **New Methods**: Add new respiratory analysis methods in the method selection section (Lines 200-300)
- **Signal Processing**: Update signal derivation algorithms in the signal processing section (Lines 400-600)
- **Apnea Detection**: Modify apnea detection algorithms in the analysis section (Lines 800-900)
- **Visualization**: Update plot configurations in the visualization section (Lines 1000-1100)

**Dependencies**:
- `vitalDSP.respiratory_analysis.*` modules
- `vitalDSP.respiratory_analysis.sleep_apnea_detection.*` modules
- `vitalDSP.respiratory_analysis.fusion.*` modules
- `vitalDSP.preprocess.*` modules
- `scipy.signal` for signal processing

**Common Issues & Fixes**:
- **Signal Quality**: Poor signal quality may affect respiratory rate estimation - add quality checks
- **Apnea Detection**: May miss subtle apnea events - adjust detection sensitivity
- **Fusion Methods**: Multimodal fusion may fail with mismatched signals - add signal alignment
- **Performance**: Complex analysis may be slow - add progress indicators

**Secondary Callbacks**:

**`update_respiratory_method_parameters()`** (Lines 1201-1300)
```python
@app.callback(
    [
        Output("peak-detection-parameters", "style"),
        Output("fft-parameters", "style"),
        Output("frequency-parameters", "style"),
        Output("apnea-parameters", "style"),
        Output("fusion-parameters", "style"),
    ],
    [Input("resp-method", "value")],
)
```
- **Purpose**: Show/hide parameter panels based on analysis method selection
- **Maintenance**: Add new analysis methods and their parameter containers

**`update_respiratory_time_range_slider()`** (Lines 1301-1400)
```python
@app.callback(
    Output("resp-time-range-slider", "value"),
    [Input("resp-start-time", "value"), Input("resp-end-time", "value")],
    prevent_initial_call=True,
)
```
- **Purpose**: Synchronize respiratory time range slider with manual inputs
- **Maintenance**: Update slider configuration for different analysis methods

---

### **7. Signal Quality Assessment Screen** (`/quality`)

**Purpose**: Comprehensive signal quality assessment and validation.

**File Location**: `src/vitalDSP_webapp/layout/pages/analysis_pages.py` (Lines 5000-6000)

**Layout Structure**:
```
┌─────────────────────────────────────────────────────────┐
│                    Page Header                          │
│         "📊 Signal Quality Assessment"                 │
├─────────────────────────────────────────────────────────┤
│                Quality Metrics                          │
│  [SNR] [Artifacts] [Baseline] [Stability] [Advanced]    │
├─────────────┬───────────────────────────────────────────┤
│   Settings  │              Visualization Area           │
│   Panel     │                                           │
│             │  ┌─────────────────────────────────────┐  │
│  - Quality  │  │        Signal Quality Plot          │  │
│    Metrics  │  │      (with quality annotations)     │  │
│  - Threshold│  │                                     │  │
│    Settings │  └─────────────────────────────────────┘  │
│  - Advanced │  ┌─────────────────────────────────────┐  │
│    Options  │  │        Quality Metrics Plot         │  │
│  - Analysis │  │      (SNR, artifacts, etc.)         │  │
│    Options  │  └─────────────────────────────────────┘  │
│             │  ┌─────────────────────────────────────┐  │
│             │  │        Quality Score Dashboard      │  │
│             │  │                                     │  │
│             │  └─────────────────────────────────────┘  │
│             │                Quality Tables             │
│             │  ┌─────────┬─────────┬─────────┬───────┐ │
│             │  │   SNR   │Artifacts│Baseline │Overall│ │
│             │  │Metrics  │Detection│Wander   │Score  │ │
│             │  └─────────┴─────────┴─────────┴───────┘ │
└─────────────┴───────────────────────────────────────────┘
```

**Key Components**:

1. **Quality Metrics**:
   - **SNR Analysis**: Signal-to-noise ratio calculation
   - **Artifact Detection**: Amplitude and statistical artifact detection
   - **Baseline Wander**: Low-frequency drift analysis
   - **Signal Stability**: Variance and stationarity tests
   - **Advanced Metrics**: Composite quality indices

2. **Visualization**:
   - **Signal Quality Plot**: Signal with quality annotations
   - **Quality Metrics Plot**: Time series of quality metrics
   - **Quality Score Dashboard**: Overall quality assessment
   - **Threshold Visualization**: Quality threshold indicators

3. **Results Tables**:
   - **SNR Metrics Table**: Signal-to-noise ratio statistics
   - **Artifact Detection Table**: Detected artifacts and locations
   - **Baseline Wander Table**: Baseline correction results
   - **Overall Score Table**: Composite quality scores

**vitalDSP Integration**:
- **Quality Assessment**: `vitalDSP.signal_quality_assessment.*`
- **Artifact Detection**: `vitalDSP.signal_quality_assessment.artifact_detection_removal`
- **Quality Indexing**: `vitalDSP.signal_quality_assessment.signal_quality_index`

#### **Callbacks Used by Signal Quality Assessment Screen**

**File Location**: `src/vitalDSP_webapp/callbacks/analysis/quality_callbacks.py`

**Main Callback: `signal_quality_callback()`** (Lines 26-1000)
```python
@app.callback(
    [
        Output("quality-main-plot", "figure"),
        Output("quality-metrics-plot", "figure"),
        Output("quality-assessment-results", "children"),
        Output("quality-issues-recommendations", "children"),
        Output("quality-detailed-analysis", "children"),
        Output("quality-score-dashboard", "children"),
        Output("store-quality-data", "data"),
        Output("store-quality-results", "data"),
    ],
    [
        Input("quality-analyze-btn", "n_clicks"),
        Input("url", "pathname"),
        Input("quality-time-range-slider", "value"),
        Input("quality-btn-nudge-m10", "n_clicks"),
        Input("quality-btn-nudge-m1", "n_clicks"),
        Input("quality-btn-nudge-p1", "n_clicks"),
        Input("quality-btn-nudge-p10", "n_clicks"),
    ],
    [
        State("quality-start-time", "value"),
        State("quality-end-time", "value"),
        State("quality-signal-type", "value"),
        State("quality-metrics", "value"),
        State("quality-thresholds", "value"),
        # ... more states
    ],
)
```

**Purpose**: Comprehensive signal quality assessment and validation

**Key Logic**:
1. **Quality Metrics Selection** (Lines 200-400):
   - SNR Analysis: Signal-to-noise ratio calculation
   - Artifact Detection: Amplitude and statistical artifact detection
   - Baseline Wander: Low-frequency drift analysis
   - Signal Stability: Variance and stationarity tests
   - Advanced Metrics: Composite quality indices

2. **Quality Assessment** (Lines 400-700):
   - Calculate selected quality metrics
   - Apply quality thresholds
   - Detect quality issues
   - Generate quality scores

3. **Results Analysis** (Lines 700-900):
   - Create quality visualization plots
   - Generate quality assessment tables
   - Provide recommendations
   - Create quality score dashboard

4. **Data Storage** (Lines 900-1000):
   - Store quality assessment results
   - Save quality metrics data
   - Update quality history

**Maintenance Notes**:
- **New Quality Metrics**: Add new quality metrics in the metrics selection section (Lines 200-300)
- **Threshold Settings**: Update quality thresholds in the threshold configuration section (Lines 300-400)
- **Quality Algorithms**: Modify quality assessment algorithms in the assessment section (Lines 400-600)
- **Visualization**: Update quality plot configurations in the visualization section (Lines 700-800)

**Dependencies**:
- `vitalDSP.signal_quality_assessment.*` modules
- `vitalDSP.signal_quality_assessment.artifact_detection_removal`
- `vitalDSP.signal_quality_assessment.signal_quality_index`
- `scipy.signal` for signal processing
- `numpy` for statistical calculations

**Common Issues & Fixes**:
- **Threshold Sensitivity**: Quality thresholds may be too strict/lenient - add adjustable thresholds
- **Metric Calculation**: Some metrics may fail with certain signal types - add signal type validation
- **Performance**: Quality assessment may be slow - add progress indicators and optimization
- **Memory Issues**: Large datasets may cause memory problems - implement chunked processing

**Secondary Callbacks**:

**`update_quality_metrics_visibility()`** (Lines 1001-1100)
```python
@app.callback(
    [
        Output("snr-parameters", "style"),
        Output("artifact-parameters", "style"),
        Output("baseline-parameters", "style"),
        Output("stability-parameters", "style"),
        Output("advanced-parameters", "style"),
    ],
    [Input("quality-metrics", "value")],
)
```
- **Purpose**: Show/hide parameter panels based on quality metrics selection
- **Maintenance**: Add new quality metrics and their parameter containers

**`update_quality_time_range_slider()`** (Lines 1101-1200)
```python
@app.callback(
    Output("quality-time-range-slider", "value"),
    [Input("quality-start-time", "value"), Input("quality-end-time", "value")],
    prevent_initial_call=True,
)
```
- **Purpose**: Synchronize quality time range slider with manual inputs
- **Maintenance**: Update slider configuration for different quality metrics

---

### **8. Advanced Features Screen** (`/features`)

**Purpose**: Advanced computational methods and machine learning integration.

**File Location**: `src/vitalDSP_webapp/layout/pages/analysis_pages.py` (Lines 6000-7000)

**Layout Structure**:
```
┌─────────────────────────────────────────────────────────┐
│                    Page Header                          │
│         "🚀 Advanced Features"                         │
├─────────────────────────────────────────────────────────┤
│                Feature Categories                       │
│  [ML Features] [Transforms] [Advanced Computation] [Export]│
├─────────────┬───────────────────────────────────────────┤
│   Settings  │              Visualization Area           │
│   Panel     │                                           │
│             │  ┌─────────────────────────────────────┐  │
│  - Feature  │  │        Feature Engineering          │  │
│    Selection│  │         Visualization               │  │
│  - ML       │  └─────────────────────────────────────┘  │
│    Methods  │  ┌─────────────────────────────────────┐  │
│  - Transform│  │        Advanced Computation          │  │
│    Settings │  │         Results Display             │  │
│  - Export   │  └─────────────────────────────────────┘  │
│    Options  │                                           │
│             │                Feature Tables             │
│             │  ┌─────────┬─────────┬─────────┬───────┐ │
│             │  │   ML    │Transform│Advanced │Export │ │
│             │  │Features │Features │Comp.    │Results│ │
│             │  └─────────┴─────────┴─────────┴───────┘ │
└─────────────┴───────────────────────────────────────────┘
```

**Key Components**:

1. **Feature Engineering**:
   - **ML Features**: Machine learning-based feature extraction
   - **Transform Features**: Wavelet, FFT, Hilbert transform features
   - **Advanced Computation**: Anomaly detection, Bayesian analysis
   - **Export Options**: Feature export in multiple formats

2. **Visualization**:
   - **Feature Engineering Plot**: Extracted features visualization
   - **Advanced Computation Plot**: ML and statistical analysis results
   - **Transform Visualization**: Wavelet and frequency domain plots
   - **Export Preview**: Feature export preview

3. **Results Tables**:
   - **ML Features Table**: Machine learning extracted features
   - **Transform Features Table**: Transform-based features
   - **Advanced Computation Table**: Statistical and ML results
   - **Export Results Table**: Export status and file information

**vitalDSP Integration**:
- **Feature Engineering**: `vitalDSP.feature_engineering.*`
- **Transforms**: `vitalDSP.transforms.*`
- **Advanced Computation**: `vitalDSP.advanced_computation.*`
- **Machine Learning**: `vitalDSP.advanced_computation.*`

#### **Callbacks Used by Advanced Features Screen**

**File Location**: `src/vitalDSP_webapp/callbacks/features/features_callbacks.py`

**Main Callback: `advanced_features_callback()`** (Lines 50-1000)
```python
@app.callback(
    [
        Output("features-engineering-plot", "figure"),
        Output("features-computation-plot", "figure"),
        Output("features-transform-plot", "figure"),
        Output("features-ml-table", "children"),
        Output("features-transform-table", "children"),
        Output("features-advanced-table", "children"),
        Output("features-export-table", "children"),
        Output("store-features-data", "data"),
    ],
    [
        Input("features-analyze-btn", "n_clicks"),
        Input("url", "pathname"),
        Input("features-time-range-slider", "value"),
        Input("features-btn-nudge-m10", "n_clicks"),
        Input("features-btn-nudge-m1", "n_clicks"),
        Input("features-btn-nudge-p1", "n_clicks"),
        Input("features-btn-nudge-p10", "n_clicks"),
    ],
    [
        State("features-start-time", "value"),
        State("features-end-time", "value"),
        State("features-categories", "value"),
        State("features-ml-methods", "value"),
        State("features-transform-methods", "value"),
        # ... more states
    ],
)
```

**Purpose**: Advanced computational methods and machine learning integration

**Key Logic**:
1. **Feature Categories Selection** (Lines 200-400):
   - ML Features: Machine learning-based feature extraction
   - Transform Features: Wavelet, FFT, Hilbert transform features
   - Advanced Computation: Anomaly detection, Bayesian analysis
   - Export Options: Feature export in multiple formats

2. **Feature Engineering** (Lines 400-600):
   - Apply selected feature extraction methods
   - Compute transform-based features
   - Generate machine learning features
   - Calculate advanced computational features

3. **Advanced Computation** (Lines 600-800):
   - Perform anomaly detection
   - Apply Bayesian analysis
   - Execute Kalman filtering
   - Run machine learning algorithms

4. **Results Generation** (Lines 800-1000):
   - Create feature visualization plots
   - Generate comprehensive feature tables
   - Export features in multiple formats
   - Store feature data

**Maintenance Notes**:
- **New Feature Types**: Add new feature categories in the category selection section (Lines 200-300)
- **ML Methods**: Implement new machine learning methods in the ML section (Lines 400-500)
- **Transform Methods**: Add new transform methods in the transform section (Lines 500-600)
- **Export Formats**: Add new export formats in the export section (Lines 800-900)

**Dependencies**:
- `vitalDSP.feature_engineering.*` modules
- `vitalDSP.transforms.*` modules
- `vitalDSP.advanced_computation.*` modules
- `scikit-learn` for machine learning
- `scipy` for advanced computations

**Common Issues & Fixes**:
- **ML Performance**: Machine learning may be slow - add progress indicators and optimization
- **Feature Selection**: Too many features may cause overfitting - add feature selection methods
- **Memory Issues**: Large feature matrices may cause memory problems - implement chunked processing
- **Export Errors**: Export may fail with large datasets - add error handling and chunked export

**Secondary Callbacks**:

**`update_features_categories_visibility()`** (Lines 1001-1100)
```python
@app.callback(
    [
        Output("ml-parameters", "style"),
        Output("transform-parameters", "style"),
        Output("advanced-parameters", "style"),
        Output("export-parameters", "style"),
    ],
    [Input("features-categories", "value")],
)
```
- **Purpose**: Show/hide parameter panels based on feature category selection
- **Maintenance**: Add new feature categories and their parameter containers

**`update_features_time_range_slider()`** (Lines 1101-1200)
```python
@app.callback(
    Output("features-time-range-slider", "value"),
    [Input("features-start-time", "value"), Input("features-end-time", "value")],
    prevent_initial_call=True,
)
```
- **Purpose**: Synchronize features time range slider with manual inputs
- **Maintenance**: Update slider configuration for different feature types

---

### **9. Settings Screen** (`/settings`)

**Purpose**: Application configuration and user preferences.

**File Location**: `src/vitalDSP_webapp/layout/pages/analysis_pages.py` (Lines 7000-8000)

**Layout Structure**:
```
┌─────────────────────────────────────────────────────────┐
│                    Page Header                          │
│              "⚙️ Settings"                             │
├─────────────────────────────────────────────────────────┤
│                Settings Categories                      │
│  [General] [Display] [Processing] [Advanced] [About]    │
├─────────────┬───────────────────────────────────────────┤
│   Settings  │              Settings Panel               │
│   Navigation│                                           │
│             │  ┌─────────────────────────────────────┐  │
│  - General  │  │        General Settings             │  │
│    Settings │  │  - Theme Selection                  │  │
│  - Display  │  │  - Language Selection               │  │
│    Settings │  │  - Default Parameters               │  │
│  - Processing│  └─────────────────────────────────────┘  │
│    Settings │  ┌─────────────────────────────────────┐  │
│  - Advanced │  │        Display Settings             │  │
│    Settings │  │  - Plot Themes                      │  │
│  - About    │  │  - UI Preferences                   │  │
│             │  └─────────────────────────────────────┘  │
│             │  ┌─────────────────────────────────────┐  │
│             │  │        Processing Settings          │  │
│             │  │  - Default Sampling Frequency       │  │
│             │  │  - Analysis Parameters              │  │
│             │  └─────────────────────────────────────┘  │
└─────────────┴───────────────────────────────────────────┘
```

**Key Components**:

1. **General Settings**:
   - **Theme Selection**: Light/Dark mode
   - **Language Selection**: Interface language
   - **Default Parameters**: Default analysis parameters
   - **User Preferences**: Personalization options

2. **Display Settings**:
   - **Plot Themes**: Plot color schemes and styles
   - **UI Preferences**: Interface layout preferences
   - **Font Settings**: Text size and font preferences
   - **Layout Options**: Sidebar and panel preferences

3. **Processing Settings**:
   - **Default Sampling Frequency**: Default data sampling rate
   - **Analysis Parameters**: Default analysis settings
   - **Filter Settings**: Default filter parameters
   - **Quality Thresholds**: Signal quality thresholds

4. **Advanced Settings**:
   - **Performance Options**: Processing performance settings
   - **Memory Management**: Data caching and memory options
   - **Debug Options**: Debugging and logging settings
   - **API Settings**: External API configuration

5. **About Section**:
   - **Version Information**: Application version and build
   - **System Information**: System requirements and status
   - **License Information**: Software license details
   - **Contact Information**: Support and contact details

**vitalDSP Integration**:
- **Configuration Management**: `vitalDSP_webapp.config.settings`
- **Settings Service**: `vitalDSP_webapp.services.settings_service`
- **User Preferences**: Persistent user settings storage

#### **Callbacks Used by Settings Screen**

**File Location**: `src/vitalDSP_webapp/callbacks/analysis/settings_callbacks.py`

**Main Callback: `settings_callback()`** (Lines 50-800)
```python
@app.callback(
    [
        Output("settings-general-panel", "children"),
        Output("settings-display-panel", "children"),
        Output("settings-processing-panel", "children"),
        Output("settings-advanced-panel", "children"),
        Output("settings-about-panel", "children"),
        Output("settings-status", "children"),
        Output("store-settings-data", "data"),
    ],
    [
        Input("settings-save-btn", "n_clicks"),
        Input("settings-reset-btn", "n_clicks"),
        Input("settings-export-btn", "n_clicks"),
        Input("settings-import-btn", "n_clicks"),
        Input("url", "pathname"),
    ],
    [
        State("settings-theme", "value"),
        State("settings-language", "value"),
        State("settings-default-sampling-freq", "value"),
        State("settings-plot-theme", "value"),
        State("settings-ui-preferences", "value"),
        # ... more states
    ],
)
```

**Purpose**: Application configuration and user preferences management

**Key Logic**:
1. **Settings Categories** (Lines 200-400):
   - General Settings: Theme, language, default parameters
   - Display Settings: Plot themes, UI preferences, font settings
   - Processing Settings: Default sampling frequency, analysis parameters
   - Advanced Settings: Performance options, memory management
   - About Section: Version information, system status

2. **Settings Management** (Lines 400-600):
   - Load current settings from storage
   - Validate settings values
   - Apply settings changes
   - Save settings to persistent storage

3. **Settings Operations** (Lines 600-800):
   - Save settings to file
   - Reset settings to defaults
   - Export settings configuration
   - Import settings from file

4. **Status Updates** (Lines 800-1000):
   - Display settings status
   - Show validation errors
   - Provide user feedback
   - Update settings history

**Maintenance Notes**:
- **New Settings**: Add new settings categories in the categories section (Lines 200-300)
- **Settings Validation**: Update validation rules in the validation section (Lines 400-500)
- **Settings Storage**: Modify storage methods in the storage section (Lines 500-600)
- **Settings UI**: Update settings UI components in the UI section (Lines 600-700)

**Dependencies**:
- `vitalDSP_webapp.config.settings`
- `vitalDSP_webapp.services.settings_service`
- `json` for settings serialization
- `os` for file operations

**Common Issues & Fixes**:
- **Settings Validation**: Invalid settings may cause errors - add comprehensive validation
- **Settings Storage**: Settings may not persist - check file permissions and storage location
- **Settings Import**: Import may fail with invalid files - add error handling and validation
- **Settings Reset**: Reset may not work properly - ensure all settings are reset to defaults

**Secondary Callbacks**:

**`update_settings_panels_visibility()`** (Lines 1001-1100)
```python
@app.callback(
    [
        Output("general-settings-panel", "style"),
        Output("display-settings-panel", "style"),
        Output("processing-settings-panel", "style"),
        Output("advanced-settings-panel", "style"),
        Output("about-settings-panel", "style"),
    ],
    [Input("settings-category-tabs", "active_tab")],
)
```
- **Purpose**: Show/hide settings panels based on category selection
- **Maintenance**: Add new settings categories and their panels

**`update_settings_status()`** (Lines 1101-1200)
```python
@app.callback(
    Output("settings-status", "children"),
    [Input("settings-save-btn", "n_clicks"), Input("settings-reset-btn", "n_clicks")],
    prevent_initial_call=True,
)
```
- **Purpose**: Update settings status display based on user actions
- **Maintenance**: Add new status messages for different operations

---

## 🎨 **Screen Design Principles**

### **Consistent Design Language**
- **Color Scheme**: Consistent color palette across all screens
- **Typography**: Unified font family and sizing
- **Spacing**: Consistent margins and padding
- **Components**: Reusable UI components

### **Responsive Design**
- **Mobile-First**: Optimized for mobile devices
- **Tablet Support**: Responsive layout for tablets
- **Desktop Enhancement**: Enhanced features for desktop
- **Adaptive Layout**: Dynamic layout adjustment

### **User Experience**
- **Intuitive Navigation**: Clear navigation patterns
- **Progressive Disclosure**: Information revealed progressively
- **Feedback Systems**: Clear user feedback and status
- **Error Handling**: Graceful error handling and recovery

### **Accessibility**
- **Keyboard Navigation**: Full keyboard accessibility
- **Screen Reader Support**: ARIA labels and descriptions
- **Color Contrast**: High contrast color schemes
- **Text Scaling**: Support for text size adjustment

---

## 🔄 **Time Domain Analysis Flow Changes**

### **New Signal Processing Flow**

The time domain analysis screen has been completely updated to follow a new signal processing workflow that prioritizes filtered data and removes redundant filtering controls.

#### **1. Filtered Data Priority**
- **Default Behavior**: Use filtered data if available from the filtering screen
- **Fallback**: Use original signal if no filtering has been performed
- **User Control**: Allow users to switch between original and filtered signals via dropdown
- **Dynamic Filtering**: When time window changes, dynamically apply stored filter parameters to new signal window

#### **2. Removed Filtering Components**
- **Filter Settings**: Removed filter family, response type, frequency cutoffs, and order controls
- **Filter Processing**: Removed real-time filtering logic from time domain analysis
- **Filter Results**: Removed filtering results table and related outputs
- **UI Simplification**: Cleaner interface focused on analysis rather than filtering

#### **3. Added Signal Source Management**
- **Signal Source Selection**: New dropdown to choose between original and filtered signal
- **Filtered Data Loading**: Logic to load pre-filtered data from global storage
- **Signal Comparison**: Side-by-side comparison of original vs filtered signals with critical points
- **Dynamic Filter Application**: Apply stored filter parameters to current time window when needed

#### **4. Enhanced Filter Type Support**
- **Multi-Filter Support**: Supports all filter types from filtering screen (traditional, advanced, artifact, neural, ensemble)
- **Parameter Extraction**: Dynamically extracts appropriate parameters for each filter type
- **Detrending Support**: Applies detrending if it was used in the original filtering
- **Error Handling**: Graceful fallback for unsupported filter types

### **Implementation Changes Completed**

#### **Layout Changes** (`src/vitalDSP_webapp/layout/pages/analysis_pages.py`)

**✅ Removed Filter Controls** (Lines 236-383):
```python
# REMOVED these components:
# - Filter family selection (filter-family)
# - Filter response type (filter-response) 
# - Low/high frequency cutoffs (filter-low-freq, filter-high-freq)
# - Filter order (filter-order)
# - Filtering results table (filtering-results-table)
```

**✅ Added Signal Source Selection** (Lines 200-250):
```python
# ADDED new signal source selection
dbc.Select(
    id="signal-source-select",
    options=[
        {"label": "Original Signal", "value": "original"},
        {"label": "Filtered Signal", "value": "filtered"},
    ],
    value="filtered",  # Default to filtered
    className="mb-3",
)
```

**✅ Updated Visualization Area** (Lines 500-600):
```python
# CHANGED from separate plots to:
# - Main signal plot (selected signal with critical points)
# - Signal comparison plot (original vs filtered with critical points)
# - Replaced filtered-signal-plot with signal-comparison-plot
# - Added signal-source-table for filter information display
```

#### **Callback Changes** (`src/vitalDSP_webapp/callbacks/analysis/vitaldsp_callbacks.py`)

**✅ Updated Callback Signature** (Lines 3487-3504):
```python
# REMOVED filter-related states:
# - State("filter-family", "value")
# - State("filter-response", "value") 
# - State("filter-low-freq", "value")
# - State("filter-high-freq", "value")
# - State("filter-order", "value")

# ADDED signal source state:
State("signal-source-select", "value")
```

**✅ Enhanced Signal Source Loading Logic** (Lines 3595-3650):
```python
def load_signal_source(signal_source, data_service, latest_data_id, signal_data, sampling_freq):
    """Load signal source (original or filtered) based on selection."""
    if signal_source == "filtered":
        # Try to load filtered data from filtering screen
        filtered_data = data_service.get_filtered_data(latest_data_id)
        filter_info = data_service.get_filter_info(latest_data_id)
        
        if filtered_data is not None:
            # Check if filtered data matches current time window
            if len(filtered_data) == len(signal_data):
                return filtered_data, "Filtered Signal", filter_info
            else:
                # Apply dynamic filtering to current window
                return apply_dynamic_filter(signal_data, filter_info, sampling_freq)
    
    # Fallback to original signal
    return signal_data, "Original Signal", None
```

**✅ Dynamic Filter Application** (Lines 4500-4700):
```python
def apply_dynamic_filter(signal_data, filter_info, sampling_freq):
    """Apply stored filter parameters to current signal window."""
    filter_type = filter_info.get("filter_type", "traditional")
    parameters = filter_info.get("parameters", {})
    detrending_applied = filter_info.get("detrending_applied", False)
    
    # Apply detrending if needed
    if detrending_applied:
        signal_data_detrended = scipy_signal.detrend(signal_data)
    else:
        signal_data_detrended = signal_data
    
    # Apply appropriate filter based on type
    if filter_type == "traditional":
        return apply_traditional_filter(signal_data_detrended, ...)
    elif filter_type == "advanced":
        return apply_advanced_filter(signal_data_detrended, ...)
    elif filter_type == "artifact":
        return apply_enhanced_artifact_removal(signal_data_detrended, ...)
    elif filter_type == "neural":
        return apply_neural_filter(signal_data_detrended, ...)
    elif filter_type == "ensemble":
        return apply_enhanced_ensemble_filter(signal_data_detrended, ...)
    else:
        return signal_data_detrended
```

**✅ Updated Analysis Logic** (Lines 3721-4000):
```python
# REMOVED filtering logic:
# - Filter design and application
# - Filter parameter processing
# - Filter performance metrics

# ENHANCED analysis logic:
# - Peak detection with vitalDSP WaveformMorphology
# - Signal quality assessment with improved metrics
# - Statistical analysis with robust calculations
# - Visualization generation with critical points
# - Dynamic signal type detection (PPG/ECG/Other)
```

**✅ Enhanced Signal Comparison Logic** (Lines 4000-4500):
```python
def create_signal_comparison_plot(original_signal, filtered_signal, time_axis, sampling_freq, signal_type):
    """Create side-by-side comparison with critical points."""
    # Trim signals to common length
    min_length = min(len(original_signal), len(filtered_signal), len(time_axis))
    original_trimmed = original_signal[:min_length]
    filtered_trimmed = filtered_signal[:min_length]
    time_trimmed = time_axis[:min_length]
    
    # Detect critical points for both signals
    original_peaks = detect_critical_points(original_trimmed, sampling_freq, signal_type)
    filtered_peaks = detect_critical_points(filtered_trimmed, sampling_freq, signal_type)
    
    # Create comparison plot with critical points
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=["Raw Signal with Critical Points", 
                                     "Filtered Signal with Critical Points"])
    
    # Add original signal with peaks
    fig.add_trace(go.Scatter(x=time_trimmed, y=original_trimmed, 
                            name="Raw Signal", line=dict(color="blue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=time_trimmed[original_peaks], y=original_trimmed[original_peaks], 
                            mode="markers", name="Critical Points", 
                            marker=dict(color="red", size=8)), row=1, col=1)
    
    # Add filtered signal with peaks
    fig.add_trace(go.Scatter(x=time_trimmed, y=filtered_trimmed, 
                            name="Filtered Signal", line=dict(color="green")), row=2, col=1)
    fig.add_trace(go.Scatter(x=time_trimmed[filtered_peaks], y=filtered_trimmed[filtered_peaks], 
                            mode="markers", name="Critical Points", 
                            marker=dict(color="red", size=8)), row=2, col=1)
    
    return fig
```

#### **Data Service Changes** (`src/vitalDSP_webapp/services/data/data_service.py`)

**✅ Added Filtered Data Storage** (Lines 100-200):
```python
def store_filtered_data(self, data_id, filtered_signal, filter_info):
    """Store filtered signal data from filtering screen."""
    if data_id not in self._data_store:
        raise ValueError(f"Data ID {data_id} not found")
    
    self._data_store[data_id]["filtered_signal"] = filtered_signal
    self._data_store[data_id]["filter_info"] = filter_info
    self._data_store[data_id]["has_filtered_data"] = True

def get_filtered_data(self, data_id):
    """Retrieve filtered signal data."""
    if data_id not in self._data_store:
        return None
    
    data = self._data_store[data_id]
    if data.get("has_filtered_data", False):
        return data.get("filtered_signal")
    return None

def has_filtered_data(self, data_id):
    """Check if filtered data is available."""
    if data_id not in self._data_store:
        return False
    
    return self._data_store[data_id].get("has_filtered_data", False)

def get_filter_info(self, data_id):
    """Retrieve filter information."""
    if data_id not in self._data_store:
        return None
    
    return self._data_store[data_id].get("filter_info")

def clear_filtered_data(self, data_id):
    """Clear stored filtered data."""
    if data_id in self._data_store:
        self._data_store[data_id]["has_filtered_data"] = False
        self._data_store[data_id].pop("filtered_signal", None)
        self._data_store[data_id].pop("filter_info", None)
        return True
    return False
```

#### **Filtering Screen Integration** (`src/vitalDSP_webapp/callbacks/analysis/signal_filtering_callbacks.py`)

**✅ Updated Filtering Callback** (Lines 1600-2000):
```python
# ADDED filtered data storage after successful filtering:
def store_filtered_results(filtered_signal, filter_info, data_service, data_id):
    """Store filtered results for use in time domain analysis."""
    data_service.store_filtered_data(data_id, filtered_signal, filter_info)
    
    # Update store-filtering-data output
    return {
        "filtered_signal": filtered_signal,
        "filter_info": filter_info,
        "timestamp": datetime.now().isoformat()
    }
```

**✅ Added Signal Type Selection** (Lines 200-300):
```python
# ADDED signal type selector to filtering layout
dbc.Select(
    id="filter-signal-type-select",
    options=[
        {"label": "PPG", "value": "ppg"},
        {"label": "ECG", "value": "ecg"},
        {"label": "Other", "value": "other"},
    ],
    value="ppg",  # Default to PPG
    className="mb-3",
)
```

**✅ Enhanced Critical Points Detection** (Lines 1000-1500):
```python
# UPDATED to use dynamic signal type and actual sampling frequency
def create_original_signal_plot(time_axis, original_signal, sampling_freq, signal_type):
    """Create plot with critical points based on signal type."""
    # Use WaveformMorphology with actual parameters
    wm = WaveformMorphology(original_signal, fs=sampling_freq, signal_type=signal_type)
    
    # Detect critical points based on signal type
    if signal_type == "ppg":
        peaks = wm.systolic_peaks if hasattr(wm, "systolic_peaks") else []
        notches = wm.dicrotic_notches if hasattr(wm, "dicrotic_notches") else []
    elif signal_type == "ecg":
        peaks = wm.r_peaks if hasattr(wm, "r_peaks") else []
        notches = wm.p_peaks if hasattr(wm, "p_peaks") else []
    else:
        peaks = detect_peaks(original_signal)  # Fallback
        notches = []
    
    # Create plot with critical points
    return create_plot_with_peaks(time_axis, original_signal, peaks, notches)
```

**✅ Updated Filter Comparison Plot** (Lines 2000-2500):
```python
# ENHANCED to include critical points for both signals
def create_filter_comparison_plot(time_axis, original_signal, filtered_signal, sampling_freq, signal_type):
    """Create comparison plot with critical points for both signals."""
    # Detect critical points for both original and filtered signals
    original_peaks = detect_critical_points(original_signal, sampling_freq, signal_type)
    filtered_peaks = detect_critical_points(filtered_signal, sampling_freq, signal_type)
    
    # Create subplots with critical points
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=["Original Signal with Critical Points", 
                                     "Filtered Signal with Critical Points"])
    
    # Add both signals with their respective critical points
    add_signal_with_peaks(fig, time_axis, original_signal, original_peaks, row=1)
    add_signal_with_peaks(fig, time_axis, filtered_signal, filtered_peaks, row=2)
    
    return fig
```

### **Latest Improvements & Features**

#### **1. Multi-Filter Type Support**
- **Traditional Filters**: Butterworth, Chebyshev, Elliptic, Bessel
- **Advanced Filters**: Kalman, Wiener, Adaptive filtering
- **Artifact Removal**: Baseline wander, motion artifacts, noise removal
- **Neural Filters**: LSTM, CNN, RNN-based filtering
- **Ensemble Methods**: Multiple filter combination techniques

#### **2. Dynamic Signal Type Detection**
- **PPG Support**: Systolic peaks, dicrotic notches, diastolic peaks
- **ECG Support**: R peaks, P waves, T waves
- **Other Signals**: Basic peak detection fallback
- **Consistent Detection**: Same critical points across all screens

#### **3. Enhanced Quality Metrics**
- **Robust SNR Calculation**: Improved signal-to-noise ratio computation
- **Signal Stability**: Coefficient of variation with proper handling of small means
- **Artifact Detection**: Statistical outlier detection and motion artifact identification
- **Baseline Wander**: Low frequency drift assessment

#### **4. Improved User Interface**
- **Sidebar Reordering**: Filtering screen moved before Time Domain Analysis
- **Default Values**: Sensible defaults for frequency (0.5-5 Hz), sampling rate (100 Hz), signal type (PPG)
- **Signal Source Selection**: Easy switching between original and filtered signals
- **Critical Points Visualization**: Clear display of detected peaks and notches

### **Benefits of New Flow**

#### **1. Separation of Concerns**
- **Filtering**: Handled exclusively in the filtering screen
- **Analysis**: Focused on signal analysis without filtering complexity
- **Data Flow**: Clear data flow from filtering to analysis

#### **2. Improved User Experience**
- **Consistent Results**: Same filtered data used across all analysis screens
- **Simplified Interface**: Cleaner time domain analysis interface
- **Better Performance**: No redundant filtering operations
- **Dynamic Filtering**: Automatic filter application when time window changes

#### **3. Enhanced Maintainability**
- **Single Source of Truth**: Filtering logic centralized in one screen
- **Easier Debugging**: Clear separation between filtering and analysis
- **Modular Design**: Each screen has focused responsibilities
- **Comprehensive Error Handling**: Graceful fallbacks for all filter types

### **Migration Guide**

#### **For Developers**:
1. **Update Layout**: Remove filter controls, add signal source selection
2. **Modify Callbacks**: Update callback signatures and logic
3. **Update Data Service**: Add filtered data storage methods
4. **Test Integration**: Ensure filtered data flows correctly between screens

#### **For Users**:
1. **Perform Filtering**: Use the filtering screen to process signals
2. **Analyze Results**: Use time domain analysis with pre-filtered data
3. **Compare Signals**: Switch between original and filtered signals as needed

---

## 🔧 **Callback Maintenance Guide**

### **General Callback Maintenance Principles**

#### **1. Callback Registration**
- **Location**: All callbacks are registered in `src/vitalDSP_webapp/callbacks/__init__.py`
- **Pattern**: Each screen has its own callback module with a registration function
- **Naming**: Registration functions follow the pattern `register_{screen_name}_callbacks(app)`

#### **2. Callback Structure**
- **Main Callback**: Each screen has one primary callback handling the main functionality
- **Secondary Callbacks**: Supporting callbacks for UI updates, parameter visibility, etc.
- **Dependencies**: Callbacks depend on specific vitalDSP modules and external libraries

#### **3. Common Maintenance Tasks**

**Adding New Features**:
1. **Identify the Screen**: Determine which screen needs the new feature
2. **Locate the Callback**: Find the main callback for that screen
3. **Add New Inputs/Outputs**: Update the callback decorator with new inputs/outputs
4. **Implement Logic**: Add the new feature logic in the appropriate section
5. **Update Dependencies**: Add any new vitalDSP module dependencies
6. **Test**: Ensure the new feature works with existing functionality

**Fixing Bugs**:
1. **Identify the Issue**: Determine which callback is causing the problem
2. **Check Dependencies**: Verify all required modules are imported
3. **Validate Parameters**: Ensure all callback parameters are properly handled
4. **Check Error Handling**: Add appropriate error handling and logging
5. **Test Fix**: Verify the fix resolves the issue without breaking other features

**Performance Optimization**:
1. **Identify Bottlenecks**: Use profiling to find slow callback sections
2. **Optimize Algorithms**: Replace slow algorithms with faster alternatives
3. **Add Caching**: Implement caching for expensive computations
4. **Chunk Processing**: Break large datasets into smaller chunks
5. **Add Progress Indicators**: Show progress for long-running operations

#### **4. Callback Debugging**

**Common Debugging Techniques**:
- **Logging**: Add comprehensive logging throughout callbacks
- **Error Handling**: Wrap callback logic in try-catch blocks
- **Parameter Validation**: Validate all input parameters
- **State Checking**: Verify callback state before processing
- **Dependency Verification**: Ensure all dependencies are available

**Debugging Tools**:
- **Dash Dev Tools**: Use Dash's built-in debugging tools
- **Browser Dev Tools**: Check for JavaScript errors
- **Python Debugger**: Use pdb for step-by-step debugging
- **Logging**: Use Python logging for detailed execution tracking

#### **5. Testing Callbacks**

**Unit Testing**:
- **Test Individual Functions**: Test callback logic functions separately
- **Mock Dependencies**: Mock vitalDSP modules for testing
- **Test Edge Cases**: Test with various input scenarios
- **Test Error Conditions**: Test error handling paths

**Integration Testing**:
- **Test Full Workflows**: Test complete user workflows
- **Test Data Flow**: Verify data flows correctly between callbacks
- **Test UI Updates**: Ensure UI updates correctly
- **Test Performance**: Verify performance under load

### **Screen-Specific Maintenance**

#### **Upload Screen Callbacks**
- **File Format Support**: Add new file formats in the parsing section
- **Column Detection**: Update auto-detection logic for new data types
- **Data Validation**: Add validation rules for new data formats
- **Error Handling**: Add specific error handling for new file types

#### **Analysis Screen Callbacks**
- **New Analysis Methods**: Add new analysis algorithms
- **Parameter Updates**: Update analysis parameters and validation
- **Visualization**: Add new plot types and configurations
- **Performance**: Optimize analysis algorithms for better performance

#### **Feature Screen Callbacks**
- **New Features**: Add new feature extraction methods
- **ML Algorithms**: Implement new machine learning algorithms
- **Transform Methods**: Add new signal transform methods
- **Export Formats**: Add new feature export formats

#### **Quality Screen Callbacks**
- **Quality Metrics**: Add new signal quality assessment metrics
- **Threshold Settings**: Update quality thresholds and validation
- **Quality Algorithms**: Implement new quality assessment algorithms
- **Visualization**: Add new quality visualization methods

### **vitalDSP Integration Maintenance**

#### **Module Updates**
- **Version Compatibility**: Ensure compatibility with new vitalDSP versions
- **API Changes**: Update code when vitalDSP APIs change
- **New Features**: Integrate new vitalDSP features
- **Deprecated Functions**: Replace deprecated vitalDSP functions

#### **Dependency Management**
- **Version Pinning**: Pin specific versions of vitalDSP modules
- **Dependency Updates**: Update dependencies regularly
- **Compatibility Testing**: Test with different vitalDSP versions
- **Fallback Handling**: Add fallbacks for missing dependencies

### **Performance Maintenance**

#### **Memory Management**
- **Data Chunking**: Process large datasets in chunks
- **Memory Cleanup**: Clean up unused data and variables
- **Caching**: Implement intelligent caching strategies
- **Garbage Collection**: Ensure proper garbage collection

#### **Processing Optimization**
- **Algorithm Optimization**: Use faster algorithms where possible
- **Parallel Processing**: Implement parallel processing for independent operations
- **Progress Indicators**: Add progress indicators for long operations
- **Background Processing**: Move heavy operations to background threads

### **Error Handling Maintenance**

#### **Error Types**
- **Data Errors**: Handle invalid or corrupted data
- **Parameter Errors**: Validate all input parameters
- **Dependency Errors**: Handle missing or incompatible dependencies
- **System Errors**: Handle system-level errors gracefully

#### **Error Recovery**
- **Graceful Degradation**: Provide fallback functionality
- **User Feedback**: Inform users of errors and recovery options
- **Logging**: Log errors for debugging and monitoring
- **Retry Logic**: Implement retry logic for transient errors

### **Documentation Maintenance**

#### **Callback Documentation**
- **Update Descriptions**: Keep callback descriptions current
- **Add Examples**: Provide usage examples for complex callbacks
- **Document Dependencies**: List all dependencies and their purposes
- **Update Maintenance Notes**: Keep maintenance notes current

#### **Code Comments**
- **Inline Comments**: Add comments explaining complex logic
- **Function Documentation**: Document all callback functions
- **Parameter Documentation**: Document all callback parameters
- **Return Value Documentation**: Document all return values

---

## 🔗 **vitalDSP Integration Deep Dive**

### **Integration Architecture**

The vitalDSP webapp is deeply integrated with the vitalDSP library, leveraging its comprehensive signal processing capabilities:

```
vitalDSP Library Integration
├── Core Modules
│   ├── physiological_features/     # HRV, PPG, ECG analysis
│   ├── respiratory_analysis/       # Respiratory rate estimation
│   ├── signal_quality_assessment/  # Quality metrics and validation
│   ├── feature_engineering/        # Feature extraction
│   ├── transforms/                 # Wavelet, FFT, Hilbert transforms
│   ├── advanced_computation/       # ML, Bayesian, Kalman filtering
│   ├── filtering/                  # Traditional and advanced filters
│   └── preprocess/                 # Signal preprocessing
├── Webapp Integration Points
│   ├── Data Service Integration    # Data management and processing
│   ├── Callback Integration        # Real-time analysis callbacks
│   ├── Error Handling              # Graceful fallback mechanisms
│   └── Performance Optimization    # Efficient processing pipelines
```

### **Module-by-Module Integration**

#### **1. Physiological Features Integration**

**Modules Used**:
```python
from vitalDSP.physiological_features import hrv_analysis, time_domain, frequency_domain
from vitalDSP.feature_engineering import ppg_light_features, ppg_autonomic_features, ecg_autonomic_features
```

**Integration Points**:
- **HRV Analysis**: `HRVFeatures` class for comprehensive heart rate variability analysis
- **PPG Features**: `PPGLightFeatureExtractor` for photoplethysmography feature extraction
- **ECG Features**: `ECGAutonomicFeatureExtractor` for electrocardiogram analysis
- **Autonomic Features**: Sympathetic and parasympathetic nervous system indicators

**Callback Usage**:
- `physiological_analysis_callback()`: Main physiological feature extraction
- Real-time feature computation with quality validation
- Multi-signal support (PPG, ECG) with adaptive processing

#### **2. Respiratory Analysis Integration**

**Modules Used**:
```python
from vitalDSP.respiratory_analysis.respiratory_analysis import RespiratoryAnalysis
from vitalDSP.respiratory_analysis.estimate_rr.peak_detection_rr import peak_detection_rr
from vitalDSP.respiratory_analysis.estimate_rr.fft_based_rr import fft_based_rr
from vitalDSP.respiratory_analysis.estimate_rr.frequency_domain_rr import frequency_domain_rr
from vitalDSP.respiratory_analysis.estimate_rr.time_domain_rr import time_domain_rr
from vitalDSP.respiratory_analysis.sleep_apnea_detection.amplitude_threshold import detect_apnea_amplitude
from vitalDSP.respiratory_analysis.fusion.multimodal_analysis import multimodal_analysis
```

**Integration Points**:
- **Multi-method RR Estimation**: Combines time-domain and frequency-domain approaches
- **Sleep Apnea Detection**: Advanced apnea detection algorithms
- **Multimodal Fusion**: Integration of PPG and ECG signals for respiratory analysis
- **Preprocessing**: `PreprocessConfig` and `preprocess_signal` for signal preparation

**Callback Usage**:
- `respiratory_analysis_callback()`: Comprehensive respiratory analysis
- Error handling with graceful fallback when modules are unavailable
- Advanced options for sleep apnea detection and multimodal analysis

#### **3. Signal Quality Assessment Integration**

**Modules Used**:
```python
from vitalDSP.signal_quality_assessment.signal_quality_index import SignalQualityIndex
from vitalDSP.signal_quality_assessment.artifact_detection_removal import ArtifactDetectionRemoval
```

**Integration Points**:
- **Quality Indexing**: Comprehensive signal quality scoring
- **Artifact Detection**: Advanced artifact identification and removal
- **SNR Analysis**: Signal-to-noise ratio computation
- **Baseline Correction**: Drift and wander correction algorithms

**Callback Usage**:
- `quality_assessment_callback()`: Real-time quality monitoring
- Multi-metric analysis with threshold-based classification
- Recommendation system for signal improvement

#### **4. Transform Integration**

**Modules Used**:
```python
from vitalDSP.transforms.wavelet_transform import WaveletTransform
from vitalDSP.transforms.fourier_transform import FourierTransform
from vitalDSP.transforms.hilbert_transform import HilbertTransform
```

**Integration Points**:
- **Wavelet Analysis**: Multi-resolution signal analysis
- **Fourier Transforms**: Frequency domain analysis
- **Hilbert Transforms**: Instantaneous feature extraction
- **Time-Frequency Analysis**: Combined time and frequency domain processing

**Callback Usage**:
- `frequency_domain_callback()`: Frequency domain analysis
- `physiological_analysis_callback()`: Transform-based feature extraction
- Real-time visualization of transform results

#### **5. Advanced Computation Integration**

**Modules Used**:
```python
from vitalDSP.advanced_computation.anomaly_detection import AnomalyDetection
from vitalDSP.advanced_computation.bayesian_analysis import GaussianProcess
from vitalDSP.advanced_computation.kalman_filter import KalmanFilter
```

**Integration Points**:
- **Anomaly Detection**: Machine learning-based anomaly identification
- **Bayesian Analysis**: Uncertainty quantification and probabilistic modeling
- **Kalman Filtering**: State estimation and noise reduction
- **Advanced ML**: Deep learning integration for signal processing

**Callback Usage**:
- `advanced_callbacks.py`: Advanced computational methods
- Real-time anomaly detection and scoring
- Probabilistic analysis with uncertainty quantification

#### **6. Filtering Integration**

**Modules Used**:
```python
# Traditional filtering through scipy.signal integration
# Advanced filtering through vitalDSP filtering modules
```

**Integration Points**:
- **Traditional Filters**: Butterworth, Chebyshev, Elliptic filters
- **Advanced Filters**: Kalman, adaptive, Wiener filtering
- **Neural Filtering**: Deep learning-based denoising
- **Ensemble Methods**: Multiple filter combination techniques

**Callback Usage**:
- `signal_filtering_callbacks.py`: Comprehensive filtering interface
- `vitaldsp_callbacks.py`: Integrated filtering in time domain analysis
- Real-time filter parameter adjustment and visualization

### **Error Handling and Fallback Mechanisms**

#### **Import Error Handling**
```python
def _import_vitaldsp_modules():
    """Import vitalDSP modules with comprehensive error handling."""
    global RespiratoryAnalysis, peak_detection_rr, fft_based_rr
    
    try:
        from vitalDSP.respiratory_analysis.respiratory_analysis import RespiratoryAnalysis
        logger.info("✓ RespiratoryAnalysis imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import RespiratoryAnalysis: {e}")
        RespiratoryAnalysis = None
```

#### **Graceful Degradation**
- **Module Availability Checks**: Verifies vitalDSP module availability before use
- **Fallback Processing**: Uses alternative methods when vitalDSP modules are unavailable
- **User Notifications**: Informs users about missing functionality
- **Logging**: Comprehensive logging of import successes and failures

### **Performance Optimization**

#### **Lazy Loading**
- Modules are imported only when needed
- Reduces startup time and memory usage
- Enables partial functionality when some modules are unavailable

#### **Caching**
- Analysis results are cached in data stores
- Reduces redundant computation
- Improves real-time responsiveness

#### **Efficient Processing**
- Batch processing for multiple analysis types
- Optimized data structures for large datasets
- Memory-efficient signal processing pipelines

### **Data Flow Integration**

```
User Input → Webapp Callbacks → vitalDSP Functions → Processed Results → UI Updates
     ↓              ↓                    ↓                    ↓              ↓
File Upload → Data Service → vitalDSP Processing → Analysis Results → Plotly Plots
     ↓              ↓                    ↓                    ↓              ↓
CSV/Excel → Load & Parse → Signal Analysis → Quality Metrics → Real-time Updates
```

### **Future Integration Opportunities**

1. **Additional vitalDSP Modules**: Integration of new vitalDSP capabilities as they become available
2. **Real-time Processing**: Enhanced real-time signal processing capabilities
3. **Machine Learning**: Deeper integration with ML-based analysis methods
4. **Cloud Processing**: Integration with cloud-based vitalDSP processing services
5. **API Integration**: RESTful API integration with vitalDSP services

---

## 🔄 Data Flow

### Data Flow Architecture

```
User Upload → Data Service → Processing → Callbacks → UI Updates
     ↓              ↓            ↓           ↓           ↓
File Upload → DataService → vitalDSP → Dash Callbacks → Plotly Plots
     ↓              ↓            ↓           ↓           ↓
CSV/Excel → Load & Parse → Signal Processing → Real-time Updates → Visualization
```

### Data Management

#### 1. **Data Service** (`services/data/data_service.py`)

**Key Methods**:
- `load_data()`: Load data from various file formats
- `process_data()`: Process and normalize signal data
- `get_data_segment()`: Extract specific time segments
- `apply_filters()`: Apply signal filtering
- `calculate_metrics()`: Compute analysis metrics

**Data Storage**:
- In-memory data storage with `dcc.Store` components
- Persistent data configuration
- Column mapping for different signal types

#### 2. **Data Stores**

**Global Stores**:
- `store-uploaded-data`: Main data storage
- `store-data-config`: Data configuration
- `store-processed-data`: Processed signal data
- `store-analysis-results`: Analysis results

**Page-specific Stores**:
- `store-time-domain-data`: Time domain analysis data
- `store-filtered-data`: Filtered signal data
- `store-frequency-data`: Frequency analysis data
- `store-filtering-data`: Filtering results

### Signal Processing Pipeline

1. **Data Upload**: User uploads CSV/Excel file
2. **Data Validation**: Check file format and structure
3. **Column Mapping**: Auto-detect time and signal columns
4. **Data Processing**: Apply sampling frequency and normalization
5. **Analysis**: Perform requested analysis (time/frequency/filtering)
6. **Visualization**: Update plots and tables with results
7. **Export**: Allow user to export results

---

## ⚙️ Configuration

### Configuration System (`config/settings.py`)

#### 1. **AppConfig Class**
```python
@dataclass
class AppConfig:
    # App metadata
    APP_NAME: str = "Vital-DSP Comprehensive Dashboard"
    APP_VERSION: str = "1.0.0"
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # File upload settings
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS: List[str] = [".csv", ".xlsx", ".xls", ".txt", ".dat"]
    
    # Data processing settings
    DEFAULT_SAMPLING_FREQ: int = 1000
    MAX_PLOT_POINTS: int = 1000
    
    # UI settings
    SIDEBAR_WIDTH: int = 280
    HEADER_HEIGHT: int = 64
```

#### 2. **ColumnMapping Class**
- Automatic column detection for different signal types
- Pattern matching for time, PPG, ECG, and other signal columns
- Configurable column mapping rules

#### 3. **UIStyles Class**
- Color schemes and theming
- Spacing and layout constants
- Component styling options

### Environment Variables

The application supports environment variable configuration:
- `HOST`: Server host (default: "0.0.0.0")
- `PORT`: Server port (default: 8000)
- `DEBUG`: Debug mode (default: false)

---

## 🔧 Services

### Data Service (`services/data/data_service.py`)

**Purpose**: Centralized data management and processing

**Key Features**:
- Multi-format file support (CSV, Excel, TXT)
- Automatic column detection and mapping
- Signal preprocessing and normalization
- Data segmentation and windowing
- Integration with vitalDSP processing functions

**Usage Example**:
```python
from vitalDSP_webapp.services.data.data_service import DataService

data_service = DataService()
df = data_service.load_data("signal_data.csv")
processed_data = data_service.process_data(df, sampling_freq=1000)
```

### Settings Service (`services/settings_service.py`)

**Purpose**: Application settings management

**Key Features**:
- User preference storage
- Configuration persistence
- Settings validation
- Default value management

---

## 🌐 API Integration

### FastAPI Endpoints (`api/endpoints.py`)

**Health Endpoints**:
- `GET /api/health`: Application health check
- `GET /healthz`: Kubernetes health check

**Future API Endpoints**:
- Data upload endpoints
- Analysis result retrieval
- Configuration management
- Export functionality

### API Architecture

```
FastAPI Server
├── / (Root) → Dash Application
├── /api/health → Health check
├── /api/data → Data management (future)
├── /api/analysis → Analysis results (future)
└── /api/export → Export functionality (future)
```

---

## 🛠️ Development Guidelines

### Code Organization

1. **Modular Structure**: Keep related functionality together
2. **Clear Naming**: Use descriptive function and variable names
3. **Documentation**: Include docstrings for all functions
4. **Error Handling**: Implement proper exception handling
5. **Logging**: Use structured logging throughout

### Callback Development

1. **Single Responsibility**: Each callback should handle one specific task
2. **Input Validation**: Validate all inputs before processing
3. **Error Handling**: Gracefully handle errors and provide user feedback
4. **Performance**: Optimize for real-time updates
5. **State Management**: Use appropriate State vs Input components

### Layout Development

1. **Responsive Design**: Ensure layouts work on different screen sizes
2. **Consistent Styling**: Use centralized CSS and theme variables
3. **Component Reuse**: Create reusable layout components
4. **Accessibility**: Follow accessibility best practices
5. **User Experience**: Design intuitive and efficient interfaces

### Testing

1. **Unit Tests**: Test individual functions and callbacks
2. **Integration Tests**: Test complete workflows
3. **UI Tests**: Test user interactions and visualizations
4. **Performance Tests**: Ensure real-time performance

### Deployment

1. **Environment Configuration**: Use environment variables for production
2. **Static Assets**: Properly serve CSS and image files
3. **Logging**: Configure appropriate logging levels
4. **Monitoring**: Implement health checks and monitoring
5. **Scaling**: Design for horizontal scaling if needed

---

## 📚 Additional Resources

- **Dash Documentation**: https://dash.plotly.com/
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Plotly Documentation**: https://plotly.com/python/
- **Bootstrap Documentation**: https://getbootstrap.com/

---

## 🔍 Troubleshooting

### Common Issues

1. **Callback Errors**: Check input/output component IDs match exactly
2. **Data Loading**: Verify file format and column structure
3. **Performance**: Reduce plot points or optimize data processing
4. **Layout Issues**: Check responsive design and CSS conflicts
5. **Memory Issues**: Clear data stores and optimize data handling

### Debug Mode

Enable debug mode by setting `DEBUG=True` in configuration or environment variable `DEBUG=true`.

---

*This documentation is maintained alongside the codebase. Please update it when making structural changes to the application.*
