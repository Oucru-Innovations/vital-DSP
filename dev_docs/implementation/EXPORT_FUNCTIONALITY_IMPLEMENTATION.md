# Export Functionality Implementation - vitalDSP Webapp

## Overview

Comprehensive export functionality has been implemented across all vitalDSP webapp analysis pages, allowing users to export their analysis results in CSV and JSON formats.

## Implementation Date
October 12, 2025

---

## New Modules Created

### 1. `src/vitalDSP_webapp/utils/export_utils.py`

**Purpose**: Core export utility functions for converting analysis data to CSV and JSON formats.

**Functions**:
- `export_filtered_signal_csv()` - Export filtered signals with timestamps to CSV
- `export_filtered_signal_json()` - Export filtered signals to JSON
- `export_features_csv()` - Export extracted features to CSV
- `export_features_json()` - Export extracted features to JSON
- `export_quality_metrics_csv()` - Export quality assessment metrics to CSV
- `export_quality_metrics_json()` - Export quality metrics to JSON
- `export_respiratory_analysis_csv()` - Export respiratory analysis to CSV
- `export_respiratory_analysis_json()` - Export respiratory analysis to JSON
- `export_transform_results_csv()` - Export transform results to CSV
- `export_transform_results_json()` - Export transform results to JSON
- `export_time_domain_analysis_csv()` - Export time domain features to CSV
- `export_time_domain_analysis_json()` - Export time domain features to JSON
- `export_frequency_domain_analysis_csv()` - Export frequency domain features to CSV
- `export_frequency_domain_analysis_json()` - Export frequency domain features to JSON

**Key Features**:
- Metadata headers in CSV files (export date, signal type, sampling frequency, etc.)
- Structured JSON output with metadata sections
- Automatic numpy array conversion for JSON serialization
- Flattening of nested dictionaries for CSV export
- Statistical summaries for large arrays

---

### 2. `src/vitalDSP_webapp/callbacks/utils/export_callbacks.py`

**Purpose**: Dash callbacks for handling export button clicks and generating downloads.

**Callback Functions**:
- `register_filtering_export_callbacks()` - Export callbacks for filtering page
- `register_time_domain_export_callbacks()` - Export callbacks for time domain page
- `register_frequency_domain_export_callbacks()` - Export callbacks for frequency domain page
- `register_physiological_export_callbacks()` - Export callbacks for physiological features page
- `register_quality_export_callbacks()` - Export callbacks for quality assessment page
- `register_respiratory_export_callbacks()` - Export callbacks for respiratory analysis page
- `register_transforms_export_callbacks()` - Export callbacks for transforms page
- `register_all_export_callbacks()` - Master function to register all export callbacks

**Callback Pattern**:
```python
@app.callback(
    Output("download-{page}-csv", "data"),
    Input("btn-export-{page}-csv", "n_clicks"),
    State("store-{page}-data", "data"),
    prevent_initial_call=True,
)
def export_page_csv(n_clicks, data):
    # Convert data to CSV format
    # Return download dict
```

---

### 3. `src/vitalDSP_webapp/utils/export_components.py`

**Purpose**: Reusable UI components for export buttons and download elements.

**Component Functions**:
- `create_export_buttons()` - Simple button group with CSV/JSON buttons
- `create_export_card()` - Card-style export section
- `create_inline_export_buttons()` - Inline buttons for action bars
- `create_dropdown_export_menu()` - Dropdown menu with export options
- `create_export_section_with_preview()` - Export section with data preview

---

## Pages Updated

### 1. **Filtering Page** (`filtering_layout()`)

**Export Capabilities**:
- Filtered signal data with timestamps
- Filter type and parameters in metadata
- Sampling frequency information

**UI Changes**:
- Added "Export" button group with CSV and JSON options next to "Apply Filter" button
- Added `dcc.Download` components for CSV and JSON
- Added `store-filtered-signal` for export data

**Button IDs**:
- `btn-export-filtered-csv`
- `btn-export-filtered-json`
- `download-filtered-csv`
- `download-filtered-json`

---

### 2. **Time Domain Analysis Page** (`time_domain_layout()`)

**Export Capabilities**:
- All time domain features (mean, std, RMS, etc.)
- Peak detection results
- Statistical measures

**UI Changes**:
- Replaced single "Export Results" button with CSV/JSON button group
- Added `dcc.Download` components
- Added `store-time-domain-features` for export data

**Button IDs**:
- `btn-export-time-domain-csv`
- `btn-export-time-domain-json`
- `download-time-domain-csv`
- `download-time-domain-json`

---

### 3. **Frequency Domain Analysis Page** (`frequency_layout()`)

**Export Capabilities**:
- Frequency domain features (power spectral density, dominant frequencies, etc.)
- FFT results
- Spectral measures

**UI Changes**:
- Export button group in action bar
- Download components added

**Button IDs**:
- `btn-export-frequency-domain-csv`
- `btn-export-frequency-domain-json`
- `download-frequency-domain-csv`
- `download-frequency-domain-json`

---

### 4. **Physiological Features Page** (`physiological_layout()`)

**Export Capabilities**:
- HRV metrics (time domain, frequency domain, nonlinear)
- PPG-specific features
- ECG-specific features
- Extracted physiological parameters

**UI Changes**:
- Replaced single "Export Results" button with CSV/JSON button group
- Added download components

**Button IDs**:
- `btn-export-physio-csv`
- `btn-export-physio-json`
- `download-physio-csv`
- `download-physio-json`

---

### 5. **Quality Assessment Page** (`quality_layout()`)

**Export Capabilities**:
- SNR (Signal-to-Noise Ratio)
- Artifact detection results
- Baseline wander metrics
- Signal quality indices (SQI)
- Overall quality score

**UI Changes**:
- Export buttons in results section
- Download components

**Button IDs**:
- `btn-export-quality-csv`
- `btn-export-quality-json`
- `download-quality-csv`
- `download-quality-json`

---

### 6. **Respiratory Analysis Page** (`respiratory_layout()`)

**Export Capabilities**:
- Respiratory rate (RR) estimates from multiple methods
- Breath detection results
- Respiratory variability metrics
- Method comparison results

**UI Changes**:
- Export button group in action bar (already existed)
- Updated to use new export system

**Button IDs**:
- `btn-export-respiratory-csv`
- `btn-export-respiratory-json`
- `download-respiratory-csv`
- `download-respiratory-json`

---

### 7. **Transforms Page** (`transforms_layout()`)

**Export Capabilities**:
- FFT results (frequencies, magnitudes, phases)
- Wavelet transform coefficients
- STFT results
- Hilbert transform results
- Other transform outputs

**UI Changes**:
- Export buttons in transform results section
- Download components

**Button IDs**:
- `btn-export-transforms-csv`
- `btn-export-transforms-json`
- `download-transforms-csv`
- `download-transforms-json`

---

## Export File Formats

### CSV Format

**Structure**:
```csv
# Export Type Export
# Export Date: 2025-10-12 12:34:56
# Signal Type: PPG
# Sampling Frequency: 100 Hz
# Duration: 10.00 seconds
# Samples: 1000
#
time,signal
0.00,1.234
0.01,1.456
...
```

**Features**:
- Metadata as comments (lines starting with `#`)
- Column headers
- Numeric data in rows
- Compatible with Excel, pandas, MATLAB, R

---

### JSON Format

**Structure**:
```json
{
  "metadata": {
    "export_date": "2025-10-12T12:34:56",
    "export_type": "filtered_signal",
    "sampling_frequency": 100,
    "duration": 10.0,
    "samples": 1000,
    "signal_type": "PPG"
  },
  "data": {
    "time": [0.00, 0.01, 0.02, ...],
    "signal": [1.234, 1.456, 1.789, ...]
  }
}
```

**Features**:
- Structured format with metadata section
- Nested objects preserved
- Arrays included
- Easy to parse programmatically

---

## Integration with App

### Step 1: Register Export Callbacks

In `src/vitalDSP_webapp/app.py` or main initialization file:

```python
from vitalDSP_webapp.callbacks.utils.export_callbacks import register_all_export_callbacks

# After creating the app
app = dash.Dash(__name__)

# Register all callbacks
register_all_export_callbacks(app)
```

### Step 2: Ensure Data Stores are Populated

Each analysis callback should populate the corresponding store:

```python
@app.callback(
    Output("store-filtered-signal", "data"),
    ...
)
def apply_filter(...):
    # Perform filtering
    filtered_signal = ...

    # Store data for export
    export_data = {
        'signal': filtered_signal.tolist(),
        'time': time_array.tolist(),
        'sampling_freq': fs,
        'filter_type': filter_type,
        'filter_params': params,
        'signal_type': signal_type
    }

    return export_data
```

---

## Usage Examples

### Example 1: Export Filtered Signal

1. User applies a Butterworth filter to PPG signal
2. Clicks "CSV" button in Export section
3. File `filtered_signal_butterworth.csv` is downloaded with:
   - Filtered signal values
   - Time stamps
   - Filter parameters in header

### Example 2: Export Quality Metrics

1. User runs signal quality assessment
2. Clicks "JSON" button
3. File `signal_quality_metrics.json` is downloaded with:
   - Overall quality score
   - SNR value
   - Artifact counts
   - Individual SQI metrics

### Example 3: Export Time Domain Features

1. User extracts time domain features
2. Clicks "CSV" button
3. File `time_domain_features.csv` is downloaded with:
   - Mean, STD, RMS values
   - Peak counts
   - All extracted features as columns

---

## Benefits

### For Users

1. **Data Portability**: Export results to work in other tools (Excel, MATLAB, Python)
2. **Record Keeping**: Save analysis results for documentation
3. **Batch Processing**: Export multiple analyses for comparison
4. **Reproducibility**: Complete metadata for reproducing analysis

### For Researchers

1. **Publication**: Export data for creating figures and tables
2. **Statistical Analysis**: Import into R, SPSS, or other statistical software
3. **Collaboration**: Share analysis results in standard formats
4. **Archival**: Save complete analysis output for long-term storage

---

## Technical Details

### Data Serialization

**Numpy Array Handling**:
```python
def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    # ... handle other types
```

**Dictionary Flattening** (for CSV):
```python
def flatten_dict(d, parent_key=''):
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            flatten_dict(v, new_key)
        else:
            flattened[new_key] = v
```

---

## Error Handling

All export callbacks include comprehensive error handling:

```python
try:
    # Export logic
    csv_content = export_function(data)
    return {'content': csv_content, 'filename': 'export.csv'}
except Exception as e:
    logger.error(f"Error exporting data: {e}")
    raise PreventUpdate
```

**Prevents**:
- App crashes from export errors
- Invalid file downloads
- User disruption

---

## Future Enhancements

### Planned Features

1. **Excel Export**: Direct .xlsx export with multiple sheets
2. **MATLAB .mat Export**: Native MATLAB format support
3. **Batch Export**: Export all analyses at once
4. **Custom Templates**: User-defined export formats
5. **Automated Reports**: Generate PDF reports with visualizations
6. **Cloud Storage**: Direct export to Google Drive, Dropbox
7. **Export Scheduling**: Automatic periodic exports

### API Endpoints

Consider adding REST API endpoints for programmatic export:

```python
@app.route('/api/export/filtered-signal/<format>')
def api_export_filtered(format):
    # Return CSV or JSON directly
    pass
```

---

## Maintenance

### Adding New Export Types

1. Create export function in `export_utils.py`
2. Add callback in `export_callbacks.py`
3. Add UI buttons in page layout
4. Add download components
5. Update documentation

### Testing

Create unit tests for export functions:

```python
def test_export_filtered_signal_csv():
    signal = np.array([1, 2, 3, 4, 5])
    time = np.array([0, 0.01, 0.02, 0.03, 0.04])

    csv_content = export_filtered_signal_csv(signal, time, 100)

    assert '# Filtered Signal Export' in csv_content
    assert 'time,signal' in csv_content
    assert '0,1' in csv_content
```

---

## Summary

**Modules Created**: 3
**Pages Updated**: 7
**Total Export Functions**: 14+
**Button Components**: 4 types
**Formats Supported**: CSV, JSON

The export functionality is now comprehensive, consistent, and user-friendly across all analysis pages in the vitalDSP webapp!
