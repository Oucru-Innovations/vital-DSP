# VitalDSP Webapp Integration Guide

## Enhanced Data Service Integration with Dash Callbacks

This guide explains how to integrate the `EnhancedDataService` with your Dash webapp callbacks.

### Table of Contents
1. [Basic Integration](#basic-integration)
2. [Progress Callbacks](#progress-callbacks)
3. [Error Handling](#error-handling)
4. [Memory Management](#memory-management)
5. [Common Patterns](#common-patterns)
6. [Performance Tips](#performance-tips)

----

### 1. Basic Integration

#### Simple File Loading

```python
from vitalDSP_webapp.services.data.enhanced_data_service import get_data_service, LoadingStrategy

# In your callback
@app.callback(
    Output('signal-graph', 'figure'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def load_and_display_signal(contents, filename):
    if contents is None:
        return {}

    # Get data service singleton
    data_service = get_data_service()

    try:
        # Automatic strategy selection based on file size
        df = data_service.load_data(
            file_path=filename,
            strategy=None  # Auto-select: Standard/Chunked/Memory-Mapped
        )

        # Create figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df['signal'].values, mode='lines'))
        return fig

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return {}
```

#### Manual Strategy Selection

```python
# For very large files, explicitly use memory-mapped strategy
if file_size_mb > 500:
    data = data_service.load_data(
        file_path=filename,
        strategy=LoadingStrategy.MEMORY_MAPPED
    )
elif file_size_mb > 50:
    data = data_service.load_data(
        file_path=filename,
        strategy=LoadingStrategy.CHUNKED
    )
else:
    data = data_service.load_data(
        file_path=filename,
        strategy=LoadingStrategy.STANDARD
    )
```

----

### 2. Progress Callbacks

#### Background Loading with Progress Updates

```python
from vitalDSP_webapp.services.data.enhanced_data_service import ProgressiveDataLoader, LoadingProgress

# Global progressive loader (or store in webapp state)
progressive_loader = ProgressiveDataLoader(max_workers=2)

@app.callback(
    Output('loading-progress', 'children'),
    Output('loading-status', 'data'),
    Input('start-loading-btn', 'n_clicks'),
    State('file-path-input', 'value'),
    prevent_initial_call=True
)
def start_background_loading(n_clicks, file_path):
    """Start background loading and return task ID"""

    def progress_callback(progress: LoadingProgress):
        # Store progress in Redis/file system for other callbacks to read
        store_progress(progress.task_id, {
            'percent_complete': progress.percent_complete,
            'loading_speed': progress.loading_speed_mbps,
            'estimated_remaining': progress.estimated_remaining_seconds,
            'status': progress.status
        })

    # Start background loading
    task_id = progressive_loader.load_in_background(
        file_path=file_path,
        strategy=LoadingStrategy.CHUNKED,
        callback=progress_callback
    )

    return f"Loading started (Task: {task_id})", {'task_id': task_id, 'status': 'loading'}

@app.callback(
    Output('loading-progress', 'children', allow_duplicate=True),
    Output('data-ready', 'data'),
    Input('progress-interval', 'n_intervals'),
    State('loading-status', 'data'),
    prevent_initial_call=True
)
def update_loading_progress(n_intervals, loading_status):
    """Poll for loading progress updates"""

    if loading_status is None or loading_status.get('status') != 'loading':
        raise PreventUpdate

    task_id = loading_status['task_id']

    # Check if loading complete
    result, error = progressive_loader.get_result(task_id, timeout=0.1)

    if result is not None:
        # Loading complete
        return "Loading complete!", {'ready': True, 'data': result}

    if error is not None:
        # Loading failed
        return f"Loading failed: {error}", {'ready': False, 'error': error}

    # Still loading - get progress
    progress = retrieve_progress(task_id)  # From storage
    if progress:
        return (
            f"Loading: {progress['percent_complete']:.1f}% | "
            f"Speed: {progress['loading_speed']:.1f} MB/s | "
            f"ETA: {progress['estimated_remaining']:.0f}s"
        ), loading_status

    return "Loading...", loading_status
```

----

### 3. Error Handling

#### Graceful Error Handling with User Feedback

```python
@app.callback(
    Output('error-message', 'children'),
    Output('signal-data', 'data'),
    Input('process-btn', 'n_clicks'),
    State('file-path', 'value'),
    prevent_initial_call=True
)
def process_with_error_handling(n_clicks, file_path):
    """Example of robust error handling"""

    data_service = get_data_service()

    try:
        # Attempt to load data
        data = data_service.load_data(file_path)

        # Process data
        processed = process_signal(data)

        return None, processed  # No error, return data

    except FileNotFoundError:
        return "Error: File not found. Please check the file path.", None

    except MemoryError:
        return (
            "Error: Insufficient memory to load file. "
            "Try closing other applications or processing a smaller file."
        ), None

    except ValueError as e:
        return f"Error: Invalid data format - {str(e)}", None

    except Exception as e:
        logger.exception("Unexpected error during processing")
        return f"Error: An unexpected error occurred - {str(e)}", None
```

#### Retry Logic for Transient Errors

```python
from time import sleep

def load_with_retry(file_path, max_retries=3, delay=1.0):
    """Load data with exponential backoff retry"""

    data_service = get_data_service()

    for attempt in range(max_retries):
        try:
            return data_service.load_data(file_path)
        except (IOError, OSError) as e:
            if attempt < max_retries - 1:
                logger.warning(f"Loading failed (attempt {attempt + 1}/{max_retries}): {e}")
                sleep(delay * (2 ** attempt))  # Exponential backoff
            else:
                raise  # Final attempt failed, re-raise exception
```

----

### 4. Memory Management

#### Proper Cleanup After Processing

```python
import gc

@app.callback(
    Output('processing-complete', 'data'),
    Input('start-processing-btn', 'n_clicks'),
    State('large-signal-data', 'data'),
    prevent_initial_call=True
)
def process_large_signal(n_clicks, signal_data):
    """Example with proper memory cleanup"""

    try:
        # Load data
        data = pd.DataFrame(signal_data)

        # Process
        result = expensive_processing(data)

        # Convert to JSON-serializable format
        output = result.to_dict()

        return output

    finally:
        # Cleanup large objects
        del data
        del result
        gc.collect()  # Force garbage collection
```

#### Using Context Managers for Automatic Cleanup

```python
from contextlib import contextmanager

@contextmanager
def managed_data_loading(file_path):
    """Context manager for automatic cleanup"""
    data_service = get_data_service()
    data = None

    try:
        data = data_service.load_data(file_path)
        yield data
    finally:
        if data is not None:
            del data
            gc.collect()

# Usage
@app.callback(...)
def process_callback(...):
    with managed_data_loading(file_path) as data:
        # Process data
        result = analyze(data)
        return result
    # Data automatically cleaned up here
```

----

### 5. Common Patterns

#### Pattern 1: Two-Stage Loading (Metadata → Full Data)

```python
# Callback 1: Load metadata only
@app.callback(
    Output('file-info', 'children'),
    Input('file-selector', 'value')
)
def display_file_info(file_path):
    """Load only metadata for quick display"""
    if not file_path:
        raise PreventUpdate

    # Get file info without loading full data
    data_service = get_data_service()
    info = data_service.get_file_info(file_path)

    return html.Div([
        html.P(f"File size: {info['size_mb']:.1f} MB"),
        html.P(f"Estimated load time: {info['estimated_load_time']:.1f}s"),
        html.P(f"Recommended strategy: {info['recommended_strategy']}")
    ])

# Callback 2: Load full data when user confirms
@app.callback(
    Output('full-data', 'data'),
    Input('load-full-data-btn', 'n_clicks'),
    State('file-selector', 'value'),
    prevent_initial_call=True
)
def load_full_data(n_clicks, file_path):
    """Load full data after user confirmation"""
    data_service = get_data_service()
    return data_service.load_data(file_path).to_dict()
```

#### Pattern 2: Chunked Processing with Incremental Display

```python
@app.callback(
    Output('signal-graph', 'figure', allow_duplicate=True),
    Output('chunk-progress', 'children'),
    Input('chunk-interval', 'n_intervals'),
    State('chunked-loader-state', 'data'),
    prevent_initial_call=True
)
def display_chunks_incrementally(n_intervals, loader_state):
    """Display signal data as chunks are loaded"""

    if loader_state is None or loader_state.get('complete'):
        raise PreventUpdate

    data_service = get_data_service()
    chunk_index = loader_state.get('current_chunk', 0)

    # Get next chunk
    chunk = data_service.get_chunk(
        file_path=loader_state['file_path'],
        chunk_index=chunk_index
    )

    if chunk is None:
        # All chunks loaded
        return dash.no_update, "Loading complete!"

    # Update figure with new chunk
    existing_data = loader_state.get('accumulated_data', [])
    existing_data.extend(chunk['signal'].values.tolist())

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=existing_data, mode='lines'))

    # Update state
    loader_state['current_chunk'] = chunk_index + 1
    loader_state['accumulated_data'] = existing_data

    progress = f"Loaded {chunk_index + 1}/{loader_state['total_chunks']} chunks"

    return fig, progress
```

#### Pattern 3: Segment-Based Analysis

```python
@app.callback(
    Output('segment-results', 'children'),
    Input('analyze-segments-btn', 'n_clicks'),
    State('signal-data', 'data'),
    State('segment-length', 'value'),
    prevent_initial_call=True
)
def analyze_by_segments(n_clicks, signal_data, segment_length_seconds):
    """Analyze signal in segments to save memory"""

    df = pd.DataFrame(signal_data)
    fs = 250  # Sampling rate
    segment_length_samples = int(segment_length_seconds * fs)

    results = []

    # Process in segments
    for start_idx in range(0, len(df), segment_length_samples):
        end_idx = min(start_idx + segment_length_samples, len(df))
        segment = df.iloc[start_idx:end_idx]

        # Analyze segment
        segment_result = analyze_segment(segment)
        results.append(segment_result)

        # Cleanup
        del segment

    # Aggregate results
    summary = aggregate_segment_results(results)

    return html.Div([
        html.H4("Segment Analysis Results"),
        html.P(f"Total segments analyzed: {len(results)}"),
        html.P(f"Average quality: {summary['avg_quality']:.2f}"),
        # ... more results
    ])
```

----

### 6. Performance Tips

#### Tip 1: Use Caching for Expensive Operations

```python
from flask_caching import Cache

cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})

@cache.memoize(timeout=3600)  # Cache for 1 hour
def expensive_feature_extraction(file_path):
    """Cache expensive operations"""
    data_service = get_data_service()
    data = data_service.load_data(file_path)
    features = extract_features(data)
    return features

@app.callback(...)
def display_features(file_path):
    # This will use cached result if available
    features = expensive_feature_extraction(file_path)
    return display_features_table(features)
```

#### Tip 2: Lazy Loading with Deferring

```python
@app.callback(
    Output('expensive-graph', 'figure'),
    Input('show-graph-btn', 'n_clicks'),
    State('signal-data', 'data'),
    prevent_initial_call=True
)
def show_expensive_graph(n_clicks, signal_data):
    """Only compute expensive visualization when user requests it"""

    if signal_data is None:
        raise PreventUpdate

    # Only load and process when user clicks button
    df = pd.DataFrame(signal_data)

    # Downsample for visualization
    downsampled = downsample_for_display(df, target_points=5000)

    fig = create_detailed_figure(downsampled)
    return fig

def downsample_for_display(df, target_points=5000):
    """Downsample large signals for browser display"""
    if len(df) <= target_points:
        return df

    # Use LTTB (Largest Triangle Three Buckets) algorithm
    indices = lttb_downsample_indices(df['signal'].values, target_points)
    return df.iloc[indices]
```

#### Tip 3: Batch Operations

```python
@app.callback(
    Output('batch-results', 'data'),
    Input('process-all-btn', 'n_clicks'),
    State('file-list', 'data'),
    prevent_initial_call=True
)
def process_multiple_files(n_clicks, file_list):
    """Process multiple files efficiently"""

    data_service = get_data_service()
    results = {}

    # Load all files with shared resources
    for file_path in file_list:
        try:
            # Use chunked loading for memory efficiency
            data = data_service.load_data(
                file_path,
                strategy=LoadingStrategy.CHUNKED
            )

            # Process
            result = quick_analysis(data)
            results[file_path] = result

            # Cleanup
            del data

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            results[file_path] = {'error': str(e)}

    return results
```

---

### Common Pitfalls to Avoid

❌ **Pitfall 1: Not cleaning up large data**
```python
# BAD
@app.callback(...)
def bad_callback(file_path):
    data = load_large_file(file_path)
    result = process(data)
    return result  # 'data' stays in memory!

# GOOD
@app.callback(...)
def good_callback(file_path):
    data = load_large_file(file_path)
    try:
        result = process(data)
        return result
    finally:
        del data
        gc.collect()
```

❌ **Pitfall 2: Blocking UI with synchronous loading**
```python
# BAD
@app.callback(...)
def bad_callback(file_path):
    data = data_service.load_data(file_path)  # Blocks UI for minutes
    return create_figure(data)

# GOOD
@app.callback(...)
def good_callback(file_path):
    # Start background loading
    task_id = progressive_loader.load_in_background(file_path, callback=on_complete)
    return f"Loading started (Task: {task_id})"
```

❌ **Pitfall 3: Loading entire file when only subset needed**
```python
# BAD
@app.callback(...)
def bad_callback(file_path, start_time, end_time):
    full_data = data_service.load_data(file_path)  # Loads entire file
    subset = full_data[start_time:end_time]
    return process(subset)

# GOOD
@app.callback(...)
def good_callback(file_path, start_time, end_time):
    # Load only needed segment with memory-mapped access
    data = data_service.load_segment(
        file_path,
        start_time=start_time,
        end_time=end_time,
        strategy=LoadingStrategy.MEMORY_MAPPED
    )
    return process(data)
```

---

### Full Example: Complete Integration

```python
from dash import Dash, html, dcc, Input, Output, State, no_update
from vitalDSP_webapp.services.data.enhanced_data_service import (
    get_data_service, ProgressiveDataLoader, LoadingStrategy, LoadingProgress
)
import plotly.graph_objs as go
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Initialize components
app = Dash(__name__)
data_service = get_data_service()
progressive_loader = ProgressiveDataLoader(max_workers=2)

# Layout
app.layout = html.Div([
    dcc.Input(id='file-path', type='text', placeholder='Enter file path'),
    html.Button('Load Data', id='load-btn'),
    html.Div(id='loading-status'),
    html.Div(id='file-info'),
    dcc.Graph(id='signal-graph'),
    dcc.Interval(id='progress-interval', interval=500, disabled=True),
    dcc.Store(id='loading-task'),
    dcc.Store(id='signal-data')
])

# Callback 1: Start loading
@app.callback(
    Output('loading-task', 'data'),
    Output('progress-interval', 'disabled'),
    Output('loading-status', 'children'),
    Input('load-btn', 'n_clicks'),
    State('file-path', 'value'),
    prevent_initial_call=True
)
def start_loading(n_clicks, file_path):
    """Start background data loading"""

    if not file_path:
        return None, True, "Please enter a file path"

    def progress_callback(progress: LoadingProgress):
        logger.info(f"Loading progress: {progress.percent_complete:.1f}%")

    try:
        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

        if file_size_mb < 50:
            # Small file - load synchronously
            data = data_service.load_data(file_path)
            return {
                'type': 'sync',
                'data': data.to_dict(),
                'status': 'complete'
            }, True, f"File loaded ({file_size_mb:.1f} MB)"

        else:
            # Large file - load in background
            task_id = progressive_loader.load_in_background(
                file_path=file_path,
                strategy=LoadingStrategy.CHUNKED,
                callback=progress_callback
            )
            return {
                'type': 'async',
                'task_id': task_id,
                'status': 'loading'
            }, False, f"Loading file ({file_size_mb:.1f} MB)..."

    except Exception as e:
        logger.error(f"Failed to start loading: {e}")
        return None, True, f"Error: {str(e)}"

# Callback 2: Monitor loading progress
@app.callback(
    Output('loading-task', 'data', allow_duplicate=True),
    Output('progress-interval', 'disabled', allow_duplicate=True),
    Output('loading-status', 'children', allow_duplicate=True),
    Input('progress-interval', 'n_intervals'),
    State('loading-task', 'data'),
    prevent_initial_call=True
)
def update_progress(n_intervals, loading_task):
    """Monitor background loading progress"""

    if loading_task is None or loading_task.get('type') != 'async':
        raise no_update

    if loading_task.get('status') == 'complete':
        raise no_update

    task_id = loading_task['task_id']

    # Check if complete
    result, error = progressive_loader.get_result(task_id, timeout=0.1)

    if result is not None:
        # Loading complete
        loading_task['data'] = result.to_dict()
        loading_task['status'] = 'complete'
        return loading_task, True, "Loading complete!"

    if error is not None:
        # Loading failed
        loading_task['status'] = 'error'
        loading_task['error'] = str(error)
        return loading_task, True, f"Loading failed: {error}"

    # Still loading
    return loading_task, False, "Loading..."

# Callback 3: Display signal
@app.callback(
    Output('signal-graph', 'figure'),
    Output('file-info', 'children'),
    Input('loading-task', 'data')
)
def display_signal(loading_task):
    """Display loaded signal"""

    if loading_task is None or loading_task.get('status') != 'complete':
        return {}, ""

    try:
        # Get data
        data_dict = loading_task.get('data')
        if data_dict is None:
            return {}, "No data"

        df = pd.DataFrame(data_dict)

        # Downsample for display if needed
        if len(df) > 10000:
            display_df = df.iloc[::len(df)//10000]  # Simple downsampling
        else:
            display_df = df

        # Create figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=display_df['signal'].values,
            mode='lines',
            name='Signal'
        ))
        fig.update_layout(
            title='Loaded Signal',
            xaxis_title='Sample',
            yaxis_title='Amplitude'
        )

        # File info
        info = html.Div([
            html.P(f"Signal length: {len(df)} samples"),
            html.P(f"Duration: {len(df)/250:.1f} seconds (assuming 250 Hz)"),
            html.P(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        ])

        return fig, info

    except Exception as e:
        logger.error(f"Failed to display signal: {e}")
        return {}, f"Error: {str(e)}"

if __name__ == '__main__':
    app.run_server(debug=True)
```

---

## Summary

**Key Takeaways:**

1. ✅ Always use `get_data_service()` to get the singleton instance
2. ✅ Let automatic strategy selection choose the optimal loader
3. ✅ Use background loading for files > 50MB
4. ✅ Clean up large data objects after use
5. ✅ Downsample data for browser display
6. ✅ Handle errors gracefully with user-friendly messages
7. ✅ Use progress callbacks for long-running operations
8. ✅ Cache expensive operations when appropriate
9. ✅ Load only needed data segments when possible
10. ✅ Use context managers for automatic cleanup

**Performance Guidelines:**

| File Size | Recommended Strategy | Loading Time (estimate) |
|-----------|---------------------|------------------------|
| < 50 MB   | Standard (synchronous) | < 1 second |
| 50-500 MB | Chunked (async with progress) | 5-30 seconds |
| > 500 MB  | Memory-Mapped (segment access) | Instant (zero-copy) |

**Memory Guidelines:**

| Data Size | Max Concurrent Files | Recommended Action |
|-----------|---------------------|-------------------|
| < 100 MB | 10+ | Direct loading |
| 100-500 MB | 3-5 | Chunked loading with cleanup |
| > 500 MB | 1-2 | Memory-mapped access, load segments only |

---

For more information, see:
- `enhanced_data_service.py` source code
- Phase 3 architecture documentation
- Performance benchmarking results
