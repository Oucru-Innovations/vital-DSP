Data Loading Guide
==================

The vitalDSP Data Loader module provides a comprehensive and unified interface for loading physiological signal data from various sources and formats. This guide covers all features and usage patterns.

.. contents:: Table of Contents
   :local:
   :depth: 3

Overview
--------

The Data Loader module supports:

* **12+ File Formats**: CSV, TSV, Excel, JSON, HDF5, EDF, WFDB, NumPy, MATLAB, Pickle, Parquet, OUCRU CSV
* **Automatic Format Detection**: Intelligently detects file format from extension
* **Multi-Channel Support**: Load and manage multiple signal channels
* **Data Validation**: Built-in validation and quality checks
* **Metadata Extraction**: Automatic extraction of sampling rates and signal properties
* **Stream Loading**: Support for real-time data streams
* **Memory Efficient**: Chunk-based loading for large files
* **Special Format Support**: OUCRU's array-per-row CSV format with automatic timestamp interpolation

Quick Start
-----------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from vitalDSP.utils.data_processing.data_loader import DataLoader

    # Load a CSV file
    loader = DataLoader('ecg_data.csv')
    data = loader.load()

    print(f"Loaded {len(data)} samples")
    print(f"Sampling rate: {loader.sampling_rate} Hz")

Convenience Functions
~~~~~~~~~~~~~~~~~~~~~

For quick loading, use the convenience functions:

.. code-block:: python

    from vitalDSP.utils.data_processing.data_loader import load_signal, load_multi_channel

    # Quick signal loading
    data = load_signal('ecg_data.csv', sampling_rate=250)

    # Load multi-channel data
    channels = load_multi_channel('multi_channel.edf', channels=['ECG', 'PPG'])

Supported Formats
-----------------

CSV Files
~~~~~~~~~

Load comma-separated values files:

.. code-block:: python

    # Basic CSV loading
    loader = DataLoader('data.csv')
    data = loader.load()

    # Load specific columns
    data = loader.load(columns=['time', 'ecg', 'ppg'])

    # Custom delimiter
    loader = DataLoader('data.txt')
    data = loader.load(delimiter=';')

    # No header
    data = loader.load(header=None)

**Requirements**: pandas

TSV Files
~~~~~~~~~

Tab-separated values files:

.. code-block:: python

    loader = DataLoader('data.tsv')
    data = loader.load()

**Requirements**: pandas

OUCRU CSV Format
~~~~~~~~~~~~~~~~

OUCRU's special CSV format where each row represents 1 second of data with signal values stored as an array:

.. code-block:: python

    from vitalDSP.utils.data_processing.data_loader import load_oucru_csv, DataLoader, DataFormat

    # Method 1: Using convenience function (recommended)
    signal, metadata = load_oucru_csv(
        'oucru_ecg.csv',
        time_column='timestamp',
        signal_column='ecg_values',
        sampling_rate=250  # Optional, auto-detected if not provided
    )

    print(f"Loaded {len(signal)} samples at {metadata['sampling_rate']} Hz")
    print(f"Duration: {metadata['duration_seconds']:.2f} seconds")
    print(f"Number of rows: {metadata['n_rows']}")

    # Method 2: Using DataLoader directly
    loader = DataLoader('oucru_ecg.csv', format=DataFormat.OUCRU_CSV, sampling_rate=250)
    data = loader.load(
        time_column='timestamp',
        signal_column='ecg_values',
        interpolate_time=True  # Generate timestamps for each sample
    )

    # Access expanded data with interpolated timestamps
    print(data.head())
    # Output:
    #                 timestamp  signal
    # 0 2024-01-01 00:00:00.000    1.20
    # 1 2024-01-01 00:00:00.004    1.21
    # 2 2024-01-01 00:00:00.008    1.22
    # ...

**Format Specification:**

The OUCRU CSV format has these characteristics:

* Each row represents exactly 1 second of data
* Signal values are stored as an array string: ``"[1.2, 1.3, 1.4, ..., 2.0]"``
* Array length equals the sampling rate (e.g., 250 samples for 250 Hz)
* Timestamps mark the start of each second
* Sampling rate can be:

  - Specified in a column (e.g., ``sampling_rate``)
  - Passed as a parameter
  - Auto-detected from array length

**Example CSV format:**

.. code-block:: text

    timestamp,ecg_values,sampling_rate
    2024-01-01 00:00:00,"[1.2, 1.3, 1.4, 1.5, 1.6]",5
    2024-01-01 00:00:01,"[1.7, 1.8, 1.9, 2.0, 2.1]",5
    2024-01-01 00:00:02,"[2.2, 2.3, 2.4, 2.5, 2.6]",5

**Features:**

* **Automatic Array Parsing**: Uses ``ast.literal_eval()`` to safely parse array strings
* **Timestamp Interpolation**: Generates precise timestamps for each sample within the second
* **Flexible Sampling Rate Detection**: Multiple methods with priority:

  1. From sampling_rate column in CSV (highest priority)
  2. From explicit ``sampling_rate`` parameter
  3. From signal type hint (``signal_type_hint='ppg'`` uses 100 Hz, ``'ecg'`` uses 128 Hz)
  4. Inferred from array length (lowest priority)

* **Data Validation**: Checks for consistent array lengths and handles missing data
* **Metadata Extraction**: Extracts duration, sample counts, and timing information

**Advanced Usage:**

.. code-block:: python

    # Load with custom column names
    signal, metadata = load_oucru_csv(
        'custom_format.csv',
        time_column='datetime',
        signal_column='ppg_signal',
        sampling_rate_column='fs'
    )

    # Use signal type hint for default sampling rate
    # PPG signals: uses default_ppg_rate=100 Hz
    signal, metadata = load_oucru_csv(
        'ppg_data.csv',
        signal_type_hint='ppg'  # Will use 100 Hz if no column/parameter
    )

    # ECG signals: uses default_ecg_rate=128 Hz
    signal, metadata = load_oucru_csv(
        'ecg_data.csv',
        signal_type_hint='ecg'  # Will use 128 Hz if no column/parameter
    )

    # Override default rates
    signal, metadata = load_oucru_csv(
        'ecg_data.csv',
        signal_type_hint='ecg',
        default_ecg_rate=250  # Use 250 Hz instead of 128 Hz
    )

    # Disable timestamp interpolation for faster loading
    loader = DataLoader('large_file.csv', format=DataFormat.OUCRU_CSV)
    data = loader.load(interpolate_time=False)
    # Returns data with row-level timestamps only

    # Handle multiple sampling rates (uses first value, warns about inconsistencies)
    signal, metadata = load_oucru_csv(
        'variable_sr.csv',
        time_column='timestamp',
        signal_column='signal'
    )

    # Access original row-based data
    row_data = metadata['row_data']  # Original CSV with array strings

**Error Handling:**

The loader handles common issues:

* **Inconsistent array lengths**: Pads or truncates to match the first row's length
* **Multiple sampling rates**: Uses the first value and warns
* **Parse errors**: Provides detailed error messages with row numbers
* **Missing timestamps**: Falls back to numeric indices

**Requirements**: pandas, numpy, ast (standard library)

Excel Files
~~~~~~~~~~~

Microsoft Excel spreadsheets:

.. code-block:: python

    # Load Excel file
    loader = DataLoader('data.xlsx')
    data = loader.load()

    # Specify sheet
    data = loader.load(sheet_name='Sheet2')

    # Load specific columns
    data = loader.load(columns=['ECG', 'PPG'])

**Requirements**: pandas, openpyxl

JSON Files
~~~~~~~~~~

JavaScript Object Notation files:

.. code-block:: python

    # Load JSON array
    loader = DataLoader('data.json')
    data = loader.load()

    # JSON with metadata
    # File format: {"sampling_rate": 250, "data": [...]}
    loader = DataLoader('data_with_meta.json')
    data = loader.load()
    print(loader.metadata)

**Requirements**: json (built-in)

HDF5 Files
~~~~~~~~~~

Hierarchical Data Format:

.. code-block:: python

    # Load HDF5
    loader = DataLoader('data.h5')
    data = loader.load(key='signal_data')

    # List available keys
    loader.load()
    print(loader.metadata['available_keys'])

**Requirements**: h5py, tables

EDF Files
~~~~~~~~~

European Data Format (medical standard):

.. code-block:: python

    # Load EDF file (returns dictionary of channels)
    loader = DataLoader('recording.edf')
    data = loader.load()

    # Load specific channels
    data = loader.load(channels=['ECG I', 'ECG II'])

    # Access channel data
    ecg = data['ECG I']

    # View metadata
    print(loader.metadata['channel_labels'])
    print(loader.metadata['start_datetime'])
    print(loader.metadata['duration'])

**Requirements**: pyedflib

**Installation**:

.. code-block:: bash

    pip install pyedflib

WFDB Files
~~~~~~~~~~

PhysioNet WFDB format:

.. code-block:: python

    # Load WFDB record (returns dictionary of channels)
    loader = DataLoader('mitdb/100.dat')
    data = loader.load()

    # Load specific channels
    data = loader.load(channels=['MLII', 'V5'])

    # Access annotations if available
    if 'annotations' in loader.metadata:
        annotations = loader.metadata['annotations']
        print(f"Found {len(annotations['sample'])} annotations")

**Requirements**: wfdb

**Installation**:

.. code-block:: bash

    pip install wfdb

NumPy Files
~~~~~~~~~~~

NumPy array files (.npy and .npz):

.. code-block:: python

    # Load .npy file
    loader = DataLoader('signal.npy')
    data = loader.load()

    # Load .npz file (returns dictionary)
    loader = DataLoader('signals.npz')
    data = loader.load()

    ecg = data['ecg']
    ppg = data['ppg']

**Requirements**: numpy

MATLAB Files
~~~~~~~~~~~~

MATLAB .mat files:

.. code-block:: python

    # Load MATLAB file (returns dictionary)
    loader = DataLoader('data.mat')
    data = loader.load()

    # Load specific variables
    data = loader.load(variable_names=['ecg', 'ppg'])

    # Access variables
    ecg = data['ecg']

**Requirements**: scipy

Pickle Files
~~~~~~~~~~~~

Python pickle format:

.. code-block:: python

    loader = DataLoader('data.pkl')
    data = loader.load()

**Requirements**: pickle (built-in)

Parquet Files
~~~~~~~~~~~~~

Apache Parquet columnar format:

.. code-block:: python

    loader = DataLoader('data.parquet')
    data = loader.load(columns=['ecg', 'ppg'])

**Requirements**: pyarrow

**Installation**:

.. code-block:: bash

    pip install pyarrow

Advanced Features
-----------------

Automatic Format Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~

The loader automatically detects file format from extension:

.. code-block:: python

    # Format automatically detected
    loader = DataLoader('data.csv')    # Detected as CSV
    loader = DataLoader('data.edf')    # Detected as EDF
    loader = DataLoader('data.json')   # Detected as JSON

    # Or specify explicitly
    loader = DataLoader('data.txt', format='csv')
    loader = DataLoader('data.bin', format=DataFormat.NUMPY)

Data Validation
~~~~~~~~~~~~~~~

Built-in validation checks for data quality:

.. code-block:: python

    # Enable validation (default)
    loader = DataLoader('data.csv', validate=True)
    data = loader.load()  # Warns about NaN, Inf, missing values

    # Disable validation
    loader = DataLoader('data.csv', validate=False)
    data = loader.load()  # No warnings

Metadata Extraction
~~~~~~~~~~~~~~~~~~~

Automatic extraction of signal metadata:

.. code-block:: python

    loader = DataLoader('data.csv')
    data = loader.load(time_column='time')

    # Get full info
    info = loader.get_info()
    print(f"Format: {info['format']}")
    print(f"Sampling rate: {info['sampling_rate']}")
    print(f"Metadata: {info['metadata']}")

    # Access specific metadata
    print(f"Columns: {loader.metadata['columns']}")
    print(f"Shape: {loader.metadata['shape']}")
    print(f"Samples: {loader.metadata['n_samples']}")

Sampling Rate Extraction
~~~~~~~~~~~~~~~~~~~~~~~~

Automatic sampling rate calculation from time column:

.. code-block:: python

    loader = DataLoader('data.csv')
    data = loader.load(time_column='time')

    # Computed from time intervals
    fs = loader.metadata['computed_sampling_rate']
    print(f"Detected sampling rate: {fs} Hz")

    # Or specify explicitly
    loader = DataLoader('data.csv', sampling_rate=250.0)

Loading from Arrays and DataFrames
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load data from in-memory objects:

.. code-block:: python

    import numpy as np
    import pandas as pd

    # From NumPy array
    signal = np.random.randn(1000)
    loader = DataLoader()
    df = loader.load_from_array(signal, sampling_rate=250.0, signal_type='ecg')

    # From DataFrame
    df_input = pd.DataFrame({'ecg': signal})
    loader = DataLoader()
    df_output = loader.load_from_dataframe(df_input, sampling_rate=250.0)

Chunk-Based Loading
~~~~~~~~~~~~~~~~~~~

Memory-efficient loading for large files:

.. code-block:: python

    # Load in chunks
    loader = DataLoader('large_file.csv')
    data = loader.load(chunk_size=10000)

    # All chunks are automatically concatenated

Multi-Channel Loading
~~~~~~~~~~~~~~~~~~~~~

Load and manage multiple signal channels:

.. code-block:: python

    # EDF multi-channel
    loader = DataLoader('recording.edf')
    data = loader.load()  # Returns dict of channels

    for channel_name, signal in data.items():
        print(f"{channel_name}: {len(signal)} samples")

    # CSV multi-channel
    loader = DataLoader('multi_channel.csv')
    data = loader.load(columns=['ECG', 'PPG', 'RESP'])

Data Export
~~~~~~~~~~~

Export data to various formats:

.. code-block:: python

    import pandas as pd

    # Load data
    loader = DataLoader('input.csv')
    data = loader.load()

    # Export to different formats
    loader.export(data, 'output.csv')
    loader.export(data, 'output.json')
    loader.export(data, 'output.xlsx')
    loader.export(data, 'output.parquet')
    loader.export(data, 'output.pkl')

    # Specify format explicitly
    loader.export(data, 'output.txt', format='csv')

Signal Types
~~~~~~~~~~~~

Specify physiological signal types:

.. code-block:: python

    from vitalDSP.utils.data_processing.data_loader import SignalType

    # Using string
    loader = DataLoader('ecg.csv', signal_type='ecg')

    # Using enum
    loader = DataLoader('ppg.csv', signal_type=SignalType.PPG)

    # Available types
    # - ECG: Electrocardiogram
    # - PPG: Photoplethysmogram
    # - EEG: Electroencephalogram
    # - RESP: Respiratory
    # - BP: Blood Pressure
    # - TEMP: Temperature
    # - SPO2: Blood Oxygen Saturation
    # - GENERIC: General signal

Stream Loading
--------------

For real-time data acquisition:

.. code-block:: python

    from vitalDSP.utils.data_processing.data_loader import StreamDataLoader

    # Serial port streaming
    loader = StreamDataLoader(
        source_type='serial',
        port='/dev/ttyUSB0',
        baudrate=9600,
        buffer_size=1000,
        sampling_rate=250.0
    )

    # Stream data
    for chunk in loader.stream(max_samples=10000):
        process_chunk(chunk)

    # Network streaming
    loader = StreamDataLoader(
        source_type='network',
        host='localhost',
        port=5000,
        buffer_size=1000
    )

    # With callback
    def on_data(chunk):
        print(f"Received {len(chunk)} samples")

    for chunk in loader.stream(callback=on_data):
        process_chunk(chunk)

Static Methods
--------------

List Supported Formats
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    formats = DataLoader.list_supported_formats()
    print("Supported formats:", formats)
    # Output: ['csv', 'tsv', 'excel', 'json', 'hdf5', 'edf', 'wfdb', ...]

Get Format Requirements
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Get requirements for specific format
    req = DataLoader.get_format_requirements('edf')
    print(f"Packages: {req['packages']}")
    print(f"Extensions: {req['extensions']}")
    print(f"Description: {req['description']}")

    # Output:
    # Packages: ['pyedflib']
    # Extensions: ['.edf']
    # Description: European Data Format (medical)

Complete Examples
-----------------

Example 1: ECG Analysis Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from vitalDSP.utils.data_processing.data_loader import DataLoader
    from vitalDSP.preprocess.preprocess_operations import preprocess_signal
    from vitalDSP.utils.signal_processing.peak_detection import PeakDetection

    # Load ECG data
    loader = DataLoader('ecg_recording.csv', signal_type='ecg')
    df = loader.load(time_column='time')

    # Get signal info
    info = loader.get_info()
    print(f"Loaded {info['metadata']['n_samples']} ECG samples")
    print(f"Sampling rate: {loader.sampling_rate} Hz")

    # Extract ECG signal
    ecg = df['ecg'].values

    # Preprocess
    ecg_filtered = preprocess_signal(ecg, loader.sampling_rate)

    # Detect R-peaks
    detector = PeakDetection()
    peaks = detector.detect_peaks(ecg_filtered, loader.sampling_rate)

    print(f"Detected {len(peaks)} heartbeats")

Example 2: Multi-Channel PPG Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from vitalDSP.utils.data_processing.data_loader import load_multi_channel

    # Load multi-channel PPG data
    channels = load_multi_channel(
        'ppg_multi.csv',
        channels=['PPG_red', 'PPG_infrared', 'PPG_green']
    )

    # Process each channel
    for channel_name, signal in channels.items():
        print(f"Processing {channel_name}")
        print(f"  Length: {len(signal)}")
        print(f"  Mean: {signal.mean():.2f}")
        print(f"  Std: {signal.std():.2f}")

Example 3: EDF File Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from vitalDSP.utils.data_processing.data_loader import DataLoader

    # Load EDF file
    loader = DataLoader('sleep_study.edf')
    data = loader.load()

    # View available channels
    print("Available channels:", loader.metadata['channel_labels'])
    print("Recording duration:", loader.metadata['duration'], "seconds")
    print("Start time:", loader.metadata['start_datetime'])

    # Access specific channels
    ecg = data['ECG']
    resp = data['RESP']

    # Get sampling rates for each channel
    ecg_fs = loader.metadata['ECG_sampling_rate']
    resp_fs = loader.metadata['RESP_sampling_rate']

    print(f"ECG: {len(ecg)} samples at {ecg_fs} Hz")
    print(f"RESP: {len(resp)} samples at {resp_fs} Hz")

Example 4: PhysioNet Database
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from vitalDSP.utils.data_processing.data_loader import DataLoader

    # Load PhysioNet WFDB record
    loader = DataLoader('physionet/mitdb/100.dat')
    data = loader.load()

    # View record info
    print("Channels:", loader.metadata['channel_names'])
    print("Sampling rate:", loader.sampling_rate, "Hz")
    print("Duration:", loader.metadata['duration'], "seconds")

    # Get annotations
    if 'annotations' in loader.metadata:
        ann = loader.metadata['annotations']
        print(f"Found {len(ann['sample'])} annotations")
        print(f"Annotation types: {set(ann['symbol'])}")

Example 5: Batch Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pathlib import Path
    from vitalDSP.utils.data_processing.data_loader import DataLoader

    # Process multiple files
    data_dir = Path('signal_data')
    results = []

    for file_path in data_dir.glob('*.csv'):
        print(f"Processing {file_path.name}")

        loader = DataLoader(file_path, sampling_rate=250.0)
        data = loader.load()

        # Process signal
        signal = data['signal'].values
        mean_value = signal.mean()

        results.append({
            'filename': file_path.name,
            'samples': len(signal),
            'mean': mean_value
        })

    # Save results
    import pandas as pd
    results_df = pd.DataFrame(results)
    results_df.to_csv('batch_results.csv', index=False)

Example 6: Format Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from vitalDSP.utils.data_processing.data_loader import DataLoader

    # Load from one format
    loader = DataLoader('data.mat')
    data = loader.load()

    # Convert to DataFrame
    import pandas as pd
    if isinstance(data, dict):
        df = pd.DataFrame(data)
    else:
        df = data

    # Export to multiple formats
    loader.export(df, 'output.csv')
    loader.export(df, 'output.json')
    loader.export(df, 'output.parquet')
    loader.export(df, 'output.xlsx')

    print("Conversion complete!")

Example 7: Real-time Data Acquisition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from vitalDSP.utils.data_processing.data_loader import StreamDataLoader
    import numpy as np

    # Setup streaming
    stream = StreamDataLoader(
        source_type='serial',
        port='/dev/ttyUSB0',
        baudrate=115200,
        buffer_size=250,  # 1 second at 250 Hz
        sampling_rate=250.0
    )

    # Collect and process data
    all_data = []

    def process_chunk(chunk):
        # Real-time processing
        mean_val = np.mean(chunk)
        print(f"Chunk mean: {mean_val:.2f}")
        all_data.append(chunk)

    # Stream for 10 seconds
    max_samples = 250 * 10  # 10 seconds at 250 Hz

    for chunk in stream.stream(callback=process_chunk, max_samples=max_samples):
        # Additional processing if needed
        pass

    # Combine all chunks
    full_signal = np.concatenate(all_data)
    print(f"Collected {len(full_signal)} total samples")

    # Save collected data
    loader = DataLoader()
    loader.export(full_signal, 'collected_data.csv')

Error Handling
--------------

Proper error handling for robust applications:

.. code-block:: python

    from vitalDSP.utils.data_processing.data_loader import DataLoader
    import warnings

    try:
        loader = DataLoader('data.csv')
        data = loader.load()

    except FileNotFoundError:
        print("Error: File not found")

    except ValueError as e:
        print(f"Error loading data: {e}")

    except ImportError as e:
        print(f"Missing required package: {e}")

    # Handle warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress warnings
        loader = DataLoader('data_with_nan.csv', validate=True)
        data = loader.load()

Best Practices
--------------

1. **Always specify sampling rate** when known:

   .. code-block:: python

       loader = DataLoader('ecg.csv', sampling_rate=250.0)

2. **Use validation for quality checks**:

   .. code-block:: python

       loader = DataLoader('data.csv', validate=True)

3. **Extract and store metadata**:

   .. code-block:: python

       info = loader.get_info()
       # Save info for reproducibility

4. **Use appropriate data types**:

   .. code-block:: python

       # For single channel
       signal = data['signal'].values  # NumPy array

       # For multi-channel
       channels = {name: data[name].values for name in data.columns}

5. **Handle large files efficiently**:

   .. code-block:: python

       data = loader.load(chunk_size=10000)

6. **Check format requirements before loading**:

   .. code-block:: python

       req = DataLoader.get_format_requirements('edf')
       print(f"Required packages: {req['packages']}")

7. **Export with metadata**:

   .. code-block:: python

       # Save metadata separately
       import json
       with open('metadata.json', 'w') as f:
           json.dump(loader.get_info(), f, indent=2)

Installation Requirements
-------------------------

Core requirements (always needed):

.. code-block:: bash

    pip install numpy pandas

Optional format-specific requirements:

.. code-block:: bash

    # Excel files
    pip install openpyxl

    # HDF5 files
    pip install h5py tables

    # EDF files (medical)
    pip install pyedflib

    # WFDB files (PhysioNet)
    pip install wfdb

    # MATLAB files
    pip install scipy

    # Parquet files
    pip install pyarrow

    # Streaming from serial ports
    pip install pyserial

    # All optional dependencies
    pip install openpyxl h5py tables pyedflib wfdb scipy pyarrow pyserial

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Issue**: "Format not supported"

**Solution**: Check if the file extension is correct and supported. Use explicit format specification:

.. code-block:: python

    loader = DataLoader('myfile.dat', format='csv')

**Issue**: "Missing required package"

**Solution**: Install the required package for your format:

.. code-block:: python

    req = DataLoader.get_format_requirements('edf')
    print(f"Install: pip install {' '.join(req['packages'])}")

**Issue**: Data contains NaN or Inf values

**Solution**: Enable validation to see warnings, then clean data:

.. code-block:: python

    loader = DataLoader('data.csv', validate=True)
    data = loader.load()
    data = data.fillna(method='ffill')  # Forward fill NaN

**Issue**: Incorrect sampling rate detection

**Solution**: Specify sampling rate explicitly:

.. code-block:: python

    loader = DataLoader('data.csv', sampling_rate=250.0)

API Reference
-------------

For detailed API documentation, see:

* :class:`vitalDSP.utils.data_processing.data_loader.DataLoader`
* :class:`vitalDSP.utils.data_processing.data_loader.StreamDataLoader`
* :class:`vitalDSP.utils.data_processing.data_loader.DataFormat`
* :class:`vitalDSP.utils.data_processing.data_loader.SignalType`

See Also
--------

* :doc:`preprocessing_guide` - Signal preprocessing techniques
* :doc:`feature_extraction` - Feature extraction methods
* :doc:`quality_assessment` - Signal quality assessment

References
----------

* EDF Format: https://www.edfplus.info/
* WFDB/PhysioNet: https://physionet.org/
* Apache Parquet: https://parquet.apache.org/

Contributing
------------

To add support for new formats, see the developer documentation.

Report issues or request features at: https://github.com/Oucru-Innovations/vital-DSP

License
-------

vitalDSP is licensed under the MIT License.
