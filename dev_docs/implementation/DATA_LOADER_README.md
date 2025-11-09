# vitalDSP Data Loader Module

A comprehensive, production-ready data loading system for physiological signal processing in vitalDSP.

## 🎯 Overview

The Data Loader module provides a unified interface for loading physiological signals from **10+ file formats** with automatic format detection, data validation, and metadata extraction.

### Key Features

- ✅ **Universal Format Support**: CSV, TSV, Excel, JSON, HDF5, EDF, WFDB, NumPy, MATLAB, Pickle, Parquet
- ✅ **Automatic Format Detection**: Intelligently detects format from file extension
- ✅ **Multi-Channel Support**: Load and manage multiple signal channels simultaneously
- ✅ **Data Validation**: Built-in quality checks for NaN, Inf, and missing values
- ✅ **Metadata Extraction**: Automatic sampling rate detection and signal properties
- ✅ **Stream Loading**: Support for real-time data acquisition
- ✅ **Memory Efficient**: Chunk-based loading for large datasets
- ✅ **Export Capabilities**: Convert between formats seamlessly

## 🚀 Quick Start

### Basic Usage

```python
from vitalDSP.utils.data_loader import DataLoader

# Load any supported format
loader = DataLoader('ecg_data.csv')
data = loader.load()

print(f"Loaded {len(data)} samples at {loader.sampling_rate} Hz")
```

### Convenience Functions

```python
from vitalDSP.utils.data_loader import load_signal, load_multi_channel

# Quick single-channel loading
data = load_signal('ecg.csv', sampling_rate=250)

# Quick multi-channel loading
channels = load_multi_channel('multi.edf', channels=['ECG', 'PPG'])
```

## 📦 Installation

### Core Requirements

```bash
pip install numpy pandas
```

### Optional Format Support

```bash
# Medical formats (EDF, WFDB)
pip install pyedflib wfdb

# Excel support
pip install openpyxl

# HDF5 support
pip install h5py tables

# MATLAB support
pip install scipy

# Parquet support
pip install pyarrow

# Real-time streaming
pip install pyserial

# Install all optional dependencies
pip install pyedflib wfdb openpyxl h5py tables scipy pyarrow pyserial
```

## 📚 Supported Formats

| Format | Extension | Description | Required Package |
|--------|-----------|-------------|------------------|
| CSV | `.csv`, `.txt` | Comma-separated values | pandas |
| TSV | `.tsv` | Tab-separated values | pandas |
| Excel | `.xlsx`, `.xls` | Microsoft Excel | openpyxl |
| JSON | `.json` | JavaScript Object Notation | (built-in) |
| HDF5 | `.h5`, `.hdf5` | Hierarchical Data Format | h5py, tables |
| EDF | `.edf` | European Data Format (medical) | pyedflib |
| WFDB | `.dat`, `.hea` | PhysioNet WFDB format | wfdb |
| NumPy | `.npy`, `.npz` | NumPy array format | numpy |
| MATLAB | `.mat` | MATLAB data files | scipy |
| Pickle | `.pkl`, `.pickle` | Python pickle | (built-in) |
| Parquet | `.parquet` | Apache Parquet columnar | pyarrow |

## 💡 Examples

### Example 1: Load ECG from CSV

```python
from vitalDSP.utils.data_loader import DataLoader

# Load with metadata
loader = DataLoader('ecg_recording.csv', sampling_rate=250.0, signal_type='ecg')
df = loader.load(time_column='time')

# Access data
ecg_signal = df['ecg'].values
time = df['time'].values

# Get metadata
info = loader.get_info()
print(f"Duration: {info['metadata']['n_samples'] / loader.sampling_rate:.1f} seconds")
```

### Example 2: Load Multi-Channel EDF (Medical Format)

```python
# Load EDF file (returns dictionary of channels)
loader = DataLoader('sleep_study.edf')
data = loader.load()

# Access specific channels
ecg = data['ECG']
resp = data['RESP']

# View metadata
print(f"Channels: {loader.metadata['channel_labels']}")
print(f"Duration: {loader.metadata['duration']} seconds")
print(f"Start time: {loader.metadata['start_datetime']}")
```

### Example 3: Load PhysioNet WFDB Data

```python
# Load WFDB record from PhysioNet
loader = DataLoader('physionet/mitdb/100.dat')
data = loader.load(channels=['MLII', 'V5'])

# Access annotations
if 'annotations' in loader.metadata:
    ann = loader.metadata['annotations']
    print(f"Found {len(ann['sample'])} annotations")
```

### Example 4: Multi-Channel CSV Processing

```python
from vitalDSP.utils.data_loader import load_multi_channel

# Load multiple channels as dictionary
channels = load_multi_channel('multi_channel.csv',
                              channels=['ECG', 'PPG', 'RESP'])

# Process each channel
for name, signal in channels.items():
    print(f"{name}: {len(signal)} samples")
    print(f"  Mean: {signal.mean():.3f}")
    print(f"  Std: {signal.std():.3f}")
```

### Example 5: Data Validation

```python
# Enable validation for quality checks
loader = DataLoader('data.csv', validate=True)
data = loader.load()  # Warns about NaN, Inf, missing values

# Manual validation
if loader.metadata.get('has_nan'):
    print("Warning: Data contains NaN values")
```

### Example 6: Format Conversion

```python
# Load from one format
loader = DataLoader('data.mat')
data = loader.load()

# Export to different formats
loader.export(data, 'output.csv')
loader.export(data, 'output.json')
loader.export(data, 'output.parquet')
loader.export(data, 'output.xlsx')
```

### Example 7: Stream Real-Time Data

```python
from vitalDSP.utils.data_loader import StreamDataLoader

# Setup serial port streaming
stream = StreamDataLoader(
    source_type='serial',
    port='/dev/ttyUSB0',
    baudrate=115200,
    buffer_size=250,  # 1 second at 250 Hz
    sampling_rate=250.0
)

# Collect data
all_data = []
for chunk in stream.stream(max_samples=2500):  # 10 seconds
    all_data.append(chunk)
    print(f"Received {len(chunk)} samples")

# Process collected data
full_signal = np.concatenate(all_data)
```

### Example 8: Load from NumPy Array

```python
import numpy as np

# Create or load array
signal = np.random.randn(1000)

# Load into vitalDSP format
loader = DataLoader()
df = loader.load_from_array(signal, sampling_rate=250.0, signal_type='ecg')

print(f"Loaded {len(df)} samples")
```

### Example 9: Batch Processing

```python
from pathlib import Path

# Process all CSV files in directory
data_dir = Path('patient_data')
results = []

for file_path in data_dir.glob('*.csv'):
    loader = DataLoader(file_path, sampling_rate=250.0)
    data = loader.load()

    signal = data['signal'].values
    results.append({
        'file': file_path.name,
        'samples': len(signal),
        'mean': signal.mean(),
        'std': signal.std()
    })

# Save results
import pandas as pd
pd.DataFrame(results).to_csv('batch_results.csv', index=False)
```

### Example 10: Automatic Sampling Rate Detection

```python
# Load without specifying sampling rate
loader = DataLoader('data.csv')
data = loader.load(time_column='time')

# Sampling rate automatically computed from time column
detected_fs = loader.metadata['computed_sampling_rate']
print(f"Detected sampling rate: {detected_fs} Hz")
```

## 🔧 Advanced Features

### Format Detection

```python
# Automatic detection from extension
loader = DataLoader('data.csv')    # Detected as CSV
loader = DataLoader('data.edf')    # Detected as EDF

# Or specify explicitly
loader = DataLoader('data.txt', format='csv')
```

### Chunk-Based Loading

```python
# Memory-efficient loading for large files
loader = DataLoader('large_file.csv')
data = loader.load(chunk_size=10000)  # Load in 10K sample chunks
```

### Metadata Extraction

```python
loader = DataLoader('data.csv')
data = loader.load(time_column='time')

# Get comprehensive info
info = loader.get_info()
print(f"Format: {info['format']}")
print(f"Sampling rate: {info['sampling_rate']}")
print(f"Samples: {info['metadata']['n_samples']}")
print(f"Shape: {info['metadata']['shape']}")
```

### Signal Type Specification

```python
from vitalDSP.utils.data_loader import SignalType

# Specify signal type
loader = DataLoader('ecg.csv', signal_type=SignalType.ECG)
loader = DataLoader('ppg.csv', signal_type='ppg')

# Available types: ECG, PPG, EEG, RESP, BP, TEMP, SPO2, GENERIC
```

## 🛠️ API Reference

### DataLoader Class

```python
DataLoader(
    file_path: Optional[str] = None,
    format: Optional[str] = None,
    sampling_rate: Optional[float] = None,
    signal_type: Optional[str] = None,
    validate: bool = True,
    **kwargs
)
```

**Methods:**
- `load()`: Load data from file
- `load_from_array()`: Load from NumPy array
- `load_from_dataframe()`: Load from pandas DataFrame
- `export()`: Export data to file
- `get_info()`: Get comprehensive data information

**Static Methods:**
- `list_supported_formats()`: List all supported formats
- `get_format_requirements()`: Get requirements for specific format

### StreamDataLoader Class

```python
StreamDataLoader(
    source_type: str,
    buffer_size: int = 1000,
    sampling_rate: Optional[float] = None,
    **kwargs
)
```

**Methods:**
- `stream()`: Stream data from source
- `stop()`: Stop streaming

### Convenience Functions

```python
load_signal(file_path, **kwargs)
load_multi_channel(file_path, channels=None, **kwargs)
```

## 📖 Documentation

- **Full Documentation**: [docs/source/data_loader_guide.rst](docs/source/data_loader_guide.rst)
- **Tutorial Notebook**: [docs/source/notebooks/data_loader_tutorial.ipynb](docs/source/notebooks/data_loader_tutorial.ipynb)
- **Test Suite**: [tests/vitalDSP/utils/test_data_loader.py](tests/vitalDSP/utils/test_data_loader.py)

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all data loader tests
pytest tests/vitalDSP/utils/test_data_loader.py -v

# Run specific test class
pytest tests/vitalDSP/utils/test_data_loader.py::TestCSVLoading -v

# Run with coverage
pytest tests/vitalDSP/utils/test_data_loader.py --cov=vitalDSP.utils.data_loader
```

## 🐛 Troubleshooting

### Issue: "Format not supported"

**Solution**: Check file extension and install required packages:

```python
# Check requirements
req = DataLoader.get_format_requirements('edf')
print(f"Install: pip install {' '.join(req['packages'])}")
```

### Issue: "Missing required package"

**Solution**: Install format-specific dependencies:

```bash
# For EDF
pip install pyedflib

# For WFDB
pip install wfdb

# For all formats
pip install pyedflib wfdb openpyxl h5py tables scipy pyarrow
```

### Issue: Data contains NaN values

**Solution**: Enable validation and clean data:

```python
loader = DataLoader('data.csv', validate=True)
data = loader.load()
data = data.fillna(method='ffill')  # or other cleaning method
```

## 📊 Performance Considerations

1. **Large Files**: Use chunk-based loading
   ```python
   data = loader.load(chunk_size=10000)
   ```

2. **Memory Efficiency**: Load only required columns
   ```python
   data = loader.load(columns=['time', 'ecg'])
   ```

3. **Format Selection**: Parquet is most efficient for large datasets
   ```python
   loader.export(data, 'output.parquet')  # Faster than CSV
   ```

## 🤝 Contributing

To add support for new formats:

1. Add format to `DataFormat` enum
2. Implement `_load_<format>()` method in `DataLoader`
3. Update format detection in `_detect_format()`
4. Add tests in `test_data_loader.py`
5. Update documentation

## 📄 License

MIT License - see LICENSE file for details

## 🔗 Links

- **Repository**: https://github.com/Oucru-Innovations/vital-DSP
- **Documentation**: https://vital-dsp.readthedocs.io
- **Issues**: https://github.com/Oucru-Innovations/vital-DSP/issues

## 📮 Support

- Report bugs or request features via [GitHub Issues](https://github.com/Oucru-Innovations/vital-DSP/issues)
- For questions, see the [Documentation](docs/source/data_loader_guide.rst)

## 🎓 Citation

If you use vitalDSP Data Loader in your research, please cite:

```bibtex
@software{vitaldsp2025,
  title={vitalDSP: Comprehensive Digital Signal Processing for Physiological Signals},
  author={vitalDSP Team},
  year={2025},
  url={https://github.com/Oucru-Innovations/vital-DSP}
}
```

## 🙏 Acknowledgments

- EDF format support powered by [pyedflib](https://github.com/holgern/pyedflib)
- WFDB support powered by [wfdb-python](https://github.com/MIT-LCP/wfdb-python)
- PhysioNet for open physiological signal datasets

---

**Made with ❤️ by the vitalDSP Team**
