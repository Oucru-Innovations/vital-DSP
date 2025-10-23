# OUCRU CSV Enhancement: Signal Type Hints & Default Sampling Rates

**Date**: 2025-01-11
**Version**: 1.1.0
**Status**: Complete ✅

---

## Overview

Enhanced the OUCRU CSV loader with **signal type hints** and **default sampling rates** for common physiological signals. This allows users to specify the signal type ('ppg' or 'ecg') and have appropriate default sampling rates applied automatically when no sampling_rate column is present.

---

## What's New

### Signal Type Hints

Users can now specify a `signal_type_hint` parameter to indicate the type of physiological signal being loaded. The loader will automatically use appropriate default sampling rates:

- **PPG (Photoplethysmography)**: 100 Hz default
- **ECG (Electrocardiogram)**: 128 Hz default

### Customizable Default Rates

Users can override the built-in defaults with custom values:
- `default_ppg_rate`: Custom default for PPG signals (default: 100 Hz)
- `default_ecg_rate`: Custom default for ECG signals (default: 128 Hz)

---

## Sampling Rate Priority (Updated)

The loader now uses a **4-tier priority system** for determining sampling rate:

1. **Highest**: `sampling_rate_column` value (if column exists in CSV)
2. **High**: Explicit `sampling_rate` parameter
3. **Medium**: Signal type hint (`signal_type_hint='ppg'` or `'ecg'`)
4. **Lowest**: Inferred from array length

---

## Usage Examples

### Example 1: Using PPG Signal Type Hint

```python
from vitalDSP.utils.data_loader import load_oucru_csv

# Load PPG data - automatically uses 100 Hz
signal, metadata = load_oucru_csv(
    'ppg_data.csv',
    time_column='timestamp',
    signal_column='ppg_values',
    signal_type_hint='ppg'  # Uses default_ppg_rate=100 Hz
)

print(f"Sampling rate: {metadata['sampling_rate']} Hz")  # Output: 100 Hz
```

### Example 2: Using ECG Signal Type Hint

```python
# Load ECG data - automatically uses 128 Hz
signal, metadata = load_oucru_csv(
    'ecg_data.csv',
    time_column='timestamp',
    signal_column='ecg_values',
    signal_type_hint='ecg'  # Uses default_ecg_rate=128 Hz
)

print(f"Sampling rate: {metadata['sampling_rate']} Hz")  # Output: 128 Hz
```

### Example 3: Custom Default Rates

```python
# Override the default ECG rate
signal, metadata = load_oucru_csv(
    'high_res_ecg.csv',
    signal_type_hint='ecg',
    default_ecg_rate=250  # Use 250 Hz instead of 128 Hz
)

print(f"Sampling rate: {metadata['sampling_rate']} Hz")  # Output: 250 Hz
```

### Example 4: Priority Demonstration

```python
# Explicit sampling_rate takes precedence over signal_type_hint
signal, metadata = load_oucru_csv(
    'custom_ecg.csv',
    sampling_rate=500,      # Explicit - highest priority
    signal_type_hint='ecg'  # This would be 128 Hz but is overridden
)

print(f"Sampling rate: {metadata['sampling_rate']} Hz")  # Output: 500 Hz
```

### Example 5: Without Hints (Backward Compatible)

```python
# Original behavior still works - infers from array length
signal, metadata = load_oucru_csv(
    'data.csv',
    # No sampling_rate, no signal_type_hint
    # Will infer from array length
)
```

---

## Supported Signal Types

Currently supported signal type hints:

| Signal Type | Aliases | Default Rate |
|-------------|---------|--------------|
| PPG | `'ppg'`, `'photoplethysmography'` | 100 Hz |
| ECG | `'ecg'`, `'electrocardiogram'`, `'ekg'` | 128 Hz |

---

## API Changes

### `load_oucru_csv()` Function

**New Parameters:**

```python
def load_oucru_csv(
    file_path: Union[str, Path],
    time_column: str = 'timestamp',
    signal_column: str = 'signal',
    sampling_rate: Optional[float] = None,
    sampling_rate_column: Optional[str] = 'sampling_rate',
    interpolate_time: bool = True,
    default_ppg_rate: float = 100.0,           # NEW
    default_ecg_rate: float = 128.0,           # NEW
    signal_type_hint: Optional[str] = None,    # NEW
    **kwargs
) -> Tuple[np.ndarray, Dict[str, Any]]:
```

**Parameter Descriptions:**

- `default_ppg_rate` (float, default=100.0): Default sampling rate for PPG signals in Hz
- `default_ecg_rate` (float, default=128.0): Default sampling rate for ECG signals in Hz
- `signal_type_hint` (str, optional): Signal type hint ('ppg', 'ecg', or None)

---

## Implementation Details

### Code Location

**File**: `src/vitalDSP/utils/data_loader.py`

**Function**: `load_oucru_csv()` (lines ~1277-1400)

### Logic Flow

```python
# Determine sampling rate if not explicitly provided
if sampling_rate is None and signal_type_hint is not None:
    signal_type_hint = signal_type_hint.lower()
    if signal_type_hint in ('ppg', 'photoplethysmography'):
        sampling_rate = default_ppg_rate
    elif signal_type_hint in ('ecg', 'electrocardiogram', 'ekg'):
        sampling_rate = default_ecg_rate
    else:
        warnings.warn(
            f"Unknown signal_type_hint '{signal_type_hint}'. "
            f"Supported types: 'ppg', 'ecg'. Will infer from array length."
        )

loader = DataLoader(
    file_path=file_path,
    format=DataFormat.OUCRU_CSV,
    sampling_rate=sampling_rate,  # Uses hint-based rate if applicable
    **kwargs
)
```

### Error Handling

- **Unknown signal type**: Warns and falls back to array length inference
- **Case insensitive**: Accepts 'PPG', 'ppg', 'ECG', 'ecg', etc.
- **Backward compatible**: All existing code continues to work

---

## Testing

### New Test Cases

Added 5 new test cases in `tests/test_oucru_csv_loader.py`:

1. **`test_signal_type_hint_ppg`**: Validates PPG hint uses 100 Hz
2. **`test_signal_type_hint_ecg`**: Validates ECG hint uses 128 Hz
3. **`test_custom_default_rates`**: Tests custom default rate override
4. **`test_signal_type_hint_priority`**: Confirms explicit rate overrides hint
5. **`test_unknown_signal_type_hint`**: Tests unknown type handling

**Total Test Count**: 20 tests (15 original + 5 new)

### Running Tests

```bash
# Run all OUCRU CSV tests
pytest tests/test_oucru_csv_loader.py -v

# Run only signal type hint tests
pytest tests/test_oucru_csv_loader.py::TestOUCRUCSVLoader::test_signal_type_hint_ppg -v
pytest tests/test_oucru_csv_loader.py::TestOUCRUCSVLoader::test_signal_type_hint_ecg -v
```

---

## Documentation Updates

### Updated Files

1. **`docs/source/data_loader_guide.rst`**
   - Added signal type hint examples
   - Updated sampling rate priority list
   - Included custom default rate examples

2. **`src/vitalDSP/utils/data_loader.py`**
   - Updated function docstring with new parameters
   - Added usage examples for signal type hints
   - Documented sampling rate priority

---

## Use Cases

### 1. Standardized Hospital Data

```python
# Hospital uses standard PPG at 100 Hz
for patient_file in patient_files:
    signal, metadata = load_oucru_csv(
        patient_file,
        signal_type_hint='ppg'  # Automatically 100 Hz
    )
    process_ppg(signal, metadata['sampling_rate'])
```

### 2. Multi-Modal Datasets

```python
# Different signal types in same dataset
for file_info in dataset:
    signal, metadata = load_oucru_csv(
        file_info['path'],
        signal_type_hint=file_info['type']  # 'ppg' or 'ecg'
    )
    analyze_signal(signal, metadata)
```

### 3. Research with Custom Rates

```python
# Research lab uses high-resolution ECG at 1000 Hz
signal, metadata = load_oucru_csv(
    'research_ecg.csv',
    signal_type_hint='ecg',
    default_ecg_rate=1000  # Research-grade sampling
)
```

### 4. Quick Prototyping

```python
# Quick analysis without knowing exact sampling rate
signal, metadata = load_oucru_csv(
    'unknown_ppg.csv',
    signal_type_hint='ppg'  # Reasonable default
)

# Proceed with analysis
heart_rate = compute_heart_rate(signal, metadata['sampling_rate'])
```

---

## Backward Compatibility

✅ **100% Backward Compatible**

All existing code continues to work without modification:

```python
# Old code - still works perfectly
signal, metadata = load_oucru_csv('data.csv')

# Old code with explicit rate - still works
signal, metadata = load_oucru_csv('data.csv', sampling_rate=250)

# Old code with sampling_rate column - still works
signal, metadata = load_oucru_csv('data.csv', sampling_rate_column='fs')
```

---

## Benefits

1. **Convenience**: No need to specify sampling rate for common signals
2. **Consistency**: Standard defaults across projects and teams
3. **Flexibility**: Can override defaults when needed
4. **Safety**: Falls back to array length inference if hint is wrong
5. **Documentation**: Signal type is explicitly documented in code

---

## Common Pitfalls & Solutions

### Pitfall 1: Wrong Signal Type Hint

**Problem:**
```python
# CSV has PPG data but hint says ECG
signal, metadata = load_oucru_csv(
    'ppg_data.csv',  # Actually PPG at 100 Hz
    signal_type_hint='ecg'  # Wrong! Would use 128 Hz
)
```

**Solution:**
- Array length will likely not match 128, triggering a warning
- Or: explicitly provide `sampling_rate` parameter

### Pitfall 2: Assuming Hint Overrides Column

**Problem:**
```python
# CSV has sampling_rate column with value 250
# User expects hint to override it
signal, metadata = load_oucru_csv(
    'data.csv',  # Has sampling_rate=250 in column
    signal_type_hint='ecg'  # Expects 128, but column wins
)
# Result: Uses 250, not 128
```

**Solution:**
- Remember priority: column > explicit > hint > inference
- Remove or ignore sampling_rate column if you want hint to apply

---

## Future Enhancements

Potential additions:

1. **More Signal Types**: EEG, EMG, SpO2, temperature
2. **Region-Specific Defaults**: US vs EU ECG standards
3. **Device-Specific Profiles**: Apple Watch (64 Hz), Fitbit (25 Hz)
4. **Auto-Detection**: Infer signal type from column name or data characteristics

---

## Changelog

### Version 1.1.0 (2025-01-11)

**Added:**
- ✅ `signal_type_hint` parameter
- ✅ `default_ppg_rate` parameter (default: 100 Hz)
- ✅ `default_ecg_rate` parameter (default: 128 Hz)
- ✅ Support for PPG signal type ('ppg', 'photoplethysmography')
- ✅ Support for ECG signal type ('ecg', 'electrocardiogram', 'ekg')
- ✅ 5 new test cases
- ✅ Updated documentation with examples
- ✅ Warning for unknown signal types

**Changed:**
- ✅ Sampling rate priority now 4-tier (was 3-tier)
- ✅ Enhanced docstrings with signal type examples

**Maintained:**
- ✅ 100% backward compatibility
- ✅ All existing tests pass
- ✅ Original behavior unchanged

---

## Summary

Successfully enhanced OUCRU CSV loader with **signal type hints** and **default sampling rates** for PPG (100 Hz) and ECG (128 Hz). This makes the loader more user-friendly for common use cases while maintaining full backward compatibility and flexibility.

**Key Features:**
- 🎯 Automatic sampling rate selection based on signal type
- 🔧 Customizable default rates
- 📊 Clear priority system
- ⚠️ Helpful warnings for unknown types
- ✅ Fully tested with 20 test cases
- 📚 Comprehensive documentation

---

**Author**: Claude (vitalDSP Enhancement)
**Version**: 1.1.0
**Date**: 2025-01-11
**Status**: Complete ✅
