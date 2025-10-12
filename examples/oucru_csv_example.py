"""
OUCRU CSV Format Loading Example

Demonstrates how to load and use OUCRU's special CSV format where each row
represents 1 second of data with signal values stored as array strings.

Example OUCRU CSV format:
    timestamp,signal,sampling_rate
    2024-01-01 00:00:00,"[1.0, 1.1, 1.2, 1.3, 1.4]",5
    2024-01-01 00:00:01,"[1.5, 1.6, 1.7, 1.8, 1.9]",5

Author: vitalDSP
Date: 2025-01-11
"""

import numpy as np
import tempfile
import os
from pathlib import Path

# Add src to path for development
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from vitalDSP.utils.data_loader import load_oucru_csv, DataLoader, DataFormat


def create_sample_oucru_file():
    """Create a sample OUCRU CSV file for demonstration."""
    # Create temporary file
    fd, filepath = tempfile.mkstemp(suffix='_oucru.csv', text=True)

    # Sample ECG data (simulated)
    content = """timestamp,ecg_values,sampling_rate
2024-01-01 10:00:00,"[0.12, 0.13, 0.14, 0.15, 0.16]",5
2024-01-01 10:00:01,"[0.17, 0.18, 0.19, 0.20, 0.21]",5
2024-01-01 10:00:02,"[0.22, 0.23, 0.24, 0.25, 0.26]",5
2024-01-01 10:00:03,"[0.27, 0.28, 0.29, 0.30, 0.31]",5
2024-01-01 10:00:04,"[0.32, 0.33, 0.34, 0.35, 0.36]",5
"""

    with os.fdopen(fd, 'w') as f:
        f.write(content)

    return filepath


def example_1_basic_loading():
    """Example 1: Basic OUCRU CSV loading."""
    print("=" * 70)
    print("Example 1: Basic OUCRU CSV Loading")
    print("=" * 70)

    # Create sample file
    filepath = create_sample_oucru_file()

    try:
        # Load using convenience function
        signal, metadata = load_oucru_csv(
            filepath,
            time_column='timestamp',
            signal_column='ecg_values',
            sampling_rate_column='sampling_rate'
        )

        # Display results
        print(f"\n✓ Loaded OUCRU CSV file successfully!")
        print(f"\n📊 Signal Information:")
        print(f"   - Number of samples: {len(signal)}")
        print(f"   - Sampling rate: {metadata['sampling_rate']} Hz")
        print(f"   - Duration: {metadata['duration_seconds']:.1f} seconds")
        print(f"   - Number of rows: {metadata['n_rows']}")
        print(f"   - Samples per row: {metadata['samples_per_row']}")

        print(f"\n📈 Signal Data (first 10 samples):")
        print(f"   {signal[:10]}")

        print(f"\n⏰ Time Range:")
        print(f"   - Start: {metadata['start_time']}")
        print(f"   - End: {metadata['end_time']}")

    finally:
        os.unlink(filepath)

    print("\n")


def example_2_dataloader_class():
    """Example 2: Using DataLoader class directly."""
    print("=" * 70)
    print("Example 2: Using DataLoader Class")
    print("=" * 70)

    filepath = create_sample_oucru_file()

    try:
        # Create DataLoader with OUCRU format
        loader = DataLoader(
            filepath,
            format=DataFormat.OUCRU_CSV,
            sampling_rate=5
        )

        # Load with timestamp interpolation
        data = loader.load(
            time_column='timestamp',
            signal_column='ecg_values',
            interpolate_time=True
        )

        print(f"\n✓ Loaded using DataLoader class!")
        print(f"\n📋 DataFrame Structure:")
        print(data.head(10))

        print(f"\n📊 DataFrame Info:")
        print(f"   - Shape: {data.shape}")
        print(f"   - Columns: {list(data.columns)}")
        print(f"   - Memory usage: {data.memory_usage(deep=True).sum() / 1024:.2f} KB")

    finally:
        os.unlink(filepath)

    print("\n")


def example_3_auto_sampling_rate():
    """Example 3: Automatic sampling rate detection."""
    print("=" * 70)
    print("Example 3: Automatic Sampling Rate Detection")
    print("=" * 70)

    # Create file without sampling_rate column
    fd, filepath = tempfile.mkstemp(suffix='_oucru_no_sr.csv', text=True)
    content = """timestamp,ppg_signal
2024-01-01 12:00:00,"[1.0, 1.1, 1.2, 1.3]"
2024-01-01 12:00:01,"[1.4, 1.5, 1.6, 1.7]"
2024-01-01 12:00:02,"[1.8, 1.9, 2.0, 2.1]"
"""

    with os.fdopen(fd, 'w') as f:
        f.write(content)

    try:
        # Load without specifying sampling rate
        signal, metadata = load_oucru_csv(
            filepath,
            time_column='timestamp',
            signal_column='ppg_signal',
            sampling_rate_column=None  # No SR column
        )

        print(f"\n✓ Sampling rate auto-detected from array length!")
        print(f"\n📊 Detected Parameters:")
        print(f"   - Sampling rate: {metadata['sampling_rate']} Hz (auto-detected)")
        print(f"   - Array length per row: {metadata['samples_per_row']}")
        print(f"   - Total samples: {len(signal)}")
        print(f"\n💡 Note: Sampling rate was inferred from array length")
        print(f"          (4 samples per row = 4 Hz)")

    finally:
        os.unlink(filepath)

    print("\n")


def example_4_no_interpolation():
    """Example 4: Loading without timestamp interpolation."""
    print("=" * 70)
    print("Example 4: Loading Without Timestamp Interpolation")
    print("=" * 70)

    filepath = create_sample_oucru_file()

    try:
        # Create loader
        loader = DataLoader(filepath, format=DataFormat.OUCRU_CSV)

        # Load without interpolation (faster)
        data_no_interp = loader.load(
            time_column='timestamp',
            signal_column='ecg_values',
            interpolate_time=False
        )

        # Load with interpolation for comparison
        loader2 = DataLoader(filepath, format=DataFormat.OUCRU_CSV)
        data_with_interp = loader2.load(
            time_column='timestamp',
            signal_column='ecg_values',
            interpolate_time=True
        )

        print(f"\n✓ Loaded with and without interpolation!")
        print(f"\n📊 Comparison:")
        print(f"\nWithout Interpolation:")
        print(f"   - Columns: {list(data_no_interp.columns)}")
        print(data_no_interp.head())

        print(f"\nWith Interpolation:")
        print(f"   - Columns: {list(data_with_interp.columns)}")
        print(data_with_interp.head())

        print(f"\n💡 Use interpolate_time=False for:")
        print(f"   - Faster loading")
        print(f"   - When sample-level timing not needed")
        print(f"   - Large files")

    finally:
        os.unlink(filepath)

    print("\n")


def example_5_integration():
    """Example 5: Integration with vitalDSP modules."""
    print("=" * 70)
    print("Example 5: Integration with vitalDSP Modules")
    print("=" * 70)

    filepath = create_sample_oucru_file()

    try:
        # Load OUCRU data
        signal, metadata = load_oucru_csv(filepath, sampling_rate=5)

        print(f"\n✓ Loaded OUCRU data: {len(signal)} samples at {metadata['sampling_rate']} Hz")

        # Basic statistics
        print(f"\n📊 Signal Statistics:")
        print(f"   - Mean: {np.mean(signal):.4f}")
        print(f"   - Std: {np.std(signal):.4f}")
        print(f"   - Min: {np.min(signal):.4f}")
        print(f"   - Max: {np.max(signal):.4f}")
        print(f"   - Range: {np.ptp(signal):.4f}")

        # Demonstrate integration potential
        print(f"\n🔗 Integration Examples:")
        print(f"   ✓ Can use with SignalFiltering.butter_lowpass_filter()")
        print(f"   ✓ Can use with PeakDetection.find_peaks()")
        print(f"   ✓ Can use with TimeDomainFeatures.extract_all_features()")
        print(f"   ✓ Can use with any vitalDSP analysis function")

        print(f"\n💡 The loaded signal array works with all vitalDSP modules!")

    finally:
        os.unlink(filepath)

    print("\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("OUCRU CSV Format Loading Examples")
    print("vitalDSP Data Loader Module")
    print("=" * 70 + "\n")

    # Run examples
    example_1_basic_loading()
    example_2_dataloader_class()
    example_3_auto_sampling_rate()
    example_4_no_interpolation()
    example_5_integration()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print("\n✅ All examples completed successfully!")
    print("\n📚 Key Takeaways:")
    print("   1. Use load_oucru_csv() for quick loading")
    print("   2. Use DataLoader for more control")
    print("   3. Sampling rate can be auto-detected")
    print("   4. Timestamp interpolation is optional")
    print("   5. Integrates seamlessly with vitalDSP")
    print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    main()
