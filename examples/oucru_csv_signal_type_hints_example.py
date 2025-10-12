"""
OUCRU CSV Signal Type Hints Example

Demonstrates the new signal type hint feature for automatic sampling rate detection.

New in v1.1.0:
- signal_type_hint parameter for PPG and ECG
- default_ppg_rate and default_ecg_rate parameters
- Automatic sampling rate based on signal type

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

from vitalDSP.utils.data_loader import load_oucru_csv


def create_sample_ppg_file():
    """Create a sample PPG OUCRU CSV file (100 samples per second)."""
    fd, filepath = tempfile.mkstemp(suffix='_ppg.csv', text=True)

    # Create PPG data with 100 samples per second (100 Hz)
    ppg_array = str(list(np.random.uniform(0.8, 1.2, 100)))  # Typical PPG range

    content = f"""timestamp,ppg_signal
2024-01-01 10:00:00,"{ppg_array}"
2024-01-01 10:00:01,"{ppg_array}"
2024-01-01 10:00:02,"{ppg_array}"
"""

    with os.fdopen(fd, 'w') as f:
        f.write(content)

    return filepath


def create_sample_ecg_file():
    """Create a sample ECG OUCRU CSV file (128 samples per second)."""
    fd, filepath = tempfile.mkstemp(suffix='_ecg.csv', text=True)

    # Create ECG data with 128 samples per second (128 Hz)
    ecg_array = str(list(np.random.uniform(-0.5, 1.5, 128)))  # Typical ECG range

    content = f"""timestamp,ecg_signal
2024-01-01 10:00:00,"{ecg_array}"
2024-01-01 10:00:01,"{ecg_array}"
2024-01-01 10:00:02,"{ecg_array}"
"""

    with os.fdopen(fd, 'w') as f:
        f.write(content)

    return filepath


def example_1_ppg_signal_type_hint():
    """Example 1: Using PPG signal type hint (automatic 100 Hz)."""
    print("=" * 70)
    print("Example 1: PPG Signal Type Hint (Automatic 100 Hz)")
    print("=" * 70)

    filepath = create_sample_ppg_file()

    try:
        # Load PPG data with signal type hint
        # No need to specify sampling_rate or sampling_rate_column!
        signal, metadata = load_oucru_csv(
            filepath,
            time_column='timestamp',
            signal_column='ppg_signal',
            signal_type_hint='ppg',  # Automatically uses 100 Hz
            sampling_rate_column=None  # No column needed
        )

        print(f"\n✓ Loaded PPG data using signal type hint!")
        print(f"\n📊 Results:")
        print(f"   - Signal type hint: 'ppg'")
        print(f"   - Sampling rate: {metadata['sampling_rate']} Hz (auto-detected)")
        print(f"   - Number of samples: {len(signal)}")
        print(f"   - Duration: {metadata['duration_seconds']:.1f} seconds")
        print(f"   - Expected samples: {metadata['n_rows'] * 100}")

        print(f"\n💡 No sampling_rate parameter or column needed!")
        print(f"   The loader automatically used 100 Hz for PPG.")

    finally:
        os.unlink(filepath)

    print("\n")


def example_2_ecg_signal_type_hint():
    """Example 2: Using ECG signal type hint (automatic 128 Hz)."""
    print("=" * 70)
    print("Example 2: ECG Signal Type Hint (Automatic 128 Hz)")
    print("=" * 70)

    filepath = create_sample_ecg_file()

    try:
        # Load ECG data with signal type hint
        signal, metadata = load_oucru_csv(
            filepath,
            time_column='timestamp',
            signal_column='ecg_signal',
            signal_type_hint='ecg',  # Automatically uses 128 Hz
            sampling_rate_column=None
        )

        print(f"\n✓ Loaded ECG data using signal type hint!")
        print(f"\n📊 Results:")
        print(f"   - Signal type hint: 'ecg'")
        print(f"   - Sampling rate: {metadata['sampling_rate']} Hz (auto-detected)")
        print(f"   - Number of samples: {len(signal)}")
        print(f"   - Duration: {metadata['duration_seconds']:.1f} seconds")
        print(f"   - Expected samples: {metadata['n_rows'] * 128}")

        print(f"\n💡 ECG signals default to 128 Hz!")

    finally:
        os.unlink(filepath)

    print("\n")


def example_3_custom_default_rates():
    """Example 3: Custom default rates for specialized equipment."""
    print("=" * 70)
    print("Example 3: Custom Default Rates (High-Resolution ECG)")
    print("=" * 70)

    # Create high-resolution ECG file (250 Hz)
    fd, filepath = tempfile.mkstemp(suffix='_hr_ecg.csv', text=True)
    ecg_array_250 = str(list(np.random.uniform(-0.5, 1.5, 250)))

    content = f"""timestamp,ecg_signal
2024-01-01 10:00:00,"{ecg_array_250}"
2024-01-01 10:00:01,"{ecg_array_250}"
"""

    with os.fdopen(fd, 'w') as f:
        f.write(content)

    try:
        # Load with custom ECG default rate
        signal, metadata = load_oucru_csv(
            filepath,
            time_column='timestamp',
            signal_column='ecg_signal',
            signal_type_hint='ecg',
            default_ecg_rate=250,  # Override standard 128 Hz
            sampling_rate_column=None
        )

        print(f"\n✓ Loaded high-resolution ECG with custom default!")
        print(f"\n📊 Results:")
        print(f"   - Signal type hint: 'ecg'")
        print(f"   - Custom default rate: 250 Hz")
        print(f"   - Actual sampling rate: {metadata['sampling_rate']} Hz")
        print(f"   - Number of samples: {len(signal)}")

        print(f"\n💡 Useful for specialized equipment or research-grade sensors!")
        print(f"   Standard ECG: 128 Hz")
        print(f"   Research ECG: 250-1000 Hz")

    finally:
        os.unlink(filepath)

    print("\n")


def example_4_priority_demonstration():
    """Example 4: Demonstrating sampling rate priority."""
    print("=" * 70)
    print("Example 4: Sampling Rate Priority Demonstration")
    print("=" * 70)

    filepath = create_sample_ppg_file()

    try:
        # Test 1: Explicit sampling_rate overrides hint
        print("\n📌 Test 1: Explicit sampling_rate (highest priority)")
        signal1, meta1 = load_oucru_csv(
            filepath,
            sampling_rate=125,  # Explicit
            signal_type_hint='ppg',  # Would be 100, but explicit wins
            sampling_rate_column=None
        )
        print(f"   - Explicit rate: 125 Hz")
        print(f"   - Signal type hint: 'ppg' (100 Hz)")
        print(f"   - Result: {meta1['sampling_rate']} Hz ✓ (explicit wins)")

        # Test 2: Signal type hint when no explicit rate
        print("\n📌 Test 2: Signal type hint (medium priority)")
        signal2, meta2 = load_oucru_csv(
            filepath,
            signal_type_hint='ppg',  # Will be used
            sampling_rate_column=None
        )
        print(f"   - Signal type hint: 'ppg' (100 Hz)")
        print(f"   - Result: {meta2['sampling_rate']} Hz ✓ (hint used)")

        # Test 3: Array length inference when no hint
        print("\n📌 Test 3: Array length inference (lowest priority)")
        signal3, meta3 = load_oucru_csv(
            filepath,
            # No sampling_rate, no signal_type_hint
            sampling_rate_column=None
        )
        print(f"   - No explicit rate or hint")
        print(f"   - Inferred from array: {meta3['sampling_rate']} Hz ✓")

        print(f"\n💡 Priority Order:")
        print(f"   1. sampling_rate_column (if exists)")
        print(f"   2. sampling_rate parameter")
        print(f"   3. signal_type_hint")
        print(f"   4. Array length inference")

    finally:
        os.unlink(filepath)

    print("\n")


def example_5_practical_workflow():
    """Example 5: Practical workflow with mixed signal types."""
    print("=" * 70)
    print("Example 5: Practical Workflow - Mixed Signal Types")
    print("=" * 70)

    # Simulate a dataset with different signal types
    ppg_file = create_sample_ppg_file()
    ecg_file = create_sample_ecg_file()

    dataset = [
        {'path': ppg_file, 'type': 'ppg', 'label': 'Patient_001_PPG'},
        {'path': ecg_file, 'type': 'ecg', 'label': 'Patient_001_ECG'},
    ]

    try:
        print(f"\n📁 Processing dataset with {len(dataset)} files...")
        print()

        for i, file_info in enumerate(dataset, 1):
            signal, metadata = load_oucru_csv(
                file_info['path'],
                signal_type_hint=file_info['type']  # Automatic rate!
            )

            print(f"File {i}: {file_info['label']}")
            print(f"   - Type: {file_info['type'].upper()}")
            print(f"   - Sampling rate: {metadata['sampling_rate']} Hz")
            print(f"   - Samples: {len(signal)}")
            print(f"   - Duration: {metadata['duration_seconds']:.1f}s")
            print()

        print(f"✓ All files processed successfully!")
        print(f"\n💡 Each file automatically used the correct sampling rate")
        print(f"   based on its signal type - no manual configuration needed!")

    finally:
        os.unlink(ppg_file)
        os.unlink(ecg_file)

    print("\n")


def example_6_comparison():
    """Example 6: Before vs After comparison."""
    print("=" * 70)
    print("Example 6: Before vs After Signal Type Hints")
    print("=" * 70)

    filepath = create_sample_ppg_file()

    try:
        print("\n❌ BEFORE (v1.0.0):")
        print("   Code needed explicit sampling rate:")
        print()
        print("   ```python")
        print("   signal, metadata = load_oucru_csv(")
        print("       'ppg_data.csv',")
        print("       sampling_rate=100  # Had to specify manually")
        print("   )")
        print("   ```")

        print("\n✅ AFTER (v1.1.0):")
        print("   Just use signal type hint:")
        print()
        print("   ```python")
        print("   signal, metadata = load_oucru_csv(")
        print("       'ppg_data.csv',")
        print("       signal_type_hint='ppg'  # Automatic 100 Hz!")
        print("   )")
        print("   ```")

        # Actually run it
        signal, metadata = load_oucru_csv(
            filepath,
            signal_type_hint='ppg'
        )

        print(f"\n📊 Result:")
        print(f"   - Sampling rate: {metadata['sampling_rate']} Hz ✓")
        print(f"   - Less code, same result!")

    finally:
        os.unlink(filepath)

    print("\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("OUCRU CSV Signal Type Hints Examples")
    print("vitalDSP Data Loader Module v1.1.0")
    print("=" * 70 + "\n")

    # Run examples
    example_1_ppg_signal_type_hint()
    example_2_ecg_signal_type_hint()
    example_3_custom_default_rates()
    example_4_priority_demonstration()
    example_5_practical_workflow()
    example_6_comparison()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print("\n✅ All examples completed successfully!")
    print("\n🎯 Key Benefits:")
    print("   1. Automatic sampling rate for PPG (100 Hz) and ECG (128 Hz)")
    print("   2. Less code - no manual sampling rate specification needed")
    print("   3. Customizable defaults for specialized equipment")
    print("   4. Clear priority system for flexibility")
    print("   5. 100% backward compatible with existing code")
    print("\n📚 Supported Signal Types:")
    print("   - PPG: signal_type_hint='ppg' → 100 Hz")
    print("   - ECG: signal_type_hint='ecg' → 128 Hz")
    print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    main()
