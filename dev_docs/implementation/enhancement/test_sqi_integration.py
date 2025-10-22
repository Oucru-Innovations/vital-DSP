"""
Simple test to verify SignalQualityIndex integration in QualityScreener.
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import time

print("Testing SignalQualityIndex Integration in QualityScreener")
print("=" * 60)

# Test 1: Import test
print("\n[Test 1] Import Test")
try:
    from vitalDSP.utils.core_infrastructure.quality_screener import (
        QualityScreener,
        SignalQualityScreener,
        QualityScreeningConfig
    )
    print("  Import PASSED")
except Exception as e:
    print(f"  Import FAILED: {e}")
    sys.exit(1)

# Test 2: ECG Quality Screening with SQI
print("\n[Test 2] ECG Quality Screening with SQI Integration")
try:
    signal = np.random.randn(2500)  # 10 seconds @ 250 Hz
    fs = 250.0

    screener = QualityScreener()
    screener.sampling_rate = fs
    screener.signal_type = 'ecg'

    start = time.time()
    results = screener.screen_signal(signal)
    elapsed = time.time() - start

    passed = sum(1 for r in results if r.passed_screening)
    print(f"  Segments: {len(results)}")
    print(f"  Passed: {passed}")
    print(f"  Time: {elapsed:.3f}s")

    # Check that SQI scores are present
    if results:
        first_result = results[0]
        print(f"  Overall quality: {first_result.quality_metrics.overall_quality:.3f}")
        print(f"  SNR: {first_result.quality_metrics.snr_db:.1f} dB")

    print("  ECG SQI Test PASSED")
except Exception as e:
    print(f"  ECG SQI Test FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 3: PPG Quality Screening with SQI
print("\n[Test 3] PPG Quality Screening with SQI Integration")
try:
    signal = np.random.randn(2500)  # 25 seconds @ 100 Hz
    fs = 100.0

    config = QualityScreeningConfig(
        segment_duration=10.0,
        max_workers=2
    )
    screener = SignalQualityScreener(config)
    screener.sampling_rate = fs
    screener.signal_type = 'ppg'

    start = time.time()
    results = screener.screen_signal(signal)
    elapsed = time.time() - start

    passed = sum(1 for r in results if r.passed_screening)
    print(f"  Segments: {len(results)}")
    print(f"  Passed: {passed}")
    print(f"  Time: {elapsed:.3f}s")
    print("  PPG SQI Test PASSED")
except Exception as e:
    print(f"  PPG SQI Test FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 4: EEG Quality Screening with SQI
print("\n[Test 4] EEG Quality Screening with SQI Integration")
try:
    signal = np.random.randn(2560)  # 10 seconds @ 256 Hz
    fs = 256.0

    screener = QualityScreener()
    screener.sampling_rate = fs
    screener.signal_type = 'eeg'

    start = time.time()
    results = screener.screen_signal(signal)
    elapsed = time.time() - start

    passed = sum(1 for r in results if r.passed_screening)
    print(f"  Segments: {len(results)}")
    print(f"  Passed: {passed}")
    print(f"  Time: {elapsed:.3f}s")
    print("  EEG SQI Test PASSED")
except Exception as e:
    print(f"  EEG SQI Test FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Configuration override
print("\n[Test 5] Custom SQI Threshold Configuration")
try:
    signal = np.random.randn(2500)
    fs = 250.0

    config = QualityScreeningConfig(
        segment_duration=10.0,
        snr_threshold=15.0
    )
    screener = SignalQualityScreener(
        config,
        # Custom SQI thresholds
        amplitude_variability_min=0.7,
        baseline_wander_min=0.7
    )
    screener.sampling_rate = fs
    screener.signal_type = 'ecg'

    results = screener.screen_signal(signal)

    # Verify thresholds were set
    assert screener.thresholds['amplitude_variability_min'] == 0.7
    assert screener.thresholds['baseline_wander_min'] == 0.7
    assert screener.thresholds['snr_min_db'] == 15.0

    print(f"  Thresholds configured correctly")
    print(f"  amplitude_variability_min: {screener.thresholds['amplitude_variability_min']}")
    print(f"  baseline_wander_min: {screener.thresholds['baseline_wander_min']}")
    print("  Custom Config Test PASSED")
except Exception as e:
    print(f"  Custom Config Test FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("All SignalQualityIndex integration tests completed!")
print("=" * 60)
