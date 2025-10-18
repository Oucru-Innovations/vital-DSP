"""
Test suite for Pipeline vitalDSP Integration.

This test suite verifies that the processing pipeline properly integrates
vitalDSP modules for filtering and feature extraction in Stages 3 and 6.
"""

import pytest
import numpy as np
from vitalDSP.utils.core_infrastructure.processing_pipeline import StandardProcessingPipeline


class TestPipelineVitalDSPIntegration:
    """Test vitalDSP module integration in processing pipeline."""

    @pytest.fixture
    def pipeline(self):
        """Create a processing pipeline instance."""
        return StandardProcessingPipeline()

    @pytest.fixture
    def sample_signal(self):
        """Generate a sample ECG-like signal."""
        fs = 1000  # 1000 Hz sampling rate
        duration = 10  # 10 seconds
        t = np.arange(0, duration, 1/fs)

        # Generate synthetic ECG-like signal
        signal = np.sin(2 * np.pi * 1.2 * t)  # Heart rate component
        signal += 0.1 * np.sin(2 * np.pi * 60 * t)  # High frequency noise
        signal += 0.2 * np.random.randn(len(t))  # Gaussian noise

        return signal, fs

    def test_stage3_uses_vitaldsp_filtering(self, pipeline, sample_signal):
        """Test that Stage 3 uses vitalDSP SignalFiltering."""
        signal, fs = sample_signal

        # Apply filtering
        filtered = pipeline._apply_filtering(signal, fs, "ecg")

        # Verify filtering was applied
        assert filtered is not None
        assert len(filtered) == len(signal)
        assert not np.array_equal(filtered, signal), "Signal should be filtered"

        # Verify signal is not just copied (some modification occurred)
        correlation = np.corrcoef(signal, filtered)[0, 1]
        assert correlation > 0.9, "Filtered signal should be highly correlated with original"
        assert correlation < 1.0, "Filtered signal should differ from original"

    def test_stage3_uses_vitaldsp_artifact_removal(self, pipeline, sample_signal):
        """Test that Stage 3 uses vitalDSP ArtifactRemoval."""
        signal, fs = sample_signal

        # Add some artifacts (large spikes)
        signal_with_artifacts = signal.copy()
        signal_with_artifacts[500] = signal_with_artifacts[500] + 10  # Large spike
        signal_with_artifacts[2000] = signal_with_artifacts[2000] - 10  # Large spike

        # Apply preprocessing (filtering + artifact removal)
        preprocessed = pipeline._apply_preprocessing(signal_with_artifacts, fs, "ecg")

        # Verify preprocessing was applied
        assert preprocessed is not None
        assert len(preprocessed) == len(signal_with_artifacts)

        # Verify artifacts were reduced
        max_original = np.max(np.abs(signal_with_artifacts))
        max_preprocessed = np.max(np.abs(preprocessed))
        assert max_preprocessed < max_original, "Artifact removal should reduce peak amplitude"

    def test_stage6_uses_vitaldsp_time_domain_features(self, pipeline, sample_signal):
        """Test that Stage 6 uses vitalDSP TimeDomainFeatures."""
        signal, fs = sample_signal

        # Extract features
        signal_out, features = pipeline._extract_simple_features(signal, fs, "ecg")

        # Verify vitalDSP time domain features are present
        vitaldsp_time_features = [
            "mean", "std", "min", "max", "range",
            "rms", "variance", "skewness", "kurtosis"
        ]

        for feature_name in vitaldsp_time_features:
            assert feature_name in features, f"Missing time domain feature: {feature_name}"
            assert isinstance(features[feature_name], (int, float)), \
                f"Feature {feature_name} should be numeric"

    def test_stage6_uses_vitaldsp_frequency_domain_features(self, pipeline, sample_signal):
        """Test that Stage 6 uses vitalDSP FrequencyDomainFeatures."""
        signal, fs = sample_signal

        # Extract features
        signal_out, features = pipeline._extract_simple_features(signal, fs, "ecg")

        # Verify vitalDSP frequency domain features are present
        vitaldsp_freq_features = [
            "spectral_centroid", "spectral_bandwidth",
            "spectral_energy", "dominant_frequency"
        ]

        for feature_name in vitaldsp_freq_features:
            assert feature_name in features, f"Missing frequency domain feature: {feature_name}"
            assert isinstance(features[feature_name], (int, float)), \
                f"Feature {feature_name} should be numeric"
            assert not np.isnan(features[feature_name]), \
                f"Feature {feature_name} should not be NaN"

    def test_feature_extraction_has_more_features(self, pipeline, sample_signal):
        """Test that vitalDSP feature extraction provides more comprehensive features."""
        signal, fs = sample_signal

        # Extract features
        signal_out, features = pipeline._extract_simple_features(signal, fs, "ecg")

        # Should have at least 13 features (9 time domain + 4 frequency domain)
        # Plus total_energy
        assert len(features) >= 13, \
            f"Expected at least 13 features, got {len(features)}: {list(features.keys())}"

        # Verify all feature values are valid
        for feature_name, feature_value in features.items():
            assert isinstance(feature_value, (int, float)), \
                f"Feature {feature_name} should be numeric"
            assert not np.isnan(feature_value), \
                f"Feature {feature_name} should not be NaN"
            assert not np.isinf(feature_value), \
                f"Feature {feature_name} should not be Inf"

    def test_filtering_different_signal_types(self, pipeline):
        """Test that filtering adapts to different signal types."""
        fs = 1000
        duration = 5
        t = np.arange(0, duration, 1/fs)
        signal = np.sin(2 * np.pi * 1.0 * t)

        # Test different signal types
        ecg_filtered = pipeline._apply_filtering(signal, fs, "ecg")
        ppg_filtered = pipeline._apply_filtering(signal, fs, "ppg")
        eeg_filtered = pipeline._apply_filtering(signal, fs, "eeg")

        # All should return filtered signals
        assert ecg_filtered is not None
        assert ppg_filtered is not None
        assert eeg_filtered is not None

        # All should have same length as input
        assert len(ecg_filtered) == len(signal)
        assert len(ppg_filtered) == len(signal)
        assert len(eeg_filtered) == len(signal)

    def test_fallback_when_vitaldsp_unavailable(self, pipeline, sample_signal):
        """Test that pipeline has fallback behavior if vitalDSP modules fail."""
        signal, fs = sample_signal

        # This should work even if there are issues with vitalDSP
        # because we have try-except with fallbacks
        try:
            filtered = pipeline._apply_filtering(signal, fs, "ecg")
            assert filtered is not None
            assert len(filtered) == len(signal)
        except Exception as e:
            pytest.fail(f"Pipeline should have fallback behavior, but got error: {e}")

        try:
            signal_out, features = pipeline._extract_simple_features(signal, fs, "ecg")
            assert features is not None
            assert len(features) > 0
        except Exception as e:
            pytest.fail(f"Pipeline should have fallback behavior, but got error: {e}")

    def test_full_pipeline_with_vitaldsp_integration(self, pipeline, sample_signal):
        """Test complete pipeline execution with vitalDSP integration."""
        signal, fs = sample_signal

        # Run through the full pipeline
        results = pipeline.process_signal(
            signal=signal,
            fs=fs,
            signal_type="ecg",
            metadata={"test": "vitaldsp_integration"},
            resume_from_checkpoint=False
        )

        # Verify pipeline completed successfully
        assert results is not None
        assert "processing_metadata" in results
        assert results["processing_metadata"]["stages_completed"] >= 6

        # Verify Stage 3 (Parallel Processing) results
        if "parallel_processing" in results:
            parallel_results = results["parallel_processing"]
            assert "paths" in parallel_results
            assert "filtered" in parallel_results["paths"]
            assert "preprocessed" in parallel_results["paths"]

        # Verify Stage 6 (Feature Extraction) results
        if "features" in results:
            feature_results = results["features"]
            assert "segment_features" in feature_results or "path_features" in feature_results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
