"""
Filtering Quality Assessment Module for Physiological Signal Processing

This module provides comprehensive quality assessment metrics for evaluating
the effectiveness of signal filtering operations on physiological signals
(ECG, PPG, EEG, etc.).

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Signal-adaptive quality thresholds
- Multiple quality metrics (SNR, smoothness, peak preservation, etc.)
- Dynamic assessment based on signal type and characteristics
- Comprehensive filtering effectiveness evaluation

Examples:
--------
Basic usage:
    >>> import numpy as np
    >>> from vitalDSP.signal_quality_assessment.filtering_quality_assessment import FilteringQualityAssessment
    >>> original = np.random.randn(1000) + np.sin(np.linspace(0, 10*np.pi, 1000))
    >>> filtered = np.sin(np.linspace(0, 10*np.pi, 1000))  # Simulated filtered signal
    >>> fqa = FilteringQualityAssessment(original, filtered, fs=250, signal_type='ECG')
    >>> quality_metrics = fqa.assess_quality()
    >>> print(f"Noise reduction: {quality_metrics['noise_reduction_score']:.2f}")
"""

import numpy as np
from typing import Dict, Tuple, Optional, Literal
from vitalDSP.utils.data_processing.validation import SignalValidator
import logging

logger = logging.getLogger(__name__)

SignalType = Literal['ECG', 'PPG', 'EEG', 'Respiratory', 'General']


class FilteringQualityAssessment:
    """
    Comprehensive filtering quality assessment for physiological signals.

    This class evaluates how well a filter has processed a physiological signal
    by considering multiple metrics that are appropriate for the specific signal
    type and filtering goals.

    The assessment adapts thresholds based on:
    - Signal type (ECG, PPG, EEG, etc.)
    - Signal characteristics (amplitude, frequency content)
    - Filtering goals (noise removal vs signal preservation)

    Attributes
    ----------
    original_signal : np.ndarray
        The original (pre-filtered) signal
    filtered_signal : np.ndarray
        The filtered signal
    fs : float
        Sampling frequency in Hz
    signal_type : str
        Type of physiological signal ('ECG', 'PPG', 'EEG', 'Respiratory', 'General')

    Examples
    --------
    >>> import numpy as np
    >>> from vitalDSP.signal_quality_assessment.filtering_quality_assessment import FilteringQualityAssessment
    >>>
    >>> # Example 1: ECG filtering assessment
    >>> original_ecg = np.random.randn(1000) * 0.1 + np.sin(2*np.pi*1.2*np.linspace(0,10,1000))
    >>> filtered_ecg = np.sin(2*np.pi*1.2*np.linspace(0,10,1000))
    >>> fqa_ecg = FilteringQualityAssessment(original_ecg, filtered_ecg, fs=250, signal_type='ECG')
    >>> results = fqa_ecg.assess_quality()
    >>> print(f"Overall quality: {results['overall_quality']}")
    >>>
    >>> # Example 2: PPG filtering assessment
    >>> original_ppg = np.random.randn(1000) * 0.05 + np.sin(2*np.pi*1.0*np.linspace(0,10,1000))
    >>> filtered_ppg = np.sin(2*np.pi*1.0*np.linspace(0,10,1000))
    >>> fqa_ppg = FilteringQualityAssessment(original_ppg, filtered_ppg, fs=128, signal_type='PPG')
    >>> results = fqa_ppg.assess_quality()
    >>> print(f"Noise reduction: {results['noise_reduction_score']:.2f}")
    """

    def __init__(
        self,
        original_signal: np.ndarray,
        filtered_signal: np.ndarray,
        fs: float = 250.0,
        signal_type: SignalType = 'General'
    ):
        """
        Initialize the FilteringQualityAssessment.

        Parameters
        ----------
        original_signal : np.ndarray
            The original (noisy/unfiltered) signal
        filtered_signal : np.ndarray
            The filtered signal
        fs : float, optional
            Sampling frequency in Hz (default: 250)
        signal_type : str, optional
            Type of signal: 'ECG', 'PPG', 'EEG', 'Respiratory', or 'General' (default: 'General')

        Raises
        ------
        ValueError
            If signals have different lengths or are invalid
        """
        # Validate inputs
        if not isinstance(original_signal, np.ndarray):
            original_signal = np.array(original_signal)
        if not isinstance(filtered_signal, np.ndarray):
            filtered_signal = np.array(filtered_signal)

        SignalValidator.validate_signal(
            original_signal, min_length=10, allow_empty=False, signal_name="original_signal"
        )
        SignalValidator.validate_signal(
            filtered_signal, min_length=10, allow_empty=False, signal_name="filtered_signal"
        )

        if len(original_signal) != len(filtered_signal):
            raise ValueError(
                f"Signal lengths must match. Got original: {len(original_signal)}, "
                f"filtered: {len(filtered_signal)}"
            )

        self.original_signal = original_signal
        self.filtered_signal = filtered_signal
        self.fs = fs
        self.signal_type = signal_type

        # Get signal-specific thresholds
        self.thresholds = self._get_adaptive_thresholds()

    def _get_adaptive_thresholds(self) -> Dict[str, Dict[str, float]]:
        """
        Get signal-adaptive quality thresholds.

        Different signal types have different acceptable ranges for quality metrics.
        For example:
        - ECG: Moderate correlation (0.7-0.9) is good - we want to remove baseline wander
        - PPG: Higher correlation (0.8-0.95) preferred - preserve pulse morphology
        - EEG: Lower correlation (0.6-0.85) acceptable - heavy filtering for artifacts

        Returns
        -------
        dict
            Nested dictionary with thresholds for each metric
        """
        if self.signal_type == 'ECG':
            return {
                'noise_reduction': {'excellent': 0.3, 'good': 0.15, 'acceptable': 0.05},
                'smoothness_improvement': {'excellent': 0.2, 'good': 0.1, 'acceptable': 0.05},
                'peak_preservation': {'excellent': 0.9, 'good': 0.8, 'acceptable': 0.7},
                'shape_similarity': {'excellent': 0.75, 'good': 0.65, 'acceptable': 0.55},
                'snr_db': {'excellent': 15, 'good': 10, 'acceptable': 5},
            }
        elif self.signal_type == 'PPG':
            return {
                'noise_reduction': {'excellent': 0.25, 'good': 0.12, 'acceptable': 0.05},
                'smoothness_improvement': {'excellent': 0.25, 'good': 0.15, 'acceptable': 0.08},
                'peak_preservation': {'excellent': 0.92, 'good': 0.85, 'acceptable': 0.75},
                'shape_similarity': {'excellent': 0.85, 'good': 0.75, 'acceptable': 0.65},
                'snr_db': {'excellent': 18, 'good': 12, 'acceptable': 7},
            }
        elif self.signal_type == 'EEG':
            return {
                'noise_reduction': {'excellent': 0.4, 'good': 0.2, 'acceptable': 0.1},
                'smoothness_improvement': {'excellent': 0.15, 'good': 0.08, 'acceptable': 0.03},
                'peak_preservation': {'excellent': 0.85, 'good': 0.75, 'acceptable': 0.65},
                'shape_similarity': {'excellent': 0.70, 'good': 0.60, 'acceptable': 0.50},
                'snr_db': {'excellent': 12, 'good': 8, 'acceptable': 4},
            }
        elif self.signal_type == 'Respiratory':
            return {
                'noise_reduction': {'excellent': 0.3, 'good': 0.15, 'acceptable': 0.08},
                'smoothness_improvement': {'excellent': 0.3, 'good': 0.2, 'acceptable': 0.1},
                'peak_preservation': {'excellent': 0.88, 'good': 0.78, 'acceptable': 0.68},
                'shape_similarity': {'excellent': 0.80, 'good': 0.70, 'acceptable': 0.60},
                'snr_db': {'excellent': 16, 'good': 10, 'acceptable': 6},
            }
        else:  # General
            return {
                'noise_reduction': {'excellent': 0.3, 'good': 0.15, 'acceptable': 0.08},
                'smoothness_improvement': {'excellent': 0.2, 'good': 0.1, 'acceptable': 0.05},
                'peak_preservation': {'excellent': 0.88, 'good': 0.78, 'acceptable': 0.68},
                'shape_similarity': {'excellent': 0.75, 'good': 0.65, 'acceptable': 0.55},
                'snr_db': {'excellent': 15, 'good': 10, 'acceptable': 5},
            }

    def calculate_noise_reduction(self) -> Tuple[float, str, str]:
        """
        Calculate noise reduction metric.

        This metric quantifies how much noise was removed by the filter.
        It's calculated as the ratio of the power of the removed component
        (difference between original and filtered) to the original signal power.

        Higher values indicate more aggressive filtering (more noise removed).

        Returns
        -------
        score : float
            Noise reduction ratio (0 to 1, where higher is more reduction)
        status : str
            Quality status ('Excellent', 'Good', 'Acceptable', 'Poor')
        assessment : str
            Detailed assessment message
        """
        try:
            # Calculate the removed component (assumed noise)
            removed_component = self.original_signal - self.filtered_signal

            # Calculate power of original and removed component
            original_power = np.mean(self.original_signal ** 2)
            removed_power = np.mean(removed_component ** 2)

            # Noise reduction ratio
            if original_power > 1e-10:
                noise_reduction_ratio = removed_power / original_power
            else:
                noise_reduction_ratio = 0.0

            # Assess quality based on thresholds
            thresholds = self.thresholds['noise_reduction']
            if noise_reduction_ratio >= thresholds['excellent']:
                status = 'Excellent'
                assessment = f'Excellent noise reduction ({noise_reduction_ratio:.1%} of signal power removed). Filter effectively removed noise while preserving signal.'
            elif noise_reduction_ratio >= thresholds['good']:
                status = 'Good'
                assessment = f'Good noise reduction ({noise_reduction_ratio:.1%} removed). Filter balanced noise removal with signal preservation.'
            elif noise_reduction_ratio >= thresholds['acceptable']:
                status = 'Acceptable'
                assessment = f'Acceptable noise reduction ({noise_reduction_ratio:.1%} removed). Minimal filtering applied, suitable for clean signals.'
            else:
                status = 'Poor'
                assessment = f'Minimal noise reduction ({noise_reduction_ratio:.1%} removed). Filter had little effect - signal may already be clean or filter settings too conservative.'

            return noise_reduction_ratio, status, assessment

        except Exception as e:
            logger.error(f"Error calculating noise reduction: {e}")
            return 0.0, 'Unknown', f'Error: {str(e)}'

    def calculate_snr_improvement(self) -> Tuple[float, str, str]:
        """
        Calculate Signal-to-Noise Ratio (SNR) improvement in dB.

        Estimates SNR by treating the difference between original and filtered
        as the noise estimate, and the filtered signal as the clean signal estimate.

        Returns
        -------
        snr_db : float
            SNR in decibels
        status : str
            Quality status
        assessment : str
            Detailed assessment
        """
        try:
            # Estimate noise as difference
            noise_estimate = self.original_signal - self.filtered_signal

            # Calculate powers
            signal_power = np.mean(self.filtered_signal ** 2)
            noise_power = np.mean(noise_estimate ** 2)

            # Calculate SNR in dB
            if noise_power > 1e-10 and signal_power > 1e-10:
                snr_db = 10 * np.log10(signal_power / noise_power)
            else:
                snr_db = 0.0

            # Assess based on thresholds
            thresholds = self.thresholds['snr_db']
            if snr_db >= thresholds['excellent']:
                status = 'Excellent'
                assessment = f'Excellent SNR ({snr_db:.1f} dB). Very high signal quality after filtering.'
            elif snr_db >= thresholds['good']:
                status = 'Good'
                assessment = f'Good SNR ({snr_db:.1f} dB). Signal quality is good for most applications.'
            elif snr_db >= thresholds['acceptable']:
                status = 'Acceptable'
                assessment = f'Acceptable SNR ({snr_db:.1f} dB). Signal usable but may benefit from additional processing.'
            else:
                status = 'Poor'
                assessment = f'Low SNR ({snr_db:.1f} dB). Signal quality may be insufficient for reliable analysis.'

            return snr_db, status, assessment

        except Exception as e:
            logger.error(f"Error calculating SNR: {e}")
            return 0.0, 'Unknown', f'Error: {str(e)}'

    def calculate_smoothness_improvement(self) -> Tuple[float, str, str]:
        """
        Calculate improvement in signal smoothness.

        Smoothness is measured by the variance of first-order differences.
        Lower variance indicates smoother signal.

        Returns
        -------
        improvement : float
            Relative improvement in smoothness (0 to 1)
        status : str
            Quality status
        assessment : str
            Detailed assessment
        """
        try:
            # Calculate smoothness using variance of differences
            orig_diff_var = np.var(np.diff(self.original_signal))
            filt_diff_var = np.var(np.diff(self.filtered_signal))

            # Calculate improvement
            if orig_diff_var > 1e-10:
                improvement = (orig_diff_var - filt_diff_var) / orig_diff_var
                # Clamp to [0, 1]
                improvement = max(0.0, min(1.0, improvement))
            else:
                improvement = 0.0

            # Assess based on thresholds
            thresholds = self.thresholds['smoothness_improvement']
            if improvement >= thresholds['excellent']:
                status = 'Excellent'
                assessment = f'Excellent smoothing ({improvement:.1%} improvement). Filter effectively reduced high-frequency noise.'
            elif improvement >= thresholds['good']:
                status = 'Good'
                assessment = f'Good smoothing ({improvement:.1%} improvement). Signal is smoother while preserving features.'
            elif improvement >= thresholds['acceptable']:
                status = 'Acceptable'
                assessment = f'Acceptable smoothing ({improvement:.1%} improvement). Moderate smoothing applied.'
            else:
                status = 'Poor'
                assessment = f'Minimal smoothing ({improvement:.1%} improvement). Filter preserved original signal roughness.'

            return improvement, status, assessment

        except Exception as e:
            logger.error(f"Error calculating smoothness: {e}")
            return 0.0, 'Unknown', f'Error: {str(e)}'

    def calculate_peak_preservation(self) -> Tuple[float, str, str]:
        """
        Calculate how well the filter preserved signal peaks.

        This is crucial for physiological signals where peaks represent
        important events (R-peaks in ECG, systolic peaks in PPG, etc.).

        Returns
        -------
        preservation_score : float
            Peak preservation score (0 to 1)
        status : str
            Quality status
        assessment : str
            Detailed assessment
        """
        try:
            from vitalDSP.utils.signal_processing.peak_detection import PeakDetection

            # Detect peaks in both signals
            orig_detector = PeakDetection(self.original_signal, method='threshold', height=np.mean(self.original_signal))
            filt_detector = PeakDetection(self.filtered_signal, method='threshold', height=np.mean(self.filtered_signal))

            orig_peaks = orig_detector.detect_peaks()
            filt_peaks = filt_detector.detect_peaks()

            if len(orig_peaks) == 0:
                return 1.0, 'Excellent', 'No peaks detected in original signal (signal may be flat or very noisy).'

            # Calculate how many original peaks were preserved (within tolerance)
            tolerance = int(0.05 * self.fs)  # 50ms tolerance
            preserved_count = 0

            for orig_peak in orig_peaks:
                # Check if there's a filtered peak within tolerance
                if np.any(np.abs(filt_peaks - orig_peak) <= tolerance):
                    preserved_count += 1

            preservation_score = preserved_count / len(orig_peaks)

            # Assess based on thresholds
            thresholds = self.thresholds['peak_preservation']
            if preservation_score >= thresholds['excellent']:
                status = 'Excellent'
                assessment = f'Excellent peak preservation ({preservation_score:.1%}). Filter maintained critical signal features.'
            elif preservation_score >= thresholds['good']:
                status = 'Good'
                assessment = f'Good peak preservation ({preservation_score:.1%}). Most important features preserved.'
            elif preservation_score >= thresholds['acceptable']:
                status = 'Acceptable'
                assessment = f'Acceptable peak preservation ({preservation_score:.1%}). Some peaks may have been smoothed.'
            else:
                status = 'Poor'
                assessment = f'Poor peak preservation ({preservation_score:.1%}). Filter may have over-smoothed important features.'

            return preservation_score, status, assessment

        except Exception as e:
            logger.error(f"Error calculating peak preservation: {e}")
            return 0.0, 'Unknown', f'Error: {str(e)}'

    def calculate_shape_similarity(self) -> Tuple[float, str, str]:
        """
        Calculate morphological similarity between original and filtered signals.

        Uses correlation but interprets it correctly for filtering:
        - We WANT some difference (noise removed)
        - But preserve overall shape

        Returns
        -------
        similarity : float
            Pearson correlation coefficient (0 to 1)
        status : str
            Quality status
        assessment : str
            Detailed assessment
        """
        try:
            from vitalDSP.utils.config_utilities.common import pearsonr

            correlation = pearsonr(self.original_signal, self.filtered_signal)

            # Assess based on signal-adaptive thresholds
            thresholds = self.thresholds['shape_similarity']
            if correlation >= thresholds['excellent']:
                status = 'Excellent'
                assessment = f'Excellent shape preservation ({correlation:.3f}). Filter preserved signal morphology while removing noise.'
            elif correlation >= thresholds['good']:
                status = 'Good'
                assessment = f'Good shape preservation ({correlation:.3f}). Signal structure maintained with noise reduction.'
            elif correlation >= thresholds['acceptable']:
                status = 'Acceptable'
                assessment = f'Acceptable shape preservation ({correlation:.3f}). Moderate filtering with some shape changes.'
            else:
                status = 'Poor'
                assessment = f'Poor shape preservation ({correlation:.3f}). Filter significantly altered signal morphology.'

            return correlation, status, assessment

        except Exception as e:
            logger.error(f"Error calculating shape similarity: {e}")
            return 0.0, 'Unknown', f'Error: {str(e)}'

    def assess_quality(self) -> Dict[str, any]:
        """
        Perform comprehensive quality assessment of filtering.

        Calculates all quality metrics and provides overall assessment.

        Returns
        -------
        dict
            Comprehensive quality assessment results containing:
            - Individual metric scores and assessments
            - Overall quality rating
            - Recommendations

        Examples
        --------
        >>> fqa = FilteringQualityAssessment(original, filtered, fs=250, signal_type='ECG')
        >>> results = fqa.assess_quality()
        >>> print(results['overall_quality'])
        'Excellent'
        >>> print(results['noise_reduction_score'])
        0.25
        """
        # Calculate all metrics
        noise_reduction_score, noise_status, noise_assessment = self.calculate_noise_reduction()
        snr_db, snr_status, snr_assessment = self.calculate_snr_improvement()
        smoothness_improvement, smooth_status, smooth_assessment = self.calculate_smoothness_improvement()
        peak_preservation, peak_status, peak_assessment = self.calculate_peak_preservation()
        shape_similarity, shape_status, shape_assessment = self.calculate_shape_similarity()

        # Calculate overall quality (weighted average of status scores)
        status_to_score = {'Excellent': 3, 'Good': 2, 'Acceptable': 1, 'Poor': 0, 'Unknown': 0}

        scores = [
            status_to_score[noise_status],
            status_to_score[snr_status],
            status_to_score[smooth_status],
            status_to_score[peak_status] * 1.5,  # Weight peak preservation higher
            status_to_score[shape_status] * 1.2,  # Weight shape similarity higher
        ]

        avg_score = np.mean(scores)

        if avg_score >= 2.5:
            overall_quality = 'Excellent'
            recommendation = 'Filter settings are optimal for this signal type.'
        elif avg_score >= 1.5:
            overall_quality = 'Good'
            recommendation = 'Filter performing well. Consider minor adjustments if specific features need better preservation.'
        elif avg_score >= 0.8:
            overall_quality = 'Acceptable'
            recommendation = 'Filter is working but could be improved. Consider adjusting cutoff frequencies or filter order.'
        else:
            overall_quality = 'Poor'
            recommendation = 'Filter settings may be inappropriate for this signal. Review filter parameters and signal characteristics.'

        return {
            'overall_quality': overall_quality,
            'recommendation': recommendation,
            'signal_type': self.signal_type,
            'sampling_frequency': self.fs,
            'metrics': {
                'noise_reduction': {
                    'score': noise_reduction_score,
                    'status': noise_status,
                    'assessment': noise_assessment,
                },
                'snr_db': {
                    'value': snr_db,
                    'status': snr_status,
                    'assessment': snr_assessment,
                },
                'smoothness_improvement': {
                    'score': smoothness_improvement,
                    'status': smooth_status,
                    'assessment': smooth_assessment,
                },
                'peak_preservation': {
                    'score': peak_preservation,
                    'status': peak_status,
                    'assessment': peak_assessment,
                },
                'shape_similarity': {
                    'score': shape_similarity,
                    'status': shape_status,
                    'assessment': shape_assessment,
                },
            }
        }
