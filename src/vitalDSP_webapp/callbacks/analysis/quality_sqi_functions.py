"""
Signal Quality Index (SQI) Functions
Provides parameter generation and computation for all vitalDSP SQI types.
"""

import numpy as np
from dash import html, dcc
import dash_bootstrap_components as dbc
import logging

logger = logging.getLogger(__name__)


# Default thresholds for each SQI type
SQI_DEFAULT_THRESHOLDS = {
    "snr_sqi": 0.7,  # SNR SQI threshold (0-1 scale)
    "baseline_wander_sqi": 0.05,  # Lower is better
    "amplitude_variability_sqi": 0.3,  # Moderate variability expected
    "zero_crossing_sqi": 0.5,  # Consistency threshold
    "waveform_similarity_sqi": 0.7,  # High similarity expected
    "signal_entropy_sqi": 0.6,  # Moderate entropy
    "energy_sqi": 0.5,  # Energy level threshold
    "kurtosis_sqi": 3.0,  # Near-Gaussian (3.0 = Gaussian)
    "skewness_sqi": 0.5,  # Near-symmetric
    "peak_to_peak_amplitude_sqi": 0.4,  # Amplitude consistency
    "ppg_signal_quality_sqi": 0.7,  # PPG-specific quality
    "respiratory_signal_quality_sqi": 0.7,  # Respiratory-specific quality
    "heart_rate_variability_sqi": 0.6,  # HRV quality
    "eeg_band_power_sqi": 0.5,  # EEG band power quality
}

# Threshold descriptions and interpretation guides
SQI_THRESHOLD_DESCRIPTIONS = {
    "snr_sqi": {
        "description": "Signal-to-Noise Ratio SQI measures the strength of the signal relative to noise.",
        "range": "0.0 - 1.0",
        "direction": "Higher is better (value > threshold = good quality)",
        "interpretation": "Values closer to 1.0 indicate less noise and better signal quality.",
    },
    "baseline_wander_sqi": {
        "description": "Measures the amount of baseline drift in the signal.",
        "range": "0.0 - 1.0",
        "direction": "Lower is better (value < threshold = good quality)",
        "interpretation": "Values closer to 0.0 indicate minimal baseline wander.",
    },
    "amplitude_variability_sqi": {
        "description": "Assesses the consistency of signal amplitude over time.",
        "range": "0.0 - 1.0",
        "direction": "Moderate is better (reasonable variability expected)",
        "interpretation": "Good quality signals show moderate amplitude variation, not too much or too little.",
    },
    "zero_crossing_sqi": {
        "description": "Measures signal stability using zero-crossing rate.",
        "range": "0.0 - 1.0",
        "direction": "Higher is better (value > threshold = consistent signal)",
        "interpretation": "Higher values indicate more consistent signal characteristics.",
    },
    "waveform_similarity_sqi": {
        "description": "Evaluates similarity of signal segments to a reference pattern.",
        "range": "0.0 - 1.0",
        "direction": "Higher is better (value > threshold = similar waveforms)",
        "interpretation": "Values closer to 1.0 indicate more consistent waveform patterns.",
    },
    "signal_entropy_sqi": {
        "description": "Measures signal complexity and information content.",
        "range": "0.0 - 1.0",
        "direction": "Moderate is better (reasonable entropy expected)",
        "interpretation": "Physiological signals typically have moderate entropy. Too high or too low may indicate issues.",
    },
    "energy_sqi": {
        "description": "Assesses signal energy levels over time.",
        "range": "0.0 - 1.0",
        "direction": "Higher is better (value > threshold = sufficient energy)",
        "interpretation": "Higher values indicate stronger, more detectable signals.",
    },
    "kurtosis_sqi": {
        "description": "Measures the 'tailedness' of the signal distribution.",
        "range": "Typically 0.0 - 10.0",
        "direction": "Near 3.0 is best (value ≈ 3.0 = Gaussian-like distribution)",
        "interpretation": "Value of 3.0 indicates Gaussian-like distribution. Deviations suggest artifacts.",
    },
    "skewness_sqi": {
        "description": "Measures the asymmetry of the signal distribution.",
        "range": "Typically -2.0 to 2.0",
        "direction": "Near 0 is best (value ≈ 0 = symmetric distribution)",
        "interpretation": "Values near 0 indicate symmetric signal. Deviations may indicate bias.",
    },
    "peak_to_peak_amplitude_sqi": {
        "description": "Evaluates consistency of peak-to-peak amplitudes.",
        "range": "0.0 - 1.0",
        "direction": "Higher is better (value > threshold = consistent amplitude)",
        "interpretation": "Higher values indicate more consistent signal amplitudes.",
    },
    "ppg_signal_quality_sqi": {
        "description": "PPG-specific quality metric assessing pulse signal characteristics.",
        "range": "0.0 - 1.0",
        "direction": "Higher is better (value > threshold = good PPG quality)",
        "interpretation": "Higher values indicate better pulse signal quality and detectability.",
    },
    "respiratory_signal_quality_sqi": {
        "description": "Respiratory-specific quality metric for breathing signal assessment.",
        "range": "0.0 - 1.0",
        "direction": "Higher is better (value > threshold = good respiratory quality)",
        "interpretation": "Higher values indicate better breathing signal quality.",
    },
    "heart_rate_variability_sqi": {
        "description": "Assesses variability in heart rate intervals.",
        "range": "0.0 - 1.0",
        "direction": "Higher is better (value > threshold = healthy HRV)",
        "interpretation": "Higher values indicate more natural heart rate variability.",
    },
    "eeg_band_power_sqi": {
        "description": "Evaluates EEG band power quality for brain signal analysis.",
        "range": "0.0 - 1.0",
        "direction": "Higher is better (value > threshold = good EEG quality)",
        "interpretation": "Higher values indicate better EEG signal quality in selected frequency band.",
    },
}


def get_sqi_parameters_layout(sqi_type):
    """
    Generate dynamic parameter layout based on selected SQI type.

    Args:
        sqi_type: Selected SQI type

    Returns:
        list: Dash components for parameters
    """

    # Get SQI description and add as informational alert
    sqi_desc = SQI_THRESHOLD_DESCRIPTIONS.get(sqi_type, {
        "description": f"Quality assessment using {sqi_type.replace('_', ' ').title()}",
        "range": "0.0 - 1.0",
        "direction": "Custom interpretation",
        "interpretation": "See documentation for specific interpretation.",
    })
    
    # Common parameters for all SQIs
    common_params = [
        html.H6(f"{sqi_type.replace('_', ' ').title()} Parameters", className="mb-3"),
        dbc.Alert(
            [
                html.Strong("Description: "),
                html.Span(sqi_desc["description"]),
                html.Br(),
                html.Strong("Threshold Range: "),
                html.Span(sqi_desc["range"]),
                html.Br(),
                html.Strong("Quality Direction: "),
                html.Span(sqi_desc["direction"]),
                html.Br(),
                html.Strong("Interpretation: "),
                html.Span(sqi_desc["interpretation"]),
            ],
            color="info",
            className="mb-3",
        ),
        dbc.Row([
            dbc.Col([
                html.Label("Window Size (samples)", className="form-label"),
                dbc.Input(
                    id={"type": "sqi-param", "param": "window_size"},
                    type="number",
                    value=1000,
                    min=100,
                    step=100,
                ),
            ], md=6),
            dbc.Col([
                html.Label("Step Size (samples)", className="form-label"),
                dbc.Input(
                    id={"type": "sqi-param", "param": "step_size"},
                    type="number",
                    value=500,
                    min=50,
                    step=50,
                ),
            ], md=6),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([
                html.Label("Threshold Type", className="form-label"),
                dbc.Select(
                    id={"type": "sqi-param", "param": "threshold_type"},
                    options=[
                        {"label": "Below Threshold = Abnormal (values >= threshold are good)", "value": "below"},
                        {"label": "Above Threshold = Abnormal (values <= threshold are good)", "value": "above"},
                        {"label": "Range (min-max)", "value": "range"},
                    ],
                    value="below",
                ),
            ], md=12),
        ], className="mb-3"),
        # Threshold inputs - single value or range
        dbc.Row(
            id={"type": "sqi-param-threshold-row", "param": "single"},
            children=[
                dbc.Col([
                    html.Label("Threshold Value", className="form-label"),
                    dbc.Input(
                        id={"type": "sqi-param", "param": "threshold"},
                        type="number",
                        value=SQI_DEFAULT_THRESHOLDS.get(sqi_type, 0.5),
                        min=0,
                        max=1,
                        step=0.05,
                    ),
                ], md=12),
            ],
            className="mb-3",
        ),
        dbc.Row(
            id={"type": "sqi-param-threshold-row", "param": "range"},
            children=[
                dbc.Col([
                    html.Label("Threshold Min Value", className="form-label"),
                    dbc.Input(
                        id={"type": "sqi-param", "param": "threshold_min"},
                        type="number",
                        value=0.2,
                        min=0,
                        max=1,
                        step=0.05,
                    ),
                ], md=6),
                dbc.Col([
                    html.Label("Threshold Max Value", className="form-label"),
                    dbc.Input(
                        id={"type": "sqi-param", "param": "threshold_max"},
                        type="number",
                        value=0.8,
                        min=0,
                        max=1,
                        step=0.05,
                    ),
                ], md=6),
            ],
            className="mb-3",
            style={"display": "none"},  # Hidden by default
        ),
        dbc.Row([
            dbc.Col([
                html.Label("Scaling Method", className="form-label"),
                dbc.Select(
                    id={"type": "sqi-param", "param": "scale"},
                    options=[
                        {"label": "Z-Score", "value": "zscore"},
                        {"label": "IQR (Interquartile Range)", "value": "iqr"},
                        {"label": "Min-Max", "value": "minmax"},
                    ],
                    value="zscore",
                ),
            ], md=12),
        ], className="mb-3"),
    ]

    # SQI-specific parameters
    specific_params = []

    if sqi_type == "baseline_wander_sqi":
        specific_params = [
            dbc.Row([
                dbc.Col([
                    html.Label("Moving Average Window", className="form-label"),
                    dbc.Input(
                        id={"type": "sqi-param", "param": "moving_avg_window"},
                        type="number",
                        value=50,
                        min=10,
                        step=10,
                    ),
                ], md=12),
            ], className="mb-3"),
        ]

    elif sqi_type == "waveform_similarity_sqi":
        specific_params = [
            dbc.Row([
                dbc.Col([
                    html.Label("Reference Window Index", className="form-label"),
                    dbc.Input(
                        id={"type": "sqi-param", "param": "reference_window_index"},
                        type="number",
                        value=0,
                        min=0,
                        step=1,
                    ),
                ], md=6),
                dbc.Col([
                    html.Label("Similarity Metric", className="form-label"),
                    dbc.Select(
                        id={"type": "sqi-param", "param": "similarity_metric"},
                        options=[
                            {"label": "Correlation", "value": "correlation"},
                            {"label": "Euclidean Distance", "value": "euclidean"},
                            {"label": "Cosine Similarity", "value": "cosine"},
                        ],
                        value="correlation",
                    ),
                ], md=6),
            ], className="mb-3"),
        ]

    elif sqi_type == "signal_entropy_sqi":
        # Note: num_bins is not configurable - the method internally uses bins=10
        specific_params = []

    elif sqi_type == "ppg_signal_quality_sqi":
        # PPG SQI doesn't need additional parameters
        specific_params = []

    elif sqi_type == "respiratory_signal_quality_sqi":
        # Respiratory SQI doesn't need additional parameters
        specific_params = []

    elif sqi_type == "heart_rate_variability_sqi":
        # HRV SQI will compute RR intervals from signal automatically
        # No additional parameters needed
        specific_params = []

    elif sqi_type == "eeg_band_power_sqi":
        specific_params = [
            dbc.Row([
                dbc.Col([
                    html.Label("EEG Band", className="form-label"),
                    dbc.Select(
                        id={"type": "sqi-param", "param": "band"},
                        options=[
                            {"label": "Delta (0.5-4 Hz)", "value": "delta"},
                            {"label": "Theta (4-8 Hz)", "value": "theta"},
                            {"label": "Alpha (8-13 Hz)", "value": "alpha"},
                            {"label": "Beta (13-30 Hz)", "value": "beta"},
                            {"label": "Gamma (30-100 Hz)", "value": "gamma"},
                        ],
                        value="alpha",
                    ),
                ], md=12),
            ], className="mb-3"),
        ]

    return common_params + specific_params


def compute_sqi(signal_data, sqi_type, params, sampling_freq=100):
    """
    Compute Signal Quality Index using vitalDSP.

    Args:
        signal_data: Input signal array
        sqi_type: Type of SQI to compute
        params: Dictionary of parameters
        sampling_freq: Sampling frequency

    Returns:
        dict: SQI results including values, segments, and metrics
    """
    try:
        from vitalDSP.signal_quality_assessment.signal_quality_index import SignalQualityIndex

        # Initialize SQI
        sqi = SignalQualityIndex(signal_data)

        # Extract common parameters (handle None values)
        window_size = params.get("window_size")
        step_size = params.get("step_size")
        # Convert to int with safe defaults
        try:
            window_size = int(window_size) if window_size not in [None, "", 0] else 1000
        except (ValueError, TypeError):
            window_size = 1000
        try:
            step_size = int(step_size) if step_size not in [None, "", 0] else 500
        except (ValueError, TypeError):
            step_size = 500
        threshold_type = params.get("threshold_type", "above")
        scale = params.get("scale", "zscore")
        
        # Handle threshold based on threshold_type
        if threshold_type == "range":
            # For range type, threshold should be a list [min, max]
            try:
                threshold_min = params.get("threshold_min")
                threshold_max = params.get("threshold_max")
                # Handle None values
                threshold_min = float(threshold_min) if threshold_min not in [None, ""] else 0.2
                threshold_max = float(threshold_max) if threshold_max not in [None, ""] else 0.8
                threshold = [threshold_min, threshold_max]
                logger.info(f"Using range threshold: {threshold}")
            except (ValueError, TypeError):
                threshold = [0.2, 0.8]  # Default range
                logger.info(f"Using default range threshold: {threshold}")
        else:
            # For single threshold (above/below)
            try:
                threshold_val = params.get("threshold")
                # Handle None values
                if threshold_val in [None, ""]:
                    threshold = float(SQI_DEFAULT_THRESHOLDS.get(sqi_type, 0.5))
                else:
                    threshold = float(threshold_val)
                logger.info(f"Using single threshold: {threshold}")
            except (ValueError, TypeError):
                threshold = float(SQI_DEFAULT_THRESHOLDS.get(sqi_type, 0.5))
                logger.info(f"Using default threshold: {threshold}")

        # Call appropriate SQI method
        if sqi_type == "snr_sqi":
            sqi_values, normal_segments, abnormal_segments = sqi.snr_sqi(
                window_size=window_size,
                step_size=step_size,
                threshold=threshold,
                threshold_type=threshold_type,
                scale=scale,
            )

        elif sqi_type == "baseline_wander_sqi":
            try:
                moving_avg_window = params.get("moving_avg_window")
                moving_avg_window = int(moving_avg_window) if moving_avg_window not in [None, ""] else 50
            except (ValueError, TypeError):
                moving_avg_window = 50
            sqi_values, normal_segments, abnormal_segments = sqi.baseline_wander_sqi(
                window_size=window_size,
                step_size=step_size,
                threshold=threshold,
                threshold_type=threshold_type,
                scale=scale,
                moving_avg_window=moving_avg_window,
            )

        elif sqi_type == "amplitude_variability_sqi":
            sqi_values, normal_segments, abnormal_segments = sqi.amplitude_variability_sqi(
                window_size=window_size,
                step_size=step_size,
                threshold=threshold,
                threshold_type=threshold_type,
                scale=scale,
            )

        elif sqi_type == "zero_crossing_sqi":
            sqi_values, normal_segments, abnormal_segments = sqi.zero_crossing_sqi(
                window_size=window_size,
                step_size=step_size,
                threshold=threshold,
                threshold_type=threshold_type,
                scale=scale,
            )

        elif sqi_type == "waveform_similarity_sqi":
            try:
                reference_window_index = params.get("reference_window_index")
                reference_window_index = int(reference_window_index) if reference_window_index not in [None, ""] else 0
            except (ValueError, TypeError):
                reference_window_index = 0
            sqi_values, normal_segments, abnormal_segments = sqi.waveform_similarity_sqi(
                window_size=window_size,
                step_size=step_size,
                threshold=threshold,
                threshold_type=threshold_type,
                scale=scale,
                reference_window_index=reference_window_index,
            )

        elif sqi_type == "signal_entropy_sqi":
            # Note: signal_entropy_sqi doesn't accept num_bins parameter
            # It internally uses bins=10
            sqi_values, normal_segments, abnormal_segments = sqi.signal_entropy_sqi(
                window_size=window_size,
                step_size=step_size,
                threshold=threshold,
                threshold_type=threshold_type,
                scale=scale,
            )

        elif sqi_type == "energy_sqi":
            sqi_values, normal_segments, abnormal_segments = sqi.energy_sqi(
                window_size=window_size,
                step_size=step_size,
                threshold=threshold,
                threshold_type=threshold_type,
                scale=scale,
            )

        elif sqi_type == "kurtosis_sqi":
            sqi_values, normal_segments, abnormal_segments = sqi.kurtosis_sqi(
                window_size=window_size,
                step_size=step_size,
                threshold=threshold,
                threshold_type=threshold_type,
                scale=scale,
            )

        elif sqi_type == "skewness_sqi":
            sqi_values, normal_segments, abnormal_segments = sqi.skewness_sqi(
                window_size=window_size,
                step_size=step_size,
                threshold=threshold,
                threshold_type=threshold_type,
                scale=scale,
            )

        elif sqi_type == "peak_to_peak_amplitude_sqi":
            sqi_values, normal_segments, abnormal_segments = sqi.peak_to_peak_amplitude_sqi(
                window_size=window_size,
                step_size=step_size,
                threshold=threshold,
                threshold_type=threshold_type,
                scale=scale,
            )

        elif sqi_type == "ppg_signal_quality_sqi":
            # PPG SQI signature: (self, window_size, step_size, threshold=None, threshold_type='below', scale='zscore')
            sqi_values, normal_segments, abnormal_segments = sqi.ppg_signal_quality_sqi(
                window_size=window_size,
                step_size=step_size,
                threshold=threshold,
                threshold_type=threshold_type,
                scale=scale,
            )

        elif sqi_type == "respiratory_signal_quality_sqi":
            # Respiratory SQI signature: (self, window_size, step_size, threshold=None, threshold_type='below', scale='zscore')
            sqi_values, normal_segments, abnormal_segments = sqi.respiratory_signal_quality_sqi(
                window_size=window_size,
                step_size=step_size,
                threshold=threshold,
                threshold_type=threshold_type,
                scale=scale,
            )

        elif sqi_type == "heart_rate_variability_sqi":
            # HRV SQI signature: (self, rr_intervals, window_size, step_size, threshold=None, threshold_type='below', scale='zscore')
            # Need to compute RR intervals from signal first
            from vitalDSP.utils.config_utilities.common import find_peaks
            
            # Compute RR intervals from signal using peak detection
            # Find peaks with adaptive threshold based on mean and std
            mean_val = np.mean(signal_data)
            std_val = np.std(signal_data)
            height_threshold = mean_val + std_val
            min_distance = max(1, int(sampling_freq * 0.3))  # Minimum 0.3 seconds between peaks
            
            peaks = find_peaks(
                signal_data,
                height=height_threshold,
                distance=min_distance,
                prominence=0.5 * std_val,
            )
            
            if len(peaks) < 2:
                logger.warning("Not enough peaks detected for HRV analysis")
                raise ValueError("Not enough peaks detected for HRV analysis. Signal may be too short or noisy.")
            
            # Compute RR intervals (in seconds)
            rr_intervals = np.diff(peaks) / sampling_freq
            
            logger.info(f"Computed {len(rr_intervals)} RR intervals from {len(peaks)} peaks")
            
            sqi_values, normal_segments, abnormal_segments = sqi.heart_rate_variability_sqi(
                rr_intervals=rr_intervals,
                window_size=min(window_size, len(rr_intervals) // 2),  # Adjust window to fit RR intervals
                step_size=min(step_size, len(rr_intervals) // 4),      # Adjust step to fit RR intervals
                threshold=threshold,
                threshold_type=threshold_type,
                scale=scale,
            )

        elif sqi_type == "eeg_band_power_sqi":
            # EEG SQI signature: (self, band_power, window_size, step_size, threshold=None, threshold_type='below', scale='zscore')
            # Need to compute band power from signal first
            from vitalDSP.frequency_transform.spectral_analysis import compute_band_power
            
            band = params.get("band", "alpha")
            
            # Define EEG band frequencies
            band_frequencies = {
                "delta": (0.5, 4),
                "theta": (4, 8),
                "alpha": (8, 13),
                "beta": (13, 30),
                "gamma": (30, 100),
            }
            
            low_freq, high_freq = band_frequencies.get(band, (8, 13))
            
            # Compute band power for the signal
            band_power = compute_band_power(signal_data, sampling_freq, low_freq, high_freq)
            
            logger.info(f"Computed {band} band power: {np.mean(band_power):.3f}")
            
            sqi_values, normal_segments, abnormal_segments = sqi.eeg_band_power_sqi(
                band_power=band_power,
                window_size=min(window_size, len(band_power) // 2),  # Adjust window to fit band power
                step_size=min(step_size, len(band_power) // 4),      # Adjust step to fit band power
                threshold=threshold,
                threshold_type=threshold_type,
                scale=scale,
            )

        else:
            raise ValueError(f"Unknown SQI type: {sqi_type}")

        # Compute statistics
        mean_sqi = float(np.mean(sqi_values))
        std_sqi = float(np.std(sqi_values))
        min_sqi = float(np.min(sqi_values))
        max_sqi = float(np.max(sqi_values))

        # Calculate quality score (percentage of normal segments)
        total_segments = len(normal_segments) + len(abnormal_segments)
        quality_score = len(normal_segments) / total_segments if total_segments > 0 else 0

        # Determine overall quality
        if quality_score >= 0.8:
            overall_quality = "EXCELLENT"
        elif quality_score >= 0.6:
            overall_quality = "GOOD"
        elif quality_score >= 0.4:
            overall_quality = "FAIR"
        else:
            overall_quality = "POOR"

        results = {
            "sqi_type": sqi_type,
            "sqi_values": sqi_values.tolist() if isinstance(sqi_values, np.ndarray) else sqi_values,
            "normal_segments": normal_segments,
            "abnormal_segments": abnormal_segments,
            "mean_sqi": mean_sqi,
            "std_sqi": std_sqi,
            "min_sqi": min_sqi,
            "max_sqi": max_sqi,
            "quality_score": quality_score,
            "overall_quality": overall_quality,
            "threshold": threshold,
            "threshold_type": threshold_type,
            "num_segments": total_segments,
            "num_normal": len(normal_segments),
            "num_abnormal": len(abnormal_segments),
        }

        logger.info(f"SQI computed successfully: {sqi_type}, Score: {quality_score:.2%}")
        return results

    except Exception as e:
        logger.error(f"Error computing SQI {sqi_type}: {e}", exc_info=True)
        raise
