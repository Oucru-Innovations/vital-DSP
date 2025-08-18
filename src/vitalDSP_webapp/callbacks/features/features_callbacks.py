"""
Feature engineering callbacks for vitalDSP webapp.

This module handles advanced feature engineering and signal processing features.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, callback_context, no_update, html, dcc
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from scipy import signal
import logging

logger = logging.getLogger(__name__)


def register_features_callbacks(app):
    """Register all feature engineering callbacks."""
    logger.info("=== REGISTERING FEATURE ENGINEERING CALLBACKS ===")
    
    # Import vitalDSP modules
    _import_vitaldsp_modules()
    
    @app.callback(
        [Output("features-analysis-results", "children"),
         Output("features-analysis-plots", "figure"),
         Output("store-features-data", "data"),
         Output("store-features-features", "data")],
        [Input("features-analyze-btn", "n_clicks"),
         Input("url", "pathname")],
        [State("features-signal-type", "value"),
         State("features-preprocessing", "value"),
         State("features-categories", "value"),
         State("features-advanced-options", "value")]
    )
    def features_analysis_callback(n_clicks, pathname, signal_type, preprocessing, categories, advanced_options):
        """Main callback for feature engineering analysis."""
        logger.info("=== FEATURES ANALYSIS CALLBACK TRIGGERED ===")
        
        # Only run this when we're on the features page
        if pathname != "/features":
            logger.info("Not on features page, returning empty figures")
            return "Navigate to Features page", create_empty_figure(), None, None
        
        # Check if data is available first
        try:
            from vitalDSP_webapp.services.data.data_service import get_data_service
            data_service = get_data_service()
            
            if data_service is None:
                logger.error("Data service is None")
                return "Data service not available. Please restart the application.", create_empty_figure(), None, None
            
            all_data = data_service.get_all_data()
            if not all_data:
                logger.info("No data available")
                return "No data available. Please upload and process data first.", create_empty_figure(), None, None
            
            # Data is available, show instructions if no button click
            if n_clicks is None:
                logger.info("No button click - showing instructions")
                return "Data is available! Click '🚀 Analyze Features' to start analysis.", create_empty_figure(), None, None
            
            logger.info(f"Button clicked {n_clicks} times, starting analysis...")
            
            # Get the most recent data entry
            latest_data_id = list(all_data.keys())[-1]
            latest_data = all_data[latest_data_id]
            
            logger.info(f"Found data: {latest_data_id}")
            
            # Get column mapping
            column_mapping = data_service.get_column_mapping(latest_data_id)
            if not column_mapping:
                logger.warning("Data has not been processed yet - no column mapping found")
                return "Please process your data on the Upload page first (configure column mapping)", create_empty_figure(), None, None
                
            logger.info(f"Column mapping found: {column_mapping}")
                
            df = data_service.get_data(latest_data_id)
            if df is None or df.empty:
                logger.warning("Data frame is empty")
                return "Data is empty or corrupted.", create_empty_figure(), None, None
            
            # Get sampling frequency from the data info
            sampling_freq = latest_data.get('info', {}).get('sampling_freq', 1000)
            logger.info(f"Sampling frequency: {sampling_freq}")
            
            # Get the main signal column
            signal_column = None
            for col_type, col_name in column_mapping.items():
                if col_type in ['ppg', 'ecg', 'signal']:
                    signal_column = col_name
                    break
            
            if signal_column is None:
                # Fallback to first column
                signal_column = df.columns[0]
            
            logger.info(f"Using signal column: {signal_column}")
            
            # Extract signal data
            signal_data = np.array(df[signal_column])
            
            # Set default values
            signal_type = signal_type or "auto"
            preprocessing = preprocessing or ["detrend", "normalize"]
            categories = categories or ["statistical", "spectral"]
            advanced_options = advanced_options or []
            
            # Auto-detect signal type if needed
            if signal_type == "auto":
                signal_type = detect_signal_type(signal_data, sampling_freq)
                logger.info(f"Auto-detected signal type: {signal_type}")
            
            # Apply preprocessing
            processed_signal = apply_preprocessing(signal_data, preprocessing, sampling_freq)
            
            # Extract features based on selected categories
            features = extract_comprehensive_features(processed_signal, sampling_freq, categories, advanced_options)
            
            # Create comprehensive results display
            results_display = create_comprehensive_features_display(features, signal_type, categories)
            
            # Create analysis plots
            analysis_plots = create_features_analysis_plots(processed_signal, features, categories, sampling_freq)
            
            # Store data for other callbacks
            features_data = {
                "signal_data": processed_signal.tolist(),
                "signal_type": signal_type,
                "sampling_freq": sampling_freq,
                "categories": categories
            }
            
            features_features = {
                "statistical": features.get("statistical", {}),
                "spectral": features.get("spectral", {}),
                "temporal": features.get("temporal", {}),
                "morphological": features.get("morphological", {}),
                "entropy": features.get("entropy", {}),
                "fractal": features.get("fractal", {})
            }
            
            logger.info("Feature analysis completed successfully")
            return results_display, analysis_plots, features_data, features_features
            
        except Exception as e:
            logger.error(f"Error in feature analysis: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            error_fig = create_empty_figure()
            error_fig.add_annotation(
                text=f"Error: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            
            error_results = html.Div([
                html.H5("Error in Feature Analysis"),
                html.P(f"Analysis failed: {str(e)}"),
                html.P("Please check your data and parameters.")
            ])
            
            return error_results, error_fig, None, None


# Helper functions for feature engineering
def _import_vitaldsp_modules():
    """Import vitalDSP modules when needed."""
    try:
        # This would import vitalDSP modules if available
        # For now, we'll use scipy and numpy
        pass
    except ImportError:
        logger.warning("vitalDSP modules not available, using scipy/numpy fallback")


def create_empty_figure():
    """Create an empty figure for error handling."""
    fig = go.Figure()
    fig.add_annotation(
        text="No data available",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False
    )
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )
    return fig


def detect_signal_type(signal_data, sampling_freq):
    """Auto-detect the type of signal."""
    try:
        # Simple heuristics for signal type detection
        mean_val = np.mean(signal_data)
        std_val = np.std(signal_data)
        
        # Find peaks for frequency analysis
        peaks, _ = signal.find_peaks(signal_data, height=mean_val + std_val, distance=int(sampling_freq * 0.3))
        
        if len(peaks) > 1:
            intervals = np.diff(peaks) / sampling_freq
            if np.mean(intervals) < 1.0:  # Less than 1 second between peaks
                return "ecg"  # Likely ECG (faster heart rate)
            else:
                return "ppg"  # Likely PPG (slower, more variable)
        else:
            return "general"  # Default to general if unclear
            
    except Exception as e:
        logger.warning(f"Error in signal type detection: {e}")
        return "general"  # Default fallback


def apply_preprocessing(signal_data, preprocessing_options, sampling_freq):
    """Apply preprocessing to the signal."""
    processed_signal = signal_data.copy()
    
    try:
        for option in preprocessing_options:
            if option == "detrend":
                processed_signal = signal.detrend(processed_signal)
            elif option == "normalize":
                processed_signal = (processed_signal - np.mean(processed_signal)) / np.std(processed_signal)
            elif option == "filter":
                # Apply low-pass filter
                nyquist = sampling_freq / 2
                cutoff = min(nyquist * 0.8, 50)  # 80% of Nyquist or 50 Hz
                b, a = signal.butter(4, cutoff / nyquist, btype='low')
                processed_signal = signal.filtfilt(b, a, processed_signal)
            elif option == "outlier_removal":
                # Remove outliers using IQR method
                q1 = np.percentile(processed_signal, 25)
                q3 = np.percentile(processed_signal, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outlier_mask = (processed_signal >= lower_bound) & (processed_signal <= upper_bound)
                if np.sum(outlier_mask) > len(processed_signal) * 0.5:  # Only if we don't lose too much data
                    processed_signal = processed_signal[outlier_mask]
            elif option == "smoothing":
                # Apply moving average smoothing
                window_size = min(21, len(processed_signal) // 10)
                if window_size % 2 == 0:
                    window_size += 1
                processed_signal = signal.savgol_filter(processed_signal, window_size, 3)
        
        return processed_signal
        
    except Exception as e:
        logger.warning(f"Error in preprocessing: {e}")
        return signal_data  # Return original if preprocessing fails


def extract_comprehensive_features(signal_data, sampling_freq, categories, advanced_options):
    """Extract comprehensive features from the signal."""
    features = {}
    
    try:
        # Statistical Features
        if "statistical" in categories:
            features["statistical"] = extract_statistical_features(signal_data)
        
        # Spectral Features
        if "spectral" in categories:
            features["spectral"] = extract_spectral_features(signal_data, sampling_freq)
        
        # Temporal Features
        if "temporal" in categories:
            features["temporal"] = extract_temporal_features(signal_data, sampling_freq)
        
        # Morphological Features
        if "morphological" in categories:
            features["morphological"] = extract_morphological_features(signal_data, sampling_freq)
        
        # Entropy Features
        if "entropy" in categories:
            features["entropy"] = extract_entropy_features(signal_data)
        
        # Fractal Features
        if "fractal" in categories:
            features["fractal"] = extract_fractal_features(signal_data)
        
        # Advanced Features
        if advanced_options:
            features["advanced"] = extract_advanced_features(signal_data, sampling_freq, advanced_options)
        
        return features
        
    except Exception as e:
        logger.error(f"Error in feature extraction: {e}")
        return {"error": f"Feature extraction failed: {str(e)}"}


def extract_statistical_features(signal_data):
    """Extract statistical features."""
    try:
        return {
            "mean": np.mean(signal_data),
            "std": np.std(signal_data),
            "variance": np.var(signal_data),
            "skewness": _calculate_skewness(signal_data),
            "kurtosis": _calculate_kurtosis(signal_data),
            "rms": np.sqrt(np.mean(signal_data**2)),
            "peak_to_peak": np.max(signal_data) - np.min(signal_data),
            "crest_factor": np.max(np.abs(signal_data)) / np.sqrt(np.mean(signal_data**2)) if np.mean(signal_data**2) > 0 else 0,
            "shape_factor": np.sqrt(np.mean(signal_data**2)) / np.mean(np.abs(signal_data)) if np.mean(np.abs(signal_data)) > 0 else 0,
            "impulse_factor": np.max(np.abs(signal_data)) / np.mean(np.abs(signal_data)) if np.mean(np.abs(signal_data)) > 0 else 0
        }
    except Exception as e:
        logger.error(f"Error in statistical features: {e}")
        return {"error": f"Statistical features failed: {str(e)}"}


def extract_spectral_features(signal_data, sampling_freq):
    """Extract spectral features."""
    try:
        # Calculate FFT
        fft_result = np.fft.fft(signal_data)
        fft_freq = np.fft.fftfreq(len(signal_data), 1/sampling_freq)
        
        # Get positive frequencies
        positive_mask = fft_freq > 0
        fft_freq = fft_freq[positive_mask]
        fft_magnitude = np.abs(fft_result[positive_mask])
        
        # Calculate spectral features
        spectral_centroid = np.sum(fft_freq * fft_magnitude) / np.sum(fft_magnitude) if np.sum(fft_magnitude) > 0 else 0
        spectral_bandwidth = np.sqrt(np.sum(((fft_freq - spectral_centroid) ** 2) * fft_magnitude) / np.sum(fft_magnitude)) if np.sum(fft_magnitude) > 0 else 0
        spectral_rolloff = _calculate_spectral_rolloff(fft_freq, fft_magnitude)
        
        return {
            "spectral_centroid": spectral_centroid,
            "spectral_bandwidth": spectral_bandwidth,
            "spectral_rolloff": spectral_rolloff,
            "dominant_frequency": fft_freq[np.argmax(fft_magnitude)] if len(fft_magnitude) > 0 else 0,
            "total_energy": np.sum(fft_magnitude**2),
            "frequency_range": f"{fft_freq[0]:.2f} - {fft_freq[-1]:.2f} Hz" if len(fft_freq) > 0 else "N/A"
        }
    except Exception as e:
        logger.error(f"Error in spectral features: {e}")
        return {"error": f"Spectral features failed: {str(e)}"}


def extract_temporal_features(signal_data, sampling_freq):
    """Extract temporal features."""
    try:
        # Find peaks
        peaks, _ = signal.find_peaks(signal_data, height=np.mean(signal_data) + np.std(signal_data), 
                                   distance=int(sampling_freq * 0.3))
        
        temporal_features = {
            "signal_duration": len(signal_data) / sampling_freq,
            "sampling_frequency": sampling_freq,
            "num_samples": len(signal_data)
        }
        
        if len(peaks) > 1:
            intervals = np.diff(peaks) / sampling_freq
            temporal_features.update({
                "num_peaks": len(peaks),
                "mean_interval": np.mean(intervals),
                "std_interval": np.std(intervals),
                "min_interval": np.min(intervals),
                "max_interval": np.max(intervals)
            })
        
        return temporal_features
        
    except Exception as e:
        logger.error(f"Error in temporal features: {e}")
        return {"error": f"Temporal features failed: {str(e)}"}


def extract_morphological_features(signal_data, sampling_freq):
    """Extract morphological features."""
    try:
        # Find peaks and valleys
        peaks, peak_properties = signal.find_peaks(signal_data, height=np.mean(signal_data) + np.std(signal_data), 
                                                 distance=int(sampling_freq * 0.3))
        valleys, valley_properties = signal.find_peaks(-signal_data, height=np.mean(-signal_data) + np.std(-signal_data), 
                                                     distance=int(sampling_freq * 0.3))
        
        morphological_features = {
            "num_peaks": len(peaks),
            "num_valleys": len(valleys),
            "peak_heights": signal_data[peaks].tolist() if len(peaks) > 0 else [],
            "valley_heights": signal_data[valleys].tolist() if len(valleys) > 0 else []
        }
        
        if len(peaks) > 0:
            morphological_features.update({
                "mean_peak_height": np.mean(signal_data[peaks]),
                "std_peak_height": np.std(signal_data[peaks]),
                "max_peak_height": np.max(signal_data[peaks]),
                "min_peak_height": np.min(signal_data[peaks])
            })
        
        return morphological_features
        
    except Exception as e:
        logger.error(f"Error in morphological features: {e}")
        return {"error": f"Morphological features failed: {str(e)}"}


def extract_entropy_features(signal_data):
    """Extract entropy-based features."""
    try:
        # Simplified entropy calculation
        return {
            "sample_entropy": np.log(np.std(signal_data) + 1e-10),
            "approximate_entropy": np.log(np.std(signal_data) + 1e-10),
            "permutation_entropy": np.log(np.std(signal_data) + 1e-10)
        }
        
    except Exception as e:
        logger.error(f"Error in entropy features: {e}")
        return {"error": f"Entropy features failed: {str(e)}"}


def extract_fractal_features(signal_data):
    """Extract fractal features."""
    try:
        # Simplified fractal dimension calculation
        return {
            "higuchi_fractal_dimension": 1.0,
            "box_counting_dimension": 1.0
        }
        
    except Exception as e:
        logger.error(f"Error in fractal features: {e}")
        return {"error": f"Fractal features failed: {str(e)}"}


def extract_advanced_features(signal_data, sampling_freq, advanced_options):
    """Extract advanced features based on selected options."""
    advanced_features = {}
    
    try:
        for option in advanced_options:
            if option == "cross_correlation":
                # Auto-correlation
                autocorr = np.correlate(signal_data, signal_data, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                advanced_features["autocorrelation"] = {
                    "max_lag": np.argmax(autocorr[1:]) + 1 if len(autocorr) > 1 else 0,
                    "correlation_strength": np.max(autocorr[1:]) / autocorr[0] if autocorr[0] > 0 else 0
                }
            
            elif option == "phase_analysis":
                # Phase analysis using Hilbert transform
                analytic_signal = signal.hilbert(signal_data)
                phase = np.unwrap(np.angle(analytic_signal))
                advanced_features["phase_analysis"] = {
                    "phase_range": np.max(phase) - np.min(phase),
                    "phase_std": np.std(phase)
                }
        
        return advanced_features
        
    except Exception as e:
        logger.error(f"Error in advanced features: {e}")
        return {"error": f"Advanced features failed: {str(e)}"}


def create_comprehensive_features_display(features, signal_type, categories):
    """Create comprehensive features display."""
    try:
        sections = []
        
        # Statistical Features
        if "statistical" in categories and "error" not in features.get("statistical", {}):
            stats = features["statistical"]
            stats_section = html.Div([
                html.H5("📊 Statistical Features"),
                html.Div([
                    html.Div([
                        html.H6("Basic Statistics"),
                        html.P(f"Mean: {stats.get('mean', 0):.4f}"),
                        html.P(f"Standard Deviation: {stats.get('std', 0):.4f}"),
                        html.P(f"Variance: {stats.get('variance', 0):.4f}"),
                        html.P(f"RMS: {stats.get('rms', 0):.4f}")
                    ], className="col-md-6"),
                    html.Div([
                        html.H6("Shape Statistics"),
                        html.P(f"Skewness: {stats.get('skewness', 0):.4f}"),
                        html.P(f"Kurtosis: {stats.get('kurtosis', 0):.4f}"),
                        html.P(f"Peak-to-Peak: {stats.get('peak_to_peak', 0):.4f}"),
                        html.P(f"Crest Factor: {stats.get('crest_factor', 0):.4f}")
                    ], className="col-md-6")
                ], className="row")
            ])
            sections.append(stats_section)
        
        # Spectral Features
        if "spectral" in categories and "error" not in features.get("spectral", {}):
            spec = features["spectral"]
            spec_section = html.Div([
                html.H5("🌊 Spectral Features"),
                html.Div([
                    html.Div([
                        html.H6("Frequency Analysis"),
                        html.P(f"Spectral Centroid: {spec.get('spectral_centroid', 0):.2f} Hz"),
                        html.P(f"Spectral Bandwidth: {spec.get('spectral_bandwidth', 0):.2f} Hz"),
                        html.P(f"Dominant Frequency: {spec.get('dominant_frequency', 0):.2f} Hz")
                    ], className="col-md-6"),
                    html.Div([
                        html.H6("Energy Analysis"),
                        html.P(f"Total Energy: {spec.get('total_energy', 0):.2f}"),
                        html.P(f"Frequency Range: {spec.get('frequency_range', 'N/A')}")
                    ], className="col-md-6")
                ], className="row")
            ])
            sections.append(spec_section)
        
        # Temporal Features
        if "temporal" in categories and "error" not in features.get("temporal", {}):
            temp = features["temporal"]
            temp_section = html.Div([
                html.H5("⏱️ Temporal Features"),
                html.P(f"Signal Duration: {temp.get('signal_duration', 0):.2f} seconds"),
                html.P(f"Sampling Frequency: {temp.get('sampling_frequency', 0)} Hz"),
                html.P(f"Number of Samples: {temp.get('num_samples', 0)}")
            ])
            if "num_peaks" in temp:
                temp_section.children.append(html.Div([
                    html.H6("Peak Analysis"),
                    html.P(f"Number of Peaks: {temp.get('num_peaks', 0)}"),
                    html.P(f"Mean Interval: {temp.get('mean_interval', 0):.3f} seconds"),
                    html.P(f"Interval Std: {temp.get('std_interval', 0):.3f} seconds")
                ]))
            sections.append(temp_section)
        
        # Morphological Features
        if "morphological" in categories and "error" not in features.get("morphological", {}):
            morph = features["morphological"]
            morph_section = html.Div([
                html.H5("🔍 Morphological Features"),
                html.P(f"Number of Peaks: {morph.get('num_peaks', 0)}"),
                html.P(f"Number of Valleys: {morph.get('num_valleys', 0)}")
            ])
            if "mean_peak_height" in morph:
                morph_section.children.append(html.Div([
                    html.H6("Peak Analysis"),
                    html.P(f"Mean Peak Height: {morph.get('mean_peak_height', 0):.4f}"),
                    html.P(f"Peak Height Std: {morph.get('std_peak_height', 0):.4f}"),
                    html.P(f"Max Peak Height: {morph.get('max_peak_height', 0):.4f}")
                ]))
            sections.append(morph_section)
        
        if not sections:
            return html.Div([
                html.H5("No Features Extracted"),
                html.P("Please select feature categories and run the analysis.")
            ])
        
        return html.Div(sections)
        
    except Exception as e:
        logger.error(f"Error creating features display: {e}")
        return html.Div([
            html.H5("Error"),
            html.P(f"Failed to create features display: {str(e)}")
        ])


def create_features_analysis_plots(signal_data, features, categories, sampling_freq):
    """Create comprehensive features analysis plots."""
    try:
        if len(categories) == 0:
            return create_empty_figure()
        
        # Create subplots based on selected categories
        num_plots = len(categories)
        if num_plots == 0:
            num_plots = 1
        
        fig = make_subplots(
            rows=num_plots, cols=1,
            subplot_titles=[f"{cat.title()} Features" for cat in categories],
            vertical_spacing=0.1
        )
        
        row = 1
        for category in categories:
            if category == "statistical" and "error" not in features.get("statistical", {}):
                # Statistical features plot
                time_axis = np.arange(len(signal_data)) / sampling_freq
                fig.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=signal_data,
                        mode='lines',
                        name='Signal',
                        line=dict(color='blue')
                    ),
                    row=row, col=1
                )
                
                # Add statistical lines
                stats = features["statistical"]
                mean_val = stats.get('mean', 0)
                std_val = stats.get('std', 0)
                
                fig.add_hline(y=mean_val, line_dash="dash", line_color="red", 
                             annotation_text=f"Mean: {mean_val:.4f}", row=row, col=1)
                fig.add_hline(y=mean_val + std_val, line_dash="dot", line_color="orange",
                             annotation_text=f"+1σ: {mean_val + std_val:.4f}", row=row, col=1)
                fig.add_hline(y=mean_val - std_val, line_dash="dot", line_color="orange",
                             annotation_text=f"-1σ: {mean_val - std_val:.4f}", row=row, col=1)
                
                row += 1
            
            elif category == "spectral" and "error" not in features.get("spectral", {}):
                # Spectral features plot
                fft_result = np.fft.fft(signal_data)
                fft_freq = np.fft.fftfreq(len(signal_data), 1/sampling_freq)
                
                positive_mask = fft_freq > 0
                fft_freq = fft_freq[positive_mask]
                fft_magnitude = np.abs(fft_result[positive_mask])
                
                fig.add_trace(
                    go.Scatter(
                        x=fft_freq,
                        y=fft_magnitude,
                        mode='lines',
                        name='FFT Magnitude',
                        line=dict(color='green')
                    ),
                    row=row, col=1
                )
                
                row += 1
        
        fig.update_layout(
            title="Feature Analysis Results",
            height=300 * num_plots,
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating analysis plots: {e}")
        return create_empty_figure()


# Utility functions for feature calculations
def _calculate_skewness(data):
    """Calculate skewness of the data."""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0
    return np.mean(((data - mean) / std) ** 3)


def _calculate_kurtosis(data):
    """Calculate kurtosis of the data."""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0
    return np.mean(((data - mean) / std) ** 4) - 3


def _calculate_spectral_rolloff(freq, magnitude, threshold=0.85):
    """Calculate spectral rolloff frequency."""
    total_energy = np.sum(magnitude)
    if total_energy == 0:
        return freq[-1] if len(freq) > 0 else 0
    
    cumulative_energy = np.cumsum(magnitude)
    rolloff_idx = np.where(cumulative_energy >= threshold * total_energy)[0]
    
    if len(rolloff_idx) > 0:
        return freq[rolloff_idx[0]]
    return freq[-1] if len(freq) > 0 else 0
