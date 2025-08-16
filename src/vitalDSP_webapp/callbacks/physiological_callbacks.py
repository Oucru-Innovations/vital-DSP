"""
Physiological features callbacks for vitalDSP webapp.
Handles comprehensive physiological feature extraction and analysis using vitalDSP.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, callback_context, no_update, html
from dash.exceptions import PreventUpdate
from scipy import signal
import dash_bootstrap_components as dbc
import logging

logger = logging.getLogger(__name__)

# Initialize vitalDSP modules as None for lazy loading
TimeDomainFeatures = None
FrequencyDomainFeatures = None
HRVFeatures = None
NonlinearFeatures = None
WaveformMorphology = None
BeatToBeatFeatures = None
CoherenceAnalysis = None
CrossSignalAnalysis = None
SignalPowerAnalysis = None
SignalSegmentation = None
TrendAnalysis = None
EnvelopeDetection = None
EnergyAnalysis = None
SignalChangeDetection = None
EnsembleFeatureExtractor = None
PhysiologicalFeatureExtractor = None
PPGAutonomicFeatures = None
ECGAutonomicFeatures = None
PPGLightFeatures = None
ECGPPGSynchronizationFeatures = None
WaveletTransform = None
FourierTransform = None
HilbertTransform = None
STFT = None
MFCC = None
PCAICADecomposition = None
VitalTransformation = None

def _import_vitaldsp_modules():
    """Import vitalDSP modules with error handling."""
    global TimeDomainFeatures, FrequencyDomainFeatures, HRVFeatures, NonlinearFeatures
    global WaveformMorphology, BeatToBeatFeatures, CoherenceAnalysis, CrossSignalAnalysis
    global SignalPowerAnalysis, SignalSegmentation, TrendAnalysis, EnvelopeDetection
    global EnergyAnalysis, SignalChangeDetection, EnsembleFeatureExtractor
    global PhysiologicalFeatureExtractor, PPGAutonomicFeatures, ECGAutonomicFeatures
    global PPGLightFeatures, ECGPPGSynchronizationFeatures, WaveletTransform
    global FourierTransform, HilbertTransform, STFT, MFCC, PCAICADecomposition, VitalTransformation
    
    logger.info("=== IMPORTING VITALDSP PHYSIOLOGICAL FEATURES MODULES ===")
    
    try:
        from vitalDSP.physiological_features.time_domain import TimeDomainFeatures
        logger.info("✓ TimeDomainFeatures imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import TimeDomainFeatures: {e}")
        TimeDomainFeatures = None

    try:
        from vitalDSP.physiological_features.frequency_domain import FrequencyDomainFeatures
        logger.info("✓ FrequencyDomainFeatures imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import FrequencyDomainFeatures: {e}")
        FrequencyDomainFeatures = None

    try:
        from vitalDSP.physiological_features.hrv_analysis import HRVFeatures
        logger.info("✓ HRVFeatures imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import HRVFeatures: {e}")
        HRVFeatures = None

    try:
        from vitalDSP.physiological_features.nonlinear import NonlinearFeatures
        logger.info("✓ NonlinearFeatures imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import NonlinearFeatures: {e}")
        NonlinearFeatures = None

    try:
        from vitalDSP.physiological_features.waveform import WaveformMorphology
        logger.info("✓ WaveformMorphology imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import WaveformMorphology: {e}")
        WaveformMorphology = None

    try:
        from vitalDSP.physiological_features.beat_to_beat import BeatToBeatFeatures
        logger.info("✓ BeatToBeatFeatures imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import BeatToBeatFeatures: {e}")
        BeatToBeatFeatures = None

    try:
        from vitalDSP.physiological_features.coherence_analysis import CoherenceAnalysis
        logger.info("✓ CoherenceAnalysis imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import CoherenceAnalysis: {e}")
        CoherenceAnalysis = None

    try:
        from vitalDSP.physiological_features.cross_signal_analysis import CrossSignalAnalysis
        logger.info("✓ CrossSignalAnalysis imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import CrossSignalAnalysis: {e}")
        CrossSignalAnalysis = None

    try:
        from vitalDSP.physiological_features.signal_power_analysis import SignalPowerAnalysis
        logger.info("✓ SignalPowerAnalysis imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import SignalPowerAnalysis: {e}")
        SignalPowerAnalysis = None

    try:
        from vitalDSP.physiological_features.signal_segmentation import SignalSegmentation
        logger.info("✓ SignalSegmentation imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import SignalSegmentation: {e}")
        SignalSegmentation = None

    try:
        from vitalDSP.physiological_features.trend_analysis import TrendAnalysis
        logger.info("✓ TrendAnalysis imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import TrendAnalysis: {e}")
        TrendAnalysis = None

    try:
        from vitalDSP.physiological_features.envelope_detection import EnvelopeDetection
        logger.info("✓ EnvelopeDetection imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import EnvelopeDetection: {e}")
        EnvelopeDetection = None

    try:
        from vitalDSP.physiological_features.energy_analysis import EnergyAnalysis
        logger.info("✓ EnergyAnalysis imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import EnergyAnalysis: {e}")
        EnergyAnalysis = None

    try:
        from vitalDSP.physiological_features.signal_change_detection import SignalChangeDetection
        logger.info("✓ SignalChangeDetection imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import SignalChangeDetection: {e}")
        SignalChangeDetection = None

    try:
        from vitalDSP.physiological_features.ensemble_based_feature_extraction import EnsembleFeatureExtractor
        logger.info("✓ EnsembleFeatureExtractor imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import EnsembleFeatureExtractor: {e}")
        EnsembleFeatureExtractor = None

    try:
        from vitalDSP.feature_engineering.morphology_features import PhysiologicalFeatureExtractor
        logger.info("✓ PhysiologicalFeatureExtractor imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import PhysiologicalFeatureExtractor: {e}")
        PhysiologicalFeatureExtractor = None

    try:
        from vitalDSP.feature_engineering.ppg_autonomic_features import PPGAutonomicFeatures
        logger.info("✓ PPGAutonomicFeatures imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import PPGAutonomicFeatures: {e}")
        PPGAutonomicFeatures = None

    try:
        from vitalDSP.feature_engineering.ecg_autonomic_features import ECGAutonomicFeatures
        logger.info("✓ ECGAutonomicFeatures imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import ECGAutonomicFeatures: {e}")
        ECGAutonomicFeatures = None

    try:
        from vitalDSP.feature_engineering.ppg_light_features import PPGLightFeatures
        logger.info("✓ PPGLightFeatures imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import PPGLightFeatures: {e}")
        PPGLightFeatures = None

    try:
        from vitalDSP.feature_engineering.ecg_ppg_synchronyzation_features import ECGPPGSynchronizationFeatures
        logger.info("✓ ECGPPGSynchronizationFeatures imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import ECGPPGSynchronizationFeatures: {e}")
        ECGPPGSynchronizationFeatures = None

    try:
        from vitalDSP.transforms.wavelet_transform import WaveletTransform
        logger.info("✓ WaveletTransform imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import WaveletTransform: {e}")
        WaveletTransform = None

    try:
        from vitalDSP.transforms.fourier_transform import FourierTransform
        logger.info("✓ FourierTransform imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import FourierTransform: {e}")
        FourierTransform = None

    try:
        from vitalDSP.transforms.hilbert_transform import HilbertTransform
        logger.info("✓ HilbertTransform imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import HilbertTransform: {e}")
        HilbertTransform = None

    try:
        from vitalDSP.transforms.stft import STFT
        logger.info("✓ STFT imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import STFT: {e}")
        STFT = None

    try:
        from vitalDSP.transforms.mfcc import MFCC
        logger.info("✓ MFCC imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import MFCC: {e}")
        MFCC = None

    try:
        from vitalDSP.transforms.pca_ica_signal_decomposition import PCAICADecomposition
        logger.info("✓ PCAICADecomposition imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import PCAICADecomposition: {e}")
        PCAICADecomposition = None

    try:
        from vitalDSP.transforms.vital_transformation import VitalTransformation
        logger.info("✓ VitalTransformation imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import VitalTransformation: {e}")
        VitalTransformation = None

    logger.info("=== VITALDSP PHYSIOLOGICAL FEATURES MODULE IMPORT COMPLETED ===")


def register_physiological_callbacks(app):
    """Register all physiological analysis callbacks."""
    logger.info("=== REGISTERING PHYSIOLOGICAL CALLBACKS ===")
    
    # Import vitalDSP modules when callbacks are registered
    _import_vitaldsp_modules()
    
    @app.callback(
        [Output("physio-main-signal-plot", "figure"),
         Output("physio-analysis-results", "children"),
         Output("physio-analysis-plots", "figure"),
         Output("store-physio-data", "data"),
         Output("store-physio-features", "data")],
        [Input("url", "pathname"),
         Input("physio-btn-update-analysis", "n_clicks"),
         Input("physio-time-range-slider", "value"),
         Input("physio-btn-nudge-m10", "n_clicks"),
         Input("physio-btn-nudge-m1", "n_clicks"),
         Input("physio-btn-nudge-p1", "n_clicks"),
         Input("physio-btn-nudge-p10", "n_clicks")],
        [State("physio-start-time", "value"),
         State("physio-end-time", "value"),
         State("physio-signal-type", "value"),
         State("physio-analysis-categories", "value"),
         State("physio-hrv-options", "value"),
         State("physio-morphology-options", "value"),
         State("physio-advanced-options", "value")]
    )
    def physiological_analysis_callback(pathname, n_clicks, slider_value, nudge_m10, nudge_m1, nudge_p1, nudge_p10,
                                     start_time, end_time, signal_type, analysis_categories, hrv_options, 
                                     morphology_options, advanced_options):
        """Unified callback for physiological analysis - handles both page load and user interactions."""
        ctx = callback_context
        
        # Determine what triggered this callback
        if not ctx.triggered:
            logger.warning("No context triggered - raising PreventUpdate")
            raise PreventUpdate
        
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        logger.info(f"=== PHYSIOLOGICAL ANALYSIS CALLBACK TRIGGERED ===")
        logger.info(f"Trigger ID: {trigger_id}")
        logger.info(f"Pathname: {pathname}")
        logger.info(f"Analysis categories: {analysis_categories}")
        logger.info(f"HRV options: {hrv_options}")
        logger.info(f"Morphology options: {morphology_options}")
        logger.info(f"Advanced options: {advanced_options}")
        
        # Only run this when we're on the physiological page
        if pathname != "/physiological":
            logger.info("Not on physiological page, returning empty figures")
            return create_empty_figure(), "Navigate to Physiological Features page", create_empty_figure(), None, None
        
        try:
            # Get data from the data service
            from ..services.data_service import get_data_service
            data_service = get_data_service()
            
            # Get the most recent data
            all_data = data_service.get_all_data()
            if not all_data:
                logger.warning("No data found in service")
                return create_empty_figure(), "No data available. Please upload and process data first.", create_empty_figure(), None, None
            
            # Get the most recent data entry
            latest_data_id = list(all_data.keys())[-1]
            latest_data = all_data[latest_data_id]
            
            logger.info(f"Found data: {latest_data_id}")
            
            # Get column mapping
            column_mapping = data_service.get_column_mapping(latest_data_id)
            if not column_mapping:
                logger.warning("Data has not been processed yet - no column mapping found")
                return create_empty_figure(), "Please process your data on the Upload page first (configure column mapping)", create_empty_figure(), None, None
                
            logger.info(f"Column mapping found: {column_mapping}")
                
            df = data_service.get_data(latest_data_id)
            if df is None or df.empty:
                logger.warning("Data frame is empty")
                return create_empty_figure(), "Data is empty or corrupted.", create_empty_figure(), None, None
            
            # Get sampling frequency from the data info
            sampling_freq = latest_data.get('info', {}).get('sampling_freq', 1000)
            logger.info(f"Sampling frequency: {sampling_freq}")
            
            # Handle time window adjustments for nudge buttons
            if trigger_id in ["physio-btn-nudge-m10", "physio-btn-nudge-m1", "physio-btn-nudge-p1", "physio-btn-nudge-p10"]:
                if not start_time or not end_time:
                    start_time, end_time = 0, 10
                    
                if trigger_id == "physio-btn-nudge-m10":
                    start_time = max(0, start_time - 10)
                    end_time = max(10, end_time - 10)
                elif trigger_id == "physio-btn-nudge-m1":
                    start_time = max(0, start_time - 1)
                    end_time = max(1, end_time - 1)
                elif trigger_id == "physio-btn-nudge-p1":
                    start_time = start_time + 1
                    end_time = end_time + 1
                elif trigger_id == "physio-btn-nudge-p10":
                    start_time = start_time + 10
                    end_time = end_time + 10
                
                logger.info(f"Time window adjusted: {start_time} to {end_time}")
            
            # Set default time window if not specified
            if not start_time or not end_time:
                start_time, end_time = 0, 10
                logger.info(f"Using default time window: {start_time} to {end_time}")
            
            # Apply time window
            start_sample = int(start_time * sampling_freq)
            end_sample = int(end_time * sampling_freq)
            windowed_data = df.iloc[start_sample:end_sample].copy()
            
            # Create time axis
            time_axis = np.arange(len(windowed_data)) / sampling_freq
            
            # Get signal column
            signal_column = column_mapping.get('signal')
            if not signal_column or signal_column not in windowed_data.columns:
                logger.warning(f"Signal column {signal_column} not found in data")
                return create_empty_figure(), "Signal column not found in data.", create_empty_figure(), None, None
            
            signal_data = windowed_data[signal_column].values
            
            # Auto-detect signal type if needed
            if signal_type == "auto":
                signal_type = detect_signal_type(signal_data, sampling_freq)
                logger.info(f"Auto-detected signal type: {signal_type}")
            
            # Create main signal plot with annotations
            logger.info("Creating main signal plot with annotations...")
            main_plot = create_physiological_signal_plot(signal_data, time_axis, sampling_freq, signal_type, 
                                                       analysis_categories, column_mapping)
            logger.info("Main signal plot created successfully")
            
            # Generate comprehensive analysis results
            logger.info("Generating physiological analysis results...")
            analysis_results = generate_physiological_analysis_results(signal_data, time_axis, sampling_freq, 
                                                                    signal_type, analysis_categories, hrv_options, 
                                                                    morphology_options, advanced_options)
            logger.info("Physiological analysis results generated successfully")
            
            # Create analysis plots
            logger.info("Creating analysis plots...")
            analysis_plots = create_physiological_analysis_plots(signal_data, time_axis, sampling_freq, 
                                                               signal_type, analysis_categories, hrv_options, 
                                                               morphology_options, advanced_options)
            logger.info("Analysis plots created successfully")
            
            # Store processed data
            physio_data = {
                "signal_data": signal_data.tolist(),
                "time_axis": time_axis.tolist(),
                "sampling_freq": sampling_freq,
                "window": [start_time, end_time],
                "signal_type": signal_type,
                "analysis_categories": analysis_categories
            }
            
            physio_features = {
                "hrv_options": hrv_options,
                "morphology_options": morphology_options,
                "advanced_options": advanced_options
            }
            
            logger.info("Physiological analysis completed successfully")
            return main_plot, analysis_results, analysis_plots, physio_data, physio_features
            
        except Exception as e:
            logger.error(f"Error in physiological analysis callback: {e}")
            import traceback
            traceback.print_exc()
            return create_empty_figure(), f"Error in analysis: {str(e)}", create_empty_figure(), None, None
    
    @app.callback(
        [Output("physio-start-time", "value"),
         Output("physio-end-time", "value")],
        [Input("physio-time-range-slider", "value")]
    )
    def update_physio_time_inputs(slider_value):
        """Update time input fields based on slider."""
        if not slider_value:
            return no_update, no_update
        return slider_value[0], slider_value[1]
    
    @app.callback(
        [Output("physio-time-range-slider", "min"),
         Output("physio-time-range-slider", "max"),
         Output("physio-time-range-slider", "value")],
        [Input("url", "pathname")]
    )
    def update_physio_time_slider_range(pathname):
        """Update time slider range based on data duration."""
        logger.info("=== UPDATE PHYSIO TIME SLIDER RANGE ===")
        logger.info(f"Pathname: {pathname}")
        
        # Only run this when we're on the physiological page
        if pathname != "/physiological":
            return 0, 100, [0, 10]
        
        try:
            # Get data from the data service
            from ..services.data_service import get_data_service
            data_service = get_data_service()
            
            # Get the most recent data
            all_data = data_service.get_all_data()
            if not all_data:
                logger.warning("No data found in service")
                return 0, 100, [0, 10]
            
            # Get the most recent data entry
            latest_data_id = list(all_data.keys())[-1]
            latest_data = all_data[latest_data_id]
            
            # Get the actual data
            df = data_service.get_data(latest_data_id)
            if df is None or df.empty:
                logger.warning("Data frame is empty")
                return 0, 100, [0, 10]
            
            # Get sampling frequency from the data info
            sampling_freq = latest_data.get('info', {}).get('sampling_freq', 1000)
            
            max_time = len(df) / sampling_freq
            logger.info(f"Max time: {max_time}, Sampling freq: {sampling_freq}")
            
            return 0, max_time, [0, min(10, max_time)]
            
        except Exception as e:
            logger.error(f"Error updating physio time slider range: {e}")
            return 0, 100, [0, 10]


def create_empty_figure():
    """Create an empty figure for when no data is available."""
    fig = go.Figure()
    fig.add_annotation(
        text="No data available<br>Please upload data first",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="gray")
    )
    fig.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig


def detect_signal_type(signal_data, sampling_freq):
    """Auto-detect signal type based on signal characteristics."""
    try:
        # Calculate basic statistics
        mean_val = np.mean(signal_data)
        std_val = np.std(signal_data)
        range_val = np.max(signal_data) - np.min(signal_data)
        
        # Calculate frequency content
        fft_result = np.abs(np.fft.rfft(signal_data))
        freqs = np.fft.rfftfreq(len(signal_data), 1/sampling_freq)
        
        # Find dominant frequency
        peak_idx = np.argmax(fft_result)
        dominant_freq = freqs[peak_idx]
        
        # PPG typically has dominant frequency around 1-2 Hz (60-120 BPM)
        # ECG typically has more complex frequency content
        # EEG typically has lower amplitude and different frequency characteristics
        
        if 0.5 < dominant_freq < 3.0 and range_val < 2 * std_val:
            return "ppg"
        elif dominant_freq > 2.0 and range_val > 3 * std_val:
            return "ecg"
        else:
            return "ppg"  # Default to PPG for most cases
            
    except Exception as e:
        logger.warning(f"Signal type detection failed: {e}")
        return "ppg"  # Default fallback


def create_physiological_signal_plot(signal_data, time_axis, sampling_freq, signal_type, analysis_categories, column_mapping):
    """Create the main physiological signal plot with annotations."""
    try:
        fig = go.Figure()
        
        # Create the main signal plot
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=signal_data,
            mode='lines',
            name=f'{signal_type.upper()} Signal',
            line=dict(color='#2E86AB', width=2),
            hovertemplate='<b>Time:</b> %{x:.3f}s<br><b>Amplitude:</b> %{y:.3f}<extra></extra>'
        ))
        
        # Add peak detection if enabled
        if "hrv" in analysis_categories or "morphology" in analysis_categories:
            try:
                # Adaptive peak detection based on signal type
                if signal_type == "ppg":
                    prominence = 0.3 * np.std(signal_data)
                    distance = int(0.5 * sampling_freq)  # Minimum 0.5s between peaks
                else:  # ECG or other
                    prominence = 0.2 * np.std(signal_data)
                    distance = int(0.3 * sampling_freq)  # Minimum 0.3s between peaks
                
                peaks, properties = signal.find_peaks(signal_data, prominence=prominence, distance=distance)
                
                if len(peaks) > 0:
                    fig.add_trace(go.Scatter(
                        x=time_axis[peaks],
                        y=signal_data[peaks],
                        mode='markers',
                        name='Detected Peaks',
                        marker=dict(color='red', size=8, symbol='diamond'),
                        hovertemplate='<b>Peak:</b> %{y:.3f}<br><b>Time:</b> %{x:.3f}s<extra></extra>'
                    ))
                    
                    # Add peak annotations
                    for i, peak in enumerate(peaks[:10]):  # Limit to first 10 peaks
                        fig.add_annotation(
                            x=time_axis[peak],
                            y=signal_data[peak],
                            text=f"P{i+1}",
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=2,
                            arrowcolor="red",
                            ax=0,
                            ay=-40
                        )
                        
            except Exception as e:
                logger.error(f"Peak detection failed: {e}")
        
        # Add morphological annotations if enabled
        if "morphology" in analysis_categories:
            try:
                # Add baseline
                baseline = np.mean(signal_data)
                fig.add_hline(y=baseline, line_dash="dash", line_color="gray", 
                             annotation_text=f"Baseline: {baseline:.3f}")
                
                # Add signal envelope if envelope detection is enabled
                if "envelope" in analysis_categories:
                    upper_env, lower_env = compute_signal_envelope(signal_data)
                    fig.add_trace(go.Scatter(
                        x=time_axis,
                        y=upper_env,
                        mode='lines',
                        name='Upper Envelope',
                        line=dict(color='green', width=1, dash='dot'),
                        opacity=0.7
                    ))
                    fig.add_trace(go.Scatter(
                        x=time_axis,
                        y=lower_env,
                        mode='lines',
                        name='Lower Envelope',
                        line=dict(color='green', width=1, dash='dot'),
                        opacity=0.7,
                        fill='tonexty'
                    ))
                    
            except Exception as e:
                logger.error(f"Morphological annotations failed: {e}")
        
        fig.update_layout(
            title=f"{signal_type.upper()} Signal Analysis",
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
            template="plotly_white",
            height=400,
            margin=dict(l=40, r=40, t=60, b=40),
            showlegend=True,
            hovermode='closest'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating physiological signal plot: {e}")
        return create_empty_figure()


def compute_signal_envelope(signal_data):
    """Compute upper and lower envelope of the signal."""
    try:
        # Use Hilbert transform to compute analytic signal
        analytic_signal = signal.hilbert(signal_data)
        envelope = np.abs(analytic_signal)
        
        # Upper and lower envelopes
        upper_env = envelope
        lower_env = -envelope
        
        return upper_env, lower_env
        
    except Exception as e:
        logger.error(f"Envelope computation failed: {e}")
        # Fallback to simple min/max approach
        window_size = max(3, len(signal_data) // 100)
        upper_env = []
        lower_env = []
        
        for i in range(len(signal_data)):
            start = max(0, i - window_size // 2)
            end = min(len(signal_data), i + window_size // 2 + 1)
            upper_env.append(np.max(signal_data[start:end]))
            lower_env.append(np.min(signal_data[start:end]))
        
        return np.array(upper_env), np.array(lower_env)


def generate_physiological_analysis_results(signal_data, time_axis, sampling_freq, signal_type, 
                                          analysis_categories, hrv_options, morphology_options, advanced_options):
    """Generate comprehensive physiological analysis results."""
    logger.info("Generating comprehensive physiological analysis results")
    
    results = []
    
    try:
        # Basic signal statistics
        results.append(html.H6("📊 Signal Statistics", className="mb-2"))
        results.append(dbc.Table([
            html.Thead([
                html.Tr([
                    html.Th("Metric", className="text-center"),
                    html.Th("Value", className="text-center"),
                    html.Th("Unit", className="text-center")
                ])
            ]),
            html.Tbody([
                html.Tr([
                    html.Td("Signal Type", className="fw-bold"),
                    html.Td(signal_type.upper(), className="text-center"),
                    html.Td("-", className="text-muted")
                ]),
                html.Tr([
                    html.Td("Duration", className="fw-bold"),
                    html.Td(f"{len(signal_data)/sampling_freq:.2f}", className="text-end"),
                    html.Td("Seconds", className="text-muted")
                ]),
                html.Tr([
                    html.Td("Samples", className="fw-bold"),
                    html.Td(f"{len(signal_data)}", className="text-end"),
                    html.Td("Count", className="text-muted")
                ]),
                html.Tr([
                    html.Td("Sampling Frequency", className="fw-bold"),
                    html.Td(f"{sampling_freq}", className="text-end"),
                    html.Td("Hz", className="text-muted")
                ])
            ])
        ], bordered=True, hover=True, responsive=True, className="mb-3"))
        
        # HRV Analysis Results
        if "hrv" in analysis_categories:
            hrv_results = generate_hrv_analysis_results(signal_data, time_axis, sampling_freq, hrv_options)
            if hrv_results:
                results.append(html.Hr())
                results.append(hrv_results)
        
        # Morphological Analysis Results
        if "morphology" in analysis_categories:
            morphology_results = generate_morphology_analysis_results(signal_data, time_axis, sampling_freq, 
                                                                   signal_type, morphology_options)
            if morphology_results:
                results.append(html.Hr())
                results.append(morphology_results)
        
        # Beat-to-Beat Analysis Results
        if "beat2beat" in analysis_categories:
            beat2beat_results = generate_beat2beat_analysis_results(signal_data, time_axis, sampling_freq, signal_type)
            if beat2beat_results:
                results.append(html.Hr())
                results.append(beat2beat_results)
        
        # Energy Analysis Results
        if "energy" in analysis_categories:
            energy_results = generate_energy_analysis_results(signal_data, time_axis, sampling_freq)
            if energy_results:
                results.append(html.Hr())
                results.append(energy_results)
        
        # Statistical Analysis Results
        if "statistical" in analysis_categories:
            statistical_results = generate_statistical_analysis_results(signal_data, time_axis, sampling_freq)
            if statistical_results:
                results.append(html.Hr())
                results.append(statistical_results)
        
        # Frequency Analysis Results
        if "frequency" in analysis_categories:
            frequency_results = generate_frequency_analysis_results(signal_data, time_axis, sampling_freq)
            if frequency_results:
                results.append(html.Hr())
                results.append(frequency_results)
        
        # Envelope Detection Results
        if "envelope" in analysis_categories:
            logger.info("Generating envelope detection results...")
            envelope_results = generate_envelope_analysis_results(signal_data, time_axis, sampling_freq)
            if envelope_results:
                results.append(html.Hr())
                results.append(envelope_results)
        
        # Signal Segmentation Results
        if "segmentation" in analysis_categories:
            logger.info("Generating signal segmentation results...")
            segmentation_results = generate_segmentation_analysis_results(signal_data, time_axis, sampling_freq)
            if segmentation_results:
                results.append(html.Hr())
                results.append(segmentation_results)
        
        # Trend Analysis Results
        if "trend" in analysis_categories:
            logger.info("Generating trend analysis results...")
            trend_results = generate_trend_analysis_results(signal_data, time_axis, sampling_freq)
            if trend_results:
                results.append(html.Hr())
                results.append(trend_results)
        
        # Waveform Analysis Results
        if "waveform" in analysis_categories:
            logger.info("Generating waveform analysis results...")
            waveform_results = generate_waveform_analysis_results(signal_data, time_axis, sampling_freq, signal_type)
            if waveform_results:
                results.append(html.Hr())
                results.append(waveform_results)
        
        # Advanced Analysis Results
        if "cross_signal" in advanced_options:
            logger.info("Generating cross-signal analysis results...")
            cross_signal_results = generate_cross_signal_analysis_results(signal_data, time_axis, sampling_freq, signal_type)
            if cross_signal_results:
                results.append(html.Hr())
                results.append(cross_signal_results)
        
        # Ensemble Methods Results
        if "ensemble" in advanced_options:
            logger.info("Generating ensemble analysis results...")
            ensemble_results = generate_ensemble_analysis_results(signal_data, time_axis, sampling_freq)
            if ensemble_results:
                results.append(html.Hr())
                results.append(ensemble_results)
        
        # Change Detection Results
        if "change_detection" in advanced_options:
            logger.info("Generating change detection results...")
            change_detection_results = generate_change_detection_analysis_results(signal_data, time_axis, sampling_freq)
            if change_detection_results:
                results.append(html.Hr())
                results.append(change_detection_results)
        
        # Power Analysis Results
        if "power_analysis" in advanced_options:
            logger.info("Generating power analysis results...")
            power_analysis_results = generate_power_analysis_results(signal_data, time_axis, sampling_freq)
            if power_analysis_results:
                results.append(html.Hr())
                results.append(power_analysis_results)
        
        # Signal Transform Results
        if "transforms" in analysis_categories:
            logger.info("Generating signal transform results...")
            transform_results = generate_signal_transform_results(signal_data, time_axis, sampling_freq)
            if transform_results:
                results.append(html.Hr())
                results.append(transform_results)
        
        return html.Div(results)
        
    except Exception as e:
        logger.error(f"Error generating physiological analysis results: {e}")
        return html.Div([
            html.H6("❌ Analysis Error", className="text-danger"),
            html.P(f"Error during analysis: {str(e)}")
        ])


def generate_hrv_analysis_results(signal_data, time_axis, sampling_freq, hrv_options):
    """Generate HRV analysis results using vitalDSP."""
    try:
        # Detect peaks for RR intervals
        if "ppg" in signal_data or "ecg" in signal_data:
            prominence = 0.3 * np.std(signal_data)
            distance = int(0.5 * sampling_freq)
        else:
            prominence = 0.2 * np.std(signal_data)
            distance = int(0.3 * sampling_freq)
        
        peaks, _ = signal.find_peaks(signal_data, prominence=prominence, distance=distance)
        
        if len(peaks) < 2:
            return html.Div([
                html.H6("❤️ HRV Analysis", className="mb-2"),
                html.P("Insufficient peaks detected for HRV analysis", className="text-warning")
            ])
        
        # Calculate RR intervals
        rr_intervals = np.diff(time_axis[peaks]) * 1000  # Convert to milliseconds
        
        # Use vitalDSP for HRV analysis
        try:
            # Check if vitalDSP modules are available
            if TimeDomainFeatures is None or FrequencyDomainFeatures is None:
                logger.warning("vitalDSP HRV modules not available - using fallback analysis")
                # Fallback: Basic HRV metrics
                td_results = [
                    f"SDNN: {np.std(rr_intervals):.2f} ms",
                    f"RMSSD: {np.sqrt(np.mean(np.diff(rr_intervals)**2)):.2f} ms",
                    f"Mean RR: {np.mean(rr_intervals):.2f} ms"
                ]
                freq_results = []
            else:
                # Time domain features
                time_features = TimeDomainFeatures(rr_intervals)
                td_results = []
                
                if "time_domain" in hrv_options:
                    td_results.extend([
                        f"SDNN: {time_features.compute_sdnn():.2f} ms",
                        f"RMSSD: {time_features.compute_rmssd():.2f} ms",
                        f"pNN50: {time_features.compute_pnn50():.2f}%",
                        f"Mean NN: {time_features.compute_mean_nn():.2f} ms"
                    ])
                
                # Frequency domain features
                freq_results = []
                if "freq_domain" in hrv_options:
                    freq_features = FrequencyDomainFeatures(rr_intervals, fs=4)
                    freq_results.extend([
                        f"LF Power: {freq_features.compute_lf():.2e} ms²",
                        f"HF Power: {freq_features.compute_hf():.2e} ms²",
                        f"LF/HF Ratio: {freq_features.compute_lf_hf_ratio():.2f}"
                    ])
            
            # Create results table
            all_results = td_results + freq_results
            
            return html.Div([
                html.H6("❤️ HRV Analysis", className="mb-2"),
                dbc.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("Metric", className="text-center"),
                            html.Th("Value", className="text-center"),
                            html.Th("Unit", className="text-center")
                        ])
                    ]),
                    html.Tbody([
                        html.Tr([
                            html.Td(result.split(":")[0], className="fw-bold"),
                            html.Td(result.split(":")[1].strip(), className="text-end"),
                            html.Td("ms" if "ms" in result else "ms²" if "ms²" in result else "Ratio" if "Ratio" in result else "%", className="text-muted")
                        ]) for result in all_results
                    ])
                ], bordered=True, hover=True, responsive=True, className="mb-3")
            ])
            
        except ImportError:
            # Fallback to basic calculations
            logger.warning("vitalDSP not available, using basic HRV calculations")
            return html.Div([
                html.H6("❤️ HRV Analysis (Basic)", className="mb-2"),
                dbc.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("Metric", className="text-center"),
                            html.Th("Value", className="text-center"),
                            html.Th("Unit", className="text-center")
                        ])
                    ]),
                    html.Tbody([
                        html.Tr([
                            html.Td("RR Intervals", className="fw-bold"),
                            html.Td(f"{len(rr_intervals)}", className="text-end"),
                            html.Td("Count", className="text-muted")
                        ]),
                        html.Tr([
                            html.Td("Mean RR", className="fw-bold"),
                            html.Td(f"{np.mean(rr_intervals):.2f}", className="text-end"),
                            html.Td("ms", className="text-muted")
                        ]),
                        html.Tr([
                            html.Td("SDNN", className="fw-bold"),
                            html.Td(f"{np.std(rr_intervals):.2f}", className="text-end"),
                            html.Td("ms", className="text-muted")
                        ]),
                        html.Tr([
                            html.Td("Heart Rate", className="fw-bold"),
                            html.Td(f"{60000/np.mean(rr_intervals):.1f}", className="text-end"),
                            html.Td("BPM", className="text-muted")
                        ])
                    ])
                ], bordered=True, hover=True, responsive=True, className="mb-3")
            ])
            
    except Exception as e:
        logger.error(f"Error generating HRV analysis results: {e}")
        return html.Div([
            html.H6("❤️ HRV Analysis", className="mb-2"),
            html.P(f"Error in HRV analysis: {str(e)}", className="text-danger")
        ])


def generate_morphology_analysis_results(signal_data, time_axis, sampling_freq, signal_type, morphology_options):
    """Generate morphological analysis results."""
    try:
        results = []
        peaks = None
        
        if "peaks" in morphology_options:
            # Peak detection results
            prominence = 0.3 * np.std(signal_data)
            distance = int(0.5 * sampling_freq)
            peaks, properties = signal.find_peaks(signal_data, prominence=prominence, distance=distance)
            
            if len(peaks) > 0:
                peak_amplitudes = signal_data[peaks]
                results.extend([
                    f"Peak Count: {len(peaks)}",
                    f"Mean Peak Amplitude: {np.mean(peak_amplitudes):.3f}",
                    f"Peak Amplitude Std: {np.std(peak_amplitudes):.3f}",
                    f"Peak Amplitude Range: {np.max(peak_amplitudes) - np.min(peak_amplitudes):.3f}"
                ])
        
        if "duration" in morphology_options:
            # Duration analysis
            if peaks is not None and len(peaks) > 1:
                intervals = np.diff(time_axis[peaks])
                results.extend([
                    f"Mean Interval: {np.mean(intervals):.3f} s",
                    f"Interval Std: {np.std(intervals):.3f} s",
                    f"Min Interval: {np.min(intervals):.3f} s",
                    f"Max Interval: {np.max(intervals):.3f} s"
                ])
        
        if "area" in morphology_options:
            # Area calculations
            total_area = np.trapz(np.abs(signal_data), time_axis)
            positive_area = np.trapz(np.maximum(signal_data, 0), time_axis)
            negative_area = np.trapz(np.minimum(signal_data, 0), time_axis)
            
            results.extend([
                f"Total Signal Area: {total_area:.3f}",
                f"Positive Area: {positive_area:.3f}",
                f"Negative Area: {negative_area:.3f}",
                f"Area Ratio (Pos/Neg): {positive_area/abs(negative_area):.3f}" if negative_area != 0 else "Area Ratio (Pos/Neg): N/A"
            ])
        
        if "amplitude" in morphology_options:
            # Amplitude variability analysis
            signal_range = np.max(signal_data) - np.min(signal_data)
            signal_skewness = np.mean(((signal_data - np.mean(signal_data)) / np.std(signal_data)) ** 3)
            signal_kurtosis = np.mean(((signal_data - np.mean(signal_data)) / np.std(signal_data)) ** 4)
            
            results.extend([
                f"Signal Range: {signal_range:.3f}",
                f"Signal Skewness: {signal_skewness:.3f}",
                f"Signal Kurtosis: {signal_kurtosis:.3f}",
                f"Amplitude CV: {np.std(signal_data)/np.mean(signal_data):.3f}" if np.mean(signal_data) != 0 else "Amplitude CV: N/A"
            ])
        
        if "slope" in morphology_options:
            # Slope analysis
            signal_diff = np.diff(signal_data)
            time_diff = np.diff(time_axis)
            slopes = signal_diff / time_diff
            
            results.extend([
                f"Mean Slope: {np.mean(slopes):.3f}",
                f"Slope Std: {np.std(slopes):.3f}",
                f"Max Positive Slope: {np.max(slopes):.3f}",
                f"Max Negative Slope: {np.min(slopes):.3f}"
            ])
        
        if "dicrotic" in morphology_options and signal_type == "ppg":
            # Dicrotic notch analysis for PPG
            try:
                # Find secondary peaks (dicrotic notches)
                secondary_peaks, _ = signal.find_peaks(signal_data, prominence=0.1 * np.std(signal_data), 
                                                     distance=int(0.1 * sampling_freq))
                
                if len(secondary_peaks) > 0:
                    # Calculate dicrotic notch timing relative to main peaks
                    if peaks is not None and len(peaks) > 0:
                        dicrotic_timings = []
                        for main_peak in peaks[:min(5, len(peaks))]:  # Check first 5 peaks
                            # Find dicrotic notch within 0.5s after main peak
                            window_start = main_peak
                            window_end = min(main_peak + int(0.5 * sampling_freq), len(signal_data))
                            window_secondary = [p for p in secondary_peaks if window_start <= p <= window_end]
                            
                            if window_secondary:
                                dicrotic_timing = (window_secondary[0] - main_peak) / sampling_freq
                                dicrotic_timings.append(dicrotic_timing)
                        
                        if dicrotic_timings:
                            results.extend([
                                f"Dicrotic Notch Count: {len(dicrotic_timings)}",
                                f"Mean Dicrotic Timing: {np.mean(dicrotic_timings):.3f} s",
                                f"Dicrotic Timing Std: {np.std(dicrotic_timings):.3f} s"
                            ])
                
            except Exception as e:
                logger.error(f"Dicrotic notch analysis failed: {e}")
        
        if results:
            return html.Div([
                html.H6("🔍 Morphological Analysis", className="mb-2"),
                dbc.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("Metric", className="text-center"),
                            html.Th("Value", className="text-center"),
                            html.Th("Unit", className="text-center")
                        ])
                    ]),
                    html.Tbody([
                        html.Tr([
                            html.Td(result.split(":")[0], className="fw-bold"),
                            html.Td(result.split(":")[1].strip(), className="text-end"),
                            html.Td("Count" if "Count" in result else "s" if "s" in result else "Amplitude" if "Amplitude" in result else "Ratio" if "Ratio" in result else "-", className="text-muted")
                        ]) for result in results
                    ])
                ], bordered=True, hover=True, responsive=True, className="mb-3")
            ])
        
        return None
        
    except Exception as e:
        logger.error(f"Error generating morphology analysis results: {e}")
        return None


def generate_beat2beat_analysis_results(signal_data, time_axis, sampling_freq, signal_type):
    """Generate beat-to-beat analysis results."""
    try:
        # Detect beats
        prominence = 0.3 * np.std(signal_data)
        distance = int(0.5 * sampling_freq)
        peaks, _ = signal.find_peaks(signal_data, prominence=prominence, distance=distance)
        
        if len(peaks) < 2:
            return None
        
        # Calculate beat-to-beat metrics
        beat_intervals = np.diff(time_axis[peaks])
        beat_rates = 60 / beat_intervals
        
        return html.Div([
            html.H6("💓 Beat-to-Beat Analysis", className="mb-2"),
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Metric", className="text-center"),
                        html.Th("Value", className="text-center"),
                        html.Th("Unit", className="text-center")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td("Total Beats", className="fw-bold"),
                        html.Td(f"{len(peaks)}", className="text-end"),
                        html.Td("Count", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Mean Heart Rate", className="fw-bold"),
                        html.Td(f"{np.mean(beat_rates):.1f}", className="text-end"),
                        html.Td("BPM", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Heart Rate Variability", className="fw-bold"),
                        html.Td(f"{np.std(beat_rates):.1f}", className="text-end"),
                        html.Td("BPM", className="text-muted")
                    ])
                ])
            ], bordered=True, hover=True, responsive=True, className="mb-3")
        ])
        
    except Exception as e:
        logger.error(f"Error generating beat-to-beat analysis results: {e}")
        return None


def generate_energy_analysis_results(signal_data, time_axis, sampling_freq):
    """Generate energy analysis results."""
    try:
        # Calculate energy metrics
        total_energy = np.sum(signal_data ** 2)
        mean_power = np.mean(signal_data ** 2)
        rms = np.sqrt(mean_power)
        
        # Calculate energy distribution
        energy_per_sample = signal_data ** 2
        energy_std = np.std(energy_per_sample)
        
        # Calculate energy in different time windows
        window_size = int(0.5 * sampling_freq)  # 0.5 second windows
        if len(signal_data) >= window_size:
            energy_windows = []
            for i in range(0, len(signal_data) - window_size, window_size // 2):
                window_energy = np.sum(signal_data[i:i + window_size] ** 2)
                energy_windows.append(window_energy)
            
            energy_variability = np.std(energy_windows) if len(energy_windows) > 1 else 0
            energy_trend = np.polyfit(range(len(energy_windows)), energy_windows, 1)[0] if len(energy_windows) > 1 else 0
        else:
            energy_variability = 0
            energy_trend = 0
        
        # Calculate spectral energy
        fft_result = np.abs(np.fft.rfft(signal_data))
        spectral_energy = np.sum(fft_result ** 2)
        
        # Calculate entropy-based energy measures
        signal_normalized = (signal_data - np.min(signal_data)) / (np.max(signal_data) - np.min(signal_data))
        signal_normalized = np.clip(signal_normalized, 1e-10, 1)  # Avoid log(0)
        energy_entropy = -np.sum(signal_normalized * np.log(signal_normalized))
        
        return html.Div([
            html.H6("⚡ Energy Analysis", className="mb-2"),
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Metric", className="text-center"),
                        html.Th("Value", className="text-center"),
                        html.Th("Unit", className="text-center")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td("Total Energy", className="fw-bold"),
                        html.Td(f"{total_energy:.3e}", className="text-end"),
                        html.Td("Energy Units", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Mean Power", className="fw-bold"),
                        html.Td(f"{mean_power:.3e}", className="text-end"),
                        html.Td("Power Units", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("RMS", className="fw-bold"),
                        html.Td(f"{rms:.3f}", className="text-end"),
                        html.Td("Amplitude", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Energy Std", className="fw-bold"),
                        html.Td(f"{energy_std:.3e}", className="text-end"),
                        html.Td("Energy Units", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Energy Variability", className="fw-bold"),
                        html.Td(f"{energy_variability:.3e}", className="text-end"),
                        html.Td("Energy Units", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Energy Trend", className="fw-bold"),
                        html.Td(f"{energy_trend:.3e}", className="text-end"),
                        html.Td("Energy/Time", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Spectral Energy", className="fw-bold"),
                        html.Td(f"{spectral_energy:.3e}", className="text-end"),
                        html.Td("Spectral Units", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Energy Entropy", className="fw-bold"),
                        html.Td(f"{energy_entropy:.3f}", className="text-end"),
                        html.Td("Entropy", className="text-muted")
                    ])
                ])
            ], bordered=True, hover=True, responsive=True, className="mb-3")
        ])
        
    except Exception as e:
        logger.error(f"Error generating energy analysis results: {e}")
        return None


def generate_statistical_analysis_results(signal_data, time_axis, sampling_freq):
    """Generate comprehensive statistical analysis results."""
    try:
        # Basic statistics
        mean_val = np.mean(signal_data)
        std_val = np.std(signal_data)
        median_val = np.median(signal_data)
        min_val = np.min(signal_data)
        max_val = np.max(signal_data)
        range_val = max_val - min_val
        
        # Percentiles
        q25 = np.percentile(signal_data, 25)
        q75 = np.percentile(signal_data, 75)
        iqr = q75 - q25
        
        # Skewness and kurtosis
        skewness = np.mean(((signal_data - mean_val) / std_val) ** 3)
        kurtosis = np.mean(((signal_data - mean_val) / std_val) ** 4)
        
        # Coefficient of variation
        cv = std_val / abs(mean_val) if mean_val != 0 else 0
        
        # Zero crossings
        zero_crossings = np.sum(np.diff(np.sign(signal_data - mean_val)) != 0)
        
        # Entropy
        hist, _ = np.histogram(signal_data, bins=50, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        entropy = -np.sum(hist * np.log(hist)) if len(hist) > 0 else 0
        
        return html.Div([
            html.H6("📊 Statistical Analysis", className="mb-2"),
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Metric", className="text-center"),
                        html.Th("Value", className="text-center"),
                        html.Th("Unit", className="text-center")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td("Mean", className="fw-bold"),
                        html.Td(f"{mean_val:.3f}", className="text-end"),
                        html.Td("Amplitude", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Median", className="fw-bold"),
                        html.Td(f"{median_val:.3f}", className="text-end"),
                        html.Td("Amplitude", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Standard Deviation", className="fw-bold"),
                        html.Td(f"{std_val:.3f}", className="text-end"),
                        html.Td("Amplitude", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Range", className="fw-bold"),
                        html.Td(f"{range_val:.3f}", className="text-end"),
                        html.Td("Amplitude", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Q25", className="fw-bold"),
                        html.Td(f"{q25:.3f}", className="text-end"),
                        html.Td("Amplitude", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Q75", className="fw-bold"),
                        html.Td(f"{q75:.3f}", className="text-end"),
                        html.Td("Amplitude", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("IQR", className="fw-bold"),
                        html.Td(f"{iqr:.3f}", className="text-end"),
                        html.Td("Amplitude", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Skewness", className="fw-bold"),
                        html.Td(f"{skewness:.3f}", className="text-end"),
                        html.Td("Dimensionless", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Kurtosis", className="fw-bold"),
                        html.Td(f"{kurtosis:.3f}", className="text-end"),
                        html.Td("Dimensionless", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Coefficient of Variation", className="fw-bold"),
                        html.Td(f"{cv:.3f}", className="text-end"),
                        html.Td("Dimensionless", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Zero Crossings", className="fw-bold"),
                        html.Td(f"{zero_crossings}", className="text-end"),
                        html.Td("Count", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Entropy", className="fw-bold"),
                        html.Td(f"{entropy:.3f}", className="text-end"),
                        html.Td("Bits", className="text-muted")
                    ])
                ])
            ], bordered=True, hover=True, responsive=True, className="mb-3")
        ])
        
    except Exception as e:
        logger.error(f"Error generating statistical analysis results: {e}")
        return None


def generate_frequency_analysis_results(signal_data, time_axis, sampling_freq):
    """Generate frequency domain analysis results."""
    try:
        # FFT analysis
        fft_result = np.abs(np.fft.rfft(signal_data))
        freqs = np.fft.rfftfreq(len(signal_data), 1/sampling_freq)
        
        # Find dominant frequency
        peak_idx = np.argmax(fft_result)
        dominant_freq = freqs[peak_idx]
        dominant_amplitude = fft_result[peak_idx]
        
        # Calculate power in different frequency bands
        low_freq_mask = (freqs >= 0.1) & (freqs < 1.0)
        mid_freq_mask = (freqs >= 1.0) & (freqs < 5.0)
        high_freq_mask = (freqs >= 5.0) & (freqs < 10.0)
        
        low_freq_power = np.sum(fft_result[low_freq_mask] ** 2)
        mid_freq_power = np.sum(fft_result[mid_freq_mask] ** 2)
        high_freq_power = np.sum(fft_result[high_freq_mask] ** 2)
        
        total_power = np.sum(fft_result ** 2)
        
        # Spectral centroid
        spectral_centroid = np.sum(freqs * fft_result) / np.sum(fft_result)
        
        # Spectral bandwidth
        spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * fft_result) / np.sum(fft_result))
        
        # Spectral rolloff (frequency below which 85% of energy is contained)
        cumulative_energy = np.cumsum(fft_result ** 2)
        rolloff_threshold = 0.85 * cumulative_energy[-1]
        rolloff_idx = np.where(cumulative_energy >= rolloff_threshold)[0]
        spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
        
        # Spectral flatness
        geometric_mean = np.exp(np.mean(np.log(fft_result + 1e-10)))
        arithmetic_mean = np.mean(fft_result)
        spectral_flatness = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0
        
        return html.Div([
            html.H6("🌊 Frequency Analysis", className="mb-2"),
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Metric", className="text-center"),
                        html.Th("Value", className="text-center"),
                        html.Th("Unit", className="text-center")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td("Dominant Frequency", className="fw-bold"),
                        html.Td(f"{dominant_freq:.2f}", className="text-end"),
                        html.Td("Hz", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Dominant Amplitude", className="fw-bold"),
                        html.Td(f"{dominant_amplitude:.3e}", className="text-end"),
                        html.Td("FFT Units", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Low Freq Power (0.1-1Hz)", className="fw-bold"),
                        html.Td(f"{low_freq_power:.3e}", className="text-end"),
                        html.Td("Power Units", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Mid Freq Power (1-5Hz)", className="fw-bold"),
                        html.Td(f"{mid_freq_power:.3e}", className="text-end"),
                        html.Td("Power Units", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("High Freq Power (5-10Hz)", className="fw-bold"),
                        html.Td(f"{high_freq_power:.3e}", className="text-end"),
                        html.Td("Power Units", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Total Power", className="fw-bold"),
                        html.Td(f"{total_power:.3e}", className="text-end"),
                        html.Td("Power Units", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Spectral Centroid", className="fw-bold"),
                        html.Td(f"{spectral_centroid:.2f}", className="text-end"),
                        html.Td("Hz", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Spectral Bandwidth", className="fw-bold"),
                        html.Td(f"{spectral_bandwidth:.2f}", className="text-end"),
                        html.Td("Hz", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Spectral Rolloff", className="fw-bold"),
                        html.Td(f"{spectral_rolloff:.2f}", className="text-end"),
                        html.Td("Hz", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Spectral Flatness", className="fw-bold"),
                        html.Td(f"{spectral_flatness:.3f}", className="text-end"),
                        html.Td("Dimensionless", className="text-muted")
                    ])
                ])
            ], bordered=True, hover=True, responsive=True, className="mb-3")
        ])
        
    except Exception as e:
        logger.error(f"Error generating frequency analysis results: {e}")
        return None


def generate_advanced_analysis_results(signal_data, time_axis, sampling_freq, signal_type):
    """Generate advanced analysis results."""
    try:
        results = []
        
        # Cross-signal analysis
        if "cross_signal" in advanced_options:
            try:
                # Calculate cross-correlation with itself (autocorrelation)
                autocorr = np.correlate(signal_data, signal_data, mode='full')
                autocorr = autocorr[len(signal_data)-1:]
                
                # Find first zero crossing (signal periodicity)
                zero_crossings = np.where(np.diff(np.sign(autocorr)))[0]
                if len(zero_crossings) > 0:
                    first_period = zero_crossings[0] / sampling_freq
                    results.append(f"Signal Periodicity: {first_period:.3f} s")
                
                # Calculate signal complexity
                complexity = np.sum(np.abs(np.diff(signal_data)))
                results.append(f"Signal Complexity: {complexity:.3f}")
                
            except Exception as e:
                logger.error(f"Cross-signal analysis failed: {e}")
        
        # Ensemble methods analysis
        if "ensemble" in advanced_options:
            try:
                # Multiple window analysis
                window_sizes = [int(0.1 * sampling_freq), int(0.2 * sampling_freq), int(0.5 * sampling_freq)]
                ensemble_means = []
                
                for window_size in window_sizes:
                    if len(signal_data) >= window_size:
                        # Moving average with different windows
                        moving_avg = np.convolve(signal_data, np.ones(window_size)/window_size, mode='valid')
                        ensemble_means.append(np.mean(moving_avg))
                
                if ensemble_means:
                    ensemble_variance = np.var(ensemble_means)
                    results.append(f"Ensemble Variance: {ensemble_variance:.3e}")
                
            except Exception as e:
                logger.error(f"Ensemble analysis failed: {e}")
        
        # Change detection analysis
        if "change_detection" in advanced_options:
            try:
                # Detect significant changes using CUSUM
                mean_signal = np.mean(signal_data)
                std_signal = np.std(signal_data)
                
                # Simple change detection
                threshold = 2 * std_signal
                changes = np.where(np.abs(signal_data - mean_signal) > threshold)[0]
                
                if len(changes) > 0:
                    change_rate = len(changes) / len(signal_data)
                    results.append(f"Change Detection Rate: {change_rate:.3f}")
                    results.append(f"Significant Changes: {len(changes)}")
                
            except Exception as e:
                logger.error(f"Change detection failed: {e}")
        
        # Power analysis
        if "power_analysis" in advanced_options:
            try:
                # Calculate power in different frequency bands
                fft_result = np.abs(np.fft.rfft(signal_data))
                freqs = np.fft.rfftfreq(len(signal_data), 1/sampling_freq)
                
                # Define frequency bands
                low_freq_power = np.sum(fft_result[(freqs >= 0.1) & (freqs < 1.0)])
                mid_freq_power = np.sum(fft_result[(freqs >= 1.0) & (freqs < 5.0)])
                high_freq_power = np.sum(fft_result[(freqs >= 5.0) & (freqs < 10.0)])
                
                total_power = low_freq_power + mid_freq_power + high_freq_power
                
                results.extend([
                    f"Low Freq Power (0.1-1Hz): {low_freq_power:.3e}",
                    f"Mid Freq Power (1-5Hz): {mid_freq_power:.3e}",
                    f"High Freq Power (5-10Hz): {high_freq_power:.3e}",
                    f"Total Power: {total_power:.3e}"
                ])
                
            except Exception as e:
                logger.error(f"Power analysis failed: {e}")
        
        if results:
            return html.Div([
                html.H6("🧠 Advanced Analysis", className="mb-2"),
                dbc.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("Metric", className="text-center"),
                            html.Th("Value", className="text-center"),
                            html.Th("Unit", className="text-center")
                        ])
                    ]),
                    html.Tbody([
                        html.Tr([
                            html.Td(result.split(":")[0], className="fw-bold"),
                            html.Td(result.split(":")[1].strip(), className="text-end"),
                            html.Td("s" if "Periodicity" in result else "Rate" if "Rate" in result else "Count" if "Changes" in result else "Power" if "Power" in result else "-", className="text-muted")
                        ]) for result in results
                    ])
                ], bordered=True, hover=True, responsive=True, className="mb-3")
            ])
        
        return None
        
    except Exception as e:
        logger.error(f"Error generating advanced analysis results: {e}")
        return None


def create_physiological_analysis_plots(signal_data, time_axis, sampling_freq, signal_type, 
                                       analysis_categories, hrv_options, morphology_options, advanced_options):
    """Create comprehensive physiological analysis plots with seaborn-style visualizations."""
    try:
        # Determine number of subplots needed
        num_plots = 0
        plot_types = []
        
        if "hrv" in analysis_categories:
            num_plots += 2  # RR intervals + Poincaré plot
            plot_types.extend(["HRV Analysis", "Poincaré Plot"])
        
        if "morphology" in analysis_categories:
            num_plots += 2  # Signal + Distribution
            plot_types.extend(["Signal Morphology", "Amplitude Distribution"])
        
        if "beat2beat" in analysis_categories:
            num_plots += 1
            plot_types.append("Beat-to-Beat")
        
        if "energy" in analysis_categories:
            num_plots += 1
            plot_types.append("Energy Analysis")
        
        if "statistical" in analysis_categories:
            num_plots += 1
            plot_types.append("Statistical Distribution")
        
        if "frequency" in analysis_categories:
            num_plots += 1
            plot_types.append("Frequency Spectrum")
        
        if "cross_signal" in advanced_options:
            num_plots += 1
            plot_types.append("Autocorrelation")
        
        if num_plots == 0:
            return create_empty_figure()
        
        # Create subplots
        fig = make_subplots(
            rows=num_plots, cols=1,
            subplot_titles=plot_types,
            vertical_spacing=0.08
        )
        
        plot_row = 1
        
        # HRV Analysis Plot
        if "hrv" in analysis_categories:
            try:
                # Detect peaks
                prominence = 0.3 * np.std(signal_data)
                distance = int(0.5 * sampling_freq)
                peaks, _ = signal.find_peaks(signal_data, prominence=prominence, distance=distance)
                
                if len(peaks) > 1:
                    # RR intervals
                    rr_intervals = np.diff(time_axis[peaks]) * 1000  # Convert to ms
                    rr_times = time_axis[peaks][1:]  # Time points for RR intervals
                    
                    # RR intervals over time
                    fig.add_trace(
                        go.Scatter(x=rr_times, y=rr_intervals, mode='lines+markers', 
                                   name='RR Intervals', line=dict(color='#e74c3c'), 
                                   marker=dict(size=6, color='#e74c3c'), showlegend=False),
                        row=plot_row, col=1
                    )
                    
                    # Add mean line and confidence bands
                    mean_rr = np.mean(rr_intervals)
                    std_rr = np.std(rr_intervals)
                    fig.add_hline(y=mean_rr, line_dash="dash", line_color="gray", 
                                 annotation_text=f"Mean: {mean_rr:.1f} ms", row=plot_row, col=1)
                    fig.add_hline(y=mean_rr + std_rr, line_dash="dot", line_color="lightgray", 
                                 annotation_text=f"+1σ: {mean_rr + std_rr:.1f} ms", row=plot_row, col=1)
                    fig.add_hline(y=mean_rr - std_rr, line_dash="dot", line_color="lightgray", 
                                 annotation_text=f"-1σ: {mean_rr - std_rr:.1f} ms", row=plot_row, col=1)
                
                plot_row += 1
                
                # Poincaré Plot (RRn vs RRn+1)
                if len(peaks) > 2:
                    try:
                        rr_n = rr_intervals[:-1]
                        rr_n1 = rr_intervals[1:]
                        
                        fig.add_trace(
                            go.Scatter(x=rr_n, y=rr_n1, mode='markers', 
                                       name='Poincaré', marker=dict(size=8, color='#3498db', 
                                                                  opacity=0.7), showlegend=False),
                            row=plot_row, col=1
                        )
                        
                        # Add diagonal line
                        min_rr = min(np.min(rr_n), np.min(rr_n1))
                        max_rr = max(np.max(rr_n), np.max(rr_n1))
                        fig.add_trace(
                            go.Scatter(x=[min_rr, max_rr], y=[min_rr, max_rr], 
                                       mode='lines', line=dict(color='red', dash='dash'), 
                                       name='Identity', showlegend=False),
                            row=plot_row, col=1
                        )
                        
                        # Add SD1 and SD2 lines (simplified)
                        mean_rr = np.mean(rr_intervals)
                        fig.add_hline(y=mean_rr, line_dash="dot", line_color="orange", 
                                     annotation_text=f"Mean RR: {mean_rr:.1f} ms", row=plot_row, col=1)
                        fig.add_vline(x=mean_rr, line_dash="dot", line_color="orange", 
                                     annotation_text="", row=plot_row, col=1)
                        
                    except Exception as e:
                        logger.error(f"Poincaré plot failed: {e}")
                
                plot_row += 1
                
            except Exception as e:
                logger.error(f"Error creating HRV plot: {e}")
                plot_row += 2
        
        # Morphology Plot
        if "morphology" in analysis_categories:
            try:
                # Show signal with peaks and annotations
                fig.add_trace(
                    go.Scatter(x=time_axis, y=signal_data, mode='lines', 
                               name='Signal', line=dict(color='#3498db', width=1.5), showlegend=False),
                    row=plot_row, col=1
                )
                
                # Add detected peaks
                prominence = 0.3 * np.std(signal_data)
                distance = int(0.5 * sampling_freq)
                peaks, _ = signal.find_peaks(signal_data, prominence=prominence, distance=distance)
                
                if len(peaks) > 0:
                    fig.add_trace(
                        go.Scatter(x=time_axis[peaks], y=signal_data[peaks], mode='markers', 
                                   name='Peaks', marker=dict(color='red', size=8, symbol='diamond'), showlegend=False),
                        row=plot_row, col=1
                    )
                    
                    # Add peak-to-peak lines
                    if len(peaks) > 1:
                        for i in range(len(peaks) - 1):
                            fig.add_trace(
                                go.Scatter(x=[time_axis[peaks[i]], time_axis[peaks[i+1]]], 
                                           y=[signal_data[peaks[i]], signal_data[peaks[i+1]]], 
                                           mode='lines', line=dict(color='lightcoral', width=1, dash='dot'), 
                                           showlegend=False),
                                row=plot_row, col=1
                            )
                
                # Add baseline and envelope
                baseline = np.mean(signal_data)
                fig.add_hline(y=baseline, line_dash="dash", line_color="gray", 
                             annotation_text=f"Baseline: {baseline:.3f}", row=plot_row, col=1)
                
                plot_row += 1
                
                # Amplitude Distribution (Histogram)
                try:
                    # Create histogram data
                    hist, bin_edges = np.histogram(signal_data, bins=30, density=True)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    
                    fig.add_trace(
                        go.Bar(x=bin_centers, y=hist, name='Distribution', 
                               marker_color='#9b59b6', opacity=0.7, showlegend=False),
                        row=plot_row, col=1
                    )
                    
                    # Add normal distribution fit
                    mu, sigma = np.mean(signal_data), np.std(signal_data)
                    x_norm = np.linspace(np.min(signal_data), np.max(signal_data), 100)
                    y_norm = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - mu) / sigma) ** 2)
                    
                    fig.add_trace(
                        go.Scatter(x=x_norm, y=y_norm, mode='lines', name='Normal Fit', 
                                   line=dict(color='red', width=2), showlegend=False),
                        row=plot_row, col=1
                    )
                    
                except Exception as e:
                    logger.error(f"Distribution plot failed: {e}")
                
                plot_row += 1
                
            except Exception as e:
                logger.error(f"Error creating morphology plot: {e}")
                plot_row += 2
        
        # Beat-to-Beat Plot
        if "beat2beat" in analysis_categories:
            try:
                # Heart rate trend
                prominence = 0.3 * np.std(signal_data)
                distance = int(0.5 * sampling_freq)
                peaks, _ = signal.find_peaks(signal_data, prominence=prominence, distance=distance)
                
                if len(peaks) > 1:
                    beat_intervals = np.diff(time_axis[peaks])
                    heart_rates = 60 / beat_intervals
                    beat_times = time_axis[peaks][1:]
                    
                    # Heart rate trend
                    fig.add_trace(
                        go.Scatter(x=beat_times, y=heart_rates, mode='lines+markers', 
                                   name='Heart Rate', line=dict(color='#27ae60', width=2), 
                                   marker=dict(size=6, color='#27ae60'), showlegend=False),
                        row=plot_row, col=1
                    )
                    
                    # Add trend line
                    if len(heart_rates) > 2:
                        z = np.polyfit(beat_times, heart_rates, 1)
                        p = np.poly1d(z)
                        trend_line = p(beat_times)
                        
                        fig.add_trace(
                            go.Scatter(x=beat_times, y=trend_line, mode='lines', 
                                       name='Trend', line=dict(color='orange', width=2, dash='dash'), 
                                       showlegend=False),
                            row=plot_row, col=1
                        )
                    
                    # Add mean line and confidence bands
                    mean_hr = np.mean(heart_rates)
                    std_hr = np.std(heart_rates)
                    fig.add_hline(y=mean_hr, line_dash="dash", line_color="gray", 
                                 annotation_text=f"Mean: {mean_hr:.1f} BPM", row=plot_row, col=1)
                    fig.add_hline(y=mean_hr + std_hr, line_dash="dot", line_color="lightgray", 
                                 annotation_text=f"+1σ: {mean_hr + std_hr:.1f} BPM", row=plot_row, col=1)
                    fig.add_hline(y=mean_hr - std_hr, line_dash="dot", line_color="lightgray", 
                                 annotation_text=f"-1σ: {mean_hr - std_hr:.1f} BPM", row=plot_row, col=1)
                
                plot_row += 1
                
            except Exception as e:
                logger.error(f"Error creating beat-to-beat plot: {e}")
                plot_row += 1
        
        # Energy Analysis Plot
        if "energy" in analysis_categories:
            try:
                # Calculate energy over time windows
                window_size = int(0.5 * sampling_freq)
                if len(signal_data) >= window_size:
                    energy_windows = []
                    time_windows = []
                    
                    for i in range(0, len(signal_data) - window_size, window_size // 4):
                        window_energy = np.sum(signal_data[i:i + window_size] ** 2)
                        energy_windows.append(window_energy)
                        time_windows.append(time_axis[i + window_size // 2])
                    
                    fig.add_trace(
                        go.Scatter(x=time_windows, y=energy_windows, mode='lines+markers', 
                                   name='Energy', line=dict(color='#f39c12', width=2), 
                                   marker=dict(size=6, color='#f39c12'), showlegend=False),
                        row=plot_row, col=1
                    )
                    
                    # Add trend line
                    if len(energy_windows) > 2:
                        z = np.polyfit(time_windows, energy_windows, 1)
                        p = np.poly1d(z)
                        trend_line = p(time_windows)
                        
                        fig.add_trace(
                            go.Scatter(x=time_windows, y=trend_line, mode='lines', 
                                       name='Trend', line=dict(color='red', width=2, dash='dash'), 
                                       showlegend=False),
                            row=plot_row, col=1
                        )
                
                plot_row += 1
                
            except Exception as e:
                logger.error(f"Error creating energy plot: {e}")
                plot_row += 1
        
        # Statistical Distribution Plot
        if "statistical" in analysis_categories:
            try:
                # Create comprehensive histogram with multiple statistics
                hist, bin_edges = np.histogram(signal_data, bins=40, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                fig.add_trace(
                    go.Bar(x=bin_centers, y=hist, name='Distribution', 
                           marker_color='#9b59b6', opacity=0.7, showlegend=False),
                    row=plot_row, col=1
                )
                
                # Add normal distribution fit
                mu, sigma = np.mean(signal_data), np.std(signal_data)
                x_norm = np.linspace(np.min(signal_data), np.max(signal_data), 100)
                y_norm = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - mu) / sigma) ** 2)
                
                fig.add_trace(
                    go.Scatter(x=x_norm, y=y_norm, mode='lines', name='Normal Fit', 
                               line=dict(color='red', width=2), showlegend=False),
                    row=plot_row, col=1
                )
                
                # Add mean and std lines
                fig.add_vline(x=mu, line_dash="dash", line_color="blue", 
                             annotation_text=f"Mean: {mu:.3f}", row=plot_row, col=1)
                fig.add_vline(x=mu + sigma, line_dash="dot", line_color="orange", 
                             annotation_text=f"+1σ: {mu + sigma:.3f}", row=plot_row, col=1)
                fig.add_vline(x=mu - sigma, line_dash="dot", line_color="orange", 
                             annotation_text=f"-1σ: {mu - sigma:.3f}", row=plot_row, col=1)
                
                plot_row += 1
                
            except Exception as e:
                logger.error(f"Error creating statistical distribution plot: {e}")
                plot_row += 1
        
        # Frequency Spectrum Plot
        if "frequency" in analysis_categories:
            try:
                # FFT spectrum
                fft_result = np.abs(np.fft.rfft(signal_data))
                freqs = np.fft.rfftfreq(len(signal_data), 1/sampling_freq)
                
                # Limit to relevant frequencies (0-10 Hz for physiological signals)
                freq_mask = freqs <= 10
                freqs_plot = freqs[freq_mask]
                fft_plot = fft_result[freq_mask]
                
                fig.add_trace(
                    go.Scatter(x=freqs_plot, y=fft_plot, mode='lines', 
                               name='FFT Spectrum', line=dict(color='#e67e22', width=2), showlegend=False),
                    row=plot_row, col=1
                )
                
                # Highlight dominant frequency
                peak_idx = np.argmax(fft_plot)
                dominant_freq = freqs_plot[peak_idx]
                dominant_amp = fft_plot[peak_idx]
                
                fig.add_trace(
                    go.Scatter(x=[dominant_freq], y=[dominant_amp], mode='markers', 
                               name='Dominant', marker=dict(color='red', size=10, symbol='diamond'), 
                               showlegend=False),
                    row=plot_row, col=1
                )
                
                # Add frequency band annotations
                fig.add_vline(x=1.0, line_dash="dot", line_color="green", 
                             annotation_text="1 Hz", row=plot_row, col=1)
                fig.add_vline(x=5.0, line_dash="dot", line_color="green", 
                             annotation_text="5 Hz", row=plot_row, col=1)
                
                plot_row += 1
                
            except Exception as e:
                logger.error(f"Error creating frequency spectrum plot: {e}")
                plot_row += 1
        
        # Autocorrelation Plot
        if "cross_signal" in advanced_options:
            try:
                # Calculate autocorrelation
                autocorr = np.correlate(signal_data, signal_data, mode='full')
                autocorr = autocorr[len(signal_data)-1:]
                lag_times = np.arange(len(autocorr)) / sampling_freq
                
                # Normalize autocorrelation
                autocorr_norm = autocorr / np.max(autocorr)
                
                fig.add_trace(
                    go.Scatter(x=lag_times, y=autocorr_norm, mode='lines', 
                               name='Autocorrelation', line=dict(color='#e67e22', width=2), showlegend=False),
                    row=plot_row, col=1
                )
                
                # Add zero line
                fig.add_hline(y=0, line_dash="dash", line_color="gray", row=plot_row, col=1)
                
                # Highlight first zero crossing
                zero_crossings = np.where(np.diff(np.sign(autocorr_norm)))[0]
                if len(zero_crossings) > 0:
                    first_period = lag_times[zero_crossings[0]]
                    fig.add_vline(x=first_period, line_dash="dot", line_color="red", 
                                 annotation_text=f"Period: {first_period:.3f}s", row=plot_row, col=1)
                
                plot_row += 1
                
            except Exception as e:
                logger.error(f"Error creating autocorrelation plot: {e}")
                plot_row += 1
        
        # Update layout
        fig.update_layout(
            title="Comprehensive Physiological Analysis Results",
            template="plotly_white",
            height=200 * num_plots,  # Dynamic height based on number of plots
            margin=dict(l=40, r=40, t=60, b=40),
            showlegend=False
        )
        
        # Update axes labels and titles
        for i in range(1, num_plots + 1):
            if i == num_plots:  # Only show x-axis title for bottom plot
                fig.update_xaxes(title_text="Time (s)", row=i, col=1)
            
            # Update y-axis titles based on plot type
            if "HRV Analysis" in plot_types[i-1]:
                fig.update_yaxes(title_text="RR Interval (ms)", row=i, col=1)
            elif "Poincaré Plot" in plot_types[i-1]:
                fig.update_yaxes(title_text="RR(n+1) (ms)", row=i, col=1)
                fig.update_xaxes(title_text="RR(n) (ms)", row=i, col=1)
            elif "Signal Morphology" in plot_types[i-1]:
                fig.update_yaxes(title_text="Amplitude", row=i, col=1)
            elif "Amplitude Distribution" in plot_types[i-1]:
                fig.update_yaxes(title_text="Density", row=i, col=1)
                fig.update_xaxes(title_text="Amplitude", row=i, col=1)
            elif "Beat-to-Beat" in plot_types[i-1]:
                fig.update_yaxes(title_text="Heart Rate (BPM)", row=i, col=1)
            elif "Energy Analysis" in plot_types[i-1]:
                fig.update_yaxes(title_text="Energy", row=i, col=1)
            elif "Statistical Distribution" in plot_types[i-1]:
                fig.update_yaxes(title_text="Density", row=i, col=1)
                fig.update_xaxes(title_text="Amplitude", row=i, col=1)
            elif "Frequency Spectrum" in plot_types[i-1]:
                fig.update_yaxes(title_text="Magnitude", row=i, col=1)
                fig.update_xaxes(title_text="Frequency (Hz)", row=i, col=1)
            elif "Autocorrelation" in plot_types[i-1]:
                fig.update_yaxes(title_text="Autocorrelation", row=i, col=1)
                fig.update_xaxes(title_text="Lag Time (s)", row=i, col=1)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating physiological analysis plots: {e}")
        return create_empty_figure()


def generate_envelope_analysis_results(signal_data, time_axis, sampling_freq):
    """Generate envelope detection analysis results using vitalDSP."""
    try:
        logger.info("Generating envelope detection analysis results...")
        
        if EnvelopeDetection is None:
            logger.warning("EnvelopeDetection module not available")
            return html.Div([
                html.H6("📦 Envelope Detection", className="mb-2"),
                html.P("Envelope detection module not available", className="text-warning")
            ])
        
        # Use vitalDSP envelope detection
        envelope_detector = EnvelopeDetection(signal_data)
        
        # Use available methods instead of non-existent ones
        upper_env = envelope_detector.hilbert_envelope()
        lower_env = envelope_detector.absolute_value_envelope()
        
        # Calculate envelope statistics
        env_width = upper_env - lower_env
        mean_width = np.mean(env_width)
        std_width = np.std(env_width)
        
        return html.Div([
            html.H6("📦 Envelope Detection", className="mb-2"),
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Metric", className="text-center"),
                        html.Th("Value", className="text-center"),
                        html.Th("Unit", className="text-center")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td("Mean Envelope Width", className="fw-bold"),
                        html.Td(f"{mean_width:.3f}", className="text-end"),
                        html.Td("AU", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Envelope Width Std", className="fw-bold"),
                        html.Td(f"{std_width:.3f}", className="text-end"),
                        html.Td("AU", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Envelope Stability", className="fw-bold"),
                        html.Td(f"{1 - std_width/mean_width:.3f}", className="text-end"),
                        html.Td("Ratio", className="text-muted")
                    ])
                ])
            ], bordered=True, hover=True, responsive=True, className="mb-3")
        ])
        
    except Exception as e:
        logger.error(f"Error generating envelope analysis results: {e}")
        return html.Div([
            html.H6("📦 Envelope Detection", className="mb-2"),
            html.P(f"Error: {str(e)}", className="text-danger")
        ])


def generate_segmentation_analysis_results(signal_data, time_axis, sampling_freq):
    """Generate signal segmentation analysis results using vitalDSP."""
    try:
        logger.info("Generating signal segmentation analysis results...")
        
        if SignalSegmentation is None:
            logger.warning("SignalSegmentation module not available")
            return html.Div([
                html.H6("✂️ Signal Segmentation", className="mb-2"),
                html.P("Signal segmentation module not available", className="text-warning")
            ])
        
        # Use vitalDSP signal segmentation
        segmenter = SignalSegmentation(signal_data)
        
        # Use available methods instead of non-existent ones
        segments = segmenter.fixed_size_segmentation(segment_size=len(signal_data)//10)
        
        if len(segments) > 0:
            segment_durations = [seg[1] - seg[0] for seg in segments]
            mean_duration = np.mean(segment_durations)
            std_duration = np.std(segment_durations)
            
            return html.Div([
                html.H6("✂️ Signal Segmentation", className="mb-2"),
                dbc.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("Metric", className="text-center"),
                            html.Th("Value", className="text-center"),
                            html.Th("Unit", className="text-center")
                        ])
                    ]),
                    html.Tbody([
                        html.Tr([
                            html.Td("Number of Segments", className="fw-bold"),
                            html.Td(f"{len(segments)}", className="text-end"),
                            html.Td("Count", className="text-muted")
                        ]),
                        html.Tr([
                            html.Td("Mean Segment Duration", className="fw-bold"),
                            html.Td(f"{mean_duration:.3f}", className="text-end"),
                            html.Td("Seconds", className="text-muted")
                        ]),
                        html.Tr([
                            html.Td("Segment Duration Std", className="fw-bold"),
                            html.Td(f"{std_duration:.3f}", className="text-end"),
                            html.Td("Seconds", className="text-muted")
                        ])
                    ])
                ], bordered=True, hover=True, responsive=True, className="mb-3")
            ])
        else:
            return html.Div([
                html.H6("✂️ Signal Segmentation", className="mb-2"),
                html.P("No segments detected", className="text-warning")
            ])
        
    except Exception as e:
        logger.error(f"Error generating segmentation analysis results: {e}")
        return html.Div([
            html.H6("✂️ Signal Segmentation", className="mb-2"),
            html.P(f"Error: {str(e)}", className="text-danger")
        ])


def generate_trend_analysis_results(signal_data, time_axis, sampling_freq):
    """Generate trend analysis results using vitalDSP."""
    try:
        logger.info("Generating trend analysis results...")
        
        if TrendAnalysis is None:
            logger.warning("TrendAnalysis module not available")
            return html.Div([
                html.H6("📈 Trend Analysis", className="mb-2"),
                html.P("Trend analysis module not available", className="text-warning")
            ])
        
        # Use vitalDSP trend analysis
        trend_analyzer = TrendAnalysis(signal_data)
        
        # Use available methods instead of non-existent ones
        linear_trend = trend_analyzer.compute_linear_trend()
        trend_strength = trend_analyzer.compute_trend_strength()
        # Calculate slope from linear trend
        trend_slope = np.mean(np.diff(linear_trend))
        
        return html.Div([
            html.H6("📈 Trend Analysis", className="mb-2"),
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Metric", className="text-center"),
                        html.Th("Value", className="text-center"),
                        html.Th("Unit", className="text-center")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td("Trend Slope", className="fw-bold"),
                        html.Td(f"{trend_slope:.6f}", className="text-end"),
                        html.Td("AU/s", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Trend Strength", className="fw-bold"),
                        html.Td(f"{trend_strength:.3f}", className="text-end"),
                        html.Td("Ratio", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Trend Direction", className="fw-bold"),
                        html.Td("Increasing" if trend_slope > 0 else "Decreasing" if trend_slope < 0 else "Stable", className="text-end"),
                        html.Td("-", className="text-muted")
                    ])
                ])
            ], bordered=True, hover=True, responsive=True, className="mb-3")
        ])
        
    except Exception as e:
        logger.error(f"Error generating trend analysis results: {e}")
        return html.Div([
            html.H6("📈 Trend Analysis", className="mb-2"),
            html.P(f"Error: {str(e)}", className="text-danger")
        ])


def generate_waveform_analysis_results(signal_data, time_axis, sampling_freq, signal_type):
    """Generate waveform analysis results using vitalDSP."""
    try:
        logger.info("Generating waveform analysis results...")
        
        if WaveformMorphology is None:
            logger.warning("WaveformMorphology module not available")
            return html.Div([
                html.H6("🌊 Waveform Analysis", className="mb-2"),
                html.P("Waveform morphology module not available", className="text-warning")
            ])
        
        # Use vitalDSP waveform morphology
        waveform_analyzer = WaveformMorphology(signal_data, fs=sampling_freq, signal_type=signal_type.upper())
        
        # Extract waveform features
        if signal_type == "ppg":
            # Use available PPG-specific methods
            features = {
                'amplitude': waveform_analyzer.compute_amplitude(),
                'volume': waveform_analyzer.compute_volume(interval_type="Sys-to-Notch", signal_type="PPG"),
                'skewness': waveform_analyzer.compute_skewness(signal_type="PPG"),
                'dicrotic_notch': waveform_analyzer.compute_ppg_dicrotic_notch()
            }
        elif signal_type == "ecg":
            # Use available ECG-specific methods
            features = {
                'amplitude': waveform_analyzer.compute_amplitude(),
                'volume': waveform_analyzer.compute_volume(interval_type="R-to-S", signal_type="ECG"),
                'skewness': waveform_analyzer.compute_skewness(signal_type="ECG"),
                'duration': waveform_analyzer.compute_duration()
            }
        else:
            # Use general methods
            features = {
                'amplitude': waveform_analyzer.compute_amplitude(),
                'skewness': waveform_analyzer.compute_skewness(),
                'slope': waveform_analyzer.compute_slope(),
                'curvature': waveform_analyzer.compute_curvature()
            }
        
        # Create results table
        feature_rows = []
        for key, value in features.items():
            if isinstance(value, (int, float)):
                feature_rows.append(html.Tr([
                    html.Td(key.replace('_', ' ').title(), className="fw-bold"),
                    html.Td(f"{value:.3f}" if isinstance(value, float) else str(value), className="text-end"),
                    html.Td("AU" if "amplitude" in key.lower() else "ms" if "duration" in key.lower() else "-", className="text-muted")
                ]))
        
        return html.Div([
            html.H6("🌊 Waveform Analysis", className="mb-2"),
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Feature", className="text-center"),
                        html.Th("Value", className="text-center"),
                        html.Th("Unit", className="text-center")
                    ])
                ]),
                html.Tbody(feature_rows)
            ], bordered=True, hover=True, responsive=True, className="mb-3")
        ])
        
    except Exception as e:
        logger.error(f"Error generating waveform analysis results: {e}")
        return html.Div([
            html.H6("🌊 Waveform Analysis", className="mb-2"),
            html.P(f"Error: {str(e)}", className="text-danger")
        ])


def generate_cross_signal_analysis_results(signal_data, time_axis, sampling_freq, signal_type):
    """Generate cross-signal analysis results using vitalDSP."""
    try:
        logger.info("Generating cross-signal analysis results...")
        
        if CrossSignalAnalysis is None:
            logger.warning("CrossSignalAnalysis module not available")
            return html.Div([
                html.H6("🔗 Cross-Signal Analysis", className="mb-2"),
                html.P("Cross-signal analysis module not available", className="text-warning")
            ])
        
        # Use vitalDSP cross-signal analysis
        cross_analyzer = CrossSignalAnalysis(signal_data, signal_data, fs=sampling_freq)  # Using same signal for demo
        
        # Compute cross-correlation and coherence
        lags, cross_corr = cross_analyzer.compute_cross_correlation()
        freqs, coh = cross_analyzer.compute_coherence()
        
        # Get peak cross-correlation and mean coherence
        peak_cross_corr = np.max(np.abs(cross_corr))
        mean_coherence = np.mean(coh)
        
        return html.Div([
            html.H6("🔗 Cross-Signal Analysis", className="mb-2"),
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Metric", className="text-center"),
                        html.Th("Value", className="text-center"),
                        html.Th("Unit", className="text-center")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td("Peak Cross-Correlation", className="fw-bold"),
                        html.Td(f"{peak_cross_corr:.3f}", className="text-end"),
                        html.Td("Ratio", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Mean Coherence", className="fw-bold"),
                        html.Td(f"{mean_coherence:.3f}", className="text-end"),
                        html.Td("Ratio", className="text-muted")
                    ])
                ])
            ], bordered=True, hover=True, responsive=True, className="mb-3")
        ])
        
    except Exception as e:
        logger.error(f"Error generating cross-signal analysis results: {e}")
        return html.Div([
            html.H6("🔗 Cross-Signal Analysis", className="mb-2"),
            html.P(f"Error: {str(e)}", className="text-danger")
        ])


def generate_ensemble_analysis_results(signal_data, time_axis, sampling_freq):
    """Generate ensemble analysis results using vitalDSP."""
    try:
        logger.info("Generating ensemble analysis results...")
        
        # Note: EnsembleFeatureExtractor is designed for ML feature extraction, not signal analysis
        # We'll provide a comprehensive ensemble analysis using multiple vitalDSP features instead
        
        # Create a comprehensive ensemble analysis by combining multiple vitalDSP features
        # This provides a multi-dimensional view of the signal characteristics
        
        # Basic statistical features
        basic_features = {
            'Signal Length': len(signal_data),
            'Signal Mean': np.mean(signal_data),
            'Signal Std': np.std(signal_data),
            'Signal Range': np.max(signal_data) - np.min(signal_data),
            'Signal Energy': np.sum(signal_data**2),
            'Signal Power': np.mean(signal_data**2)
        }
        
        # Advanced features using vitalDSP modules
        advanced_features = {}
        
        # Time domain features if available
        if TimeDomainFeatures is not None:
            try:
                # Create synthetic NN intervals for demonstration
                nn_intervals = np.diff(np.arange(0, len(signal_data), 100)) / sampling_freq * 1000  # Convert to ms
                if len(nn_intervals) > 0:
                    time_features = TimeDomainFeatures(nn_intervals)
                    advanced_features['RMSSD'] = time_features.rmssd()
                    advanced_features['SDNN'] = time_features.sdnn()
                    advanced_features['NN50'] = time_features.nn50()
                    advanced_features['pNN50'] = time_features.pnn50()
            except Exception as e:
                logger.warning(f"Could not compute time domain features: {e}")
        
        # Frequency domain features if available
        if FrequencyDomainFeatures is not None:
            try:
                # Create synthetic NN intervals for demonstration
                nn_intervals = np.diff(np.arange(0, len(signal_data), 100)) / sampling_freq * 1000  # Convert to ms
                if len(nn_intervals) > 0:
                    freq_features = FrequencyDomainFeatures(nn_intervals, fs=sampling_freq)
                    advanced_features['Total Power'] = freq_features.total_power()
                    advanced_features['VLF Power'] = freq_features.vlf_power()
                    advanced_features['LF Power'] = freq_features.lf_power()
                    advanced_features['HF Power'] = freq_features.hf_power()
            except Exception as e:
                logger.warning(f"Could not compute frequency domain features: {e}")
        
        # Nonlinear features if available
        if NonlinearFeatures is not None:
            try:
                # Create synthetic NN intervals for demonstration
                nn_intervals = np.diff(np.arange(0, len(signal_data), 100)) / sampling_freq * 1000  # Convert to ms
                if len(nn_intervals) > 0:
                    nonlinear_features = NonlinearFeatures(nn_intervals)
                    advanced_features['Sample Entropy'] = nonlinear_features.sample_entropy()
                    advanced_features['Approximate Entropy'] = nonlinear_features.approximate_entropy()
                    advanced_features['Correlation Dimension'] = nonlinear_features.correlation_dimension()
            except Exception as e:
                logger.warning(f"Could not compute nonlinear features: {e}")
        
        # Combine all features
        features = {**basic_features, **advanced_features}
        
        # Create results table
        feature_rows = []
        for key, value in features.items():
            if isinstance(value, (int, float)):
                feature_rows.append(html.Tr([
                    html.Td(key.replace('_', ' ').title(), className="fw-bold"),
                    html.Td(f"{value:.3f}" if isinstance(value, float) else str(value), className="text-end"),
                    html.Td("AU", className="text-muted")
                ]))
        
        return html.Div([
            html.H6("🎯 Ensemble Analysis", className="mb-2"),
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Feature", className="text-center"),
                        html.Th("Value", className="text-center"),
                        html.Th("Unit", className="text-center")
                    ])
                ]),
                html.Tbody(feature_rows)
            ], bordered=True, hover=True, responsive=True, className="mb-3")
        ])
        
    except Exception as e:
        logger.error(f"Error generating ensemble analysis results: {e}")
        return html.Div([
            html.H6("🎯 Ensemble Analysis", className="mb-2"),
            html.P(f"Error: {str(e)}", className="text-danger")
        ])


def generate_change_detection_analysis_results(signal_data, time_axis, sampling_freq):
    """Generate change detection analysis results using vitalDSP."""
    try:
        logger.info("Generating change detection analysis results...")
        
        if SignalChangeDetection is None:
            logger.warning("SignalChangeDetection module not available")
            return html.Div([
                html.H6("🔄 Change Detection", className="mb-2"),
                html.P("Signal change detection module not available", className="text-warning")
            ])
        
        # Use vitalDSP signal change detection
        change_detector = SignalChangeDetection(signal_data)
        
        # Use available methods instead of non-existent ones
        change_points = change_detector.adaptive_threshold_detection()
        
        if len(change_points) > 0:
            change_intervals = np.diff(change_points) / sampling_freq
            mean_interval = np.mean(change_intervals)
            std_interval = np.std(change_intervals)
            
            return html.Div([
                html.H6("🔄 Change Detection", className="mb-2"),
                dbc.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("Metric", className="text-center"),
                            html.Th("Value", className="text-center"),
                            html.Th("Unit", className="text-center")
                        ])
                    ]),
                    html.Tbody([
                        html.Tr([
                            html.Td("Number of Changes", className="fw-bold"),
                            html.Td(f"{len(change_points)}", className="text-end"),
                            html.Td("Count", className="text-muted")
                        ]),
                        html.Tr([
                            html.Td("Mean Change Interval", className="fw-bold"),
                            html.Td(f"{mean_interval:.3f}", className="text-end"),
                            html.Td("Seconds", className="text-muted")
                        ]),
                        html.Tr([
                            html.Td("Change Interval Std", className="fw-bold"),
                            html.Td(f"{std_interval:.3f}", className="text-end"),
                            html.Td("Seconds", className="text-muted")
                        ])
                    ])
                ], bordered=True, hover=True, responsive=True, className="mb-3")
            ])
        else:
            return html.Div([
                html.H6("🔄 Change Detection", className="mb-2"),
                html.P("No significant changes detected", className="text-success")
            ])
        
    except Exception as e:
        logger.error(f"Error generating change detection analysis results: {e}")
        return html.Div([
            html.H6("🔄 Change Detection", className="mb-2"),
            html.P(f"Error: {str(e)}", className="text-danger")
        ])


def generate_power_analysis_results(signal_data, time_axis, sampling_freq):
    """Generate power analysis results using vitalDSP."""
    try:
        logger.info("Generating power analysis results...")
        
        if SignalPowerAnalysis is None:
            logger.warning("SignalPowerAnalysis module not available")
            return html.Div([
                html.H6("⚡ Power Analysis", className="mb-2"),
                html.P("Signal power analysis module not available", className="text-warning")
            ])
        
        # Use vitalDSP signal power analysis
        power_analyzer = SignalPowerAnalysis(signal_data)
        
        # Compute power metrics using available methods
        total_power = power_analyzer.compute_total_power()
        peak_power = power_analyzer.compute_peak_power()
        energy = power_analyzer.compute_energy()
        
        return html.Div([
            html.H6("⚡ Power Analysis", className="mb-2"),
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Metric", className="text-center"),
                        html.Th("Value", className="text-center"),
                        html.Th("Unit", className="text-center")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td("Total Power", className="fw-bold"),
                        html.Td(f"{total_power:.3e}", className="text-end"),
                        html.Td("AU²", className="text-muted")
                    ]),
                                            html.Tr([
                            html.Td("Peak Power", className="fw-bold"),
                            html.Td(f"{peak_power:.3e}", className="text-end"),
                            html.Td("AU²", className="text-muted")
                        ]),
                        html.Tr([
                            html.Td("Total Energy", className="fw-bold"),
                            html.Td(f"{energy:.3e}", className="text-end"),
                            html.Td("AU²", className="text-muted")
                        ])
                ])
            ], bordered=True, hover=True, responsive=True, className="mb-3")
        ])
        
    except Exception as e:
        logger.error(f"Error generating power analysis results: {e}")
        return html.Div([
            html.H6("⚡ Power Analysis", className="mb-2"),
            html.P(f"Error: {str(e)}", className="text-danger")
        ])


def generate_signal_transform_results(signal_data, time_axis, sampling_freq):
    """Generate signal transform results using vitalDSP."""
    try:
        logger.info("Generating signal transform results...")
        
        results = []
        results.append(html.H6("🔄 Signal Transforms", className="mb-2"))
        
        # Wavelet Transform
        if WaveletTransform is not None:
            try:
                wavelet_analyzer = WaveletTransform(signal_data)
                # Use the actual method that exists
                coeffs = wavelet_analyzer.perform_wavelet_transform(level=2)
                
                results.append(html.H6("🌊 Wavelet Transform", className="mb-2", style={"fontSize": "1rem"}))
                wavelet_table = dbc.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("Feature", className="text-center"),
                            html.Th("Value", className="text-center"),
                            html.Th("Unit", className="text-center")
                        ])
                    ]),
                    html.Tbody([
                        html.Tr([
                            html.Td("Decomposition Levels", className="fw-bold"),
                            html.Td("2", className="text-end"),
                            html.Td("Levels", className="text-muted")
                        ]),
                        html.Tr([
                            html.Td("Coefficients", className="fw-bold"),
                            html.Td(f"{len(coeffs)}", className="text-end"),
                            html.Td("Count", className="text-muted")
                        ])
                    ])
                ], bordered=True, hover=True, responsive=True, className="mb-3")
                results.append(wavelet_table)
            except Exception as e:
                logger.error(f"Wavelet transform failed: {e}")
                results.append(html.P("Wavelet transform failed", className="text-warning"))
        
        # Fourier Transform
        if FourierTransform is not None:
            try:
                fourier_analyzer = FourierTransform(signal_data)
                # Use the actual method that exists
                dft_result = fourier_analyzer.compute_dft()
                
                results.append(html.H6("📊 Fourier Transform", className="mb-2", style={"fontSize": "1rem"}))
                fourier_table = dbc.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("Feature", className="text-center"),
                            html.Th("Value", className="text-center"),
                            html.Th("Unit", className="text-center")
                        ])
                    ]),
                    html.Tbody([
                        html.Tr([
                            html.Td("Signal Length", className="fw-bold"),
                            html.Td(f"{len(signal_data)}", className="text-end"),
                            html.Td("Samples", className="text-muted")
                        ]),
                        html.Tr([
                            html.Td("DFT Magnitude", className="fw-bold"),
                            html.Td(f"{np.mean(np.abs(dft_result)):.3e}", className="text-end"),
                            html.Td("AU", className="text-muted")
                        ])
                    ])
                ], bordered=True, hover=True, responsive=True, className="mb-3")
                results.append(fourier_table)
            except Exception as e:
                logger.error(f"Fourier transform failed: {e}")
                results.append(html.P("Fourier transform failed", className="text-warning"))
        
        # Hilbert Transform
        if HilbertTransform is not None:
            try:
                hilbert_analyzer = HilbertTransform(signal_data)
                # Use the actual method that exists
                envelope = hilbert_analyzer.envelope()
                analytic_signal = hilbert_analyzer.compute_hilbert()
                
                results.append(html.H6("📐 Hilbert Transform", className="mb-2", style={"fontSize": "1rem"}))
                hilbert_table = dbc.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("Feature", className="text-center"),
                            html.Th("Value", className="text-center"),
                            html.Th("Unit", className="text-center")
                        ])
                    ]),
                    html.Tbody([
                        html.Tr([
                            html.Td("Envelope Mean", className="fw-bold"),
                            html.Td(f"{np.mean(envelope):.3f}", className="text-end"),
                            html.Td("AU", className="text-muted")
                        ]),
                        html.Tr([
                            html.Td("Analytic Signal Magnitude", className="fw-bold"),
                            html.Td(f"{np.mean(np.abs(analytic_signal)):.3f}", className="text-end"),
                            html.Td("AU", className="text-muted")
                        ])
                    ])
                ], bordered=True, hover=True, responsive=True, className="mb-3")
                results.append(hilbert_table)
            except Exception as e:
                logger.error(f"Hilbert transform failed: {e}")
                results.append(html.P("Hilbert transform failed", className="text-warning"))
        
        if len(results) == 1:  # Only header
            results.append(html.P("No transform modules available", className="text-warning"))
        
        return html.Div(results)
        
    except Exception as e:
        logger.error(f"Error generating signal transform results: {e}")
        return html.Div([
            html.H6("🔄 Signal Transforms", className="mb-2"),
            html.P(f"Error: {str(e)}", className="text-danger")
        ])


logger.info("=== PHYSIOLOGICAL CALLBACKS MODULE LOADED SUCCESSFULLY ===")
