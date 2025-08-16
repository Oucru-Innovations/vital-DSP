"""
Respiratory analysis callbacks for vitalDSP webapp.
Handles comprehensive respiratory rate estimation and breathing pattern analysis using vitalDSP.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, callback_context, no_update, html, dcc
from dash.exceptions import PreventUpdate
from scipy import signal
import dash_bootstrap_components as dbc
import logging

# Initialize logger first
logger = logging.getLogger(__name__)

# Initialize vitalDSP modules as None
RespiratoryAnalysis = None
peak_detection_rr = None
fft_based_rr = None
frequency_domain_rr = None
time_domain_rr = None
detect_apnea_amplitude = None
detect_apnea_pauses = None
multimodal_analysis = None
ppg_ecg_fusion = None
respiratory_cardiac_fusion = None
PPGRespiratoryAutonomicFeatures = None
ECGPpgSynchronizationFeatures = None
PreprocessConfig = None
preprocess_signal = None

def _import_vitaldsp_modules():
    """Import vitalDSP modules with error handling."""
    global RespiratoryAnalysis, peak_detection_rr, fft_based_rr, frequency_domain_rr
    global time_domain_rr, detect_apnea_amplitude, detect_apnea_pauses, multimodal_analysis
    global ppg_ecg_fusion, respiratory_cardiac_fusion, PPGRespiratoryAutonomicFeatures
    global ECGPpgSynchronizationFeatures, PreprocessConfig, preprocess_signal
    
    logger.info("=== IMPORTING VITALDSP MODULES ===")
    
    try:
        from vitalDSP.respiratory_analysis.respiratory_analysis import RespiratoryAnalysis
        logger.info("✓ RespiratoryAnalysis imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import RespiratoryAnalysis: {e}")
        RespiratoryAnalysis = None

    try:
        from vitalDSP.respiratory_analysis.estimate_rr.peak_detection_rr import peak_detection_rr
        logger.info("✓ peak_detection_rr imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import peak_detection_rr: {e}")
        peak_detection_rr = None

    try:
        from vitalDSP.respiratory_analysis.estimate_rr.fft_based_rr import fft_based_rr
        logger.info("✓ fft_based_rr imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import fft_based_rr: {e}")
        fft_based_rr = None

    try:
        from vitalDSP.respiratory_analysis.estimate_rr.frequency_domain_rr import frequency_domain_rr
        logger.info("✓ frequency_domain_rr imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import frequency_domain_rr: {e}")
        frequency_domain_rr = None

    try:
        from vitalDSP.respiratory_analysis.estimate_rr.time_domain_rr import time_domain_rr
        logger.info("✓ time_domain_rr imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import time_domain_rr: {e}")
        time_domain_rr = None

    try:
        from vitalDSP.respiratory_analysis.sleep_apnea_detection.amplitude_threshold import detect_apnea_amplitude
        logger.info("✓ detect_apnea_amplitude imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import detect_apnea_amplitude: {e}")
        detect_apnea_amplitude = None

    try:
        from vitalDSP.respiratory_analysis.sleep_apnea_detection.pause_detection import detect_apnea_pauses
        logger.info("✓ detect_apnea_pauses imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import detect_apnea_pauses: {e}")
        detect_apnea_pauses = None

    try:
        from vitalDSP.respiratory_analysis.fusion.multimodal_analysis import multimodal_analysis
        logger.info("✓ multimodal_analysis imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import multimodal_analysis: {e}")
        multimodal_analysis = None

    try:
        from vitalDSP.respiratory_analysis.fusion.ppg_ecg_fusion import ppg_ecg_fusion
        logger.info("✓ ppg_ecg_fusion imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import ppg_ecg_fusion: {e}")
        ppg_ecg_fusion = None

    try:
        from vitalDSP.respiratory_analysis.fusion.respiratory_cardiac_fusion import respiratory_cardiac_fusion
        logger.info("✓ respiratory_cardiac_fusion imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import respiratory_cardiac_fusion: {e}")
        respiratory_cardiac_fusion = None

    try:
        from vitalDSP.feature_engineering.ppg_autonomic_features import PPGRespiratoryAutonomicFeatures
        logger.info("✓ PPGRespiratoryAutonomicFeatures imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import PPGRespiratoryAutonomicFeatures: {e}")
        PPGRespiratoryAutonomicFeatures = None

    try:
        from vitalDSP.respiratory_analysis.ecg_ppg_synchronyzation_features import ECGPpgSynchronizationFeatures
        logger.info("✓ ECGPpgSynchronizationFeatures imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import ECGPpgSynchronizationFeatures: {e}")
        ECGPpgSynchronizationFeatures = None

    try:
        from vitalDSP.preprocess.preprocess_operations import PreprocessConfig, preprocess_signal
        logger.info("✓ PreprocessConfig and preprocess_signal imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import PreprocessConfig/preprocess_signal: {e}")
        PreprocessConfig = None
        preprocess_signal = None

    logger.info("=== VITALDSP MODULE IMPORT COMPLETED ===")


def register_respiratory_callbacks(app):
    """Register all respiratory analysis callbacks."""
    logger.info("=== REGISTERING RESPIRATORY CALLBACKS ===")
    logger.info(f"App type: {type(app)}")
    
    # Import vitalDSP modules when callbacks are registered
    _import_vitaldsp_modules()
    
    @app.callback(
        [Output("resp-main-signal-plot", "figure"),
         Output("resp-analysis-results", "children"),
         Output("resp-analysis-plots", "figure"),
         Output("store-resp-data", "data"),
         Output("store-resp-features", "data")],
        [Input("url", "pathname"),
         Input("resp-btn-update-analysis", "n_clicks"),
         Input("resp-time-range-slider", "value"),
         Input("resp-btn-nudge-m10", "n_clicks"),
         Input("resp-btn-nudge-m1", "n_clicks"),
         Input("resp-btn-nudge-p1", "n_clicks"),
         Input("resp-btn-nudge-p10", "n_clicks")],
        [State("resp-start-time", "value"),
         State("resp-end-time", "value"),
         State("resp-signal-type", "value"),
         State("resp-estimation-methods", "value"),
         State("resp-advanced-options", "value"),
         State("resp-preprocessing-options", "value"),
         State("resp-low-cut", "value"),
         State("resp-high-cut", "value"),
         State("resp-min-breath-duration", "value"),
         State("resp-max-breath-duration", "value")]
    )
    def respiratory_analysis_callback(pathname, n_clicks, slider_value, nudge_m10, nudge_m1, nudge_p1, nudge_p10,
                                   start_time, end_time, signal_type, estimation_methods, advanced_options, 
                                   preprocessing_options, low_cut, high_cut, min_breath_duration, max_breath_duration):
        """Unified callback for respiratory analysis - handles both page load and user interactions."""
        ctx = callback_context
        
        # Determine what triggered this callback
        if not ctx.triggered:
            logger.warning("No context triggered - raising PreventUpdate")
            raise PreventUpdate
        
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        logger.info(f"=== RESPIRATORY ANALYSIS CALLBACK TRIGGERED ===")
        logger.info(f"Trigger ID: {trigger_id}")
        logger.info(f"Pathname: {pathname}")
        logger.info(f"All callback parameters:")
        logger.info(f"  - n_clicks: {n_clicks}")
        logger.info(f"  - slider_value: {slider_value}")
        logger.info(f"  - start_time: {start_time}")
        logger.info(f"  - end_time: {end_time}")
        logger.info(f"  - signal_type: {signal_type}")
        logger.info(f"  - estimation_methods: {estimation_methods}")
        logger.info(f"  - advanced_options: {advanced_options}")
        logger.info(f"  - preprocessing_options: {preprocessing_options}")
        logger.info(f"  - low_cut: {low_cut}")
        logger.info(f"  - high_cut: {high_cut}")
        logger.info(f"  - min_breath_duration: {min_breath_duration}")
        logger.info(f"  - max_breath_duration: {max_breath_duration}")
        
        # Only run this when we're on the respiratory page
        if pathname != "/respiratory":
            logger.info("Not on respiratory page, returning empty figures")
            return create_empty_figure(), "Navigate to Respiratory Analysis page", create_empty_figure(), None, None
        
        # If this is the first time loading the page (no button clicks), show a message
        if not ctx.triggered or ctx.triggered[0]["prop_id"].split(".")[0] == "url":
            logger.info("First time loading respiratory page, attempting to load data")
        
        try:
            # Get data from the data service
            logger.info("Attempting to get data service...")
            from ..services.data_service import get_data_service
            data_service = get_data_service()
            logger.info("Data service retrieved successfully")
            
            # Get the most recent data
            logger.info("Retrieving all data from service...")
            all_data = data_service.get_all_data()
            logger.info(f"All data keys: {list(all_data.keys()) if all_data else 'None'}")
            
            if not all_data:
                logger.warning("No data found in service")
                return create_empty_figure(), "No data available. Please upload and process data first.", create_empty_figure(), None, None
            
            # Get the most recent data entry
            latest_data_id = list(all_data.keys())[-1]
            latest_data = all_data[latest_data_id]
            logger.info(f"Latest data ID: {latest_data_id}")
            logger.info(f"Latest data info: {latest_data.get('info', 'No info')}")
            
            # Get column mapping
            logger.info("Retrieving column mapping...")
            column_mapping = data_service.get_column_mapping(latest_data_id)
            logger.info(f"Column mapping: {column_mapping}")
            
            if not column_mapping:
                logger.warning("Data has not been processed yet - no column mapping found")
                return create_empty_figure(), "Please process your data on the Upload page first (configure column mapping)", create_empty_figure(), None, None
                
            # Get the actual data
            logger.info("Retrieving data frame...")
            df = data_service.get_data(latest_data_id)
            logger.info(f"Data frame shape: {df.shape if df is not None else 'None'}")
            logger.info(f"Data frame columns: {list(df.columns) if df is not None else 'None'}")
            
            if df is None or df.empty:
                logger.warning("Data frame is empty")
                return create_empty_figure(), "Data is empty or corrupted.", create_empty_figure(), None, None
            
            # Get sampling frequency from the data info
            sampling_freq = latest_data.get('info', {}).get('sampling_freq', 1000)
            logger.info(f"Sampling frequency: {sampling_freq}")
            
            # Handle time window adjustments for nudge buttons
            if trigger_id in ["resp-btn-nudge-m10", "resp-btn-nudge-m1", "resp-btn-nudge-p1", "resp-btn-nudge-p10"]:
                if not start_time or not end_time:
                    start_time, end_time = 0, 10
                    
                if trigger_id == "resp-btn-nudge-m10":
                    start_time = max(0, start_time - 10)
                    end_time = max(10, end_time - 10)
                elif trigger_id == "resp-btn-nudge-m1":
                    start_time = max(0, start_time - 1)
                    end_time = max(1, end_time - 1)
                elif trigger_id == "resp-btn-nudge-p1":
                    start_time = start_time + 1
                    end_time = end_time + 1
                elif trigger_id == "resp-btn-nudge-p10":
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
            logger.info(f"Signal column from mapping: {signal_column}")
            logger.info(f"Available columns in windowed data: {list(windowed_data.columns)}")
            
            if not signal_column or signal_column not in windowed_data.columns:
                logger.warning(f"Signal column {signal_column} not found in data")
                return create_empty_figure(), "Signal column not found in data.", create_empty_figure(), None, None
            
            signal_data = windowed_data[signal_column].values
            logger.info(f"Signal data shape: {signal_data.shape}")
            logger.info(f"Signal data range: {np.min(signal_data):.3f} to {np.max(signal_data):.3f}")
            logger.info(f"Signal data mean: {np.mean(signal_data):.3f}")
            
            # Auto-detect signal type if needed
            if signal_type == "auto":
                logger.info("Auto-detecting signal type...")
                signal_type = detect_respiratory_signal_type(signal_data, sampling_freq)
                logger.info(f"Auto-detected signal type: {signal_type}")
            
            logger.info(f"Final signal type: {signal_type}")
            logger.info(f"Estimation methods: {estimation_methods}")
            logger.info(f"Advanced options: {advanced_options}")
            logger.info(f"Preprocessing options: {preprocessing_options}")
            
            # Create main respiratory signal plot with annotations
            logger.info("Creating main respiratory signal plot with annotations...")
            main_plot = create_respiratory_signal_plot(signal_data, time_axis, sampling_freq, signal_type, 
                                                     estimation_methods, preprocessing_options, low_cut, high_cut)
            logger.info("Main respiratory signal plot created successfully")
            logger.info(f"Main plot type: {type(main_plot)}")
            
            # Generate comprehensive respiratory analysis results
            logger.info("Generating comprehensive respiratory analysis results...")
            analysis_results = generate_comprehensive_respiratory_analysis(signal_data, time_axis, sampling_freq, 
                                                                        signal_type, estimation_methods, advanced_options,
                                                                        preprocessing_options, low_cut, high_cut,
                                                                        min_breath_duration, max_breath_duration)
            logger.info("Respiratory analysis results generated successfully")
            logger.info(f"Analysis results type: {type(analysis_results)}")
            logger.info(f"Analysis results content length: {len(str(analysis_results)) if analysis_results else 'None'}")
            
            # Create respiratory analysis plots
            logger.info("Creating respiratory analysis plots...")
            analysis_plots = create_comprehensive_respiratory_plots(signal_data, time_axis, sampling_freq, 
                                                                  signal_type, estimation_methods, advanced_options,
                                                                  preprocessing_options, low_cut, high_cut)
            logger.info("Respiratory analysis plots created successfully")
            logger.info(f"Analysis plots type: {type(analysis_plots)}")
            
            # Store processed data
            resp_data = {
                "signal_data": signal_data.tolist(),
                "time_axis": time_axis.tolist(),
                "sampling_freq": sampling_freq,
                "window": [start_time, end_time],
                "signal_type": signal_type,
                "estimation_methods": estimation_methods
            }
            
            resp_features = {
                "advanced_options": advanced_options,
                "preprocessing_options": preprocessing_options,
                "filter_params": {"low_cut": low_cut, "high_cut": high_cut},
                "breath_constraints": {"min_duration": min_breath_duration, "max_duration": max_breath_duration}
            }
            
            logger.info("Respiratory analysis completed successfully")
            logger.info("Returning results:")
            logger.info(f"  - Main plot: {type(main_plot)}")
            logger.info(f"  - Analysis results: {type(analysis_results)}")
            logger.info(f"  - Analysis plots: {type(analysis_plots)}")
            logger.info(f"  - Resp data: {type(resp_data)}")
            logger.info(f"  - Resp features: {type(resp_features)}")
            
            return main_plot, analysis_results, analysis_plots, resp_data, resp_features
            
        except Exception as e:
            logger.error(f"Error in respiratory analysis callback: {e}")
            import traceback
            traceback.print_exc()
            return create_empty_figure(), f"Error in analysis: {str(e)}", create_empty_figure(), None, None
    
    @app.callback(
        [Output("resp-start-time", "value"),
         Output("resp-end-time", "value")],
        [Input("resp-time-range-slider", "value")]
    )
    def update_resp_time_inputs(slider_value):
        """Update time input fields based on slider."""
        if not slider_value:
            return no_update, no_update
        return slider_value[0], slider_value[1]
    
    @app.callback(
        [Output("resp-time-range-slider", "min"),
         Output("resp-time-range-slider", "max"),
         Output("resp-time-range-slider", "value")],
        [Input("url", "pathname")]
    )
    def update_resp_time_slider_range(pathname):
        """Update time slider range based on data duration."""
        logger.info("=== UPDATE RESP TIME SLIDER RANGE ===")
        logger.info(f"Pathname: {pathname}")
        
        # Only run this when we're on the respiratory page
        if pathname != "/respiratory":
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
            logger.error(f"Error updating resp time slider range: {e}")
            return 0, 100, [0, 10]
    
    logger.info("=== RESPIRATORY CALLBACKS REGISTERED SUCCESSFULLY ===")


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


def detect_respiratory_signal_type(signal_data, sampling_freq):
    """Auto-detect respiratory signal type based on signal characteristics."""
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
        
        # Respiratory signals typically have dominant frequency around 0.2-0.5 Hz (12-30 BPM)
        # PPG respiratory component is usually lower amplitude
        # ECG respiratory component is more complex
        
        if 0.1 < dominant_freq < 0.8 and range_val < 2 * std_val:
            return "ppg"
        elif 0.1 < dominant_freq < 1.0 and range_val > 3 * std_val:
            return "respiratory"
        elif dominant_freq > 0.5:
            return "ecg"
        else:
            return "ppg"  # Default to PPG for most cases
            
    except Exception as e:
        logger.warning(f"Respiratory signal type detection failed: {e}")
        return "ppg"  # Default fallback


def create_respiratory_signal_plot(signal_data, time_axis, sampling_freq, signal_type, 
                                  estimation_methods, preprocessing_options, low_cut, high_cut):
    """Create the main respiratory signal plot with annotations."""
    logger.info("=== CREATE RESPIRATORY SIGNAL PLOT STARTED ===")
    logger.info(f"Input parameters:")
    logger.info(f"  - signal_data shape: {signal_data.shape}")
    logger.info(f"  - time_axis shape: {time_axis.shape}")
    logger.info(f"  - sampling_freq: {sampling_freq}")
    logger.info(f"  - signal_type: {signal_type}")
    logger.info(f"  - estimation_methods: {estimation_methods}")
    logger.info(f"  - preprocessing_options: {preprocessing_options}")
    logger.info(f"  - low_cut: {low_cut}")
    logger.info(f"  - high_cut: {high_cut}")
    
    try:
        fig = go.Figure()
        
        # Apply preprocessing if selected
        logger.info("Applying preprocessing...")
        processed_signal = signal_data.copy()
        if preprocessing_options and "bandpass" in preprocessing_options:
            try:
                # Design bandpass filter for respiratory frequencies
                nyquist = sampling_freq / 2
                low = low_cut / nyquist
                high = high_cut / nyquist
                
                if low < high < 1.0:
                    b, a = signal.butter(4, [low, high], btype='bandpass')
                    processed_signal = signal.filtfilt(b, a, processed_signal)
                    logger.info(f"Applied bandpass filter: {low_cut}-{high_cut} Hz")
            except Exception as e:
                logger.error(f"Bandpass filtering failed: {e}")
        
        # Apply additional preprocessing options
        if preprocessing_options and "baseline_correction" in preprocessing_options:
            try:
                # Remove baseline wander using high-pass filter
                baseline_cutoff = 0.01  # 0.01 Hz cutoff for baseline
                baseline_nyquist = baseline_cutoff / nyquist
                if baseline_nyquist < 1.0:
                    b_baseline, a_baseline = signal.butter(2, baseline_nyquist, btype='highpass')
                    processed_signal = signal.filtfilt(b_baseline, a_baseline, processed_signal)
                    logger.info("Applied baseline correction")
            except Exception as e:
                logger.error(f"Baseline correction failed: {e}")
        
        if preprocessing_options and "moving_average" in preprocessing_options:
            try:
                # Apply moving average smoothing
                window_size = int(0.1 * sampling_freq)  # 100ms window
                if window_size > 0:
                    processed_signal = np.convolve(processed_signal, np.ones(window_size)/window_size, mode='same')
                    logger.info("Applied moving average smoothing")
            except Exception as e:
                logger.error(f"Moving average smoothing failed: {e}")
        
        # Create the main signal plot
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=processed_signal,
            mode='lines',
            name=f'{signal_type.upper()} Signal (Processed)',
            line=dict(color='#2E86AB', width=2),
            hovertemplate='<b>Time:</b> %{x:.3f}s<br><b>Amplitude:</b> %{y:.3f}<extra></extra>'
        ))
        
        # Add original signal if preprocessing was applied
        if preprocessing_options and len(preprocessing_options) > 0:
            fig.add_trace(go.Scatter(
                x=time_axis,
                y=signal_data,
                mode='lines',
                name=f'{signal_type.upper()} Signal (Original)',
                line=dict(color='#95A5A6', width=1, dash='dot'),
                opacity=0.7,
                hovertemplate='<b>Time:</b> %{x:.3f}s<br><b>Amplitude:</b> %{y:.3f}<extra></extra>'
            ))
        
        # Add breathing pattern detection if enabled
        if estimation_methods and "peaks" in estimation_methods:
            try:
                # Detect breathing peaks
                prominence = 0.3 * np.std(processed_signal)
                distance = int(0.5 * sampling_freq)  # Minimum 0.5s between breaths
                
                peaks, properties = signal.find_peaks(processed_signal, prominence=prominence, distance=distance)
                
                if len(peaks) > 0:
                    fig.add_trace(go.Scatter(
                        x=time_axis[peaks],
                        y=processed_signal[peaks],
                        mode='markers',
                        name='Breathing Peaks',
                        marker=dict(color='red', size=8, symbol='diamond'),
                        hovertemplate='<b>Breath:</b> %{y:.3f}<br><b>Time:</b> %{x:.3f}s<extra></extra>'
                    ))
                    
                    # Add breath annotations
                    for i, peak in enumerate(peaks[:10]):  # Limit to first 10 breaths
                        fig.add_annotation(
                            x=time_axis[peak],
                            y=processed_signal[peak],
                            text=f"B{i+1}",
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=2,
                            arrowcolor="red",
                            ax=0,
                            ay=-40
                        )
                    
                    # Add breathing rate annotation
                    if len(peaks) > 1:
                        breath_intervals = np.diff(peaks) / sampling_freq
                        breathing_rate = 60 / np.mean(breath_intervals)
                        fig.add_annotation(
                            x=0.02, y=0.98, xref='paper', yref='paper',
                            text=f"Breathing Rate: {breathing_rate:.1f} BPM",
                            showarrow=False,
                            bgcolor="rgba(255,255,255,0.8)",
                            bordercolor="red",
                            borderwidth=2
                        )
                        
            except Exception as e:
                logger.error(f"Breathing peak detection failed: {e}")
        
        # Add baseline
        baseline = np.mean(processed_signal)
        fig.add_hline(y=baseline, line_dash="dash", line_color="gray", 
                      annotation_text=f"Baseline: {baseline:.3f}")
        
        logger.info("Updating plot layout...")
        fig.update_layout(
            title=f"{signal_type.upper()} Respiratory Signal Analysis",
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
            template="plotly_white",
            height=400,
            margin=dict(l=40, r=40, t=60, b=40),
            showlegend=True,
            hovermode='closest'
        )
        
        logger.info("=== CREATE RESPIRATORY SIGNAL PLOT COMPLETED ===")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating respiratory signal plot: {e}")
        return create_empty_figure()


def generate_comprehensive_respiratory_analysis(signal_data, time_axis, sampling_freq, signal_type, 
                                              estimation_methods, advanced_options, preprocessing_options, 
                                              low_cut, high_cut, min_breath_duration, max_breath_duration):
    """Generate comprehensive respiratory analysis results using vitalDSP."""
    logger.info("=== GENERATE COMPREHENSIVE RESPIRATORY ANALYSIS STARTED ===")
    logger.info(f"Input parameters:")
    logger.info(f"  - signal_data shape: {signal_data.shape}")
    logger.info(f"  - time_axis shape: {time_axis.shape}")
    logger.info(f"  - sampling_freq: {sampling_freq}")
    logger.info(f"  - signal_type: {signal_type}")
    logger.info(f"  - estimation_methods: {estimation_methods}")
    logger.info(f"  - advanced_options: {advanced_options}")
    logger.info(f"  - preprocessing_options: {preprocessing_options}")
    logger.info(f"  - low_cut: {low_cut}")
    logger.info(f"  - high_cut: {high_cut}")
    logger.info(f"  - min_breath_duration: {min_breath_duration}")
    logger.info(f"  - max_breath_duration: {max_breath_duration}")
    
    try:
        results = []
        
        # Initialize vitalDSP RespiratoryAnalysis
        if RespiratoryAnalysis is None:
            logger.warning("RespiratoryAnalysis module not available - using fallback analysis")
            results.append(html.Div([
                html.H5("⚠️ Using Fallback Analysis", className="text-warning"),
                html.P("vitalDSP modules not available. Using basic signal processing methods.", className="text-muted")
            ]))
            
            # Fallback: Basic respiratory rate estimation using peak detection
            try:
                # Simple peak detection for breathing
                prominence = 0.3 * np.std(signal_data)
                distance = int(0.5 * sampling_freq)  # Minimum 0.5s between breaths
                peaks, _ = signal.find_peaks(signal_data, prominence=prominence, distance=distance)
                
                if len(peaks) > 1:
                    breath_intervals = np.diff(peaks) / sampling_freq
                    rr_bpm = 60 / np.mean(breath_intervals)
                    rr_std = 60 * np.std(breath_intervals) / (np.mean(breath_intervals) ** 2)
                    
                    results.append(html.Div([
                        html.Strong("Fallback Peak Detection: "),
                        html.Span(f"{rr_bpm:.2f} BPM", className="text-success")
                    ], className="mb-2"))
                    
                    results.append(html.Div([
                        html.Strong("Breathing Variability: "),
                        html.Span(f"{rr_std:.2f} BPM", className="text-info")
                    ], className="mb-2"))
                    
                    results.append(html.Div([
                        html.Strong("Number of Breaths: "),
                        html.Span(f"{len(peaks)}", className="text-info")
                    ], className="mb-2"))
                else:
                    results.append(html.Div([
                        html.Strong("Fallback Analysis: "),
                        html.Span("Insufficient peaks detected", className="text-warning")
                    ], className="mb-2"))
                    
            except Exception as e:
                logger.error(f"Fallback analysis failed: {e}")
                results.append(html.Div([
                    html.Strong("Fallback Analysis: "),
                    html.Span("Failed", className="text-danger")
                ], className="mb-2"))
            
            # Continue with basic statistics
        else:
            logger.info("Initializing vitalDSP RespiratoryAnalysis...")
            resp_analysis = RespiratoryAnalysis(signal_data, sampling_freq)
            logger.info("RespiratoryAnalysis initialized successfully")
        
        # Create preprocessing configuration
        if PreprocessConfig is None:
            logger.error("PreprocessConfig module not available - using default settings")
            preprocess_config = None
        else:
            logger.info("Creating preprocessing configuration...")
            preprocess_config = PreprocessConfig(
                filter_type='bandpass',
                lowcut=low_cut,
                highcut=high_cut,
                respiratory_mode=True,
                preprocess=True
            )
            logger.info(f"PreprocessConfig created: {preprocess_config}")
        
        # Respiratory Rate Estimation using multiple methods
        logger.info(f"Processing estimation methods: {estimation_methods}")
        if estimation_methods:
            logger.info("Adding respiratory rate estimation header...")
            results.append(html.H5("🫁 Respiratory Rate Estimation", className="mb-3"))
            
            for method in estimation_methods:
                logger.info(f"Processing method: {method}")
                try:
                    if method == "peaks":
                        logger.info("Computing respiratory rate using peaks method...")
                        rr = resp_analysis.compute_respiratory_rate(
                            method="peaks",
                            min_breath_duration=min_breath_duration,
                            max_breath_duration=max_breath_duration,
                            preprocess_config=preprocess_config
                        )
                        logger.info(f"Peaks method result: {rr:.2f} BPM")
                        results.append(html.Div([
                            html.Strong("Peak Detection Method: "),
                            html.Span(f"{rr:.2f} BPM", className="text-success")
                        ], className="mb-2"))
                        
                    elif method == "zero_crossing":
                        rr = resp_analysis.compute_respiratory_rate(
                            method="zero_crossing",
                            min_breath_duration=min_breath_duration,
                            max_breath_duration=max_breath_duration,
                            preprocess_config=preprocess_config
                        )
                        results.append(html.Div([
                            html.Strong("Zero Crossing Method: "),
                            html.Span(f"{rr:.2f} BPM", className="text-success")
                        ], className="mb-2"))
                        
                    elif method == "time_domain":
                        if time_domain_rr is None:
                            logger.warning("time_domain_rr function not available")
                            results.append(html.Div([
                                html.Strong("Time Domain Method: "),
                                html.Span("Function not available", className="text-warning")
                            ], className="mb-2"))
                        else:
                            rr = time_domain_rr(
                                signal_data, sampling_freq,
                                preprocess='bandpass',
                                lowcut=low_cut, highcut=high_cut
                            )
                            results.append(html.Div([
                                html.Strong("Time Domain Method: "),
                                html.Span(f"{rr:.2f} BPM", className="text-success")
                            ], className="mb-2"))
                        
                    elif method == "frequency_domain":
                        if frequency_domain_rr is None:
                            logger.warning("frequency_domain_rr function not available")
                            results.append(html.Div([
                                html.Strong("Frequency Domain Method: "),
                                html.Span("Function not available", className="text-warning")
                            ], className="mb-2"))
                        else:
                            rr = frequency_domain_rr(
                                signal_data, sampling_freq,
                                preprocess='bandpass',
                                lowcut=low_cut, highcut=high_cut
                            )
                            results.append(html.Div([
                                html.Strong("Frequency Domain Method: "),
                                html.Span(f"{rr:.2f} BPM", className="text-success")
                            ], className="mb-2"))
                        
                    elif method == "fft_based":
                        if fft_based_rr is None:
                            logger.warning("fft_based_rr function not available")
                            results.append(html.Div([
                                html.Strong("FFT-based Method: "),
                                html.Span("Function not available", className="text-warning")
                            ], className="mb-2"))
                        else:
                            rr = fft_based_rr(
                                signal_data, sampling_freq,
                                preprocess='bandpass',
                                lowcut=low_cut, highcut=high_cut
                            )
                            results.append(html.Div([
                                html.Strong("FFT-based Method: "),
                                html.Span(f"{rr:.2f} BPM", className="text-success")
                            ], className="mb-2"))
                        
                    elif method == "counting":
                        if peak_detection_rr is None:
                            logger.warning("peak_detection_rr function not available")
                            results.append(html.Div([
                                html.Strong("Counting Method: "),
                                html.Span("Function not available", className="text-warning")
                            ], className="mb-2"))
                        else:
                            rr = peak_detection_rr(
                                signal_data, sampling_freq,
                                preprocess='bandpass',
                                lowcut=low_cut, highcut=high_cut
                            )
                            results.append(html.Div([
                                html.Strong("Counting Method: "),
                                html.Span(f"{rr:.2f} BPM", className="text-success")
                            ], className="mb-2"))
                        
                except Exception as e:
                    logger.warning(f"Method {method} failed: {e}")
                    results.append(html.Div([
                        html.Strong(f"{method.replace('_', ' ').title()} Method: "),
                        html.Span("Failed", className="text-danger")
                    ], className="mb-2"))
        
        # Advanced Analysis Features
        logger.info(f"Processing advanced options: {advanced_options}")
        if advanced_options:
            logger.info("Adding advanced analysis section...")
            results.append(html.Hr())
            results.append(html.H5("🔬 Advanced Analysis", className="mb-3"))
            
            # Sleep Apnea Detection
            if "sleep_apnea" in advanced_options:
                logger.info("Processing sleep apnea detection...")
                try:
                    # Detect apnea events using amplitude threshold
                    apnea_threshold = 0.3 * np.std(signal_data)
                    logger.info(f"Apnea threshold: {apnea_threshold:.3f}")
                    
                    if detect_apnea_amplitude is None:
                        logger.warning("detect_apnea_amplitude function not available")
                        apnea_events = []
                    else:
                        apnea_events = detect_apnea_amplitude(
                            signal_data, sampling_freq, 
                            threshold=apnea_threshold, 
                            min_duration=5
                        )
                    
                    # Also detect apnea events using pause detection
                    if detect_apnea_pauses is None:
                        logger.warning("detect_apnea_pauses function not available")
                        pause_apnea_events = []
                    else:
                        pause_apnea_events = detect_apnea_pauses(
                            signal_data, sampling_freq, 
                            min_pause_duration=5
                        )
                    
                    total_apnea_events = len(apnea_events) + len(pause_apnea_events)
                    
                    if total_apnea_events > 0:
                        results.append(html.Div([
                            html.Strong("Sleep Apnea Events Detected: "),
                            html.Span(f"{total_apnea_events} events", className="text-warning")
                        ], className="mb-2"))
                        
                        if apnea_events:
                            results.append(html.Div([
                                html.Small(f"Amplitude-based: {len(apnea_events)} events", className="text-muted")
                            ], className="ms-3"))
                            
                        if pause_apnea_events:
                            results.append(html.Div([
                                html.Small(f"Pause-based: {len(pause_apnea_events)} events", className="text-muted")
                            ], className="ms-3"))
                            
                        # Show details of first few events
                        all_events = apnea_events + pause_apnea_events
                        for i, (start, end) in enumerate(all_events[:3]):  # Show first 3
                            results.append(html.Div([
                                html.Small(f"Event {i+1}: {start:.1f}s - {end:.1f}s", className="text-muted")
                            ], className="ms-3"))
                    else:
                        results.append(html.Div([
                            html.Strong("Sleep Apnea Detection: "),
                            html.Span("No events detected", className="text-success")
                        ], className="mb-2"))
                        
                except Exception as e:
                    logger.warning(f"Sleep apnea detection failed: {e}")
                    results.append(html.Div([
                        html.Strong("Sleep Apnea Detection: "),
                        html.Span("Failed", className="text-danger")
                        ], className="mb-2"))
            
            # Breathing Pattern Analysis
            if "breathing_pattern" in advanced_options:
                try:
                    # Analyze breathing pattern variability
                    peaks, _ = signal.find_peaks(signal_data, distance=int(0.5 * sampling_freq))
                    if len(peaks) > 1:
                        breath_intervals = np.diff(peaks) / sampling_freq
                        variability = np.std(breath_intervals)
                        mean_interval = np.mean(breath_intervals)
                        
                        results.append(html.Div([
                            html.Strong("Breathing Pattern Analysis: "),
                            html.Br(),
                            html.Small(f"Mean interval: {mean_interval:.2f}s", className="text-muted"),
                            html.Br(),
                            html.Small(f"Variability: {variability:.2f}s", className="text-muted")
                        ], className="mb-2"))
                        
                except Exception as e:
                    logger.warning(f"Breathing pattern analysis failed: {e}")
            
            # Respiratory Variability
            if "respiratory_variability" in advanced_options:
                try:
                    if signal_type == "ppg":
                        if PPGRespiratoryAutonomicFeatures is None:
                            logger.warning("PPGRespiratoryAutonomicFeatures not available")
                            results.append(html.Div([
                                html.Strong("Respiratory Variability (PPG): "),
                                html.Span("Function not available", className="text-warning")
                            ], className="mb-2"))
                        else:
                            # Use PPG respiratory autonomic features
                            ppg_features = PPGRespiratoryAutonomicFeatures(signal_data, sampling_freq)
                            rrv = ppg_features.compute_rrv()
                            rsa = ppg_features.compute_rsa()
                            
                            results.append(html.Div([
                                html.Strong("Respiratory Variability (PPG): "),
                                html.Br(),
                                html.Small(f"RRV: {rrv:.4f}s", className="text-muted"),
                                html.Br(),
                                html.Small(f"RSA: {rsa:.4f}s", className="text-muted")
                            ], className="mb-2"))
                        
                except Exception as e:
                    logger.warning(f"Respiratory variability analysis failed: {e}")
            
            # Respiratory-Cardiac Fusion Analysis
            if "cardiac_fusion" in advanced_options:
                try:
                    if respiratory_cardiac_fusion is None:
                        logger.warning("respiratory_cardiac_fusion function not available")
                        results.append(html.Div([
                            html.Strong("Respiratory-Cardiac Fusion RR: "),
                            html.Span("Function not available", className="text-warning")
                        ], className="mb-2"))
                    else:
                        # For demonstration, use the same signal as both respiratory and cardiac
                        # In real applications, you would have separate signals
                        fusion_rr = respiratory_cardiac_fusion(
                            signal_data, signal_data, sampling_freq,
                            preprocess='bandpass',
                            lowcut=low_cut, highcut=high_cut
                        )
                        
                        results.append(html.Div([
                            html.Strong("Respiratory-Cardiac Fusion RR: "),
                            html.Span(f"{fusion_rr:.2f} BPM", className="text-info")
                        ], className="mb-2"))
                    
                except Exception as e:
                    logger.warning(f"Respiratory-cardiac fusion failed: {e}")
            
            # Multimodal Fusion
            if "multimodal_fusion" in advanced_options:
                try:
                    if multimodal_analysis is None:
                        logger.warning("multimodal_analysis function not available")
                        results.append(html.Div([
                            html.Strong("Multimodal Fusion RR: "),
                            html.Span("Function not available", className="text-warning")
                        ], className="mb-2"))
                    else:
                        # For demonstration, use the same signal as multiple modalities
                        # In real applications, you would have different signal types
                        multimodal_rr = multimodal_analysis(
                            [signal_data, signal_data],  # Using same signal for demo
                            sampling_freq,
                            preprocess='bandpass',
                            lowcut=low_cut, highcut=high_cut
                        )
                        
                        results.append(html.Div([
                            html.Strong("Multimodal Fusion RR: "),
                            html.Span(f"{multimodal_rr:.2f} BPM", className="text-info")
                        ], className="mb-2"))
                    
                except Exception as e:
                    logger.warning(f"Multimodal fusion failed: {e}")
            
            # Quality Assessment
            if "quality_assessment" in advanced_options:
                try:
                    # Basic signal quality metrics
                    snr = 10 * np.log10(np.var(signal_data) / np.var(signal_data - np.mean(signal_data)))
                    dynamic_range = np.max(signal_data) - np.min(signal_data)
                    
                    # Calculate signal-to-noise ratio using different methods
                    # Method 1: Variance-based SNR
                    signal_power = np.var(signal_data)
                    noise_power = np.var(signal_data - np.mean(signal_data))
                    snr_variance = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
                    
                    # Method 2: Peak-to-peak SNR
                    peak_to_peak = np.max(signal_data) - np.min(signal_data)
                    rms_noise = np.sqrt(np.mean((signal_data - np.mean(signal_data))**2))
                    snr_pp = 20 * np.log10(peak_to_peak / (2 * rms_noise)) if rms_noise > 0 else 0
                    
                    results.append(html.Div([
                        html.Strong("Signal Quality Assessment: "),
                        html.Br(),
                        html.Small(f"Variance SNR: {snr_variance:.2f} dB", className="text-muted"),
                        html.Br(),
                        html.Small(f"Peak-to-Peak SNR: {snr_pp:.2f} dB", className="text-muted"),
                        html.Br(),
                        html.Small(f"Dynamic Range: {dynamic_range:.3f}", className="text-muted"),
                        html.Br(),
                        html.Small(f"Signal Power: {signal_power:.3f}", className="text-muted")
                    ], className="mb-2"))
                    
                except Exception as e:
                    logger.warning(f"Quality assessment failed: {e}")
        
        # Summary Statistics
        logger.info("Adding summary statistics section...")
        results.append(html.Hr())
        results.append(html.H5("📊 Signal Statistics", className="mb-3"))
        
        mean_val = np.mean(signal_data)
        std_val = np.std(signal_data)
        min_val = np.min(signal_data)
        max_val = np.max(signal_data)
        
        results.append(html.Div([
            html.Small(f"Mean: {mean_val:.3f}", className="text-muted"),
            html.Br(),
            html.Small(f"Std: {std_val:.3f}", className="text-muted"),
            html.Br(),
            html.Small(f"Range: {min_val:.3f} to {max_val:.3f}", className="text-muted"),
            html.Br(),
            html.Small(f"Duration: {len(signal_data)/sampling_freq:.1f}s", className="text-muted")
        ], className="mb-2"))
        
        # Additional Respiratory Metrics
        logger.info("Calculating additional respiratory metrics...")
        try:
            # Calculate respiratory rate variability from peak intervals
            peaks, _ = signal.find_peaks(signal_data, distance=int(0.5 * sampling_freq))
            logger.info(f"Found {len(peaks)} peaks for additional metrics")
            if len(peaks) > 1:
                breath_intervals = np.diff(peaks) / sampling_freq
                rr_mean = 60 / np.mean(breath_intervals)
                rr_std = 60 * np.std(breath_intervals) / (np.mean(breath_intervals) ** 2)
                
                results.append(html.Div([
                    html.Strong("Respiratory Rate Metrics: "),
                    html.Br(),
                    html.Small(f"Mean RR: {rr_mean:.1f} BPM", className="text-muted"),
                    html.Br(),
                    html.Small(f"RR Variability: {rr_std:.1f} BPM", className="text-muted"),
                    html.Br(),
                    html.Small(f"Number of breaths: {len(peaks)}", className="text-muted")
                ], className="mb-2"))
                
        except Exception as e:
            logger.warning(f"Additional respiratory metrics failed: {e}")
        
        logger.info(f"Final results list length: {len(results)}")
        logger.info("=== GENERATE COMPREHENSIVE RESPIRATORY ANALYSIS COMPLETED ===")
        return html.Div(results)
        
    except Exception as e:
        logger.error(f"Error generating respiratory analysis results: {e}")
        return html.Div([
            html.H5("❌ Analysis Failed", className="text-danger"),
            html.P(f"Error: {str(e)}", className="text-muted")
        ])


def create_comprehensive_respiratory_plots(signal_data, time_axis, sampling_freq, signal_type, 
                                         estimation_methods, advanced_options, preprocessing_options, 
                                         low_cut, high_cut):
    """Create comprehensive respiratory analysis plots."""
    logger.info("=== CREATE COMPREHENSIVE RESPIRATORY PLOTS STARTED ===")
    logger.info(f"Input parameters:")
    logger.info(f"  - signal_data shape: {signal_data.shape}")
    logger.info(f"  - time_axis shape: {time_axis.shape}")
    logger.info(f"  - sampling_freq: {sampling_freq}")
    logger.info(f"  - signal_type: {signal_type}")
    logger.info(f"  - estimation_methods: {estimation_methods}")
    logger.info(f"  - advanced_options: {advanced_options}")
    logger.info(f"  - preprocessing_options: {preprocessing_options}")
    logger.info(f"  - low_cut: {low_cut}")
    logger.info(f"  - high_cut: {high_cut}")
    
    try:
        # Create subplots for different analyses
        logger.info("Creating subplots...")
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Time Domain Signal', 'Frequency Domain', 
                           'Breathing Pattern', 'Respiratory Rate Over Time',
                           'Sleep Apnea Detection', 'Signal Quality'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # 1. Time Domain Signal
        fig.add_trace(
            go.Scatter(x=time_axis, y=signal_data, mode='lines', name='Original Signal',
                      line=dict(color='#2E86AB', width=2)),
            row=1, col=1
        )
        
        # Apply preprocessing if selected
        if preprocessing_options and "bandpass" in preprocessing_options:
            try:
                nyquist = sampling_freq / 2
                low = low_cut / nyquist
                high = high_cut / nyquist
                
                if low < high < 1.0:
                    b, a = signal.butter(4, [low, high], btype='bandpass')
                    filtered_signal = signal.filtfilt(b, a, signal_data)
                    fig.add_trace(
                        go.Scatter(x=time_axis, y=filtered_signal, mode='lines', 
                                 name='Filtered Signal', line=dict(color='#E74C3C', width=2)),
                        row=1, col=1
                    )
            except Exception as e:
                logger.error(f"Filtering failed: {e}")
        
        # 2. Frequency Domain
        fft_result = np.abs(np.fft.rfft(signal_data))
        freqs = np.fft.rfftfreq(len(signal_data), 1/sampling_freq)
        
        fig.add_trace(
            go.Scatter(x=freqs, y=fft_result, mode='lines', name='FFT Spectrum',
                      line=dict(color='#9B59B6', width=2)),
            row=1, col=2
        )
        
        # Highlight respiratory frequency band
        resp_mask = (freqs >= 0.1) & (freqs <= 0.5)
        if np.any(resp_mask):
            fig.add_trace(
                go.Scatter(x=freqs[resp_mask], y=fft_result[resp_mask], 
                          mode='lines', name='Respiratory Band',
                          line=dict(color='#E67E22', width=3)),
                row=1, col=2
            )
            
            # Add respiratory frequency annotations
            peak_freq_idx = np.argmax(fft_result[resp_mask])
            peak_freq = freqs[resp_mask][peak_freq_idx]
            peak_magnitude = fft_result[resp_mask][peak_freq_idx]
            
            fig.add_annotation(
                x=peak_freq, y=peak_magnitude,
                text=f"Peak: {peak_freq:.2f} Hz<br>({peak_freq*60:.1f} BPM)",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#E67E22",
                ax=0.1, ay=-40,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#E67E22"
            )
        
        # 3. Breathing Pattern
        if estimation_methods and "peaks" in estimation_methods:
            try:
                prominence = 0.3 * np.std(signal_data)
                distance = int(0.5 * sampling_freq)
                peaks, _ = signal.find_peaks(signal_data, prominence=prominence, distance=distance)
                
                if len(peaks) > 0:
                    fig.add_trace(
                        go.Scatter(x=time_axis[peaks], y=signal_data[peaks], 
                                 mode='markers', name='Breathing Peaks',
                                 marker=dict(color='red', size=8, symbol='diamond')),
                        row=2, col=1
                    )
                    
                    # Add breath intervals
                    if len(peaks) > 1:
                        breath_intervals = np.diff(peaks) / sampling_freq
                        interval_times = time_axis[peaks[1:]]
                        fig.add_trace(
                            go.Scatter(x=interval_times, y=breath_intervals, 
                                     mode='lines+markers', name='Breath Intervals',
                                     line=dict(color='#27AE60', width=2)),
                            row=2, col=2
                        )
                        
            except Exception as e:
                logger.error(f"Breathing pattern analysis failed: {e}")
        
        # 4. Sleep Apnea Detection
        if advanced_options and "sleep_apnea" in advanced_options:
            try:
                # Amplitude-based apnea detection
                apnea_threshold = 0.3 * np.std(signal_data)
                
                if detect_apnea_amplitude is None:
                    logger.warning("detect_apnea_amplitude function not available for plots")
                    apnea_events = []
                else:
                    apnea_events = detect_apnea_amplitude(
                        signal_data, sampling_freq, 
                        threshold=apnea_threshold, 
                        min_duration=5
                    )
                
                # Pause-based apnea detection
                if detect_apnea_pauses is None:
                    logger.warning("detect_apnea_pauses function not available for plots")
                    pause_apnea_events = []
                else:
                    pause_apnea_events = detect_apnea_pauses(
                        signal_data, sampling_freq, 
                        min_pause_duration=5
                    )
                
                # Plot all apnea events
                all_apnea_events = apnea_events + pause_apnea_events
                if all_apnea_events:
                    for start, end in all_apnea_events:
                        start_idx = int(start * sampling_freq)
                        end_idx = int(end * sampling_freq)
                        if start_idx < len(time_axis) and end_idx < len(time_axis):
                            fig.add_trace(
                                go.Scatter(x=time_axis[start_idx:end_idx], 
                                         y=signal_data[start_idx:end_idx],
                                         mode='lines', name='Apnea Event',
                                         line=dict(color='red', width=3)),
                                row=3, col=1
                            )
                    
                    # Add apnea threshold line
                    fig.add_hline(y=apnea_threshold, line_dash="dash", line_color="red",
                                annotation_text="Apnea Threshold", row=3, col=1)
                    fig.add_hline(y=-apnea_threshold, line_dash="dash", line_color="red",
                                row=3, col=1)
                            
            except Exception as e:
                logger.error(f"Sleep apnea detection failed: {e}")
        
        # 5. Signal Quality
        try:
            # Calculate moving average for trend
            window_size = int(0.5 * sampling_freq)
            if window_size > 0:
                moving_avg = np.convolve(signal_data, np.ones(window_size)/window_size, mode='same')
                fig.add_trace(
                    go.Scatter(x=time_axis, y=moving_avg, mode='lines', 
                             name='Moving Average', line=dict(color='#F39C12', width=2)),
                    row=3, col=2
                )
                
                # Add signal envelope
                upper_envelope = moving_avg + 2 * np.std(signal_data)
                lower_envelope = moving_avg - 2 * np.std(signal_data)
                
                fig.add_trace(
                    go.Scatter(x=time_axis, y=upper_envelope, mode='lines', 
                             name='Upper Envelope', line=dict(color='#E74C3C', width=1, dash='dot'),
                             opacity=0.7),
                    row=3, col=2
                )
                
                fig.add_trace(
                    go.Scatter(x=time_axis, y=lower_envelope, mode='lines', 
                             name='Lower Envelope', line=dict(color='#E74C3C', width=1, dash='dot'),
                             opacity=0.7),
                    row=3, col=2
                )
                
        except Exception as e:
            logger.error(f"Signal quality analysis failed: {e}")
        
        # Update layout
        logger.info("Updating plot layout...")
        fig.update_layout(
            title="Comprehensive Respiratory Analysis",
            template="plotly_white",
            height=800,
            showlegend=True,
            margin=dict(l=40, r=40, t=80, b=40)
        )
        
        # Update axes labels
        logger.info("Updating axes labels...")
        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)
        fig.update_yaxes(title_text="Magnitude", row=1, col=2)
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="Amplitude", row=2, col=1)
        fig.update_xaxes(title_text="Time (s)", row=2, col=2)
        fig.update_yaxes(title_text="Interval (s)", row=2, col=2)
        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
        fig.update_yaxes(title_text="Amplitude", row=3, col=1)
        fig.update_xaxes(title_text="Time (s)", row=3, col=2)
        fig.update_yaxes(title_text="Amplitude", row=3, col=2)
        
        logger.info("=== CREATE COMPREHENSIVE RESPIRATORY PLOTS COMPLETED ===")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating comprehensive respiratory plots: {e}")
        return create_empty_figure()
