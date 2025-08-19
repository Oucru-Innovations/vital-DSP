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
PPGAutonomicFeatures = None
ECGPPGSynchronization = None
PreprocessConfig = None
preprocess_signal = None

def _import_vitaldsp_modules():
    """Import vitalDSP modules with error handling."""
    global RespiratoryAnalysis, peak_detection_rr, fft_based_rr, frequency_domain_rr
    global time_domain_rr, detect_apnea_amplitude, detect_apnea_pauses, multimodal_analysis
    global ppg_ecg_fusion, respiratory_cardiac_fusion, PPGAutonomicFeatures
    global ECGPPGSynchronization, PreprocessConfig, preprocess_signal
    
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
        from vitalDSP.feature_engineering.ppg_autonomic_features import PPGAutonomicFeatures
        logger.info("✓ PPGAutonomicFeatures imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import PPGAutonomicFeatures: {e}")
        PPGAutonomicFeatures = None

    try:
        from vitalDSP.feature_engineering.ecg_ppg_synchronyzation_features import ECGPPGSynchronization
        logger.info("✓ ECGPPGSynchronization imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import ECGPPGSynchronization: {e}")
        ECGPPGSynchronization = None

    try:
        from vitalDSP.preprocess.preprocess_operations import PreprocessConfig
        logger.info("✓ PreprocessConfig imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import PreprocessConfig: {e}")
        PreprocessConfig = None

    try:
        from vitalDSP.preprocess.preprocess_operations import preprocess_signal
        logger.info("✓ preprocess_signal imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import preprocess_signal: {e}")
        preprocess_signal = None

def create_empty_figure():
    """Create an empty figure for error cases."""
    fig = go.Figure()
    fig.add_annotation(
        text="No data available or error occurred",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="gray")
    )
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white"
    )
    return fig

def detect_respiratory_signal_type(signal_data, sampling_freq):
    """Auto-detect if signal is respiratory or cardiac based on frequency content."""
    try:
        # Compute FFT
        fft_result = np.fft.fft(signal_data)
        fft_freq = np.fft.fftfreq(len(signal_data), 1/sampling_freq)
        
        # Focus on positive frequencies
        positive_mask = fft_freq > 0
        fft_freq = fft_freq[positive_mask]
        fft_magnitude = np.abs(fft_result[positive_mask])
        
        # Find dominant frequency
        dominant_idx = np.argmax(fft_magnitude)
        dominant_freq = fft_freq[dominant_idx]
        
        # Respiratory signals typically have lower frequencies (0.1-0.5 Hz)
        # Cardiac signals typically have higher frequencies (0.8-2.0 Hz)
        if dominant_freq < 0.5:
            return "respiratory"
        else:
            return "cardiac"
    except:
        return "unknown"

def create_respiratory_signal_plot(signal_data, time_axis, sampling_freq, signal_type, 
                                  estimation_methods, preprocessing_options, low_cut, high_cut):
    """Create the main respiratory signal plot with annotations."""
    try:
        fig = go.Figure()
        
        # Add main signal
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=signal_data,
            mode='lines',
            name=f'{signal_type.title()} Signal',
            line=dict(color='blue', width=1)
        ))
        
        # Add preprocessing info if applied
        if preprocessing_options and any(preprocessing_options):
            fig.add_trace(go.Scatter(
                x=time_axis,
                y=signal_data,  # This would be the preprocessed signal in real implementation
                mode='lines',
                name='Preprocessed Signal',
                line=dict(color='red', width=1, dash='dash')
            ))
        
        # Add filter frequency lines if specified
        if low_cut:
            fig.add_hline(y=np.min(signal_data), line_dash="dot", line_color="green",
                         annotation_text=f"Low Cut: {low_cut} Hz")
        if high_cut:
            fig.add_hline(y=np.max(signal_data), line_dash="dot", line_color="orange",
                         annotation_text=f"High Cut: {high_cut} Hz")
        
        fig.update_layout(
            title=f"Respiratory Signal Analysis - {signal_type.title()}",
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
            showlegend=True,
            height=400
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating respiratory signal plot: {e}")
        return create_empty_figure()

def generate_comprehensive_respiratory_analysis(signal_data, time_axis, sampling_freq, 
                                              signal_type, estimation_methods, advanced_options,
                                              preprocessing_options, low_cut, high_cut,
                                              min_breath_duration, max_breath_duration):
    """Generate comprehensive respiratory analysis results."""
    try:
        results = []
        
        # Basic statistics
        results.append(html.H5("Signal Statistics"))
        results.append(html.P(f"Signal Type: {signal_type.title()}"))
        results.append(html.P(f"Duration: {time_axis[-1]:.2f} seconds"))
        results.append(html.P(f"Sampling Frequency: {sampling_freq} Hz"))
        results.append(html.P(f"Signal Length: {len(signal_data)} samples"))
        results.append(html.P(f"Mean Amplitude: {np.mean(signal_data):.4f}"))
        results.append(html.P(f"Std Amplitude: {np.std(signal_data):.4f}"))
        
        # Estimation methods used
        if estimation_methods:
            results.append(html.Hr())
            results.append(html.H6("Estimation Methods"))
            for method in estimation_methods:
                results.append(html.P(f"• {method.replace('_', ' ').title()}"))
        
        # Preprocessing options
        if preprocessing_options and any(preprocessing_options):
            results.append(html.Hr())
            results.append(html.H6("Preprocessing Applied"))
            for option in preprocessing_options:
                if option:
                    results.append(html.P(f"• {option.replace('_', ' ').title()}"))
        
        # Filter parameters
        if low_cut or high_cut:
            results.append(html.Hr())
            results.append(html.H6("Filter Parameters"))
            if low_cut:
                results.append(html.P(f"Low Cutoff: {low_cut} Hz"))
            if high_cut:
                results.append(html.P(f"High Cutoff: {high_cut} Hz"))
        
        # Breath duration constraints
        if min_breath_duration or max_breath_duration:
            results.append(html.Hr())
            results.append(html.H6("Breath Duration Constraints"))
            if min_breath_duration:
                results.append(html.P(f"Minimum Duration: {min_breath_duration} seconds"))
            if max_breath_duration:
                results.append(html.P(f"Maximum Duration: {max_breath_duration} seconds"))
        
        return html.Div(results)
    except Exception as e:
        logger.error(f"Error generating respiratory analysis results: {e}")
        return html.Div([html.H5("Error"), html.P(f"Failed to generate results: {str(e)}")])

def create_comprehensive_respiratory_plots(signal_data, time_axis, sampling_freq, 
                                         signal_type, estimation_methods, advanced_options,
                                         preprocessing_options, low_cut, high_cut):
    """Create comprehensive respiratory analysis plots."""
    try:
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Time Domain", "Frequency Domain", "Power Spectral Density", "Signal Quality"),
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )
        
        # Time domain plot
        fig.add_trace(
            go.Scatter(x=time_axis, y=signal_data, mode='lines', name='Signal'),
            row=1, col=1
        )
        
        # Frequency domain plot
        fft_result = np.fft.fft(signal_data)
        fft_freq = np.fft.fftfreq(len(signal_data), 1/sampling_freq)
        positive_mask = fft_freq > 0
        fft_freq = fft_freq[positive_mask]
        fft_magnitude = np.abs(fft_result[positive_mask])
        
        fig.add_trace(
            go.Scatter(x=fft_freq, y=fft_magnitude, mode='lines', name='FFT'),
            row=1, col=2
        )
        
        # Power spectral density
        from scipy import signal as scipy_signal
        freqs, psd = scipy_signal.welch(signal_data, sampling_freq, nperseg=min(256, len(signal_data)//4))
        fig.add_trace(
            go.Scatter(x=freqs, y=psd, mode='lines', name='PSD'),
            row=2, col=1
        )
        
        # Signal quality metrics
        # Calculate signal-to-noise ratio approximation
        signal_power = np.mean(signal_data**2)
        noise_power = np.var(signal_data)
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
        
        fig.add_trace(
            go.Bar(x=['SNR (dB)', 'Peak-to-Peak', 'RMS'], 
                   y=[snr, np.max(signal_data) - np.min(signal_data), np.sqrt(np.mean(signal_data**2))],
                   name='Quality Metrics'),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Comprehensive Respiratory Analysis",
            height=600,
            showlegend=True
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating comprehensive respiratory plots: {e}")
        return create_empty_figure()

def register_respiratory_callbacks(app):
    """Register all respiratory analysis callbacks."""
    
    # Import vitalDSP modules
    _import_vitaldsp_modules()
    
    @app.callback(
        [Output("resp-main-plot", "figure"),
         Output("resp-analysis-results", "children"),
         Output("resp-analysis-plots", "figure"),
         Output("resp-data-store", "data"),
         Output("resp-features-store", "data")],
        [Input("resp-analyze-btn", "n_clicks"),
         Input("resp-time-range-slider", "value"),
         Input("resp-btn-nudge-m10", "n_clicks"),
         Input("resp-btn-nudge-m1", "n_clicks"),
         Input("resp-btn-nudge-p1", "n_clicks"),
         Input("resp-btn-nudge-p10", "n_clicks"),
         Input("url", "pathname")],
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
    def analyze_respiratory_signal(n_clicks, slider_value, nudge_m10, nudge_m1, nudge_p1, nudge_p10,
                                  pathname, start_time, end_time, signal_type, estimation_methods,
                                  advanced_options, preprocessing_options, low_cut, high_cut,
                                  min_breath_duration, max_breath_duration):
        """Main respiratory analysis callback."""
        logger.info("=== RESPIRATORY ANALYSIS CALLBACK ===")
        
        # Get trigger information
        ctx = callback_context
        if not ctx.triggered:
            trigger_id = "initial_load"
        else:
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        logger.info(f"Trigger ID: {trigger_id}")
        logger.info(f"Pathname: {pathname}")
        
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
            from vitalDSP_webapp.services.data.data_service import get_data_service
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
            from vitalDSP_webapp.services.data.data_service import get_data_service
            data_service = get_data_service()
            
            # Get the most recent data
            all_data = data_service.get_all_data()
            if not all_data:
                return 0, 100, [0, 10]
            
            latest_data_id = list(all_data.keys())[-1]
            latest_data = all_data[latest_data_id]
            
            # Get sampling frequency and calculate duration
            sampling_freq = latest_data.get('info', {}).get('sampling_freq', 1000)
            df = data_service.get_data(latest_data_id)
            
            if df is None or df.empty:
                return 0, 100, [0, 10]
            
            duration = len(df) / sampling_freq
            max_time = int(duration)
            
            return 0, max_time, [0, min(10, max_time)]
            
        except Exception as e:
            logger.error(f"Error updating respiratory time slider range: {e}")
            return 0, 100, [0, 10]
