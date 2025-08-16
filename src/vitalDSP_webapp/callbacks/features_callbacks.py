"""
Feature engineering callbacks for vitalDSP webapp.
This module handles all feature engineering operations including PPG autonomic features,
ECG autonomic features, morphology features, light-based features, and synchronization features.
"""

import logging
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback_context, no_update
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from dash import html, dcc

# Initialize logger
logger = logging.getLogger(__name__)

# Global variables for vitalDSP modules (initialized to None)
PPGAutonomicFeatures = None
PhysiologicalFeatureExtractor = None
ECGExtractor = None
PPGLightFeatureExtractor = None
ECGPPGSynchronization = None

def _import_vitaldsp_modules():
    """Import vitalDSP modules with error handling."""
    global PPGAutonomicFeatures, PhysiologicalFeatureExtractor, ECGExtractor
    global PPGLightFeatureExtractor, ECGPPGSynchronization
    
    logger.info("=== IMPORTING VITALDSP FEATURE ENGINEERING MODULES ===")
    
    # Check if vitalDSP package is available
    try:
        import vitalDSP
        logger.info(f"✓ vitalDSP package found: {vitalDSP.__file__}")
    except ImportError as e:
        logger.error(f"✗ vitalDSP package not found: {e}")
        return
    
    # Check feature_engineering module
    try:
        import vitalDSP.feature_engineering
        logger.info(f"✓ vitalDSP.feature_engineering module found: {vitalDSP.feature_engineering.__file__}")
    except ImportError as e:
        logger.error(f"✗ vitalDSP.feature_engineering module not found: {e}")
        return
    
    try:
        from vitalDSP.feature_engineering.ppg_autonomic_features import PPGAutonomicFeatures
        logger.info("✓ PPGAutonomicFeatures imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import PPGAutonomicFeatures: {e}")
        logger.error(f"Available modules in feature_engineering: {dir(vitalDSP.feature_engineering)}")
        PPGAutonomicFeatures = None
    
    try:
        from vitalDSP.feature_engineering.morphology_features import PhysiologicalFeatureExtractor
        logger.info("✓ PhysiologicalFeatureExtractor imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import PhysiologicalFeatureExtractor: {e}")
        PhysiologicalFeatureExtractor = None
    
    try:
        from vitalDSP.feature_engineering.ecg_autonomic_features import ECGExtractor
        logger.info("✓ ECGExtractor imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import ECGExtractor: {e}")
        ECGExtractor = None
    
    try:
        from vitalDSP.feature_engineering.ppg_light_features import PPGLightFeatureExtractor
        logger.info("✓ PPGLightFeatureExtractor imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import PPGLightFeatureExtractor: {e}")
        PPGLightFeatureExtractor = None
    
    try:
        from vitalDSP.feature_engineering.ecg_ppg_synchronyzation_features import ECGPPGSynchronization
        logger.info("✓ ECGPPGSynchronization imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import ECGPPGSynchronization: {e}")
        ECGPPGSynchronization = None
    
    logger.info("=== VITALDSP FEATURE ENGINEERING MODULE IMPORT COMPLETED ===")

def convert_numpy_to_python(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if hasattr(obj, 'item'):
        return obj.item()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.dtype):
        return str(obj)  # Convert numpy dtype to string
    elif isinstance(obj, np.bool_):
        return bool(obj)  # Convert numpy boolean to Python boolean
    elif isinstance(obj, np.void):
        return str(obj)  # Convert numpy void to string
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    else:
        return obj

def clean_plotly_figure(fig):
    """Ensure Plotly figure is JSON serializable by converting all numpy types."""
    # Convert the figure to a dict and clean it
    fig_dict = fig.to_dict()
    
    def clean_dict(d):
        """Recursively clean dictionary values."""
        if isinstance(d, dict):
            return {k: clean_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [clean_dict(item) for item in d]
        elif hasattr(d, 'item'):
            return d.item()
        elif isinstance(d, np.integer):
            return int(d)
        elif isinstance(d, np.floating):
            return float(d)
        elif isinstance(d, np.ndarray):
            return d.tolist()
        elif isinstance(d, np.dtype):
            return str(d)  # Convert numpy dtype to string
        elif isinstance(d, np.bool_):
            return bool(d)  # Convert numpy boolean to Python boolean
        elif isinstance(d, np.void):
            return str(d)  # Convert numpy void to string
        else:
            return d
    
    cleaned_dict = clean_dict(fig_dict)
    
    # Create a new figure from the cleaned dict
    return go.Figure(cleaned_dict)

def create_empty_figure():
    """Create an empty figure for when no data is available."""
    fig = go.Figure()
    fig.add_annotation(
        text="No data available for analysis",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="gray")
    )
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    return fig

def generate_basic_features(signal_data, sampling_freq):
    """Generate basic features when vitalDSP modules are not available."""
    logger.info("Generating basic features using numpy/scipy...")
    
    features = {}
    
    try:
        # Basic statistical features
        features['mean'] = float(np.mean(signal_data))
        features['std'] = float(np.std(signal_data))
        features['min'] = float(np.min(signal_data))
        features['max'] = float(np.max(signal_data))
        features['range'] = float(np.max(signal_data) - np.min(signal_data))
        features['variance'] = float(np.var(signal_data))
        
        # Basic frequency domain features
        fft_vals = np.abs(np.fft.fft(signal_data))
        features['dominant_freq'] = float(np.argmax(fft_vals) * sampling_freq / len(signal_data))
        features['spectral_energy'] = float(np.sum(fft_vals**2))
        
        # Basic time domain features
        features['zero_crossings'] = int(np.sum(np.diff(np.signbit(signal_data).astype(int)) != 0))
        
        logger.info(f"Basic features generated: {len(features)} features")
        
    except Exception as e:
        logger.error(f"Error generating basic features: {e}")
        features = {'error': str(e)}
    
    return features

def generate_comprehensive_feature_analysis(signal_data, sampling_freq, signal_type="ppg"):
    """
    Generate comprehensive feature analysis using all available vitalDSP feature engineering modules.
    
    Parameters
    ----------
    signal_data : np.ndarray
        Input signal data
    sampling_freq : float
        Sampling frequency
    signal_type : str
        Type of signal ("ppg", "ecg", or "both")
    
    Returns
    -------
    dict
        Dictionary containing all extracted features
    """
    logger.info(f"=== GENERATING COMPREHENSIVE FEATURE ANALYSIS ===")
    logger.info(f"Signal type: {signal_type}")
    logger.info(f"Signal shape: {signal_data.shape}")
    logger.info(f"Sampling frequency: {sampling_freq}")
    
    features = {}
    
    # Check if any vitalDSP modules are available
    if all([PPGAutonomicFeatures is None, PhysiologicalFeatureExtractor is None, 
            ECGExtractor is None, PPGLightFeatureExtractor is None, ECGPPGSynchronization is None]):
        logger.warning("No vitalDSP modules available, using basic feature extraction")
        return generate_basic_features(signal_data, sampling_freq)
    
    # PPG Autonomic Features
    if PPGAutonomicFeatures is not None:
        logger.info("Extracting PPG autonomic features...")
        try:
            ppg_features = PPGAutonomicFeatures(signal_data, sampling_freq)
            
            # Respiratory Rate Variability (RRV)
            try:
                rrv = ppg_features.compute_rrv()
                features['rrv'] = rrv
                logger.info(f"RRV computed: {rrv}")
            except Exception as e:
                logger.warning(f"Failed to compute RRV: {e}")
                features['rrv'] = None
            
            # Respiratory Sinus Arrhythmia (RSA)
            try:
                rsa = ppg_features.compute_rsa()
                features['rsa'] = rsa
                logger.info(f"RSA computed: {rsa}")
            except Exception as e:
                logger.warning(f"Failed to compute RSA: {e}")
                features['rsa'] = None
            
            # Fractal Dimension
            try:
                fractal_dim = ppg_features.compute_fractal_dimension()
                features['fractal_dimension'] = fractal_dim
                logger.info(f"Fractal dimension computed: {fractal_dim}")
            except Exception as e:
                logger.warning(f"Failed to compute fractal dimension: {e}")
                features['fractal_dimension'] = None
            
            # Detrended Fluctuation Analysis (DFA)
            try:
                dfa = ppg_features.compute_dfa()
                features['dfa'] = dfa
                logger.info(f"DFA computed: {dfa}")
            except Exception as e:
                logger.warning(f"Failed to compute DFA: {e}")
                features['dfa'] = None
                
        except Exception as e:
            logger.error(f"Failed to initialize PPG autonomic features: {e}")
            features['ppg_autonomic'] = None
    else:
        logger.warning("PPGAutonomicFeatures not available")
        features['ppg_autonomic'] = None
    
    # Morphology Features
    if PhysiologicalFeatureExtractor is not None:
        logger.info("Extracting morphology features...")
        try:
            morphology_extractor = PhysiologicalFeatureExtractor(signal_data, sampling_freq)
            
            # Extract comprehensive features
            try:
                morphology_features = morphology_extractor.extract_features(signal_type="PPG")
                logger.info(f"Morphology features extracted: {len(morphology_features)} features")
                
                # Process each morphology feature individually
                for key, value in morphology_features.items():
                    if value is not None and not np.isnan(value):
                        # Convert numpy types to Python types
                        if hasattr(value, 'item'):
                            features[f"morphology_{key}"] = value.item()
                        elif isinstance(value, np.integer):
                            features[f"morphology_{key}"] = int(value)
                        elif isinstance(value, np.floating):
                            features[f"morphology_{key}"] = float(value)
                        else:
                            features[f"morphology_{key}"] = value
                        logger.info(f"Morphology {key}: {features[f'morphology_{key}']}")
                    else:
                        logger.warning(f"Morphology {key} is None or NaN")
                        
            except Exception as e:
                logger.warning(f"Failed to extract morphology features: {e}")
                
        except Exception as e:
            logger.error(f"Failed to initialize morphology feature extractor: {e}")
    else:
        logger.warning("PhysiologicalFeatureExtractor not available")
    
    # ECG Autonomic Features (if ECG data available)
    if ECGExtractor is not None and signal_type in ["ecg", "both"]:
        logger.info("Extracting ECG autonomic features...")
        try:
            ecg_extractor = ECGExtractor(signal_data, sampling_freq)
            
            # P-wave duration
            try:
                p_wave_duration = ecg_extractor.compute_p_wave_duration()
                features['p_wave_duration'] = p_wave_duration
                logger.info(f"P-wave duration computed: {p_wave_duration}")
            except Exception as e:
                logger.warning(f"Failed to compute P-wave duration: {e}")
                features['p_wave_duration'] = None
            
            # PR interval
            try:
                pr_interval = ecg_extractor.compute_pr_interval()
                features['pr_interval'] = pr_interval
                logger.info(f"PR interval computed: {pr_interval}")
            except Exception as e:
                logger.warning(f"Failed to compute PR interval: {e}")
                features['pr_interval'] = None
            
            # QRS duration
            try:
                qrs_duration = ecg_extractor.compute_qrs_duration()
                features['qrs_duration'] = qrs_duration
                logger.info(f"QRS duration computed: {qrs_duration}")
            except Exception as e:
                logger.warning(f"Failed to compute QRS duration: {e}")
                features['qrs_duration'] = None
            
            # ST segment
            try:
                st_segment = ecg_extractor.compute_st_segment()
                features['st_segment'] = st_segment
                logger.info(f"ST segment computed: {st_segment}")
            except Exception as e:
                logger.warning(f"Failed to compute ST segment: {e}")
                features['st_segment'] = None
            
            # QT interval
            try:
                qt_interval = ecg_extractor.compute_qt_interval()
                features['qt_interval'] = qt_interval
                logger.info(f"QT interval computed: {qt_interval}")
            except Exception as e:
                logger.warning(f"Failed to compute QT interval: {e}")
                features['qt_interval'] = None
            
            # Arrhythmia detection
            try:
                arrhythmias = ecg_extractor.detect_arrhythmias()
                features['arrhythmias'] = arrhythmias
                logger.info(f"Arrhythmias detected: {arrhythmias}")
            except Exception as e:
                logger.warning(f"Failed to detect arrhythmias: {e}")
                features['arrhythmias'] = None
                
        except Exception as e:
            logger.error(f"Failed to initialize ECG feature extractor: {e}")
            features['ecg_autonomic'] = None
    else:
        logger.warning("ECGExtractor not available or ECG data not provided")
        features['ecg_autonomic'] = None
    
    # PPG Light Features (if IR/Red data available)
    if PPGLightFeatureExtractor is not None:
        logger.info("Extracting PPG light features...")
        try:
            # For now, we'll use the main signal data
            # In a real scenario, you'd need separate IR and Red channels
            light_extractor = PPGLightFeatureExtractor(signal_data, signal_data, sampling_freq)
            
            # SpO2 estimation
            try:
                spo2_values, spo2_times = light_extractor.calculate_spo2()
                if len(spo2_values) > 0:
                    spo2_mean = np.mean(spo2_values)
                    # Convert numpy scalar to Python scalar
                    if hasattr(spo2_mean, 'item'):
                        features['spo2'] = spo2_mean.item()
                    elif isinstance(spo2_mean, np.floating):
                        features['spo2'] = float(spo2_mean)
                    else:
                        features['spo2'] = spo2_mean
                    logger.info(f"SpO2 estimated: {features['spo2']}")
                else:
                    features['spo2'] = None
                    logger.warning("No SpO2 values computed")
            except Exception as e:
                logger.warning(f"Failed to estimate SpO2: {e}")
                features['spo2'] = None
            
            # Perfusion Index (PI)
            try:
                pi_values, pi_times = light_extractor.calculate_perfusion_index()
                if len(pi_values) > 0:
                    pi_mean = np.mean(pi_values)
                    # Convert numpy scalar to Python scalar
                    if hasattr(pi_mean, 'item'):
                        features['perfusion_index'] = pi_mean.item()
                    elif isinstance(pi_mean, np.floating):
                        features['perfusion_index'] = float(pi_mean)
                    else:
                        features['perfusion_index'] = pi_mean
                    logger.info(f"Perfusion index computed: {features['perfusion_index']}")
                else:
                    features['perfusion_index'] = None
                    logger.warning("No PI values computed")
            except Exception as e:
                logger.warning(f"Failed to compute perfusion index: {e}")
                features['perfusion_index'] = None
            
            # Respiratory Rate from PPG
            try:
                rr_values, rr_times = light_extractor.calculate_respiratory_rate()
                if len(rr_values) > 0:
                    rr_mean = np.mean(rr_values)
                    # Convert numpy scalar to Python scalar
                    if hasattr(rr_mean, 'item'):
                        features['respiratory_rate_ppg'] = rr_mean.item()
                    elif isinstance(rr_mean, np.floating):
                        features['respiratory_rate_ppg'] = float(rr_mean)
                    else:
                        features['respiratory_rate_ppg'] = None
                    logger.info(f"Respiratory rate from PPG estimated: {features['respiratory_rate_ppg']}")
                else:
                    features['respiratory_rate_ppg'] = None
                    logger.warning("No RR values computed")
            except Exception as e:
                logger.warning(f"Failed to estimate respiratory rate from PPG: {e}")
                features['respiratory_rate_ppg'] = None
            
            # Photoplethysmogram Ratio (PPR)
            try:
                ppr_values, ppr_times = light_extractor.calculate_ppr()
                if len(ppr_values) > 0:
                    ppr_mean = np.mean(ppr_values)
                    # Convert numpy scalar to Python scalar
                    if hasattr(ppr_mean, 'item'):
                        features['ppr'] = ppr_mean.item()
                    elif isinstance(ppr_mean, np.floating):
                        features['ppr'] = float(ppr_mean)
                    else:
                        features['ppr'] = ppr_mean
                    logger.info(f"PPR computed: {features['ppr']}")
                else:
                    features['ppr'] = None
                    logger.warning("No PPR values computed")
            except Exception as e:
                logger.warning(f"Failed to compute PPR: {e}")
                features['ppr'] = None
                
        except Exception as e:
            logger.error(f"Failed to initialize PPG light feature extractor: {e}")
    else:
        logger.warning("PPGLightFeatureExtractor not available")
    
    # ECG-PPG Synchronization Features (if both signals available)
    if ECGPPGSynchronization is not None and signal_type == "both":
        logger.info("Extracting ECG-PPG synchronization features...")
        try:
            # For now, we'll use the same signal data for both
            # In a real scenario, you'd need separate ECG and PPG signals
            sync_extractor = ECGPPGSynchronization(signal_data, signal_data, sampling_freq)
            
            # Pulse Transit Time (PTT)
            try:
                ptt = sync_extractor.compute_ptt()
                features['ptt'] = ptt
                logger.info(f"PTT computed: {ptt}")
            except Exception as e:
                logger.warning(f"Failed to compute PTT: {e}")
                features['ptt'] = None
            
            # Pulse Arrival Time (PAT)
            try:
                pat = sync_extractor.compute_pat()
                features['pat'] = pat
                logger.info(f"PAT computed: {pat}")
            except Exception as e:
                logger.warning(f"Failed to compute PAT: {e}")
                features['pat'] = None
            
            # Electromechanical Delay (EMD)
            try:
                emd = sync_extractor.compute_emd()
                features['emd'] = emd
                logger.info(f"EMD computed: {emd}")
            except Exception as e:
                logger.warning(f"Failed to compute EMD: {e}")
                features['emd'] = None
            
            # Respiratory Sinus Arrhythmia (RSA)
            try:
                rsa_sync = sync_extractor.compute_rsa()
                features['rsa_sync'] = rsa_sync
                logger.info(f"RSA from synchronization computed: {rsa_sync}")
            except Exception as e:
                logger.warning(f"Failed to compute RSA from synchronization: {e}")
                features['rsa_sync'] = None
            
            # Heart Rate - Pulse Rate synchronization
            try:
                hr_pr_sync = sync_extractor.compute_hr_pr_synchronization()
                features['hr_pr_sync'] = hr_pr_sync
                logger.info(f"HR-PR synchronization computed: {hr_pr_sync}")
            except Exception as e:
                logger.warning(f"Failed to compute HR-PR synchronization: {e}")
                features['hr_pr_sync'] = None
                
        except Exception as e:
            logger.error(f"Failed to initialize ECG-PPG synchronization extractor: {e}")
            features['ecg_ppg_sync'] = None
    else:
        logger.warning("ECGPPGSynchronization not available or both signals not provided")
        features['ecg_ppg_sync'] = None
    
    logger.info("=== FEATURE EXTRACTION COMPLETED ===")
    logger.info(f"Total features extracted: {len(features)}")
    
    return features

def create_comprehensive_feature_plots(signal_data, features, sampling_freq):
    """
    Create comprehensive plots for feature analysis results.
    
    Parameters
    ----------
    signal_data : np.ndarray
        Input signal data
    features : dict
        Extracted features
    sampling_freq : float
        Sampling frequency
    
    Returns
    -------
    plotly.graph_objects.Figure
        Combined figure with all feature plots
    """
    logger.info("=== CREATING COMPREHENSIVE FEATURE PLOTS ===")
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Original Signal",
            "Feature Summary",
            "Autonomic Features",
            "Morphology Features",
            "Light Features",
            "Synchronization Features"
        ),
        specs=[
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}]
        ]
    )
    
    # Original signal - convert numpy arrays to Python lists for JSON serialization
    time_axis = (np.arange(len(signal_data)) / sampling_freq).tolist()
    signal_data_list = signal_data.tolist()
    
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=signal_data_list,
            mode='lines',
            name='Signal',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )
    
    # Feature summary bar chart
    feature_names = []
    feature_values = []
    
    for key, value in features.items():
        if value is not None and not isinstance(value, (dict, list)):
            try:
                # Convert numpy types to Python types
                if hasattr(value, 'item'):
                    value = value.item()  # Convert numpy scalar to Python scalar
                elif isinstance(value, np.integer):
                    value = int(value)
                elif isinstance(value, np.floating):
                    value = float(value)
                feature_names.append(str(key))
                feature_values.append(float(value))
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not convert feature {key}={value} to float: {e}")
                continue
    
    if feature_names:
        fig.add_trace(
            go.Bar(
                x=feature_names,
                y=feature_values,
                name='Feature Values',
                marker_color='lightblue'
            ),
            row=1, col=2
        )
    
    # Autonomic features
    if features.get('rrv') is not None:
        try:
            rrv_value = features['rrv']
            if hasattr(rrv_value, 'item'):
                rrv_value = rrv_value.item()
            elif isinstance(rrv_value, np.integer):
                rrv_value = int(rrv_value)
            elif isinstance(rrv_value, np.floating):
                rrv_value = float(rrv_value)
            fig.add_trace(
                go.Scatter(
                    x=['RRV'],
                    y=[float(rrv_value)],
                    mode='markers',
                    name='RRV',
                    marker=dict(color='red', size=10)
                ),
                row=2, col=1
            )
        except Exception as e:
            logger.warning(f"Could not plot RRV: {e}")
    
    if features.get('rsa') is not None:
        try:
            rsa_value = features['rsa']
            if hasattr(rsa_value, 'item'):
                rsa_value = rsa_value.item()
            elif isinstance(rsa_value, np.integer):
                rsa_value = int(rsa_value)
            elif isinstance(rsa_value, np.floating):
                rsa_value = float(rsa_value)
            fig.add_trace(
                go.Scatter(
                    x=['RSA'],
                    y=[float(rsa_value)],
                    mode='markers',
                    name='RSA',
                    marker=dict(color='green', size=10)
                ),
                row=2, col=1
            )
        except Exception as e:
            logger.warning(f"Could not plot RSA: {e}")
    
    # Morphology features
    if features.get('troughs') is not None:
        try:
            troughs_value = features['troughs']
            if hasattr(troughs_value, 'item'):
                troughs_value = troughs_value.item()
            elif isinstance(troughs_value, np.integer):
                troughs_value = int(troughs_value)
            elif isinstance(troughs_value, np.floating):
                troughs_value = float(troughs_value)
            fig.add_trace(
                go.Scatter(
                    x=['Troughs'],
                    y=[float(troughs_value)],
                    mode='markers',
                    name='Troughs',
                    marker=dict(color='orange', size=10)
                ),
                row=2, col=2
            )
        except Exception as e:
            logger.warning(f"Could not plot Troughs: {e}")
    
    # Light features
    if features.get('spo2') is not None:
        try:
            spo2_value = features['spo2']
            if hasattr(spo2_value, 'item'):
                spo2_value = spo2_value.item()
            elif isinstance(spo2_value, np.integer):
                spo2_value = int(spo2_value)
            elif isinstance(spo2_value, np.floating):
                spo2_value = float(spo2_value)
            fig.add_trace(
                go.Scatter(
                    x=['SpO2'],
                    y=[float(spo2_value)],
                    mode='markers',
                    name='SpO2',
                    marker=dict(color='purple', size=10)
                ),
                row=3, col=1
            )
        except Exception as e:
            logger.warning(f"Could not plot SpO2: {e}")
    
    # Synchronization features
    if features.get('ptt') is not None:
        try:
            ptt_value = features['ptt']
            if hasattr(ptt_value, 'item'):
                ptt_value = ptt_value.item()
            elif isinstance(ptt_value, np.integer):
                ptt_value = int(ptt_value)
            elif isinstance(ptt_value, np.floating):
                ptt_value = float(ptt_value)
            fig.add_trace(
                go.Scatter(
                    x=['PTT'],
                    y=[float(ptt_value)],
                    mode='markers',
                    name='PTT',
                    marker=dict(color='brown', size=10)
                ),
                row=3, col=2
            )
        except Exception as e:
            logger.warning(f"Could not plot PTT: {e}")
    
    # Update layout
    fig.update_layout(
        title="Comprehensive Feature Analysis Results",
        height=800,
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Update axes
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    logger.info("=== FEATURE PLOTS CREATED SUCCESSFULLY ===")
    
    return fig

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
         State("features-preprocessing", "value")]
    )
    def features_analysis_callback(n_clicks, pathname, signal_type, preprocessing):
        """Main callback for feature engineering analysis."""
        logger.info("=== FEATURES ANALYSIS CALLBACK TRIGGERED ===")
        
        # Only run this when we're on the features page
        if pathname != "/features":
            logger.info("Not on features page, returning empty figures")
            return "Navigate to Features page", create_empty_figure(), None, None
        
        # Check if data is available first
        try:
            from ..services.data_service import get_data_service
            data_service = get_data_service()
            
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
            logger.info(f"Latest data keys: {list(latest_data.keys())}")
            logger.info(f"Latest data info keys: {list(latest_data.get('info', {}).keys())}")
            logger.info(f"Latest data metadata keys: {list(latest_data.get('metadata', {}).keys())}")
            
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
            sampling_freq = latest_data.get('info', {}).get('sampling_freq', 100)
            logger.info(f"Sampling frequency: {sampling_freq}")
            
            # If no sampling frequency in info, try to get it from metadata
            if sampling_freq == 100:  # Default value, might not be real
                metadata = latest_data.get('metadata', {})
                if 'sampling_freq' in metadata:
                    sampling_freq = metadata['sampling_freq']
                elif 'fs' in metadata:
                    sampling_freq = metadata['fs']
                elif 'frequency' in metadata:
                    sampling_freq = metadata['frequency']
                logger.info(f"Updated sampling frequency from metadata: {sampling_freq}")
            
            # Get the main signal column (assuming it's the first mapped column)
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
            signal_type = signal_type or "ppg"
            preprocessing = preprocessing or []
            
            logger.info(f"Data shape: {signal_data.shape}")
            logger.info(f"Signal data type: {type(signal_data)}")
            logger.info(f"Signal data range: {signal_data.min():.3f} to {signal_data.max():.3f}")
            logger.info(f"Signal data mean: {signal_data.mean():.3f}")
            logger.info(f"Sampling frequency: {sampling_freq}")
            logger.info(f"Signal type: {signal_type}")
            logger.info(f"Preprocessing options: {preprocessing}")
            
            # Apply preprocessing if selected
            if preprocessing:
                logger.info("Applying preprocessing...")
                # Add preprocessing logic here if needed
                pass
            
            # Generate comprehensive feature analysis
            features = generate_comprehensive_feature_analysis(signal_data, sampling_freq, signal_type)
            
            # Create plots
            plots = create_comprehensive_feature_plots(signal_data, features, sampling_freq)
            
            # Clean the Plotly figure to ensure JSON serialization
            try:
                plots = clean_plotly_figure(plots)
                logger.info("Plotly figure cleaned successfully")
            except Exception as e:
                logger.error(f"Failed to clean Plotly figure: {e}")
                # Create a simple fallback figure
                plots = create_empty_figure()
                plots.update_layout(title="Feature Analysis Results (Simplified View)")
                logger.info("Using fallback figure due to serialization error")
            
            # Create results display
            results_html = create_feature_results_display(features)
            
            # Convert features to JSON-serializable format
            features_serializable = convert_numpy_to_python(features)
            
            logger.info("=== FEATURES ANALYSIS COMPLETED SUCCESSFULLY ===")
            
            return results_html, plots, latest_data, features_serializable
            
        except Exception as e:
            logger.error(f"Error in features analysis callback: {e}")
            import traceback
            traceback.print_exc()
            error_msg = f"Error during feature analysis: {str(e)}"
            return error_msg, create_empty_figure(), None, None
    
    logger.info("=== FEATURE ENGINEERING CALLBACKS REGISTERED SUCCESSFULLY ===")

def create_feature_results_display(features):
    """Create HTML display for feature analysis results."""
    logger.info("Creating feature results display...")
    
    if not features:
        return html.Div("No features extracted.")
    
    # Group features by category
    autonomic_features = {}
    morphology_features = {}
    light_features = {}
    sync_features = {}
    ecg_features = {}
    other_features = {}
    
    for key, value in features.items():
        if key in ['rrv', 'rsa', 'fractal_dimension', 'dfa']:
            autonomic_features[key] = value
        elif key.startswith('morphology_'):
            morphology_features[key.replace('morphology_', '')] = value
        elif key in ['spo2', 'perfusion_index', 'respiratory_rate_ppg', 'ppr']:
            light_features[key] = value
        elif key in ['ptt', 'pat', 'emd', 'rsa_sync', 'hr_pr_sync']:
            sync_features[key] = value
        elif key in ['p_wave_duration', 'pr_interval', 'qrs_duration', 'st_segment', 'qt_interval', 'arrhythmias']:
            ecg_features[key] = value
        else:
            other_features[key] = value
    
    # Create feature cards
    cards = []
    
    if autonomic_features:
        autonomic_card = dbc.Card([
            dbc.CardHeader("🧠 Autonomic Features", className="bg-primary text-white"),
            dbc.CardBody([
                html.Div([
                    html.Strong(f"{key.replace('_', ' ').title()}: "),
                    html.Span(f"{value:.4f}" if isinstance(value, (int, float)) or (hasattr(value, 'item') and isinstance(value.item(), (int, float))) else str(value))
                ]) for key, value in autonomic_features.items() if value is not None
            ])
        ], className="mb-3")
        cards.append(autonomic_card)
    
    if morphology_features:
        morphology_card = dbc.Card([
            dbc.CardHeader("📊 Morphology Features", className="bg-success text-white"),
            dbc.CardBody([
                html.Div([
                    html.Strong(f"{key.replace('_', ' ').title()}: "),
                    html.Span(f"{value:.4f}" if isinstance(value, (int, float)) or (hasattr(value, 'item') and isinstance(value.item(), (int, float))) else str(value))
                ]) for key, value in morphology_features.items() if value is not None
            ])
        ], className="mb-3")
        cards.append(morphology_card)
    
    if light_features:
        light_card = dbc.Card([
            dbc.CardHeader("💡 Light-Based Features", className="bg-warning text-dark"),
            dbc.CardBody([
                html.Div([
                    html.Strong(f"{key.replace('_', ' ').title()}: "),
                    html.Span(f"{value:.4f}" if isinstance(value, (int, float)) or (hasattr(value, 'item') and isinstance(value.item(), (int, float))) else str(value))
                ]) for key, value in light_features.items() if value is not None
            ])
        ], className="mb-3")
        cards.append(light_card)
    
    if sync_features:
        sync_card = dbc.Card([
            dbc.CardHeader("🔗 Synchronization Features", className="bg-info text-white"),
            dbc.CardBody([
                html.Div([
                    html.Strong(f"{key.replace('_', ' ').title()}: "),
                    html.Span(f"{value:.4f}" if isinstance(value, (int, float)) or (hasattr(value, 'item') and isinstance(value.item(), (int, float))) else str(value))
                ]) for key, value in sync_features.items() if value is not None
            ])
        ], className="mb-3")
        cards.append(sync_card)
    
    if ecg_features:
        ecg_card = dbc.Card([
            dbc.CardHeader("❤️ ECG Features", className="bg-danger text-white"),
            dbc.CardBody([
                html.Div([
                    html.Strong(f"{key.replace('_', ' ').title()}: "),
                    html.Span(f"{value:.4f}" if isinstance(value, (int, float)) or (hasattr(value, 'item') and isinstance(value.item(), (int, float))) else str(value))
                ]) for key, value in ecg_features.items() if value is not None
            ])
        ], className="mb-3")
        cards.append(ecg_card)
    
    if other_features:
        other_card = dbc.Card([
            dbc.CardHeader("📈 Other Features", className="bg-secondary text-white"),
            dbc.CardBody([
                html.Div([
                    html.Strong(f"{key.replace('_', ' ').title()}: "),
                    html.Span(f"{value:.4f}" if isinstance(value, (int, float)) or (hasattr(value, 'item') and isinstance(value.item(), (int, float))) else str(value))
                ]) for key, value in other_features.items() if value is not None
            ])
        ], className="mb-3")
        cards.append(other_card)
    
    return html.Div([
        html.H4("🎯 Feature Analysis Results", className="mb-3"),
        html.Div(cards)
    ])
