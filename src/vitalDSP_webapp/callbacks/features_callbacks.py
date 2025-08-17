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
    """Import vitalDSP modules with error handling and comprehensive module checking."""
    global PPGAutonomicFeatures, PhysiologicalFeatureExtractor, ECGExtractor
    global PPGLightFeatureExtractor, ECGPPGSynchronization
    
    logger.info("=== IMPORTING VITALDSP FEATURE ENGINEERING MODULES ===")
    
    # Check if vitalDSP package is available
    try:
        import vitalDSP
        logger.info(f"✓ vitalDSP package found: {vitalDSP.__file__}")
        logger.info(f"✓ vitalDSP version: {getattr(vitalDSP, '__version__', 'unknown')}")
    except ImportError as e:
        logger.error(f"✗ vitalDSP package not found: {e}")
        return
    
    # Check feature_engineering module
    try:
        import vitalDSP.feature_engineering
        logger.info(f"✓ vitalDSP.feature_engineering module found: {vitalDSP.feature_engineering.__file__}")
        
        # List all available submodules
        available_modules = [attr for attr in dir(vitalDSP.feature_engineering) 
                           if not attr.startswith('_') and not attr.startswith('__')]
        logger.info(f"✓ Available feature_engineering modules: {available_modules}")
        
    except ImportError as e:
        logger.error(f"✗ vitalDSP.feature_engineering module not found: {e}")
        return
    
    # Import PPG Autonomic Features
    try:
        from vitalDSP.feature_engineering.ppg_autonomic_features import PPGAutonomicFeatures
        logger.info("✓ PPGAutonomicFeatures imported successfully")
        
        # Test the class
        if hasattr(PPGAutonomicFeatures, '__init__'):
            logger.info("✓ PPGAutonomicFeatures class is properly defined")
        else:
            logger.warning("⚠ PPGAutonomicFeatures class may not be properly defined")
            
    except Exception as e:
        logger.error(f"✗ Failed to import PPGAutonomicFeatures: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        PPGAutonomicFeatures = None
    
    # Import Physiological Feature Extractor
    try:
        from vitalDSP.feature_engineering.morphology_features import PhysiologicalFeatureExtractor
        logger.info("✓ PhysiologicalFeatureExtractor imported successfully")
        
        # Test the class
        if hasattr(PhysiologicalFeatureExtractor, '__init__'):
            logger.info("✓ PhysiologicalFeatureExtractor class is properly defined")
        else:
            logger.warning("⚠ PhysiologicalFeatureExtractor class may not be properly defined")
            
    except Exception as e:
        logger.error(f"✗ Failed to import PhysiologicalFeatureExtractor: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        PhysiologicalFeatureExtractor = None
    
    # Import ECG Extractor
    try:
        from vitalDSP.feature_engineering.ecg_autonomic_features import ECGExtractor
        logger.info("✓ ECGExtractor imported successfully")
        
        # Test the class
        if hasattr(ECGExtractor, '__init__'):
            logger.info("✓ ECGExtractor class is properly defined")
        else:
            logger.warning("⚠ ECGExtractor class may not be properly defined")
            
    except Exception as e:
        logger.error(f"✗ Failed to import ECGExtractor: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        ECGExtractor = None
    
    # Import PPG Light Feature Extractor
    try:
        from vitalDSP.feature_engineering.ppg_light_features import PPGLightFeatureExtractor
        logger.info("✓ PPGLightFeatureExtractor imported successfully")
        
        # Test the class
        if hasattr(PPGLightFeatureExtractor, '__init__'):
            logger.info("✓ PPGLightFeatureExtractor class is properly defined")
        else:
            logger.warning("⚠ PPGLightFeatureExtractor class may not be properly defined")
            
    except Exception as e:
        logger.error(f"✗ Failed to import PPGLightFeatureExtractor: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        PPGLightFeatureExtractor = None
    
    # Import ECG-PPG Synchronization
    try:
        from vitalDSP.feature_engineering.ecg_ppg_synchronyzation_features import ECGPPGSynchronization
        logger.info("✓ ECGPPGSynchronization imported successfully")
        
        # Test the class
        if hasattr(ECGPPGSynchronization, '__init__'):
            logger.info("✓ ECGPPGSynchronization class is properly defined")
        else:
            logger.warning("⚠ ECGPPGSynchronization class may not be properly defined")
            
    except Exception as e:
        logger.error(f"✗ Failed to import ECGPPGSynchronization: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        ECGPPGSynchronization = None
    
    # Summary of available modules
    available_modules = []
    if PPGAutonomicFeatures: available_modules.append("PPGAutonomicFeatures")
    if PhysiologicalFeatureExtractor: available_modules.append("PhysiologicalFeatureExtractor")
    if ECGExtractor: available_modules.append("ECGExtractor")
    if PPGLightFeatureExtractor: available_modules.append("PPGLightFeatureExtractor")
    if ECGPPGSynchronization: available_modules.append("ECGPPGSynchronization")
    
    logger.info(f"✓ Successfully imported {len(available_modules)} modules: {available_modules}")
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
    elif hasattr(obj, 'dtype'):  # Handle any object with dtype attribute
        try:
            if hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            else:
                return str(obj)
        except:
            return str(obj)
    elif str(type(obj)).startswith("<class 'numpy."):  # Catch any numpy types we missed
        try:
            if hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            else:
                return str(obj)
        except:
            return str(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    else:
        return obj

def clean_data_dict(data):
    """Clean data dictionary to ensure all values are JSON serializable."""
    if not isinstance(data, dict):
        return str(data)
    
    cleaned_data = {}
    
    for key, value in data.items():
        try:
            if value is None:
                cleaned_data[key] = None
            elif isinstance(value, (str, int, float, bool)):
                cleaned_data[key] = value
            elif hasattr(value, 'item'):  # numpy scalar
                cleaned_data[key] = value.item()
            elif hasattr(value, 'tolist'):  # numpy array
                cleaned_data[key] = value.tolist()
            elif isinstance(value, np.ndarray):
                cleaned_data[key] = value.tolist()
            elif isinstance(value, np.integer):
                cleaned_data[key] = int(value)
            elif isinstance(value, np.floating):
                cleaned_data[key] = float(value)
            elif isinstance(value, np.bool_):
                cleaned_data[key] = bool(value)
            elif isinstance(value, dict):
                cleaned_data[key] = clean_data_dict(value)
            elif isinstance(value, list):
                cleaned_data[key] = [clean_data_dict(item) if isinstance(item, dict) else 
                                    (item.item() if hasattr(item, 'item') else 
                                     item.tolist() if hasattr(item, 'tolist') else item) 
                                    for item in value]
            else:
                # Try to convert to string as last resort
                cleaned_data[key] = str(value)
                logger.warning(f"Data key {key} converted to string: {value}")
        except Exception as e:
            logger.error(f"Failed to clean data key {key}: {e}")
            cleaned_data[key] = str(value)
    
    return cleaned_data

def clean_features_dict(features):
    """Clean features dictionary to ensure all values are JSON serializable."""
    cleaned_features = {}
    
    for key, value in features.items():
        try:
            if value is None:
                cleaned_features[key] = None
            elif isinstance(value, (str, int, float, bool)):
                cleaned_features[key] = value
            elif hasattr(value, 'item'):  # numpy scalar
                cleaned_features[key] = value.item()
            elif hasattr(value, 'tolist'):  # numpy array
                cleaned_features[key] = value.tolist()
            elif isinstance(value, np.ndarray):
                cleaned_features[key] = value.tolist()
            elif isinstance(value, np.integer):
                cleaned_features[key] = int(value)
            elif isinstance(value, np.floating):
                cleaned_features[key] = float(value)
            elif isinstance(value, np.bool_):
                cleaned_features[key] = bool(value)
            elif isinstance(value, dict):
                cleaned_features[key] = clean_features_dict(value)
            elif isinstance(value, list):
                cleaned_features[key] = [clean_features_dict(item) if isinstance(item, dict) else 
                                       (item.item() if hasattr(item, 'item') else 
                                        item.tolist() if hasattr(item, 'tolist') else item) 
                                       for item in value]
            else:
                # Try to convert to string as last resort
                cleaned_features[key] = str(value)
                logger.warning(f"Feature {key} converted to string: {value}")
        except Exception as e:
            logger.error(f"Failed to clean feature {key}: {e}")
            cleaned_features[key] = str(value)
    
    return cleaned_features

def test_json_serialization(obj, name="object"):
    """Test if an object can be JSON serialized and return a serializable version."""
    import json
    try:
        json.dumps(obj)
        logger.info(f"✓ {name} is JSON serializable")
        return obj
    except (TypeError, ValueError) as e:
        logger.warning(f"✗ {name} is not JSON serializable: {e}")
        # Try to convert to a more basic format
        try:
            if hasattr(obj, 'to_dict'):
                converted = obj.to_dict()
                json.dumps(converted)
                logger.info(f"✓ {name} converted to dict and is now serializable")
                return converted
            elif hasattr(obj, 'tolist'):
                converted = obj.tolist()
                json.dumps(converted)
                logger.info(f"✓ {name} converted to list and is now serializable")
                return converted
            else:
                logger.warning(f"✗ Could not convert {name} to serializable format")
                return str(obj)
        except Exception as conv_e:
            logger.error(f"✗ Failed to convert {name}: {conv_e}")
            return str(obj)

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
        elif hasattr(d, 'dtype'):  # Handle any object with dtype attribute
            try:
                if hasattr(d, 'item'):
                    return d.item()
                elif hasattr(d, 'tolist'):
                    return d.tolist()
                else:
                    return str(d)
            except:
                return str(d)
        elif str(type(d)).startswith("<class 'numpy."):  # Catch any numpy types we missed
            try:
                if hasattr(d, 'item'):
                    return d.item()
                elif hasattr(d, 'tolist'):
                    return d.tolist()
                else:
                    return str(d)
            except:
                return str(d)
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

def generate_comprehensive_feature_analysis_enhanced(signal_data, sampling_freq, signal_type="auto", preprocessing_options=None):
    """Generate comprehensive feature analysis with enhanced error handling, progress tracking, and fallback mechanisms."""
    logger.info("=== GENERATING COMPREHENSIVE FEATURE ANALYSIS (ENHANCED) ===")
    logger.info(f"Signal type: {signal_type}")
    logger.info(f"Signal shape: {signal_data.shape}")
    logger.info(f"Sampling frequency: {sampling_freq}")
    
    # Initialize features dictionary and tracking
    features = {}
    extraction_stats = {
        'successful': 0,
        'failed': 0,
        'warnings': 0,
        'start_time': pd.Timestamp.now()
    }
    
    # Auto-detect signal type if needed
    if signal_type == "auto":
        try:
            signal_type = detect_signal_type(signal_data, sampling_freq)
            logger.info(f"✓ Auto-detected signal type: {signal_type}")
        except Exception as e:
            logger.warning(f"⚠ Signal type auto-detection failed, using 'auto': {e}")
            signal_type = "auto"
    
    # Apply preprocessing if specified
    if preprocessing_options:
        logger.info(f"🔄 Applying preprocessing: {preprocessing_options}")
        try:
            original_signal = signal_data.copy()
            signal_data = apply_preprocessing(signal_data, sampling_freq, preprocessing_options)
            logger.info("✓ Preprocessing applied successfully")
            logger.info(f"  Signal range before: {np.min(original_signal):.3f} to {np.max(original_signal):.3f}")
            logger.info(f"  Signal range after: {np.min(signal_data):.3f} to {np.max(signal_data):.3f}")
        except Exception as e:
            logger.warning(f"⚠ Preprocessing failed, using original signal: {e}")
            extraction_stats['warnings'] += 1
    
    # Extract PPG autonomic features
    if signal_type in ["ppg", "auto"]:
        logger.info("=== EXTRACTING PPG AUTONOMIC FEATURES ===")
        try:
            if PPGAutonomicFeatures:
                ppg_extractor = PPGAutonomicFeatures()
                logger.info("✓ PPGAutonomicFeatures instance created successfully")
                
                # Extract RRV (Respiratory Rate Variability)
                try:
                    rrv = ppg_extractor.extract_rrv(signal_data, sampling_freq)
                    features['rrv'] = rrv
                    logger.info(f"✓ RRV computed: {rrv}")
                    extraction_stats['successful'] += 1
                except Exception as e:
                    logger.warning(f"⚠ RRV extraction failed: {e}")
                    features['rrv'] = None
                    extraction_stats['failed'] += 1
                
                # Extract RSA (Respiratory Sinus Arrhythmia)
                try:
                    rsa = ppg_extractor.extract_rsa(signal_data, sampling_freq)
                    features['rsa'] = rsa
                    logger.info(f"✓ RSA computed: {rsa}")
                    extraction_stats['successful'] += 1
                except Exception as e:
                    logger.warning(f"⚠ RSA extraction failed: {e}")
                    features['rsa'] = None
                    extraction_stats['failed'] += 1
                
                # Extract fractal dimension
                try:
                    fractal_dim = ppg_extractor.extract_fractal_dimension(signal_data)
                    features['fractal_dimension'] = fractal_dim
                    logger.info(f"✓ Fractal dimension computed: {fractal_dim}")
                    extraction_stats['successful'] += 1
                except Exception as e:
                    logger.warning(f"⚠ Fractal dimension extraction failed: {e}")
                    features['fractal_dimension'] = None
                    extraction_stats['failed'] += 1
                
                # Extract DFA (Detrended Fluctuation Analysis)
                try:
                    dfa = ppg_extractor.extract_dfa(signal_data)
                    features['dfa'] = dfa
                    logger.info(f"✓ DFA computed: {dfa}")
                    extraction_stats['successful'] += 1
                except Exception as e:
                    logger.warning(f"⚠ DFA extraction failed: {e}")
                    features['dfa'] = None
                    extraction_stats['failed'] += 1
                
            else:
                logger.warning("⚠ PPGAutonomicFeatures not available")
                features.update({
                    'rrv': None, 'rsa': None, 'fractal_dimension': None, 'dfa': None
                })
                extraction_stats['warnings'] += 4
                
        except Exception as e:
            logger.error(f"✗ PPG autonomic feature extraction failed: {e}")
            features.update({
                'rrv': None, 'rsa': None, 'fractal_dimension': None, 'dfa': None
            })
            extraction_stats['failed'] += 4
    
    # Extract morphology features
    logger.info("=== EXTRACTING MORPHOLOGY FEATURES ===")
    try:
        if PhysiologicalFeatureExtractor:
            morph_extractor = PhysiologicalFeatureExtractor()
            logger.info("✓ PhysiologicalFeatureExtractor instance created successfully")
            
            # Extract comprehensive morphology features
            try:
                morph_features = morph_extractor.extract_features(signal_data, sampling_freq)
                features.update(morph_features)
                logger.info(f"✓ Morphology features extracted: {len(morph_features)} features")
                extraction_stats['successful'] += len(morph_features)
                
                # Log individual morphology features with validation
                for key, value in morph_features.items():
                    if value is None or (hasattr(value, '__len__') and len(value) == 0):
                        logger.warning(f"⚠ Morphology {key} is None or empty")
                        extraction_stats['warnings'] += 1
                    elif hasattr(value, '__len__') and np.any(np.isnan(value)):
                        logger.warning(f"⚠ Morphology {key} contains NaN values")
                        extraction_stats['warnings'] += 1
                    else:
                        logger.info(f"✓ Morphology {key}: {value}")
                        
            except Exception as e:
                logger.error(f"✗ Morphology feature extraction failed: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                # Add default morphology features
                default_morph_features = {
                    'systolic_duration': None, 'diastolic_duration': None,
                    'systolic_area': None, 'diastolic_area': None,
                    'systolic_slope': None, 'diastolic_slope': None,
                    'signal_skewness': None, 'peak_trend_slope': None,
                    'heart_rate': None, 'systolic_amplitude_variability': None,
                    'diastolic_amplitude_variability': None
                }
                features.update(default_morph_features)
                extraction_stats['failed'] += len(default_morph_features)
        else:
            logger.warning("⚠ PhysiologicalFeatureExtractor not available")
            # Add default morphology features
            default_morph_features = {
                'systolic_duration': None, 'diastolic_duration': None,
                'systolic_area': None, 'diastolic_area': None,
                'systolic_slope': None, 'diastolic_slope': None,
                'signal_skewness': None, 'peak_trend_slope': None,
                'heart_rate': None, 'systolic_amplitude_variability': None,
                'diastolic_amplitude_variability': None
            }
            features.update(default_morph_features)
            extraction_stats['warnings'] += len(default_morph_features)
            
    except Exception as e:
        logger.error(f"✗ Morphology feature extraction failed: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        # Add default morphology features
        default_morph_features = {
            'systolic_duration': None, 'diastolic_duration': None,
            'systolic_area': None, 'diastolic_area': None,
            'systolic_slope': None, 'diastolic_slope': None,
            'signal_skewness': None, 'peak_trend_slope': None,
            'heart_rate': None, 'systolic_amplitude_variability': None,
            'diastolic_amplitude_variability': None
        }
        features.update(default_morph_features)
        extraction_stats['failed'] += len(default_morph_features)
    
    # Extract ECG autonomic features (if ECG data available)
    if signal_type in ["ecg", "auto"]:
        logger.info("=== EXTRACTING ECG AUTONOMIC FEATURES ===")
        try:
            if ECGExtractor:
                ecg_extractor = ECGExtractor()
                logger.info("✓ ECGExtractor instance created successfully")
                
                try:
                    ecg_features = ecg_extractor.extract_features(signal_data, sampling_freq)
                    features['ecg_autonomic'] = ecg_features
                    logger.info(f"✓ ECG autonomic features extracted: {len(ecg_features)} features")
                    extraction_stats['successful'] += len(ecg_features)
                except Exception as e:
                    logger.error(f"✗ ECG feature extraction failed: {e}")
                    features['ecg_autonomic'] = None
                    extraction_stats['failed'] += 1
            else:
                logger.warning("⚠ ECGExtractor not available or ECG data not provided")
                features['ecg_autonomic'] = None
                extraction_stats['warnings'] += 1
        except Exception as e:
            logger.error(f"✗ ECG autonomic feature extraction failed: {e}")
            features['ecg_autonomic'] = None
            extraction_stats['failed'] += 1
    
    # Extract PPG light-based features
    logger.info("=== EXTRACTING PPG LIGHT FEATURES ===")
    try:
        if PPGLightFeatureExtractor:
            light_extractor = PPGLightFeatureExtractor()
            logger.info("✓ PPGLightFeatureExtractor instance created successfully")
            
            # Extract SpO2
            try:
                spo2 = light_extractor.estimate_spo2(signal_data, sampling_freq)
                features['spo2'] = spo2
                logger.info(f"✓ SpO2 estimated: {spo2}")
                extraction_stats['successful'] += 1
            except Exception as e:
                logger.warning(f"⚠ SpO2 estimation failed: {e}")
                features['spo2'] = None
                extraction_stats['failed'] += 1
            
            # Extract perfusion index
            try:
                pi = light_extractor.compute_perfusion_index(signal_data, sampling_freq)
                features['perfusion_index'] = pi
                logger.info(f"✓ Perfusion index computed: {pi}")
                extraction_stats['successful'] += 1
            except Exception as e:
                logger.warning(f"⚠ Perfusion index computation failed: {e}")
                features['perfusion_index'] = None
                extraction_stats['failed'] += 1
            
            # Extract respiratory rate from PPG
            try:
                rr_ppg = light_extractor.estimate_respiratory_rate(signal_data, sampling_freq)
                features['respiratory_rate_ppg'] = rr_ppg
                logger.info(f"✓ Respiratory rate from PPG estimated: {rr_ppg}")
                extraction_stats['successful'] += 1
            except Exception as e:
                logger.warning(f"⚠ Respiratory rate estimation failed: {e}")
                features['respiratory_rate_ppg'] = None
                extraction_stats['failed'] += 1
            
            # Extract PPR (Pulse Pressure Ratio)
            try:
                ppr = light_extractor.compute_ppr(signal_data, sampling_freq)
                features['ppr'] = ppr
                logger.info(f"✓ PPR computed: {ppr}")
                extraction_stats['successful'] += 1
            except Exception as e:
                logger.warning(f"⚠ PPR computation failed: {e}")
                features['ppr'] = None
                extraction_stats['failed'] += 1
                
        else:
            logger.warning("⚠ PPGLightFeatureExtractor not available")
            features.update({
                'spo2': None, 'perfusion_index': None, 'respiratory_rate_ppg': None, 'ppr': None
            })
            extraction_stats['warnings'] += 4
            
    except Exception as e:
        logger.error(f"✗ PPG light feature extraction failed: {e}")
        features.update({
            'spo2': None, 'perfusion_index': None, 'respiratory_rate_ppg': None, 'ppr': None
        })
        extraction_stats['failed'] += 4
    
    # Extract ECG-PPG synchronization features (if both signals available)
    logger.info("=== EXTRACTING ECG-PPG SYNCHRONIZATION FEATURES ===")
    try:
        if ECGPPGSynchronization:
            sync_extractor = ECGPPGSynchronization()
            logger.info("✓ ECGPPGSynchronization instance created successfully")
            
            try:
                sync_features = sync_extractor.extract_synchronization_features(signal_data, sampling_freq)
                features['ecg_ppg_sync'] = sync_features
                logger.info(f"✓ ECG-PPG synchronization features extracted: {len(sync_features)} features")
                extraction_stats['successful'] += len(sync_features)
            except Exception as e:
                logger.error(f"✗ ECG-PPG synchronization extraction failed: {e}")
                features['ecg_ppg_sync'] = None
                extraction_stats['failed'] += 1
        else:
            logger.warning("⚠ ECGPPGSynchronization not available or both signals not provided")
            features['ecg_ppg_sync'] = None
            extraction_stats['warnings'] += 1
    except Exception as e:
        logger.error(f"✗ ECG-PPG synchronization feature extraction failed: {e}")
        features['ecg_ppg_sync'] = None
        extraction_stats['failed'] += 1
    
    # Add fallback basic features if no vitalDSP features were extracted
    if extraction_stats['successful'] == 0:
        logger.warning("⚠ No vitalDSP features extracted, adding fallback basic features")
        try:
            basic_features = generate_basic_features(signal_data, sampling_freq)
            features.update(basic_features)
            extraction_stats['successful'] += len(basic_features)
            logger.info(f"✓ Added {len(basic_features)} fallback basic features")
        except Exception as e:
            logger.error(f"✗ Fallback feature generation failed: {e}")
    
    # Final summary and metadata
    extraction_stats['end_time'] = pd.Timestamp.now()
    extraction_stats['duration'] = (extraction_stats['end_time'] - extraction_stats['start_time']).total_seconds()
    
    logger.info("=== FEATURE EXTRACTION COMPLETED (ENHANCED) ===")
    logger.info(f"Total features extracted: {len(features)}")
    logger.info(f"Extraction statistics: {extraction_stats['successful']} successful, {extraction_stats['failed']} failed, {extraction_stats['warnings']} warnings")
    logger.info(f"Extraction duration: {extraction_stats['duration']:.2f} seconds")
    
    # Add extraction metadata
    features['_extraction_metadata'] = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'signal_type': signal_type,
        'sampling_freq': sampling_freq,
        'signal_length': len(signal_data),
        'extraction_stats': extraction_stats,
        'preprocessing_applied': preprocessing_options is not None,
        'vitaldsp_modules_available': {
            'PPGAutonomicFeatures': PPGAutonomicFeatures is not None,
            'PhysiologicalFeatureExtractor': PhysiologicalFeatureExtractor is not None,
            'ECGExtractor': ECGExtractor is not None,
            'PPGLightFeatureExtractor': PPGLightFeatureExtractor is not None,
            'ECGPPGSynchronization': ECGPPGSynchronization is not None
        }
    }
    
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
    
    # Ensure all data is Python native types
    if not isinstance(time_axis, list):
        time_axis = [float(t) for t in time_axis]
    if not isinstance(signal_data_list, list):
        signal_data_list = [float(s) for s in signal_data_list]
    
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
    
    # Final safety check - ensure the figure doesn't contain any numpy types
    try:
        fig_dict = fig.to_dict()
        # Test if the figure can be JSON serialized
        import json
        json.dumps(fig_dict)
        logger.info("✓ Figure is JSON serializable")
    except Exception as e:
        logger.warning(f"✗ Figure contains non-serializable data: {e}")
        # Create a simplified fallback figure
        fig = create_empty_figure()
        fig.update_layout(title="Feature Analysis Results (Simplified View)")
        logger.info("Using fallback figure due to serialization issues")
    
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
            
            # Ensure signal data is in standard numpy types (not pandas nullable types)
            if signal_data.dtype.kind in ['O', 'U']:  # Object or unicode string type
                logger.warning(f"Signal data has dtype {signal_data.dtype}, converting to float64")
                signal_data = signal_data.astype(np.float64)
            elif signal_data.dtype.name.startswith('Int'):  # Pandas nullable integer types
                logger.warning(f"Signal data has pandas nullable dtype {signal_data.dtype}, converting to float64")
                signal_data = signal_data.astype(np.float64)
            elif signal_data.dtype.name.startswith('Float'):  # Pandas nullable float types
                logger.warning(f"Signal data has pandas nullable dtype {signal_data.dtype}, converting to float64")
                signal_data = signal_data.astype(np.float64)
            
            # Handle any NaN or inf values
            if np.any(np.isnan(signal_data)) or np.any(np.isinf(signal_data)):
                logger.warning("Signal data contains NaN or inf values, replacing with 0")
                signal_data = np.nan_to_num(signal_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Validate signal data
            if len(signal_data) == 0:
                logger.error("Signal data is empty")
                return "Signal data is empty. Please check your data.", create_empty_figure(), None, None
            
            if len(signal_data) < 100:  # Minimum length for meaningful analysis
                logger.warning(f"Signal data is very short ({len(signal_data)} samples), analysis may be limited")
            
            if sampling_freq <= 0:
                logger.error(f"Invalid sampling frequency: {sampling_freq}")
                return f"Invalid sampling frequency: {sampling_freq}. Please check your data.", create_empty_figure(), None, None
            
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
            
            # Check if features were generated successfully
            if not features:
                logger.warning("No features were generated")
                features = {"error": "No features could be extracted from the signal"}
            elif isinstance(features, dict) and "error" in features:
                logger.warning(f"Feature extraction returned error: {features['error']}")
            
            # Clean features to ensure they're JSON serializable
            features = clean_features_dict(features)
            
            # Log raw features for debugging
            logger.info("=== RAW FEATURE ENGINEERING RESULTS ===")
            for key, value in features.items():
                logger.info(f"Feature {key}: {value} (type: {type(value)})")
                if hasattr(value, 'dtype'):
                    logger.info(f"  - dtype: {value.dtype}")
                if hasattr(value, 'shape'):
                    logger.info(f"  - shape: {value.shape}")
            logger.info("=== END RAW FEATURES ===")
            
            # Create plots
            plots = create_comprehensive_feature_plots(signal_data, features, sampling_freq)
            
            # Ensure plots is defined
            if plots is None:
                logger.warning("No plots were created, using empty figure")
                plots = create_empty_figure()
                plots.update_layout(title="Feature Analysis Results (No Plots Available)")
            
            # Clean the Plotly figure to ensure JSON serialization
            try:
                plots = clean_plotly_figure(plots)
                logger.info("Plotly figure cleaned successfully")
                
                # Test JSON serialization of the cleaned plots
                plots_serializable = test_json_serialization(plots, "cleaned plots")
                if plots_serializable != plots:
                    plots = plots_serializable
                    logger.info("Plots converted to serializable format")
                
            except Exception as e:
                logger.error(f"Failed to clean Plotly figure: {e}")
                # Create a simple fallback figure
                plots = create_empty_figure()
                plots.update_layout(title="Feature Analysis Results (Simplified View)")
                logger.info("Using fallback figure due to serialization error")
            
            # Create results display
            results_html = create_feature_results_display(features)
            
            # Ensure results_html is defined
            if results_html is None:
                logger.warning("No results HTML was created, using default message")
                results_html = html.Div("Feature analysis completed but no results display was generated.")
            
            # Convert features to JSON-serializable format
            features_serializable = convert_numpy_to_python(features)
            
            # Ensure features_serializable is defined
            if features_serializable is None:
                features_serializable = {"error": "Failed to convert features"}
            
            # Test JSON serialization of the cleaned features
            features_serializable = test_json_serialization(features_serializable, "cleaned features")
            
            # Log cleaned features for debugging
            logger.info("=== CLEANED FEATURES FOR JSON SERIALIZATION ===")
            for key, value in features_serializable.items():
                logger.info(f"Cleaned feature {key}: {value} (type: {type(value)})")
            logger.info("=== END CLEANED FEATURES ===")
            
            logger.info("=== FEATURES ANALYSIS COMPLETED SUCCESSFULLY ===")
            
            # Ensure all required variables are defined
            if 'results_html' not in locals() or results_html is None:
                results_html = html.Div("Feature analysis completed but no results display was generated.")
            if 'plots' not in locals() or plots is None:
                plots = create_empty_figure()
                plots.update_layout(title="Feature Analysis Results (No Plots Available)")
            if 'latest_data' not in locals() or latest_data is None:
                latest_data = {"error": "No data available"}
            if 'features_serializable' not in locals() or features_serializable is None:
                features_serializable = {"error": "No features available"}
            
            # Final safety check - ensure all return values are JSON serializable
            try:
                # Clean latest_data to ensure it's serializable
                latest_data_clean = clean_data_dict(latest_data)
                
                # Ensure latest_data_clean is defined
                if latest_data_clean is None:
                    latest_data_clean = {"error": "Failed to clean data"}
                
                # Test serialization of all return values
                test_json_serialization(results_html, "results_html")
                test_json_serialization(plots, "plots")
                test_json_serialization(latest_data_clean, "latest_data_clean")
                test_json_serialization(features_serializable, "features_serializable")
                logger.info("✓ All return values are JSON serializable")
            except Exception as e:
                logger.error(f"✗ JSON serialization test failed: {e}")
                # If any value fails, convert to string representation
                if not isinstance(results_html, str):
                    results_html = str(results_html)
                if not isinstance(plots, (dict, str)):
                    plots = str(plots)
                if not isinstance(latest_data_clean, dict):
                    latest_data_clean = str(latest_data_clean)
                if not isinstance(features_serializable, dict):
                    features_serializable = str(features_serializable)
                logger.info("Converted non-serializable values to strings")
            
            # Final check - ensure all variables are defined
            if 'results_html' not in locals() or results_html is None:
                results_html = "Feature analysis completed but no results display was generated."
            if 'plots' not in locals() or plots is None:
                plots = create_empty_figure()
                plots.update_layout(title="Feature Analysis Results (No Plots Available)")
            if 'latest_data_clean' not in locals() or latest_data_clean is None:
                latest_data_clean = {"error": "No data available"}
            if 'features_serializable' not in locals() or features_serializable is None:
                features_serializable = {"error": "No features available"}
            
            return results_html, plots, latest_data_clean, features_serializable
            
        except Exception as e:
            logger.error(f"Error in features analysis callback: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Try to get more information about the error
            try:
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Error args: {e.args}")
                if hasattr(e, '__dict__'):
                    logger.error(f"Error attributes: {e.__dict__}")
            except Exception as log_e:
                logger.error(f"Failed to log error details: {log_e}")
            
            error_msg = f"Error during feature analysis: {str(e)}"
            return error_msg, create_empty_figure(), None, None
    
    logger.info("=== FEATURE ENGINEERING CALLBACKS REGISTERED SUCCESSFULLY ===")

def features_analysis_callback_enhanced(n_clicks, pathname, signal_type, preprocessing):
    """Enhanced callback for feature engineering analysis with comprehensive error handling and progress tracking."""
    logger.info("=== ENHANCED FEATURES ANALYSIS CALLBACK TRIGGERED ===")
    
    # Initialize tracking variables
    start_time = pd.Timestamp.now()
    callback_stats = {
        'start_time': start_time,
        'steps_completed': [],
        'errors': [],
        'warnings': [],
        'performance_metrics': {}
    }
    
    # Only run this when we're on the features page
    if pathname != "/features":
        logger.info("Not on features page, returning empty figures")
        return "Navigate to Features page", create_empty_figure(), None, None
    
    # Check if data is available first
    try:
        from ..services.data_service import get_data_service
        data_service = get_data_service()
        
        if data_service is None:
            error_msg = "Data service is None"
            logger.error(error_msg)
            callback_stats['errors'].append(error_msg)
            return "Data service not available. Please restart the application.", create_empty_figure(), None, None
        
        # Step 1: Get all data
        logger.info("🔄 Step 1: Retrieving data from service...")
        all_data = data_service.get_all_data()
        if not all_data:
            error_msg = "No data available"
            logger.info(error_msg)
            callback_stats['warnings'].append(error_msg)
            return "No data available. Please upload and process data first.", create_empty_figure(), None, None
        
        callback_stats['steps_completed'].append("data_retrieval")
        logger.info(f"✓ Retrieved {len(all_data)} data entries")
        
        # Data is available, show instructions if no button click
        if n_clicks is None:
            logger.info("No button click - showing instructions")
            return "Data is available! Click '🚀 Analyze Features' to start analysis.", create_empty_figure(), None, None
        
        logger.info(f"Button clicked {n_clicks} times, starting analysis...")
        
        # Step 2: Get the most recent data entry
        logger.info("🔄 Step 2: Processing latest data...")
        latest_data_id = list(all_data.keys())[-1]
        latest_data = all_data[latest_data_id]
        
        logger.info(f"✓ Found data: {latest_data_id}")
        logger.info(f"✓ Latest data keys: {list(latest_data.keys())}")
        logger.info(f"✓ Latest data info keys: {list(latest_data.get('info', {}).keys())}")
        logger.info(f"✓ Latest data metadata keys: {list(latest_data.get('metadata', {}).keys())}")
        callback_stats['steps_completed'].append("data_processing")
        
        # Step 3: Get column mapping
        logger.info("🔄 Step 3: Checking column mapping...")
        column_mapping = data_service.get_column_mapping(latest_data_id)
        if not column_mapping:
            error_msg = "Data has not been processed yet - no column mapping found"
            logger.warning(error_msg)
            callback_stats['warnings'].append(error_msg)
            return "Please process your data on the Upload page first (configure column mapping)", create_empty_figure(), None, None
        
        logger.info(f"✓ Column mapping found: {column_mapping}")
        callback_stats['steps_completed'].append("column_mapping_validation")
        
        # Step 4: Get data frame
        logger.info("🔄 Step 4: Retrieving data frame...")
        df = data_service.get_data(latest_data_id)
        if df is None or df.empty:
            error_msg = "Data frame is empty"
            logger.warning(error_msg)
            callback_stats['errors'].append(error_msg)
            return "Data is empty or corrupted.", create_empty_figure(), None, None
        
        logger.info(f"✓ Data frame retrieved, shape: {df.shape}")
        callback_stats['steps_completed'].append("dataframe_retrieval")
        
        # Step 5: Get sampling frequency
        logger.info("🔄 Step 5: Processing sampling frequency...")
        sampling_freq = latest_data.get('info', {}).get('sampling_freq', 100)
        logger.info(f"✓ Initial sampling frequency: {sampling_freq}")
        
        # If no sampling frequency in info, try to get it from metadata
        if sampling_freq == 100:  # Default value, might not be real
            metadata = latest_data.get('metadata', {})
            if 'sampling_freq' in metadata:
                sampling_freq = metadata['sampling_freq']
            elif 'fs' in metadata:
                sampling_freq = metadata['fs']
            elif 'frequency' in metadata:
                sampling_freq = metadata['frequency']
            logger.info(f"✓ Updated sampling frequency from metadata: {sampling_freq}")
        
        callback_stats['steps_completed'].append("sampling_freq_processing")
        
        # Step 6: Determine signal column
        logger.info("🔄 Step 6: Determining signal column...")
        signal_column = None
        for col_type, col_name in column_mapping.items():
            if col_type in ['ppg', 'ecg', 'signal']:
                signal_column = col_name
                break
        
        if signal_column is None:
            # Fallback to first column
            signal_column = df.columns[0]
            logger.warning(f"⚠ No specific signal column found, using fallback: {signal_column}")
            callback_stats['warnings'].append(f"Using fallback signal column: {signal_column}")
        
        logger.info(f"✓ Using signal column: {signal_column}")
        callback_stats['steps_completed'].append("signal_column_determination")
        
        # Step 7: Extract and validate signal data
        logger.info("🔄 Step 7: Extracting and validating signal data...")
        signal_data = np.array(df[signal_column])
        
        # Ensure signal data is in standard numpy types (not pandas nullable types)
        if signal_data.dtype.kind in ['O', 'U']:  # Object or unicode string type
            logger.warning(f"⚠ Signal data has dtype {signal_data.dtype}, converting to float64")
            signal_data = signal_data.astype(np.float64)
            callback_stats['warnings'].append(f"Converted {signal_data.dtype} to float64")
        elif signal_data.dtype.name.startswith('Int'):  # Pandas nullable integer types
            logger.warning(f"⚠ Signal data has pandas nullable dtype {signal_data.dtype}, converting to float64")
            signal_data = signal_data.astype(np.float64)
            callback_stats['warnings'].append(f"Converted {signal_data.dtype} to float64")
        elif signal_data.dtype.name.startswith('Float'):  # Pandas nullable float types
            logger.warning(f"⚠ Signal data has pandas nullable dtype {signal_data.dtype}, converting to float64")
            signal_data = signal_data.astype(np.float64)
            callback_stats['warnings'].append(f"Converted {signal_data.dtype} to float64")
        
        # Handle any NaN or inf values
        if np.any(np.isnan(signal_data)) or np.any(np.isinf(signal_data)):
            logger.warning("⚠ Signal data contains NaN or inf values, replacing with 0")
            signal_data = np.nan_to_num(signal_data, nan=0.0, posinf=0.0, neginf=0.0)
            callback_stats['warnings'].append("Replaced NaN/inf values with 0")
        
        logger.info(f"✓ Signal data extracted, shape: {signal_data.shape}")
        logger.info(f"✓ Signal data type: {type(signal_data)}")
        logger.info(f"✓ Signal data range: {signal_data.min():.3f} to {signal_data.max():.3f}")
        logger.info(f"✓ Signal data mean: {signal_data.mean():.3f}")
        callback_stats['steps_completed'].append("signal_extraction")
        
        # Step 8: Validate signal data quality
        logger.info("🔄 Step 8: Validating signal data quality...")
        if len(signal_data) == 0:
            error_msg = "Signal data is empty"
            logger.error(error_msg)
            callback_stats['errors'].append(error_msg)
            return error_msg, create_empty_figure(), None, None
        
        if len(signal_data) < 100:  # Minimum length for meaningful analysis
            warning_msg = f"Signal data is very short ({len(signal_data)} samples), analysis may be limited"
            logger.warning(warning_msg)
            callback_stats['warnings'].append(warning_msg)
        
        if sampling_freq <= 0:
            error_msg = f"Invalid sampling frequency: {sampling_freq}"
            logger.error(error_msg)
            callback_stats['errors'].append(error_msg)
            return f"Invalid sampling frequency: {sampling_freq}. Please check your data.", create_empty_figure(), None, None
        
        callback_stats['steps_completed'].append("data_validation")
        
        # Step 9: Set signal type and preprocessing
        logger.info("🔄 Step 9: Setting signal type and preprocessing...")
        signal_type = signal_type or "ppg"
        preprocessing = preprocessing or []
        
        logger.info(f"✓ Signal type: {signal_type}")
        logger.info(f"✓ Preprocessing options: {preprocessing}")
        callback_stats['steps_completed'].append("parameters_setup")
        
        # Step 10: Apply preprocessing if selected
        if preprocessing:
            logger.info("🔄 Step 10: Applying preprocessing...")
            try:
                # Add preprocessing logic here if needed
                logger.info("✓ Preprocessing applied (placeholder)")
                callback_stats['steps_completed'].append("preprocessing")
            except Exception as e:
                logger.warning(f"⚠ Preprocessing failed: {e}")
                callback_stats['warnings'].append(f"Preprocessing failed: {e}")
        
        # Step 11: Generate comprehensive feature analysis
        logger.info("🔄 Step 11: Generating comprehensive feature analysis...")
        try:
            features = generate_comprehensive_feature_analysis_enhanced(signal_data, sampling_freq, signal_type, preprocessing)
            logger.info(f"✓ Features generated successfully: {len(features)} features")
            callback_stats['steps_completed'].append("feature_generation")
        except Exception as e:
            logger.error(f"✗ Feature generation failed: {e}")
            callback_stats['errors'].append(f"Feature generation failed: {e}")
            # Try fallback feature generation
            try:
                logger.info("🔄 Attempting fallback feature generation...")
                features = generate_basic_features(signal_data, sampling_freq)
                logger.info(f"✓ Fallback features generated: {len(features)} features")
                callback_stats['steps_completed'].append("fallback_feature_generation")
            except Exception as fallback_e:
                logger.error(f"✗ Fallback feature generation also failed: {fallback_e}")
                callback_stats['errors'].append(f"Fallback feature generation failed: {fallback_e}")
                return f"Feature generation failed: {str(e)}", create_empty_figure(), {}, {}
        
        # Check if features were generated successfully
        if not features:
            error_msg = "No features were generated"
            logger.error(error_msg)
            callback_stats['errors'].append(error_msg)
            return error_msg, create_empty_figure(), None, None
        elif isinstance(features, dict) and "error" in features:
            warning_msg = f"Feature extraction returned error: {features['error']}"
            logger.warning(warning_msg)
            callback_stats['warnings'].append(warning_msg)
        
        # Step 12: Clean features for JSON serialization
        logger.info("🔄 Step 12: Cleaning features for JSON serialization...")
        try:
            features = clean_features_dict(features)
            logger.info("✓ Features cleaned successfully")
            callback_stats['steps_completed'].append("feature_cleaning")
        except Exception as e:
            logger.error(f"✗ Feature cleaning failed: {e}")
            callback_stats['errors'].append(f"Feature cleaning failed: {e}")
        
        # Step 13: Log raw features for debugging
        logger.info("=== RAW FEATURE ENGINEERING RESULTS ===")
        for key, value in features.items():
            logger.info(f"Feature {key}: {value} (type: {type(value)})")
            if hasattr(value, 'dtype'):
                logger.info(f"  - dtype: {value.dtype}")
            if hasattr(value, 'shape'):
                logger.info(f"  - shape: {value.shape}")
        logger.info("=== END RAW FEATURES ===")
        
        # Step 14: Create comprehensive feature plots
        logger.info("🔄 Step 14: Creating comprehensive feature plots...")
        try:
            plots = create_comprehensive_feature_plots(signal_data, features, sampling_freq)
            if plots is None:
                logger.warning("⚠ Failed to create plots, using empty figure")
                plots = create_empty_figure()
                plots.update_layout(title="Feature Analysis Results (No Plots Available)")
            logger.info("✓ Feature plots created successfully")
            callback_stats['steps_completed'].append("plot_creation")
        except Exception as e:
            logger.error(f"✗ Plot creation failed: {e}")
            callback_stats['errors'].append(f"Plot creation failed: {e}")
            plots = create_empty_figure()
            plots.update_layout(title="Feature Analysis Results (Error Creating Plots)")
        
        # Step 15: Clean the Plotly figure for JSON serialization
        logger.info("🔄 Step 15: Cleaning Plotly figure for JSON serialization...")
        try:
            plots = clean_plotly_figure(plots)
            logger.info("✓ Plotly figure cleaned successfully")
            callback_stats['steps_completed'].append("plot_cleaning")
        except Exception as e:
            logger.error(f"✗ Plot cleaning failed: {e}")
            callback_stats['errors'].append(f"Plot cleaning failed: {e}")
        
        # Step 16: Test JSON serialization of plots
        logger.info("🔄 Step 16: Testing JSON serialization of plots...")
        try:
            plots_serializable = test_json_serialization(plots, "cleaned plots")
            if plots_serializable != plots:
                plots = plots_serializable
                logger.info("✓ Plots converted to serializable format")
            callback_stats['steps_completed'].append("plot_serialization_test")
        except Exception as e:
            logger.error(f"✗ Plot serialization test failed: {e}")
            callback_stats['errors'].append(f"Plot serialization test failed: {e}")
        
        # Step 17: Create results display
        logger.info("🔄 Step 17: Creating feature results display...")
        try:
            results_html = create_feature_results_display(features)
            if results_html is None:
                logger.warning("⚠ No results HTML was created, using default message")
                results_html = html.Div("Feature analysis completed but no results display was generated.")
            logger.info("✓ Feature results display created successfully")
            callback_stats['steps_completed'].append("results_display_creation")
        except Exception as e:
            logger.error(f"✗ Results display creation failed: {e}")
            callback_stats['errors'].append(f"Results display creation failed: {e}")
            results_html = html.Div([
                html.H4("Feature Analysis Results"),
                html.P(f"Analysis completed with errors: {str(e)}")
            ])
        
        # Step 18: Convert features to JSON-serializable format
        logger.info("🔄 Step 18: Converting features to JSON-serializable format...")
        try:
            features_serializable = convert_numpy_to_python(features)
            if features_serializable is None:
                features_serializable = {"error": "Failed to convert features"}
            logger.info("✓ Features converted to JSON-serializable format")
            callback_stats['steps_completed'].append("feature_conversion")
        except Exception as e:
            logger.error(f"✗ Feature conversion failed: {e}")
            callback_stats['errors'].append(f"Feature conversion failed: {e}")
            features_serializable = {"error": f"Feature conversion failed: {str(e)}"}
        
        # Step 19: Test JSON serialization of features
        logger.info("🔄 Step 19: Testing JSON serialization of features...")
        try:
            features_serializable = test_json_serialization(features_serializable, "cleaned features")
            logger.info("✓ Features serialization test passed")
            callback_stats['steps_completed'].append("features_serialization_test")
        except Exception as e:
            logger.error(f"✗ Features serialization test failed: {e}")
            callback_stats['errors'].append(f"Features serialization test failed: {e}")
        
        # Step 20: Log cleaned features for debugging
        logger.info("=== CLEANED FEATURES FOR JSON SERIALIZATION ===")
        for key, value in features_serializable.items():
            logger.info(f"Cleaned feature {key}: {value} (type: {type(value)})")
        logger.info("=== END CLEANED FEATURES ===")
        
        # Step 21: Final safety check - ensure all return values are JSON serializable
        logger.info("🔄 Step 21: Final JSON serialization safety check...")
        try:
            # Clean latest_data to ensure it's serializable
            latest_data_clean = clean_data_dict(latest_data)
            
            # Ensure latest_data_clean is defined
            if latest_data_clean is None:
                latest_data_clean = {"error": "Failed to clean data"}
            
            # Test serialization of all return values
            test_json_serialization(results_html, "results_html")
            test_json_serialization(plots, "plots")
            test_json_serialization(latest_data_clean, "latest_data_clean")
            test_json_serialization(features_serializable, "features_serializable")
            logger.info("✓ All return values are JSON serializable")
            callback_stats['steps_completed'].append("final_serialization_check")
        except Exception as e:
            logger.error(f"✗ JSON serialization test failed: {e}")
            callback_stats['errors'].append(f"JSON serialization test failed: {e}")
            # If any value fails, convert to string representation
            if not isinstance(results_html, str):
                results_html = str(results_html)
            if not isinstance(plots, (dict, str)):
                plots = str(plots)
            if not isinstance(latest_data_clean, dict):
                latest_data_clean = str(latest_data_clean)
            if not isinstance(features_serializable, dict):
                features_serializable = str(features_serializable)
            logger.info("Converted non-serializable values to strings")
        
        # Step 22: Final variable validation
        logger.info("🔄 Step 22: Final variable validation...")
        if 'results_html' not in locals() or results_html is None:
            results_html = "Feature analysis completed but no results display was generated."
        if 'plots' not in locals() or plots is None:
            plots = create_empty_figure()
            plots.update_layout(title="Feature Analysis Results (No Plots Available)")
        if 'latest_data_clean' not in locals() or latest_data_clean is None:
            latest_data_clean = {"error": "No data available"}
        if 'features_serializable' not in locals() or features_serializable is None:
            features_serializable = {"error": "No features available"}
        
        # Final callback statistics
        end_time = pd.Timestamp.now()
        callback_stats['end_time'] = end_time
        callback_stats['duration'] = (end_time - start_time).total_seconds()
        callback_stats['total_steps'] = len(callback_stats['steps_completed'])
        callback_stats['success_rate'] = len(callback_stats['steps_completed']) / 22  # Total expected steps
        
        # Performance metrics
        callback_stats['performance_metrics'] = {
            'signal_length': len(signal_data),
            'sampling_freq': sampling_freq,
            'features_extracted': len(features),
            'memory_usage_mb': signal_data.nbytes / (1024 * 1024)
        }
        
        logger.info("=== ENHANCED FEATURES ANALYSIS COMPLETED SUCCESSFULLY ===")
        logger.info(f"✓ Total steps completed: {callback_stats['total_steps']}/22")
        logger.info(f"✓ Success rate: {callback_stats['success_rate']:.1%}")
        logger.info(f"✓ Total duration: {callback_stats['duration']:.2f} seconds")
        logger.info(f"✓ Errors encountered: {len(callback_stats['errors'])}")
        logger.info(f"✓ Warnings encountered: {len(callback_stats['warnings'])}")
        logger.info(f"✓ Performance metrics: {callback_stats['performance_metrics']}")
        
        # Add callback metadata to features
        if isinstance(features_serializable, dict):
            features_serializable['_callback_metadata'] = callback_stats
        
        return results_html, plots, latest_data_clean, features_serializable
        
    except Exception as e:
        logger.error(f"✗ Critical error in enhanced features analysis callback: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Try to get more information about the error
        try:
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error args: {e.args}")
            if hasattr(e, '__dict__'):
                logger.error(f"Error attributes: {e.__dict__}")
        except Exception as log_e:
            logger.error(f"Failed to log error details: {log_e}")
        
        # Add error to callback stats
        callback_stats['errors'].append(f"Critical error: {str(e)}")
        callback_stats['end_time'] = pd.Timestamp.now()
        callback_stats['duration'] = (callback_stats['end_time'] - start_time).total_seconds()
        
        # Return comprehensive error message
        error_message = f"An error occurred during feature analysis: {str(e)}"
        error_details = html.Div([
            html.H4("❌ Feature Analysis Failed", className="text-danger"),
            html.P(f"Error: {str(e)}"),
            html.Hr(),
            html.H6("Debug Information:"),
            html.Ul([
                html.Li(f"Steps completed: {len(callback_stats['steps_completed'])}"),
                html.Li(f"Errors: {len(callback_stats['errors'])}"),
                html.Li(f"Warnings: {len(callback_stats['warnings'])}"),
                html.Li(f"Duration: {callback_stats['duration']:.2f} seconds")
            ]),
            html.Hr(),
            html.P("Please check the logs for more details and try again.")
        ])
        
        return error_details, create_empty_figure(), {"error": str(e)}, {"error": str(e), "_callback_metadata": callback_stats}

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

def create_feature_results_display_enhanced(features):
    """Create enhanced HTML display for feature analysis results with better formatting and error handling."""
    logger.info("Creating enhanced feature results display...")
    
    if not features:
        return html.Div([
            html.H4("❌ No Features Extracted", className="mb-4 text-danger"),
            html.P("No features could be extracted from the signal. Please check your data and try again.")
        ])
    
    # Check for errors in features
    if isinstance(features, dict) and "error" in features:
        return html.Div([
            html.H4("❌ Feature Extraction Error", className="mb-4 text-danger"),
            html.P(f"Error: {features['error']}"),
            html.Hr(),
            html.P("Please check your data and try again.")
        ])
    
    # Create a comprehensive results display
    results_sections = []
    
    # Basic statistics section
    if any(key in features for key in ['mean', 'std', 'min', 'max', 'variance', 'zero_crossings']):
        basic_stats = []
        for key in ['mean', 'std', 'min', 'max', 'variance', 'zero_crossings']:
            if key in features and features[key] is not None:
                if isinstance(features[key], (int, float)):
                    basic_stats.append(html.Li(f"{key.upper()}: {features[key]:.3f}"))
                else:
                    basic_stats.append(html.Li(f"{key.upper()}: {features[key]}"))
        
        if basic_stats:
            results_sections.append(html.Div([
                html.H5("📊 Basic Statistics", className="mb-3 text-primary"),
                html.Ul(basic_stats, className="list-unstyled")
            ], className="col-md-6 mb-3"))
    
    # PPG features section
    ppg_features = []
    for key in ['rrv', 'rsa', 'fractal_dimension', 'dfa']:
        if key in features and features[key] is not None:
            if isinstance(features[key], (int, float)):
                ppg_features.append(html.Li(f"{key.upper()}: {features[key]:.3f}"))
            else:
                ppg_features.append(html.Li(f"{key.upper()}: {features[key]}"))
    
    if ppg_features:
        results_sections.append(html.Div([
            html.H5("💓 PPG Features", className="mb-3 text-success"),
            html.Ul(ppg_features, className="list-unstyled")
        ], className="col-md-6 mb-3"))
    
    # Light-based features section
    light_features = []
    for key in ['spo2', 'perfusion_index', 'respiratory_rate_ppg', 'ppr']:
        if key in features and features[key] is not None:
            if isinstance(features[key], (int, float)):
                light_features.append(html.Li(f"{key.upper()}: {features[key]:.3f}"))
            else:
                light_features.append(html.Li(f"{key.upper()}: {features[key]}"))
    
    if light_features:
        results_sections.append(html.Div([
            html.H5("💡 Light-Based Features", className="mb-3 text-info"),
            html.Ul(light_features, className="list-unstyled")
        ], className="col-md-6 mb-3"))
    
    # Morphology features section
    morph_features = []
    for key in ['systolic_duration', 'diastolic_duration', 'heart_rate', 'systolic_area', 'diastolic_area']:
        if key in features and features[key] is not None:
            if isinstance(features[key], (int, float)):
                morph_features.append(html.Li(f"{key.upper()}: {features[key]:.3f}"))
            else:
                morph_features.append(html.Li(f"{key.upper()}: {features[key]}"))
    
    if morph_features:
        results_sections.append(html.Div([
            html.H5("🔬 Morphology Features", className="mb-3 text-warning"),
            html.Ul(morph_features, className="list-unstyled")
        ], className="col-md-6 mb-3"))
    
    # ECG features section
    if 'ecg_autonomic' in features and features['ecg_autonomic']:
        ecg_features = []
        for key, value in features['ecg_autonomic'].items():
            if value is not None:
                if isinstance(value, (int, float)):
                    ecg_features.append(html.Li(f"{key.upper()}: {value:.3f}"))
                else:
                    ecg_features.append(html.Li(f"{key.upper()}: {value}"))
        
        if ecg_features:
            results_sections.append(html.Div([
                html.H5("⚡ ECG Features", className="mb-3 text-danger"),
                html.Ul(ecg_features, className="list-unstyled")
            ], className="col-md-6 mb-3"))
    
    # Synchronization features section
    if 'ecg_ppg_sync' in features and features['ecg_ppg_sync']:
        sync_features = []
        for key, value in features['ecg_ppg_sync'].items():
            if value is not None:
                if isinstance(value, (int, float)):
                    sync_features.append(html.Li(f"{key.upper()}: {value:.3f}"))
                else:
                    sync_features.append(html.Li(f"{key.upper()}: {value}"))
        
        if sync_features:
            results_sections.append(html.Div([
                html.H5("🔄 Synchronization Features", className="mb-3 text-secondary"),
                html.Ul(sync_features, className="list-unstyled")
            ], className="col-md-6 mb-3"))
    
    # Frequency domain features section
    freq_features = []
    for key in ['dominant_freq', 'spectral_energy']:
        if key in features and features[key] is not None:
            if isinstance(features[key], (int, float)):
                freq_features.append(html.Li(f"{key.upper()}: {features[key]:.3f}"))
            else:
                freq_features.append(html.Li(f"{key.upper()}: {features[key]}"))
    
    if freq_features:
        results_sections.append(html.Div([
            html.H5("🌊 Frequency Domain Features", className="mb-3 text-primary"),
            html.Ul(freq_features, className="list-unstyled")
        ], className="col-md-6 mb-3"))
    
    # If no specific features found, show all available features
    if not results_sections:
        all_features = []
        for key, value in features.items():
            if key.startswith('_') or key in ['error', 'warning']:
                continue
            if value is not None:
                if isinstance(value, (int, float)):
                    all_features.append(html.Li(f"{key}: {value:.3f}"))
                else:
                    all_features.append(html.Li(f"{key}: {value}"))
        
        if all_features:
            results_sections.append(html.Div([
                html.H5("📋 All Features", className="mb-3 text-dark"),
                html.Ul(all_features, className="list-unstyled")
            ], className="col-12 mb-3"))
    
    # Add callback metadata if available
    metadata_section = None
    if isinstance(features, dict) and '_callback_metadata' in features:
        metadata = features['_callback_metadata']
        metadata_section = html.Div([
            html.H5("📈 Analysis Metadata", className="mb-3 text-muted"),
            html.Div([
                html.P(f"✅ Steps completed: {metadata.get('total_steps', 0)}/22"),
                html.P(f"📊 Success rate: {metadata.get('success_rate', 0):.1%}"),
                html.P(f"⏱️ Duration: {metadata.get('duration', 0):.2f} seconds"),
                html.P(f"❌ Errors: {len(metadata.get('errors', []))}"),
                html.P(f"⚠️ Warnings: {len(metadata.get('warnings', []))}")
            ], className="text-muted")
        ], className="col-12 mb-3")
    
    # Create the main results container
    if results_sections:
        return html.Div([
            html.H4("🎯 Feature Analysis Results", className="mb-4 text-success"),
            html.Div([
                html.Div(results_sections, className="row"),
                metadata_section
            ] if metadata_section else [html.Div(results_sections, className="row")])
        ])
    else:
        return html.Div([
            html.H4("❌ No Features Extracted", className="mb-4 text-danger"),
            html.P("No features could be extracted from the signal. Please check your data and try again.")
        ])

def handle_asynchronous_response_error():
    """Handle the asynchronous response error that can occur in Dash callbacks."""
    logger.warning("⚠️ Asynchronous response error detected - implementing fallback mechanism")
    
    # Create a fallback response that doesn't require async operations
    fallback_response = html.Div([
        html.H4("⚠️ Analysis in Progress", className="mb-4 text-warning"),
        html.P("The feature analysis is taking longer than expected. This usually indicates:"),
        html.Ul([
            html.Li("Complex signal processing operations"),
            html.Li("Large dataset analysis"),
            html.Li("System resource constraints")
        ], className="mb-3"),
        html.Hr(),
        html.P("Please wait for the analysis to complete, or try again with a smaller dataset."),
        html.P("If the issue persists, check the application logs for more details.")
    ])
    
    return fallback_response, create_empty_figure(), {"status": "processing"}, {"status": "processing"}

def create_progress_tracker():
    """Create a progress tracking component for long-running operations."""
    return html.Div([
        html.H5("🔄 Analysis Progress", className="mb-3"),
        html.Div([
            html.Div(className="progress-bar", style={"width": "0%", "transition": "width 0.5s"}),
        ], className="progress mb-3", style={"height": "20px"}),
        html.Div(id="progress-text", children="Initializing..."),
        html.Div(id="progress-details", children="", className="text-muted small")
    ], id="progress-tracker", className="mb-4")

def create_error_summary(errors, warnings):
    """Create a comprehensive error and warning summary."""
    if not errors and not warnings:
        return None
    
    error_sections = []
    
    if errors:
        error_sections.append(html.Div([
            html.H6("❌ Errors", className="text-danger"),
            html.Ul([html.Li(str(error)) for error in errors], className="text-danger")
        ]))
    
    if warnings:
        error_sections.append(html.Div([
            html.H6("⚠️ Warnings", className="text-warning"),
            html.Ul([html.Li(str(warning)) for warning in warnings], className="text-warning")
        ]))
    
    return html.Div([
        html.H5("📋 Analysis Summary", className="mb-3"),
        html.Div(error_sections)
    ], className="mb-4")
