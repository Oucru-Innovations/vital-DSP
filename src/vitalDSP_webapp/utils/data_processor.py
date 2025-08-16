"""
Data processing utilities for vitalDSP webapp.
Handles data validation, processing, and quality assessment.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import logging

from ..config.settings import app_config, column_mapping

logger = logging.getLogger(__name__)


class DataProcessor:
    """Utility class for data processing operations."""
    
    @staticmethod
    def validate_file_extension(filename: str) -> bool:
        """
        Validate if file extension is supported.
        
        Parameters
        ----------
        filename : str
            Name of the file to validate
            
        Returns
        -------
        bool
            True if extension is supported, False otherwise
        """
        if not filename:
            return False
        
        file_ext = Path(filename).suffix.lower()
        return file_ext in app_config.ALLOWED_EXTENSIONS
    
    @staticmethod
    def read_file(file_path: str, filename: str) -> Optional[pd.DataFrame]:
        """
        Read file based on its extension.
        
        Parameters
        ----------
        file_path : str
            Path to the file
        filename : str
            Name of the file
            
        Returns
        -------
        Optional[pd.DataFrame]
            DataFrame if successful, None otherwise
        """
        try:
            file_ext = Path(filename).suffix.lower()
            
            if file_ext == '.csv':
                return pd.read_csv(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
            elif file_ext in ['.txt', '.dat']:
                # Try to read as CSV first, then as space-separated
                try:
                    return pd.read_csv(file_path, sep=None, engine='python')
                except:
                    return pd.read_csv(file_path, sep='\s+', engine='python')
            else:
                logger.warning(f"Unsupported file extension: {file_ext}")
                return None
                
        except Exception as e:
            logger.error(f"Error reading file {filename}: {str(e)}")
            return None
    
    @staticmethod
    def read_uploaded_content(content: str, filename: str) -> Optional[pd.DataFrame]:
        """
        Read uploaded file content.
        
        Parameters
        ----------
        content : str
            Base64 encoded file content
        filename : str
            Name of the uploaded file
            
        Returns
        -------
        Optional[pd.DataFrame]
            DataFrame if successful, None otherwise
        """
        try:
            import base64
            import io
            
            # Decode base64 content
            content_type, content_string = content.split(",")
            decoded = base64.b64decode(content_string)
            
            file_ext = Path(filename).suffix.lower()
            
            if file_ext == '.csv':
                return pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            elif file_ext in ['.xlsx', '.xls']:
                return pd.read_excel(io.BytesIO(decoded))
            elif file_ext in ['.txt', '.dat']:
                try:
                    return pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep=None, engine='python')
                except:
                    return pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep='\s+', engine='python')
            else:
                logger.warning(f"Unsupported file extension: {file_ext}")
                return None
                
        except Exception as e:
            logger.error(f"Error reading uploaded content {filename}: {str(e)}")
            return None
    
    @staticmethod
    def auto_detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
        """
        Auto-detect columns based on naming patterns.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to analyze
            
        Returns
        -------
        Dict[str, Optional[str]]
            Dictionary with detected column mappings
        """
        columns = df.columns.tolist()
        mapping = {
            'time': None,
            'signal': None,
            'red': None,
            'ir': None
        }
        
        for col in columns:
            col_lower = col.lower()
            
            # Time column detection
            if any(pattern in col_lower for pattern in column_mapping.TIME_PATTERNS):
                if not mapping['time']:
                    mapping['time'] = col
            
            # Signal column detection
            if any(pattern in col_lower for pattern in column_mapping.SIGNAL_PATTERNS):
                if not mapping['signal']:
                    mapping['signal'] = col
            
            # RED channel detection
            if any(pattern in col_lower for pattern in column_mapping.RED_PATTERNS):
                mapping['red'] = col
            
            # IR channel detection
            if any(pattern in col_lower for pattern in column_mapping.IR_PATTERNS):
                mapping['ir'] = col
        
        # If no specific signal column found, use first numeric column
        if not mapping['signal'] and len(df.columns) > 0:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                mapping['signal'] = numeric_cols[0]
        
        return mapping
    
    @staticmethod
    def assess_signal_quality(df: pd.DataFrame, signal_col: str) -> Dict[str, Any]:
        """
        Assess signal quality based on various metrics.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the signal
        signal_col : str
            Name of the signal column
            
        Returns
        -------
        Dict[str, Any]
            Quality assessment results
        """
        if signal_col not in df.columns:
            return {'quality_status': 'Unknown', 'error': 'Signal column not found'}
        
        signal_data = df[signal_col].dropna()
        if len(signal_data) == 0:
            return {'quality_status': 'Unknown', 'error': 'No valid signal data'}
        
        # Basic statistics
        min_val = float(signal_data.min())
        max_val = float(signal_data.max())
        mean_val = float(signal_data.mean())
        std_val = float(signal_data.std())
        
        # SNR estimation
        signal_power = np.mean(signal_data**2)
        noise_power = np.var(signal_data)
        snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
        
        # Artifact detection (outliers using IQR)
        q75, q25 = np.percentile(signal_data, [75, 25])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        artifact_count = np.sum((signal_data < lower_bound) | (signal_data > upper_bound))
        artifact_ratio = artifact_count / len(signal_data)
        
        # Quality assessment
        if snr_db > app_config.SNR_EXCELLENT_THRESHOLD and artifact_ratio < app_config.ARTIFACT_EXCELLENT_RATIO:
            quality_status = "Excellent"
        elif snr_db > app_config.SNR_GOOD_THRESHOLD and artifact_ratio < app_config.ARTIFACT_GOOD_RATIO:
            quality_status = "Good"
        elif snr_db > app_config.SNR_FAIR_THRESHOLD and artifact_ratio < app_config.ARTIFACT_FAIR_RATIO:
            quality_status = "Fair"
        else:
            quality_status = "Poor"
        
        return {
            'quality_status': quality_status,
            'min_value': min_val,
            'max_value': max_val,
            'mean_value': mean_val,
            'std_value': std_val,
            'snr_db': snr_db,
            'artifact_count': int(artifact_count),
            'artifact_ratio': float(artifact_ratio),
            'total_samples': len(signal_data)
        }
    
    @staticmethod
    def process_uploaded_data(df: pd.DataFrame, filename: str, sampling_freq: int, time_unit: str) -> Dict[str, Any]:
        """
        Process uploaded data and extract comprehensive information.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to process
        filename : str
            Name of the uploaded file
        sampling_freq : int
            Sampling frequency in Hz
        time_unit : str
            Time unit for the data
            
        Returns
        -------
        Dict[str, Any]
            Comprehensive data information
        """
        rows, cols = df.shape
        
        # Calculate duration
        duration_sec = rows / sampling_freq if sampling_freq else 0
        
        # Memory usage
        size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Signal quality assessment
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        quality_info = {}
        
        if len(numeric_cols) > 0:
            signal_col = numeric_cols[0]
            quality_info = DataProcessor.assess_signal_quality(df, signal_col)
        else:
            quality_info = {
                'quality_status': 'Unknown',
                'min_value': 0,
                'max_value': 0,
                'snr_db': 0,
                'artifact_count': 0
            }
        
        return {
            'filename': filename,
            'size_mb': size_mb,
            'format': Path(filename).suffix.upper() if filename else "Unknown",
            'rows': rows,
            'columns': cols,
            'duration_sec': duration_sec,
            'sampling_rate': sampling_freq,
            'time_unit': time_unit,
            'quality_status': quality_info.get('quality_status', 'Unknown'),
            'min_value': quality_info.get('min_value', 0),
            'max_value': quality_info.get('max_value', 0),
            'snr_db': quality_info.get('snr_db', 0),
            'artifact_count': quality_info.get('artifact_count', 0),
            'total_samples': quality_info.get('total_samples', rows)
        }
    
    @staticmethod
    def generate_sample_ppg_data(sampling_freq: int = 1000, duration: int = 50) -> pd.DataFrame:
        """
        Generate sample PPG data for testing purposes.
        
        Parameters
        ----------
        sampling_freq : int
            Sampling frequency in Hz
        duration : int
            Duration in seconds
            
        Returns
        -------
        pd.DataFrame
            Generated sample PPG data
        """
        # Time axis
        t = np.linspace(0, duration, int(duration * sampling_freq))
        
        # Generate realistic PPG signal components
        # Main cardiac component (1.2 Hz = 72 BPM)
        cardiac_freq = 1.2
        cardiac_signal = np.sin(2 * np.pi * cardiac_freq * t)
        
        # Respiratory component (0.2 Hz = 12 breaths/min)
        resp_freq = 0.2
        respiratory_signal = 0.3 * np.sin(2 * np.pi * resp_freq * t)
        
        # Add harmonics and noise
        harmonic_1 = 0.2 * np.sin(2 * np.pi * 2 * cardiac_freq * t)
        harmonic_2 = 0.1 * np.sin(2 * np.pi * 3 * cardiac_freq * t)
        
        # Combine signals
        base_signal = cardiac_signal + respiratory_signal + harmonic_1 + harmonic_2
        
        # Add amplitude modulation and baseline drift
        amplitude_mod = 1 + 0.2 * np.sin(2 * np.pi * 0.05 * t)
        baseline_drift = 0.1 * np.sin(2 * np.pi * 0.02 * t)
        
        # Final signal
        ppg_signal = amplitude_mod * base_signal + baseline_drift
        
        # Scale to realistic values
        ppg_signal = ppg_signal * 5000 + 7000
        
        # Add realistic noise
        noise = np.random.normal(0, 200, len(t))
        ppg_signal = ppg_signal + noise
        
        # Generate other channels
        red_signal = 230000 + 1000 * np.sin(2 * np.pi * cardiac_freq * t) + np.random.normal(0, 100, len(t))
        ir_signal = 275000 + 1000 * np.sin(2 * np.pi * cardiac_freq * t) + np.random.normal(0, 100, len(t))
        
        # Heart rate and SpO2
        hr_base = 72
        hr_variations = 5 * np.sin(2 * np.pi * 0.1 * t) + np.random.normal(0, 2, len(t))
        heart_rate = np.round(hr_base + hr_variations).astype(int)
        
        spo2_base = 98
        spo2_variations = np.random.normal(0, 1, len(t))
        spo2 = np.round(spo2_base + spo2_variations).astype(int)
        
        # Create DataFrame
        df = pd.DataFrame({
            'TIMESTAMP_MS': (t * 1000).astype(int),
            'PULSE_BPM': heart_rate,
            'SPO2_PCT': spo2,
            'PLETH': ppg_signal.astype(int),
            'RED_ADC': red_signal.astype(int),
            'IR_ADC': ir_signal.astype(int)
        })
        
        return df
    
    @staticmethod
    def validate_column_mapping(df: pd.DataFrame, mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        Validate column mapping against the DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate against
        mapping : Dict[str, str]
            Column mapping to validate
            
        Returns
        -------
        Dict[str, Any]
            Validation results
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'mapping': mapping
        }
        
        # Check if all mapped columns exist
        for col_type, col_name in mapping.items():
            if col_name and col_name not in df.columns:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"Column '{col_name}' not found in data")
        
        # Check if required columns are mapped
        if not mapping.get('time'):
            validation_result['warnings'].append("No time column mapped")
        
        if not mapping.get('signal'):
            validation_result['is_valid'] = False
            validation_result['errors'].append("No signal column mapped")
        
        # Check data types
        if mapping.get('time') and mapping['time'] in df.columns:
            time_col = df[mapping['time']]
            if not pd.api.types.is_numeric_dtype(time_col):
                validation_result['warnings'].append("Time column should be numeric")
        
        if mapping.get('signal') and mapping['signal'] in df.columns:
            signal_col = df[mapping['signal']]
            if not pd.api.types.is_numeric_dtype(signal_col):
                validation_result['is_valid'] = False
                validation_result['errors'].append("Signal column must be numeric")
        
        return validation_result
