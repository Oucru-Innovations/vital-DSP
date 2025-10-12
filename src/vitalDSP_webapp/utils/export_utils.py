"""
Export utilities for vitalDSP webapp.

This module provides functions to export data in various formats (CSV, JSON, Excel)
for different analysis pages.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import io
import base64


def prepare_download_data(data: Union[str, bytes], filename: str, content_type: str) -> Dict[str, str]:
    """
    Prepare data for download via Dash dcc.Download component.

    Args:
        data: Data to download (string or bytes)
        filename: Name of the file
        content_type: MIME type of the file

    Returns:
        dict: Dictionary with 'content', 'filename', and 'type' keys
    """
    if isinstance(data, bytes):
        # Encode bytes to base64 string
        data = base64.b64encode(data).decode('utf-8')
        return {
            'content': data,
            'filename': filename,
            'base64': True,
            'type': content_type
        }
    return {
        'content': data,
        'filename': filename,
        'type': content_type
    }


def export_filtered_signal_csv(signal: np.ndarray, time: np.ndarray,
                                sampling_freq: float,
                                metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Export filtered signal data to CSV format with timestamps.

    Args:
        signal: Filtered signal array
        time: Time array
        sampling_freq: Sampling frequency
        metadata: Optional metadata dict

    Returns:
        str: CSV content as string
    """
    df = pd.DataFrame({
        'time': time,
        'signal': signal
    })

    # Add metadata as comments
    csv_buffer = io.StringIO()
    csv_buffer.write(f"# Filtered Signal Export\n")
    csv_buffer.write(f"# Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    csv_buffer.write(f"# Sampling Frequency: {sampling_freq} Hz\n")
    csv_buffer.write(f"# Duration: {len(signal)/sampling_freq:.2f} seconds\n")
    csv_buffer.write(f"# Samples: {len(signal)}\n")

    if metadata:
        for key, value in metadata.items():
            csv_buffer.write(f"# {key}: {value}\n")

    csv_buffer.write("#\n")
    df.to_csv(csv_buffer, index=False)

    return csv_buffer.getvalue()


def export_filtered_signal_json(signal: np.ndarray, time: np.ndarray,
                                 sampling_freq: float,
                                 metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Export filtered signal data to JSON format.

    Args:
        signal: Filtered signal array
        time: Time array
        sampling_freq: Sampling frequency
        metadata: Optional metadata dict

    Returns:
        str: JSON content as string
    """
    export_data = {
        'metadata': {
            'export_date': datetime.now().isoformat(),
            'sampling_frequency': sampling_freq,
            'duration': len(signal) / sampling_freq,
            'samples': len(signal),
            **(metadata or {})
        },
        'data': {
            'time': time.tolist(),
            'signal': signal.tolist()
        }
    }

    return json.dumps(export_data, indent=2)


def export_features_csv(features: Dict[str, Any],
                        signal_type: str = "Unknown") -> str:
    """
    Export extracted features to CSV format.

    Args:
        features: Dictionary of extracted features
        signal_type: Type of signal (PPG, ECG, etc.)

    Returns:
        str: CSV content as string
    """
    # Flatten nested dictionaries
    flattened = {}

    def flatten_dict(d, parent_key=''):
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                flatten_dict(v, new_key)
            elif isinstance(v, (list, np.ndarray)):
                # For arrays, store as JSON string or first few values
                if len(v) <= 10:
                    flattened[new_key] = str(v)
                else:
                    flattened[f"{new_key}_mean"] = np.mean(v)
                    flattened[f"{new_key}_std"] = np.std(v)
                    flattened[f"{new_key}_min"] = np.min(v)
                    flattened[f"{new_key}_max"] = np.max(v)
            else:
                flattened[new_key] = v

    flatten_dict(features)

    # Create DataFrame
    df = pd.DataFrame([flattened])

    # Add metadata header
    csv_buffer = io.StringIO()
    csv_buffer.write(f"# Feature Extraction Export\n")
    csv_buffer.write(f"# Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    csv_buffer.write(f"# Signal Type: {signal_type}\n")
    csv_buffer.write(f"# Features Count: {len(flattened)}\n")
    csv_buffer.write("#\n")

    df.to_csv(csv_buffer, index=False)

    return csv_buffer.getvalue()


def export_features_json(features: Dict[str, Any],
                         signal_type: str = "Unknown") -> str:
    """
    Export extracted features to JSON format.

    Args:
        features: Dictionary of extracted features
        signal_type: Type of signal (PPG, ECG, etc.)

    Returns:
        str: JSON content as string
    """
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj

    export_data = {
        'metadata': {
            'export_date': datetime.now().isoformat(),
            'signal_type': signal_type,
            'features_count': len(features)
        },
        'features': convert_to_serializable(features)
    }

    return json.dumps(export_data, indent=2)


def export_quality_metrics_csv(quality_results: Dict[str, Any]) -> str:
    """
    Export signal quality assessment metrics to CSV format.

    Args:
        quality_results: Dictionary of quality assessment results

    Returns:
        str: CSV content as string
    """
    csv_buffer = io.StringIO()
    csv_buffer.write(f"# Signal Quality Assessment Export\n")
    csv_buffer.write(f"# Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    csv_buffer.write("#\n")

    # Overall scores
    if 'overall_score' in quality_results:
        overall = quality_results['overall_score']
        csv_buffer.write("# Overall Quality\n")
        csv_buffer.write(f"Score,{overall.get('score', 'N/A')}\n")
        csv_buffer.write(f"Percentage,{overall.get('percentage', 'N/A')}\n")
        csv_buffer.write(f"Quality,{overall.get('quality', 'N/A')}\n")
        csv_buffer.write("#\n")

    # Individual metrics
    metrics_data = []
    for metric_name, metric_data in quality_results.items():
        if metric_name == 'overall_score':
            continue
        if isinstance(metric_data, dict):
            row = {'metric': metric_name}
            for key, value in metric_data.items():
                if not isinstance(value, (list, dict)):
                    row[key] = value
            metrics_data.append(row)

    if metrics_data:
        df = pd.DataFrame(metrics_data)
        df.to_csv(csv_buffer, index=False)

    return csv_buffer.getvalue()


def export_quality_metrics_json(quality_results: Dict[str, Any]) -> str:
    """
    Export signal quality assessment metrics to JSON format.

    Args:
        quality_results: Dictionary of quality assessment results

    Returns:
        str: JSON content as string
    """
    # Convert numpy types to native Python types
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj

    export_data = {
        'metadata': {
            'export_date': datetime.now().isoformat(),
            'export_type': 'signal_quality_assessment'
        },
        'quality_metrics': convert_to_serializable(quality_results)
    }

    return json.dumps(export_data, indent=2)


def export_respiratory_analysis_csv(respiratory_data: Dict[str, Any]) -> str:
    """
    Export respiratory analysis results to CSV format.

    Args:
        respiratory_data: Dictionary of respiratory analysis results

    Returns:
        str: CSV content as string
    """
    csv_buffer = io.StringIO()
    csv_buffer.write(f"# Respiratory Analysis Export\n")
    csv_buffer.write(f"# Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    csv_buffer.write("#\n")

    # Respiratory rate estimates
    if 'respiratory_rate' in respiratory_data:
        rr = respiratory_data['respiratory_rate']
        csv_buffer.write("# Respiratory Rate\n")
        if isinstance(rr, dict):
            for key, value in rr.items():
                if not isinstance(value, (list, dict)):
                    csv_buffer.write(f"{key},{value}\n")
        else:
            csv_buffer.write(f"respiratory_rate,{rr}\n")
        csv_buffer.write("#\n")

    # Additional metrics
    metrics_rows = []
    for key, value in respiratory_data.items():
        if key == 'respiratory_rate':
            continue
        if isinstance(value, dict):
            row = {'metric': key}
            for k, v in value.items():
                if not isinstance(v, (list, dict)):
                    row[k] = v
            metrics_rows.append(row)
        elif not isinstance(value, (list, dict)):
            metrics_rows.append({'metric': key, 'value': value})

    if metrics_rows:
        df = pd.DataFrame(metrics_rows)
        df.to_csv(csv_buffer, index=False)

    return csv_buffer.getvalue()


def export_respiratory_analysis_json(respiratory_data: Dict[str, Any]) -> str:
    """
    Export respiratory analysis results to JSON format.

    Args:
        respiratory_data: Dictionary of respiratory analysis results

    Returns:
        str: JSON content as string
    """
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj

    export_data = {
        'metadata': {
            'export_date': datetime.now().isoformat(),
            'export_type': 'respiratory_analysis'
        },
        'respiratory_data': convert_to_serializable(respiratory_data)
    }

    return json.dumps(export_data, indent=2)


def export_transform_results_csv(transform_data: Dict[str, Any],
                                  transform_type: str = "Unknown") -> str:
    """
    Export transform results to CSV format.

    Args:
        transform_data: Dictionary of transform results
        transform_type: Type of transform (FFT, Wavelet, etc.)

    Returns:
        str: CSV content as string
    """
    csv_buffer = io.StringIO()
    csv_buffer.write(f"# Transform Results Export\n")
    csv_buffer.write(f"# Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    csv_buffer.write(f"# Transform Type: {transform_type}\n")
    csv_buffer.write("#\n")

    # Try to create a structured dataframe
    df_data = {}
    for key, value in transform_data.items():
        if isinstance(value, np.ndarray) and value.ndim == 1:
            df_data[key] = value
        elif isinstance(value, list) and len(value) > 0:
            df_data[key] = value
        elif not isinstance(value, (dict, list, np.ndarray)):
            # Scalar values go in header
            csv_buffer.write(f"# {key}: {value}\n")

    if df_data:
        # Ensure all arrays have same length
        max_len = max(len(v) for v in df_data.values())
        for key in list(df_data.keys()):
            if len(df_data[key]) < max_len:
                # Pad with NaN
                df_data[key] = np.pad(df_data[key], (0, max_len - len(df_data[key])),
                                      constant_values=np.nan)

        df = pd.DataFrame(df_data)
        df.to_csv(csv_buffer, index=False)

    return csv_buffer.getvalue()


def export_transform_results_json(transform_data: Dict[str, Any],
                                   transform_type: str = "Unknown") -> str:
    """
    Export transform results to JSON format.

    Args:
        transform_data: Dictionary of transform results
        transform_type: Type of transform (FFT, Wavelet, etc.)

    Returns:
        str: JSON content as string
    """
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, complex):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj

    export_data = {
        'metadata': {
            'export_date': datetime.now().isoformat(),
            'transform_type': transform_type,
            'export_type': 'transform_results'
        },
        'transform_data': convert_to_serializable(transform_data)
    }

    return json.dumps(export_data, indent=2)


def export_time_domain_analysis_csv(analysis_results: Dict[str, Any]) -> str:
    """
    Export time domain analysis results to CSV format.

    Args:
        analysis_results: Dictionary of time domain analysis results

    Returns:
        str: CSV content as string
    """
    return export_features_csv(analysis_results, signal_type="Time Domain")


def export_time_domain_analysis_json(analysis_results: Dict[str, Any]) -> str:
    """
    Export time domain analysis results to JSON format.

    Args:
        analysis_results: Dictionary of time domain analysis results

    Returns:
        str: JSON content as string
    """
    return export_features_json(analysis_results, signal_type="Time Domain")


def export_frequency_domain_analysis_csv(analysis_results: Dict[str, Any]) -> str:
    """
    Export frequency domain analysis results to CSV format.

    Args:
        analysis_results: Dictionary of frequency domain analysis results

    Returns:
        str: CSV content as string
    """
    return export_features_csv(analysis_results, signal_type="Frequency Domain")


def export_frequency_domain_analysis_json(analysis_results: Dict[str, Any]) -> str:
    """
    Export frequency domain analysis results to JSON format.

    Args:
        analysis_results: Dictionary of frequency domain analysis results

    Returns:
        str: JSON content as string
    """
    return export_features_json(analysis_results, signal_type="Frequency Domain")
