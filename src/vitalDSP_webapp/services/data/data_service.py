"""
Data service for vitalDSP webapp.

This module provides data management and processing services.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class DataService:
    """Service for managing data operations."""
    
    def __init__(self):
        self.current_data: Optional[pd.DataFrame] = None
        self.data_config: Dict[str, Any] = {}
        self._data_store: Dict[str, Any] = {}
        self._column_mappings: Dict[str, Dict[str, str]] = {}
        self._next_id = 1
    
    def load_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load data from file path."""
        try:
            file_path = Path(file_path)
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() == '.txt':
                df = pd.read_csv(file_path, sep='\t')
            elif file_path.suffix.lower() == '.mat':
                # For .mat files, we'd need scipy.io
                logger.warning(".mat files not yet supported")
                return None
            else:
                logger.error(f"Unsupported file format: {file_path.suffix}")
                return None
            
            self.current_data = df
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
    
    def process_data(self, df: pd.DataFrame, sampling_freq: float, time_unit: str = "seconds") -> Dict[str, Any]:
        """Process uploaded data and return metadata."""
        try:
            # Basic data validation
            if df.empty:
                return {"error": "Data is empty"}
            
            # Calculate basic statistics
            signal_data = df.iloc[:, 1].values if len(df.columns) > 1 else df.iloc[:, 0].values
            
            # Convert time unit if needed
            if time_unit == "milliseconds":
                sampling_freq = sampling_freq / 1000
            elif time_unit == "minutes":
                sampling_freq = sampling_freq * 60
            
            duration = len(signal_data) / sampling_freq
            
            return {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "sampling_freq": sampling_freq,
                "time_unit": time_unit,
                "duration": duration,
                "signal_length": len(signal_data),
                "mean": float(np.mean(signal_data)),
                "std": float(np.std(signal_data)),
                "min": float(np.min(signal_data)),
                "max": float(np.max(signal_data))
            }
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return {"error": str(e)}
    
    def store_data(self, df: pd.DataFrame, info: Dict[str, Any]) -> str:
        """Store data with a unique ID and return the ID."""
        try:
            data_id = f"data_{self._next_id}"
            self._next_id += 1
            
            logger.info(f"=== STORING DATA ===")
            logger.info(f"Data ID: {data_id}")
            logger.info(f"Data shape: {df.shape}")
            logger.info(f"Data columns: {list(df.columns)}")
            logger.info(f"Data info: {info}")
            
            self._data_store[data_id] = {
                "data": df,
                "info": info,
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
            # Check if custom column mapping is provided in info
            if 'column_mapping' in info and info['column_mapping']:
                logger.info("Using custom column mapping from info")
                column_mapping = info['column_mapping']
                logger.info(f"Custom column mapping: {column_mapping}")
            else:
                # Auto-generate column mapping only if no custom mapping provided
                logger.info("No custom column mapping found, auto-detecting columns...")
                column_mapping = self._auto_detect_columns(df)
                logger.info(f"Auto-detected column mapping: {column_mapping}")
            
            self._column_mappings[data_id] = column_mapping
            
            logger.info(f"Data stored with ID: {data_id}")
            logger.info(f"Final column mapping: {column_mapping}")
            return data_id
            
        except Exception as e:
            logger.error(f"Error storing data: {e}")
            return None
    
    def get_data(self, data_id: str) -> Optional[pd.DataFrame]:
        """Get data by ID."""
        if data_id in self._data_store:
            return self._data_store[data_id]["data"]
        return None
    
    def get_all_data(self) -> Dict[str, Any]:
        """Get all stored data."""
        return self._data_store
    
    def get_column_mapping(self, data_id: str) -> Dict[str, str]:
        """Get column mapping for a specific data ID."""
        mapping = self._column_mappings.get(data_id, {})
        logger.info(f"Getting column mapping for {data_id}: {mapping}")
        return mapping
    
    def update_column_mapping(self, data_id: str, mapping: Dict[str, str]):
        """Update column mapping for a specific data ID."""
        self._column_mappings[data_id] = mapping
        logger.info(f"Column mapping updated for {data_id}: {mapping}")
    
    def _auto_detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Auto-detect column types based on data characteristics."""
        mapping = {}
        
        if len(df.columns) >= 2:
            # Look for specific column types based on names first
            for col in df.columns:
                col_lower = col.lower()
                
                # Time-related columns - only set if not already set
                # Be more specific about time column detection to avoid false positives
                if "time" not in mapping and (any(keyword in col_lower for keyword in ['time', 'timestamp']) or col_lower == 't'):
                    mapping["time"] = col
                
                # Signal columns - prioritize waveform/pleth columns
                elif any(keyword in col_lower for keyword in ['waveform', 'pleth', 'pl']):
                    mapping["signal"] = col
                    logger.info(f"Found waveform/pleth column: {col}")
                
                # Other signal columns
                elif any(keyword in col_lower for keyword in ['signal', 'ppg', 'ecg']):
                    if "signal" not in mapping:  # Only set if no waveform/pleth found
                        mapping["signal"] = col
                        logger.info(f"Found signal column: {col}")
                
                # RED channel (for pulse oximetry) - only set if not already set
                elif "red" not in mapping and any(keyword in col_lower for keyword in ['red']):
                    mapping["red"] = col
                
                # IR channel (for pulse oximetry) - only set if not already set  
                elif "ir" not in mapping and any(keyword in col_lower for keyword in ['ir', 'infrared']):
                    mapping["ir"] = col
            
            # If no specific columns found, use defaults based on position
            # Priority: time = first column, signal = second column (if available)
            # ALWAYS assign time to first column if not already detected
            if "time" not in mapping and len(df.columns) > 0:
                mapping["time"] = df.columns[0]
                logger.info(f"Using default time column: {df.columns[0]}")
                
            # Only assign signal column if not already detected
            if "signal" not in mapping:
                if len(df.columns) > 1:
                    mapping["signal"] = df.columns[1]
                    logger.info(f"Using default signal column: {df.columns[1]}")
                elif len(df.columns) == 1:
                    # If only one column, use it for signal (time was already assigned)
                    mapping["signal"] = df.columns[0]
                    logger.info(f"Using single column for signal: {df.columns[0]}")
        
        logger.info(f"Auto-detected column mapping: {mapping}")
        return mapping
    
    def get_data_info(self, data_id: str) -> Optional[Dict[str, Any]]:
        """Get data info by ID."""
        if data_id in self._data_store:
            return self._data_store[data_id]["info"]
        return None
    
    def get_current_data(self) -> Optional[pd.DataFrame]:
        """Get current data."""
        return self.current_data
    
    def get_config(self) -> Dict[str, Any]:
        """Get current data configuration."""
        return self.data_config
    
    def update_config(self, config: Dict[str, Any]):
        """Update data configuration."""
        self.data_config.update(config)
    
    def clear_data(self, data_id: str):
        """Clear data by ID."""
        if data_id in self._data_store:
            del self._data_store[data_id]
        if data_id in self._column_mappings:
            del self._column_mappings[data_id]
        logger.info(f"Data cleared for ID: {data_id}")
    
    def clear_all_data(self):
        """Clear all stored data."""
        self.current_data = None
        self.data_config.clear()
        self._data_store.clear()
        self._column_mappings.clear()
        self._next_id = 1
        logger.info("All data cleared")
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current data configuration."""
        return self.data_config.copy()  # Return a copy, not the original
    
    def set_column_mapping(self, data_id: str, mapping: Dict[str, str]):
        """Set column mapping for a specific data ID."""
        self._column_mappings[data_id] = mapping
        logger.info(f"Column mapping set for {data_id}: {mapping}")
    
    def get_data_summary(self) -> Optional[Dict[str, Any]]:
        """Get summary of current data."""
        if self.current_data is None or self.current_data.empty:
            return None
        
        return {
            "shape": self.current_data.shape,
            "columns": self.current_data.columns.tolist(),
            "data_config": self.data_config
        }


# Global instance
_data_service = DataService()


def get_data_service() -> DataService:
    """Get the global data service instance."""
    return _data_service
