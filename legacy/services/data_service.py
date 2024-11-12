"""
Data service for vitalDSP webapp.
Handles data storage, retrieval, and management operations.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import tempfile
import os
from datetime import datetime
import uuid

from ..config.settings import app_config


class DataService:
    """Service for managing data operations in the webapp."""
    
    def __init__(self):
        """Initialize the data service."""
        self._data_store: Dict[str, Any] = {}
        self._config_store: Dict[str, Any] = {}
        self._column_mappings: Dict[str, Dict[str, str]] = {}
        self._analysis_results: Dict[str, Any] = {}
        self._session_id = str(uuid.uuid4())
        
        # Ensure data directory exists
        os.makedirs(app_config.UPLOAD_FOLDER, exist_ok=True)
    
    def store_data(self, data: pd.DataFrame, metadata: Dict[str, Any]) -> str:
        """
        Store data with metadata and return a unique identifier.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to store
        metadata : Dict[str, Any]
            Metadata about the data
            
        Returns
        -------
        str
            Unique identifier for the stored data
        """
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"=== STORE_DATA CALLED ===")
        logger.info(f"Data shape: {data.shape}")
        logger.info(f"Data columns: {data.columns.tolist()}")
        logger.info(f"Metadata: {metadata}")
        
        data_id = str(uuid.uuid4())
        logger.info(f"Generated data_id: {data_id}")
        
        # Store data
        self._data_store[data_id] = {
            'dataframe': data.to_dict('records'),
            'columns': data.columns.tolist(),
            'shape': data.shape,
            'dtypes': data.dtypes.to_dict(),
            'metadata': metadata,
            'timestamp': datetime.now().isoformat(),
            'session_id': self._session_id
        }
        
        logger.info(f"Data stored successfully with ID: {data_id}")
        logger.info(f"Data store now contains {len(self._data_store)} entries")
        logger.info(f"Data store keys: {list(self._data_store.keys())}")
        
        return data_id
    
    def get_data(self, data_id: str) -> Optional[pd.DataFrame]:
        """
        Retrieve data by ID.
        
        Parameters
        ----------
        data_id : str
            The unique identifier for the data
            
        Returns
        -------
        Optional[pd.DataFrame]
            The stored data or None if not found
        """
        if data_id not in self._data_store:
            return None
        
        data_info = self._data_store[data_id]
        return pd.DataFrame(data_info['dataframe'])
    
    def get_metadata(self, data_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for stored data.
        
        Parameters
        ----------
        data_id : str
            The unique identifier for the data
            
        Returns
        -------
        Optional[Dict[str, Any]]
            Metadata for the data or None if not found
        """
        if data_id not in self._data_store:
            return None
        
        return self._data_store[data_id]['metadata']
    
    def store_column_mapping(self, data_id: str, mapping: Dict[str, str]) -> None:
        """
        Store column mapping for a dataset.
        
        Parameters
        ----------
        data_id : str
            The unique identifier for the data
        mapping : Dict[str, str]
            Column mapping (e.g., {'time': 'timestamp', 'signal': 'pleth'})
        """
        self._column_mappings[data_id] = mapping
    
    def get_column_mapping(self, data_id: str) -> Optional[Dict[str, str]]:
        """
        Get column mapping for a dataset.
        
        Parameters
        ----------
        data_id : str
            The unique identifier for the data
            
        Returns
        -------
        Optional[Dict[str, str]]
            Column mapping or None if not found
        """
        return self._column_mappings.get(data_id)
    
    def get_all_data(self) -> Dict[str, Any]:
        """
        Get all stored data with their metadata and info.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of all stored data with their metadata and info
        """
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"=== GET_ALL_DATA CALLED ===")
        logger.info(f"Data store keys: {list(self._data_store.keys())}")
        logger.info(f"Data store size: {len(self._data_store)}")
        
        all_data = {}
        for data_id, data_info in self._data_store.items():
            logger.info(f"Processing data_id: {data_id}")
            logger.info(f"Data info keys: {list(data_info.keys())}")
            
            all_data[data_id] = {
                'dataframe': data_info['dataframe'],
                'columns': data_info['columns'],
                'shape': data_info['shape'],
                'dtypes': data_info['dtypes'],
                'metadata': data_info['metadata'],
                'timestamp': data_info['timestamp'],
                'session_id': data_info['session_id'],
                'info': data_info['metadata']  # Store metadata as 'info' for compatibility
            }
            
            logger.info(f"Added data_id {data_id} to all_data")
        
        logger.info(f"Returning {len(all_data)} data entries")
        return all_data
    
    def update_data_info(self, data_id: str, info_updates: Dict[str, Any]) -> bool:
        """
        Update data info/metadata with additional information.
        
        Parameters
        ----------
        data_id : str
            The unique identifier for the data
        info_updates : Dict[str, Any]
            Additional info to merge with existing metadata
            
        Returns
        -------
        bool
            True if update was successful, False if data not found
        """
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"=== UPDATE_DATA_INFO CALLED ===")
        logger.info(f"data_id: {data_id}")
        logger.info(f"info_updates: {info_updates}")
        logger.info(f"Data store keys: {list(self._data_store.keys())}")
        
        if data_id not in self._data_store:
            logger.error(f"Data ID {data_id} not found in data store")
            return False
        
        # Merge new info with existing metadata
        current_metadata = self._data_store[data_id]['metadata']
        logger.info(f"Current metadata: {current_metadata}")
        
        updated_metadata = {**current_metadata, **info_updates}
        logger.info(f"Updated metadata: {updated_metadata}")
        
        # Update the metadata
        self._data_store[data_id]['metadata'] = updated_metadata
        
        logger.info(f"Successfully updated data info for {data_id}")
        return True
    
    def store_config(self, config: Dict[str, Any]) -> str:
        """
        Store configuration data.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration data to store
            
        Returns
        -------
        str
            Unique identifier for the stored configuration
        """
        config_id = str(uuid.uuid4())
        self._config_store[config_id] = {
            'config': config,
            'timestamp': datetime.now().isoformat(),
            'session_id': self._session_id
        }
        return config_id
    
    def get_config(self, config_id: str) -> Optional[Dict[str, Any]]:
        """
        Get stored configuration.
        
        Parameters
        ----------
        config_id : str
            The unique identifier for the configuration
            
        Returns
        -------
        Optional[Dict[str, Any]]
            Configuration data or None if not found
        """
        return self._config_store.get(config_id, {}).get('config')
    
    def store_analysis_result(self, data_id: str, analysis_type: str, result: Any) -> str:
        """
        Store analysis results.
        
        Parameters
        ----------
        data_id : str
            The unique identifier for the data
        analysis_type : str
            Type of analysis performed
        result : Any
            Analysis result to store
            
        Returns
        -------
        str
            Unique identifier for the stored result
        """
        result_id = str(uuid.uuid4())
        
        if data_id not in self._analysis_results:
            self._analysis_results[data_id] = {}
        
        self._analysis_results[data_id][analysis_type] = {
            'result': result,
            'timestamp': datetime.now().isoformat(),
            'result_id': result_id
        }
        
        return result_id
    
    def get_analysis_result(self, data_id: str, analysis_type: str) -> Optional[Any]:
        """
        Get stored analysis result.
        
        Parameters
        ----------
        data_id : str
            The unique identifier for the data
        analysis_type : str
            Type of analysis to retrieve
            
        Returns
        -------
        Optional[Any]
            Analysis result or None if not found
        """
        if data_id not in self._analysis_results:
            return None
        
        return self._analysis_results[data_id].get(analysis_type, {}).get('result')
    
    def list_stored_data(self) -> List[Dict[str, Any]]:
        """
        List all stored datasets with basic information.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of stored data information
        """
        data_list = []
        for data_id, data_info in self._data_store.items():
            data_list.append({
                'id': data_id,
                'filename': data_info['metadata'].get('filename', 'Unknown'),
                'shape': data_info['shape'],
                'timestamp': data_info['timestamp'],
                'has_mapping': data_id in self._column_mappings,
                'has_analysis': data_id in self._analysis_results
            })
        
        return data_list
    
    def clear_data(self, data_id: str) -> bool:
        """
        Clear stored data and related information.
        
        Parameters
        ----------
        data_id : str
            The unique identifier for the data
            
        Returns
        -------
        bool
            True if data was cleared, False if not found
        """
        if data_id not in self._data_store:
            return False
        
        # Remove data
        del self._data_store[data_id]
        
        # Remove related mappings
        if data_id in self._column_mappings:
            del self._column_mappings[data_id]
        
        # Remove related analysis results
        if data_id in self._analysis_results:
            del self._analysis_results[data_id]
        
        return True
    
    def clear_session(self) -> None:
        """Clear all data for the current session."""
        self._data_store.clear()
        self._config_store.clear()
        self._column_mappings.clear()
        self._analysis_results.clear()
        self._session_id = str(uuid.uuid4())
    
    def export_data(self, data_id: str, format: str = 'csv') -> Optional[str]:
        """
        Export data to a file.
        
        Parameters
        ----------
        data_id : str
            The unique identifier for the data
        format : str
            Export format ('csv', 'xlsx', 'json')
            
        Returns
        -------
        Optional[str]
            Path to exported file or None if export failed
        """
        data = self.get_data(data_id)
        if data is None:
            return None
        
        metadata = self.get_metadata(data_id)
        filename = metadata.get('filename', 'exported_data')
        base_name = Path(filename).stem
        
        try:
            if format == 'csv':
                file_path = os.path.join(app_config.UPLOAD_FOLDER, f"{base_name}_export.csv")
                data.to_csv(file_path, index=False)
            elif format == 'xlsx':
                file_path = os.path.join(app_config.UPLOAD_FOLDER, f"{base_name}_export.xlsx")
                data.to_excel(file_path, index=False)
            elif format == 'json':
                file_path = os.path.join(app_config.UPLOAD_FOLDER, f"{base_name}_export.json")
                data.to_json(file_path, orient='records', indent=2)
            else:
                return None
            
            return file_path
        except Exception:
            return None
    
    def get_data_summary(self, data_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of stored data.
        
        Parameters
        ----------
        data_id : str
            The unique identifier for the data
            
        Returns
        -------
        Optional[Dict[str, Any]]
            Data summary or None if not found
        """
        data = self.get_data(data_id)
        if data is None:
            return None
        
        metadata = self.get_metadata(data_id)
        mapping = self.get_column_mapping(data_id)
        
        # Calculate basic statistics
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        stats = {}
        
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    stats[col] = {
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()),
                        'count': int(len(col_data))
                    }
        
        return {
            'id': data_id,
            'filename': metadata.get('filename', 'Unknown'),
            'shape': data.shape,
            'columns': data.columns.tolist(),
            'dtypes': data.dtypes.to_dict(),
            'metadata': metadata,
            'column_mapping': mapping,
            'statistics': stats,
            'timestamp': self._data_store[data_id]['timestamp'],
            'has_analysis': data_id in self._analysis_results
        }


# Global data service instance
data_service = DataService()


def get_data_service() -> DataService:
    """Get the global data service instance."""
    return data_service
