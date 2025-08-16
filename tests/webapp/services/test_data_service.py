import pytest
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json
from datetime import datetime

# Add the src directory to the Python path so tests can import vitalDSP_webapp modules
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from vitalDSP_webapp.services.data_service import DataService, get_data_service


class TestDataService:
    """Test class for DataService."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.data_service = DataService()
        
        # Create test data
        self.test_df = pd.DataFrame({
            'time': np.arange(100),
            'signal': np.random.randn(100),
            'red': np.random.randn(100),
            'ir': np.random.randn(100)
        })
        
        self.test_metadata = {
            'filename': 'test.csv',
            'sampling_freq': 1000,
            'time_unit': 'ms',
            'upload_method': 'file_upload'
        }

    def test_init(self):
        """Test DataService initialization."""
        service = DataService()
        
        assert hasattr(service, '_data_store')
        assert hasattr(service, '_config_store')
        assert hasattr(service, '_column_mappings')
        assert hasattr(service, '_analysis_results')
        assert hasattr(service, '_session_id')
        
        assert isinstance(service._data_store, dict)
        assert isinstance(service._config_store, dict)
        assert isinstance(service._column_mappings, dict)
        assert isinstance(service._analysis_results, dict)
        assert isinstance(service._session_id, str)

    @patch('vitalDSP_webapp.services.data_service.app_config')
    def test_init_creates_upload_folder(self, mock_app_config):
        """Test that initialization creates the upload folder."""
        mock_app_config.UPLOAD_FOLDER = '/tmp/test_uploads'
        
        with patch('os.makedirs') as mock_makedirs:
            service = DataService()
            mock_makedirs.assert_called_once_with('/tmp/test_uploads', exist_ok=True)

    def test_store_data(self):
        """Test storing data with metadata."""
        data_id = self.data_service.store_data(self.test_df, self.test_metadata)
        
        assert isinstance(data_id, str)
        assert data_id in self.data_service._data_store
        
        stored_data = self.data_service._data_store[data_id]
        assert 'dataframe' in stored_data
        assert 'columns' in stored_data
        assert 'shape' in stored_data
        assert 'dtypes' in stored_data
        assert 'metadata' in stored_data
        assert 'timestamp' in stored_data
        assert 'session_id' in stored_data
        
        # Check data integrity
        assert stored_data['shape'] == self.test_df.shape
        assert stored_data['columns'] == self.test_df.columns.tolist()
        assert stored_data['metadata'] == self.test_metadata

    def test_store_data_multiple_datasets(self):
        """Test storing multiple datasets."""
        df1 = pd.DataFrame({'col1': [1, 2, 3]})
        df2 = pd.DataFrame({'col2': [4, 5, 6]})
        
        id1 = self.data_service.store_data(df1, {'filename': 'file1.csv'})
        id2 = self.data_service.store_data(df2, {'filename': 'file2.csv'})
        
        assert id1 != id2
        assert len(self.data_service._data_store) == 2
        assert id1 in self.data_service._data_store
        assert id2 in self.data_service._data_store

    def test_get_data_existing(self):
        """Test retrieving existing data."""
        data_id = self.data_service.store_data(self.test_df, self.test_metadata)
        retrieved_df = self.data_service.get_data(data_id)
        
        assert retrieved_df is not None
        assert isinstance(retrieved_df, pd.DataFrame)
        pd.testing.assert_frame_equal(retrieved_df, self.test_df)

    def test_get_data_nonexistent(self):
        """Test retrieving non-existent data."""
        retrieved_df = self.data_service.get_data('nonexistent_id')
        assert retrieved_df is None

    def test_get_metadata_existing(self):
        """Test retrieving metadata for existing data."""
        data_id = self.data_service.store_data(self.test_df, self.test_metadata)
        retrieved_metadata = self.data_service.get_metadata(data_id)
        
        assert retrieved_metadata is not None
        assert retrieved_metadata == self.test_metadata

    def test_get_metadata_nonexistent(self):
        """Test retrieving metadata for non-existent data."""
        retrieved_metadata = self.data_service.get_metadata('nonexistent_id')
        assert retrieved_metadata is None

    def test_store_column_mapping(self):
        """Test storing column mapping."""
        data_id = self.data_service.store_data(self.test_df, self.test_metadata)
        
        column_mapping = {
            'time': 'time',
            'signal': 'signal',
            'red': 'red',
            'ir': 'ir'
        }
        
        self.data_service.store_column_mapping(data_id, column_mapping)
        
        assert data_id in self.data_service._column_mappings
        assert self.data_service._column_mappings[data_id] == column_mapping

    def test_get_column_mapping_existing(self):
        """Test retrieving existing column mapping."""
        data_id = self.data_service.store_data(self.test_df, self.test_metadata)
        
        column_mapping = {
            'time': 'time',
            'signal': 'signal',
            'red': 'red',
            'ir': 'ir'
        }
        
        self.data_service.store_column_mapping(data_id, column_mapping)
        retrieved_mapping = self.data_service.get_column_mapping(data_id)
        
        assert retrieved_mapping == column_mapping

    def test_get_column_mapping_nonexistent(self):
        """Test retrieving column mapping for non-existent data."""
        retrieved_mapping = self.data_service.get_column_mapping('nonexistent_id')
        assert retrieved_mapping is None

    def test_store_analysis_result(self):
        """Test storing analysis results."""
        data_id = self.data_service.store_data(self.test_df, self.test_metadata)
        
        analysis_result = {
            'hrv_features': {'mean_hr': 75.5, 'sdnn': 45.2},
            'quality_metrics': {'snr': 25.3}
        }
        
        self.data_service.store_analysis_result(data_id, 'hrv_analysis', analysis_result)
        
        assert data_id in self.data_service._analysis_results
        assert 'hrv_analysis' in self.data_service._analysis_results[data_id]
        assert self.data_service._analysis_results[data_id]['hrv_analysis'] == analysis_result

    def test_get_analysis_result_existing(self):
        """Test retrieving existing analysis results."""
        data_id = self.data_service.store_data(self.test_df, self.test_metadata)
        
        analysis_result = {'feature1': 'value1'}
        self.data_service.store_analysis_result(data_id, 'test_analysis', analysis_result)
        
        retrieved_result = self.data_service.get_analysis_result(data_id, 'test_analysis')
        assert retrieved_result == analysis_result

    def test_get_analysis_result_nonexistent(self):
        """Test retrieving analysis results for non-existent data."""
        retrieved_result = self.data_service.get_analysis_result('nonexistent_id', 'test_analysis')
        assert retrieved_result is None

    def test_get_analysis_result_nonexistent_analysis(self):
        """Test retrieving non-existent analysis type for existing data."""
        data_id = self.data_service.store_data(self.test_df, self.test_metadata)
        
        retrieved_result = self.data_service.get_analysis_result(data_id, 'nonexistent_analysis')
        assert retrieved_result is None

    def test_list_stored_data(self):
        """Test listing all stored data IDs."""
        df1 = pd.DataFrame({'col1': [1, 2]})
        df2 = pd.DataFrame({'col2': [3, 4]})
        
        id1 = self.data_service.store_data(df1, {'filename': 'file1.csv'})
        id2 = self.data_service.store_data(df2, {'filename': 'file2.csv'})
        
        stored_ids = self.data_service.list_stored_data()
        
        assert len(stored_ids) == 2
        assert id1 in stored_ids
        assert id2 in stored_ids

    def test_list_stored_data_empty(self):
        """Test listing stored data when none exist."""
        stored_ids = self.data_service.list_stored_data()
        assert stored_ids == []

    def test_remove_data_existing(self):
        """Test removing existing data."""
        data_id = self.data_service.store_data(self.test_df, self.test_metadata)
        
        # Store some additional data
        self.data_service.store_column_mapping(data_id, {'time': 'time'})
        self.data_service.store_analysis_result(data_id, 'test', {'result': 'value'})
        
        # Remove the data
        success = self.data_service.remove_data(data_id)
        
        assert success is True
        assert data_id not in self.data_service._data_store
        assert data_id not in self.data_service._column_mappings
        assert data_id not in self.data_service._analysis_results

    def test_remove_data_nonexistent(self):
        """Test removing non-existent data."""
        success = self.data_service.remove_data('nonexistent_id')
        assert success is False

    def test_get_data_info_existing(self):
        """Test getting comprehensive data info."""
        data_id = self.data_service.store_data(self.test_df, self.test_metadata)
        
        info = self.data_service.get_data_info(data_id)
        
        assert info is not None
        assert 'data_id' in info
        assert 'shape' in info
        assert 'columns' in info
        assert 'metadata' in info
        assert 'timestamp' in info
        assert info['data_id'] == data_id
        assert info['shape'] == self.test_df.shape

    def test_get_data_info_nonexistent(self):
        """Test getting data info for non-existent data."""
        info = self.data_service.get_data_info('nonexistent_id')
        assert info is None

    def test_update_data_info(self):
        """Test updating data info."""
        data_id = self.data_service.store_data(self.test_df, self.test_metadata)
        
        new_info = {
            'sampling_freq': 2000,
            'time_unit': 's',
            'new_field': 'new_value'
        }
        
        self.data_service.update_data_info(data_id, new_info)
        
        # Check that new info was added to metadata
        stored_data = self.data_service._data_store[data_id]
        for key, value in new_info.items():
            assert stored_data['metadata'][key] == value

    def test_update_data_info_nonexistent(self):
        """Test updating data info for non-existent data."""
        new_info = {'field': 'value'}
        success = self.data_service.update_data_info('nonexistent_id', new_info)
        assert success is False

    def test_get_session_data(self):
        """Test getting all data for current session."""
        # Store data in current session
        data_id1 = self.data_service.store_data(self.test_df, self.test_metadata)
        
        # Create new service with different session
        other_service = DataService()
        other_df = pd.DataFrame({'col': [1, 2, 3]})
        other_id = other_service.store_data(other_df, {'filename': 'other.csv'})
        
        # Get session data from original service
        session_data = self.data_service.get_session_data()
        
        assert data_id1 in session_data
        assert other_id not in session_data

    def test_clear_session_data(self):
        """Test clearing all data for current session."""
        data_id1 = self.data_service.store_data(self.test_df, self.test_metadata)
        
        # Create data in another session
        other_service = DataService()
        other_df = pd.DataFrame({'col': [1, 2, 3]})
        other_id = other_service.store_data(other_df, {'filename': 'other.csv'})
        
        # Clear session data
        self.data_service.clear_session_data()
        
        # Check that only current session data was cleared
        assert data_id1 not in self.data_service._data_store
        assert other_id in other_service._data_store

    def test_export_data_csv(self):
        """Test exporting data to CSV."""
        data_id = self.data_service.store_data(self.test_df, self.test_metadata)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            success = self.data_service.export_data_csv(data_id, tmp_path)
            assert success is True
            
            # Check that file was created and contains data
            assert os.path.exists(tmp_path)
            exported_df = pd.read_csv(tmp_path)
            pd.testing.assert_frame_equal(exported_df, self.test_df)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_export_data_csv_nonexistent(self):
        """Test exporting non-existent data to CSV."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            success = self.data_service.export_data_csv('nonexistent_id', tmp_path)
            assert success is False
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_get_data_summary(self):
        """Test getting data summary statistics."""
        data_id = self.data_service.store_data(self.test_df, self.test_metadata)
        
        summary = self.data_service.get_data_summary(data_id)
        
        assert summary is not None
        assert 'rows' in summary
        assert 'columns' in summary
        assert 'dtypes' in summary
        assert 'memory_usage' in summary
        assert summary['rows'] == 100
        assert summary['columns'] == 4

    def test_get_data_summary_nonexistent(self):
        """Test getting summary for non-existent data."""
        summary = self.data_service.get_data_summary('nonexistent_id')
        assert summary is None

    def test_validate_data_id(self):
        """Test validating data ID format."""
        # Valid UUID format
        valid_id = "550e8400-e29b-41d4-a716-446655440000"
        assert self.data_service._validate_data_id(valid_id) is True
        
        # Invalid format
        invalid_id = "invalid-id"
        assert self.data_service._validate_data_id(invalid_id) is False

    def test_get_data_size(self):
        """Test getting data size in memory."""
        data_id = self.data_service.store_data(self.test_df, self.test_metadata)
        
        size = self.data_service.get_data_size(data_id)
        
        assert size > 0
        assert isinstance(size, int)

    def test_get_data_size_nonexistent(self):
        """Test getting size for non-existent data."""
        size = self.data_service.get_data_size('nonexistent_id')
        assert size == 0


class TestDataServiceSingleton:
    """Test the singleton pattern for DataService."""

    def test_get_data_service_singleton(self):
        """Test that get_data_service returns the same instance."""
        service1 = get_data_service()
        service2 = get_data_service()
        
        assert service1 is service2
        assert isinstance(service1, DataService)

    def test_singleton_persistence(self):
        """Test that singleton persists across calls."""
        service1 = get_data_service()
        
        # Store some data
        df = pd.DataFrame({'col': [1, 2, 3]})
        data_id = service1.store_data(df, {'filename': 'test.csv'})
        
        # Get service again
        service2 = get_data_service()
        
        # Check that data is still there
        retrieved_df = service2.get_data(data_id)
        assert retrieved_df is not None
        pd.testing.assert_frame_equal(retrieved_df, df)
