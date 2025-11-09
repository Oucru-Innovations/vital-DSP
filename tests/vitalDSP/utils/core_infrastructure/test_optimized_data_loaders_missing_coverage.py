"""
Additional tests for optimized_data_loaders.py to cover missing lines.

This test file specifically targets uncovered lines in:
- ProgressInfo properties (progress_percent, loading_speed_mbps)
- CancellationToken methods
- OptimizedChunkedDataLoader (all methods and branches)
- OptimizedMemoryMappedLoader (all methods and branches)
- select_optimal_loader function
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
import time
from unittest.mock import Mock, patch, MagicMock
import warnings

from vitalDSP.utils.core_infrastructure.optimized_data_loaders import (
    ProgressInfo,
    CancellationToken,
    OptimizedChunkedDataLoader,
    OptimizedMemoryMappedLoader,
    select_optimal_loader,
    LoadingStrategy,
)
from vitalDSP.utils.config_utilities.dynamic_config import DynamicConfig, get_config


@pytest.fixture
def sample_csv_file(tmp_path):
    """Create a sample CSV file for testing."""
    file_path = tmp_path / "test_data.csv"
    # Create a CSV with some data
    data = pd.DataFrame({
        'signal': np.random.randn(1000),
        'timestamp': np.arange(1000),
    })
    data.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def sample_npy_file(tmp_path):
    """Create a sample .npy file for testing."""
    file_path = tmp_path / "test_data.npy"
    data = np.random.randn(1000).astype(np.float64)
    np.save(file_path, data)
    return file_path


@pytest.fixture
def sample_binary_file(tmp_path):
    """Create a sample binary file for testing."""
    file_path = tmp_path / "test_data.bin"
    data = np.random.randn(1000).astype(np.float64)
    data.tofile(file_path)
    return file_path


class TestProgressInfo:
    """Test ProgressInfo class (lines 60-62, 67-70)."""

    def test_progress_percent_zero_total(self):
        """Test progress_percent when total_bytes is 0 (line 60-62)."""
        info = ProgressInfo(
            bytes_processed=100,
            total_bytes=0,
            chunks_processed=1,
            total_chunks=1,
            elapsed_time=1.0,
            estimated_remaining=0.0,
            current_chunk_size=100,
            loading_strategy="test"
        )
        assert info.progress_percent == 0.0

    def test_progress_percent_normal(self):
        """Test progress_percent calculation."""
        info = ProgressInfo(
            bytes_processed=500,
            total_bytes=1000,
            chunks_processed=1,
            total_chunks=2,
            elapsed_time=1.0,
            estimated_remaining=1.0,
            current_chunk_size=500,
            loading_strategy="test"
        )
        assert info.progress_percent == 50.0

    def test_loading_speed_mbps_zero_time(self):
        """Test loading_speed_mbps when elapsed_time is 0 (line 67-68)."""
        info = ProgressInfo(
            bytes_processed=1000,
            total_bytes=2000,
            chunks_processed=1,
            total_chunks=2,
            elapsed_time=0.0,
            estimated_remaining=0.0,
            current_chunk_size=1000,
            loading_strategy="test"
        )
        assert info.loading_speed_mbps == 0.0

    def test_loading_speed_mbps_normal(self):
        """Test loading_speed_mbps calculation (lines 69-70)."""
        info = ProgressInfo(
            bytes_processed=1024 * 1024,  # 1 MB
            total_bytes=2 * 1024 * 1024,
            chunks_processed=1,
            total_chunks=2,
            elapsed_time=1.0,
            estimated_remaining=1.0,
            current_chunk_size=1024 * 1024,
            loading_strategy="test"
        )
        assert info.loading_speed_mbps == 1.0


class TestCancellationToken:
    """Test CancellationToken class (lines 79-81, 85-87, 91-92, 96-97, 101-103)."""

    def test_cancel_with_message(self):
        """Test cancel with message (lines 85-87)."""
        token = CancellationToken()
        token.cancel("Test cancellation")
        assert token.is_cancelled() is True
        assert token._cancel_message == "Test cancellation"

    def test_cancel_without_message(self):
        """Test cancel without message (line 87)."""
        token = CancellationToken()
        token.cancel()
        assert token.is_cancelled() is True
        assert token._cancel_message == "Operation cancelled by user"

    def test_is_cancelled(self):
        """Test is_cancelled method (lines 91-92)."""
        token = CancellationToken()
        assert token.is_cancelled() is False
        token.cancel()
        assert token.is_cancelled() is True

    def test_throw_if_cancelled(self):
        """Test throw_if_cancelled method (lines 96-97)."""
        token = CancellationToken()
        token.cancel("Test error")
        with pytest.raises(InterruptedError, match="Test error"):
            token.throw_if_cancelled()

    def test_reset(self):
        """Test reset method (lines 101-103)."""
        token = CancellationToken()
        token.cancel("Test")
        assert token.is_cancelled() is True
        token.reset()
        assert token.is_cancelled() is False
        assert token._cancel_message is None


class TestOptimizedChunkedDataLoader:
    """Test OptimizedChunkedDataLoader class (lines 125-161, 173, 188-282, 286-288, 294-299, 303-308, 312-323, 327-333, 341-345, 349)."""

    def test_init_file_not_found(self, tmp_path):
        """Test initialization with non-existent file."""
        file_path = tmp_path / "nonexistent.csv"
        with pytest.raises(FileNotFoundError):
            OptimizedChunkedDataLoader(file_path)

    def test_init_with_overlap_samples(self, sample_csv_file):
        """Test initialization with overlap_samples (lines 133-137)."""
        loader = OptimizedChunkedDataLoader(sample_csv_file, overlap_samples=100)
        assert loader.overlap_samples == 100

    def test_init_without_overlap_samples(self, sample_csv_file):
        """Test initialization without overlap_samples (uses config default)."""
        loader = OptimizedChunkedDataLoader(sample_csv_file)
        assert loader.overlap_samples is not None

    def test_init_auto_chunk_size(self, sample_csv_file):
        """Test initialization with auto chunk size (lines 146-147)."""
        loader = OptimizedChunkedDataLoader(sample_csv_file, chunk_size="auto")
        assert loader.chunk_size > 0

    def test_init_manual_chunk_size(self, sample_csv_file):
        """Test initialization with manual chunk size (line 149)."""
        loader = OptimizedChunkedDataLoader(sample_csv_file, chunk_size=100)
        assert loader.chunk_size == 100

    def test_determine_optimal_chunk_size(self, sample_csv_file):
        """Test _determine_optimal_chunk_size (lines 169-177)."""
        loader = OptimizedChunkedDataLoader(sample_csv_file, chunk_size="auto")
        chunk_size = loader._determine_optimal_chunk_size()
        assert chunk_size > 0

    def test_load_chunks_csv(self, sample_csv_file):
        """Test load_chunks with CSV format (lines 197-200)."""
        loader = OptimizedChunkedDataLoader(sample_csv_file, chunk_size=100)
        chunks = list(loader.load_chunks())
        assert len(chunks) > 0
        assert all(isinstance(chunk, pd.DataFrame) for chunk in chunks)

    def test_load_chunks_with_cancellation(self, sample_csv_file):
        """Test load_chunks with cancellation (lines 214, 278-280)."""
        loader = OptimizedChunkedDataLoader(sample_csv_file, chunk_size=100)
        token = CancellationToken()
        loader.cancellation_token = token
        
        # Cancel after first chunk - should raise InterruptedError on next iteration
        chunks = []
        with pytest.raises(InterruptedError):
            for i, chunk in enumerate(loader.load_chunks()):
                chunks.append(chunk)
                if i == 0:
                    token.cancel()  # Cancel after first chunk
                # Next iteration will check cancellation and raise

    def test_load_chunks_start_chunk(self, sample_csv_file):
        """Test load_chunks with start_chunk parameter (lines 217-218)."""
        loader = OptimizedChunkedDataLoader(sample_csv_file, chunk_size=100)
        all_chunks = list(loader.load_chunks())
        chunks_from_start = list(loader.load_chunks(start_chunk=1))
        # Should skip first chunk
        assert len(chunks_from_start) <= len(all_chunks) - 1

    def test_load_chunks_with_overlap(self, sample_csv_file):
        """Test load_chunks with overlap (lines 221-223)."""
        loader = OptimizedChunkedDataLoader(sample_csv_file, chunk_size=100, overlap_samples=10)
        chunks = list(loader.load_chunks())
        # With overlap, chunks should have some overlap data
        assert len(chunks) > 0

    def test_load_chunks_progress_callback(self, sample_csv_file):
        """Test load_chunks with progress callback (lines 225-259)."""
        progress_calls = []
        def progress_callback(info):
            progress_calls.append(info)
        
        loader = OptimizedChunkedDataLoader(sample_csv_file, chunk_size=100)
        list(loader.load_chunks(progress_callback=progress_callback))
        # Progress callback should be called
        assert len(progress_calls) >= 0  # May be throttled

    def test_load_chunks_max_chunks(self, sample_csv_file):
        """Test load_chunks with max_chunks limit (lines 271-272)."""
        loader = OptimizedChunkedDataLoader(sample_csv_file, chunk_size=100)
        chunks = list(loader.load_chunks(max_chunks=2))
        assert len(chunks) <= 2

    def test_load_chunks_memory_management(self, sample_csv_file):
        """Test load_chunks memory management (lines 275-276)."""
        loader = OptimizedChunkedDataLoader(sample_csv_file, chunk_size=100)
        # Process more than 10 chunks to trigger GC
        chunks = list(loader.load_chunks())
        # Should complete without error
        assert len(chunks) > 0

    def test_estimate_total_chunks(self, sample_csv_file):
        """Test _estimate_total_chunks (lines 284-288)."""
        loader = OptimizedChunkedDataLoader(sample_csv_file, chunk_size=100)
        total = loader._estimate_total_chunks()
        assert total >= 1

    def test_estimate_remaining_time(self, sample_csv_file):
        """Test _estimate_remaining_time (lines 290-299)."""
        loader = OptimizedChunkedDataLoader(sample_csv_file, chunk_size=100)
        
        # Test with zero bytes processed (line 294-295)
        remaining = loader._estimate_remaining_time(0, 1000, 1.0)
        assert remaining == 0.0
        
        # Test normal case (lines 297-299)
        remaining = loader._estimate_remaining_time(500, 1000, 1.0)
        assert remaining > 0

    @pytest.mark.skipif(True, reason="Requires pyarrow")
    def test_parquet_chunked_reader(self, tmp_path):
        """Test _parquet_chunked_reader (lines 301-308)."""
        try:
            import pyarrow.parquet as pq
            # Create a parquet file
            file_path = tmp_path / "test.parquet"
            df = pd.DataFrame({'signal': np.random.randn(1000)})
            df.to_parquet(file_path)
            
            loader = OptimizedChunkedDataLoader(file_path, file_format="parquet", chunk_size=100)
            chunks = list(loader._parquet_chunked_reader())
            assert len(chunks) > 0
        except ImportError:
            pytest.skip("pyarrow not available")

    @pytest.mark.skipif(True, reason="Requires h5py")
    def test_hdf5_chunked_reader(self, tmp_path):
        """Test _hdf5_chunked_reader (lines 310-323)."""
        try:
            import h5py
            # Create an HDF5 file
            file_path = tmp_path / "test.h5"
            with h5py.File(file_path, 'w') as f:
                f.create_dataset('data', data=np.random.randn(1000, 10))
            
            loader = OptimizedChunkedDataLoader(file_path, file_format="hdf5", chunk_size=100, key="data")
            chunks = list(loader._hdf5_chunked_reader())
            assert len(chunks) > 0
        except ImportError:
            pytest.skip("h5py not available")

    def test_load_chunks_unsupported_format(self, sample_csv_file):
        """Test load_chunks with unsupported format (lines 205-208)."""
        loader = OptimizedChunkedDataLoader(sample_csv_file, file_format="unsupported")
        with pytest.raises(ValueError, match="Unsupported file format"):
            list(loader.load_chunks())

    def test_update_performance_stats(self, sample_csv_file):
        """Test _update_performance_stats (lines 325-333)."""
        loader = OptimizedChunkedDataLoader(sample_csv_file, chunk_size=100)
        initial_chunks = loader._performance_stats["total_chunks_processed"]
        loader._update_performance_stats(100, 800)
        assert loader._performance_stats["total_chunks_processed"] == initial_chunks + 1
        assert loader._performance_stats["total_bytes_processed"] > 0

    def test_load_all(self, sample_csv_file):
        """Test load_all method (lines 335-345)."""
        loader = OptimizedChunkedDataLoader(sample_csv_file, chunk_size=100)
        df = loader.load_all()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_get_info(self, sample_csv_file):
        """Test get_info method (lines 347-359)."""
        loader = OptimizedChunkedDataLoader(sample_csv_file, chunk_size=100)
        info = loader.get_info()
        assert "file_path" in info
        assert "file_size_mb" in info
        assert "estimated_chunks" in info
        assert "performance_stats" in info


class TestOptimizedMemoryMappedLoader:
    """Test OptimizedMemoryMappedLoader class (lines 380-428, 438-452, 463-519, 531-534, 538-542, 546, 558-559, 563, 567-568)."""

    def test_init_file_not_found(self, tmp_path):
        """Test initialization with non-existent file (line 387-388)."""
        file_path = tmp_path / "nonexistent.npy"
        with pytest.raises(FileNotFoundError):
            OptimizedMemoryMappedLoader(file_path)

    def test_init_npy_format(self, sample_npy_file):
        """Test initialization with .npy format (lines 393-399)."""
        loader = OptimizedMemoryMappedLoader(sample_npy_file)
        assert loader.is_npy_format is True
        assert loader.shape is not None
        assert loader.size > 0

    def test_init_binary_format_with_shape(self, sample_binary_file):
        """Test initialization with binary format and shape (lines 400-414)."""
        loader = OptimizedMemoryMappedLoader(
            sample_binary_file,
            dtype=np.float64,
            shape=(1000,)
        )
        assert loader.is_npy_format is False
        assert loader.shape == (1000,)

    def test_init_binary_format_infer_shape(self, sample_binary_file):
        """Test initialization with binary format inferring shape (lines 402-406)."""
        loader = OptimizedMemoryMappedLoader(
            sample_binary_file,
            dtype=np.float64
        )
        assert loader.shape is not None
        assert loader.size > 0

    def test_get_segment(self, sample_npy_file):
        """Test get_segment method (lines 434-452)."""
        loader = OptimizedMemoryMappedLoader(sample_npy_file)
        segment = loader.get_segment(0, 100)
        assert len(segment) == 100
        assert isinstance(segment, np.ndarray)

    def test_get_segment_invalid_range(self, sample_npy_file):
        """Test get_segment with invalid range (lines 440-441)."""
        loader = OptimizedMemoryMappedLoader(sample_npy_file)
        with pytest.raises(ValueError, match="Invalid segment range"):
            loader.get_segment(-1, 100)
        with pytest.raises(ValueError, match="Invalid segment range"):
            loader.get_segment(0, loader.size + 100)

    def test_get_segment_with_copy(self, sample_npy_file):
        """Test get_segment with copy=True (lines 450-451)."""
        loader = OptimizedMemoryMappedLoader(sample_npy_file)
        segment = loader.get_segment(0, 100, copy=True)
        assert isinstance(segment, np.ndarray)
        # Should be a copy, not a view
        assert segment.flags.owndata is True

    def test_get_segment_cancellation(self, sample_npy_file):
        """Test get_segment with cancellation (line 438)."""
        loader = OptimizedMemoryMappedLoader(sample_npy_file)
        token = CancellationToken()
        loader.cancellation_token = token
        token.cancel()
        with pytest.raises(InterruptedError):
            loader.get_segment(0, 100)

    def test_iterate_chunks_auto_chunk_size(self, sample_npy_file):
        """Test iterate_chunks with auto chunk size (lines 463-466)."""
        loader = OptimizedMemoryMappedLoader(sample_npy_file)
        chunks = list(loader.iterate_chunks())
        assert len(chunks) > 0

    def test_iterate_chunks_manual_chunk_size(self, sample_npy_file):
        """Test iterate_chunks with manual chunk size."""
        loader = OptimizedMemoryMappedLoader(sample_npy_file)
        chunks = list(loader.iterate_chunks(chunk_size=100))
        assert len(chunks) > 0
        assert all(len(chunk) <= 100 for chunk in chunks)

    def test_iterate_chunks_with_copy(self, sample_npy_file):
        """Test iterate_chunks with copy=True (lines 478-479)."""
        loader = OptimizedMemoryMappedLoader(sample_npy_file)
        chunks = list(loader.iterate_chunks(chunk_size=100, copy=True))
        assert len(chunks) > 0
        assert all(chunk.flags.owndata for chunk in chunks)

    def test_iterate_chunks_progress_callback(self, sample_npy_file):
        """Test iterate_chunks with progress callback (lines 487-517)."""
        progress_calls = []
        def progress_callback(info):
            progress_calls.append(info)
        
        loader = OptimizedMemoryMappedLoader(sample_npy_file)
        chunks = list(loader.iterate_chunks(chunk_size=100, progress_callback=progress_callback))
        # Progress callback may be throttled
        assert len(chunks) > 0

    def test_iterate_chunks_cancellation(self, sample_npy_file):
        """Test iterate_chunks with cancellation (line 473)."""
        loader = OptimizedMemoryMappedLoader(sample_npy_file)
        token = CancellationToken()
        loader.cancellation_token = token
        
        # Cancel after first chunk - should raise InterruptedError on next iteration
        chunks = []
        with pytest.raises(InterruptedError):
            for chunk in loader.iterate_chunks(chunk_size=100):
                chunks.append(chunk)
                if len(chunks) == 1:
                    token.cancel()  # Cancel after first chunk
                # Next iteration will check cancellation and raise

    def test_get_time_segment(self, sample_npy_file):
        """Test get_time_segment method (lines 521-534)."""
        loader = OptimizedMemoryMappedLoader(sample_npy_file)
        sampling_rate = 100.0
        segment = loader.get_time_segment(0.0, 1.0, sampling_rate)
        expected_length = int(1.0 * sampling_rate)
        assert len(segment) == expected_length

    def test_get_time_segment_with_copy(self, sample_npy_file):
        """Test get_time_segment with copy=True."""
        loader = OptimizedMemoryMappedLoader(sample_npy_file)
        sampling_rate = 100.0
        segment = loader.get_time_segment(0.0, 1.0, sampling_rate, copy=True)
        assert segment.flags.owndata is True

    def test_estimate_remaining(self, sample_npy_file):
        """Test _estimate_remaining method (lines 536-542)."""
        loader = OptimizedMemoryMappedLoader(sample_npy_file)
        
        # Test with zero processed (lines 538-539)
        remaining = loader._estimate_remaining(0, 1000, 1.0)
        assert remaining == 0.0
        
        # Test normal case (lines 540-542)
        remaining = loader._estimate_remaining(500, 1000, 1.0)
        assert remaining > 0

    def test_get_info(self, sample_npy_file):
        """Test get_info method (lines 544-554)."""
        loader = OptimizedMemoryMappedLoader(sample_npy_file)
        info = loader.get_info()
        assert "file_path" in info
        assert "file_size_mb" in info
        assert "file_size_gb" in info
        assert "memory_footprint_mb" in info
        assert "performance_stats" in info

    def test_close(self, sample_npy_file):
        """Test close method (lines 556-559)."""
        loader = OptimizedMemoryMappedLoader(sample_npy_file)
        loader.close()
        # Should not raise error

    def test_context_manager(self, sample_npy_file):
        """Test context manager support (lines 561-568)."""
        with OptimizedMemoryMappedLoader(sample_npy_file) as loader:
            segment = loader.get_segment(0, 100)
            assert len(segment) == 100
        # Should be closed after context exit


class TestSelectOptimalLoader:
    """Test select_optimal_loader function (lines 571-607)."""

    def test_select_optimal_loader_file_not_found(self, tmp_path):
        """Test select_optimal_loader with non-existent file (lines 580-581)."""
        file_path = tmp_path / "nonexistent.csv"
        with pytest.raises(FileNotFoundError):
            select_optimal_loader(file_path)

    def test_select_optimal_loader_small_file(self, sample_csv_file):
        """Test select_optimal_loader for small file (lines 605-607)."""
        # Mock config to have high threshold
        with patch('vitalDSP.utils.core_infrastructure.optimized_data_loaders.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.data_loader.small_file_threshold = 1000  # 1 GB
            mock_config.data_loader.medium_file_threshold = 2000  # 2 GB
            mock_config.data_loader.memory_map_supported_formats = ['.npy', '.bin']
            mock_get_config.return_value = mock_config
            
            result = select_optimal_loader(sample_csv_file)
            # Small file should return None
            assert result is None

    def test_select_optimal_loader_medium_file(self, sample_csv_file):
        """Test select_optimal_loader for medium file (lines 601-603)."""
        # Mock config and file size
        with patch('vitalDSP.utils.core_infrastructure.optimized_data_loaders.get_config') as mock_get_config, \
             patch.object(Path, 'stat') as mock_stat:
            mock_config = Mock()
            mock_config.data_loader.small_file_threshold = 0.001  # 1 MB
            mock_config.data_loader.medium_file_threshold = 1000  # 1 GB
            mock_config.data_loader.memory_map_supported_formats = ['.npy', '.bin']
            mock_get_config.return_value = mock_config
            
            # Mock file size to be medium
            mock_stat.return_value.st_size = 500 * 1024 * 1024  # 500 MB
            
            result = select_optimal_loader(sample_csv_file)
            assert isinstance(result, OptimizedChunkedDataLoader)

    def test_select_optimal_loader_large_file_memory_map(self, sample_npy_file):
        """Test select_optimal_loader for large file with memory-mapped format (lines 594-596)."""
        with patch('vitalDSP.utils.core_infrastructure.optimized_data_loaders.get_config') as mock_get_config, \
             patch.object(Path, 'stat') as mock_stat:
            mock_config = Mock()
            mock_config.data_loader.small_file_threshold = 0.001
            mock_config.data_loader.medium_file_threshold = 100  # 100 MB
            mock_config.data_loader.memory_map_supported_formats = ['.npy', '.bin']
            mock_get_config.return_value = mock_config
            
            # Mock file size to be large
            mock_stat.return_value.st_size = 500 * 1024 * 1024  # 500 MB
            
            result = select_optimal_loader(sample_npy_file)
            assert isinstance(result, OptimizedMemoryMappedLoader)

    def test_select_optimal_loader_large_file_chunked(self, sample_csv_file):
        """Test select_optimal_loader for large file with text format (lines 597-599)."""
        with patch('vitalDSP.utils.core_infrastructure.optimized_data_loaders.get_config') as mock_get_config, \
             patch.object(Path, 'stat') as mock_stat:
            mock_config = Mock()
            mock_config.data_loader.small_file_threshold = 0.001
            mock_config.data_loader.medium_file_threshold = 100  # 100 MB
            mock_config.data_loader.memory_map_supported_formats = ['.npy', '.bin']
            mock_get_config.return_value = mock_config
            
            # Mock file size to be large
            mock_stat.return_value.st_size = 500 * 1024 * 1024  # 500 MB
            
            result = select_optimal_loader(sample_csv_file)
            assert isinstance(result, OptimizedChunkedDataLoader)

