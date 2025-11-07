"""Basic tests for neural_network_filtering.py module."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

# Test data
SAMPLE_DATA = np.sin(2 * np.pi * np.linspace(0, 10, 1000)) + 0.1 * np.random.randn(1000)
SAMPLE_FREQ = 100

try:
    from vitalDSP.advanced_computation.neural_network_filtering import NeuralNetworkFiltering
    AVAILABLE = True
except ImportError:
    AVAILABLE = False

@pytest.mark.skipif(not AVAILABLE, reason="Module not available")
class TestNeuralNetworkFiltering:
    def test_init(self):
        """Test basic initialization"""
        nnf = NeuralNetworkFiltering(SAMPLE_DATA)
        assert nnf is not None
        assert len(nnf.signal) == len(SAMPLE_DATA)
        assert nnf.network_type == "feedforward"  # Default network type
        assert nnf.epochs == 100  # Default epochs
    
    def test_init_with_parameters(self):
        """Test initialization with custom parameters"""
        nnf = NeuralNetworkFiltering(
            SAMPLE_DATA,
            network_type="feedforward",
            hidden_layers=[32, 32],
            learning_rate=0.01,
            epochs=50,
            batch_size=16,
            dropout_rate=0.3,
            batch_norm=False
        )
        assert nnf is not None
        assert nnf.network_type == "feedforward"
        assert nnf.hidden_layers == [32, 32]
        assert nnf.learning_rate == 0.01
        assert nnf.epochs == 50
        assert nnf.batch_size == 16
        assert nnf.dropout_rate == 0.3
        assert nnf.batch_norm == False
    
    def test_lstm_filter(self):
        """Test LSTM-based filtering using recurrent network type"""
        # Use shorter epochs and smaller data for faster testing
        test_data = SAMPLE_DATA[:100]  # Use smaller dataset
        nnf = NeuralNetworkFiltering(test_data, network_type="recurrent", recurrent_type="lstm", epochs=1)
        # Train the network first
        nnf.train()
        # Apply the filter
        filtered = nnf.apply_filter()
        assert isinstance(filtered, np.ndarray)
        assert len(filtered) > 0
    
    def test_cnn_filter(self):
        """Test CNN-based filtering using convolutional network type"""
        # Use shorter epochs for faster testing
        nnf = NeuralNetworkFiltering(SAMPLE_DATA, network_type="convolutional", epochs=1)
        # Train the network first
        nnf.train()
        # Apply the filter
        filtered = nnf.apply_filter()
        assert isinstance(filtered, np.ndarray)
        assert len(filtered) > 0
    
    def test_autoencoder_filter(self):
        """Test feedforward network filtering (closest to autoencoder)"""
        # Use shorter epochs for faster testing
        nnf = NeuralNetworkFiltering(SAMPLE_DATA, network_type="feedforward", epochs=1)
        # Train the network first
        nnf.train()
        # Apply the filter
        filtered = nnf.apply_filter()
        assert isinstance(filtered, np.ndarray)
        assert len(filtered) > 0
        assert filtered.shape == (len(SAMPLE_DATA) - 1, 1)  # Expected output shape
    
    def test_evaluate_method(self):
        """Test the evaluate method for performance assessment"""
        nnf = NeuralNetworkFiltering(SAMPLE_DATA, network_type="feedforward", epochs=1)
        nnf.train()
        # Create a test signal
        test_signal = SAMPLE_DATA + 0.05 * np.random.randn(len(SAMPLE_DATA))
        mse = nnf.evaluate(test_signal)
        assert isinstance(mse, (float, np.floating))
        assert mse >= 0  # MSE should be non-negative

@pytest.mark.skipif(not AVAILABLE, reason="Module not available")
class TestErrorHandling:
    def test_init_empty_data(self):
        """Test initialization with empty data"""
        nnf = NeuralNetworkFiltering(np.array([]))
        assert nnf is not None
        assert len(nnf.signal) == 0
    
    def test_init_single_point(self):
        """Test initialization with single data point"""
        nnf = NeuralNetworkFiltering(np.array([1.0]))
        assert nnf is not None
        assert len(nnf.signal) == 1
    
    def test_init_with_different_network_types(self):
        """Test initialization with different network types"""
        # Feedforward (should work)
        nnf1 = NeuralNetworkFiltering(SAMPLE_DATA, network_type="feedforward")
        assert nnf1.network_type == "feedforward"
        
        # Convolutional (should work)
        nnf2 = NeuralNetworkFiltering(SAMPLE_DATA, network_type="convolutional")
        assert nnf2.network_type == "convolutional"
        
        # Recurrent (should work)
        nnf3 = NeuralNetworkFiltering(SAMPLE_DATA, network_type="recurrent")
        assert nnf3.network_type == "recurrent"
    
    def test_invalid_network_type(self):
        """Test that invalid network type raises ValueError"""
        with pytest.raises(ValueError, match="Unknown network type"):
            NeuralNetworkFiltering(SAMPLE_DATA, network_type="invalid_type")
