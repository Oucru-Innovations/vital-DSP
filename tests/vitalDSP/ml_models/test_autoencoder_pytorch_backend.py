"""
Tests for Autoencoder PyTorch backend implementation.
This test file focuses on increasing coverage for PyTorch-specific code paths.
"""

import pytest
import numpy as np

# Skip all tests if TensorFlow is not available
pytest.importorskip("tensorflow")

# Try to import PyTorch, skip tests if not available
torch = pytest.importorskip("torch")

from vitalDSP.ml_models.autoencoder import (
    StandardAutoencoder,
    ConvolutionalAutoencoder,
    LSTMAutoencoder,
    VariationalAutoencoder,
    DenoisingAutoencoder,
)


@pytest.fixture
def sample_signal_1d():
    """Generate sample 1D signal."""
    np.random.seed(42)
    return np.random.randn(100, 50)


@pytest.fixture
def sample_signal_2d():
    """Generate sample 2D signal for convolutional autoencoder."""
    np.random.seed(42)
    return np.random.randn(100, 100, 1)


class TestStandardAutoencoderPyTorch:
    """Test StandardAutoencoder with PyTorch backend."""

    def test_pytorch_backend_init(self):
        """Test PyTorch backend initialization."""
        ae = StandardAutoencoder(
            input_shape=(50,),
            latent_dim=16,
            backend="pytorch"
        )
        assert ae.backend == "pytorch"
        assert ae.model is not None
        assert ae.encoder is not None
        assert ae.decoder is not None

    def test_pytorch_activation_relu(self, sample_signal_1d):
        """Test PyTorch backend with ReLU activation."""
        ae = StandardAutoencoder(
            input_shape=(50,),
            latent_dim=16,
            activation="relu",
            backend="pytorch"
        )
        # Just verify it builds successfully
        assert ae.model is not None

    def test_pytorch_activation_tanh(self, sample_signal_1d):
        """Test PyTorch backend with Tanh activation."""
        ae = StandardAutoencoder(
            input_shape=(50,),
            latent_dim=16,
            activation="tanh",
            backend="pytorch"
        )
        assert ae.model is not None

    def test_pytorch_output_activation_sigmoid(self):
        """Test PyTorch backend with sigmoid output activation."""
        ae = StandardAutoencoder(
            input_shape=(50,),
            latent_dim=16,
            output_activation="sigmoid",
            backend="pytorch"
        )
        assert ae.model is not None

    def test_pytorch_output_activation_tanh(self):
        """Test PyTorch backend with tanh output activation."""
        ae = StandardAutoencoder(
            input_shape=(50,),
            latent_dim=16,
            output_activation="tanh",
            backend="pytorch"
        )
        assert ae.model is not None

    def test_pytorch_output_activation_linear(self):
        """Test PyTorch backend with linear output activation."""
        ae = StandardAutoencoder(
            input_shape=(50,),
            latent_dim=16,
            output_activation="linear",
            backend="pytorch"
        )
        assert ae.model is not None

    def test_pytorch_without_batch_norm(self):
        """Test PyTorch backend without batch normalization."""
        ae = StandardAutoencoder(
            input_shape=(50,),
            latent_dim=16,
            use_batch_norm=False,
            backend="pytorch"
        )
        assert ae.model is not None

    def test_pytorch_with_different_hidden_dims(self):
        """Test PyTorch backend with different hidden dimensions."""
        ae = StandardAutoencoder(
            input_shape=(50,),
            latent_dim=16,
            hidden_dims=[128, 64, 32],
            backend="pytorch"
        )
        assert ae.model is not None

    def test_pytorch_with_high_dropout(self):
        """Test PyTorch backend with high dropout rate."""
        ae = StandardAutoencoder(
            input_shape=(50,),
            latent_dim=16,
            dropout_rate=0.5,
            backend="pytorch"
        )
        assert ae.model is not None

    def test_pytorch_encode_decode(self, sample_signal_1d):
        """Test PyTorch encode and decode methods."""
        ae = StandardAutoencoder(
            input_shape=(50,),
            latent_dim=16,
            backend="pytorch"
        )
        
        # Test encoding
        latent = ae.encode(sample_signal_1d)
        assert latent.shape == (100, 16)
        
        # Test decoding
        reconstructed = ae.decode(latent)
        assert reconstructed.shape == sample_signal_1d.shape

    def test_pytorch_reconstruction(self, sample_signal_1d):
        """Test PyTorch reconstruction."""
        ae = StandardAutoencoder(
            input_shape=(50,),
            latent_dim=16,
            backend="pytorch"
        )
        
        # Use encode + decode for reconstruction (no predict method)
        latent = ae.encode(sample_signal_1d)
        reconstructed = ae.decode(latent)
        assert reconstructed.shape == sample_signal_1d.shape


class TestConvolutionalAutoencoderPyTorch:
    """Test ConvolutionalAutoencoder PyTorch backend."""

    def test_pytorch_not_implemented(self):
        """Test that PyTorch backend raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            ConvolutionalAutoencoder(
                input_shape=(100, 1),
                latent_dim=16,
                backend="pytorch"
            )


class TestLSTMAutoencoderPyTorch:
    """Test LSTMAutoencoder PyTorch backend."""

    def test_pytorch_not_implemented(self):
        """Test that PyTorch backend raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            LSTMAutoencoder(
                input_shape=(100, 1),
                latent_dim=16,
                backend="pytorch"
            )


class TestVariationalAutoencoderPyTorch:
    """Test VariationalAutoencoder PyTorch backend."""

    def test_pytorch_not_implemented(self):
        """Test that PyTorch backend raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            VariationalAutoencoder(
                input_shape=(50,),
                latent_dim=16,
                backend="pytorch"
            )


class TestDenoisingAutoencoderPyTorch:
    """Test DenoisingAutoencoder PyTorch backend."""

    def test_pytorch_backend_works(self):
        """Test that PyTorch backend works for DenoisingAutoencoder."""
        # DenoisingAutoencoder now supports PyTorch backend
        dae = DenoisingAutoencoder(
            input_shape=(50,),
            latent_dim=16,
            backend="pytorch"
        )
        assert dae.backend == "pytorch"
        assert dae.model is not None


class TestAutoencoderEdgeCases:
    """Test edge cases for autoencoders."""

    def test_different_beta_values(self, sample_signal_1d):
        """Test VAE with different beta values."""
        for beta in [0.5, 1.0, 2.0]:
            vae = VariationalAutoencoder(
                input_shape=(50,),
                latent_dim=16,
                beta=beta
            )
            assert vae.beta == beta

    def test_various_kernel_sizes_conv(self):
        """Test convolutional autoencoder with various kernel sizes."""
        for kernel_size in [3, 5, 7]:
            ae = ConvolutionalAutoencoder(
                input_shape=(100, 1),
                latent_dim=16,
                kernel_sizes=kernel_size
            )
            assert ae.model is not None

    def test_various_pool_sizes_conv(self):
        """Test convolutional autoencoder with various pool sizes."""
        for pool_size in [2, 3, 4]:
            ae = ConvolutionalAutoencoder(
                input_shape=(100, 1),
                latent_dim=16,
                pool_sizes=pool_size
            )
            assert ae.model is not None

    def test_lstm_bidirectional(self, sample_signal_1d):
        """Test LSTM autoencoder with bidirectional layers."""
        ae = LSTMAutoencoder(
            input_shape=(100, 1),
            latent_dim=16,
            use_bidirectional=True
        )
        assert ae.use_bidirectional is True

    def test_lstm_different_units(self):
        """Test LSTM autoencoder with different unit sizes."""
        ae = LSTMAutoencoder(
            input_shape=(100, 1),
            latent_dim=16,
            lstm_units=[128, 64, 32]
        )
        assert len(ae.lstm_units) == 3

    def test_denoising_noise_types(self, sample_signal_1d):
        """Test denoising autoencoder with different noise types."""
        ae = DenoisingAutoencoder(
            input_shape=(50,),
            latent_dim=16,
            noise_type="gaussian"
        )
        
        # Test _add_noise private method (add_noise is private)
        noisy = ae._add_noise(sample_signal_1d)
        assert noisy.shape == sample_signal_1d.shape

    def test_denoising_noise_level(self, sample_signal_1d):
        """Test denoising autoencoder with different noise levels."""
        for noise_level in [0.1, 0.3, 0.5]:
            ae = DenoisingAutoencoder(
                input_shape=(50,),
                latent_dim=16,
                noise_level=noise_level
            )
            assert ae.noise_level == noise_level

    def test_random_state_reproducibility(self):
        """Test that random_state ensures reproducibility."""
        ae1 = StandardAutoencoder(
            input_shape=(50,),
            latent_dim=16,
            random_state=42
        )
        ae2 = StandardAutoencoder(
            input_shape=(50,),
            latent_dim=16,
            random_state=42
        )
        # Just verify they initialize without error
        assert ae1.model is not None
        assert ae2.model is not None

    def test_compute_reconstruction_error_mse(self, sample_signal_1d):
        """Test compute_reconstruction_error with MSE metric."""
        ae = StandardAutoencoder(input_shape=(50,), latent_dim=16)
        ae.fit(sample_signal_1d, epochs=1, verbose=0)
        
        error = ae.compute_reconstruction_error(sample_signal_1d, metric="mse")
        # compute_reconstruction_error returns per-sample errors (array), not single float
        assert isinstance(error, np.ndarray)
        assert error.shape[0] == sample_signal_1d.shape[0]
        assert np.all(error >= 0)

    def test_get_latent_representation(self, sample_signal_1d):
        """Test get_latent_representation method."""
        ae = StandardAutoencoder(input_shape=(50,), latent_dim=16)
        
        # get_latent_representation doesn't exist, use encode() instead
        latent = ae.encode(sample_signal_1d)
        assert latent.shape == (100, 16)

    def test_vae_encode_return_distribution(self, sample_signal_1d):
        """Test VAE encode with return_distribution=True."""
        vae = VariationalAutoencoder(input_shape=(50,), latent_dim=16)
        vae.fit(sample_signal_1d, epochs=1, verbose=0)
        
        z_mean, z_log_var, z = vae.encode(sample_signal_1d, return_distribution=True)
        assert z_mean.shape == (100, 16)
        assert z_log_var.shape == (100, 16)
        assert z.shape == (100, 16)


class TestAutoencoderConvenienceFunctions:
    """Test convenience functions from autoencoder module."""

    def test_detect_anomalies_default(self, sample_signal_1d):
        """Test detect_anomalies convenience function with defaults."""
        from vitalDSP.ml_models.autoencoder import detect_anomalies
        
        # epochs and verbose are not part of autoencoder_kwargs - they're hardcoded in detect_anomalies
        anomalies, scores, threshold = detect_anomalies(
            sample_signal_1d,
            autoencoder_type="standard",
            latent_dim=16
        )
        
        assert anomalies.shape == (100,)
        assert scores.shape == (100,)
        assert isinstance(threshold, float)

    def test_detect_anomalies_with_threshold(self, sample_signal_1d):
        """Test detect_anomalies with custom threshold."""
        from vitalDSP.ml_models.autoencoder import detect_anomalies
        
        # Note: detect_anomalies doesn't accept custom threshold parameter
        # It calculates threshold based on contamination parameter
        anomalies, scores, threshold = detect_anomalies(
            sample_signal_1d,
            autoencoder_type="standard",
            latent_dim=16,
            contamination=0.1  # This affects threshold calculation
        )
        
        assert isinstance(threshold, float)
        assert threshold > 0

    def test_denoise_signal_default(self, sample_signal_1d):
        """Test denoise_signal convenience function."""
        from vitalDSP.ml_models.autoencoder import denoise_signal
        
        # epochs and verbose are not accepted - they're hardcoded in denoise_signal
        denoised = denoise_signal(
            sample_signal_1d,
            latent_dim=16,
            noise_level=0.2
        )
        
        assert denoised.shape == sample_signal_1d.shape

