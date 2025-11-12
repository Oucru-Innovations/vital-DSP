import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from scipy.signal import hilbert
from vitalDSP.transforms.hilbert_transform import HilbertTransform


class TestHilbertTransform:

    def test_hilbert_transform(self):
        # Create a test signal
        signal = np.sin(np.linspace(0, 10, 100))

        # Instantiate the HilbertTransform class
        ht = HilbertTransform(signal)

        # Compute the Hilbert Transform using the class method
        analytic_signal = ht.compute_hilbert()

        # Compute the Hilbert Transform using scipy for validation
        expected_analytic_signal = hilbert(signal)

        # Assert that the result is close to the expected value
        assert_almost_equal(analytic_signal, expected_analytic_signal, decimal=5)

    def test_envelope(self):
        # Create a test signal
        signal = np.sin(np.linspace(0, 10, 100))

        # Instantiate the HilbertTransform class
        ht = HilbertTransform(signal)

        # Compute the envelope using the class method
        envelope = ht.envelope()

        # Compute the envelope using scipy for validation
        expected_envelope = np.abs(hilbert(signal))

        # Assert that the result is close to the expected value
        assert_almost_equal(envelope, expected_envelope, decimal=5)

    def test_instantaneous_phase(self):
        # Create a test signal
        signal = np.sin(np.linspace(0, 10, 100))

        # Instantiate the HilbertTransform class
        ht = HilbertTransform(signal)

        # Compute the instantaneous phase using the class method
        phase = ht.instantaneous_phase()

        # Compute the phase using scipy for validation
        expected_phase = np.angle(hilbert(signal))

        # Assert that the result is close to the expected value
        assert_almost_equal(phase, expected_phase, decimal=5)

    def test_hilbert_transform_odd_length(self):
        """Test Hilbert Transform with odd-length signal.
        
        This test covers lines 91-92 in hilbert_transform.py where
        the else branch is executed when N % 2 != 0.
        """
        # Create a test signal with odd length (101 samples)
        signal = np.sin(np.linspace(0, 10, 101))

        # Instantiate the HilbertTransform class
        ht = HilbertTransform(signal)

        # Compute the Hilbert Transform using the class method
        analytic_signal = ht.compute_hilbert()

        # Compute the Hilbert Transform using scipy for validation
        expected_analytic_signal = hilbert(signal)

        # Assert that the result is close to the expected value
        assert_almost_equal(analytic_signal, expected_analytic_signal, decimal=5)

    def test_envelope_odd_length(self):
        """Test envelope computation with odd-length signal."""
        # Create a test signal with odd length
        signal = np.sin(np.linspace(0, 10, 99))

        # Instantiate the HilbertTransform class
        ht = HilbertTransform(signal)

        # Compute the envelope using the class method
        envelope = ht.envelope()

        # Compute the envelope using scipy for validation
        expected_envelope = np.abs(hilbert(signal))

        # Assert that the result is close to the expected value
        assert_almost_equal(envelope, expected_envelope, decimal=5)

    def test_instantaneous_phase_odd_length(self):
        """Test instantaneous phase computation with odd-length signal."""
        # Create a test signal with odd length
        signal = np.sin(np.linspace(0, 10, 97))

        # Instantiate the HilbertTransform class
        ht = HilbertTransform(signal)

        # Compute the instantaneous phase using the class method
        phase = ht.instantaneous_phase()

        # Compute the phase using scipy for validation
        expected_phase = np.angle(hilbert(signal))

        # Assert that the result is close to the expected value
        assert_almost_equal(phase, expected_phase, decimal=5)

    def test_hilbert_transform_small_odd_length(self):
        """Test Hilbert Transform with a very small odd-length signal."""
        # Create a test signal with small odd length (5 samples)
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Instantiate the HilbertTransform class
        ht = HilbertTransform(signal)

        # Compute the Hilbert Transform using the class method
        analytic_signal = ht.compute_hilbert()

        # Compute the Hilbert Transform using scipy for validation
        expected_analytic_signal = hilbert(signal)

        # Assert that the result is close to the expected value
        assert_almost_equal(analytic_signal, expected_analytic_signal, decimal=5)

    def test_hilbert_transform_single_sample(self):
        """Test Hilbert Transform with single sample (edge case)."""
        # Create a test signal with single sample
        signal = np.array([1.0])

        # Instantiate the HilbertTransform class
        ht = HilbertTransform(signal)

        # Compute the Hilbert Transform using the class method
        analytic_signal = ht.compute_hilbert()

        # Compute the Hilbert Transform using scipy for validation
        expected_analytic_signal = hilbert(signal)

        # Assert that the result is close to the expected value
        assert_almost_equal(analytic_signal, expected_analytic_signal, decimal=5)


# To run the tests, simply execute: pytest test_hilbert_transform.py
