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


# To run the tests, simply execute: pytest test_hilbert_transform.py
