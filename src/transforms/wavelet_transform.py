import numpy as np
from utils.mother_wavelets import Wavelet
class WaveletTransform:
    """
    A class to perform Discrete Wavelet Transform (DWT) on signals using different mother wavelets.
    
    Methods:
    - perform_wavelet_transform: Computes the DWT of the signal.
    - perform_inverse_wavelet_transform: Reconstructs the signal using the inverse DWT.
    """

    def __init__(self, signal, wavelet_name='haar'):
        """
        Initialize the WaveletTransform class with the signal and select the mother wavelet.

        Parameters:
        signal (numpy.ndarray): The input signal to be transformed.
        wavelet_name (str): Name of the wavelet to be used (e.g., 'haar', 'db4').
        """
        self.signal = signal
        self.original_length = len(signal)  # Store the original length of the signal
        self.wavelet_name = wavelet_name
        
        # Retrieve the wavelet filters (low_pass, high_pass) from MotherWavelets
        wavelet_class = Wavelet()
        wavelet_method = getattr(wavelet_class, wavelet_name, None)
        
        if wavelet_method is None:
            raise ValueError(f"Wavelet '{wavelet_name}' not found in MotherWavelets.")
        
        # Call the wavelet method to get the wavelet coefficients
        self.low_pass, self.high_pass = wavelet_method()

        # Ensure the wavelet filters are numpy arrays
        if isinstance(self.low_pass, (float, int)):
            self.low_pass = np.array([self.low_pass])
        if isinstance(self.high_pass, (float, int)):
            self.high_pass = np.array([self.high_pass])

    def _wavelet_decompose(self, data):
        """
        Perform a single-level wavelet transform on the data using the specified wavelet function.

        Parameters:
        data (numpy.ndarray): The input data to be transformed.

        Returns:
        tuple: (approximation coefficients, detail coefficients)
        """
        output_length = len(data) // 2
        approximation = np.zeros(output_length)
        detail = np.zeros(output_length)

        filter_len = len(self.low_pass)

        for i in range(output_length):
            approximation[i] = np.dot(self.low_pass, data[2 * i:2 * i + filter_len])
            detail[i] = np.dot(self.high_pass, data[2 * i:2 * i + filter_len])

        return approximation, detail

    def perform_wavelet_transform(self, level=1):
        """
        Perform the Discrete Wavelet Transform (DWT) on the signal.

        Parameters:
        level (int): The number of decomposition levels.

        Returns:
        list: Wavelet coefficients, where each element corresponds to one level of decomposition.
        
        Example Usage:
        >>> signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
        >>> wavelet_transform = WaveletTransform(signal, wavelet_name='db4')
        >>> coeffs = wavelet_transform.perform_wavelet_transform(level=3)
        >>> print(coeffs)
        """
        coeffs = []
        data = self.signal.copy()
        for _ in range(level):
            if len(data) % 2 != 0:  # Ensure even length by padding with zero
                data = np.append(data, 0)
            approximation, detail = self._wavelet_decompose(data)
            coeffs.append(detail)
            data = approximation
        coeffs.append(data)  # Final approximation at the highest level
        return coeffs

    def _wavelet_reconstruct(self, approximation, detail):
        """
        Perform a single-level inverse wavelet transform using the specified wavelet function.

        Parameters:
        approximation (numpy.ndarray): Approximation coefficients.
        detail (numpy.ndarray): Detail coefficients.

        Returns:
        numpy.ndarray: Reconstructed data at this level.
        """
        output_length = len(approximation) * 2
        data = np.zeros(output_length)

        filter_len = len(self.low_pass)

        for i in range(len(approximation)):
            for j in range(filter_len):
                data[2 * i + j] += (approximation[i] * self.low_pass[j] +
                                    detail[i] * self.high_pass[j]) / np.sqrt(2)

        return data

    def perform_inverse_wavelet_transform(self, coeffs):
        """
        Perform the Inverse Discrete Wavelet Transform (IDWT) to reconstruct the signal.

        Parameters:
        coeffs (list): Wavelet coefficients from the wavelet transform.

        Returns:
        numpy.ndarray: Reconstructed signal from the wavelet coefficients.
        
        Example Usage:
        >>> signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
        >>> wavelet_transform = WaveletTransform(signal, wavelet_name='db4')
        >>> coeffs = wavelet_transform.perform_wavelet_transform(level=3)
        >>> reconstructed_signal = wavelet_transform.perform_inverse_wavelet_transform(coeffs)
        >>> print(reconstructed_signal)
        """
        data = coeffs[-1]  # Start with the final approximation
        for detail in reversed(coeffs[:-1]):
            data = self._wavelet_reconstruct(data, detail)
        return data[:self.original_length]  # Trim to the original length of the signal