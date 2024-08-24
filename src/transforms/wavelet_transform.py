import numpy as np
from utils.mother_wavelets import Wavelet

class WaveletTransform:
    """
    A class to perform Discrete Wavelet Transform (DWT) on signals using different mother wavelets.

    Methods
    -------
    perform_wavelet_transform : method
        Computes the DWT of the signal.
    perform_inverse_wavelet_transform : method
        Reconstructs the signal using the inverse DWT.
    """

    def __init__(self, signal, wavelet_name="haar"):
        """
        Initialize the WaveletTransform class with the signal and select the mother wavelet.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal to be transformed.
        wavelet_name : str, optional
            Name of the wavelet to be used (default is 'haar').

        Raises
        ------
        ValueError
            If the specified wavelet name is not found in the Wavelet class.
        """
        self.signal = signal
        self.original_length = len(signal)  # Store the original length of the signal
        self.wavelet_name = wavelet_name

        # Retrieve the wavelet filters (low_pass, high_pass) from the Wavelet class
        wavelet_class = Wavelet()
        wavelet_method = getattr(wavelet_class, wavelet_name, None)

        if wavelet_method is None:
            raise ValueError(f"Wavelet '{wavelet_name}' not found in Wavelet class.")

        # Call the wavelet method to get the wavelet coefficients
        filters = wavelet_method()

        if isinstance(filters, tuple) and len(filters) == 2:
            self.low_pass, self.high_pass = filters
        else:
            # If only one filter is returned, assume it's a low-pass filter
            self.low_pass = filters
            self.high_pass = np.array([1, -1])  # Use a default or dummy high-pass filter

        # Ensure the wavelet filters are numpy arrays
        if isinstance(self.low_pass, (float, int)):
            self.low_pass = np.array([self.low_pass])
        if isinstance(self.high_pass, (float, int)):
            self.high_pass = np.array([self.high_pass])

    def _wavelet_decompose(self, data):
        """
        Perform a single-level wavelet transform on the data using the specified wavelet function.

        Parameters
        ----------
        data : numpy.ndarray
            The input data to be transformed.

        Returns
        -------
        tuple
            Approximation coefficients and detail coefficients.
        """
        output_length = len(data) // 2
        approximation = np.zeros(output_length)
        detail = np.zeros(output_length)

        filter_len = len(self.low_pass)

        for i in range(output_length):
            data_segment = data[2 * i : 2 * i + filter_len]

            # If the data segment is shorter than the filter, pad it with zeros
            if len(data_segment) < filter_len:
                data_segment = np.pad(
                    data_segment, (0, filter_len - len(data_segment)), "constant"
                )

            approximation[i] = np.dot(self.low_pass, data_segment)
            detail[i] = np.dot(self.high_pass, data_segment)

        return approximation, detail

    def perform_wavelet_transform(self, level=1):
        """
        Perform the Discrete Wavelet Transform (DWT) on the signal.

        Parameters
        ----------
        level : int, optional
            The number of decomposition levels (default is 1).

        Returns
        -------
        list
            Wavelet coefficients, where each element corresponds to one level of decomposition.

        Examples
        --------
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

        Parameters
        ----------
        approximation : numpy.ndarray
            Approximation coefficients.
        detail : numpy.ndarray
            Detail coefficients.

        Returns
        -------
        numpy.ndarray
            Reconstructed data at this level.
        """
        output_length = len(approximation) * 2
        data = np.zeros(output_length)

        filter_len = len(self.low_pass)

        for i in range(len(approximation)):
            for j in range(filter_len):
                index = 2 * i + j
                if index < output_length:
                    data[index] += (
                        approximation[i] * self.low_pass[j]
                        + detail[i] * self.high_pass[j]
                    ) / np.sqrt(2)

        return data

    def perform_inverse_wavelet_transform(self, coeffs):
        """
        Perform the Inverse Discrete Wavelet Transform (IDWT) to reconstruct the signal.

        Parameters
        ----------
        coeffs : list
            Wavelet coefficients from the wavelet transform.

        Returns
        -------
        numpy.ndarray
            Reconstructed signal from the wavelet coefficients.

        Examples
        --------
        >>> signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
        >>> wavelet_transform = WaveletTransform(signal, wavelet_name='db4')
        >>> coeffs = wavelet_transform.perform_wavelet_transform(level=3)
        >>> reconstructed_signal = wavelet_transform.perform_inverse_wavelet_transform(coeffs)
        >>> print(reconstructed_signal)
        """
        data = coeffs[-1]  # Start with the final approximation
        for detail in reversed(coeffs[:-1]):
            data = self._wavelet_reconstruct(data, detail)
        return data[: self.original_length]  # Trim to the original length of the signal
