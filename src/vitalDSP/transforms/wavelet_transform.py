"""
Wavelet Transform Module for Physiological Signal Processing

This module provides comprehensive wavelet transform capabilities for physiological
signals including ECG, PPG, EEG, and other vital signs. It implements Discrete
Wavelet Transform (DWT) with multiple mother wavelets and inverse transform
capabilities for signal analysis and reconstruction.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Discrete Wavelet Transform (DWT) implementation
- Multiple mother wavelets (Haar, Daubechies, Coiflets, etc.)
- Inverse Wavelet Transform for signal reconstruction
- Multi-level decomposition capabilities
- Signal length preservation options
- Integration with mother wavelet utilities

Examples:
--------
Basic wavelet transform:
    >>> import numpy as np
    >>> from vitalDSP.transforms.wavelet_transform import WaveletTransform
    >>> signal = np.sin(np.linspace(0, 10, 1000)) + np.random.normal(0, 0.1, 1000)
    >>> wt = WaveletTransform(signal, wavelet_name="haar")
    >>> coefficients = wt.perform_wavelet_transform()
    >>> print(f"Coefficients shape: {len(coefficients)}")

Signal reconstruction:
    >>> reconstructed = wt.perform_inverse_wavelet_transform(coefficients)
    >>> print(f"Reconstruction error: {np.mean((signal - reconstructed)**2):.6f}")

Different wavelets:
    >>> wt_db4 = WaveletTransform(signal, wavelet_name="db4")
    >>> wt_coif2 = WaveletTransform(signal, wavelet_name="coif2")
    >>> db4_coeffs = wt_db4.perform_wavelet_transform()
    >>> coif2_coeffs = wt_coif2.perform_wavelet_transform()
"""

import numpy as np
from vitalDSP.utils.signal_processing.mother_wavelets import Wavelet
from scipy.signal import convolve


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

    def __init__(self, signal, wavelet_name="haar", same_length=True):
        """
        Initialize the WaveletTransform class with the signal and select the mother wavelet.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal to be transformed.
        wavelet_name : str, optional
            Name of the wavelet to be used (default is 'haar').
        same_length : bool, optional
            If True, the transformed signal will have the same length as the original (default is True).

        Raises
        ------
        ValueError
            If the specified wavelet name is not found in the Wavelet class.
        """
        self.signal = signal
        self.original_length = len(signal)  # Store the original length of the signal
        self.wavelet_name = wavelet_name
        self.same_length = same_length  # Option to maintain the same length

        # Retrieve the wavelet filters (low_pass, high_pass) from the Wavelet class
        wavelet_class = Wavelet()

        # Handle common wavelet naming conventions
        wavelet_mapping = {
            "haar": lambda: wavelet_class.haar(),
            "db1": lambda: wavelet_class.db(order=1),
            "db2": lambda: wavelet_class.db(order=2),
            "db3": lambda: wavelet_class.db(order=3),
            "db4": lambda: wavelet_class.db(order=4),
            "db5": lambda: wavelet_class.db(order=5),
            "db6": lambda: wavelet_class.db(order=6),
            "db7": lambda: wavelet_class.db(order=7),
            "db8": lambda: wavelet_class.db(order=8),
            "sym1": lambda: wavelet_class.sym(order=1),
            "sym2": lambda: wavelet_class.sym(order=2),
            "sym3": lambda: wavelet_class.sym(order=3),
            "sym4": lambda: wavelet_class.sym(order=4),
            "sym5": lambda: wavelet_class.sym(order=5),
            "sym6": lambda: wavelet_class.sym(order=6),
            "sym7": lambda: wavelet_class.sym(order=7),
            "sym8": lambda: wavelet_class.sym(order=8),
            "coif1": lambda: wavelet_class.coif(order=1),
            "coif2": lambda: wavelet_class.coif(order=2),
            "coif3": lambda: wavelet_class.coif(order=3),
            "coif4": lambda: wavelet_class.coif(order=4),
            "coif5": lambda: wavelet_class.coif(order=5),
            # Common abbreviations for continuous wavelets
            "mexh": lambda: wavelet_class.mexican_hat(),
            "morl": lambda: wavelet_class.morlet(),
            "mexican_hat": lambda: wavelet_class.mexican_hat(),
            "morlet": lambda: wavelet_class.morlet(),
        }

        # Get the wavelet method
        if wavelet_name in wavelet_mapping:
            wavelet_method = wavelet_mapping[wavelet_name]
        else:
            # Try direct method lookup as fallback
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
            self.high_pass = np.array(
                [1, -1]
            )  # Use a default or dummy high-pass filter

        # Ensure the wavelet filters are numpy arrays
        self.low_pass = np.asarray(self.low_pass)
        self.high_pass = np.asarray(self.high_pass)

    def _wavelet_decompose(self, data):
        """
        OPTIMIZED: Perform a single-level wavelet transform using vectorized convolution.

        Parameters
        ----------
        data : numpy.ndarray
            The input data to be transformed.

        Returns
        -------
        tuple
            Approximation coefficients and detail coefficients.
        """
        output_length = len(data)
        filter_len = len(self.low_pass)

        # Apply padding based on the same_length option
        if self.same_length:
            padded_data = np.pad(data, (filter_len // 2, filter_len // 2), "reflect")
        else:
            padded_data = np.pad(data, (0, filter_len - 1), "constant")

        # OPTIMIZATION: Use vectorized convolution instead of loops
        try:
            from scipy.signal import convolve

            # OPTIMIZATION: Vectorized convolution for O(n log n) complexity
            approximation = convolve(padded_data, self.low_pass[::-1], mode="valid")
            detail = convolve(padded_data, self.high_pass[::-1], mode="valid")

            # Ensure output length matches expected length
            if len(approximation) > output_length:
                approximation = approximation[:output_length]
            if len(detail) > output_length:
                detail = detail[:output_length]

        except ImportError:
            # Fallback to original implementation if scipy not available
            approximation = np.zeros(output_length)
            detail = np.zeros(output_length)

            # Iterate over the signal and apply the filters
            for i in range(output_length):
                data_segment = padded_data[i : i + filter_len]

                if len(data_segment) == len(self.low_pass):
                    approximation[i] = np.dot(self.low_pass, data_segment)
                if len(data_segment) == len(self.high_pass):
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
        >>> wavelet_transform = WaveletTransform(signal, wavelet_name='db')
        >>> coeffs = wavelet_transform.perform_wavelet_transform(level=3)
        >>> print(coeffs)
        """
        coeffs = []
        data = self.signal.copy()
        for _ in range(level):
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
        # Convolve approximation and detail coefficients with the corresponding filters
        approx_conv = convolve(approximation, self.low_pass, mode="full")
        detail_conv = convolve(detail, self.high_pass, mode="full")

        # Combine the convolved signals and scale appropriately
        data = (approx_conv + detail_conv) / np.sqrt(2)

        # Trim the data to maintain the original length if required
        if self.same_length:
            data = data[: len(approximation)]

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
        >>> wavelet_transform = WaveletTransform(signal, wavelet_name='db')
        >>> coeffs = wavelet_transform.perform_wavelet_transform(level=3)
        >>> reconstructed_signal = wavelet_transform.perform_inverse_wavelet_transform(coeffs)
        >>> print(reconstructed_signal)
        """
        data = coeffs[-1]  # Start with the final approximation
        for detail in reversed(coeffs[:-1]):
            data = self._wavelet_reconstruct(data, detail)
        return data[: self.original_length] if self.same_length else data
