import numpy as np
from utils.wavelet import Wavelet

class WaveletTransform:
    # No change in structure, only in added wavelets
    ...

    def compute_wavelet_transform(self, wavelet_type='haar', order=4, **kwargs):
        """
        Compute the Wavelet Transform of the signal using the specified wavelet type and order.

        Parameters:
        wavelet_type (str): The type of wavelet to use ('haar', 'db', 'sym', 'coif', 'gauchy', 'mexican_hat', 'morlet', 'meyer', 'biorthogonal', 'reverse_biorthogonal', 'complex_gaussian', 'shannon', 'cmor', 'fbsp', 'mhat', 'custom').
        order (int): The order of the wavelet (used for 'db', 'sym', and 'coif').
        kwargs: Additional parameters for specific wavelets.

        Returns:
        list of numpy.ndarray: The wavelet coefficients at different scales.
        """
        wavelet = Wavelet()
        if wavelet_type == 'haar':
            mother_wavelet = wavelet.haar()
        elif wavelet_type == 'db':
            mother_wavelet = wavelet.db(order)
        elif wavelet_type == 'sym':
            mother_wavelet = wavelet.sym(order)
        elif wavelet_type == 'coif':
            mother_wavelet = wavelet.coif(order)
        elif wavelet_type == 'gauchy':
            mother_wavelet = wavelet.gauchy(**kwargs)
        elif wavelet_type == 'mexican_hat':
            mother_wavelet = wavelet.mexican_hat(**kwargs)
        elif wavelet_type == 'morlet':
            mother_wavelet = wavelet.morlet(**kwargs)
        elif wavelet_type == 'meyer':
            mother_wavelet = wavelet.meyer(**kwargs)
        elif wavelet_type == 'biorthogonal':
            mother_wavelet = wavelet.biorthogonal(**kwargs)
        elif wavelet_type == 'reverse_biorthogonal':
            mother_wavelet = wavelet.reverse_biorthogonal(**kwargs)
        elif wavelet_type == 'complex_gaussian':
            mother_wavelet = wavelet.complex_gaussian(**kwargs)
        elif wavelet_type == 'shannon':
            mother_wavelet = wavelet.shannon(**kwargs)
        elif wavelet_type == 'cmor':
            mother_wavelet = wavelet.cmor(**kwargs)
        elif wavelet_type == 'fbsp':
            mother_wavelet = wavelet.fbsp(**kwargs)
        elif wavelet_type == 'mhat':
            mother_wavelet = wavelet.mhat(**kwargs)
        else:
            raise ValueError("Invalid wavelet_type. Must be 'haar', 'db', 'sym', 'coif', 'gauchy', 'mexican_hat', 'morlet', 'meyer', 'biorthogonal', 'reverse_biorthogonal', 'complex_gaussian', 'shannon', 'cmor', 'fbsp', 'mhat', or 'custom'.")

        coeffs = []
        current_signal = self.signal.copy()
        while len(current_signal) >= len(mother_wavelet):
            coeff = np.convolve(current_signal, mother_wavelet, mode='valid')[::2]
            coeffs.append(coeff)
            current_signal = np.convolve(current_signal, mother_wavelet[::-1], mode='valid')[::2]
        return coeffs

    def inverse_wavelet_transform(self, coeffs, wavelet_type='haar', order=4, **kwargs):
        """
        Reconstruct the signal from its wavelet coefficients.

        Parameters:
        coeffs (list of numpy.ndarray): The wavelet coefficients at different scales.
        wavelet_type (str): The type of wavelet used for the transformation.
        order (int): The order of the wavelet used.
        kwargs: Additional parameters for specific wavelets.

        Returns:
        numpy.ndarray: The reconstructed time-domain signal.
        """
        wavelet = Wavelet()
        if wavelet_type == 'haar':
            mother_wavelet = wavelet.haar()
        elif wavelet_type == 'db':
            mother_wavelet = wavelet.db(order)
        elif wavelet_type == 'sym':
            mother_wavelet = wavelet.sym(order)
        elif wavelet_type == 'coif':
            mother_wavelet = wavelet.coif(order)
        elif wavelet_type == 'gauchy':
            mother_wavelet = wavelet.gauchy(**kwargs)
        elif wavelet_type == 'mexican_hat':
            mother_wavelet = wavelet.mexican_hat(**kwargs)
        elif wavelet_type == 'morlet':
            mother_wavelet = wavelet.morlet(**kwargs)
        elif wavelet_type == 'meyer':
            mother_wavelet = wavelet.meyer(**kwargs)
        elif wavelet_type == 'biorthogonal':
            mother_wavelet = wavelet.biorthogonal(**kwargs)
        elif wavelet_type == 'reverse_biorthogonal':
            mother_wavelet = wavelet.reverse_biorthogonal(**kwargs)
        elif wavelet_type == 'complex_gaussian':
            mother_wavelet = wavelet.complex_gaussian(**kwargs)
        elif wavelet_type == 'shannon':
            mother_wavelet = wavelet.shannon(**kwargs)
        elif wavelet_type == 'cmor':
            mother_wavelet = wavelet.cmor(**kwargs)
        elif wavelet_type == 'fbsp':
            mother_wavelet = wavelet.fbsp(**kwargs)
        elif wavelet_type == 'mhat':
            mother_wavelet = wavelet.mhat(**kwargs)
        else:
            raise ValueError("Invalid wavelet_type. Must be 'haar', 'db', 'sym', 'coif', 'gauchy', 'mexican_hat', 'morlet', 'meyer', 'biorthogonal', 'reverse_biorthogonal', 'complex_gaussian', 'shannon', 'cmor', 'fbsp', 'mhat', or 'custom'.")

        reconstructed_signal = coeffs[-1]
        for coeff in reversed(coeffs[:-1]):
            expanded_coeff = np.repeat(reconstructed_signal, 2)
            reconstructed_signal = np.convolve(expanded_coeff, mother_wavelet[::-1], mode='full')[:len(coeff)]
            reconstructed_signal += coeff
        return reconstructed_signal
