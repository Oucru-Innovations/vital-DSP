import numpy as np

class Wavelet:
    """
    A class for generating different types of mother wavelets based on their mathematical formulas.

    Methods
    -------
    haar : static method
        Generates a Haar wavelet.
    db : static method
        Generates a Daubechies wavelet.
    sym : static method
        Generates a Symlet wavelet.
    coif : static method
        Generates a Coiflet wavelet.
    gauchy : static method
        Generates a Gaussian wavelet.
    mexican_hat : static method
        Generates a Mexican Hat wavelet.
    morlet : static method
        Generates a Morlet wavelet.
    meyer : static method
        Generates a Meyer wavelet.
    biorthogonal : static method
        Generates a Biorthogonal wavelet.
    reverse_biorthogonal : static method
        Generates a Reverse Biorthogonal wavelet.
    complex_gaussian : static method
        Generates a Complex Gaussian wavelet.
    shannon : static method
        Generates a Shannon wavelet.
    cmor : static method
        Generates a Complex Morlet wavelet.
    fbsp : static method
        Generates a Frequency B-Spline wavelet.
    mhat : static method
        Generates a Modified Mexican Hat wavelet.
    custom_wavelet : static method
        Generates a custom wavelet provided by the user.
    """

    @staticmethod
    def haar():
        """
        Generate a Haar wavelet.
        Best for detecting sudden changes in signals (e.g., step functions).

        Returns
        -------
        wavelet : numpy.ndarray
            The Haar wavelet.

        Examples
        --------
        >>> haar_wavelet = Wavelet.haar()
        >>> print(haar_wavelet)
        [0.70710678 0.70710678]
        """
        return np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])

    @staticmethod
    def db(order=4):
        """
        Generate a Daubechies wavelet of a given order using the wavelet formula.
        Suitable for analyzing smooth signals with sharp discontinuities like ECG signals.

        Parameters
        ----------
        order : int
            The order of the Daubechies wavelet.

        Returns
        -------
        wavelet_coeffs : numpy.ndarray
            The Daubechies wavelet coefficients.

        Examples
        --------
        >>> db_wavelet = Wavelet.db(order=4)
        >>> print(db_wavelet)
        """
        p = np.poly1d([1])
        for k in range(1, order + 1):
            p = np.convolve(p, np.poly1d([1, 1]))
        p = np.polyder(p)
        roots = np.roots(p)
        angles = np.angle(roots)
        indices = np.argsort(angles)
        roots = roots[indices]
        wavelet_coeffs = np.polyval(p, roots)
        wavelet_coeffs = wavelet_coeffs / np.sqrt(np.sum(wavelet_coeffs**2))
        return wavelet_coeffs

    @staticmethod
    def sym(order=4):
        """
        Generate a Symlet wavelet of a given order using the wavelet formula.
        Ideal for symmetric analysis, minimizing edge effects in signal processing.

        Parameters
        ----------
        order : int
            The order of the Symlet wavelet.

        Returns
        -------
        wavelet_coeffs : numpy.ndarray
            The Symlet wavelet coefficients.

        Examples
        --------
        >>> sym_wavelet = Wavelet.sym(order=4)
        >>> print(sym_wavelet)
        """
        p = np.poly1d([1])
        for k in range(1, order + 1):
            p = np.convolve(p, np.poly1d([1, 1]))
        p = np.polyder(p)
        roots = np.roots(p)
        angles = np.angle(roots)
        indices = np.argsort(angles)
        roots = roots[indices]
        wavelet_coeffs = np.polyval(p, roots)
        wavelet_coeffs = wavelet_coeffs / np.sqrt(np.sum(wavelet_coeffs**2))
        return wavelet_coeffs

    @staticmethod
    def coif(order=1):
        """
        Generate a Coiflet wavelet of a given order using the wavelet formula.
        Provides near-symmetry and smoothness, making it ideal for subtle signal variations.

        Parameters
        ----------
        order : int
            The order of the Coiflet wavelet.

        Returns
        -------
        wavelet_coeffs : numpy.ndarray
            The Coiflet wavelet coefficients.

        Examples
        --------
        >>> coif_wavelet = Wavelet.coif(order=1)
        >>> print(coif_wavelet)
        """
        p = np.poly1d([1])
        for k in range(1, order + 1):
            p = np.convolve(p, np.poly1d([1, 1]))
        p = np.polyder(p)
        roots = np.roots(p)
        angles = np.angle(roots)
        indices = np.argsort(angles)
        roots = roots[indices]
        wavelet_coeffs = np.polyval(p, roots)
        wavelet_coeffs = wavelet_coeffs / np.sqrt(np.sum(wavelet_coeffs**2))
        return wavelet_coeffs

    @staticmethod
    def gauchy(sigma=1.0, N=6):
        """
        Generate a Gaussian wavelet.
        Excellent for isolating specific features with a Gaussian shape in the time domain, such as spikes in EEG signals.

        Parameters
        ----------
        sigma : float
            The standard deviation of the Gaussian function.
        N : int
            The number of points in the wavelet.

        Returns
        -------
        wavelet : numpy.ndarray
            The Gaussian wavelet coefficients.

        Examples
        --------
        >>> gauchy_wavelet = Wavelet.gauchy(sigma=1.0, N=6)
        >>> print(gauchy_wavelet)
        """
        t = np.arange(-N // 2, N // 2)
        return np.exp(-(t**2) / (2 * sigma**2))

    @staticmethod
    def mexican_hat(sigma=1.0, N=6):
        """
        Generate a Mexican Hat wavelet (also known as the Ricker wavelet).
        Commonly used in detecting peaks and ridges, particularly in EEG spike detection.

        Parameters
        ----------
        sigma : float
            The standard deviation of the Gaussian function.
        N : int
            The number of points in the wavelet.

        Returns
        -------
        wavelet : numpy.ndarray
            The Mexican Hat wavelet coefficients.

        Examples
        --------
        >>> mexican_hat_wavelet = Wavelet.mexican_hat(sigma=1.0, N=6)
        >>> print(mexican_hat_wavelet)
        """
        t = np.arange(-N // 2, N // 2)
        return (1 - (t / sigma) ** 2) * np.exp(-(t**2) / (2 * sigma**2))

    @staticmethod
    def morlet(sigma=1.0, N=6, f=1.0):
        """
        Generate a Morlet wavelet.
        Useful for time-frequency analysis, frequently applied in EEG and seismic signal analysis.

        Parameters
        ----------
        sigma : float
            The standard deviation of the Gaussian function.
        N : int
            The number of points in the wavelet.
        f : float
            The central frequency of the Morlet wavelet.

        Returns
        -------
        wavelet : numpy.ndarray
            The Morlet wavelet coefficients.

        Examples
        --------
        >>> morlet_wavelet = Wavelet.morlet(sigma=1.0, N=6, f=1.0)
        >>> print(morlet_wavelet)
        """
        t = np.arange(-N // 2, N // 2)
        return np.cos(2 * np.pi * f * t) * np.exp(-(t**2) / (2 * sigma**2))

    @staticmethod
    def meyer(N=6):
        """
        Generate a Meyer wavelet.
        Smooth wavelet, suitable for analyzing slowly varying signals without sharp edges.

        Parameters
        ----------
        N : int
            The number of points in the wavelet.

        Returns
        -------
        wavelet : numpy.ndarray
            The Meyer wavelet coefficients.

        Examples
        --------
        >>> meyer_wavelet = Wavelet.meyer(N=6)
        >>> print(meyer_wavelet)
        """
        t = np.linspace(-np.pi, np.pi, N)
        return np.sin(np.pi * t / 2) * np.cos(np.pi * t)

    @staticmethod
    def biorthogonal(N=6, p=2, q=2):
        """
        Generate a Biorthogonal wavelet.
        Provides exact reconstruction with linear phase, useful in image compression (e.g., JPEG 2000).

        Parameters
        ----------
        N : int
            The number of points in the wavelet.
        p : int
            The order of the Biorthogonal wavelet.
        q : int
            The order of the Reverse Biorthogonal wavelet.

        Returns
        -------
        wavelet : numpy.ndarray
            The Biorthogonal wavelet coefficients.

        Examples
        --------
        >>> biorthogonal_wavelet = Wavelet.biorthogonal(N=6, p=2, q=2)
        >>> print(biorthogonal_wavelet)
        """
        t = np.linspace(-1, 1, N)
        return np.polyval([1, 0, 0], t) * np.sin(np.pi * p * t) * np.sin(np.pi * q * t)

    @staticmethod
    def reverse_biorthogonal(N=6, p=2, q=2):
        """
        Generate a Reverse Biorthogonal wavelet.
        Similar to Biorthogonal but reversed, useful in de-noising applications.

        Parameters
        ----------
        N : int
            The number of points in the wavelet.
        p : int
            The order of the Biorthogonal wavelet.
        q : int
            The order of the Reverse Biorthogonal wavelet.

        Returns
        -------
        wavelet : numpy.ndarray
            The Reverse Biorthogonal wavelet coefficients.

        Examples
        --------
        >>> reverse_biorthogonal_wavelet = Wavelet.reverse_biorthogonal(N=6, p=2, q=2)
        >>> print(reverse_biorthogonal_wavelet)
        """
        t = np.linspace(-1, 1, N)
        return np.polyval([1, 0, 0], t) * np.cos(np.pi * p * t) * np.cos(np.pi * q * t)

    @staticmethod
    def complex_gaussian(sigma=1.0, N=6):
        """
        Generate a Complex Gaussian wavelet.
        Applied in phase and frequency modulation, suitable for radar and communication signals.

        Parameters
        ----------
        sigma : float
            The standard deviation of the Gaussian function.
        N : int
            The number of points in the wavelet.

        Returns
        -------
        wavelet : numpy.ndarray
            The Complex Gaussian wavelet coefficients.

        Examples
        --------
        >>> complex_gaussian_wavelet = Wavelet.complex_gaussian(sigma=1.0, N=6)
        >>> print(complex_gaussian_wavelet)
        """
        t = np.arange(-N // 2, N // 2)
        return np.exp(-(t**2) / (2 * sigma**2)) * np.exp(1j * np.pi * t)

    @staticmethod
    def shannon(N=6):
        """
        Generate a Shannon wavelet.
        Ideal for signals with sharp time-frequency localization, such as short pulses.

        Parameters
        ----------
        N : int
            The number of points in the wavelet.

        Returns
        -------
        wavelet : numpy.ndarray
            The Shannon wavelet coefficients.

        Examples
        --------
        >>> shannon_wavelet = Wavelet.shannon(N=6)
        >>> print(shannon_wavelet)
        """
        t = np.linspace(-1, 1, N)
        return np.sinc(t)

    @staticmethod
    def cmor(sigma=1.0, N=6, f=1.0):
        """
        Generate a Complex Morlet wavelet.
        Used for time-frequency localization, particularly in non-stationary signal analysis.

        Parameters
        ----------
        sigma : float
            The standard deviation of the Gaussian function.
        N : int
            The number of points in the wavelet.
        f : float
            The central frequency of the Complex Morlet wavelet.

        Returns
        -------
        wavelet : numpy.ndarray
            The Complex Morlet wavelet coefficients.

        Examples
        --------
        >>> cmor_wavelet = Wavelet.cmor(sigma=1.0, N=6, f=1.0)
        >>> print(cmor_wavelet)
        """
        t = np.arange(-N // 2, N // 2)
        return np.exp(-(t**2) / (2 * sigma**2)) * np.exp(1j * 2 * np.pi * f * t)

    @staticmethod
    def fbsp(N=6, m=5, s=0.5):
        """
        Generate a Frequency B-Spline wavelet.
        Useful for analyzing frequency-modulated signals, often applied in audio signal processing.

        Parameters
        ----------
        N : int
            The number of points in the wavelet.
        m : int
            The order of the B-spline.
        s : float
            The scaling factor.

        Returns
        -------
        wavelet : numpy.ndarray
            The Frequency B-Spline wavelet coefficients.

        Examples
        --------
        >>> fbsp_wavelet = Wavelet.fbsp(N=6, m=5, s=0.5)
        >>> print(fbsp_wavelet)
        """
        t = np.linspace(-1, 1, N)
        return np.exp(-(t**2) / (2 * s**2)) * np.sin(np.pi * m * t)

    @staticmethod
    def mhat(sigma=1.0, N=6):
        """
        Generate a Modified Mexican Hat wavelet.
        Applied in edge detection in images and other signals with sharp transitions.

        Parameters
        ----------
        sigma : float
            The standard deviation of the Gaussian function.
        N : int
            The number of points in the wavelet.

        Returns
        -------
        wavelet : numpy.ndarray
            The Modified Mexican Hat wavelet coefficients.

        Examples
        --------
        >>> mhat_wavelet = Wavelet.mhat(sigma=1.0, N=6)
        >>> print(mhat_wavelet)
        """
        t = np.arange(-N // 2, N // 2)
        return (1 - (t / sigma) ** 2) * np.exp(-(t**2) / (2 * sigma**2))

    @staticmethod
    def custom_wavelet(wavelet):
        """
        Use a custom wavelet provided by the user.

        Parameters
        ----------
        wavelet : numpy.ndarray
            The custom wavelet coefficients.

        Returns
        -------
        wavelet : numpy.ndarray
            The custom wavelet.

        Examples
        --------
        >>> custom_wavelet = Wavelet.custom_wavelet(np.array([0.2, 0.5, 0.2]))
        >>> print(custom_wavelet)
        """
        return wavelet
