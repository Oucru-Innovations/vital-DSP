import numpy as np

class Wavelet:
    """
    A class for generating different types of mother wavelets based on their mathematical formulas.

    Wavelets are used in various signal processing applications, particularly in analyzing non-stationary signals where both time and frequency localization is important. This class provides methods to generate common wavelets used in discrete wavelet transforms (DWT).

    Methods
    -------
    haar() -> numpy.ndarray
        Generates a Haar wavelet, best for detecting sudden changes in signals.
    db(order=4) -> numpy.ndarray
        Generates a Daubechies wavelet, suitable for analyzing smooth signals with sharp discontinuities.
    sym(order=4) -> numpy.ndarray
        Generates a Symlet wavelet, ideal for symmetric analysis and minimizing edge effects.
    coif(order=1) -> numpy.ndarray
        Generates a Coiflet wavelet, providing near-symmetry and smoothness, ideal for subtle signal variations.
    gauchy(sigma=1.0, N=6) -> numpy.ndarray
        Generates a Gaussian wavelet, excellent for isolating features with a Gaussian shape in the time domain.
    mexican_hat(sigma=1.0, N=6) -> numpy.ndarray
        Generates a Mexican Hat wavelet, commonly used for detecting peaks and ridges.
    morlet(sigma=1.0, N=6, f=1.0) -> numpy.ndarray
        Generates a Morlet wavelet, useful for time-frequency analysis.
    meyer(N=6) -> numpy.ndarray
        Generates a Meyer wavelet, suitable for analyzing slowly varying signals without sharp edges.
    biorthogonal(N=6, p=2, q=2) -> numpy.ndarray
        Generates a Biorthogonal wavelet, useful in image compression and providing exact reconstruction with linear phase.
    reverse_biorthogonal(N=6, p=2, q=2) -> numpy.ndarray
        Generates a Reverse Biorthogonal wavelet, useful in de-noising applications.
    complex_gaussian(sigma=1.0, N=6) -> numpy.ndarray
        Generates a Complex Gaussian wavelet, applied in phase and frequency modulation.
    shannon(N=6) -> numpy.ndarray
        Generates a Shannon wavelet, ideal for signals with sharp time-frequency localization.
    cmor(sigma=1.0, N=6, f=1.0) -> numpy.ndarray
        Generates a Complex Morlet wavelet, used for time-frequency localization in non-stationary signals.
    fbsp(N=6, m=5, s=0.5) -> numpy.ndarray
        Generates a Frequency B-Spline wavelet, useful for analyzing frequency-modulated signals.
    mhat(sigma=1.0, N=6) -> numpy.ndarray
        Generates a Modified Mexican Hat wavelet, applied in edge detection in images and other signals.
    custom_wavelet(wavelet: numpy.ndarray) -> numpy.ndarray
        Use a custom wavelet provided by the user.
    """

    @staticmethod
    def haar():
        """
        Generate a Haar wavelet.

        The Haar wavelet is the simplest wavelet, best suited for detecting sudden changes in a signal, such as step functions or sharp transitions.

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

        Daubechies wavelets are used for analyzing signals with sharp discontinuities, such as ECG signals. They provide a good balance between compact support and smoothness.

        Parameters
        ----------
        order : int, optional
            The order of the Daubechies wavelet (default is 4).

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

        Symlet wavelets are nearly symmetric, minimizing edge effects in signal processing. They are particularly useful when symmetric analysis is required.

        Parameters
        ----------
        order : int, optional
            The order of the Symlet wavelet (default is 4).

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

        Coiflet wavelets are nearly symmetric and provide smooth waveforms, making them ideal for analyzing subtle variations in signals.

        Parameters
        ----------
        order : int, optional
            The order of the Coiflet wavelet (default is 1).

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

        Gaussian wavelets are ideal for isolating specific features with a Gaussian shape in the time domain, such as spikes in EEG signals.

        Parameters
        ----------
        sigma : float, optional
            The standard deviation of the Gaussian function (default is 1.0).
        N : int, optional
            The number of points in the wavelet (default is 6).

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

        The Mexican Hat wavelet is commonly used in detecting peaks and ridges in signals, particularly useful in EEG spike detection.

        Parameters
        ----------
        sigma : float, optional
            The standard deviation of the Gaussian function (default is 1.0).
        N : int, optional
            The number of points in the wavelet (default is 6).

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

        Morlet wavelets are particularly useful for time-frequency analysis and are frequently applied in EEG and seismic signal analysis.

        Parameters
        ----------
        sigma : float, optional
            The standard deviation of the Gaussian function (default is 1.0).
        N : int, optional
            The number of points in the wavelet (default is 6).
        f : float, optional
            The central frequency of the Morlet wavelet (default is 1.0).

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

        Meyer wavelets are smooth and suitable for analyzing slowly varying signals without sharp edges.

        Parameters
        ----------
        N : int, optional
            The number of points in the wavelet (default is 6).

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

        Biorthogonal wavelets provide exact reconstruction with linear phase, making them useful in image compression (e.g., JPEG 2000).

        Parameters
        ----------
        N : int, optional
            The number of points in the wavelet (default is 6).
        p : int, optional
            The order of the Biorthogonal wavelet (default is 2).
        q : int, optional
            The order of the Reverse Biorthogonal wavelet (default is 2).

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

        Reverse Biorthogonal wavelets are similar to Biorthogonal wavelets but with reversed reconstruction properties, making them useful in de-noising applications.

        Parameters
        ----------
        N : int, optional
            The number of points in the wavelet (default is 6).
        p : int, optional
            The order of the Biorthogonal wavelet (default is 2).
        q : int, optional
            The order of the Reverse Biorthogonal wavelet (default is 2).

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

        Complex Gaussian wavelets are applied in phase and frequency modulation, making them suitable for radar and communication signals.

        Parameters
        ----------
        sigma : float, optional
            The standard deviation of the Gaussian function (default is 1.0).
        N : int, optional
            The number of points in the wavelet (default is 6).

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

        Shannon wavelets are ideal for signals with sharp time-frequency localization, such as short pulses.

        Parameters
        ----------
        N : int, optional
            The number of points in the wavelet (default is 6).

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

        Complex Morlet wavelets are used for time-frequency localization, particularly in non-stationary signal analysis.

        Parameters
        ----------
        sigma : float, optional
            The standard deviation of the Gaussian function (default is 1.0).
        N : int, optional
            The number of points in the wavelet (default is 6).
        f : float, optional
            The central frequency of the Complex Morlet wavelet (default is 1.0).

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

        Frequency B-Spline wavelets are useful for analyzing frequency-modulated signals, often applied in audio signal processing.

        Parameters
        ----------
        N : int, optional
            The number of points in the wavelet (default is 6).
        m : int, optional
            The order of the B-spline (default is 5).
        s : float, optional
            The scaling factor (default is 0.5).

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

        Modified Mexican Hat wavelets are applied in edge detection in images and other signals with sharp transitions.

        Parameters
        ----------
        sigma : float, optional
            The standard deviation of the Gaussian function (default is 1.0).
        N : int, optional
            The number of points in the wavelet (default is 6).

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

        This method allows the use of a custom wavelet defined by the user, which can be applied in specific or novel signal processing tasks.

        Parameters
        ----------
        wavelet : numpy.ndarray
            The custom wavelet coefficients provided by the user.

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
