"""
Utility Functions Module for Physiological Signal Processing

This module provides comprehensive capabilities for physiological
signal processing including ECG, PPG, EEG, and other vital signs.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Object-oriented design with comprehensive classes
- Multiple processing methods and functions
- NumPy integration for numerical computations

Examples:
---------
Basic usage:
    >>> import numpy as np
    >>> from vitalDSP.signal_processing.mother_wavelets import MotherWavelets
    >>> signal = np.random.randn(1000)
    >>> processor = MotherWavelets(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""

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
        lo = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)])
        hi = np.array([1.0 / np.sqrt(2), -1.0 / np.sqrt(2)])
        return lo, hi

    @staticmethod
    def db(order=4):
        """
        Generate Daubechies wavelet low-pass and high-pass filter coefficients.

        Returns the standard orthonormal Daubechies scaling (low-pass) and
        wavelet (high-pass) filter pair for the given vanishing-moment order.
        Known exact coefficients are used for orders 1-8; higher orders fall
        back to a symmetric approximation.

        Parameters
        ----------
        order : int, optional
            The order (number of vanishing moments) of the Daubechies wavelet
            (default is 4).

        Returns
        -------
        low_pass : numpy.ndarray
            Scaling filter coefficients (real-valued).
        high_pass : numpy.ndarray
            Wavelet filter coefficients (real-valued).

        Examples
        --------
        >>> lo, hi = Wavelet.db(order=4)
        >>> print(lo)
        """
        # Standard Daubechies scaling (low-pass) filter coefficients (db1–db8)
        _DB_LO = {
            1: np.array([0.7071067811865476, 0.7071067811865476]),
            2: np.array(
                [
                    0.4829629131445341,
                    0.8365163037378079,
                    0.2241438680420134,
                    -0.1294095225512604,
                ]
            ),
            3: np.array(
                [
                    0.3326705529500825,
                    0.8068915093110926,
                    0.4598775021184915,
                    -0.1350110200102546,
                    -0.0854412738820267,
                    0.0352262918857096,
                ]
            ),
            4: np.array(
                [
                    0.2303778133088964,
                    0.7148465705529154,
                    0.6308807679298587,
                    -0.0279837694168599,
                    -0.1870348117190931,
                    0.0308413818355607,
                    0.0328830116668852,
                    -0.0105974017850690,
                ]
            ),
            5: np.array(
                [
                    0.1601023979741929,
                    0.6038292697974185,
                    0.7243085284377729,
                    0.1384281459013207,
                    -0.2422948870663824,
                    -0.0322448695846381,
                    0.0775714938400459,
                    -0.0062414902127983,
                    -0.0125807519990820,
                    0.0033357252854738,
                ]
            ),
            6: np.array(
                [
                    0.1115407433501095,
                    0.4946238903984530,
                    0.7511339080210954,
                    0.3152503517092432,
                    -0.2262646939654100,
                    -0.1297668675672624,
                    0.0975016055873224,
                    0.0275228655303053,
                    -0.0315820810014865,
                    0.0005538422009938,
                    0.0047772575109455,
                    -0.0010773010853085,
                ]
            ),
            7: np.array(
                [
                    0.0778520540850037,
                    0.3965393194819173,
                    0.7291320908462368,
                    0.4697822874052152,
                    -0.1439060039285212,
                    -0.2240361849938500,
                    0.0713092192668312,
                    0.0806126091510659,
                    -0.0380299369350122,
                    -0.0165745416306664,
                    0.0125509985560993,
                    0.0004295779729214,
                    -0.0018016407039998,
                    0.0003537137999745,
                ]
            ),
            8: np.array(
                [
                    0.0544158422431049,
                    0.3128715909144659,
                    0.6756307362973195,
                    0.5853546836548691,
                    -0.0158291052563823,
                    -0.2840155429615469,
                    0.0004724845739124,
                    0.1287474266204837,
                    -0.0173693010018083,
                    -0.0440882539307971,
                    0.0139810279173995,
                    0.0087460940474061,
                    -0.0048703529934520,
                    -0.0003917403733770,
                    0.0006754494064506,
                    -0.0001174767841248,
                ]
            ),
        }

        if order in _DB_LO:
            lo = _DB_LO[order].astype(np.float64)
        else:
            # Generic fallback: uniform low-pass of length 2*order
            lo = np.ones(2 * order, dtype=np.float64)

        # Enforce unit energy to absorb any floating-point transcription error
        lo /= np.sqrt(np.sum(lo**2))
        # The QMF high-pass filter is the alternating-flip of the low-pass
        hi = lo[::-1] * np.array([-1 if i % 2 == 0 else 1 for i in range(len(lo))])
        return lo, hi

    @staticmethod
    def sym(order=4):
        """
        Generate Symlet wavelet low-pass and high-pass filter coefficients.

        Symlets are nearly symmetric modifications of the Daubechies wavelets.
        Known exact coefficients are used for orders 2-8; other orders fall
        back to the corresponding Daubechies filters.

        Parameters
        ----------
        order : int, optional
            The order of the Symlet wavelet (default is 4).

        Returns
        -------
        low_pass : numpy.ndarray
            Scaling filter coefficients (real-valued).
        high_pass : numpy.ndarray
            Wavelet filter coefficients (real-valued).

        Examples
        --------
        >>> lo, hi = Wavelet.sym(order=4)
        >>> print(lo)
        """
        _SYM_LO = {
            2: np.array(
                [
                    -0.1294095225512604,
                    0.2241438680420134,
                    0.8365163037378079,
                    0.4829629131445341,
                ]
            ),
            3: np.array(
                [
                    0.0352262918857096,
                    -0.0854412738820267,
                    -0.1350110200102546,
                    0.4598775021184915,
                    0.8068915093110926,
                    0.3326705529500825,
                ]
            ),
            4: np.array(
                [
                    -0.0757657147893406,
                    -0.0296355276459541,
                    0.4976186676324578,
                    0.8037387518052163,
                    0.2978577956052774,
                    -0.0992195435769354,
                    -0.0126039672622612,
                    0.0322231006040427,
                ]
            ),
            5: np.array(
                [
                    0.0273330683451645,
                    -0.0295194909260734,
                    -0.2474985700611133,
                    0.4937216264309682,
                    0.7973510013999680,
                    0.3486048809346920,
                    -0.0676328290612938,
                    -0.1400842430439987,
                    0.0186477833781657,
                    0.0355423381571987,
                ]
            ),
            6: np.array(
                [
                    0.0154041093272182,
                    0.0034907120843304,
                    -0.1179901111484105,
                    -0.0483117425859981,
                    0.4910559419276396,
                    0.7876411410287941,
                    0.3379294217278806,
                    -0.0726375227866000,
                    -0.1663176508429673,
                    0.0046623726483920,
                    0.0121764802753706,
                    -0.0039466905989270,
                ]
            ),
            7: np.array(
                [
                    0.0202767858330571,
                    -0.0170857564509860,
                    -0.1641586756690713,
                    0.0621656457376411,
                    0.4625362936739696,
                    0.7565821109183484,
                    0.3773551352142397,
                    -0.0322431756784820,
                    -0.2088830398028700,
                    0.0321067045087432,
                    0.0454282907382478,
                    -0.0155049631071818,
                    -0.0051526807186808,
                    0.0022025450523861,
                ]
            ),
            8: np.array(
                [
                    -0.0033824159510061,
                    0.0005421323316990,
                    0.0316950878103452,
                    -0.0076074873249766,
                    -0.1432942383510542,
                    0.0612733590679088,
                    0.4813596512592012,
                    0.7771857517005235,
                    0.3644418948359564,
                    -0.0519458381078751,
                    -0.2272180351088840,
                    0.0363585669159919,
                    0.0510097726033448,
                    -0.0210602925126954,
                    -0.0053068792350380,
                    0.0033357252854738,
                ]
            ),
        }

        if order in _SYM_LO:
            lo = _SYM_LO[order].astype(np.float64)
        else:
            # Fall back to Daubechies for unsupported orders
            lo, _ = Wavelet.db(order)

        # Enforce unit energy (||lo||^2 = 1) to guard against transcription drift
        lo /= np.sqrt(np.sum(lo**2))
        hi = lo[::-1] * np.array([-1 if i % 2 == 0 else 1 for i in range(len(lo))])
        return lo, hi

    @staticmethod
    def coif(order=1):
        """
        Generate a Coiflet wavelet of a given order.

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
        # Predefined Coiflet coefficients for orders 1-4
        coif_coeffs = {
            1: np.array(
                [
                    0.038580777748,
                    -0.126969125396,
                    -0.077161555496,
                    0.607491641386,
                    0.745687558934,
                    0.226584265197,
                ]
            ),
            2: np.array(
                [
                    0.016387336463,
                    -0.041464936781,
                    -0.067372554722,
                    0.386110066823,
                    0.812723635445,
                    0.417005184424,
                    -0.076488599078,
                    -0.059434418646,
                    0.023680171947,
                    0.005611434819,
                    -0.001823208870,
                    -0.000720549445,
                ]
            ),
            3: np.array(
                [
                    0.007800708325,
                    -0.013532377880,
                    -0.044663748330,
                    0.191500822714,
                    0.479360089564,
                    0.876501559633,
                    0.417566506506,
                    -0.054463372698,
                    -0.042916387274,
                    0.016727319306,
                    0.004870352993,
                    -0.001456841295,
                    -0.000590847816,
                    0.000149764800,
                    0.000043512627,
                    -0.000014991303,
                ]
            ),
            4: np.array(
                [
                    0.003793512864,
                    -0.004882816378,
                    -0.027219029917,
                    0.093057364604,
                    0.237689909049,
                    0.619330888566,
                    0.687750162028,
                    0.087734129625,
                    -0.070928535954,
                    0.008464837484,
                    0.004258746704,
                    -0.000539645345,
                    -0.000080661204,
                    0.000004626171,
                    0.000001465842,
                    -0.000000095539,
                ]
            ),
        }

        if order in coif_coeffs:
            lo = coif_coeffs[order].astype(np.float64)
        else:
            # Fallback to Haar for unsupported orders
            lo = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)])

        # Enforce unit energy (||lo||^2 = 1)
        lo /= np.sqrt(np.sum(lo**2))
        hi = lo[::-1] * np.array([-1 if i % 2 == 0 else 1 for i in range(len(lo))])
        return lo, hi

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
        t = np.arange(-N // 2, N - N // 2, dtype=np.float64)
        # Meyer wavelet: derivative of the Meyer scaling function
        # psi(t) = (4/(3*pi)) * cos(2*pi*t) * sinc(4t/3) + (8/(3*pi)) * sin(2*pi*t) * sinc(8t/3)
        # Use a numerically stable sinc (np.sinc uses normalized sinc = sin(pi*x)/(pi*x))
        wavelet = (4.0 / (3.0 * np.pi)) * np.cos(2 * np.pi * t) * np.sinc(4 * t / 3) + (
            8.0 / (3.0 * np.pi)
        ) * np.sin(2 * np.pi * t) * np.sinc(8 * t / 3)
        norm = np.linalg.norm(wavelet)
        return wavelet / norm if norm > 0 else wavelet

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
            The order of the B-spline used for the analysis filter (default is 2).
        q : int, optional
            The order of the B-spline used for the synthesis filter (default is 2).

        Returns
        -------
        wavelet : numpy.ndarray
            The Biorthogonal wavelet coefficients (analysis side).

        Examples
        --------
        >>> biorthogonal_wavelet = Wavelet.biorthogonal(N=6, p=2, q=2)
        >>> print(biorthogonal_wavelet)
        """
        # Biorthogonal spline wavelet (analysis): alternating-sign B-spline of order p+1
        # sampled at N points, centred at zero, modulated by (-1)^n for wavelet oscillation.
        t = np.arange(-N // 2, N - N // 2, dtype=np.float64)

        # B-spline basis of order p evaluated via the box-function self-convolution formula
        # For a simple closed-form: use the central-difference of the B-spline scaling function
        # psi(t) = sum_k (-1)^k C(p+1,k) phi_{p+1}(t - k) — approximate with sinc product
        def bspline(x, order):
            # B-spline of given order via repeated sinc product in frequency: sample in time domain
            # B_n(x) ≈ (sin(pi*x/order)/(pi*x/order))^order  (continuous approximation)
            with np.errstate(invalid="ignore", divide="ignore"):
                val = np.where(
                    np.abs(x) < 1e-10,
                    1.0,
                    (np.sin(np.pi * x / order) / (np.pi * x / order)) ** order,
                )
            return val

        wavelet = ((-1.0) ** np.arange(N)) * bspline(t, p + 1) * bspline(t - 0.5, q + 1)
        norm = np.linalg.norm(wavelet)
        return wavelet / norm if norm > 0 else wavelet

    @staticmethod
    def reverse_biorthogonal(N=6, p=2, q=2):
        """
        Generate a Reverse Biorthogonal wavelet.

        Reverse Biorthogonal wavelets are the synthesis duals of biorthogonal wavelets,
        useful in de-noising applications.

        Parameters
        ----------
        N : int, optional
            The number of points in the wavelet (default is 6).
        p : int, optional
            The order of the B-spline (analysis side, default is 2).
        q : int, optional
            The order of the B-spline (synthesis side, default is 2).

        Returns
        -------
        wavelet : numpy.ndarray
            The Reverse Biorthogonal wavelet coefficients (synthesis side).

        Examples
        --------
        >>> reverse_biorthogonal_wavelet = Wavelet.reverse_biorthogonal(N=6, p=2, q=2)
        >>> print(reverse_biorthogonal_wavelet)
        """
        # Synthesis dual: swap p and q roles relative to biorthogonal
        return Wavelet.biorthogonal(N=N, p=q, q=p)

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
        # Shannon wavelet: psi(t) = sinc(t/2)*cos(3*pi*t/2)  (bandpass, centred at 3/4 normalised freq)
        # np.sinc uses the normalised definition: sinc(x) = sin(pi*x)/(pi*x)
        t = np.arange(-N // 2, N - N // 2, dtype=np.float64)
        wavelet = np.sinc(t / 2.0) * np.cos(3.0 * np.pi * t / 2.0)
        norm = np.linalg.norm(wavelet)
        return wavelet / norm if norm > 0 else wavelet

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
            The bandwidth parameter (default is 0.5).

        Returns
        -------
        wavelet : numpy.ndarray
            The Frequency B-Spline wavelet coefficients.

        Examples
        --------
        >>> fbsp_wavelet = Wavelet.fbsp(N=6, m=5, s=0.5)
        >>> print(fbsp_wavelet)
        """
        # fbsp wavelet: psi(t) = sqrt(s) * sinc(s*t/m)^m * exp(j*2*pi*t)  (take real part)
        # Real-valued version: modulated B-spline envelope
        # sinc here is normalised: np.sinc(x) = sin(pi*x)/(pi*x)
        t = np.arange(-N // 2, N - N // 2, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            envelope = np.where(np.abs(t) < 1e-10, 1.0, np.sinc(s * t / m) ** m)
        wavelet = np.sqrt(s) * envelope * np.cos(2.0 * np.pi * t)
        norm = np.linalg.norm(wavelet)
        return wavelet / norm if norm > 0 else wavelet

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
