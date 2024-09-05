from scipy.signal import coherence
from vitalDSP.preprocess.preprocess import preprocess_signal
from vitalDSP.respiratory_analysis.respiratory_analysis import PreprocessConfig


class CoherenceAnalysis:
    """
    A class for performing coherence analysis between two physiological signals
    (e.g., ECG, PPG, respiratory signals), with built-in preprocessing to handle noise,
    delay, and signal lag.

    Attributes
    ----------
    signal1 : numpy.ndarray
        The first signal to analyze (e.g., ECG).
    signal2 : numpy.ndarray
        The second signal to analyze (e.g., PPG).
    fs : int
        The sampling frequency of the signals in Hz.
    """

    def __init__(self, signal1, signal2, fs=1000):
        """
        Initializes the CoherenceAnalysis object.

        Parameters
        ----------
        signal1 : numpy.ndarray
            The first signal to analyze (e.g., ECG).
        signal2 : numpy.ndarray
            The second signal to analyze (e.g., PPG).
        fs : int, optional
            The sampling frequency of the signals in Hz. Default is 1000 Hz.
        """
        self.signal1 = signal1
        self.signal2 = signal2
        self.fs = fs

    def preprocess_signals(self, preprocess_config1=None, preprocess_config2=None):
        """
        Preprocesses the input signals using filtering, noise reduction, and delay compensation.

        Parameters
        ----------
        preprocess_config1 : PreprocessConfig, optional
            Configuration for preprocessing the first signal (e.g., ECG).
        preprocess_config2 : PreprocessConfig, optional
            Configuration for preprocessing the second signal (e.g., PPG).

        Returns
        -------
        preprocessed_signal1 : numpy.ndarray
            The preprocessed first signal.
        preprocessed_signal2 : numpy.ndarray
            The preprocessed second signal.

        Examples
        --------
        >>> ca = CoherenceAnalysis(signal1, signal2, fs=1000)
        >>> preprocessed_signal1, preprocessed_signal2 = ca.preprocess_signals(preprocess_config1, preprocess_config2)
        """
        if preprocess_config1 is None:
            preprocess_config1 = PreprocessConfig()

        if preprocess_config2 is None:
            preprocess_config2 = PreprocessConfig()

        # Preprocess the two signals to eliminate noise, apply filtering, and delay compensation
        preprocessed_signal1 = preprocess_signal(
            signal=self.signal1,
            sampling_rate=self.fs,
            filter_type=preprocess_config1.filter_type,
            noise_reduction_method=preprocess_config1.noise_reduction_method,
            lowcut=preprocess_config1.lowcut,
            highcut=preprocess_config1.highcut,
            order=preprocess_config1.order,
            wavelet_name=preprocess_config1.wavelet_name,
            level=preprocess_config1.level,
            window_length=preprocess_config1.window_length,
            polyorder=preprocess_config1.polyorder,
            kernel_size=preprocess_config1.kernel_size,
            sigma=preprocess_config1.sigma,
            respiratory_mode=False,
        )

        preprocessed_signal2 = preprocess_signal(
            signal=self.signal2,
            sampling_rate=self.fs,
            filter_type=preprocess_config2.filter_type,
            noise_reduction_method=preprocess_config2.noise_reduction_method,
            lowcut=preprocess_config2.lowcut,
            highcut=preprocess_config2.highcut,
            order=preprocess_config2.order,
            wavelet_name=preprocess_config2.wavelet_name,
            level=preprocess_config2.level,
            window_length=preprocess_config2.window_length,
            polyorder=preprocess_config2.polyorder,
            kernel_size=preprocess_config2.kernel_size,
            sigma=preprocess_config2.sigma,
            respiratory_mode=False,
        )

        return preprocessed_signal1, preprocessed_signal2

    def align_signals(self, signal1, signal2):
        """
        Aligns the two signals by compensating for delay or lag using cross-correlation.

        Parameters
        ----------
        signal1 : numpy.ndarray
            The first signal (preprocessed).
        signal2 : numpy.ndarray
            The second signal (preprocessed).

        Returns
        -------
        aligned_signal1 : numpy.ndarray
            The aligned first signal.
        aligned_signal2 : numpy.ndarray
            The aligned second signal.

        Examples
        --------
        >>> ca = CoherenceAnalysis(signal1, signal2, fs=1000)
        >>> aligned_signal1, aligned_signal2 = ca.align_signals(preprocessed_signal1, preprocessed_signal2)
        """
        from scipy.signal import correlate

        # Cross-correlate the signals to find the delay
        correlation = correlate(signal1, signal2)
        delay = correlation.argmax() - (len(signal1) - 1)

        # Adjust the signals based on the detected delay
        if delay > 0:
            aligned_signal1 = signal1[delay:]
            aligned_signal2 = signal2[: len(signal2) - delay]
        else:
            aligned_signal1 = signal1[: len(signal1) + delay]
            aligned_signal2 = signal2[-delay:]

        return aligned_signal1, aligned_signal2

    def compute_coherence(
        self, preprocess_config1=None, preprocess_config2=None, nperseg=256
    ):
        """
        Computes the coherence between two signals after preprocessing and alignment.

        Parameters
        ----------
        preprocess_config1 : PreprocessConfig, optional
            Preprocessing configuration for the first signal.
        preprocess_config2 : PreprocessConfig, optional
            Preprocessing configuration for the second signal.
        nperseg : int, optional
            Length of each segment for computing the coherence. Default is 256.

        Returns
        -------
        f : numpy.ndarray
            Array of sample frequencies.
        Cxy : numpy.ndarray
            Coherence between the two signals, ranging from 0 to 1.

        Examples
        --------
        >>> signal1 = np.sin(np.linspace(0, 10, 1000)) + np.random.normal(0, 0.2, 1000)
        >>> signal2 = np.sin(np.linspace(0, 10, 1000) + 0.1) + np.random.normal(0, 0.2, 1000)
        >>> ca = CoherenceAnalysis(signal1, signal2, fs=1000)
        >>> f, Cxy = ca.compute_coherence(nperseg=256)
        >>> print(f"Frequencies: {f}")
        >>> print(f"Coherence: {Cxy}")
        """
        # Preprocess and align the signals
        preprocessed_signal1, preprocessed_signal2 = self.preprocess_signals(
            preprocess_config1, preprocess_config2
        )
        aligned_signal1, aligned_signal2 = self.align_signals(
            preprocessed_signal1, preprocessed_signal2
        )

        # Compute coherence between aligned signals
        f, Cxy = coherence(
            aligned_signal1, aligned_signal2, fs=self.fs, nperseg=nperseg
        )
        return f, Cxy

    def plot_coherence(self, f, Cxy):
        """
        Plots the coherence between two signals.

        Parameters
        ----------
        f : numpy.ndarray
            Array of sample frequencies.
        Cxy : numpy.ndarray
            Coherence between the two signals.

        Examples
        --------
        >>> signal1 = np.sin(np.linspace(0, 10, 1000)) + np.random.normal(0, 0.2, 1000)
        >>> signal2 = np.sin(np.linspace(0, 10, 1000) + 0.1) + np.random.normal(0, 0.2, 1000)
        >>> ca = CoherenceAnalysis(signal1, signal2, fs=1000)
        >>> f, Cxy = ca.compute_coherence(nperseg=256)
        >>> ca.plot_coherence(f, Cxy)
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.semilogy(f, Cxy)
        plt.title("Coherence between two signals")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Coherence")
        plt.grid()
        plt.show()
