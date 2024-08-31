import numpy as np


class NonlinearFeatures:
    """
    A class for computing nonlinear features from physiological signals (ECG, PPG, EEG).

    Attributes:
        signal (np.array): The physiological signal (ECG, PPG, EEG).
        fs (int): The sampling frequency of the signal in Hz. Default is 1000 Hz.
    """

    def __init__(self, signal, fs=1000):
        """
        Initializes the NonlinearFeatures object.

        Args:
            signal (np.array): The physiological signal.
            fs (int): The sampling frequency of the signal in Hz. Default is 1000 Hz.
        """
        self.signal = np.array(signal)
        self.fs = fs  # Sampling frequency

    def compute_sample_entropy(self, m=2, r=0.2):
        """
        Computes the sample entropy of the signal. Sample entropy is a measure of signal complexity,
        specifically used for detecting the regularity and unpredictability of fluctuations in a signal.

        Args:
            m (int): Embedding dimension (default is 2).
            r (float): Tolerance (default is 0.2).

        Returns:
            float: The computed sample entropy of the signal.

        Example:
            >>> ecg_signal = [...]  # Sample ECG signal
            >>> nf = NonlinearFeatures(ecg_signal)
            >>> sample_entropy = nf.compute_sample_entropy()
            >>> print(f"Sample Entropy: {sample_entropy}")
        """
        N = len(self.signal)

        def _phi(m):
            x = np.array([self.signal[i : i + m] for i in range(N - m + 1)])
            C = np.sum(
                [
                    np.sum(np.abs(x[i] - x[j]) < r)
                    for i in range(len(x))
                    for j in range(len(x))
                    if i != j
                ]
            )
            return C / ((N - m + 1) ** 2 - (N - m + 1))

        return -np.log(_phi(m + 1) / _phi(m))

    def compute_approximate_entropy(self, m=2, r=0.2):
        """
        Computes the approximate entropy of the signal. Approximate entropy quantifies the
        unpredictability and regularity of signal patterns.

        Args:
            m (int): Embedding dimension (default is 2).
            r (float): Tolerance (default is 0.2).

        Returns:
            float: The computed approximate entropy of the signal.

        Example:
            >>> ppg_signal = [...]  # Sample PPG signal
            >>> nf = NonlinearFeatures(ppg_signal)
            >>> approx_entropy = nf.compute_approximate_entropy()
            >>> print(f"Approximate Entropy: {approx_entropy}")
        """

        def _phi(m):
            x = np.array(
                [self.signal[i : i + m] for i in range(len(self.signal) - m + 1)]
            )
            C = np.sum(
                [
                    np.max(np.abs(x[i] - x[j])) < r
                    for i in range(len(x))
                    for j in range(len(x))
                    if i != j
                ]
            )
            return C / (len(self.signal) - m + 1)

        return _phi(m) - _phi(m + 1)

    def compute_fractal_dimension(self):
        """
        Computes the fractal dimension of the signal using Higuchi's method. Fractal dimension
        is a measure of complexity, reflecting how the signal fills space as its scale changes.

        Returns:
            float: The fractal dimension of the signal.

        Example:
            >>> eeg_signal = [...]  # Sample EEG signal
            >>> nf = NonlinearFeatures(eeg_signal)
            >>> fractal_dimension = nf.compute_fractal_dimension()
            >>> print(f"Fractal Dimension: {fractal_dimension}")
        """

        def _higuchi_fd(signal, kmax):
            Lmk = np.zeros((kmax, kmax))
            N = len(signal)
            for k in range(1, kmax + 1):
                for m in range(0, k):
                    Lm = 0
                    for i in range(1, int((N - m) / k)):
                        Lm += abs(signal[m + i * k] - signal[m + (i - 1) * k])
                    Lmk[m, k - 1] = Lm * (N - 1) / ((int((N - m) / k) * k * k))

            Lk = np.sum(Lmk, axis=0) / kmax
            return -np.polyfit(np.log(range(1, kmax + 1)), np.log(Lk), 1)[0]

        return _higuchi_fd(self.signal, kmax=10)

    def compute_lyapunov_exponent(self):
        """
        Computes the largest Lyapunov exponent (LLE) of the signal. LLE measures the rate at
        which nearby trajectories in phase space diverge, indicating chaotic behavior in the signal.

        Returns:
            float: The largest Lyapunov exponent of the signal.

        Example:
            >>> ecg_signal = [...]  # Sample ECG signal
            >>> nf = NonlinearFeatures(ecg_signal)
            >>> lyapunov_exponent = nf.compute_lyapunov_exponent()
            >>> print(f"Largest Lyapunov Exponent: {lyapunov_exponent}")
        """
        N = len(self.signal)
        epsilon = np.std(self.signal) * 0.1
        max_t = int(N / 10)

        def _distance(x, y):
            return np.sqrt(np.sum((x - y) ** 2))

        def _lyapunov(time_delay, dim, max_t):
            phase_space = np.array([self.signal[i::time_delay] for i in range(dim)]).T
            divergences = []
            for i in range(len(phase_space) - max_t):
                d0 = _distance(phase_space[i], phase_space[i + 1])
                d1 = _distance(phase_space[i + max_t], phase_space[i + max_t + 1])
                if d0 > epsilon and d1 > epsilon:
                    divergences.append(np.log(d1 / d0))
            return np.mean(divergences)

        return _lyapunov(time_delay=5, dim=2, max_t=max_t)
