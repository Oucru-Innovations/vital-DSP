import numpy as np

# from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from scipy.spatial import distance as sp_distance

class NonlinearFeatures:
    """
    A class for computing nonlinear (geometric) features from physiological signals (ECG, PPG, EEG).

    Attributes
    ----------
    signal : np.array
        The physiological signal (e.g., ECG, PPG, EEG).
    fs : int
        The sampling frequency of the signal in Hz. Default is 1000 Hz.

    Methods
    -------
    compute_sample_entropy(m=2, r=0.2)
        Computes the sample entropy of the signal, measuring its complexity.
    compute_approximate_entropy(m=2, r=0.2)
        Computes the approximate entropy of the signal, quantifying its unpredictability.
    compute_fractal_dimension(kmax=10)
        Computes the fractal dimension of the signal using Higuchi's method.
    compute_lyapunov_exponent()
        Computes the largest Lyapunov exponent, indicating the presence of chaos in the signal.
    compute_dfa(order=1)
        Computes the detrended fluctuation analysis (DFA) for assessing fractal scaling.
    compute_poincare_features()
        Computes Poincaré plot features (SD1 and SD2) to assess short- and long-term HRV variability.
    compute_recurrence_features(threshold=0.2)
        Computes features from the recurrence plot, including recurrence rate, determinism, and laminarity.
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

        signal = np.asarray(self.signal)
        N = len(signal)

        if np.all(signal == 0) or np.std(signal) == 0:
            return 0  # Return 0 for constant or zero signals
        if N < m + 1:
            return 0  # Return 0 for signals too short for meaningful entropy

        # Normalize the signal to have zero mean and unit variance
        signal = (signal - np.mean(signal)) / np.std(signal)
        r *= np.std(signal)  # Scale tolerance; std=1 after normalization, so unchanged

        # Create embedded vectors for dimensions m and m+1
        embedded_m = np.array([signal[i : i + m] for i in range(N - m + 1)])
        embedded_m1 = np.array([signal[i : i + m + 1] for i in range(N - m)])

        def _phi(embedded, tol):
            n = len(embedded)
            if n <= 1:
                return 0.0
            tree = KDTree(embedded)
            # Use <= tol; adjust tol slightly if strict < is needed, but for practicality, use <=
            counts = tree.query_ball_point(embedded, r=tol, p=np.inf, return_length=True)
            total_double = np.sum(counts) - n  # Subtract self-matches
            num_pairs = n * (n - 1) / 2.0
            return total_double / (2.0 * num_pairs) if num_pairs > 0 else 0.0

        phi_m = _phi(embedded_m, r)
        phi_m1 = _phi(embedded_m1, r)

        if phi_m == 0 or phi_m1 == 0:
            return 0  # Avoid log of zero

        return -np.log(phi_m1 / phi_m)

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

        signal = np.asarray(self.signal)
        N = len(signal)

        if np.all(signal == 0) or np.std(signal) == 0:
            return 0  # Return 0 for constant or zero signals
        if N <= m + 1:
            return 0  # Return 0 for signals too short for meaningful entropy

        # Normalize the signal to have zero mean and unit variance
        signal = (signal - np.mean(signal)) / np.std(signal)
        r *= np.std(signal)  # Scale tolerance to the signal's standard deviation

        def _phi(m):
            # Create embedded vectors for dimension m
            embedded = np.array([signal[i : i + m] for i in range(N - m + 1)])
            # Build a KDTree for efficient neighbor searches
            tree = KDTree(embedded)

            # Query neighbors within radius r using Chebyshev (max) distance
            counts = tree.query_ball_point(embedded, r, p=np.inf)
            # Compute C_i values
            C = np.array([len(c) / (N - m + 1) for c in counts])

            # Handle zero counts to avoid log(0)
            C = np.where(C == 0, np.finfo(float).eps, C)

            # Compute phi
            phi = np.sum(np.log(C)) / (N - m + 1)
            return phi

        # Compute phi for m and m+1
        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)

        # Approximate entropy is the difference between the two phi values
        return phi_m - phi_m1

    def compute_fractal_dimension(self, kmax=10):
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

        if len(self.signal) < kmax:
            return 0  # Return 0 for signals too short for the given kmax

        def _higuchi_fd(signal, kmax):
            Lmk = np.zeros((kmax, kmax))
            N = len(signal)
            for k in range(1, kmax + 1):
                for m in range(0, k):
                    Lm = 0
                    for i in range(1, int((N - m) / k)):
                        Lm += abs(signal[m + i * k] - signal[m + (i - 1) * k])
                    if int((N - m) / k) == 0:
                        return 0
                    Lmk[m, k - 1] = Lm * (N - 1) / ((int((N - m) / k) * k * k))

            Lk = np.sum(Lmk, axis=0) / kmax
            log_range = np.log(np.arange(1, kmax + 1))
            if np.any(Lk == 0):
                return 0  # Return 0 to avoid division by zero in polyfit
            return -np.polyfit(log_range, np.log(Lk), 1)[0]

        return _higuchi_fd(self.signal, kmax)

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
        if N < 3:
            return 0  # Not enough data for meaningful computation

        epsilon = np.std(self.signal) * 0.1
        max_t = min(
            int(N / 10), N - 3
        )  # Ensure max_t is not larger than the length of the phase space

        def _distance(x, y):
            return np.sqrt(np.sum((x - y) ** 2))

        def _lyapunov(time_delay, dim, max_t):
            if max_t <= 1:
                return 0  # Prevent division errors with too short signals
            phase_space = np.array([self.signal[i::time_delay] for i in range(dim)]).T

            divergences = []
            for i in range(len(phase_space) - max_t - 1):
                d0 = _distance(phase_space[i], phase_space[i + 1])
                d1 = _distance(phase_space[i + max_t], phase_space[i + max_t + 1])
                if d0 > epsilon and d1 > epsilon:
                    divergences.append(np.log(d1 / d0))

            if len(divergences) == 0:
                return 0  # Return 0 if no valid divergences were found
            return np.mean(divergences)

        return _lyapunov(time_delay=5, dim=2, max_t=max_t)

    def compute_dfa(self, order=1):
        """
        Computes the Detrended Fluctuation Analysis (DFA) of the signal. DFA is used to assess
        the fractal scaling properties of time-series data, especially in physiological signals.

        Args:
            order (int): The order of the polynomial fit for detrending. Default is 1 (linear detrending).

        Returns:
            float: The DFA scaling exponent (α).

        Example:
            >>> ecg_signal = [...]  # Sample ECG signal
            >>> nf = NonlinearFeatures(ecg_signal)
            >>> dfa = nf.compute_dfa(order=1)
            >>> print(f"DFA Scaling Exponent: {dfa}")
        """
        import numpy as np

        signal = np.asarray(self.signal)
        N = len(signal)
        if N < 4:
            return 0  # Not enough data for DFA computation

        # Step 1: Compute the integrated signal
        integrated_signal = np.cumsum(signal - np.mean(signal))

        # Step 2: Define scales (segment lengths)
        scales = np.unique(
            np.floor(np.logspace(np.log10(4), np.log10(N // 4), num=20))
        ).astype(int)
        fluctuation_sizes = []

        for scale in scales:
            # Number of segments
            n_segments = N // scale
            if n_segments < 2:
                continue  # Need at least 2 segments for reliable estimation

            # Truncate the integrated signal to make it divisible by scale
            truncated_length = n_segments * scale
            integrated_truncated = integrated_signal[:truncated_length]

            # Reshape into segments
            segments = integrated_truncated.reshape((n_segments, scale))

            # Create the time vector for polynomial fitting
            x = np.arange(scale)

            if order == 1:
                # Linear detrending can be vectorized
                X = np.vstack([x, np.ones_like(x)]).T  # Design matrix
                # Precompute pseudoinverse of X
                XtX_inv_Xt = np.linalg.pinv(X)
                # Compute coefficients for all segments
                coeffs = XtX_inv_Xt @ segments.T  # Shape: (2, n_segments)
                # Compute trends
                trends = X @ coeffs  # Shape: (scale, n_segments)
                trends = trends.T  # Shape: (n_segments, scale)
            else:
                # For higher-order polynomials, process each segment individually
                trends = np.zeros_like(segments)
                for i in range(n_segments):
                    coeffs = np.polyfit(x, segments[i], order)
                    trends[i] = np.polyval(coeffs, x)

            # Compute fluctuations
            residuals = segments - trends
            rms = np.sqrt(np.mean(residuals**2, axis=1))
            # Append the mean fluctuation for this scale
            fluctuation_sizes.append(np.mean(rms))

        # Convert scales and fluctuation sizes to numpy arrays
        fluctuation_sizes = np.array(fluctuation_sizes)
        scales = scales[: len(fluctuation_sizes)]

        # Logarithms of scales and fluctuation sizes
        log_scales = np.log(scales)
        log_fluctuation_sizes = np.log(fluctuation_sizes)

        # Linear regression to find the scaling exponent (alpha)
        dfa_alpha = np.polyfit(log_scales, log_fluctuation_sizes, 1)[0]
        return dfa_alpha

    def compute_poincare_features(self, nn_intervals):
        """
        Computes the SD1 and SD2 features from the Poincaré plot of the NN intervals. SD1 reflects
        short-term HRV, while SD2 reflects long-term HRV.

        Returns:
            tuple: SD1 (short-term HRV), SD2 (long-term HRV).

        Example:
            >>> nf = NonlinearFeatures(signal)
            >>> nn_intervals = [800, 810, 790, 805, 795]
            >>> sd1, sd2 = nf.compute_poincare_features(nn_intervals=nn_intervals)
            >>> print(f"SD1: {sd1}, SD2: {sd2}")
        """
        nn_intervals = np.asarray(nn_intervals)
        x1 = nn_intervals[:-1]
        x2 = nn_intervals[1:]

        # Compute the differences and sums
        diff = x2 - x1
        # summ = x2 + x1

        # Compute variances
        var_diff = np.var(diff, ddof=1)
        var_nn = np.var(nn_intervals, ddof=1)

        # Calculate SD1 and SD2
        sd1 = np.sqrt(var_diff / 2)
        sd2 = np.sqrt(2 * var_nn - var_diff / 2)

        return sd1, sd2

    def compute_recurrence_features(self, threshold=0.2, sample_size=10000):
        """
        Computes approximate recurrence features by sampling point pairs. This approach significantly
        reduces computation time for large datasets by avoiding the full pairwise distance calculations.

        Args:
            threshold (float): The threshold to define recurrences. Default is 0.2.
            sample_size (int): The number of point pairs to sample. Default is 10,000.

        Returns:
            dict: A dictionary containing approximate recurrence rate, determinism, and laminarity.

        Example:
            >>> ecg_signal = [...]  # Sample ECG signal
            >>> nf = NonlinearFeatures(ecg_signal)
            >>> rqa_features = nf.compute_recurrence_features(threshold=0.2, sample_size=10000)
            >>> print(rqa_features)
        """
        signal = np.asarray(self.signal)
        N = len(signal)

        if N < 2:
            return {
                "recurrence_rate": 0,
                "determinism": 0,
                "laminarity": 0,
            }

        # Normalize the signal to zero mean and unit variance
        signal = (signal - np.mean(signal)) / np.std(signal)

        # Create phase space (embedding dimension = 2)
        phase_space = np.column_stack((signal[:-1], signal[1:]))
        M = len(phase_space)

        # Total number of possible pairs (excluding self-pairs)
        total_pairs = M * (M - 1) // 2

        # Adjust sample size if necessary
        sample_size = min(sample_size, total_pairs)

        # Sample random indices without replacement
        idx_pairs = np.random.choice(total_pairs, size=sample_size, replace=False)
        idx1 = idx_pairs // M
        idx2 = idx_pairs % M

        # Ensure idx1 < idx2 to avoid duplicate pairs
        mask = idx1 < idx2
        idx1 = idx1[mask]
        idx2 = idx2[mask]
        sample_size = len(idx1)

        # Compute distances between sampled pairs
        diffs = phase_space[idx1] - phase_space[idx2]
        distances = np.sqrt(np.sum(diffs**2, axis=1))

        # Compute approximate recurrence rate
        recurrences = distances < threshold
        recurrence_rate = np.sum(recurrences) / sample_size

        # Approximate determinism and laminarity
        # Sort indices to detect line structures
        sorted_indices = np.argsort(idx1)
        idx1_sorted = idx1[sorted_indices]
        idx2_sorted = idx2[sorted_indices]
        recurrences_sorted = recurrences[sorted_indices]

        # Initialize variables for line detection
        diag_lengths = []
        vert_lengths = []
        current_diag_len = 1
        current_vert_len = 1

        for i in range(1, sample_size):
            if recurrences_sorted[i]:
                # Check for diagonal lines (idx1 and idx2 increase by 1)
                if (
                    idx1_sorted[i] - idx1_sorted[i - 1] == 1
                    and idx2_sorted[i] - idx2_sorted[i - 1] == 1
                ):
                    current_diag_len += 1
                else:
                    if current_diag_len > 1:
                        diag_lengths.append(current_diag_len)
                    current_diag_len = 1

                # Check for vertical lines (idx1 increases by 1, idx2 same)
                if (
                    idx1_sorted[i] - idx1_sorted[i - 1] == 1
                    and idx2_sorted[i] == idx2_sorted[i - 1]
                ):
                    current_vert_len += 1
                else:
                    if current_vert_len > 1:
                        vert_lengths.append(current_vert_len)
                    current_vert_len = 1
            else:
                if current_diag_len > 1:
                    diag_lengths.append(current_diag_len)
                current_diag_len = 1
                if current_vert_len > 1:
                    vert_lengths.append(current_vert_len)
                current_vert_len = 1

        # Append the last lengths if needed
        if current_diag_len > 1:
            diag_lengths.append(current_diag_len)
        if current_vert_len > 1:
            vert_lengths.append(current_vert_len)

        # Calculate determinism
        if diag_lengths:
            det = np.sum(diag_lengths) / np.sum(recurrences)
        else:
            det = 0

        # Calculate laminarity
        if vert_lengths:
            laminarity = np.sum(vert_lengths) / np.sum(recurrences)
        else:
            laminarity = 0

        return {
            "recurrence_rate": recurrence_rate,
            "determinism": det,
            "laminarity": laminarity,
        }

import pandas as pd
import os
if __name__ == "__main__":
    ppg_signal = np.random.rand(1000)
    fname = '20190109T151032.026+0700_1050000_1080000.csv'
    PATH = 'D:\Workspace\Data\\24EIa\output\sample'
    
    ppg_signal = pd.read_csv(os.path.join(PATH, fname))['PLETH'].values
    fs = 100
    features = NonlinearFeatures(ppg_signal, fs)
    sample_entropy = features.compute_sample_entropy()
    approximate_entropy = features.compute_approximate_entropy()
    fractal_dimension = features.compute_fractal_dimension()
    lyapunov_exponent = features.compute_lyapunov_exponent()
    dfa = features.compute_dfa()
    poincare_features = features.compute_poincare_features()
    recurrence_features = features.compute_recurrence_features()
    print(f"Sample Entropy: {sample_entropy}, Approximate Entropy: {approximate_entropy}, Fractal Dimension: {fractal_dimension}, Lyapunov Exponent: {lyapunov_exponent}, DFA: {dfa}, Poincaré Features: {poincare_features}, Recurrence Features: {recurrence_features}")