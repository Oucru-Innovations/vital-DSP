import numpy as np
import matplotlib.pyplot as plt


class NonlinearAnalysis:
    """
    Nonlinear Analysis for examining chaotic signals, such as Heart Rate Variability (HRV).

    This class provides methods for analyzing the chaotic behavior of signals using techniques
    like the estimation of Lyapunov exponents, generation of Poincaré plots, and calculation of
    correlation dimensions.

    Methods
    -------
    lyapunov_exponent(max_iter=1000, epsilon=1e-8)
        Estimates the largest Lyapunov exponent to assess chaos in the signal.
    poincare_plot()
        Generates a Poincaré plot to visualize the dynamics of the signal.
    correlation_dimension(radius=0.1)
        Estimates the correlation dimension of the signal.

    Example Usage
    -------------
    >>> signal = np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.normal(size=100)
    >>> nonlinear_analysis = NonlinearAnalysis(signal)
    >>> lyapunov = nonlinear_analysis.lyapunov_exponent()
    >>> print("Lyapunov Exponent:", lyapunov)

    >>> nonlinear_analysis.poincare_plot()

    >>> correlation_dim = nonlinear_analysis.correlation_dimension()
    >>> print("Correlation Dimension:", correlation_dim)
    """

    def __init__(self, signal):
        """
        Initialize the NonlinearAnalysis class with the provided signal.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal to be analyzed for nonlinear dynamics.
        """
        self.signal = signal

    def lyapunov_exponent(self, max_iter=1000, epsilon=1e-8):
        """
        Estimate the largest Lyapunov exponent of the signal.

        The Lyapunov exponent is a measure of the rate of separation of infinitesimally
        close trajectories in a chaotic system. A positive Lyapunov exponent indicates chaos.

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of iterations to compute the exponent (default is 1000).
        epsilon : float, optional
            Small perturbation used to calculate divergence, to avoid division by zero (default is 1e-8).

        Returns
        -------
        float
            The estimated largest Lyapunov exponent.

        Examples
        --------
        >>> signal = np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.normal(size=100)
        >>> nonlinear_analysis = NonlinearAnalysis(signal)
        >>> lyapunov = nonlinear_analysis.lyapunov_exponent()
        >>> print("Lyapunov Exponent:", lyapunov)
        """
        n = len(self.signal)
        distances = []
        for i in range(n - max_iter):
            dist = np.abs(self.signal[i + 1 : i + max_iter] - self.signal[i])
            dist[dist < epsilon] = epsilon  # Avoid log(0)
            distances.append(np.log(np.abs(dist / dist[0])))

        lyapunov = np.mean(distances) / max_iter
        return lyapunov

    def poincare_plot(self):
        """
        Generate a Poincaré plot to visualize the dynamics of the signal.

        The Poincaré plot is a scatter plot of the signal against its delayed version.
        It is often used to visualize periodic and chaotic dynamics in the signal.

        Returns
        -------
        matplotlib.figure.Figure
            The generated Poincaré plot.

        Examples
        --------
        >>> signal = np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.normal(size=100)
        >>> nonlinear_analysis = NonlinearAnalysis(signal)
        >>> nonlinear_analysis.poincare_plot()
        """
        x = self.signal[:-1]
        y = self.signal[1:]

        plt.scatter(x, y)
        plt.xlabel(r"$x_{n}$")
        plt.ylabel(r"$x_{n+1}$")
        plt.title("Poincaré Plot")
        plt.show()

    def correlation_dimension(self, radius=0.1):
        """
        Estimate the correlation dimension of the signal using the Grassberger-Procaccia method.

        The correlation dimension is a measure of the fractal dimension of the attractor
        in the phase space of the signal. It provides insight into the complexity of the signal.

        Parameters
        ----------
        radius : float, optional
            Radius within which points are considered neighbors (default is 0.1).

        Returns
        -------
        float
            The estimated correlation dimension.

        Examples
        --------
        >>> signal = np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.normal(size=100)
        >>> nonlinear_analysis = NonlinearAnalysis(signal)
        >>> correlation_dim = nonlinear_analysis.correlation_dimension()
        >>> print("Correlation Dimension:", correlation_dim)
        """
        n = len(self.signal)
        count = 0

        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(self.signal[i] - self.signal[j]) < radius:
                    count += 1

        correlation_dim = np.log(count) / np.log(radius)
        return correlation_dim
