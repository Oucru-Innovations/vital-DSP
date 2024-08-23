import numpy as np
import matplotlib.pyplot as plt

class NonlinearAnalysis:
    """
    Nonlinear Analysis for analyzing chaotic signals like Heart Rate Variability (HRV).

    Methods:
    - lyapunov_exponent: Estimates the largest Lyapunov exponent to assess chaos in the signal.
    - poincare_plot: Generates a Poincaré plot to visualize signal dynamics.
    - correlation_dimension: Estimates the correlation dimension of the signal.

    Example Usage:
    --------------
    signal = np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.normal(size=100)
    nonlinear_analysis = NonlinearAnalysis(signal)
    lyapunov = nonlinear_analysis.lyapunov_exponent()
    print("Lyapunov Exponent:", lyapunov)

    nonlinear_analysis.poincare_plot()
    correlation_dim = nonlinear_analysis.correlation_dimension()
    print("Correlation Dimension:", correlation_dim)
    """

    def __init__(self, signal):
        self.signal = signal

    def lyapunov_exponent(self, max_iter=1000, epsilon=1e-8):
        """
        Estimate the largest Lyapunov exponent of the signal.

        Parameters:
        max_iter (int): Maximum number of iterations.
        epsilon (float): Small perturbation used to calculate divergence.

        Returns:
        float: The largest Lyapunov exponent.
        """
        n = len(self.signal)
        distances = []
        for i in range(n - max_iter):
            dist = np.abs(self.signal[i + 1:i + max_iter] - self.signal[i])
            dist[dist < epsilon] = epsilon  # Avoid log(0)
            distances.append(np.log(np.abs(dist / dist[0])))

        lyapunov = np.mean(distances) / max_iter
        return lyapunov

    def poincare_plot(self):
        """
        Generate a Poincaré plot to visualize the dynamics of the signal.

        Returns:
        matplotlib.figure.Figure: The generated Poincaré plot.
        """
        

        x = self.signal[:-1]
        y = self.signal[1:]

        plt.scatter(x, y)
        plt.xlabel(r'$x_{n}$')
        plt.ylabel(r'$x_{n+1}$')
        plt.title("Poincaré Plot")
        plt.show()

    def correlation_dimension(self, radius=0.1):
        """
        Estimate the correlation dimension of the signal using the Grassberger-Procaccia method.

        Parameters:
        radius (float): Radius within which points are considered neighbors.

        Returns:
        float: The estimated correlation dimension.
        """
        n = len(self.signal)
        count = 0

        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(self.signal[i] - self.signal[j]) < radius:
                    count += 1

        correlation_dim = np.log(count) / np.log(radius)
        return correlation_dim
