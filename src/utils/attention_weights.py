import numpy as np


class AttentionWeights:
    """
    A class that provides a collection of predefined attention weights for signal processing.

    Methods:
    - uniform: Uniform attention where all parts of the signal are treated equally.
    - linear: Linearly increasing or decreasing attention.
    - gaussian: Gaussian distribution for attention weights.
    - exponential: Exponentially increasing or decreasing attention.
    - custom_weights: Accepts custom attention weights provided by the user.
    """

    @staticmethod
    def uniform(size):
        """
        Generate uniform attention weights.

        Parameters:
        size (int): Size of the attention window.

        Returns:
        numpy.ndarray: The uniform attention weights.

        Example:
        >>> weights = AttentionWeights.uniform(size=5)
        >>> print(weights)
        """
        return np.ones(size) / size

    @staticmethod
    def linear(size, ascending=True):
        """
        Generate linearly increasing or decreasing attention weights.

        Parameters:
        size (int): Size of the attention window.
        ascending (bool): If True, weights increase linearly; otherwise, they decrease.

        Returns:
        numpy.ndarray: The linear attention weights.

        Example:
        >>> weights = AttentionWeights.linear(size=5, ascending=True)
        >>> print(weights)
        """
        if ascending:
            return np.linspace(1, size, size) / np.sum(np.linspace(1, size, size))
        else:
            return np.linspace(size, 1, size) / np.sum(np.linspace(size, 1, size))

    @staticmethod
    def gaussian(size, sigma=1.0):
        """
        Generate Gaussian-distributed attention weights.

        Parameters:
        size (int): Size of the attention window.
        sigma (float): Standard deviation of the Gaussian distribution.

        Returns:
        numpy.ndarray: The Gaussian attention weights.

        Example:
        >>> weights = AttentionWeights.gaussian(size=5, sigma=1.0)
        >>> print(weights)
        """
        center = size // 2
        x = np.arange(size) - center
        weights = np.exp(-0.5 * (x / sigma) ** 2)
        return weights / np.sum(weights)

    @staticmethod
    def exponential(size, ascending=True, base=2.0):
        """
        Generate exponentially increasing or decreasing attention weights.

        Parameters:
        size (int): Size of the attention window.
        ascending (bool): If True, weights increase exponentially; otherwise, they decrease.
        base (float): The base of the exponential function.

        Returns:
        numpy.ndarray: The exponential attention weights.

        Example:
        >>> weights = AttentionWeights.exponential(size=5, ascending=True, base=2.0)
        >>> print(weights)
        """
        if ascending:
            weights = base ** np.arange(size)
        else:
            weights = base ** np.arange(size)[::-1]
        return weights / np.sum(weights)

    @staticmethod
    def custom_weights(weights):
        """
        Use custom attention weights provided by the user.

        Parameters:
        weights (numpy.ndarray): The custom attention weights.

        Returns:
        numpy.ndarray: The custom attention weights.

        Example:
        >>> weights = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        >>> custom_weights = AttentionWeights.custom_weights(weights)
        >>> print(custom_weights)
        """
        return weights / np.sum(weights)
