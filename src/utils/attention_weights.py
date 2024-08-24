import numpy as np

class AttentionWeights:
    """
    A class that provides a collection of predefined attention weights for signal processing.

    This class offers various methods to generate attention weights, which can be applied to signals in tasks such as filtering, feature extraction, and data weighting. Attention weights determine the importance of different parts of a signal, allowing for more focused analysis on specific sections.

    Methods
    -------
    uniform : static method
        Generates uniform attention where all parts of the signal are treated equally.
    linear : static method
        Generates linearly increasing or decreasing attention.
    gaussian : static method
        Generates Gaussian-distributed attention weights.
    exponential : static method
        Generates exponentially increasing or decreasing attention.
    custom_weights : static method
        Accepts custom attention weights provided by the user.
    """

    @staticmethod
    def uniform(size):
        """
        Generate uniform attention weights.

        Uniform attention weights assign equal importance to all elements in the signal, making this method suitable for scenarios where each part of the signal is equally relevant.

        Parameters
        ----------
        size : int
            The size of the attention window or the number of elements in the signal.

        Returns
        -------
        numpy.ndarray
            The uniform attention weights, with each element having equal value.

        Examples
        --------
        >>> weights = AttentionWeights.uniform(size=5)
        >>> print(weights)
        [0.2 0.2 0.2 0.2 0.2]
        """
        return np.ones(size) / size

    @staticmethod
    def linear(size, ascending=True):
        """
        Generate linearly increasing or decreasing attention weights.

        Linear attention weights are useful when a gradual change in importance across the signal is desired. This can be applied when the beginning or end of a signal is more relevant.

        Parameters
        ----------
        size : int
            The size of the attention window or the number of elements in the signal.
        ascending : bool, optional
            If True, weights increase linearly; if False, weights decrease linearly (default is True).

        Returns
        -------
        numpy.ndarray
            The linear attention weights, either increasing or decreasing based on the `ascending` parameter.

        Examples
        --------
        >>> weights = AttentionWeights.linear(size=5, ascending=True)
        >>> print(weights)
        [0.06666667 0.13333333 0.2        0.26666667 0.33333333]

        >>> weights = AttentionWeights.linear(size=5, ascending=False)
        >>> print(weights)
        [0.33333333 0.26666667 0.2        0.13333333 0.06666667]
        """
        if ascending:
            return np.linspace(1, size, size) / np.sum(np.linspace(1, size, size))
        else:
            return np.linspace(size, 1, size) / np.sum(np.linspace(size, 1, size))

    @staticmethod
    def gaussian(size, sigma=1.0):
        """
        Generate Gaussian-distributed attention weights.

        Gaussian attention weights emphasize the central parts of the signal more than the edges, following a bell-shaped curve. This method is ideal for scenarios where the middle portion of the signal is more relevant.

        Parameters
        ----------
        size : int
            The size of the attention window or the number of elements in the signal.
        sigma : float, optional
            The standard deviation of the Gaussian distribution, controlling the spread (default is 1.0).

        Returns
        -------
        numpy.ndarray
            The Gaussian attention weights, centered on the middle of the window.

        Examples
        --------
        >>> weights = AttentionWeights.gaussian(size=5, sigma=1.0)
        >>> print(weights)
        [0.05448868 0.24420134 0.40261995 0.24420134 0.05448868]
        """
        center = size // 2
        x = np.arange(size) - center
        weights = np.exp(-0.5 * (x / sigma) ** 2)
        return weights / np.sum(weights)

    @staticmethod
    def exponential(size, ascending=True, base=2.0):
        """
        Generate exponentially increasing or decreasing attention weights.

        Exponential attention weights rapidly increase or decrease, making them suitable for emphasizing certain parts of the signal much more than others. This method can be used when the importance of signal components grows or diminishes exponentially.

        Parameters
        ----------
        size : int
            The size of the attention window or the number of elements in the signal.
        ascending : bool, optional
            If True, weights increase exponentially; if False, weights decrease (default is True).
        base : float, optional
            The base of the exponential function, controlling the rate of increase or decrease (default is 2.0).

        Returns
        -------
        numpy.ndarray
            The exponential attention weights, either increasing or decreasing based on the `ascending` parameter.

        Examples
        --------
        >>> weights = AttentionWeights.exponential(size=5, ascending=True, base=2.0)
        >>> print(weights)
        [0.03125 0.0625  0.125   0.25    0.5    ]

        >>> weights = AttentionWeights.exponential(size=5, ascending=False, base=2.0)
        >>> print(weights)
        [0.5    0.25   0.125  0.0625 0.03125]
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

        This method allows users to define their own attention weights, which can be tailored to specific applications or signal characteristics.

        Parameters
        ----------
        weights : numpy.ndarray
            The custom attention weights provided by the user.

        Returns
        -------
        numpy.ndarray
            The normalized custom attention weights.

        Examples
        --------
        >>> weights = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        >>> custom_weights = AttentionWeights.custom_weights(weights)
        >>> print(custom_weights)
        [0.1 0.2 0.4 0.2 0.1]
        """
        return weights / np.sum(weights)
