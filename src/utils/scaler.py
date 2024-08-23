import numpy as np


class StandardScaler:
    """
    Custom implementation of the Standard Scaler.

    Methods:
    - fit: Calculate mean and standard deviation for scaling.
    - transform: Scale the signal using the mean and standard deviation.
    - fit_transform: Fit to data and then transform it.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, signal):
        """
        Fit the scaler to the signal.

        Parameters:
        - signal (numpy.ndarray): The input signal.
        """
        self.mean_ = np.mean(signal)
        self.std_ = np.std(signal)

    def transform(self, signal):
        """
        Transform the signal using the fitted scaler.

        Parameters:
        - signal (numpy.ndarray): The input signal.

        Returns:
        - scaled_signal (numpy.ndarray): The scaled signal.
        """
        return (signal - self.mean_) / self.std_

    def fit_transform(self, signal):
        """
        Fit to data, then transform it.

        Parameters:
        - signal (numpy.ndarray): The input signal.

        Returns:
        - scaled_signal (numpy.ndarray): The scaled signal.
        """
        self.fit(signal)
        return self.transform(signal)
