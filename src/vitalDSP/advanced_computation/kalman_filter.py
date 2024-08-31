import numpy as np


class KalmanFilter:
    """
    A Kalman Filter for real-time filtering and continuous monitoring of signals.

    The Kalman Filter is an optimal recursive data processing algorithm that estimates the state
    of a dynamic system from a series of noisy measurements. It is widely used in control systems,
    signal processing, and time series analysis.

    Parameters
    ----------
    initial_state : numpy.ndarray
        The initial state estimate of the system.
    initial_covariance : numpy.ndarray
        The initial covariance estimate of the state.
    process_covariance : numpy.ndarray
        The process noise covariance matrix, representing the uncertainty in the system dynamics.
    measurement_covariance : numpy.ndarray
        The measurement noise covariance matrix, representing the uncertainty in the measurements.

    Methods
    -------
    filter(signal, measurement_matrix, transition_matrix, control_input=None, control_matrix=None)
        Apply the Kalman filter to the input signal to obtain a filtered estimate of the state.

    Examples
    --------
    >>> signal = np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.normal(size=100)
    >>> initial_state = np.array([0])
    >>> initial_covariance = np.array([[1]])
    >>> process_covariance = np.array([[1e-5]])
    >>> measurement_covariance = np.array([[1e-1]])
    >>> measurement_matrix = np.array([[1]])
    >>> transition_matrix = np.array([[1]])

    >>> kalman_filter = KalmanFilter(initial_state, initial_covariance, process_covariance, measurement_covariance)
    >>> filtered_signal = kalman_filter.filter(signal, measurement_matrix, transition_matrix)
    >>> print("Filtered Signal:", filtered_signal)
    """

    def __init__(
        self,
        initial_state,
        initial_covariance,
        process_covariance,
        measurement_covariance,
    ):
        """
        Initialize the KalmanFilter with the initial state, covariance, process noise, and measurement noise.

        Parameters
        ----------
        initial_state : numpy.ndarray
            The initial state estimate of the system.
        initial_covariance : numpy.ndarray
            The initial covariance estimate of the state.
        process_covariance : numpy.ndarray
            The process noise covariance matrix.
        measurement_covariance : numpy.ndarray
            The measurement noise covariance matrix.
        """
        self.state = initial_state
        self.covariance = initial_covariance
        self.process_covariance = process_covariance
        self.measurement_covariance = measurement_covariance

    def filter(
        self,
        signal,
        measurement_matrix,
        transition_matrix,
        control_input=None,
        control_matrix=None,
    ):
        """
        Apply the Kalman filter to the input signal.

        The Kalman filter recursively estimates the state of a dynamic system from noisy measurements.
        At each time step, it predicts the next state, updates the estimate based on the actual measurement,
        and provides a filtered signal as output.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal to be filtered.
        measurement_matrix : numpy.ndarray
            The measurement matrix that maps the true state space into the observed space.
        transition_matrix : numpy.ndarray
            The state transition matrix that describes how the state evolves from one time step to the next.
        control_input : numpy.ndarray or None, optional
            An optional control input to the system (default is None).
        control_matrix : numpy.ndarray or None, optional
            An optional control matrix that maps the control input to the state space (default is None).

        Returns
        -------
        numpy.ndarray
            The filtered signal, where each element corresponds to the estimated state at each time step.

        Examples
        --------
        >>> signal = np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.normal(size=100)
        >>> measurement_matrix = np.array([[1]])
        >>> transition_matrix = np.array([[1]])
        >>> kalman_filter = KalmanFilter(initial_state, initial_covariance, process_covariance, measurement_covariance)
        >>> filtered_signal = kalman_filter.filter(signal, measurement_matrix, transition_matrix)
        >>> print(filtered_signal)
        """
        filtered_signal = []

        for measurement in signal:
            # Prediction step
            self.state = transition_matrix @ self.state
            if control_input is not None and control_matrix is not None:
                self.state += control_matrix @ control_input
            self.covariance = (
                transition_matrix @ self.covariance @ transition_matrix.T
                + self.process_covariance
            )

            # Update step
            innovation = measurement - measurement_matrix @ self.state
            innovation_covariance = (
                measurement_matrix @ self.covariance @ measurement_matrix.T
                + self.measurement_covariance
            )
            kalman_gain = (
                self.covariance
                @ measurement_matrix.T
                @ np.linalg.inv(innovation_covariance)
            )
            self.state = self.state + kalman_gain @ innovation
            self.covariance = (
                np.eye(len(self.state)) - kalman_gain @ measurement_matrix
            ) @ self.covariance

            filtered_signal.append(self.state.copy())

        return np.array(filtered_signal)
