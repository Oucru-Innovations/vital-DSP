import numpy as np

class KalmanFilter:
    """
    Kalman Filter for real-time filtering and continuous monitoring of signals.

    Methods:
    - filter: Apply the Kalman filter to the signal.

    Example Usage:
    --------------
    signal = np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.normal(size=100)
    initial_state = np.array([0])
    initial_covariance = np.array([[1]])
    process_covariance = np.array([[1e-5]])
    measurement_covariance = np.array([[1e-1]])
    measurement_matrix = np.array([[1]])
    transition_matrix = np.array([[1]])

    kalman_filter = KalmanFilter(initial_state, initial_covariance, process_covariance, measurement_covariance)
    filtered_signal = kalman_filter.filter(signal, measurement_matrix, transition_matrix)
    print("Filtered Signal:", filtered_signal)
    """

    def __init__(self, initial_state, initial_covariance, process_covariance, measurement_covariance):
        self.state = initial_state
        self.covariance = initial_covariance
        self.process_covariance = process_covariance
        self.measurement_covariance = measurement_covariance

    def filter(self, signal, measurement_matrix, transition_matrix, control_input=None, control_matrix=None):
        """
        Apply the Kalman filter to the signal.

        Parameters:
        signal (numpy.ndarray): The signal to filter.
        measurement_matrix (numpy.ndarray): The measurement matrix.
        transition_matrix (numpy.ndarray): The state transition matrix.
        control_input (numpy.ndarray or None): Optional control input.
        control_matrix (numpy.ndarray or None): Optional control matrix.

        Returns:
        numpy.ndarray: The filtered signal.
        """
        filtered_signal = []

        for measurement in signal:
            # Prediction
            self.state = transition_matrix @ self.state
            if control_input is not None and control_matrix is not None:
                self.state += control_matrix @ control_input
            self.covariance = transition_matrix @ self.covariance @ transition_matrix.T + self.process_covariance

            # Update
            innovation = measurement - measurement_matrix @ self.state
            innovation_covariance = measurement_matrix @ self.covariance @ measurement_matrix.T + self.measurement_covariance
            kalman_gain = self.covariance @ measurement_matrix.T @ np.linalg.inv(innovation_covariance)
            self.state = self.state + kalman_gain @ innovation
            self.covariance = (np.eye(len(self.state)) - kalman_gain @ measurement_matrix) @ self.covariance

            filtered_signal.append(self.state.copy())

        return np.array(filtered_signal)
