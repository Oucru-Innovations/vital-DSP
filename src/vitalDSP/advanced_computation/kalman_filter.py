import numpy as np
import warnings


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
        # Input validation and numerical stability checks
        self._validate_matrices(initial_state, initial_covariance, process_covariance, measurement_covariance)
        
        self.state = initial_state.astype(float)  # Ensuring state is float
        self.covariance = initial_covariance.astype(float)  # Ensuring covariance is float
        self.process_covariance = process_covariance.astype(float)
        self.measurement_covariance = measurement_covariance.astype(float)
        
        # Numerical stability checks
        self._check_numerical_stability()
    
    def _validate_matrices(self, initial_state, initial_covariance, process_covariance, measurement_covariance):
        """Validate input matrices for proper dimensions and properties."""
        # Check dimensions
        if initial_state.ndim != 1:
            raise ValueError("Initial state must be 1-dimensional")
        
        if initial_covariance.ndim != 2:
            raise ValueError("Initial covariance must be 2-dimensional")
        
        if process_covariance.ndim != 2:
            raise ValueError("Process covariance must be 2-dimensional")
        
        if measurement_covariance.ndim != 2:
            raise ValueError("Measurement covariance must be 2-dimensional")
        
        # Check for positive definiteness of covariance matrices
        if not np.all(np.linalg.eigvals(initial_covariance) > 0):
            raise ValueError("Initial covariance must be positive definite")
        
        if not np.all(np.linalg.eigvals(process_covariance) > 0):
            raise ValueError("Process covariance must be positive definite")
        
        if not np.all(np.linalg.eigvals(measurement_covariance) > 0):
            raise ValueError("Measurement covariance must be positive definite")
    
    def _check_numerical_stability(self):
        """Check for potential numerical stability issues."""
        # Check for very small eigenvalues that could cause numerical issues
        initial_eigenvals = np.linalg.eigvals(self.covariance)
        process_eigenvals = np.linalg.eigvals(self.process_covariance)
        measurement_eigenvals = np.linalg.eigvals(self.measurement_covariance)
        
        min_eigenval = min(np.min(initial_eigenvals), np.min(process_eigenvals), np.min(measurement_eigenvals))
        
        if min_eigenval <= 1e-12:
            raise ValueError(f"Very small eigenvalues detected ({min_eigenval:.2e}). "
                           f"This may cause numerical instability.")
        
        # Check condition numbers
        initial_cond = np.linalg.cond(self.covariance)
        process_cond = np.linalg.cond(self.process_covariance)
        measurement_cond = np.linalg.cond(self.measurement_covariance)
        
        max_cond = max(initial_cond, process_cond, measurement_cond)
        
        if max_cond > 1e12:
            raise ValueError(f"High condition number detected ({max_cond:.2e}). "
                           f"This may cause numerical instability.")

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
                self.state += control_matrix @ control_input.astype(
                    float
                )  # Ensure control_input is float
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
            
            # Numerical stability check for innovation covariance
            if np.linalg.cond(innovation_covariance) > 1e12:
                warnings.warn("High condition number in innovation covariance. Using pseudoinverse.")
                kalman_gain = (
                    self.covariance
                    @ measurement_matrix.T
                    @ np.linalg.pinv(innovation_covariance)
                )
            else:
                kalman_gain = (
                    self.covariance
                    @ measurement_matrix.T
                    @ np.linalg.inv(innovation_covariance)
                )
            
            self.state = self.state + kalman_gain @ innovation
            
            # Joseph form for numerical stability
            I = np.eye(len(self.state))
            self.covariance = (
                (I - kalman_gain @ measurement_matrix) @ self.covariance @ 
                (I - kalman_gain @ measurement_matrix).T + 
                kalman_gain @ self.measurement_covariance @ kalman_gain.T
            )
            
            # Ensure covariance remains positive definite
            self.covariance = self._ensure_positive_definite(self.covariance)

            filtered_signal.append(self.state.copy())

        return np.array(filtered_signal)
    
    def _ensure_positive_definite(self, matrix):
        """Ensure matrix remains positive definite using eigenvalue clipping."""
        eigenvals, eigenvecs = np.linalg.eigh(matrix)
        
        # Clip eigenvalues to prevent numerical issues
        min_eigenval = 1e-12
        eigenvals = np.maximum(eigenvals, min_eigenval)
        
        # Reconstruct matrix
        return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
