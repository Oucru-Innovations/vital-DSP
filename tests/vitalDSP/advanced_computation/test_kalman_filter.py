import pytest
import numpy as np
import warnings
from vitalDSP.advanced_computation.kalman_filter import KalmanFilter


@pytest.fixture
def setup_kalman_filter():
    initial_state = np.array([0])
    initial_covariance = np.array([[1]])
    process_covariance = np.array([[1e-5]])
    measurement_covariance = np.array([[1e-1]])
    kalman_filter = KalmanFilter(
        initial_state, initial_covariance, process_covariance, measurement_covariance
    )
    return kalman_filter


def test_kalman_filter_initialization():
    initial_state = np.array([0])
    initial_covariance = np.array([[1]])
    process_covariance = np.array([[1e-5]])
    measurement_covariance = np.array([[1e-1]])

    kalman_filter = KalmanFilter(
        initial_state, initial_covariance, process_covariance, measurement_covariance
    )

    assert np.array_equal(kalman_filter.state, initial_state)
    assert np.array_equal(kalman_filter.covariance, initial_covariance)
    assert np.array_equal(kalman_filter.process_covariance, process_covariance)
    assert np.array_equal(kalman_filter.measurement_covariance, measurement_covariance)


def test_kalman_filter_no_control(setup_kalman_filter):
    kalman_filter = setup_kalman_filter
    signal = np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.normal(size=100)
    measurement_matrix = np.array([[1]])
    transition_matrix = np.array([[1]])

    filtered_signal = kalman_filter.filter(
        signal, measurement_matrix, transition_matrix
    )

    assert isinstance(filtered_signal, np.ndarray)
    assert filtered_signal.shape[0] == len(signal)
    assert filtered_signal.shape[1] == len(kalman_filter.state)


def test_kalman_filter_with_control():
    initial_state = np.array([0])
    initial_covariance = np.array([[1]])
    process_covariance = np.array([[1e-5]])
    measurement_covariance = np.array([[1e-1]])

    kalman_filter = KalmanFilter(
        initial_state, initial_covariance, process_covariance, measurement_covariance
    )

    signal = np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.normal(size=100)
    measurement_matrix = np.array([[1]])
    transition_matrix = np.array([[1]])
    control_input = np.array([1])
    control_matrix = np.array([[0.1]])

    filtered_signal = kalman_filter.filter(
        signal, measurement_matrix, transition_matrix, control_input, control_matrix
    )

    assert isinstance(filtered_signal, np.ndarray)
    assert filtered_signal.shape[0] == len(signal)
    assert filtered_signal.shape[1] == len(kalman_filter.state)


def test_kalman_filter_control_none(setup_kalman_filter):
    kalman_filter = setup_kalman_filter
    signal = np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.normal(size=100)
    measurement_matrix = np.array([[1]])
    transition_matrix = np.array([[1]])

    # control_input and control_matrix are None
    filtered_signal = kalman_filter.filter(
        signal,
        measurement_matrix,
        transition_matrix,
        control_input=None,
        control_matrix=None,
    )

    assert isinstance(filtered_signal, np.ndarray)
    assert filtered_signal.shape[0] == len(signal)


def test_kalman_filter_update_step(setup_kalman_filter):
    kalman_filter = setup_kalman_filter
    signal = np.sin(np.linspace(0, 10, 10))  # Using a shorter signal for simplicity
    measurement_matrix = np.array([[1]])
    transition_matrix = np.array([[1]])

    # Initial state and covariance
    state = kalman_filter.state
    covariance = kalman_filter.covariance
    process_covariance = kalman_filter.process_covariance
    measurement_covariance = kalman_filter.measurement_covariance

    for i, measurement in enumerate(signal):
        # Prediction step (state and covariance)
        predicted_state = transition_matrix @ state
        predicted_covariance = (
            transition_matrix @ covariance @ transition_matrix.T + process_covariance
        )

        # Innovation and innovation covariance
        innovation = measurement - (measurement_matrix @ predicted_state)
        innovation_covariance = (
            measurement_matrix @ predicted_covariance @ measurement_matrix.T
            + measurement_covariance
        )

        # Kalman gain
        kalman_gain = (
            predicted_covariance
            @ measurement_matrix.T
            @ np.linalg.inv(innovation_covariance)
        )

        # Update step (state and covariance)
        updated_state = predicted_state + kalman_gain @ innovation
        updated_covariance = (
            np.eye(len(state)) - kalman_gain @ measurement_matrix
        ) @ predicted_covariance

        # Check that state and covariance match the manual calculation
        kalman_filter.filter([measurement], measurement_matrix, transition_matrix)
        assert np.allclose(kalman_filter.state, updated_state, atol=1e-5)
        assert np.allclose(kalman_filter.covariance, updated_covariance, atol=1e-5)

        # Update the state and covariance for the next iteration
        state = updated_state
        covariance = updated_covariance


def test_kalman_filter_covariance_update(setup_kalman_filter):
    kalman_filter = setup_kalman_filter
    signal = np.sin(np.linspace(0, 10, 10))
    measurement_matrix = np.array([[1]])
    transition_matrix = np.array([[1]])

    kalman_filter.filter(signal, measurement_matrix, transition_matrix)

    # Check that covariance updates at each step
    assert np.all(
        kalman_filter.covariance <= np.eye(1) * 1
    )  # Covariance should not grow excessively


def test_initial_state_not_1d():
    """Test that ValueError is raised when initial_state is not 1-dimensional."""
    initial_state = np.array([[0]])  # 2D instead of 1D
    initial_covariance = np.array([[1]])
    process_covariance = np.array([[1e-5]])
    measurement_covariance = np.array([[1e-1]])

    with pytest.raises(ValueError, match="Initial state must be 1-dimensional"):
        KalmanFilter(
            initial_state, initial_covariance, process_covariance, measurement_covariance
        )


def test_initial_covariance_not_2d():
    """Test that ValueError is raised when initial_covariance is not 2-dimensional."""
    initial_state = np.array([0])
    initial_covariance = np.array([1])  # 1D instead of 2D
    process_covariance = np.array([[1e-5]])
    measurement_covariance = np.array([[1e-1]])

    with pytest.raises(ValueError, match="Initial covariance must be 2-dimensional"):
        KalmanFilter(
            initial_state, initial_covariance, process_covariance, measurement_covariance
        )


def test_process_covariance_not_2d():
    """Test that ValueError is raised when process_covariance is not 2-dimensional."""
    initial_state = np.array([0])
    initial_covariance = np.array([[1]])
    process_covariance = np.array([1e-5])  # 1D instead of 2D
    measurement_covariance = np.array([[1e-1]])

    with pytest.raises(ValueError, match="Process covariance must be 2-dimensional"):
        KalmanFilter(
            initial_state, initial_covariance, process_covariance, measurement_covariance
        )


def test_measurement_covariance_not_2d():
    """Test that ValueError is raised when measurement_covariance is not 2-dimensional."""
    initial_state = np.array([0])
    initial_covariance = np.array([[1]])
    process_covariance = np.array([[1e-5]])
    measurement_covariance = np.array([1e-1])  # 1D instead of 2D

    with pytest.raises(ValueError, match="Measurement covariance must be 2-dimensional"):
        KalmanFilter(
            initial_state, initial_covariance, process_covariance, measurement_covariance
        )


def test_process_covariance_not_positive_definite():
    """Test that ValueError is raised when process_covariance is not positive definite."""
    initial_state = np.array([0])
    initial_covariance = np.array([[1]])
    # Create a non-positive definite matrix (negative eigenvalue)
    process_covariance = np.array([[-1]])  # Negative definite
    measurement_covariance = np.array([[1e-1]])

    with pytest.raises(ValueError, match="Process covariance must be positive definite"):
        KalmanFilter(
            initial_state, initial_covariance, process_covariance, measurement_covariance
        )


def test_measurement_covariance_not_positive_definite():
    """Test that ValueError is raised when measurement_covariance is not positive definite."""
    initial_state = np.array([0])
    initial_covariance = np.array([[1]])
    process_covariance = np.array([[1e-5]])
    # Create a non-positive definite matrix (negative eigenvalue)
    measurement_covariance = np.array([[-1]])  # Negative definite

    with pytest.raises(ValueError, match="Measurement covariance must be positive definite"):
        KalmanFilter(
            initial_state, initial_covariance, process_covariance, measurement_covariance
        )


def test_high_condition_number():
    """Test that ValueError is raised when condition number is too high (> 1e12)."""
    # Create matrices with very high condition number
    # Need eigenvalues > 1e-12 to pass eigenvalue check, but condition number > 1e12
    # Condition number = max(eigenvalue) / min(eigenvalue)
    # So we need max > 1e12 * min, and min > 1e-12
    # Example: min = 1e-10, max = 1e3 gives condition number = 1e13 > 1e12
    initial_state = np.array([0, 0])  # Match dimension of 2x2 covariance matrices
    initial_covariance = np.array([[1e-10, 0], [0, 1e3]])
    process_covariance = np.array([[1e-10, 0], [0, 1e3]])
    measurement_covariance = np.array([[1e-10, 0], [0, 1e3]])

    with pytest.raises(ValueError, match="High condition number detected"):
        KalmanFilter(
            initial_state, initial_covariance, process_covariance, measurement_covariance
        )


def test_innovation_covariance_high_condition_number():
    """Test that warning is issued and pseudoinverse is used when innovation covariance has high condition number."""
    # Create a scenario where innovation covariance becomes ill-conditioned during filtering
    # We need a 2D measurement space to get a 2x2 innovation covariance matrix with high condition number
    initial_state = np.array([0.0, 0.0])
    
    # Start with reasonable covariance values that pass initialization
    initial_covariance = np.array([[1.0, 0.0], [0.0, 1.0]])
    process_covariance = np.array([[0.01, 0.0], [0.0, 0.01]])
    # Use small measurement covariance that passes init but contributes to ill-conditioning
    measurement_covariance = np.array([[1e-10, 0.0], [0.0, 1e-10]])

    kalman_filter = KalmanFilter(
        initial_state, initial_covariance, process_covariance, measurement_covariance
    )

    # Manually set covariance to create high condition number in innovation covariance
    # innovation_covariance = measurement_matrix @ covariance @ measurement_matrix.T + measurement_covariance
    # We want this to have condition number > 1e12
    # Use a covariance with very different eigenvalues
    kalman_filter.covariance = np.array([[1e12, 0.0], [0.0, 1e-12]])

    # Signal should be an array where each element is a measurement
    # For 2D measurements, we can use a list of arrays or a 2D array
    signal = np.array([[1.0, 1.0]])  # 2D array: one measurement with 2 dimensions
    measurement_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])  # 2x2 identity, projects 2D state to 2D measurement
    transition_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])

    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        filtered_signal = kalman_filter.filter(signal, measurement_matrix, transition_matrix)

        # Check that warning was issued
        assert len(w) > 0
        assert any(
            "High condition number in innovation covariance" in str(warning.message)
            for warning in w
        )

    # Check that filtering still works (returns valid result)
    assert isinstance(filtered_signal, np.ndarray)
    assert filtered_signal.shape[0] == len(signal)
    assert filtered_signal.shape[1] == len(initial_state)
