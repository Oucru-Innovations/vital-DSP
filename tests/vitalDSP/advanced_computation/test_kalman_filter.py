import pytest
import numpy as np
from vitalDSP.advanced_computation.kalman_filter import KalmanFilter

@pytest.fixture
def setup_kalman_filter():
    initial_state = np.array([0])
    initial_covariance = np.array([[1]])
    process_covariance = np.array([[1e-5]])
    measurement_covariance = np.array([[1e-1]])
    kalman_filter = KalmanFilter(initial_state, initial_covariance, process_covariance, measurement_covariance)
    return kalman_filter

def test_kalman_filter_initialization():
    initial_state = np.array([0])
    initial_covariance = np.array([[1]])
    process_covariance = np.array([[1e-5]])
    measurement_covariance = np.array([[1e-1]])
    
    kalman_filter = KalmanFilter(initial_state, initial_covariance, process_covariance, measurement_covariance)
    
    assert np.array_equal(kalman_filter.state, initial_state)
    assert np.array_equal(kalman_filter.covariance, initial_covariance)
    assert np.array_equal(kalman_filter.process_covariance, process_covariance)
    assert np.array_equal(kalman_filter.measurement_covariance, measurement_covariance)

def test_kalman_filter_no_control(setup_kalman_filter):
    kalman_filter = setup_kalman_filter
    signal = np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.normal(size=100)
    measurement_matrix = np.array([[1]])
    transition_matrix = np.array([[1]])
    
    filtered_signal = kalman_filter.filter(signal, measurement_matrix, transition_matrix)
    
    assert isinstance(filtered_signal, np.ndarray)
    assert filtered_signal.shape[0] == len(signal)
    assert filtered_signal.shape[1] == len(kalman_filter.state)

def test_kalman_filter_with_control():
    initial_state = np.array([0])
    initial_covariance = np.array([[1]])
    process_covariance = np.array([[1e-5]])
    measurement_covariance = np.array([[1e-1]])
    
    kalman_filter = KalmanFilter(initial_state, initial_covariance, process_covariance, measurement_covariance)
    
    signal = np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.normal(size=100)
    measurement_matrix = np.array([[1]])
    transition_matrix = np.array([[1]])
    control_input = np.array([1])
    control_matrix = np.array([[0.1]])
    
    filtered_signal = kalman_filter.filter(signal, measurement_matrix, transition_matrix, control_input, control_matrix)
    
    assert isinstance(filtered_signal, np.ndarray)
    assert filtered_signal.shape[0] == len(signal)
    assert filtered_signal.shape[1] == len(kalman_filter.state)

def test_kalman_filter_control_none(setup_kalman_filter):
    kalman_filter = setup_kalman_filter
    signal = np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.normal(size=100)
    measurement_matrix = np.array([[1]])
    transition_matrix = np.array([[1]])
    
    # control_input and control_matrix are None
    filtered_signal = kalman_filter.filter(signal, measurement_matrix, transition_matrix, control_input=None, control_matrix=None)
    
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
    assert np.all(kalman_filter.covariance <= np.eye(1) * 1)  # Covariance should not grow excessively
