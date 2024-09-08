import pytest
import numpy as np
from scipy.optimize import minimize
from scipy.special import erf  # Import the correct erf function
from vitalDSP.advanced_computation.bayesian_analysis import GaussianProcess, BayesianOptimization

# Fixtures for test signals and GaussianProcess
@pytest.fixture
def sample_gp():
    return GaussianProcess(length_scale=1.0, noise=1e-5)

@pytest.fixture
def X_train():
    return np.array([[0.1], [0.4], [0.7]])

@pytest.fixture
def y_train(X_train):
    return np.sin(3 * X_train) - X_train ** 2 + 0.7 * X_train

@pytest.fixture
def X_new():
    return np.array([[0.2], [0.5]])

def test_rbf_kernel(sample_gp, X_train):
    # Test RBF kernel output
    K = sample_gp._rbf_kernel(X_train, X_train)
    assert K.shape == (len(X_train), len(X_train))
    assert np.all(K >= 0)  # Kernel values should be non-negative

def test_predict_no_training_data(sample_gp, X_new):
    # Ensure error is raised if predict is called before updating the GP model
    with pytest.raises(ValueError, match="The GP model has not been updated"):
        sample_gp.predict(X_new)

def test_update(sample_gp, X_train, y_train):
    # Test that the GP model is correctly updated
    sample_gp.update(X_train, y_train)
    assert sample_gp.X_train is not None
    assert sample_gp.y_train is not None
    assert sample_gp.K is not None

def test_predict_with_training_data(sample_gp, X_train, y_train, X_new):
    # Test prediction with a trained GP model
    sample_gp.update(X_train, y_train)
    mean, variance = sample_gp.predict(X_new)
    assert mean.shape == (len(X_new),)
    assert variance.shape == (len(X_new),)
    assert np.all(variance >= 0)  # Variance should be non-negative

# Fixtures for BayesianOptimization
@pytest.fixture
def sample_objective_function():
    def objective_function(x):
        # Ensure the input x is always converted to a numpy array
        x = np.array(x)
        return -np.sin(3 * x) - x ** 2 + 0.7 * x
    return objective_function

@pytest.fixture
def sample_bayesian_optimization(sample_objective_function):
    return BayesianOptimization(sample_objective_function, bounds=(0, 2))

def test_acquisition_function(sample_bayesian_optimization):
    # Test the acquisition function with properly shaped inputs
    X_samples = [[0.1], [0.5]]  # Use lists instead of np arrays
    Y_samples = [sample_bayesian_optimization.func(x) for x in X_samples]
    sample_bayesian_optimization.X_samples = X_samples
    sample_bayesian_optimization.Y_samples = Y_samples
    sample_bayesian_optimization.gp.update(np.array(X_samples), np.array(Y_samples))
    
    # The test input must have 2D shape (n_samples, n_features)
    X_new = np.array([[0.2], [0.5]])
    ei = sample_bayesian_optimization.acquisition(X_new)
    
    assert ei.shape == (len(X_new),)
    assert np.all(ei >= 0)  # Expected Improvement should be non-negative

def test_propose_location(sample_bayesian_optimization):
    # Provide initial training data to update the GP model
    X_samples = [[0.1], [0.5]]  # Use lists instead of np arrays
    Y_samples = [sample_bayesian_optimization.func(x) for x in X_samples]
    sample_bayesian_optimization.X_samples = X_samples
    sample_bayesian_optimization.Y_samples = Y_samples
    sample_bayesian_optimization.gp.update(np.array(X_samples), np.array(Y_samples))

    # Test the location proposal method with proper shape
    X_next = sample_bayesian_optimization.propose_location()
    
    # X_next should be a 1D array as expected by 'minimize'
    assert X_next.shape == (1,)
    assert 0 <= X_next <= 2  # Proposed location should be within bounds

def test_optimize(sample_bayesian_optimization):
    # Provide initial training data to update the GP model
    X_samples = [[0.1], [0.5]]  # Use lists instead of np arrays
    Y_samples = [sample_bayesian_optimization.func(x) for x in X_samples]
    sample_bayesian_optimization.X_samples = X_samples
    sample_bayesian_optimization.Y_samples = Y_samples
    sample_bayesian_optimization.gp.update(np.array(X_samples), np.array(Y_samples))

    # Test the optimization process
    best_x, best_y = sample_bayesian_optimization.optimize(n_iter=5, random_seed=42)
    
    # Add debug prints to see what's happening
    print("Best X:", best_x)
    print("Best Y:", best_y)
    print("X_samples:", sample_bayesian_optimization.X_samples)
    print("Y_samples:", sample_bayesian_optimization.Y_samples)
    
    assert 0 <= best_x <= 2  # Ensure the best parameter is within the bounds
    assert isinstance(best_y.item(), float)  # Ensure best_y is a valid float value
    assert len(sample_bayesian_optimization.X_samples) == 7  # 2 initial samples + 5 iterations
    assert len(sample_bayesian_optimization.Y_samples) == 7  # 2 initial samples + 5 iterations
