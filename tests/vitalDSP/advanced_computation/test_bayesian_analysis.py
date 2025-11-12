import pytest
import numpy as np
from scipy.optimize import minimize
from scipy.special import erf  # Import the correct erf function
from vitalDSP.advanced_computation.bayesian_analysis import (
    GaussianProcess,
    BayesianOptimization,
)


# Fixtures for test signals and GaussianProcess
@pytest.fixture
def sample_gp():
    return GaussianProcess(length_scale=1.0, noise=1e-5)


@pytest.fixture
def X_train():
    return np.array([[0.1], [0.4], [0.7]])


@pytest.fixture
def y_train(X_train):
    return np.sin(3 * X_train) - X_train**2 + 0.7 * X_train


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
        return -np.sin(3 * x) - x**2 + 0.7 * x

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

    assert 0 <= best_x <= 2  # Ensure the best parameter is within the bounds
    assert isinstance(best_y.item(), float)  # Ensure best_y is a valid float value
    assert (
        len(sample_bayesian_optimization.X_samples) == 7
    )  # 2 initial samples + 5 iterations
    assert (
        len(sample_bayesian_optimization.Y_samples) == 7
    )  # 2 initial samples + 5 iterations


class TestBayesianAnalysisMissingCoverage:
    """Tests to cover missing lines in bayesian_analysis.py."""

    def test_optimize_with_random_seed(self, sample_objective_function):
        """Test optimize when random_seed is provided.
        
        This test covers line 391 in bayesian_analysis.py where
        np.random.seed(random_seed) is called.
        """
        optimizer = BayesianOptimization(sample_objective_function, bounds=(0, 2))
        
        # Test with random_seed set
        best_x1, best_y1 = optimizer.optimize(n_iter=3, random_seed=42)
        
        # Reset and run again with same seed - should get same results
        optimizer2 = BayesianOptimization(sample_objective_function, bounds=(0, 2))
        best_x2, best_y2 = optimizer2.optimize(n_iter=3, random_seed=42)
        
        # Results should be identical with same seed
        assert np.allclose(best_x1, best_x2)
        assert np.allclose(best_y1, best_y2)

    def test_optimize_empty_x_samples_initialization(self, sample_objective_function):
        """Test optimize when X_samples is empty (initialization).
        
        This test covers lines 405-412 in bayesian_analysis.py where
        initialization loop is executed when len(self.X_samples) == 0.
        """
        optimizer = BayesianOptimization(sample_objective_function, bounds=(0, 2))
        
        # Ensure X_samples is empty
        assert len(optimizer.X_samples) == 0
        
        # Run optimize - should initialize with 3 random samples
        best_x, best_y = optimizer.optimize(n_iter=2, random_seed=42)
        
        # Should have at least 3 initial samples + 2 iterations = 5 samples
        assert len(optimizer.X_samples) >= 5
        assert len(optimizer.Y_samples) >= 5
        assert 0 <= best_x <= 2

    def test_optimize_y_samples_array_normalization(self, sample_objective_function):
        """Test optimize when Y_samples contains arrays/lists.
        
        This test covers line 422 in bayesian_analysis.py where
        y = float(np.atleast_1d(y).flatten()[0]) is executed.
        """
        optimizer = BayesianOptimization(sample_objective_function, bounds=(0, 2))
        
        # Pre-populate Y_samples with arrays/lists to trigger normalization
        # X_samples should be numpy arrays (as expected by the code)
        optimizer.X_samples = [np.array([[0.1]]), np.array([[0.5]])]
        optimizer.Y_samples = [np.array([1.5]), [2.0]]  # Arrays/lists instead of scalars
        
        # Run optimize - should normalize Y_samples
        best_x, best_y = optimizer.optimize(n_iter=2, random_seed=42)
        
        # All Y_samples should be scalars after normalization
        assert all(isinstance(y, (int, float, np.number)) for y in optimizer.Y_samples)
        assert isinstance(best_y, (int, float, np.number))

    def test_optimize_linalgerror_fallback(self, sample_objective_function):
        """Test optimize when LinAlgError occurs (fallback to random sampling).
        
        This test covers lines 437-444 in bayesian_analysis.py where
        LinAlgError is caught and random sampling fallback is used.
        """
        from unittest.mock import patch
        
        optimizer = BayesianOptimization(sample_objective_function, bounds=(0, 2))
        
        # Pre-populate with some samples (as numpy arrays)
        optimizer.X_samples = [np.array([[0.1]]), np.array([[0.5]])]
        optimizer.Y_samples = [1.0, 2.0]
        
        # Mock gp.update to raise LinAlgError
        original_update = optimizer.gp.update
        call_count = [0]
        
        def mock_update(X_train, y_train):
            call_count[0] += 1
            if call_count[0] == 1:  # First call raises LinAlgError
                raise np.linalg.LinAlgError("Singular matrix")
            return original_update(X_train, y_train)
        
        optimizer.gp.update = mock_update
        
        # Run optimize - should fall back to random sampling
        best_x, best_y = optimizer.optimize(n_iter=2, random_seed=42)
        
        # Should still return valid results
        assert 0 <= best_x <= 2
        assert isinstance(best_y, (int, float, np.number))
        # Should have more samples (including random fallback samples)
        assert len(optimizer.X_samples) >= 2
        assert len(optimizer.Y_samples) >= 2