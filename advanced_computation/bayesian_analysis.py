import numpy as np
from scipy.optimize import minimize

class GaussianProcess:
    """
    Gaussian Process for Bayesian Optimization.

    Methods:
    - predict: Predicts the mean and variance of the objective function at new points.
    - update: Updates the GP with new observations.

    Example Usage:
    --------------
    gp = GaussianProcess()
    X_train = np.array([[0.1], [0.4], [0.7]])
    y_train = np.sin(3 * X_train) - X_train ** 2 + 0.7 * X_train
    gp.update(X_train, y_train)
    X_new = np.array([[0.2], [0.5]])
    mean, variance = gp.predict(X_new)
    print("Predicted Mean:", mean)
    print("Predicted Variance:", variance)
    """

    def __init__(self, length_scale=1.0, noise=1e-10):
        self.length_scale = length_scale
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.K = None

    def _rbf_kernel(self, X1, X2):
        """
        Radial Basis Function (RBF) kernel (also known as Gaussian kernel).

        Parameters:
        X1, X2 (numpy.ndarray): Arrays of input points.

        Returns:
        numpy.ndarray: Kernel matrix.
        """
        sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
        return np.exp(-0.5 / self.length_scale ** 2 * sqdist)

    def predict(self, X):
        """
        Predict the mean and variance of the objective function at new points.

        Parameters:
        X (numpy.ndarray): Points at which to predict.

        Returns:
        tuple: Predicted mean and variance.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("The GP model has not been updated with any training data.")
        
        K_s = self._rbf_kernel(self.X_train, X)
        K_ss = self._rbf_kernel(X, X)
        K_inv = np.linalg.inv(self.K + self.noise ** 2 * np.eye(len(self.X_train)))

        mu_s = K_s.T.dot(K_inv).dot(self.y_train)
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

        return mu_s.ravel(), np.diag(cov_s)

    def update(self, X_train, y_train):
        """
        Update the GP model with new observations.

        Parameters:
        X_train (numpy.ndarray): Observed input points.
        y_train (numpy.ndarray): Observed output values.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.K = self._rbf_kernel(X_train, X_train)


class BayesianOptimization:
    """
    Bayesian Optimization for parameter tuning in signal processing.

    Methods:
    - optimize: Performs Bayesian optimization to find the best parameters.

    Example Usage:
    --------------
    def objective_function(x):
        return -np.sin(3 * x) - x ** 2 + 0.7 * x

    bayesian_optimizer = BayesianOptimization(objective_function, bounds=(0, 2))
    best_x, best_y = bayesian_optimizer.optimize(n_iter=10)
    print("Best X:", best_x)
    print("Best Y:", best_y)
    """

    def __init__(self, func, bounds, length_scale=1.0, noise=1e-10):
        """
        Initialize BayesianOptimization with the objective function and bounds.

        Parameters:
        func (callable): The objective function to be optimized.
        bounds (tuple): The bounds within which to search for the optimal parameters.
        length_scale (float): Length scale parameter for the Gaussian Process kernel.
        noise (float): Noise level in the observations.
        """
        self.func = func
        self.bounds = bounds
        self.gp = GaussianProcess(length_scale, noise)
        self.X_samples = []
        self.Y_samples = []

    def acquisition(self, X, xi=0.01):
        """
        Expected Improvement (EI) acquisition function.

        Parameters:
        X (numpy.ndarray): Points at which to evaluate the acquisition function.
        xi (float): Exploration-exploitation trade-off parameter.

        Returns:
        numpy.ndarray: EI values at the provided points.
        """
        mu, sigma = self.gp.predict(X)
        mu_sample = np.max(self.Y_samples)
        
        imp = mu - mu_sample - xi
        Z = imp / sigma
        ei = imp * self._cdf(Z) + sigma * self._pdf(Z)
        ei[sigma == 0.0] = 0.0

        return ei

    def _cdf(self, x):
        """
        Cumulative Distribution Function (CDF) for the standard normal distribution.

        Parameters:
        x (numpy.ndarray): Input array.

        Returns:
        numpy.ndarray: CDF values.
        """
        return 0.5 * (1 + np.erf(x / np.sqrt(2)))

    def _pdf(self, x):
        """
        Probability Density Function (PDF) for the standard normal distribution.

        Parameters:
        x (numpy.ndarray): Input array.

        Returns:
        numpy.ndarray: PDF values.
        """
        return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)

    def propose_location(self, n_restarts=10):
        """
        Propose the next sampling location by optimizing the acquisition function.

        Parameters:
        n_restarts (int): Number of restarts for the optimization to avoid local optima.

        Returns:
        numpy.ndarray: Proposed next sampling point.
        """
        best_value = None
        best_x = None
        
        for _ in range(n_restarts):
            x_init = np.random.uniform(self.bounds[0], self.bounds[1], size=(1, 1))
            res = minimize(lambda x: -self.acquisition(np.atleast_2d(x)), x_init, bounds=[self.bounds])
            if best_value is None or res.fun < best_value:
                best_value = res.fun
                best_x = res.x

        return best_x

    def optimize(self, n_iter=10, random_seed=None):
        """
        Perform Bayesian optimization to find the best parameters.

        Parameters:
        n_iter (int): Number of iterations for the optimization.
        random_seed (int or None): Random seed for reproducibility.

        Returns:
        tuple: The best parameters and the corresponding function value.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        for _ in range(n_iter):
            if len(self.X_samples) > 0:
                self.gp.update(np.array(self.X_samples), np.array(self.Y_samples))

            X_next = self.propose_location()
            Y_next = self.func(X_next)

            self.X_samples.append(X_next)
            self.Y_samples.append(Y_next)

        best_index = np.argmax(self.Y_samples)
        best_x = self.X_samples[best_index]
        best_y = self.Y_samples[best_index]

        return best_x, best_y
