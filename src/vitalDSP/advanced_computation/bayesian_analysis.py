import numpy as np
from scipy.optimize import minimize
from scipy.special import erf


class GaussianProcess:
    """
    A class implementing Gaussian Process (GP) for Bayesian Optimization.

    Gaussian Process models are used to predict the mean and variance of an unknown function based on observed data. This is particularly useful in Bayesian Optimization, where the GP helps in selecting the next sample point to evaluate.

    Methods
    -------
    predict(X)
        Predict the mean and variance of the objective function at new points X.
    update(X_train, y_train)
        Update the GP model with new observations.

    Example Usage
    -------------
    >>> gp = GaussianProcess(length_scale=1.0, noise=1e-10)
    >>> X_train = np.array([[0.1], [0.4], [0.7]])
    >>> y_train = np.sin(3 * X_train) - X_train ** 2 + 0.7 * X_train
    >>> gp.update(X_train, y_train)
    >>> X_new = np.array([[0.2], [0.5]])
    >>> mean, variance = gp.predict(X_new)
    >>> print("Predicted Mean:", mean)
    >>> print("Predicted Variance:", variance)
    """

    def __init__(self, length_scale=1.0, noise=1e-10):
        """
        Initialize the Gaussian Process model.

        Parameters
        ----------
        length_scale : float, optional
            The length scale parameter for the RBF kernel. Controls the smoothness of the function (default is 1.0).
        noise : float, optional
            The noise level in the observations. Used to regularize the covariance matrix (default is 1e-10).
        """
        self.length_scale = length_scale
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.K = None

    def _rbf_kernel(self, X1, X2):
        """
        Radial Basis Function (RBF) kernel, also known as the Gaussian kernel.

        Parameters
        ----------
        X1 : numpy.ndarray
            The first set of input points.
        X2 : numpy.ndarray
            The second set of input points.

        Returns
        -------
        numpy.ndarray
            The kernel matrix computed between X1 and X2.
        """
        sqdist = (
            np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        )
        return np.exp(-0.5 / self.length_scale**2 * sqdist)

    def predict(self, X):
        """
        Predict the mean and variance of the objective function at new points.

        Parameters
        ----------
        X : numpy.ndarray
            The points at which to predict the mean and variance.

        Returns
        -------
        tuple
            The predicted mean and variance at the input points X.

        Raises
        ------
        ValueError
            If the GP model has not been updated with any training data.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError(
                "The GP model has not been updated with any training data."
            )
        K_s = self._rbf_kernel(self.X_train, X)
        K_ss = self._rbf_kernel(X, X)
        K_inv = np.linalg.inv(self.K + self.noise**2 * np.eye(len(self.X_train)))

        mu_s = K_s.T.dot(K_inv).dot(self.y_train)
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

        return mu_s.ravel(), np.diag(cov_s)

    def update(self, X_train, y_train):
        """
        Update the GP model with new observations.

        Parameters
        ----------
        X_train : numpy.ndarray
            The observed input points.
        y_train : numpy.ndarray
            The observed output values corresponding to X_train.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.K = self._rbf_kernel(X_train, X_train)


class BayesianOptimization:
    """
    A class implementing Bayesian Optimization for parameter tuning.

    Bayesian Optimization is an efficient method for finding the minimum or maximum of an objective function, especially when the function is expensive to evaluate. It uses a Gaussian Process model to predict the function's behavior and an acquisition function to determine the next point to sample.

    Methods
    -------
    optimize(n_iter=10, random_seed=None)
        Perform Bayesian optimization to find the best parameters.
    acquisition(X, xi=0.01)
        Compute the Expected Improvement (EI) at new points.
    propose_location(n_restarts=10)
        Propose the next sampling location based on the acquisition function.

    Example Usage
    -------------
    >>> def objective_function(x):
    >>>     return -np.sin(3 * x) - x ** 2 + 0.7 * x
    >>>
    >>> bayesian_optimizer = BayesianOptimization(objective_function, bounds=(0, 2))
    >>> best_x, best_y = bayesian_optimizer.optimize(n_iter=10)
    >>> print("Best X:", best_x)
    >>> print("Best Y:", best_y)
    """

    def __init__(self, func, bounds, length_scale=1.0, noise=1e-10):
        """
        Initialize BayesianOptimization with the objective function and bounds.

        Parameters
        ----------
        func : callable
            The objective function to be optimized.
        bounds : tuple
            The bounds within which to search for the optimal parameters.
        length_scale : float, optional
            Length scale parameter for the Gaussian Process kernel (default is 1.0).
        noise : float, optional
            Noise level in the observations (default is 1e-10).
        """
        self.func = func
        self.bounds = bounds
        self.gp = GaussianProcess(length_scale, noise)
        self.X_samples = []
        self.Y_samples = []

    def acquisition(self, X, xi=0.01):
        """
        Expected Improvement (EI) acquisition function.

        The EI acquisition function is used to balance exploration and exploitation in Bayesian Optimization. It selects points that have the potential to improve upon the best observed value.

        Parameters
        ----------
        X : numpy.ndarray
            The points at which to evaluate the acquisition function.
        xi : float, optional
            Exploration-exploitation trade-off parameter (default is 0.01).

        Returns
        -------
        numpy.ndarray
            The EI values at the provided points.
        """
        mu, sigma = self.gp.predict(X)
        mu_sample = np.max(self.Y_samples)
        imp = mu - mu_sample - xi
        
        # Avoid divide by zero
        sigma_safe = np.where(sigma == 0.0, 1e-10, sigma)
        Z = imp / sigma_safe
        
        ei = imp * self._cdf(Z) + sigma * self._pdf(Z)
        ei[sigma == 0.0] = 0.0

        return ei

    def _cdf(self, x):
        """
        Cumulative Distribution Function (CDF) for the standard normal distribution.

        Parameters
        ----------
        x : numpy.ndarray
            The input array.

        Returns
        -------
        numpy.ndarray
            The CDF values.
        """
        return 0.5 * (1 + erf(x / np.sqrt(2)))

    def _pdf(self, x):
        """
        Probability Density Function (PDF) for the standard normal distribution.

        Parameters
        ----------
        x : numpy.ndarray
            The input array.

        Returns
        -------
        numpy.ndarray
            The PDF values.
        """
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

    def propose_location(self, n_restarts=10):
        """
        Propose the next sampling location by optimizing the acquisition function.

        Parameters
        ----------
        n_restarts : int, optional
            Number of restarts for the optimization to avoid local optima (default is 10).

        Returns
        -------
        numpy.ndarray
            The proposed next sampling point.
        """
        best_value = None
        best_x = None

        for _ in range(n_restarts):
            x_init = np.random.uniform(
                self.bounds[0], self.bounds[1], size=(1,)
            ).flatten()  # Flatten x_init
            res = minimize(
                lambda x: -self.acquisition(np.atleast_2d(x)),
                x_init,
                bounds=[self.bounds],
            )
            if best_value is None or res.fun < best_value:
                best_value = res.fun
                best_x = res.x

        return best_x

    def optimize(self, n_iter=10, random_seed=None):
        """
        Perform Bayesian optimization to find the best parameters.

        Parameters
        ----------
        n_iter : int, optional
            Number of iterations for the optimization (default is 10).
        random_seed : int or None, optional
            Random seed for reproducibility (default is None).

        Returns
        -------
        tuple
            The best parameters and the corresponding function value.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        for _ in range(n_iter):
            if len(self.X_samples) > 0:
                # Convert to numpy arrays for GP update
                self.gp.update(np.array(self.X_samples), np.array(self.Y_samples))

            X_next = self.propose_location()
            Y_next = self.func(X_next)

            # Append new sample to the lists
            self.X_samples.append(X_next)
            self.Y_samples.append(Y_next)

        best_index = np.argmax(self.Y_samples)
        best_x = self.X_samples[best_index]
        best_y = self.Y_samples[best_index]

        return best_x, best_y
