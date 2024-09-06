import numpy as np


def correlation_based_artifact_detection(signals, threshold=0.5):
    """
    Detect artifacts by analyzing the correlation between multiple signals.

    Parameters
    ----------
    signals : list of numpy.ndarray
        List of signals to analyze for artifacts.
    threshold : float, optional (default=0.5)
        The correlation threshold below which points are considered artifacts.

    Returns
    -------
    artifact_indices : numpy.ndarray
        Indices of the detected artifacts across all signals.

    Examples
    --------
    >>> signal1 = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01))
    >>> signal2 = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01) + 0.5)
    >>> artifacts = correlation_based_artifact_detection([signal1, signal2], threshold=0.7)
    >>> print(artifacts)
    """
    if len(signals) == 0:
        raise ValueError("Signal list cannot be empty")

    combined_signal = np.mean(np.array(signals), axis=0)
    correlations = np.array(
        [np.corrcoef(combined_signal, signal)[0, 1] for signal in signals]
    )

    # Adjust to select indices where the correlation is below the threshold
    artifact_indices = np.where(correlations < threshold)[0]

    return artifact_indices


def energy_ratio_artifact_detection(signals, window_size=100, threshold=0.5):
    """
    Detect artifacts by analyzing the energy ratio between multiple signals.
    This version doesn't involve mutual information, but checks for energy ratios.

    Parameters
    ----------
    signals : list of numpy.ndarray
        List of signals to analyze for artifacts.
    window_size : int, optional (default=100)
        The size of the window for local analysis.
    threshold : float, optional (default=0.5)
        The energy ratio threshold below which points are considered artifacts.

    Returns
    -------
    artifact_indices : numpy.ndarray
        Indices of the detected artifacts across all signals.

    Examples
    --------
    >>> signal1 = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01))
    >>> signal2 = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01) + 0.5)
    >>> artifacts = energy_ratio_artifact_detection([signal1, signal2], window_size=50, threshold=0.6)
    >>> print(artifacts)
    """
    if len(signals) == 0:
        raise ValueError("Signal list cannot be empty")

    artifact_indices = []
    combined_signal = np.mean(np.array(signals), axis=0)

    for i in range(0, len(combined_signal) - window_size + 1, window_size):
        window_energies = [
            np.sum(signal[i : i + window_size] ** 2) for signal in signals
        ]
        if np.max(window_energies) == 0:
            continue  # Avoid division by zero in energy ratio calculation
        energy_ratio = np.min(window_energies) / np.max(window_energies)

        if energy_ratio < threshold:
            artifact_indices.extend(range(i, i + window_size))

    return np.array(artifact_indices)


def mutual_information_artifact_detection(signals, num_bins=10, threshold=0.1):
    """
    Detect artifacts using mutual information between multiple signals.

    Parameters
    ----------
    signals : list of numpy.ndarray
        List of signals to analyze for artifacts.
    num_bins : int, optional (default=10)
        Number of bins to use for histogram estimation in mutual information.
    threshold : float, optional (default=0.1)
        The mutual information threshold below which points are considered artifacts.

    Returns
    -------
    artifact_indices : numpy.ndarray
        Indices of the detected artifacts across all signals.

    Examples
    --------
    >>> signal1 = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01))
    >>> signal2 = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01) + 0.5)
    >>> artifacts = mutual_information_artifact_detection([signal1, signal2], threshold=0.05)
    >>> print(artifacts)
    """
    if threshold < 0:
        raise ValueError("Threshold must be a non-negative value.")
    if len(signals) == 0:
        raise ValueError("Signal list cannot be empty")

    def mutual_information(x, y, num_bins):
        # Create a 2D histogram for the joint distribution
        joint_histogram = np.histogram2d(x, y, bins=num_bins)[0]
        joint_prob = joint_histogram / np.sum(joint_histogram)

        # Marginal distributions for x and y
        x_marginal = np.sum(
            joint_prob, axis=1, keepdims=True
        )  # Reshape to (num_bins, 1)
        y_marginal = np.sum(
            joint_prob, axis=0, keepdims=True
        )  # Reshape to (1, num_bins)

        # Get non-zero joint probabilities and their corresponding marginals
        non_zero_joint = joint_prob[joint_prob > 0]

        # Create a meshgrid of x and y marginals matching the shape of the joint_prob
        x_marginal_mesh, y_marginal_mesh = np.meshgrid(
            x_marginal.flatten(), y_marginal.flatten(), indexing="ij"
        )

        # Select the marginals that correspond to the non-zero joint probabilities
        valid_x_marginals = x_marginal_mesh[joint_prob > 0]
        valid_y_marginals = y_marginal_mesh[joint_prob > 0]

        # Compute mutual information (using only non-zero entries)
        mi = np.sum(
            non_zero_joint
            * np.log(
                non_zero_joint / (valid_x_marginals * valid_y_marginals + 1e-9)
            )  # To avoid division by zero
        )
        return mi

    artifact_indices = []
    combined_signal = np.mean(np.array(signals), axis=0)

    for i in range(len(combined_signal)):
        mutual_infos = [
            mutual_information(combined_signal, signal, num_bins) for signal in signals
        ]
        if np.min(mutual_infos) < threshold:
            artifact_indices.append(i)

    return np.array(artifact_indices)
