"""
Signal Transforms Module for Physiological Signal Processing

This module provides comprehensive capabilities for physiological
signal processing including ECG, PPG, EEG, and other vital signs.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Object-oriented design with comprehensive classes
- Multiple processing methods and functions
- NumPy integration for numerical computations
- SciPy integration for advanced signal processing
- Performance optimization

Examples:
--------
Basic usage:
    >>> import numpy as np
    >>> from vitalDSP.transforms.vital_transformation import VitalTransformation
    >>> signal = np.random.randn(1000)
    >>> processor = VitalTransformation(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""

import numpy as np
from scipy import signal
from vitalDSP.filtering.artifact_removal import ArtifactRemoval
from vitalDSP.filtering.advanced_signal_filtering import AdvancedSignalFiltering
from vitalDSP.filtering.signal_filtering import BandpassFilter, SignalFiltering


class VitalTransformation:
    """
    A class to perform comprehensive signal processing on ECG and PPG signals using advanced filtering and artifact removal techniques.

    This class provides a series of transformations aimed at enhancing the quality of ECG and PPG signals by eliminating noise, detrending, normalizing, and enhancing critical points for easier detection. Each transformation step is modular and customizable.

    Parameters
    ----------
    signal : numpy.ndarray
        The input ECG or PPG signal to be transformed.
    fs : int, optional
        Sampling frequency of the signal. Default is 256 Hz.
    signal_type : str, optional
        Type of signal. Options: 'ECG', 'PPG', 'EEG'. Default is 'ECG'.

    Methods
    -------
    apply_transformations(options=None, method_order=None)
        Apply a sequence of transformations to process the signal, with options for customization.
    apply_artifact_removal(method='baseline_correction', options=None)
        Apply artifact removal using various techniques such as mean subtraction, baseline correction, or wavelet denoising.
    apply_bandpass_filter(options=None)
        Apply bandpass filtering with customizable filter type, cutoff frequencies, and order.
    apply_detrending(options=None)
        Detrend the signal with customizable options.
    apply_normalization(options=None)
        Normalize the signal to a specified range.
    apply_smoothing(options=None)
        Smooth the signal using different methods such as moving average or Gaussian.
    apply_enhancement(options=None)
        Enhance critical points in the signal using methods such as squaring or absolute value.
    apply_advanced_filtering(options=None)
        Apply advanced signal filtering techniques using the AdvancedSignalFiltering class.

    Examples
    --------
    >>> import numpy as np
    >>> from vitalDSP.transforms.vital_transformation import VitalTransformation
    >>>
    >>> # Example 1: Basic ECG signal transformation
    >>> ecg_signal = np.random.randn(1000)  # Simulated ECG signal
    >>> transformer = VitalTransformation(ecg_signal, fs=256, signal_type="ECG")
    >>> transformed_signal = transformer.apply_transformations()
    >>> print(f"Transformed signal shape: {transformed_signal.shape}")
    >>>
    >>> # Example 2: PPG signal with custom options
    >>> ppg_signal = np.random.randn(2000)  # Simulated PPG signal
    >>> transformer_ppg = VitalTransformation(ppg_signal, fs=128, signal_type="PPG")
    >>> options = {
    ...     'artifact_removal': 'baseline_correction',
    ...     'artifact_removal_options': {'cutoff': 0.5},
    ...     'bandpass_filter': {'lowcut': 0.5, 'highcut': 8.0, 'filter_order': 4, 'filter_type': 'butter'},
    ...     'detrending': {'detrend_type': 'linear'},
    ...     'normalization': {'normalization_range': (0, 1)}
    ... }
    >>> transformed_ppg = transformer_ppg.apply_transformations(options=options)
    >>> print(f"Transformed PPG shape: {transformed_ppg.shape}")
    >>>
    >>> # Example 3: Individual transformation methods
    >>> transformer = VitalTransformation(ecg_signal, fs=256, signal_type="ECG")
    >>> detrended = transformer.apply_detrending(method='linear')
    >>> normalized = transformer.apply_normalization(normalization_range=(0, 1))
    >>> filtered = transformer.apply_bandpass_filter(options={'lowcut': 0.5, 'highcut': 40, 'order': 4})
    >>> print(f"Detrended signal range: [{detrended.min():.3f}, {detrended.max():.3f}]")
    >>> print(f"Normalized signal range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    """

    def __init__(self, signal, fs=256, signal_type="ECG"):
        self.signal = signal
        self.fs = fs
        self.signal_type = signal_type

    def apply_transformations(self, options=None, method_order=None):
        """
        Apply a sequence of transformations to process the signal.

        Parameters
        ----------
        options : dict, optional
            A dictionary of options to customize each step in the transformation process. Default options will be used if not provided.
        method_order : list, optional
            A list specifying the order in which to apply the methods. Default is the order defined in the class.

        Returns
        -------
        transformed_signal : numpy.ndarray
            The fully transformed signal.

        Examples
        --------
        >>> signal = np.random.randn(1000)  # Example signal
        >>> options = {
        >>>     'artifact_removal': 'baseline_correction',
        >>>     'artifact_removal_options': {'cutoff': 0.5},
        >>>     'bandpass_filter': {'lowcut': 0.5, 'highcut': 30, 'filter_order': 4, 'filter_type': 'butter'},
        >>>     'detrending': {'detrend_type': 'linear'},
        >>>     'normalization': {'normalization_range': (0, 1)},
        >>>     'smoothing': {'smoothing_method': 'moving_average', 'window_size': 5, 'iterations': 2},
        >>>     'enhancement': {'enhance_method': 'square'},
        >>>     'advanced_filtering': {'filter_type': 'kalman_filter', 'options': {'R': 0.1, 'Q': 0.01}},
        >>> }
        >>> method_order = ['artifact_removal', 'bandpass_filter', 'detrending', 'normalization', 'smoothing', 'enhancement', 'advanced_filtering']
        >>> transformer = ECG_PPG_Transformation(signal, fs=256, signal_type='ECG')
        >>> transformed_signal = transformer.apply_transformations(options, method_order)
        >>> print(transformed_signal)
        """
        if options is None:
            options = {
                "artifact_removal": "baseline_correction",
                "artifact_removal_options": {"cutoff": 0.5},
                "bandpass_filter": {
                    "lowcut": 0.5,
                    "highcut": 30,
                    "filter_order": 4,
                    "filter_type": "butter",
                },
                "detrending": {"detrend_type": "linear"},
                "normalization": {"normalization_range": (0, 1)},
                "smoothing": {
                    "smoothing_method": "moving_average",
                    "window_size": 5,
                    "iterations": 2,
                },
                "enhancement": {"enhance_method": "square"},
                "advanced_filtering": {
                    "filter_type": "kalman_filter",
                    "options": {"R": 0.1, "Q": 0.01},
                },
            }

        if method_order is None:
            method_order = [
                "artifact_removal",
                "bandpass_filter",
                "detrending",
                "normalization",
                "smoothing",
                "enhancement",
                "advanced_filtering",
            ]

        # Filter out options that aren't actual methods
        method_order = [m for m in method_order if not m.endswith("_options")]

        for method in method_order:
            if method == "artifact_removal":
                self.apply_artifact_removal(
                    options.get("artifact_removal", "baseline_correction"),
                    options.get("artifact_removal_options", {}),
                )
            elif method == "bandpass_filter":
                self.apply_bandpass_filter(options.get("bandpass_filter", {}))
            elif method == "detrending":
                self.apply_detrending(options.get("detrending", {}))
            elif method == "normalization":
                self.apply_normalization(options.get("normalization", {}))
            elif method == "smoothing":
                self.apply_smoothing(options.get("smoothing", {}))
            elif method == "enhancement":
                self.apply_enhancement(options.get("enhancement", {}))
            elif method == "advanced_filtering":
                self.apply_advanced_filtering(options.get("advanced_filtering", {}))
            else:
                raise ValueError(f"Unknown method: {method}")

        return self.signal

    def apply_artifact_removal(self, method="baseline_correction", options=None):
        """
        Apply artifact removal to the signal.

        Parameters
        ----------
        method : str, optional
            The artifact removal method to use. Default is 'baseline_correction'.
        options : dict, optional
            Additional options specific to the selected method.

        Returns
        -------
        None

        Examples
        --------
        >>> signal = np.random.randn(1000)  # Example signal
        >>> transformer = ECG_PPG_Transformation(signal, fs=256, signal_type='ECG')
        >>> transformer.apply_artifact_removal(method='baseline_correction', options={'cutoff': 0.5})
        >>> print(transformer.signal)
        """
        if options is None:
            {"lowcut": 0.5, "highcut": 30, "filter_order": 4, "filter_type": "butter"}

        artifact_removal = ArtifactRemoval(self.signal)

        if method == "mean_subtraction":
            self.signal = artifact_removal.mean_subtraction()
        elif method == "baseline_correction":
            self.signal = artifact_removal.baseline_correction(
                cutoff=options.get("cutoff", 0.5), fs=self.fs
            )
        elif method == "median_filter_removal":
            self.signal = artifact_removal.median_filter_removal(
                kernel_size=options.get("kernel_size", 3)
            )
        elif method == "wavelet_denoising":
            self.signal = artifact_removal.wavelet_denoising(
                wavelet_type=options.get("wavelet_type", "db"),
                level=options.get("level", 1),
                order=options.get("order", 4),
            )
        elif method == "adaptive_filtering":
            reference_signal = options.get("reference_signal", None)
            if reference_signal is None:
                raise ValueError("Reference signal is required for adaptive filtering.")
            self.signal = artifact_removal.adaptive_filtering(
                reference_signal,
                learning_rate=options.get("learning_rate", 0.01),
                num_iterations=options.get("num_iterations", 100),
            )
        elif method == "notch_filter":
            self.signal = artifact_removal.notch_filter(
                freq=options.get("freq", 50), fs=self.fs, Q=options.get("Q", 30)
            )
        elif method == "pca_artifact_removal":
            self.signal = artifact_removal.pca_artifact_removal(
                num_components=options.get("num_components", 1)
            )
        elif method == "ica_artifact_removal":
            self.signal = artifact_removal.ica_artifact_removal(
                num_components=options.get("num_components", 1),
                max_iterations=options.get("max_iterations", 1000),
                tol=options.get("tol", 1e-5),
                seed=options.get("seed", 23),
            )
        else:
            raise ValueError(f"Unknown artifact removal method: {method}")

    def apply_bandpass_filter(self, options=None):
        """
        Apply bandpass filtering with customizable filter type, cutoff frequencies, and order.

        Parameters
        ----------
        options : dict, optional
            A dictionary of filter options.

        Returns
        -------
        None

        Examples
        --------
        >>> signal = np.random.randn(1000)  # Example signal
        >>> transformer = ECG_PPG_Transformation(signal, fs=256, signal_type='ECG')
        >>> transformer.apply_bandpass_filter(options={'lowcut': 0.5, 'highcut': 30, 'filter_order': 4, 'filter_type': 'butter'})
        >>> print(transformer.signal)
        """
        # Ensure the signal length is long enough for the filter
        if options is None:
            options = {
                "lowcut": 0.5,
                "highcut": 30,
                "filter_order": 4,
                "filter_type": "butter",
            }
        if len(self.signal) < 3 * options.get("filter_order", 4):
            raise ValueError("Signal too short for the specified filter order.")
        lowcut = options.get("lowcut", 0.2)
        highcut = options.get("highcut", 3.0)
        filter_order = options.get("filter_order", 4)
        filter_type = options.get("filter_type", "butter")

        bandpass_filter = BandpassFilter(band_type=filter_type, fs=self.fs)

        # Check signal length and adjust filter order if necessary
        padlen = 3 * max(
            len(bandpass_filter.signal_bypass(lowcut, filter_order, btype="low")[0]),
            len(bandpass_filter.signal_bypass(lowcut, filter_order, btype="low")[1]),
        )
        if len(self.signal) <= padlen:
            filter_order = max(
                1, int(len(self.signal) / 3)
            )  # Reduce filter order dynamically

        # Apply bandpass filter
        self.signal = bandpass_filter.signal_highpass_filter(
            data=self.signal,
            cutoff=lowcut,
            order=filter_order,
            a_pass=options.get("a_pass", 3),
            rp=options.get("rp", 4),
            rs=options.get("rs", 40),
        )

        self.signal = bandpass_filter.signal_lowpass_filter(
            data=self.signal,
            cutoff=highcut,
            order=filter_order,
            a_pass=options.get("a_pass", 3),
            rp=options.get("rp", 4),
            rs=options.get("rs", 40),
        )

    def apply_detrending(self, options=None):
        """
        Detrend the signal with customizable options using pure NumPy implementation.

        This method removes trends from the signal, which is crucial for physiological
        signal processing. Detrending eliminates baseline wander and drift that can
        interfere with feature extraction and analysis.

        Parameters
        ----------
        options : dict, optional
            A dictionary of options for detrending. Default options:
            {
                'detrend_type': 'linear',  # 'linear', 'constant', or 'polynomial'
                'polynomial_order': 3,      # Used when detrend_type='polynomial'
                'break_points': None,       # For piecewise detrending
            }

        Returns
        -------
        None
            Modifies self.signal in place.

        Notes
        -----
        Detrending Types:
        - **linear**: Removes linear trend using least squares fitting
        - **constant**: Removes DC component (mean value)
        - **polynomial**: Removes polynomial trend of specified order

        The implementation uses pure NumPy for better performance and removes
        the scipy dependency.

        Clinical Applications:
        - ECG: Removes baseline wander due to respiration or movement
        - PPG: Eliminates DC shifts and slow baseline drift
        - General: Ensures zero-mean signal for accurate feature extraction

        Examples
        --------
        >>> import numpy as np
        >>> from vitalDSP.transforms.vital_transformation import VitalTransformation
        >>>
        >>> # Example 1: Linear detrending (default)
        >>> signal_with_drift = np.random.randn(1000) + np.linspace(0, 10, 1000)
        >>> transformer = VitalTransformation(signal_with_drift, fs=256, signal_type='ECG')
        >>> transformer.apply_detrending(options={'detrend_type': 'linear'})
        >>> print(f"Signal mean after detrending: {np.mean(transformer.signal):.6f}")
        >>>
        >>> # Example 2: Constant (mean) detrending
        >>> ecg_signal = np.random.randn(1000) + 5.0  # Signal with DC offset
        >>> transformer2 = VitalTransformation(ecg_signal, fs=256, signal_type='ECG')
        >>> transformer2.apply_detrending(options={'detrend_type': 'constant'})
        >>> print(f"Signal mean after constant detrend: {np.mean(transformer2.signal):.6f}")
        >>>
        >>> # Example 3: Polynomial detrending
        >>> ppg_signal = np.random.randn(1000) + 0.001 * np.arange(1000)**2
        >>> transformer3 = VitalTransformation(ppg_signal, fs=128, signal_type='PPG')
        >>> transformer3.apply_detrending(options={
        ...     'detrend_type': 'polynomial',
        ...     'polynomial_order': 2
        ... })
        >>> print(f"Detrended PPG signal length: {len(transformer3.signal)}")
        """
        if options is None:
            options = {
                "detrend_type": "linear",
                "polynomial_order": 3,
                "break_points": None,
            }

        detrend_type = options.get("detrend_type", "linear")

        # Get signal length
        N = len(self.signal)

        if detrend_type == "constant":
            # Remove mean (DC component)
            self.signal = self.signal - np.mean(self.signal)

        elif detrend_type == "linear":
            # Remove linear trend using least squares
            # Create time vector
            x = np.arange(N)

            # Fit linear model: y = mx + b
            # Using normal equations: [m, b] = (X^T X)^-1 X^T y
            # where X = [x, ones]
            X = np.vstack([x, np.ones(N)]).T

            # Solve least squares
            coeffs = np.linalg.lstsq(X, self.signal, rcond=None)[0]

            # Compute and subtract trend
            trend = coeffs[0] * x + coeffs[1]
            self.signal = self.signal - trend

        elif detrend_type == "polynomial":
            # Remove polynomial trend
            polynomial_order = options.get("polynomial_order", 3)

            # Create time vector
            x = np.arange(N)

            # Fit polynomial
            coeffs = np.polyfit(x, self.signal, polynomial_order)

            # Compute and subtract polynomial trend
            trend = np.polyval(coeffs, x)
            self.signal = self.signal - trend

        else:
            raise ValueError(
                f"Unknown detrending method: {detrend_type}. "
                f"Supported methods: 'constant', 'linear', 'polynomial'"
            )

    def apply_normalization(self, options=None):
        """
        Normalize the signal to a specified range.

        Parameters
        ----------
        options : dict, optional
            A dictionary of normalization options.

        Returns
        -------
        None

        Examples
        --------
        >>> signal = np.random.randn(1000)  # Example signal
        >>> transformer = ECG_PPG_Transformation(signal, fs=256, signal_type='ECG')
        >>> transformer.apply_normalization(options={'normalization_range': (0, 1)})
        >>> print(transformer.signal)
        """
        if options is None:
            options = {"normalization_range": (0, 1)}

        normalization_range = options.get("normalization_range", (0, 1))
        min_val = np.min(self.signal)
        max_val = np.max(self.signal)

        # Avoid divide by zero warning
        if max_val - min_val > 0:
            self.signal = (self.signal - min_val) / (max_val - min_val) * (
                normalization_range[1] - normalization_range[0]
            ) + normalization_range[0]
        else:
            # If signal is constant, set to middle of range
            self.signal = np.full_like(
                self.signal, (normalization_range[0] + normalization_range[1]) / 2
            )

    def apply_smoothing(self, options=None):
        """
        Smooth the signal using different methods such as moving average or Gaussian.

        Parameters
        ----------
        options : dict, optional
            Options to customize the smoothing process.

        Returns
        -------
        None

        Examples
        --------
        >>> signal = np.random.randn(1000)  # Example signal
        >>> transformer = ECG_PPG_Transformation(signal, fs=256, signal_type='ECG')
        >>> transformer.apply_smoothing(options={'smoothing_method': 'moving_average', 'window_size': 5, 'iterations': 2})
        >>> print(transformer.signal)
        """
        if options is None:
            options = {
                "smoothing_method": "moving_average",
                "window_size": 5,
                "iterations": 2,
            }

        smoothing_method = options.get("smoothing_method", "moving_average")
        signal_filtering = SignalFiltering(self.signal)

        if smoothing_method == "moving_average":
            window_size = options.get("window_size", 5)
            iterations = options.get("iterations", 1)
            self.signal = signal_filtering.moving_average(
                window_size, iterations=iterations
            )
        elif smoothing_method == "gaussian":
            sigma = options.get("sigma", 1.0)
            iterations = options.get("iterations", 1)
            self.signal = signal_filtering.gaussian(sigma=sigma, iterations=iterations)
        else:
            raise ValueError(f"Unknown smoothing method: {smoothing_method}")

    def apply_enhancement(self, options=None):
        """
        Enhance critical points in the signal using methods such as squaring or absolute value.

        Parameters
        ----------
        options : dict, optional
            Options to customize the enhancement process.

        Returns
        -------
        None

        Examples
        --------
        >>> signal = np.random.randn(1000)  # Example signal
        >>> transformer = ECG_PPG_Transformation(signal, fs=256, signal_type='ECG')
        >>> transformer.apply_enhancement(options={'enhance_method': 'square'})
        >>> print(transformer.signal)
        """
        if options is None:
            options = {"enhance_method": "square"}

        enhance_method = options.get("enhance_method", "square")

        if enhance_method == "square":
            self.signal = np.square(self.signal)
        elif enhance_method == "abs":
            self.signal = np.abs(self.signal)
        elif enhance_method == "gradient":
            # Use the gradient to enhance rapid changes in the signal
            self.signal = np.gradient(self.signal)
        else:
            raise ValueError(f"Unknown enhancement method: {enhance_method}")

    def apply_advanced_filtering(self, options=None):
        """
        Apply advanced signal filtering using the AdvancedSignalFiltering class.

        Parameters
        ----------
        options : dict, optional
            Options to customize the advanced filtering process.

        Returns
        -------
        None

        Examples
        --------
        >>> signal = np.random.randn(1000)  # Example signal
        >>> transformer = ECG_PPG_Transformation(signal, fs=256, signal_type='ECG')
        >>> transformer.apply_advanced_filtering(options={'filter_type': 'kalman_filter', 'R': 0.1, 'Q': 0.01})
        >>> print(transformer.signal)
        """
        if options is None:
            options = {"filter_type": "kalman_filter", "R": 0.1, "Q": 0.01}

        filter_type = options.get("filter_type", "kalman_filter")
        advanced_filtering = AdvancedSignalFiltering(
            self.signal
        )  # Initialize with the signal

        if filter_type == "kalman_filter":
            R = options.get("R", 0.1)
            Q = options.get("Q", 0.01)
            self.signal = advanced_filtering.kalman_filter(
                R=R, Q=Q
            )  # Call method from instance
        elif filter_type == "optimization_based_filtering":
            target_signal = options.get("target_signal", self.signal)
            loss_type = options.get("loss_type", "mse")
            learning_rate = options.get("learning_rate", 0.01)
            iterations = options.get("iterations", 100)
            self.signal = advanced_filtering.optimization_based_filtering(
                target=target_signal,
                loss_type=loss_type,
                learning_rate=learning_rate,
                iterations=iterations,
            )
        elif filter_type == "gradient_descent_filter":
            target_signal = options.get("target_signal", self.signal)
            learning_rate = options.get("learning_rate", 0.01)
            iterations = options.get("iterations", 100)
            self.signal = advanced_filtering.gradient_descent_filter(
                target=target_signal, learning_rate=learning_rate, iterations=iterations
            )
        elif filter_type == "ensemble_filtering":
            filters = options.get("filters", [advanced_filtering.kalman_filter])
            method = options.get("method", "mean")
            weights = options.get("weights", None)
            num_iterations = options.get("num_iterations", 10)
            learning_rate = options.get("learning_rate", 0.01)
            self.signal = advanced_filtering.ensemble_filtering(
                filters=filters,
                method=method,
                weights=weights,
                num_iterations=num_iterations,
                learning_rate=learning_rate,
            )
        elif filter_type == "convolution_based_filter":
            kernel_type = options.get("kernel_type", "smoothing")
            kernel_size = options.get("kernel_size", 3)
            self.signal = advanced_filtering.convolution_based_filter(
                kernel_type=kernel_type, kernel_size=kernel_size
            )
        elif filter_type == "attention_based_filter":
            attention_type = options.get("attention_type", "uniform")
            size = options.get("size", 5)
            self.signal = advanced_filtering.attention_based_filter(
                attention_type=attention_type, size=size, **options.get("kwargs", {})
            )
        elif filter_type == "adaptive_filtering":
            desired_signal = options.get("desired_signal", self.signal)
            mu = options.get("mu", 0.5)
            filter_order = options.get("filter_order", 4)
            self.signal = advanced_filtering.adaptive_filtering(
                desired_signal=desired_signal, mu=mu, filter_order=filter_order
            )
        else:
            raise ValueError(f"Unknown advanced filtering method: {filter_type}")
