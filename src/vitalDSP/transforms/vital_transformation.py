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
        Type of signal ('ecg' or 'ppg'). Default is 'ecg'.

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
    >>> transformer = ECG_PPG_Transformation(signal, fs=256, signal_type='ecg')
    >>> transformed_signal = transformer.apply_transformations(options, method_order)
    >>> print(transformed_signal)
    """

    def __init__(self, signal, fs=256, signal_type="ecg"):
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
        >>> transformer = ECG_PPG_Transformation(signal, fs=256, signal_type='ecg')
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
        >>> transformer = ECG_PPG_Transformation(signal, fs=256, signal_type='ecg')
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
        >>> transformer = ECG_PPG_Transformation(signal, fs=256, signal_type='ecg')
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
        Detrend the signal with customizable options.

        Parameters
        ----------
        options : dict, optional
            A dictionary of options for detrending.

        Returns
        -------
        None

        Examples
        --------
        >>> signal = np.random.randn(1000)  # Example signal
        >>> transformer = ECG_PPG_Transformation(signal, fs=256, signal_type='ecg')
        >>> transformer.apply_detrending(options={'detrend_type': 'linear'})
        >>> print(transformer.signal)
        """
        if options is None:
            options = {
                "detrend_type": "linear",
                "lowcut": 0.5,
                "highcut": 30,
                "filter_order": 4,
                "filter_type": "butter",
            }

        detrend_type = options.get("detrend_type", "linear")
        if detrend_type == "linear":
            self.signal = signal.detrend(self.signal, type="linear")
        elif detrend_type == "constant":
            self.signal = signal.detrend(self.signal, type="constant")
        else:
            raise ValueError(f"Unknown detrending method: {detrend_type}")

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
        >>> transformer = ECG_PPG_Transformation(signal, fs=256, signal_type='ecg')
        >>> transformer.apply_normalization(options={'normalization_range': (0, 1)})
        >>> print(transformer.signal)
        """
        if options is None:
            options = {"normalization_range": (0, 1)}

        normalization_range = options.get("normalization_range", (0, 1))
        min_val = np.min(self.signal)
        max_val = np.max(self.signal)
        self.signal = (self.signal - min_val) / (max_val - min_val) * (
            normalization_range[1] - normalization_range[0]
        ) + normalization_range[0]

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
        >>> transformer = ECG_PPG_Transformation(signal, fs=256, signal_type='ecg')
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
        >>> transformer = ECG_PPG_Transformation(signal, fs=256, signal_type='ecg')
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
        >>> transformer = ECG_PPG_Transformation(signal, fs=256, signal_type='ecg')
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
