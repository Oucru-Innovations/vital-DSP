import numpy as np
from vitalDSP.utils.peak_detection import PeakDetection
from vitalDSP.preprocess.preprocess_operations import estimate_baseline
from scipy.stats import skew, linregress
import logging as logger
from scipy.signal import savgol_filter
from vitalDSP.transforms.vital_transformation import VitalTransformation


class WaveformMorphology:
    """
    A class for computing morphological features from physiological waveforms (ECG, PPG, EEG).

    Attributes:
        waveform (np.array): The waveform signal (ECG, PPG, EEG).
        fs (int): The sampling frequency of the signal in Hz. Default is 1000 Hz.
        signal_type (str): The type of signal ('ECG', 'PPG', 'EEG').
        simple_mode (bool, optional): If True, uses simplified diastolic peak detection (midpoint-based). Default is True.
    """

    def __init__(
        self,
        waveform,
        fs=256,
        qrs_ratio=0.05,
        signal_type="ECG",
        peak_config=None,
        simple_mode=True,
        options=None,
    ):
        """
        Initializes the WaveformMorphology object.

        Args:
            waveform (np.array): The waveform signal.
            fs (int): The sampling frequency of the signal in Hz. Default is 256 Hz.
            qrs_ratio (float): The ratio for QRS detection. Default is 0.05.
            signal_type (str): The type of signal (ECG, PPG, EEG).
            peak_config : dict, optional
                Configuration for peak detection parameters. If None, default settings are used.
                For ECG: {
                    "distance": 50,
                    "window_size": 7,
                    "threshold_factor": 1.6,
                    "search_window": 6,
                }
                For PPG: {
                    "distance": 50,
                    "window_size": 7,
                    "threshold_factor": 0.8,
                    "search_window": 6,
                    "fs": fs,
                    "dicrotic_config": {
                        "slope_quartile_factor": 0.25,
                        "search_range_start": 0.1,
                        "search_range_end": 0.3,
                    },
                    "diastolic_config": {  # New settings for diastolic peaks
                        "search_range_start": 0.2,  # Start 20% after notch
                        "search_range_end": 0.8,    # End 80% before trough
                        "min_height_factor": 0.1,   # Min height as fraction of notch-to-trough amp
                        "min_distance": 5           # Min samples from notch
                    }
                }
        """
        self.waveform = np.array(waveform)
        self.fs = fs  # Sampling frequency
        self.signal_type = signal_type  # 'ECG', 'PPG', 'EEG'
        self.qrs_ratio = qrs_ratio
        self.simple_mode = simple_mode
        self._cache = {}  # Cache for computed results

        # Precompute derivatives for reuse
        self._signal_derivative = np.diff(self.waveform)
        self._signal_second_derivative = np.diff(self._signal_derivative)
        # self._smoothed_signal = savgol_filter(self.waveform, window_length=5, polyorder=2)
        if options is not None:
            # options = {
            # 'artifact_removal': 'baseline_correction',
            # 'artifact_removal_options': {'cutoff': 0.5},
            # 'bandpass_filter': {'lowcut': 0.2, 'highcut': 10, 'filter_order': 4, 'filter_type': 'butter'},
            # 'detrending': {'detrend_type': 'linear'},
            # 'normalization': {'normalization_range': (0, 20)},
            # "smoothing": {
            #     "smoothing_method": "moving_average",
            #     "window_size": 5,
            #     "iterations": 2,
            # },
            # 'enhancement': {'enhance_method': 'square'},
            # 'advanced_filtering': {'filter_type': 'kalman_filter', 'options': {'R': 0.1, 'Q': 0.01}},
            # }
            method_order = options.keys()
            transformer = VitalTransformation(
                self.waveform, fs=fs, signal_type=signal_type
            )
            self._smoothed_signal = transformer.apply_transformations(
                options, method_order
            )
        else:
            self._smoothed_signal = savgol_filter(
                self.waveform, window_length=5, polyorder=2
            )

        # Default peak detection configurations
        default_ecg_config = {
            "distance": 50,
            "window_size": 7,
            "threshold_factor": 1.6,
            "search_window": 6,
        }

        default_ppg_config = {
            "distance": 50,
            "window_size": 7,
            "threshold_factor": 0.8,
            "search_window": 6,
            "fs": self.fs,
            "dicrotic_config": {
                "slope_quartile_factor": 0.25,
                "search_range_start": 0.1,
                "search_range_end": 0.3,
            },
            "diastolic_config": {
                "search_range_start": 0.2,
                "search_range_end": 0.8,
                "min_height_factor": 0.1,
                "min_distance": 5,
                "min_distance_to_trough": 5,  # New: Min distance from trough
                "slope_window": 3,  # New: Window for slope calculation
            },
        }

        # Store the configuration for later use
        self.peak_config = (
            peak_config
            if peak_config is not None
            else (default_ecg_config if signal_type == "ECG" else default_ppg_config)
        )

        if signal_type == "ECG":
            detector = PeakDetection(self.waveform, "ecg_r_peak", **(self.peak_config))
            self.r_peaks = detector.detect_peaks()
        elif signal_type == "PPG":
            detector = PeakDetection(
                self.waveform, "ppg_systolic_peaks", **(self.peak_config)
            )
            self.systolic_peaks = detector.detect_peaks()

        elif signal_type == "EEG":
            detector = PeakDetection(self.waveform)
            self.eeg_peaks = detector.detect_peaks()
        else:
            raise ValueError("Invalid signal type. Supported types are ECG, PPG, EEG.")

        self.q_valleys = None
        self.s_valleys = None
        self.diastolic_troughs = None
        self.diastolic_peaks = None
        self.t_peaks = None
        self.p_peaks = None
        self.dicrotic_notches = None

    def _cache_result(self, key, method, *args, **kwargs):
        """Cache the result of a method call."""
        cache_key = (key, tuple(args), frozenset(kwargs.items()))
        if cache_key not in self._cache:
            self._cache[cache_key] = method(*args, **kwargs)
        return self._cache[cache_key]

    def detect_troughs(self, systolic_peaks=None):
        """
        Detects the troughs (valleys) in the PPG waveform between systolic peaks.
        In simple_mode, uses the minimum value between adjacent peaks; otherwise, uses flat segment detection.

        Parameters
        ----------
        systolic_peaks : np.array
            Indices of detected systolic peaks in the PPG waveform.

        Returns
        -------
        troughs : np.array
            Indices of the detected troughs between the systolic peaks.

        Example
        -------
        >>> waveform = np.sin(np.linspace(0, 2*np.pi, 100))  # Simulated PPG signal
        >>> wm = WaveformMorphology(waveform, signal_type="PPG")
        >>> peaks = PeakDetection(waveform).detect_peaks()
        >>> troughs = wm.detect_troughs(peaks)
        >>> print(f"Troughs: {troughs}")
        """
        if self.signal_type != "PPG":
            logger.warning(
                "This function is designed to work with PPG signals. "
                "Other signal types may result in unexpected behaviour"
            )
        if systolic_peaks is None:
            systolic_peaks = self.systolic_peaks

        diastolic_troughs = []

        if self.simple_mode:
            # Simple mode: Find the minimum value between adjacent systolic peaks
            for i in range(len(systolic_peaks) - 1):
                peak_start = systolic_peaks[i]
                peak_end = systolic_peaks[i + 1]
                if peak_end <= peak_start:
                    continue
                segment = self.waveform[peak_start:peak_end]
                trough_idx = peak_start + np.argmin(segment)  # Index of minimum value
                diastolic_troughs.append(trough_idx)
        else:
            # Original mode: Use derivative and flat segments
            signal_derivative = np.diff(self.waveform)
            for i in range(len(systolic_peaks) - 1):
                peak_start = systolic_peaks[i]
                peak_end = systolic_peaks[i + 1]
                search_start = peak_start + (peak_end - peak_start) // 2
                search_end = peak_end

                if search_start >= search_end or search_start < 0:
                    continue

                # Calculate adaptive flatness threshold based on MAD of the derivative
                local_derivative = signal_derivative[search_start:search_end]
                mad_derivative = np.median(
                    np.abs(local_derivative - np.median(local_derivative))
                )
                adaptive_threshold = 0.5 * mad_derivative  # 50% of local MAD

                # Identify flat segments
                flat_segment = [
                    j
                    for j in range(search_start, search_end - 1)
                    if abs(signal_derivative[j]) < adaptive_threshold
                ]

                # Find midpoint of the longest flat segment
                if flat_segment:
                    flat_segment_groups = np.split(
                        flat_segment, np.where(np.diff(flat_segment) != 1)[0] + 1
                    )
                    longest_flat_segment = max(flat_segment_groups, key=len)
                    trough_index = longest_flat_segment[len(longest_flat_segment) // 2]
                    diastolic_troughs.append(trough_index)

        diastolic_troughs = np.array(diastolic_troughs)
        self.diastolic_troughs = diastolic_troughs
        return diastolic_troughs
        # # Calculate the derivative of the signal for slope information
        # signal_derivative = np.diff(self.waveform)

        # for i in range(len(systolic_peaks) - 1):
        #     # Define the search range as the midpoint between two adjacent systolic peaks
        #     peak_start = systolic_peaks[i]
        #     peak_end = systolic_peaks[i + 1]
        #     search_start = peak_start + (peak_end - peak_start) // 2
        #     search_end = peak_end

        #     # Ensure search range is valid
        #     if search_start >= search_end or search_start < 0:
        #         continue

        #     # Narrow down to the section within the search range
        #     search_section = self.waveform[search_start:search_end]
        #     search_derivative = signal_derivative[search_start:search_end-1]

        #     # Find the index where the slope is closest to zero with a preference for a negative slope
        #     candidate_troughs = [
        #         idx for idx, slope in enumerate(search_derivative) if slope <= 0
        #     ]

        #     if candidate_troughs:
        #         trough_index = candidate_troughs[np.argmin(np.abs(search_derivative[candidate_troughs]))]
        #         diastolic_troughs.append(search_start + trough_index)
        # return np.array(diastolic_troughs)

    def detect_dicrotic_notches(self, systolic_peaks=None, diastolic_troughs=None):
        """
        Detects the dicrotic notches in a PPG waveform using second derivative.

        Parameters
        ----------
        systolic_peaks : np.array
            Indices of detected systolic peaks in the PPG waveform.
        diastolic_troughs : np.array
            Indices of the detected troughs between the systolic peaks.

        Returns
        -------
        notches : np.array
            Indices of detected dicrotic notches in the PPG waveform.
        """
        if self.signal_type != "PPG":
            raise ValueError("Notches can only be detected for PPG signals.")
        if systolic_peaks is None:
            systolic_peaks = self.systolic_peaks
        if diastolic_troughs is None:
            diastolic_troughs = self.detect_troughs(systolic_peaks=systolic_peaks)

        notches = []
        signal_derivative = np.diff(self.waveform)
        min_length = min(len(systolic_peaks) - 1, len(diastolic_troughs))
        systolic_peaks = systolic_peaks[: min_length + 1]
        diastolic_troughs = diastolic_troughs[:min_length]

        if self.simple_mode:
            # Simple mode: Full range, closest to zero derivative
            for i in range(min_length):
                peak = systolic_peaks[i]
                trough = diastolic_troughs[i]
                if trough <= peak:
                    notches.append(peak + (trough - peak) // 2)  # Fallback to midpoint
                    continue

                # search_start = peak
                # search_end = trough
                # search_deriv = signal_derivative[search_start:search_end]
                interval_length = trough - peak
                search_start = peak + int(
                    interval_length * 0.15
                )  # Start of 2nd quarter
                search_end = peak + int(interval_length * 0.45)  # End of 3rd quarter
                search_start = min(max(search_start, peak + 1), trough - 1)
                search_end = min(max(search_end, search_start + 1), trough)
                search_deriv = signal_derivative[search_start:search_end]

                if search_deriv.size > 0:
                    notch_idx = search_start + np.argmin(
                        np.abs(search_deriv)
                    )  # Closest to zero
                    notches.append(
                        max(peak + 1, min(notch_idx, trough - 1) - 2)
                    )  # Ensure bounds
                else:
                    notches.append(peak + (trough - peak) // 2)  # Fallback if empty

            notches = np.array(notches)
            target_length = len(diastolic_troughs)
            if len(notches) < target_length and target_length - len(notches) <= 1:
                # Impute missing notches with derivative closest to zero
                for i in range(len(notches), target_length):
                    peak = systolic_peaks[i]
                    trough = diastolic_troughs[i]
                    if trough <= peak:
                        imputed_notch = peak + (trough - peak) // 3
                    else:
                        search_deriv = signal_derivative[peak:trough]
                        min_deriv_idx = np.argmin(np.abs(search_deriv))
                        imputed_notch = peak + min_deriv_idx
                        imputed_notch = max(peak + 1, min(imputed_notch, trough - 1))
                    notches = np.append(notches, imputed_notch)
        else:
            # Non-simple mode: Search 2nd and 3rd quarters (25% to 75%) of [peak, trough]
            for i in range(min_length):
                peak = systolic_peaks[i]
                trough = diastolic_troughs[i]
                if trough <= peak:
                    notches.append(peak + (trough - peak) // 2)
                    continue

                # Divide [peak, trough] into 4 parts, take 2nd and 3rd (25% to 75%)
                interval_length = trough - peak
                search_start = peak + int(
                    interval_length * 0.25
                )  # Start of 2nd quarter
                search_end = peak + int(interval_length * 0.75)  # End of 3rd quarter
                search_start = min(max(search_start, peak + 1), trough - 1)
                search_end = min(max(search_end, search_start + 1), trough)

                search_deriv = signal_derivative[search_start:search_end]
                if search_deriv.size > 0:
                    notch_idx = search_start + np.argmin(
                        np.abs(search_deriv)
                    )  # Closest to zero
                    notches.append(max(search_start, min(notch_idx, search_end - 1)))
                else:
                    notches.append(search_start + (search_end - search_start) // 2)

        # Sort and ensure length consistency
        notches = np.sort(notches)[: len(diastolic_troughs)]
        self.dicrotic_notches = notches
        return notches

    def detect_diastolic_peak(self, notches=None, diastolic_troughs=None):
        """
        Detect diastolic peaks in PPG signals based on notches and diastolic troughs.

        Parameters
        ----------
        notches : list of int
            Indices of detected notches in the PPG signal.
        diastolic_troughs : list of int
            Indices of diastolic troughs in the PPG signal.

        Returns
        -------
        diastolic_peaks : list of int
            Indices of diastolic peaks detected in the PPG signal.
        """
        if self.signal_type != "PPG":
            raise ValueError("Diastolic peaks can only be detected for PPG signals.")
        if diastolic_troughs is None:
            diastolic_troughs = self.detect_troughs(systolic_peaks=self.systolic_peaks)
        if notches is None:
            notches = self.detect_dicrotic_notches(
                systolic_peaks=self.systolic_peaks, diastolic_troughs=diastolic_troughs
            )

        # Get diastolic config from peak_config
        # diastolic_config = self.peak_config.get(
        #     "diastolic_config",
        #     {
        #         "search_range_start": 0.2,
        #         "search_range_end": 0.4,
        #         "min_height_factor": 0.1,
        #         "min_distance": 8,
        #         "min_distance_to_trough": 8,
        #         "slope_window": 5,
        #     },
        # )

        diastolic_peaks = []
        min_length = min(len(notches), len(diastolic_troughs))
        notches = notches[:min_length]
        diastolic_troughs = diastolic_troughs[:min_length]

        signal_derivative = np.diff(self.waveform)
        signal_second_derivative = np.diff(signal_derivative)
        diastolic_peaks = []

        if self.simple_mode:
            # Simple mode: Full range [notch, trough], find flattest derivative
            for i in range(min_length):
                notch = notches[i]
                trough = diastolic_troughs[i]
                if trough <= notch + 2:  # Need at least 3 points for derivative
                    diastolic_peaks.append(notch + (trough - notch) // 2)
                    continue

                # search_start = notch
                # search_end = trough
                # Divide [notch, trough] into parts, take 2nd and 3rd (20% to 45%)
                interval_length = trough - notch
                search_start = notch + int(
                    interval_length * 0.1
                )  # Start of 2nd quarter
                search_end = notch + int(interval_length * 0.45)  # End of 3rd quarter
                search_start = min(max(search_start, notch + 1), trough - 1)
                search_end = min(max(search_end, search_start + 1), trough)

                search_deriv = signal_derivative[search_start:search_end]
                if search_deriv.size > 0:
                    min_deriv_idx = np.argmin(np.abs(search_deriv))
                    peak_idx = search_start + min_deriv_idx + 2  # Shift forth 2 units
                    peak_idx = max(
                        notch + 1, min(peak_idx, trough - 1)
                    )  # Ensure bounds
                    diastolic_peaks.append(peak_idx)
                else:
                    diastolic_peaks.append(notch + (trough - notch) // 3)
        else:
            # Non-simple mode: Search 2nd and 3rd quarters (25% to 75%) of [notch, trough]
            for i in range(min_length):
                notch = notches[i]
                trough = diastolic_troughs[i]
                if trough <= notch + 2:  # Need at least 3 points for derivative
                    diastolic_peaks.append(notch + (trough - notch) // 2)
                    continue

                # Divide [notch, trough] into 4 parts, take 2nd and 3rd (25% to 75%)
                interval_length = trough - notch
                search_start = notch + int(
                    interval_length * 0.25
                )  # Start of 2nd quarter
                search_end = notch + int(interval_length * 0.75)  # End of 3rd quarter
                search_start = min(max(search_start, notch + 1), trough - 1)
                search_end = min(max(search_end, search_start + 1), trough)

                search_deriv = signal_second_derivative[search_start:search_end]
                if search_deriv.size > 0:
                    min_deriv_idx = np.argmin(np.abs(search_deriv))
                    peak_idx = search_start + min_deriv_idx + 2  # Shift forth 3 units
                    peak_idx = max(
                        search_start, min(peak_idx, search_end - 1)
                    )  # Ensure bounds
                    diastolic_peaks.append(peak_idx)
                else:
                    diastolic_peaks.append(
                        search_start + (search_end - search_start) // 2
                    )

        diastolic_peaks = np.array(diastolic_peaks)
        self.diastolic_peaks = diastolic_peaks
        return diastolic_peaks
        # # signal_derivative = np.diff(self.waveform)

        # # Light smoothing to stabilize slope detection
        # # smoothed_signal = savgol_filter(self.waveform, window_length=5, polyorder=2)
        # smoothed_signal = self._smoothed_signal

        # for i in range(min_length):
        #     notch = notches[i]
        #     trough = diastolic_troughs[i]
        #     min_dist = diastolic_config["min_distance"]
        #     min_dist_trough = diastolic_config["min_distance_to_trough"]

        #     if trough <= notch + min_dist + min_dist_trough:
        #         diastolic_peaks.append(notch + (trough - notch) // 2)
        #         continue

        #     search_start = notch + min_dist
        #     search_end = trough - min_dist_trough

        #     if (
        #         search_end <= search_start + 2
        #     ):  # Ensure at least 3 samples for slope calc
        #         diastolic_peaks.append(notch + (trough - notch) // 2)
        #         continue

        #     if self.simple_mode:
        #         # Simple mode: Find point with smallest slope
        #         search_segment = smoothed_signal[search_start:search_end]
        #         slopes = np.abs(np.diff(search_segment))  # Absolute slope
        #         if len(slopes) > 0:
        #             min_slope_idx = np.argmin(slopes)
        #             peak_idx = search_start + min_slope_idx
        #             # Ensure it's not too close to boundaries
        #             if peak_idx < search_start + 1 or peak_idx > search_end - 1:
        #                 peak_idx = search_start + (search_end - search_start) // 2
        #             diastolic_peaks.append(peak_idx)
        #         else:
        #             diastolic_peaks.append(notch + (trough - notch) // 2)
        #     else:
        #         # Non-simple mode: Use find_peaks
        #         search_segment = self.waveform[search_start:search_end]
        #         notch_value = self.waveform[notch]
        #         trough_value = self.waveform[trough]
        #         min_height = (notch_value - trough_value) * diastolic_config[
        #             "min_height_factor"
        #         ] + trough_value

        #         peaks, _ = find_peaks(
        #             search_segment,
        #             height=min_height,
        #             distance=diastolic_config["min_distance"],
        #         )

        #         if peaks.size > 0:
        #             peak_idx = search_start + peaks[0]
        #             diastolic_peaks.append(peak_idx)
        #         else:
        #             peak_idx = search_start + np.argmax(search_segment)
        #             diastolic_peaks.append(peak_idx)

        # # min_length = min(len(notches), len(diastolic_troughs))
        # # notches = notches[:min_length]
        # # diastolic_troughs = diastolic_troughs[:min_length]

        # # for i in range(min_length):
        # #     notch = notches[i]
        # #     trough = diastolic_troughs[i]
        # #     if trough <= notch:
        # #         diastolic_peaks.append(notch)  # Fallback to notch
        # #         continue

        # #     # Search full range from notch to trough
        # #     search_segment = self.waveform[notch:trough]
        # #     peak_idx = notch + np.argmax(search_segment)  # Find max value
        # #     diastolic_peaks.append(peak_idx)

        # diastolic_peaks = np.array(diastolic_peaks)
        # self.diastolic_peaks = diastolic_peaks
        # return diastolic_peaks

        # for i in range(len(notches)):
        #     notch = notches[i]
        #     trough = (
        #         diastolic_troughs[i]
        #         if i < len(diastolic_troughs)
        #         else len(self.waveform) - 1
        #     )

        #     # Define the initial search range: notch to halfway to the trough
        #     search_start = notch
        #     search_end = notch + (trough - notch) // 2

        #     if search_end <= search_start or search_end > len(self.waveform):
        #         diastolic_peaks.append(notch)
        #         continue

        #     search_segment = self.waveform[search_start:search_end]
        #     segment_derivative = np.diff(search_segment)

        #     candidate_peaks = [
        #         idx
        #         for idx in range(1, len(segment_derivative))
        #         if segment_derivative[idx - 1] < 0 and segment_derivative[idx] >= 0
        #     ]

        #     # Convert candidates to absolute indices and select the most prominent peak if available
        #     if candidate_peaks:
        #         candidate_indices = [search_start + idx for idx in candidate_peaks]
        #         diastolic_peak_idx = max(
        #             candidate_indices, key=lambda x: self.waveform[x]
        #         )
        #         diastolic_peaks.append(diastolic_peak_idx)
        #     else:
        #         # Fallback assignment: if no diastolic peak is detected, use the notch as the diastolic peak
        #         diastolic_peaks.append(notch)

        # # min_length = min(len(notches), len(diastolic_troughs))
        # # notches = notches[:min_length]
        # # diastolic_troughs = diastolic_troughs[:min_length]

        # # for i in range(min_length):
        # #     notch = notches[i]
        # #     trough = diastolic_troughs[i]
        # #     if trough <= notch:
        # #         diastolic_peaks.append(notch)  # Fallback if order is wrong
        # #         continue
        # #     search_segment = self.waveform[notch:trough]
        # #     peak_idx = notch + np.argmax(search_segment)  # Find max in full range
        # #     diastolic_peaks.append(peak_idx)

        # diastolic_peaks = np.array(diastolic_peaks)
        # self.diastolic_peaks = diastolic_peaks
        # return diastolic_peaks

    def detect_q_valley(self, r_peaks=None):
        """
        Detects the Q valley (local minimum) in the ECG waveform just before each R peak.

        Parameters
        ----------
        r_peaks : np.array
            Indices of detected R peaks in the ECG waveform.

        Returns
        -------
        q_valleys : list of int
            Indices of the Q valley (local minimum) for each R peak.
        """
        if self.signal_type != "ECG":
            raise ValueError("Q valleys can only be detected for ECG signals.")
        if r_peaks is None:
            r_peaks = self.r_peaks
        q_valleys = []
        for i, r_peak in enumerate(r_peaks):
            # Set the end of the search range to be the R peak
            search_end = r_peak

            # Determine the start of the search range
            if i == 0:
                # For the first R peak, start from the beginning of the signal
                search_start = max(
                    0, search_end - int(self.fs * self.qrs_ratio)
                )  # Approx 200ms window
            else:
                # For subsequent R peaks, set the start as the midpoint to the previous R peak
                # search_start = (r_peaks[i - 1] + r_peak) // 2
                search_start = r_peak - int(self.qrs_ratio * self.fs)

            # Ensure the search range is valid
            if search_start < search_end:
                # Extract the signal segment within the search range
                signal_segment = self.waveform[search_start:search_end]

                # Detect the Q valley as the minimum point in the segment
                q_valley_idx = np.argmin(signal_segment) + search_start
                q_valleys.append(q_valley_idx)
        q_valleys = np.array(q_valleys)
        self.q_valleys = q_valleys
        return q_valleys

    def detect_s_valley(self, r_peaks=None):
        """
        Detects the S valleys (local minima after each R peak).

        Parameters
        ----------
        r_peaks : list of int
            Indices of detected R peaks in the ECG waveform.

        Returns
        -------
        s_valleys : list of int
            Indices of the S valleys for each R peak.
        """
        if self.signal_type != "ECG":
            raise ValueError("Q valleys can only be detected for ECG signals.")
        if r_peaks is None:
            r_peaks = self.r_peaks

        s_valleys = []

        for i, r_peak in enumerate(r_peaks):
            # Set the start of the search range to be the R peak
            search_start = r_peak

            # Determine the end of the search range
            if i == len(r_peaks) - 1:
                # For the last R peak, set the end to the end of the signal or a 200ms window after the R peak
                search_end = min(
                    len(self.waveform) - 1, search_start + int(self.fs * 0.1)
                )
            else:
                # For other R peaks, set the end as the midpoint to the next R peak
                # search_end = (r_peak + r_peaks[i + 1]) // 2
                search_end = r_peak + int(self.qrs_ratio * self.fs)

            # Ensure the search range is valid
            if search_start < search_end:
                # Extract the signal segment within the search range
                signal_segment = self.waveform[search_start:search_end]

                # Detect the S valley as the minimum point in the segment
                s_valley_idx = np.argmin(signal_segment) + search_start
                s_valleys.append(s_valley_idx)
        s_valleys = np.array(s_valleys)
        self.s_valleys = s_valleys
        return s_valleys

    def detect_p_peak(self, r_peaks=None, q_valleys=None):
        """
        Detects the P peak (local maximum) in the ECG waveform just before each Q valley.

        Parameters
        ----------
        q_valleys : list of int
            Indices of detected Q valleys in the ECG waveform.

        Returns
        -------
        p_peaks : list of int
            Indices of the P peaks (local maximum) for each Q valley.
        """
        if self.signal_type != "ECG":
            raise ValueError("P peaks can only be detected for ECG signals.")
        if r_peaks is None:
            r_peaks = self.r_peaks
        if q_valleys is None:
            q_valleys = self.detect_q_valley(r_peaks=r_peaks)
        p_peaks = []

        for i in range(1, len(r_peaks)):
            # Define search range with midpoint as start and Q valley as end
            midpoint = (r_peaks[i - 1] + r_peaks[i]) // 2

            q_valley = q_valleys[i - 1]  # Corresponding Q valley for the R peak pair
            # Trick to shift the starting point of the search range
            if q_valley < midpoint:
                q_valley = q_valleys[i]

            # Ensure search range has positive length
            if midpoint < q_valley:
                search_start = midpoint
                search_end = q_valley
            else:
                # Skip if search range is invalid
                continue

            # Extract signal segment within search range
            signal_segment = self.waveform[search_start:search_end]

            # Detect the maximum in this segment
            p_peak_idx = np.argmax(signal_segment) + search_start
            p_peaks.append(p_peak_idx)
        p_peaks = np.array(p_peaks)
        self.p_peaks = p_peaks
        return p_peaks

    def detect_t_peak(self, r_peaks=None, s_valleys=None):
        """
        Detect T peaks (local maxima) between the S valley and the midpoint to the next R peak.

        Parameters
        ----------
        r_peaks : list of int
            Indices of detected R peaks in the ECG waveform.
        s_valleys : list of int
            Indices of detected S valleys in the ECG waveform.

        Returns
        -------
        t_peaks : list of int
            Indices of the T peaks for each S valley.
        """
        if self.signal_type != "ECG":
            raise ValueError("T peaks can only be detected for ECG signals.")
        if r_peaks is None:
            r_peaks = self.r_peaks
        if s_valleys is None:
            s_valleys = self.detect_s_valley(r_peaks=r_peaks)

        t_peaks = []

        for i in range(len(s_valleys)):
            # Define S valley as the start of the search range
            s_valley = s_valleys[i]

            # Determine the end of the search range
            # For the last S valley, restrict the end point within signal bounds
            if i < len(r_peaks) - 1:
                midpoint = (r_peaks[i] + r_peaks[i + 1]) // 2
            else:
                # For the last R peak, limit the midpoint to signal length
                midpoint = len(self.waveform) - 1

            # Check if search range is valid
            if s_valley < midpoint:
                search_start = s_valley
                search_end = midpoint

                # Extract the signal segment within the search range
                signal_segment = self.waveform[search_start:search_end]

                # Detect the T peak as the maximum point in the segment
                t_peak_idx = np.argmax(signal_segment) + search_start
                t_peaks.append(t_peak_idx)
        t_peaks = np.array(t_peaks)
        self.t_peaks = t_peaks  # Store the detected T peaks for future use
        return t_peaks

    def detect_q_session(self, p_peaks=None, q_valleys=None, r_peaks=None):
        """
        Detects the Q sessions (start and end) in the ECG waveform based on R peaks.

        Parameters
        ----------
        r_peaks : np.array
            Indices of detected R peaks in the ECG waveform.

        Returns
        -------
        q_sessions : list of tuples
            Each tuple contains the start and end index of a Q session.

        Example
        -------
        >>> waveform = np.sin(np.linspace(0, 2*np.pi, 100))  # Simulated ECG signal
        >>> wm = WaveformMorphology(waveform, signal_type="ECG")
        >>> r_peaks = PeakDetection(waveform).detect_peaks()
        >>> q_sessions = wm.detect_q_session(r_peaks)
        >>> print(f"Q Sessions: {q_sessions}")
        """
        if self.signal_type != "ECG":
            raise ValueError("Q sessions can only be detected for ECG signals.")
        if r_peaks is None:
            r_peaks = self.r_peaks
        if q_valleys is None:
            q_valleys = self.detect_q_valley(r_peaks=r_peaks)
        if p_peaks is None:
            p_peaks = self.detect_p_peak(r_peaks=r_peaks, q_valleys=q_valleys)
        q_sessions = []

        # Trick to ensure that P peak comes before Q valley at the start
        if q_valleys[0] < p_peaks[0]:
            q_valleys = q_valleys[
                1:
            ]  # Skip the first Q valley if it comes before the first P peak
            r_peaks = r_peaks[1:]  # Also skip the corresponding first R peak

        # Iterate over detected peaks and valleys
        for i in range(len(q_valleys)):
            # Ensure indices are within valid bounds
            if i < len(p_peaks) and i < len(r_peaks):
                # Define start of Q session as the midpoint between P peak and Q valley
                start = (p_peaks[i] + q_valleys[i]) // 2

                # Set search range from Q valley to R peak
                search_start = q_valleys[i]
                search_end = r_peaks[i]

                # Check for valid search range
                if search_end > search_start:
                    # Find the end of Q session: closest point to start signal value within search range
                    start_value = self.waveform[start]
                    end = search_start + np.argmin(
                        np.abs(self.waveform[search_start:search_end] - start_value)
                    )

                    # Append the session to the q_sessions list
                    q_sessions.append((start, end))
        q_sessions = np.array(q_sessions)
        self.q_sessions = q_sessions  # Store the detected Q sessions for future use
        return q_sessions

    def detect_s_session(self, t_peaks=None, s_valleys=None, r_peaks=None):
        """
        Detects the S sessions (start and end) in the ECG waveform based on R peaks.

        Parameters
        ----------
        r_peaks : np.array
            Indices of detected R peaks in the ECG waveform.

        Returns
        -------
        s_sessions : list of tuples
            Each tuple contains the start and end index of an S session.

        Example
        -------
        >>> waveform = np.sin(np.linspace(0, 2*np.pi, 100))  # Simulated ECG signal
        >>> wm = WaveformMorphology(waveform, signal_type="ECG")
        >>> r_peaks = PeakDetection(waveform).detect_peaks()
        >>> s_sessions = wm.detect_s_session(r_peaks)
        >>> print(f"S Sessions: {s_sessions}")
        """
        if self.signal_type != "ECG":
            raise ValueError("S sessions can only be detected for ECG signals.")
        if r_peaks is None:
            r_peaks = self.r_peaks
        if s_valleys is None:
            s_valleys = self.detect_s_valley(r_peaks=r_peaks)
        if t_peaks is None:
            t_peaks = self.detect_t_peak(r_peaks=r_peaks, s_valleys=s_valleys)

        s_sessions = []
        # Ensure all input arrays are aligned in size
        for i in range(1, min(len(r_peaks), len(s_valleys), len(t_peaks))):
            r_peak = r_peaks[i]
            s_valley = s_valleys[i]
            t_peak = t_peaks[i]

            # Calculate the midpoint between S valley and T peak for the end point
            s_end = s_valley + (t_peak - s_valley) // 2

            # Determine the start point within the range from R peak to S valley,
            # choosing the point closest in value to s_end
            # Ensure the search range is valid and not empty
            if r_peak < s_valley:
                search_range = self.waveform[r_peak:s_valley]
                if len(search_range) > 0:
                    # Find the start point within the search range closest to s_end's value
                    start_index_within_range = np.argmin(
                        np.abs(search_range - self.waveform[s_end])
                    )
                    s_start = r_peak + start_index_within_range

                    # Only add the session if s_start is before s_end
                    if s_start < s_end:
                        s_sessions.append((s_start, s_end))
        s_sessions = np.array(s_sessions)
        self.s_sessions = s_sessions  # Store the detected S sessions for future use
        return s_sessions

    def detect_r_session(self, rpeaks=None, q_sessions=None, s_sessions=None):
        """
        Detects the R session (start and end) in the ECG waveform based on Q and S sessions.

        Parameters
        ----------
        q_session : tuple
            The start and end index of the Q session.
        s_session : tuple
            The start and end index of the S session.

        Returns
        -------
        r_sessions : list of tuples
            Each tuple contains the start and end index of an R session.
        """
        if rpeaks is None:
            rpeaks = self.r_peaks

        # If q_sessions or s_sessions are explicitly provided as empty arrays, respect that
        if (q_sessions is not None and len(q_sessions) == 0) or (s_sessions is not None and len(s_sessions) == 0):
            return np.array([])

        # Only try to get sessions from instance if they weren't provided
        if q_sessions is None:
            q_sessions = self.q_sessions
        if s_sessions is None:
            s_sessions = self.s_sessions

        # If sessions are None or empty, try to detect them
        if q_sessions is None or len(q_sessions) == 0:
            q_sessions = self.detect_q_session(r_peaks=rpeaks)
        if s_sessions is None or len(s_sessions) == 0:
            s_sessions = self.detect_s_session(r_peaks=rpeaks)

        # If still empty after detection, return empty array
        if len(q_sessions) == 0 or len(s_sessions) == 0:
            return np.array([])

        r_sessions = []

        # Ensure all input arrays are aligned in size and skip the first item to avoid boundary issues
        for i in range(min(len(q_sessions), len(s_sessions))):
            q_end = q_sessions[i][1]
            s_start = s_sessions[i][0]

            # Ensure there is a valid interval for the R session
            if q_end < s_start:
                r_sessions.append((q_end, s_start))
        r_sessions = np.array(r_sessions)
        self.r_sessions = r_sessions  # Store the detected R sessions for future use
        return r_sessions

    def detect_qrs_session(self, rpeaks=None, q_session=None, s_session=None):
        """
        Detects the QRS complex sessions (start and end) in the ECG waveform.

        Parameters
        ----------
        r_peaks : np.array
            Indices of detected R peaks in the ECG waveform.

        Returns
        -------
        qrs_sessions : np.ndarray
            Each tuple contains the start and end index of a QRS session.

        Example
        -------
        >>> waveform = np.sin(np.linspace(0, 2*np.pi, 100))  # Simulated ECG signal
        >>> wm = WaveformMorphology(waveform, signal_type="ECG")
        >>> r_peaks = PeakDetection(waveform).detect_peaks()
        >>> qrs_sessions = wm.detect_qrs_session(r_peaks)
        >>> print(f"QRS Sessions: {qrs_sessions}")
        """
        if rpeaks is None:
            rpeaks = self.r_peaks
        if q_session is None:
            q_session = self.detect_q_session(r_peaks=rpeaks)
        if s_session is None:
            s_session = self.detect_s_session(r_peaks=rpeaks)
        # Ensure all input arrays are aligned in size
        if len(q_session) == 0 or len(s_session) == 0:
            return np.array([])

        qrs_sessions = [(q[0], s[1]) for q, s in zip(q_session, s_session)]
        return np.array(qrs_sessions)

    def detect_ppg_session(self, troughs=None):
        """
        Detects PPG sessions between consecutive troughs.

        Parameters
        ----------
        troughs : np.array, optional
            Indices of detected troughs in the PPG waveform.

        Returns
        -------
        np.ndarray
            Array of tuples containing start and end indices of PPG sessions.
        """
        if troughs is None:
            troughs = self.detect_troughs(systolic_peaks=self.systolic_peaks)
        ppg_sessions = [
            (trough_start, trough_end)
            for trough_start, trough_end in zip(troughs[:-1], troughs[1:])
        ]
        return np.array(ppg_sessions)

    def detect_ecg_session(self, p_peaks=None, t_peaks=None):
        """
        Detects the ECG session (start and end) based on flat lines before the P peak and after the T peak.

        Parameters
        ----------
        p_peaks : np.array, optional
            Indices of detected P peaks in the ECG waveform.
        t_peaks : np.array, optional
            Indices of detected T peaks in the ECG waveform.

        Returns
        -------
        ecg_sessions : list of tuples
            Each tuple contains the start and end index of an ECG session.
        """
        if self.signal_type != "ECG":
            raise ValueError("ECG sessions can only be detected for ECG signals.")

        if p_peaks is None:
            p_peaks = self.detect_p_peak()
        if t_peaks is None:
            t_peaks = self.detect_t_peak()

        ecg_sessions = []

        # Calculate the derivative of the signal to identify flat regions
        signal_derivative = np.diff(self.waveform)
        threshold = 0.02 * np.std(signal_derivative)  # Adaptive threshold for flatness

        for p, t in zip(p_peaks, t_peaks):
            # Detect flat line before the P peak
            start = p
            for i in range(p, 0, -1):
                if abs(signal_derivative[i - 1]) > threshold:
                    start = i
                    break

            # Detect flat line after the T peak
            end = t
            for i in range(t, len(signal_derivative) - 1):
                if abs(signal_derivative[i]) > threshold:
                    end = i
                    break

            ecg_sessions.append((start, end))

        ecg_sessions = np.array(ecg_sessions)
        self.ecg_sessions = ecg_sessions
        return ecg_sessions

    def compute_amplitude(
        self,
        interval_type="R-to-S",
        baseline_method="moving_average",
        compare_to_baseline=False,
        signal_type="ECG",
    ):
        """
        Computes the amplitude (max-min) of the waveform for specified intervals.

        Parameters
        ----------
        interval_type : str, optional
            The interval type to calculate the amplitude for:
            - For ECG:
            - "R-to-S": Between R peak and S valley.
            - "R-to-Q": Between Q valley and R peak.
            - "P-to-Q": Between P peak and Q valley.
            - "T-to-S": Between S valley and T peak.
            - "T-to-Baseline": Between T peak and baseline.
            - "R-to-Baseline": Between R peak and baseline.
            - "S-to-Baseline": Between S valley and baseline.
            - For PPG:
            - "Sys-to-Notch": Between systolic peak and dicrotic notch.
            - "Notch-to-Dia": Between dicrotic notch and diastolic peak.
            - "Sys-to-Dia": Between systolic and diastolic peaks.
            - "Sys-to-Baseline": Between systolic peak and baseline.
            - "Notch-to-Baseline": Between dicrotic notch and baseline.
            - "Dia-to-Baseline": Between diastolic peak and baseline.

        signal_type : str, optional
            The type of signal: "ECG" or "PPG". Default is "ECG".

        compare_to_baseline : bool, optional
            If True, compute amplitudes relative to a baseline.

        Returns
        -------
        List[float]: A list of amplitude values for each interval within a single complex.

        Examples
        --------
        >>> amplitude_values = wm.compute_amplitude(interval_type="R-to-S", signal_type="ECG")
        >>> print(f"Amplitude values for each complex in R-to-S interval: {amplitude_values}")
        """
        # Baseline calculation
        baseline = estimate_baseline(self.waveform, self.fs, method=baseline_method)

        # Automatically compute peaks and valleys if they are not already computed
        if signal_type == "ECG":
            if interval_type == "R-to-S":
                if self.s_valleys is None:
                    self.detect_s_valley()
                peaks, valleys, require_peak_first = self.r_peaks, self.s_valleys, True
            elif interval_type == "R-to-Q":
                if self.q_valleys is None:
                    self.detect_q_valley()
                peaks, valleys, require_peak_first = self.r_peaks, self.q_valleys, False
            elif interval_type == "P-to-Q":
                if self.q_valleys is None:
                    self.detect_q_valley()
                if self.p_peaks is None:
                    self.detect_p_peak()
                peaks, valleys, require_peak_first = self.p_peaks, self.q_valleys, True
            elif interval_type == "T-to-S":
                if self.s_valleys is None:
                    self.detect_s_valley()
                if self.t_peaks is None:
                    self.detect_t_peak()
                peaks, valleys, require_peak_first = self.t_peaks, self.s_valleys, False
            elif interval_type == "T-to-Baseline":
                if self.t_peaks is None:
                    self.detect_t_peak()
                peaks, valleys, require_peak_first = self.t_peaks, None, False
            elif interval_type == "R-to-Baseline":
                peaks, valleys, require_peak_first = self.r_peaks, None, False
            elif interval_type == "S-to-Baseline":
                if self.s_valleys is None:
                    self.detect_s_valley()
                peaks, valleys, require_peak_first = self.s_valleys, None, False
            else:
                raise ValueError("Invalid interval_type for ECG.")
        elif signal_type == "PPG":
            if interval_type == "Sys-to-Notch":
                if self.dicrotic_notches is None:
                    self.detect_dicrotic_notches()
                peaks, valleys, require_peak_first = (
                    self.systolic_peaks,
                    self.dicrotic_notches,
                    True,
                )
            elif interval_type == "Notch-to-Dia":
                if self.dicrotic_notches is None:
                    self.detect_dicrotic_notches()
                if self.diastolic_peaks is None:
                    self.detect_diastolic_peak()
                peaks, valleys, require_peak_first = (
                    self.dicrotic_notches,
                    self.diastolic_peaks,
                    False,  # Keep as False since we want volume when notch comes before diastolic peak
                )
            elif interval_type == "Sys-to-Dia":
                if self.diastolic_peaks is None:
                    self.detect_diastolic_peak()
                peaks, valleys, require_peak_first = (
                    self.systolic_peaks,
                    self.diastolic_peaks,
                    True,
                )
            elif interval_type == "Sys-to-Baseline":
                peaks, valleys, require_peak_first = self.systolic_peaks, None, False
            elif interval_type == "Notch-to-Baseline":
                if self.dicrotic_notches is None:
                    self.detect_dicrotic_notches()
                peaks, valleys, require_peak_first = self.dicrotic_notches, None, False
            elif interval_type == "Dia-to-Baseline":
                if self.diastolic_peaks is None:
                    self.detect_diastolic_peak()
                peaks, valleys, require_peak_first = self.diastolic_peaks, None, False
            else:
                raise ValueError("Invalid interval_type for PPG.")
        else:
            raise ValueError("Invalid signal type. Supported types are ECG, PPG, EEG.")

        # Compute amplitude for each interval or baseline comparison
        amplitudes = []
        for i, peak in enumerate(peaks):
            if valleys is None:
                # Baseline comparison only
                amplitude = abs(self.waveform[peak] - baseline)
                amplitudes.append(amplitude)
            elif i < len(valleys):
                valley = valleys[i]
                if (require_peak_first and peak < valley) or (
                    not require_peak_first and valley < peak
                ):
                    amplitude = abs(self.waveform[peak] - self.waveform[valley])
                    amplitudes.append(amplitude)

        return np.array(amplitudes)

    def compute_volume(self, interval_type="P-to-T", signal_type="ECG", mode="peak"):
        """
        Compute the area under the curve between two sets of peaks and valleys for specified intervals.

        Parameters
        ----------
        interval_type : str, optional
            The interval type to calculate the volume for:
            - For ECG:
            - "P-to-T": Entire complex from P peak to T peak.
            - "R-to-S": Between R peak and S valley.
            - "R-to-Q": Between Q valley and R peak.
            - "P-to-Q": Between P peak and Q valley.
            - "T-to-S": Between S valley and T peak.
            - For PPG:
            - "Sys-to-Notch": Between systolic peak and dicrotic notch.
            - "Notch-to-Dia": Between dicrotic notch and diastolic peak.
            - "Sys-to-Dia": Between systolic and diastolic peaks.
            - "Sys-to-Sys": Between consecutive systolic peaks (full PPG complex).

            Default is "P-to-T" for ECG and "Sys-to-Notch" for PPG.

        signal_type : str, optional
            The type of signal: "ECG" or "PPG". Default is "ECG".

        mode : str, optional
            The area computation method ("peak" or "trough"). Default is "peak".
            - "peak": Computes the area under the curve.
            - "trough": Computes the area bounded by troughs.

        Returns
        -------
        List[float]: A list of volume values, each representing the area for the specified interval
                    within a single complex.

        Examples
        --------
        >>> volume_values = wm.compute_volume(interval_type="R-to-S", signal_type="ECG")
        >>> print(f"Volume values for each complex in R-to-S interval: {volume_values}")
        """

        # Automatically compute peaks and valleys if they are not already computed
        if signal_type == "ECG":
            if interval_type == "R-to-S":
                if self.s_valleys is None:
                    self.detect_s_valley()
                peaks, valleys, require_peak_first = self.r_peaks, self.s_valleys, True
            elif interval_type == "R-to-Q":
                if self.q_valleys is None:
                    self.detect_q_valley()
                peaks, valleys, require_peak_first = self.r_peaks, self.q_valleys, False
            elif interval_type == "P-to-Q":
                if self.q_valleys is None:
                    self.detect_q_valley()
                if self.p_peaks is None:
                    self.detect_p_peak()
                peaks, valleys, require_peak_first = self.p_peaks, self.q_valleys, True
            elif interval_type == "T-to-S":
                if self.s_valleys is None:
                    self.detect_s_valley()
                if self.t_peaks is None:
                    self.detect_t_peak()
                peaks, valleys, require_peak_first = self.t_peaks, self.s_valleys, False
            else:
                raise ValueError("Invalid interval_type for ECG.")
        elif signal_type == "PPG":
            if interval_type == "Sys-to-Notch":
                if self.dicrotic_notches is None:
                    self.detect_dicrotic_notches()
                peaks, valleys, require_peak_first = (
                    self.systolic_peaks,
                    self.dicrotic_notches,
                    True,
                )
            elif interval_type == "Notch-to-Dia":
                if self.dicrotic_notches is None:
                    self.detect_dicrotic_notches()
                if self.diastolic_peaks is None:
                    self.detect_diastolic_peak()
                peaks, valleys, require_peak_first = (
                    self.dicrotic_notches,
                    self.diastolic_peaks,
                    False,
                )
            elif interval_type == "Sys-to-Dia":
                if self.diastolic_peaks is None:
                    self.detect_diastolic_peak()
                peaks, valleys, require_peak_first = (
                    self.systolic_peaks,
                    self.diastolic_peaks,
                    True,
                )
            elif interval_type == "Sys-to-Sys":
                # Full PPG complex: consecutive systolic peaks
                if self.systolic_peaks is None or len(self.systolic_peaks) < 2:
                    raise ValueError(
                        "At least two systolic peaks required for Sys-to-Sys interval."
                    )
                peaks = self.systolic_peaks[:-1]  # Start points
                valleys = self.systolic_peaks[1:]  # End points
                require_peak_first = True
            else:
                raise ValueError("Invalid interval_type for PPG.")
        else:
            raise ValueError("signal_type must be 'ECG' or 'PPG'.")

        # Compute area for each interval
        volumes = []
        for start, end in zip(peaks, valleys):
            if (require_peak_first and start < end) or (
                not require_peak_first and start < end
            ):  # Changed condition to always check start < end
                if mode == "peak":
                    volume = np.trapz(
                        self.waveform[start : end + 1]
                    )  # Integrate over interval
                elif mode == "trough":
                    volume = np.trapz(
                        self.waveform[min(start, end) : max(start, end) + 1]
                    )
                else:
                    raise ValueError("Volume mode must be 'peak' or 'trough'.")
                volumes.append(volume)

        return np.array(volumes)

    def compute_skewness(self, signal_type="ECG"):
        """
        Compute the skewness of each complex in the signal.

        Parameters
        ----------
        signal_type : str, optional
            The type of signal, either "ECG" or "PPG". Default is "ECG".

        Returns
        -------
        List[float]: A list of skewness values, one for each complex.

        Examples
        --------
        >>> signal = np.random.randn(1000)
        >>> extractor = PhysiologicalFeatureExtractor(signal)
        >>> skewness_values = extractor.compute_skewness()
        >>> print(f"Skewness values for each complex: {skewness_values}")
        """

        # Define complex points based on signal type
        if signal_type == "ECG":
            sessions = self.detect_ecg_session()
        elif signal_type == "PPG":
            sessions = self.detect_ppg_session()
        else:
            raise ValueError("signal_type must be 'ECG' or 'PPG'.")

        # Compute skewness for each complex
        skewness_values = []
        for session in sessions:
            # Ensure valid intervals
            start = session[0]
            end = session[1]
            # Compute skewness for the complex segment if valid intervals are found
            if end > start:
                complex_segment = self.waveform[start : end + 1]
                skewness_values.append(skew(complex_segment))

        return np.array(skewness_values)

    def compute_duration(self, sessions=None, mode="Custom"):
        """
        Computes the duration of the QRS complex in an ECG waveform.

        Returns:
            float: The QRS duration in milliseconds.

        Example:
            >>> ecg_signal = [...]  # Sample ECG signal
            >>> wm = WaveformMorphology(ecg_signal, signal_type="ECG")
            >>> qrs_duration = wm.compute_qrs_duration()
            >>> print(f"QRS Duration: {qrs_duration} ms")
        """
        if mode not in ["ECG", "PPG", "QRS", "Custom"]:
            raise ValueError(
                "Duration can only be computed for ECG signals, PPG signals or QRS complexes.\
                            Please use Custom or Custom"
            )
        if sessions is None:
            if mode == "ECG":
                sessions = self.detect_ecg_session()
            elif mode == "PPG":
                sessions = self.detect_ppg_session()
            elif mode == "QRS":
                sessions = self.detect_qrs_session()
        session_durations = [
            (session[1] - session[0]) / self.fs for session in sessions
        ]
        return np.array(session_durations)

    def compute_ppg_dicrotic_notch(self):
        """
        Detects the dicrotic notch in a PPG waveform and computes its timing.

        Returns:
            float: The timing of the dicrotic notch in milliseconds.

        Example:
            >>> ppg_signal = [...]  # Sample PPG signal
            >>> wm = WaveformMorphology(ppg_signal, signal_type="PPG")
            >>> notch_timing = wm.compute_ppg_dicrotic_notch()
            >>> print(f"Dicrotic Notch Timing: {notch_timing} ms")
        """
        if self.signal_type != "PPG":
            raise ValueError("Dicrotic notch can only be computed for PPG signals.")

        # Detect peaks and dicrotic notches in the PPG signal
        peak_detector = PeakDetection(self.waveform, method="ppg_first_derivative")
        systolic_peaks = peak_detector.detect_peaks()

        # dicrotic_notch_detector = PeakDetection(
        #     self.waveform, method="ppg_second_derivative"
        # )
        # dicrotic_notches = dicrotic_notch_detector.detect_peaks()
        dicrotic_notches = self.detect_dicrotic_notches(systolic_peaks=systolic_peaks)

        # Compute the time difference between systolic peak and dicrotic notch
        notch_timing = 0.0
        # for peak, notch in zip(systolic_peaks, dicrotic_notches):
        #     if notch > peak:
        #         notch_timing += (
        #             (notch - peak) * 1000 / self.fs
        #         )  # Convert to milliseconds

        # return notch_timing / len(systolic_peaks) if len(systolic_peaks) > 0 else 0.0

        valid_pairs = 0

        for peak, notch in zip(systolic_peaks, dicrotic_notches):
            if notch > peak:
                notch_timing += (
                    (notch - peak) * 1000 / self.fs
                )  # Convert to milliseconds
                valid_pairs += 1

        return notch_timing / valid_pairs if valid_pairs > 0 else 0.0

    def compute_eeg_wavelet_features(self):
        """
        Computes EEG wavelet features by applying a wavelet transform and extracting relevant
        frequency bands.

        Returns:
            dict: A dictionary of extracted EEG wavelet features (e.g., delta, theta, alpha).

        Example:
            >>> eeg_signal = [...]  # Sample EEG signal
            >>> wm = WaveformMorphology(eeg_signal, signal_type="EEG")
            >>> wavelet_features = wm.compute_eeg_wavelet_features()
            >>> print(wavelet_features)
        """
        if self.signal_type != "EEG":
            raise ValueError("Wavelet features can only be computed for EEG signals.")

        wavelet_coeffs = np.abs(np.convolve(self.waveform, np.ones(10), "same"))

        # Extract frequency bands based on wavelet decomposition
        delta = wavelet_coeffs[: len(wavelet_coeffs) // 5]
        theta = wavelet_coeffs[len(wavelet_coeffs) // 5 : len(wavelet_coeffs) // 4]
        alpha = wavelet_coeffs[len(wavelet_coeffs) // 4 : len(wavelet_coeffs) // 3]

        return {
            "delta_power": np.sum(delta),
            "theta_power": np.sum(theta),
            "alpha_power": np.sum(alpha),
        }

    def compute_slope(self, points=None, option=None, window=3, slope_unit="radians"):
        """
        Compute the slope of the waveform at specified points or critical points.

        Parameters
        ----------
        points : list or np.array, optional
            Indices of the points where the slope is to be calculated. Ignored if `option` is specified.
        option : str, optional
            Specifies the critical points to use for slope calculation. Options for ECG are "p_peaks", "q_valleys",
            "r_peaks", "s_valleys", "t_peaks". Options for PPG are "troughs", "systolic_peaks", "diastolic_peaks".
        window : int, optional
            The number of points before and after each point to consider for slope calculation.

        Returns
        -------
        slopes : np.array
            Slope values at the specified points or critical points.

        Example
        -------
        >>> waveform = np.sin(np.linspace(0, 2 * np.pi, 100))
        >>> wm = WaveformMorphology(waveform, signal_type="ECG")
        >>> slopes = wm.compute_slope(option="r_peaks")
        >>> print(f"Slopes at R peaks: {slopes}")
        """
        if option:
            if option == "p_peaks":
                points = (
                    self.p_peaks if self.p_peaks is not None else self.detect_p_peak()
                )
            elif option == "q_valleys":
                points = (
                    self.q_valleys
                    if self.q_valleys is not None
                    else self.detect_q_valley()
                )
            elif option == "r_peaks":
                points = self.r_peaks
            elif option == "s_valleys":
                points = (
                    self.s_valleys
                    if self.s_valleys is not None
                    else self.detect_s_valley()
                )
            elif option == "t_peaks":
                points = (
                    self.t_peaks if self.t_peaks is not None else self.detect_t_peak()
                )
            elif option == "troughs":
                points = (
                    self.diastolic_troughs
                    if self.diastolic_troughs is not None
                    else self.detect_troughs()
                )
            elif option == "systolic_peaks":
                points = self.systolic_peaks
            elif option == "diastolic_peaks":
                points = (
                    self.diastolic_peaks
                    if self.diastolic_peaks is not None
                    else self.detect_diastolic_peak()
                )
                # from plotly import graph_objects as go
                # fig = go.Figure()
                # fig.add_trace(go.Scatter(x=np.arange(len(self.waveform)), y=self.waveform, mode='lines'))
                # fig.add_trace(go.Scatter(x=self.dicrotic_notches, y=self.waveform[self.dicrotic_notches], mode='markers', name='Dicrotic Notches'))
                # fig.add_trace(go.Scatter(x=self.diastolic_peaks, y=self.waveform[self.diastolic_peaks], mode='markers', name='Diastolic Peaks'))
                # fig.add_trace(go.Scatter(x=self.systolic_peaks, y=self.waveform[self.systolic_peaks], mode='markers', name='Systolic Peaks'))
                # fig.add_trace(go.Scatter(x=self.diastolic_troughs, y=self.waveform[self.diastolic_troughs], mode='markers', name='Diastolic Troughs'))
                # fig.show()
            else:
                raise ValueError(f"Invalid option '{option}' for critical points.")

        if points is None:
            raise ValueError("No points specified for slope calculation.")

        slopes = []
        for point in points:
            # Define the window around the point
            start = max(0, point - window)
            end = min(len(self.waveform) - 1, point + window)
            if end <= start + 1:  # Need at least 2 points for a slope
                slopes.append(0.0)  # Flat slope if window is too small
                continue

            # Extract segment and corresponding x-axis (in samples)
            x = np.arange(start, end + 1)
            y = self.waveform[start : end + 1]

            # Fit a line using least-squares (polyfit degree 1)
            coeffs = np.polyfit(x, y, 1)  # coeffs[0] is slope, coeffs[1] is intercept
            raw_slope = coeffs[0]  # Slope in signal units per sample

            # Convert to angle based on sampling frequency-adjusted x-axis
            # dx = (end - start) / self.fs  # Time span in seconds
            # dy = self.waveform[end] - self.waveform[start]  # Amplitude change
            # raw_slope = dy / dx  # Slope in signal units per second

            if slope_unit == "degrees":
                slope = np.degrees(np.arctan(raw_slope))  # Convert to degrees
            elif slope_unit == "radians":
                slope = np.arctan(raw_slope)  # Convert to radians
            elif slope_unit == "raw":
                slope = raw_slope  # Keep as signal units per sample
            else:
                raise ValueError("slope_unit must be 'degrees', 'radians', or 'raw'.")

            slopes.append(slope)

        return np.array(slopes)

    def compute_curvature(self, points=None, option=None, window=3):
        """
        Compute the curvature of the waveform at specified points or critical points.

        Parameters
        ----------
        points : list or np.array, optional
            Indices of the points where the curvature is to be calculated. Ignored if `option` is specified.
        option : str, optional
            Specifies the critical points to use for curvature calculation. Options for ECG are "p_peaks", "q_valleys",
            "r_peaks", "s_valleys", "t_peaks". Options for PPG are "troughs", "systolic_peaks", "diastolic_peaks".
        window : int, optional
            The number of points before and after each point to consider for curvature calculation.

        Returns
        -------
        curvatures : np.array
            Curvature values at the specified points or critical points.

        Example
        -------
        >>> waveform = np.sin(np.linspace(0, 2 * np.pi, 100))
        >>> wm = WaveformMorphology(waveform, signal_type="PPG")
        >>> curvatures = wm.compute_curvature(option="systolic_peaks")
        >>> print(f"Curvatures at systolic peaks: {curvatures}")
        """
        if option:
            if option == "p_peaks":
                points = (
                    self.p_peaks if self.p_peaks is not None else self.detect_p_peak()
                )
            elif option == "q_valleys":
                points = (
                    self.q_valleys
                    if self.q_valleys is not None
                    else self.detect_q_valley()
                )
            elif option == "r_peaks":
                points = (
                    self.r_peaks if self.r_peaks is not None else self.detect_r_peaks()
                )
            elif option == "s_valleys":
                points = (
                    self.s_valleys
                    if self.s_valleys is not None
                    else self.detect_s_valley()
                )
            elif option == "t_peaks":
                points = (
                    self.t_peaks if self.t_peaks is not None else self.detect_t_peak()
                )
            elif option == "troughs":
                points = (
                    self.diastolic_troughs
                    if self.diastolic_troughs is not None
                    else self.detect_troughs()
                )
            elif option == "systolic_peaks":
                points = (
                    self.systolic_peaks
                    if self.systolic_peaks is not None
                    else self.detect_systolic_peaks()
                )
            elif option == "diastolic_peaks":
                points = (
                    self.diastolic_peaks
                    if self.diastolic_peaks is not None
                    else self.detect_diastolic_peak()
                )
            else:
                raise ValueError(f"Invalid option '{option}' for critical points.")

        if points is None:
            raise ValueError("No points specified for curvature calculation.")

        curvatures = []
        for point in points:
            start = max(0, point - window)
            end = min(len(self.waveform) - 1, point + window)
            dy_dx = np.gradient(self.waveform[start : end + 1])
            d2y_dx2 = np.gradient(dy_dx)

            if len(d2y_dx2) > 1:
                curvature = d2y_dx2[len(d2y_dx2) // 2]
            else:
                curvature = 0
            curvatures.append(curvature)

        return np.array(curvatures)

    def get_duration(
        self,
        start_points=None,
        end_points=None,
        session_type="systolic",
        summary_type="mean",
    ):
        """
        Computes the duration based on detected start and end points for a specified session type.

        Parameters
        ----------
        start_points : list, optional
            List of detected start indices in the signal. If None, defaults based on `session_type`.
        end_points : list, optional
            List of detected end indices in the signal. If None, defaults based on `session_type`.
        session_type : str, optional
            Specifies the type of session: "systolic", "diastolic", "qrs", or "Custom".
            - For "systolic" and "diastolic", appropriate start and end points will be detected if not provided.
            - For "qrs", QRS sessions are automatically detected and do not require start or end points.
            - For "Custom", start and end points must be provided as arguments.
        summary_type : str, optional
            Type of summary to apply to the computed durations. Options are 'mean', 'median',
            '2nd_quartile', '3rd_quartile', or 'full' (returns all durations).

        Returns
        -------
        float or list
            The summarized duration based on `summary_type`, or a list of durations if `summary_type` is 'full'.

        Notes
        -----
        Logs an error and returns NaN if no valid sessions are detected or if an invalid summary type is provided.
        """
        valid_summary_types = ["mean", "median", "sum"]
        if summary_type not in valid_summary_types:
            raise ValueError(
                "Invalid summary_type. Supported types are 'mean', 'median', and 'sum'."
            )
        try:
            # Handle each session type to determine start and end points
            if session_type == "systolic":
                start_points = (
                    start_points if start_points is not None else self.detect_troughs()
                )
                end_points = (
                    end_points
                    if end_points is not None
                    else self.detect_dicrotic_notches(diastolic_troughs=start_points)
                )
            elif session_type == "diastolic":
                start_points = (
                    start_points
                    if start_points is not None
                    else self.detect_dicrotic_notches()
                )
                end_points = (
                    end_points if end_points is not None else self.detect_troughs()
                )
            elif session_type == "qrs":
                sessions = self.detect_qrs_session()
                if not sessions:
                    raise ValueError("No valid QRS sessions for duration calculation.")
                return self._summarize_list(
                    self.compute_duration(sessions), summary_type
                )
            elif session_type == "Custom":
                if start_points is None or end_points is None:
                    raise ValueError(
                        "For 'Custom' session type, start_points and end_points must be provided."
                    )
            else:
                raise ValueError(f"Unsupported session type: {session_type}")

            # Validate start and end points
            if (
                start_points is None
                or end_points is None
                or len(start_points) == 0
                or len(end_points) == 0
            ):
                raise ValueError(
                    "Start or end points detection returned an empty list."
                )

            # Adjust lists for alignment (e.g., if the first end point is before the first start point)
            if session_type in ["systolic", "diastolic", "Custom"]:
                if end_points[0] < start_points[0]:
                    end_points = end_points[1:]
                min_length = min(len(start_points), len(end_points))

                if min_length == 0:
                    raise ValueError(
                        f"No valid pairs of start and end points for {session_type} duration calculation."
                    )

                # Compute durations based on aligned start and end points
                sessions = [
                    (start, end)
                    for start, end in zip(
                        start_points[:min_length], end_points[:min_length]
                    )
                    if end > start
                ]
                if not sessions:
                    raise ValueError(
                        f"No valid durations computed for {session_type} sessions."
                    )

                durations = self.compute_duration(sessions)
                return self._summarize_list(durations, summary_type)

        except ValueError as e:
            logger.error(f"Value error in get_duration for {session_type} session: {e}")
            return np.nan
        except Exception as e:
            logger.error(
                f"Unexpected error in get_duration for {session_type} session: {e}"
            )
            return np.nan

    def _summarize_list(self, values, summary_type):
        """
        Helper function to apply summary type to a list of values.

        Parameters
        ----------
        values : list
            List of values to summarize.
        summary_type : str
            Summary type: 'mean', 'median', '2nd_quartile', '3rd_quartile', or 'full'.

        Returns
        -------
        float or list
            The summarized value or the original list if `summary_type` is 'full'.
        """
        if summary_type == "mean":
            return np.mean(values)
        elif summary_type == "median":
            return np.median(values)
        elif summary_type == "2nd_quartile":
            return np.percentile(values, 25)
        elif summary_type == "3rd_quartile":
            return np.percentile(values, 75)
        elif summary_type == "full":
            return values
        else:
            raise ValueError(f"Unsupported summary type: {summary_type}")

    def get_area(self, interval_type, signal_type="PPG", summary_type="mean"):
        """
        Computes the area of the specified interval in the signal with the chosen summary type.

        Parameters
        ----------
        interval_type : str
            Type of interval to compute the area for. Options include:
            - For PPG: "Sys-to-Notch", "Notch-to-Dia", "Sys-to-Dia"
            - For ECG: "R-to-Q", "R-to-S", "QRS", "T-to-S"
        signal_type : str, optional
            The type of signal: "PPG" or "ECG". Default is "PPG".
        summary_type : str, optional
            Specifies the summary statistic to return:
            - 'mean': Mean of all areas
            - 'median': Median of all areas
            - '2nd_quartile': 25th percentile (1st quartile)
            - '3rd_quartile': 75th percentile (3rd quartile)
            - 'full': Returns all areas as a list
            Default is 'mean'.

        Returns
        -------
        float or list
            The summary statistic for the area, or the list of all areas if `summary_type` is 'full'.
            Returns NaN if an error occurs.

        Notes
        -----
        Logs an error and returns NaN if the area computation fails or if an invalid summary type is provided.
        """
        valid_summary_types = ["mean", "median", "sum"]
        if summary_type not in valid_summary_types:
            raise ValueError(
                "Invalid summary_type. Supported types are 'mean', 'median', and 'sum'."
            )
        try:
            # If interval type is "QRS", compute combined area in a single step
            if interval_type == "QRS" and signal_type == "ECG":
                areas_r_to_q = self.compute_volume(
                    interval_type="R-to-Q", signal_type=signal_type
                )
                areas_r_to_s = self.compute_volume(
                    interval_type="R-to-S", signal_type=signal_type
                )
                if areas_r_to_q is None or areas_r_to_s is None:
                    raise ValueError(
                        "QRS area computation failed due to invalid sub-interval areas."
                    )
                min_length = min(len(areas_r_to_q), len(areas_r_to_s))
                if len(areas_r_to_q) != len(areas_r_to_s):
                    areas_r_to_q = areas_r_to_q[:min_length]
                    areas_r_to_s = areas_r_to_s[:min_length]
                areas = np.array(areas_r_to_q) + np.array(areas_r_to_s)

            elif interval_type == "Notch-to-Dia" and signal_type == "PPG":
                areas_sys_to_sys = self.compute_volume(
                    interval_type="Sys-to-Sys", signal_type=signal_type
                )
                areas_sys_to_notch = self.compute_volume(
                    interval_type="Sys-to-Notch", signal_type=signal_type
                )
                if areas_sys_to_sys is None or areas_sys_to_notch is None:
                    raise ValueError(
                        "Notch-to-Dia computation failed due to invalid Sys-to-Sys or Sys-to-Notch areas."
                    )
                min_length = min(len(areas_sys_to_sys), len(areas_sys_to_notch))
                if len(areas_sys_to_sys) != len(areas_sys_to_notch):
                    areas_sys_to_sys = areas_sys_to_sys[:min_length]
                    areas_sys_to_notch = areas_sys_to_notch[:min_length]
                areas = np.array(areas_sys_to_sys) - np.array(areas_sys_to_notch)
            else:
                areas = self.compute_volume(
                    interval_type=interval_type, signal_type=signal_type
                )

            # Validate computed areas
            if (
                areas is None
                or not isinstance(areas, (list, np.ndarray))
                or len(areas) == 0
            ):
                raise ValueError(
                    f"Computed areas for {interval_type} are None or empty."
                )

            # Normalize by sampling frequency to get area in amplitude × seconds
            areas = areas / self.fs

            return self._summarize_list(areas, summary_type)

        except ValueError as e:
            logger.error(f"Value error in get_area for {interval_type} interval: {e}")
            return np.nan
        except Exception as e:
            logger.error(
                f"Unexpected error in get_area for {interval_type} interval: {e}"
            )
            return np.nan

    def get_slope(
        self, slope_type="systolic", window=5, summary_type="mean", slope_unit="radians"
    ):
        """
        Computes the slope of the specified type (systolic, diastolic, QRS) using the chosen summary type.

        Parameters
        ----------
        slope_type : str, optional
            The type of slope to calculate. Options are:
            - "systolic": Slope from systolic peaks.
            - "diastolic": Slope from diastolic peaks.
            - "qrs": Slope from QRS R peaks.
            Default is "systolic".
        window : int, optional
            The window size for slope calculation. Default is 5.
        summary_type : str, optional
            Specifies the summary statistic to return:
            - 'mean': Mean of all slopes
            - 'median': Median of all slopes
            - '2nd_quartile': 25th percentile (1st quartile)
            - '3rd_quartile': 75th percentile (3rd quartile)
            - 'full': Returns all slopes as a list
            Default is 'mean'.

        Returns
        -------
        float or list
            The summary statistic for the slopes, or the list of slopes if `summary_type` is 'full'.
            Returns NaN if an error occurs.
        """
        try:
            # Determine the option based on slope_type
            options_map = {
                "systolic": "systolic_peaks",
                "diastolic": "diastolic_peaks",
                "qrs": "r_peaks",
            }

            if slope_type not in options_map:
                raise ValueError(f"Unsupported slope type: {slope_type}")

            # Compute slopes based on specified option
            slopes = self.compute_slope(
                option=options_map[slope_type], window=window, slope_unit=slope_unit
            )

            # Validate slopes array
            if (
                slopes is None
                or not isinstance(slopes, (list, np.ndarray))
                or len(slopes) == 0
            ):
                raise ValueError(f"Computed slopes for {slope_type} are None or empty.")

            # Apply the specified summary type
            return self._summarize_list(slopes, summary_type)

        except ValueError as e:
            logger.error(f"Value error in get_slope for {slope_type}: {e}")
            return np.nan
        except Exception as e:
            logger.error(f"Unexpected error in get_slope for {slope_type}: {e}")
            return np.nan

    def get_signal_skewness(self, signal_type="PPG", summary_type="mean"):
        """
        Computes the skewness of the signal based on the specified signal type and summary type.

        Parameters
        ----------
        signal_type : str, optional
            Type of signal to compute skewness for (e.g., 'PPG' or 'ECG'). Default is 'PPG'.
        summary_type : str, optional
            Type of summary to apply to the computed skewness values. Options are 'mean', 'median',
            '2nd_quartile', '3rd_quartile', or 'full' (returns all skewness values).

        Returns
        -------
        float or list
            The summarized skewness value, or a list of skewness values if summary_type is 'full'.
        """
        try:
            # Calculate skewness based on the signal type
            skewness_list = self.compute_skewness(signal_type=signal_type)

            # Ensure skewness_list is a valid, non-empty list or array
            if skewness_list is None or not isinstance(
                skewness_list, (list, np.ndarray)
            ):
                raise ValueError("Computed skewness is None or invalid.")

            if len(skewness_list) == 0:
                raise ValueError("Computed skewness is an empty array.")

            return self._summarize_list(skewness_list, summary_type)

        except ValueError as e:
            logger.error(f"Value error in get_signal_skewness: {e}")
            return np.nan
        except Exception as e:
            logger.error(f"Unexpected error in get_signal_skewness: {e}")
            return np.nan

    def get_peak_trend_slope(
        self, peaks=None, method="linear_regression", window_size=5
    ):
        """
        Calculate the trend slope of peak values using specified method.

        Parameters
        ----------
        peaks : list or np.ndarray
            The y-values of detected peaks (peak amplitudes).
        method : str, optional
            The method to calculate trend slope. Options are 'linear_regression',
            'moving_average', and 'rate_of_change'. Default is 'linear_regression'.
        window_size : int, optional
            The window size for moving average calculation. Used only when method='moving_average'.
            Default is 5.

        Returns
        -------
        float or np.ndarray
            The calculated trend slope. If 'moving_average' method is selected, returns an array of slopes.
        """
        try:
            if peaks is None:
                if self.signal_type == "ECG":
                    peaks = self.r_peaks
                if self.signal_type == "PPG":
                    peaks = self.systolic_peaks
            if peaks.size == 0:
                return 0.0
            # Check if peaks is a valid array-like input
            if not isinstance(peaks, (list, np.ndarray)) or len(peaks) == 0:
                raise ValueError("Peaks data is None, invalid, or empty.")

            peaks = np.array(peaks)  # Ensure peaks is an np.ndarray

            if method == "linear_regression":
                # Use linear regression to compute the slope of the trend
                x = np.arange(len(peaks))
                slope, _, _, _, _ = linregress(x, peaks)
                return slope

            elif method == "moving_average":
                # Compute the slope based on the moving average trend
                if len(peaks) < window_size:
                    raise ValueError(
                        "Window size is greater than the number of peak values."
                    )

                moving_averages = np.convolve(
                    peaks, np.ones(window_size) / window_size, mode="valid"
                )
                slopes = np.diff(moving_averages) / window_size
                return np.mean(slopes)

            elif method == "rate_of_change":
                # Calculate the overall rate of change across the entire signal
                if len(peaks) < 2:
                    raise ValueError(
                        "Rate of change requires at least two peak values."
                    )

                overall_slope = (peaks[-1] - peaks[0]) / (len(peaks) - 1)
                return overall_slope

            else:
                raise ValueError(f"Unsupported method: {method}")

        except ValueError as e:
            logger.error(f"Value error in get_peak_trend_slope: {e}")
            return np.nan
        except Exception as e:
            logger.error(f"Unexpected error in get_peak_trend_slope: {e}")
            return np.nan

    def get_amplitude_variability(
        self,
        interval_type="Sys-to-Baseline",
        baseline_method="moving_average",
        signal_type="PPG",
        method="std_dev",
    ):
        """
        Calculates the variability in amplitude over the specified interval or baseline comparison.

        Parameters
        ----------
        interval_type : str
            The interval type for amplitude calculation, with default "Sys-to-Baseline".

        signal_type : str
            The type of signal: "PPG" or "ECG".

        method : str, optional
            The method to calculate variability: "std" for standard deviation or "cv" for coefficient of variation.

        Returns
        -------
        float : Variability of the amplitudes.

        Examples
        --------
        >>> variability = wm.get_amplitude_variability(interval_type="Sys-to-Baseline", signal_type="PPG")
        >>> print(f"Amplitude variability (Sys-to-Baseline): {variability}")
        """
        # Early validation for method
        valid_methods = ["std_dev", "cv", "interquartile_range"]
        if method not in valid_methods:
            raise ValueError(f"Unsupported variability method: {method}")
        try:
            # Compute amplitudes based on the interval and signal type
            amplitudes = self.compute_amplitude(
                interval_type=interval_type,
                signal_type=signal_type,
                baseline_method=baseline_method,
            )
            # Flatten amplitudes if needed
            if isinstance(amplitudes, np.ndarray) and amplitudes.ndim > 1:
                amplitudes = amplitudes.flatten()

            # Ensure amplitudes is a valid, non-empty array
            if amplitudes is None or amplitudes.size == 0:
                raise ValueError(
                    "No amplitudes calculated; ensure peaks and baselines are detected properly."
                )

            if method == "std_dev":
                # Standard deviation of amplitude variability
                variability = np.std(amplitudes)

            elif method == "cv":
                # Coefficient of Variation (CV)
                mean_amplitude = np.mean(amplitudes)
                if mean_amplitude == 0:
                    raise ValueError(
                        "Mean amplitude is zero, cannot compute coefficient of variation."
                    )
                variability = np.std(amplitudes) / mean_amplitude

            elif method == "interquartile_range":
                # Interquartile Range (IQR)
                q1 = np.percentile(amplitudes, 25)
                q3 = np.percentile(amplitudes, 75)
                variability = q3 - q1

            return variability

        except ValueError as e:
            logger.error(f"Value error in get_systolic_amplitude_variability: {e}")
            return np.nan
        except Exception as e:
            logger.error(f"Unexpected error in get_systolic_amplitude_variability: {e}")
            return np.nan

    def get_qrs_amplitude(self, summary_type="mean"):
        """
        Calculates the QRS amplitude by finding the maximum amplitude
        between the R-to-S and R-to-Q intervals.

        Parameters
        ----------
        summary_type : str, optional
            Specifies the summary statistic to return:
            - 'mean': Mean of all QRS amplitudes
            - 'median': Median of all QRS amplitudes
            - '2nd_quartile': 25th percentile (1st quartile)
            - '3rd_quartile': 75th percentile (3rd quartile)
            - 'full': Returns all QRS amplitudes as a list
            Default is 'mean'.

        Returns
        -------
        float or list
            The summary statistic for QRS amplitude, or the list of amplitudes if `summary_type` is 'full'.
            Returns NaN if an error occurs.

        Notes
        -----
        Logs an error and returns NaN if an invalid summary type is provided or if amplitude computation fails.
        """
        if np.all(self.waveform == 0):
            return 0
        try:
            # Compute R-to-S and R-to-Q amplitudes
            rs_amplitudes = np.array(self.compute_amplitude(interval_type="R-to-S"))
            qr_amplitudes = np.array(self.compute_amplitude(interval_type="R-to-Q"))

            # Verify amplitude arrays are valid and non-empty
            if rs_amplitudes.size == 0 or qr_amplitudes.size == 0:
                raise ValueError(
                    "R-to-S or R-to-Q amplitude array is empty or invalid."
                )

            # Check if the lengths of the two areas are equal
            min_length = min(len(rs_amplitudes), len(qr_amplitudes))
            if len(rs_amplitudes) != len(qr_amplitudes):
                # Log a warning if lengths differ
                logger.warning(
                    "Mismatch in lengths for R-to-Q and R-to-S areas. Truncating to minimum length."
                )
                # Truncate both arrays to the minimum length to align them
                rs_amplitudes = rs_amplitudes[:min_length]
                qr_amplitudes = qr_amplitudes[:min_length]
            # Calculate maximum amplitude between R-S and R-Q intervals using vectorized operation
            qrs_amplitudes = np.maximum(rs_amplitudes, qr_amplitudes)

            return self._summarize_list(qrs_amplitudes, summary_type)

        except ValueError as e:
            logger.error(f"Value error in get_qrs_amplitude: {e}")
            return np.nan
        except Exception as e:
            logger.error(f"Unexpected error in get_qrs_amplitude: {e}")
            return np.nan

    def get_heart_rate(self, summary_type="mean"):
        """
        Computes the heart rate based on R-R intervals.

        Parameters
        ----------
        summary_type : str, optional
            Specifies the summary statistic to return:
            - 'mean': Mean heart rate
            - 'median': Median heart rate
            - '2nd_quartile': 25th percentile (1st quartile)
            - '3rd_quartile': 75th percentile (3rd quartile)
            - 'full': Returns all computed heart rates as a list
            Default is 'mean'.

        Returns
        -------
        float or list
            The summary statistic for heart rates, or a list of all heart rates if `summary_type` is 'full'.
            Returns NaN if an error occurs.

        Notes
        -----
        Logs an error and returns NaN if an invalid summary type is provided or if heart rate computation fails.
        """
        try:
            # Compute R-R intervals in seconds
            if self.signal_type == "ECG":
                rr_intervals = np.diff(self.r_peaks) / self.fs
            else:
                rr_intervals = np.diff(self.systolic_peaks) / self.fs

            # Ensure rr_intervals is valid and non-empty
            if rr_intervals.size == 0:
                raise ValueError("No R-R intervals found; cannot compute heart rate.")

            # Calculate heart rates from R-R intervals (in beats per minute)
            heart_rates = 60 / rr_intervals

            # Dictionary to map summary types to computations
            summary_methods = {
                "mean": np.mean(heart_rates),
                "median": np.median(heart_rates),
                "2nd_quartile": np.percentile(heart_rates, 25),
                "3rd_quartile": np.percentile(heart_rates, 75),
                "full": heart_rates.tolist(),
            }

            # Retrieve and return the computed summary based on the selected type
            if summary_type in summary_methods:
                return summary_methods[summary_type]
            else:
                raise ValueError(f"Unsupported summary type: {summary_type}")

        except ValueError as e:
            logger.error(f"Value error in get_heart_rate: {e}")
            return np.nan
        except Exception as e:
            logger.error(f"Unexpected error in get_heart_rate: {e}")
            return np.nan
