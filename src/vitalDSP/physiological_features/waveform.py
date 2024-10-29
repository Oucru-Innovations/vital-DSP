import numpy as np
from vitalDSP.utils.peak_detection import PeakDetection
from scipy.stats import skew


class WaveformMorphology:
    """
    A class for computing morphological features from physiological waveforms (ECG, PPG, EEG).

    Attributes:
        waveform (np.array): The waveform signal (ECG, PPG, EEG).
        fs (int): The sampling frequency of the signal in Hz. Default is 1000 Hz.
        signal_type (str): The type of signal ('ECG', 'PPG', 'EEG').
    """

    def __init__(self, waveform, fs=256, qrs_ratio=0.05, signal_type="ECG"):
        """
        Initializes the WaveformMorphology object.

        Args:
            waveform (np.array): The waveform signal.
            fs (int): The sampling frequency of the signal in Hz. Default is 1000 Hz.
            signal_type (str): The type of signal (ECG, PPG, EEG).
        """
        self.waveform = np.array(waveform)
        self.fs = fs  # Sampling frequency
        self.signal_type = signal_type  # 'ECG', 'PPG', 'EEG'
        self.qrs_ratio = qrs_ratio

    def detect_troughs(self, systolic_peaks=None):
        """
        Detects the troughs (valleys) in the PPG waveform between systolic peaks.

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
        if systolic_peaks is None:
            detector = PeakDetection(
                self.waveform,
                "ppg_systolic_peaks",
                **{
                    "distance": 50,
                    "window_size": 7,
                    "threshold_factor": 1.6,
                    "search_window": 6,
                    "fs": self.fs,
                }
            )
            systolic_peaks = detector.detect_peaks()
        diastolic_troughs = []
        signal_derivative = np.diff(self.waveform)

        for i in range(len(systolic_peaks) - 1):
            # Define the search range between two adjacent systolic peaks
            peak_start = systolic_peaks[i]
            peak_end = systolic_peaks[i + 1]
            search_start = peak_start + (peak_end - peak_start) // 2
            search_end = peak_end

            if search_start >= search_end or search_start < 0:
                continue

            # Calculate the adaptive flatness threshold based on the MAD of the derivative within the search range
            local_derivative = signal_derivative[search_start:search_end]
            mad_derivative = np.median(
                np.abs(local_derivative - np.median(local_derivative))
            )
            adaptive_threshold = (
                0.5 * mad_derivative
            )  # Flatness threshold set as 50% of local MAD

            # Identify flat segments based on the adaptive threshold
            flat_segment = []
            for j in range(search_start, search_end - 1):
                if abs(signal_derivative[j]) < adaptive_threshold:
                    flat_segment.append(j)

            # Find the midpoint of the longest flat segment (if multiple segments are detected)
            if flat_segment:
                flat_segment_groups = np.split(
                    flat_segment, np.where(np.diff(flat_segment) != 1)[0] + 1
                )
                longest_flat_segment = max(flat_segment_groups, key=len)
                trough_index = longest_flat_segment[len(longest_flat_segment) // 2]
                diastolic_troughs.append(trough_index)

        return np.array(diastolic_troughs)
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

    def detect_notches(self, systolic_peaks=None, diastolic_troughs=None):
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

        Example
        -------
        >>> waveform = np.sin(np.linspace(0, 2*np.pi, 100))  # Simulated PPG signal
        >>> wm = WaveformMorphology(waveform, signal_type="PPG")
        >>> notches = wm.detect_notches()
        >>> print(f"Dicrotic Notches: {notches}")
        """
        if self.signal_type != "PPG":
            raise ValueError("Notches can only be detected for PPG signals.")
        if systolic_peaks is None:
            detector = PeakDetection(
                self.waveform,
                "ppg_systolic_peaks",
                **{
                    "distance": 50,
                    "window_size": 7,
                    "threshold_factor": 1.6,
                    "search_window": 6,
                    "fs": self.fs,
                }
            )
            systolic_peaks = detector.detect_peaks()
        if diastolic_troughs is None:
            diastolic_troughs = self.detect_troughs(systolic_peaks=systolic_peaks)
        notches = []

        # Calculate the first and second derivatives of the signal
        signal_derivative = np.diff(self.waveform)
        signal_second_derivative = np.diff(signal_derivative)

        for i in range(min(len(systolic_peaks), len(diastolic_troughs))):
            peak = systolic_peaks[i]
            trough = diastolic_troughs[i]

            # Define the search range: keep it close to the peak, 10-30% of the distance to the trough
            search_start = peak + int(
                (trough - peak) * 0.1
            )  # Start 10% toward the trough
            search_end = peak + int((trough - peak) * 0.3)  # End 30% toward the trough

            # Ensure the search range is within bounds
            search_start = min(max(search_start, peak), trough)
            search_end = min(max(search_end, peak), trough)

            # Extract search section in the signal and derivatives
            # search_section = self.waveform[search_start:search_end]
            search_derivative = signal_derivative[search_start : search_end - 1]
            search_second_derivative = signal_second_derivative[
                search_start : search_end - 2
            ]

            # Identify candidate notches with slope close to zero
            candidate_notches = [
                idx
                for idx in range(len(search_derivative))
                if abs(search_derivative[idx])
                < 0.05  # Adjust threshold for "close to zero" slope
            ]

            # Filter candidates where the second derivative indicates leveling (positive values)
            candidate_indices = [
                idx
                for idx in candidate_notches
                if idx < len(search_second_derivative)
                and search_second_derivative[idx] > 0
            ]

            # Select the best candidate notch based on minimum absolute slope
            if candidate_indices:
                best_notch_idx = candidate_indices[
                    np.argmin(np.abs(search_derivative[candidate_indices]))
                ]
                notches.append(search_start + best_notch_idx)

        return np.array(notches)

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
        if diastolic_troughs is None:
            diastolic_troughs = self.detect_troughs()
        if notches is None:
            notches = self.detect_notches(diastolic_troughs=diastolic_troughs)
        diastolic_peaks = []

        for i in range(len(notches)):
            notch = notches[i]
            trough = (
                diastolic_troughs[i]
                if i < len(diastolic_troughs)
                else len(self.waveform) - 1
            )

            # Define the initial search range: notch to halfway to the trough
            search_start = notch
            search_end = notch + (trough - notch) // 2

            if search_end <= search_start or search_end > len(self.waveform):
                diastolic_peaks.append(notch)
                continue

            search_segment = self.waveform[search_start:search_end]
            segment_derivative = np.diff(search_segment)

            candidate_peaks = [
                idx
                for idx in range(1, len(segment_derivative))
                if segment_derivative[idx - 1] < 0 and segment_derivative[idx] >= 0
            ]

            # Convert candidates to absolute indices and select the most prominent peak if available
            if candidate_peaks:
                candidate_indices = [search_start + idx for idx in candidate_peaks]
                diastolic_peak_idx = max(
                    candidate_indices, key=lambda x: self.waveform[x]
                )
                diastolic_peaks.append(diastolic_peak_idx)
            else:
                # Fallback assignment: if no diastolic peak is detected, use the notch as the diastolic peak
                diastolic_peaks.append(notch)

        return diastolic_peaks

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
            detector = PeakDetection(
                self.waveform,
                "ecg_r_peak",
                **{
                    "distance": 50,
                    "window_size": 7,
                    "threshold_factor": 1.6,
                    "search_window": 6,
                }
            )
            r_peaks = detector.detect_peaks()

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
            detector = PeakDetection(
                self.waveform,
                "ecg_r_peak",
                **{
                    "distance": 50,
                    "window_size": 7,
                    "threshold_factor": 1.6,
                    "search_window": 6,
                }
            )
            r_peaks = detector.detect_peaks()

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
            detector = PeakDetection(
                self.waveform,
                "ecg_r_peak",
                **{
                    "distance": 50,
                    "window_size": 7,
                    "threshold_factor": 1.6,
                    "search_window": 6,
                }
            )
            r_peaks = detector.detect_peaks()
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
            detector = PeakDetection(
                self.waveform,
                "ecg_r_peak",
                **{
                    "distance": 50,
                    "window_size": 7,
                    "threshold_factor": 1.6,
                    "search_window": 6,
                }
            )
            r_peaks = detector.detect_peaks()
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
            detector = PeakDetection(
                self.waveform,
                "ecg_r_peak",
                **{
                    "distance": 50,
                    "window_size": 7,
                    "threshold_factor": 1.6,
                    "search_window": 6,
                }
            )
            r_peaks = detector.detect_peaks()
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
            detector = PeakDetection(
                self.waveform,
                "ecg_r_peak",
                **{
                    "distance": 50,
                    "window_size": 7,
                    "threshold_factor": 1.6,
                    "search_window": 6,
                }
            )
            r_peaks = detector.detect_peaks()
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

        return s_sessions

    def detect_r_session(self, rpeaks=None, q_session=None, s_session=None):
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
        if q_session is None:
            q_session = self.detect_q_session(r_peaks=rpeaks)
        if s_session is None:
            s_session = self.detect_s_session(r_peaks=rpeaks)
        # Ensure all input arrays are aligned in size
        if len(q_session) == 0 or len(s_session) == 0:
            return []
        r_sessions = []

        # Ensure all input arrays are aligned in size and skip the first item to avoid boundary issues
        for i in range(min(len(q_session), len(s_session))):
            q_end = q_session[i][1]
            s_start = s_session[i][0]

            # Ensure there is a valid interval for the R session
            if q_end < s_start:
                r_sessions.append((q_end, s_start))

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
        qrs_sessions : list of tuples
            Each tuple contains the start and end index of a QRS session.

        Example
        -------
        >>> waveform = np.sin(np.linspace(0, 2*np.pi, 100))  # Simulated ECG signal
        >>> wm = WaveformMorphology(waveform, signal_type="ECG")
        >>> r_peaks = PeakDetection(waveform).detect_peaks()
        >>> qrs_sessions = wm.detect_qrs_session(r_peaks)
        >>> print(f"QRS Sessions: {qrs_sessions}")
        """
        if q_session is None:
            q_session = self.detect_q_session(r_peaks=rpeaks)
        if s_session is None:
            s_session = self.detect_s_session(r_peaks=rpeaks)
        # Ensure all input arrays are aligned in size
        if len(q_session) == 0 or len(s_session) == 0:
            return []

        qrs_sessions = [(q[0], s[1]) for q, s in zip(q_session, s_session)]
        return qrs_sessions

    def compute_amplitude(self, interval_type="R-to-S", signal_type="ECG"):
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
            - For PPG:
            - "Sys-to-Notch": Between systolic peak and dicrotic notch.
            - "Notch-to-Dia": Between dicrotic notch and diastolic peak.
            - "Sys-to-Dia": Between systolic and diastolic peaks.

        signal_type : str, optional
            The type of signal: "ECG" or "PPG". Default is "ECG".

        Returns
        -------
        List[float]: A list of amplitude values for each interval within a single complex.

        Examples
        --------
        >>> amplitude_values = wm.compute_amplitude(interval_type="R-to-S", signal_type="ECG")
        >>> print(f"Amplitude values for each complex in R-to-S interval: {amplitude_values}")
        """

        if signal_type == "ECG":
            # Set peaks and valleys for ECG intervals
            if interval_type == "R-to-S":
                peaks = self.r_peaks
                valleys = self.s_valleys
                require_peak_first = True
            elif interval_type == "R-to-Q":
                peaks = self.r_peaks
                valleys = self.q_valleys
                require_peak_first = False
            elif interval_type == "P-to-Q":
                peaks = self.p_peaks
                valleys = self.q_valleys
                require_peak_first = True
            elif interval_type == "T-to-S":
                peaks = self.t_peaks
                valleys = self.s_valleys
                require_peak_first = False
            else:
                raise ValueError("Invalid interval_type for ECG.")

        elif signal_type == "PPG":
            # Set points for PPG intervals
            if interval_type == "Sys-to-Notch":
                peaks = self.sys_peaks
                valleys = self.dicrotic_notches
                require_peak_first = True
            elif interval_type == "Notch-to-Dia":
                peaks = self.dicrotic_notches
                valleys = self.dia_peaks
                require_peak_first = False
            elif interval_type == "Sys-to-Dia":
                peaks = self.sys_peaks
                valleys = self.dia_peaks
                require_peak_first = True
            else:
                raise ValueError("Invalid interval_type for PPG.")
        else:
            raise ValueError("signal_type must be 'ECG' or 'PPG'.")

        # Compute amplitude for each interval
        amplitudes = []
        for peak, valley in zip(peaks, valleys):
            if (require_peak_first and peak < valley) or (
                not require_peak_first and valley < peak
            ):
                amplitude = abs(self.waveform[peak] - self.waveform[valley])
                amplitudes.append(amplitude)

        return amplitudes

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

        if signal_type == "ECG":
            # Set peaks and valleys for ECG intervals
            if interval_type == "P-to-T":
                start_points = self.p_peaks
                end_points = self.t_peaks
                require_peak_first = True
            elif interval_type == "R-to-S":
                start_points = self.r_peaks
                end_points = self.s_valleys
                require_peak_first = True
            elif interval_type == "R-to-Q":
                start_points = self.q_valleys
                end_points = self.r_peaks
                require_peak_first = False
            elif interval_type == "P-to-Q":
                start_points = self.p_peaks
                end_points = self.q_valleys
                require_peak_first = True
            elif interval_type == "T-to-S":
                start_points = self.s_valleys
                end_points = self.t_peaks
                require_peak_first = False
            else:
                raise ValueError("Invalid interval_type for ECG.")

        elif signal_type == "PPG":
            # Set points for PPG intervals
            if interval_type == "Sys-to-Notch":
                start_points = self.sys_peaks
                end_points = self.dicrotic_notches
                require_peak_first = True
            elif interval_type == "Notch-to-Dia":
                start_points = self.dicrotic_notches
                end_points = self.dia_peaks
                require_peak_first = False
            elif interval_type == "Sys-to-Dia":
                start_points = self.sys_peaks
                end_points = self.dia_peaks
                require_peak_first = True
            else:
                raise ValueError("Invalid interval_type for PPG.")
        else:
            raise ValueError("signal_type must be 'ECG' or 'PPG'.")

        # Compute area for each interval
        volumes = []
        for start, end in zip(start_points, end_points):
            if (require_peak_first and start < end) or (
                not require_peak_first and end < start
            ):
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

        return volumes

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
            starts = self.p_peaks
            ends = self.t_peaks
        elif signal_type == "PPG":
            starts = self.sys_peaks
            ends = self.dia_peaks
        else:
            raise ValueError("signal_type must be 'ECG' or 'PPG'.")

        # Compute skewness for each complex
        skewness_values = []
        for start, end in zip(starts, ends):
            # Ensure valid intervals
            if end > start:
                complex_segment = self.waveform[start : end + 1]
                skewness_values.append(skew(complex_segment))

        return skewness_values

    def compute_qrs_duration(self):
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
        if self.signal_type != "ECG":
            raise ValueError("QRS duration can only be computed for ECG signals.")

        # Detect R-peaks in the ECG signal
        peak_detector = PeakDetection(self.waveform, method="ecg_r_peak")
        r_peaks = peak_detector.detect_peaks()
        if len(r_peaks) == 0:
            return np.nan  # Return NaN if no peaks are detected
        # Detect Q and S points around R peaks
        q_points = []
        s_points = []
        for r_peak in r_peaks:
            # Q point detection
            q_start = max(0, r_peak - int(self.fs * 0.04))
            q_end = r_peak
            q_segment = self.waveform[q_start:q_end]
            if len(q_segment) > 0:
                q_point = np.argmin(q_segment) + q_start
                q_points.append(q_point)
            else:
                q_points.append(q_start)

            # S point detection
            s_start = r_peak
            s_end = min(len(self.waveform), r_peak + int(self.fs * 0.04))
            s_segment = self.waveform[s_start:s_end]
            if len(s_segment) > 0:
                s_point = np.argmin(s_segment) + s_start
                s_points.append(s_point)
            else:
                s_points.append(s_end)

        # Compute QRS durations
        qrs_durations = [
            (s_points[i] - q_points[i]) / self.fs
            for i in range(len(r_peaks))
            if s_points[i] > q_points[i]
        ]
        qrs_duration = np.mean(qrs_durations) if qrs_durations else 0.0

        return qrs_duration

    def compute_duration(self, peaks1, peaks2, mode):
        """
        Compute the mean duration between two sets of peaks.

        Parameters
        ----------
        peaks1 : numpy.ndarray
            The first set of peaks (e.g., systolic peaks).
        peaks2 : numpy.ndarray
            The second set of peaks (e.g., diastolic peaks).
        mode : str, optional
            The type of duration computation method ("peak" or "trough"). Default is "peak".

        Returns
        -------
        duration : float
            The mean duration between the two sets of peaks in seconds.

        Examples
        --------
        >>> peaks1 = np.array([100, 200, 300])
        >>> peaks2 = np.array([150, 250, 350])
        >>> extractor = PhysiologicalFeatureExtractor(np.random.randn(1000))
        >>> duration = extractor.compute_duration(peaks1, peaks2)
        >>> print(duration)
        """
        if mode == "peak":
            durations = [
                (p2 - p1) / self.fs for p1, p2 in zip(peaks1, peaks2) if p1 < p2
            ]
        elif mode == "trough":
            durations = [(t - p) / self.fs for p, t in zip(peaks1, peaks2)]
        else:
            raise ValueError("Duration mode must be 'peak' or 'trough'.")
        return np.mean(durations) if durations else 0.0

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
        dicrotic_notches = self.detect_notches(systolic_peaks=systolic_peaks)

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

    def compute_wave_ratio(self, peaks, notch_points):
        """
        Computes the ratio of systolic to diastolic volumes in a PPG waveform.

        Args:
            peaks (np.array): Indices of systolic peaks.
            notch_points (np.array): Indices of dicrotic notches.

        Returns:
            float: The ratio of systolic to diastolic areas.

        Example:
            >>> systolic_peaks = [5, 15, 25]
            >>> dicrotic_notches = [10, 20, 30]
            >>> wave_ratio = wm.compute_wave_ratio(systolic_peaks, dicrotic_notches)
            >>> print(f"Systolic/Diastolic Ratio: {wave_ratio}")
        """
        if self.signal_type != "PPG":
            raise ValueError("Wave ratio can only be computed for PPG signals.")

        systolic_volume = self.compute_volume(peaks)
        diastolic_volume = self.compute_volume(notch_points)

        if diastolic_volume == 0:
            return np.inf  # Avoid division by zero

        return systolic_volume / diastolic_volume

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

    def compute_slope(self, peaks1, peaks2, mode="peak"):
        """
        Compute the mean slope between two sets of peaks.

        Parameters
        ----------
        peaks1 : numpy.ndarray
            The first set of peaks.
        peaks2 : numpy.ndarray
            The second set of peaks.
        mode : str, optional
        The type of duration computation method ("peak" or "trough"). Default is "peak".

        Returns
        -------
        slope : float
            The mean slope between the two sets of peaks.

        Examples
        --------
        >>> peaks1 = np.array([100, 200, 300])
        >>> peaks2 = np.array([150, 250, 350])
        >>> extractor = PhysiologicalFeatureExtractor(np.random.randn(1000))
        >>> slope = extractor.compute_slope(peaks1, peaks2)
        >>> print(slope)
        """
        if mode == "peak":
            slopes = [
                (self.waveform[p2] - self.waveform[p1]) / (p2 - p1)
                for p1, p2 in zip(peaks1, peaks2)
                if p1 < p2
            ]
        elif mode == "trough":
            slopes = [
                (self.waveform[p] - self.waveform[t]) / (p - t)
                for p, t in zip(peaks1, peaks2)
            ]
        else:
            raise ValueError("Slope mode must be 'peak' or 'trough'.")
        return np.mean(slopes) if slopes else 0.0

    def compute_curvature(self):
        """
        Computes the curvature of the waveform by analyzing its second derivative.

        Returns:
            float: The average curvature of the waveform.

        Example:
            >>> signal = [0.5, 0.7, 1.0, 0.8, 0.6]
            >>> wm = WaveformMorphology(signal)
            >>> curvature = wm.compute_curvature()
            >>> print(f"Curvature: {curvature}")
        """
        first_derivative = np.gradient(self.waveform)
        second_derivative = np.gradient(first_derivative)
        curvature = np.abs(second_derivative) / (1 + first_derivative**2) ** (3 / 2)
        return np.mean(curvature)


# from vitalDSP.utils.synthesize_data import generate_ecg_signal, generate_synthetic_ppg
# from plotly import graph_objects as go
# from vitalDSP.notebooks import process_in_chunks
# import os

# if __name__ == "__main__":
#     # Function to plot Q sessions
#     sfecg = 256
#     N = 15
#     Anoise = 0.05
#     hrmean = 70
#     fs = 256
#     duration = 60
#     step_size = fs * 20
#     FILE_PATH = os.path.join("sample_data", "public", "ecg_small.csv")
#     ecg_signal, date_col = process_in_chunks(FILE_PATH, data_type='ecg', fs=fs)
#     ecg_signal = np.array(ecg_signal)

#     # ecg_signal = generate_ecg_signal(
#     #     sfecg=sfecg, N=N, Anoise=Anoise, hrmean=hrmean
#     # )

#     detector = PeakDetection(
#         ecg_signal,"ecg_r_peak", **{
#             "distance": 50,
#             "window_size": 7,
#             "threshold_factor":1.6,
#             "search_window":6}
#         )

#     rpeaks = detector.detect_peaks()

#     waveform = WaveformMorphology(ecg_signal, fs=256, signal_type="ECG")
#     q_valleys = waveform.detect_q_valley(r_peaks=rpeaks)
#     p_peaks = waveform.detect_p_peak(r_peaks=rpeaks,q_valleys=q_valleys)
#     s_valleys = waveform.detect_s_valley(r_peaks=rpeaks)
#     t_peaks = waveform.detect_t_peak(r_peaks=rpeaks, s_valleys=s_valleys)
#     q_sessions = waveform.detect_q_session(p_peaks, q_valleys, rpeaks)
#     s_sessions = waveform.detect_s_session(t_peaks, s_valleys, rpeaks)

#     fig = go.Figure()
#         # Plot the ECG signal
#     fig.add_trace(go.Scatter(x=np.arange(len(ecg_signal)), y=ecg_signal, mode="lines", name="ECG Signal"))

#     # Plot R-peaks
#     fig.add_trace(go.Scatter(x=rpeaks, y=ecg_signal[rpeaks], mode="markers", name="R Peaks", marker=dict(color="red", size=8)))
#     fig.add_trace(go.Scatter(x=q_valleys, y=ecg_signal[q_valleys], mode="markers", name="Q Valleys", marker=dict(color="green", size=8)))
#     fig.add_trace(go.Scatter(x=s_valleys, y=ecg_signal[s_valleys], mode="markers", name="S Valleys", marker=dict(size=8)))
#     fig.add_trace(go.Scatter(x=p_peaks, y=ecg_signal[p_peaks], mode="markers", name="P Peaks", marker=dict(size=8)))
#     fig.add_trace(go.Scatter(x=t_peaks, y=ecg_signal[t_peaks], mode="markers", name="T Peaks", marker=dict(size=8)))

#     # Define margin for highlighting
#     margin = 10  # Adjust this value as needed for visual spacing
#     # Highlight each Q session with lines above and below
#     for (start, end) in q_sessions:
#         x_vals = np.arange(start, end + 1)
#         y_vals = ecg_signal[start:end + 1]

#         # Upper boundary trace
#         fig.add_trace(go.Scatter(
#             x=x_vals,
#             y=y_vals + margin,  # Shift upward by margin
#             mode="lines",
#             line=dict(color="mediumaquamarine", dash="solid"),
#             showlegend=False
#         ))

#         # Lower boundary trace
#         fig.add_trace(go.Scatter(
#             x=x_vals,
#             y=y_vals - margin,  # Shift downward by margin
#             mode="lines",
#             line=dict(color="mediumaquamarine", dash="solid"),
#             showlegend=False
#         ))

#     for (start, end) in s_sessions:
#         x_vals = np.arange(start, end + 1)
#         y_vals = ecg_signal[start:end + 1]

#         # Upper boundary trace
#         fig.add_trace(go.Scatter(
#             x=x_vals,
#             y=y_vals + margin,  # Shift upward by margin
#             mode="lines",
#             line=dict(color="seagreen", dash="solid"),
#             showlegend=False
#         ))

#         # Lower boundary trace
#         fig.add_trace(go.Scatter(
#             x=x_vals,
#             y=y_vals - margin,  # Shift downward by margin
#             mode="lines",
#             line=dict(color="seagreen", dash="solid"),
#             showlegend=False
#         ))

#     fig.update_layout(
#             title="ECG Signal with QRS-peaks/valleys and P, T peaks",
#             xaxis_title="Samples",
#             yaxis_title="Amplitude",
#             showlegend=True
#     )
#     fig.show()

#     # fs = 100
#     # time, ppg_signal = generate_synthetic_ppg(
#     #     duration=10, sampling_rate=fs, noise_level=0.01, heart_rate=60, display=False
#     # )
#     # waveform = WaveformMorphology(ppg_signal, fs=fs, signal_type="PPG")
#     # detector = PeakDetection(
#     #     ppg_signal,
#     #     "ppg_systolic_peaks",
#     #     **{
#     #         "distance": 50,
#     #         "window_size": 7,
#     #         "threshold_factor": 1.6,
#     #         "search_window": 6,
#     #         "fs": fs,
#     #     }
#     # )
#     # systolic_peaks = detector.detect_peaks()
#     # troughs = waveform.detect_troughs(systolic_peaks=systolic_peaks)
#     # notches = waveform.detect_notches(
#     #     systolic_peaks=systolic_peaks, diastolic_troughs=troughs
#     # )
#     # diastolic_peaks = waveform.detect_diastolic_peak(
#     #     notches=notches, diastolic_troughs=troughs
#     # )
#     # # detector = PeakDetection(
#     # #     ppg_signal,"abp_diastolic", **{
#     # #         "distance": 50,
#     # #         "window_size": 7,
#     # #         "threshold_factor":1.6,
#     # #         "search_window":6}
#     # # )
#     # # diastolic_peaks = detector.detect_peaks()

#     # fig = go.Figure()
#     # fig.add_trace(go.Scatter(x=time, y=ppg_signal, mode="lines", name="PPG Signal"))
#     # fig.add_trace(
#     #     go.Scatter(
#     #         x=time[systolic_peaks],
#     #         y=ppg_signal[systolic_peaks],
#     #         mode="markers",
#     #         name="Systolic Peaks",
#     #     )
#     # )
#     # fig.add_trace(
#     #     go.Scatter(
#     #         x=time[troughs],
#     #         y=ppg_signal[troughs],
#     #         name="Diastolic Troughs",
#     #         mode="markers",
#     #     )
#     # )
#     # fig.add_trace(
#     #     go.Scatter(
#     #         x=time[notches], y=ppg_signal[notches], name="Notches", mode="markers"
#     #     )
#     # )
#     # fig.add_trace(
#     #     go.Scatter(
#     #         x=time[diastolic_peaks],
#     #         y=ppg_signal[diastolic_peaks],
#     #         name="Diastolic Peaks",
#     #         mode="markers",
#     #     )
#     # )
#     # fig.show()
