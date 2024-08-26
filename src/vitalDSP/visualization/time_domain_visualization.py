import plotly.graph_objs as go


class TrendAnalysisVisualizer:
    def __init__(self, trend_analysis):
        self.trend_analysis = trend_analysis

    def plot_trend(self, original_signal, trend, title="Trend Analysis"):
        fig = go.Figure()

        # Original signal
        fig.add_trace(
            go.Scatter(
                x=list(range(len(original_signal))),
                y=original_signal,
                mode="lines",
                name="Original Signal",
            )
        )

        # Trend line
        fig.add_trace(
            go.Scatter(x=list(range(len(trend))), y=trend, mode="lines", name="Trend")
        )

        fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Amplitude")
        fig.show()

    def plot_moving_average(self, moving_avg, window_size, title="Moving Average"):
        self.plot_trend(
            self.trend_analysis.signal,
            moving_avg,
            f"{title} (Window Size: {window_size})",
        )

    def plot_weighted_moving_average(
        self, weighted_avg, title="Weighted Moving Average"
    ):
        self.plot_trend(self.trend_analysis.signal, weighted_avg, title)

    def plot_exponential_smoothing(
        self, exp_smoothed, alpha, title="Exponential Smoothing"
    ):
        self.plot_trend(
            self.trend_analysis.signal, exp_smoothed, f"{title} (Alpha: {alpha})"
        )

    def plot_linear_trend(self, linear_trend, title="Linear Trend"):
        self.plot_trend(self.trend_analysis.signal, linear_trend, title)

    def plot_polynomial_trend(self, polynomial_trend, degree, title="Polynomial Trend"):
        self.plot_trend(
            self.trend_analysis.signal, polynomial_trend, f"{title} (Degree: {degree})"
        )


class SignalSegmentationVisualizer:
    def __init__(self, signal_segmentation):
        self.signal_segmentation = signal_segmentation

    def plot_segments(self, segments, title="Signal Segmentation"):
        fig = go.Figure()

        for i, segment in enumerate(segments):
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(segment))),
                    y=segment,
                    mode="lines",
                    name=f"Segment {i+1}",
                )
            )

        fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Amplitude")
        fig.show()


class SignalPowerAnalysisVisualizer:
    def __init__(self, signal_power_analysis):
        self.signal_power_analysis = signal_power_analysis

    def plot_signal_with_power(self, signal, power, title="Signal with Power"):
        fig = go.Figure()

        # Original signal
        fig.add_trace(
            go.Scatter(
                x=list(range(len(signal))),
                y=signal,
                mode="lines",
                name="Original Signal",
            )
        )

        # Signal power (as a line or area under the curve)
        fig.add_trace(
            go.Scatter(x=list(range(len(signal))), y=power, mode="lines", name="Power")
        )

        fig.update_layout(
            title=title, xaxis_title="Time", yaxis_title="Amplitude/Power"
        )
        fig.show()

    def plot_psd(self, freqs, psd, title="Power Spectral Density"):
        fig = go.Figure()

        # Power Spectral Density
        fig.add_trace(go.Scatter(x=freqs, y=psd, mode="lines", name="PSD"))

        fig.update_layout(
            title=title,
            xaxis_title="Frequency (Hz)",
            yaxis_title="Power/Frequency (dB/Hz)",
        )
        fig.show()


class SignalChangeDetectionVisualizer:
    def __init__(self, signal_change_detection):
        self.signal_change_detection = signal_change_detection

    def plot_changes(
        self, original_signal, change_points, title="Signal Change Detection"
    ):
        fig = go.Figure()

        # Original signal
        fig.add_trace(
            go.Scatter(
                x=list(range(len(original_signal))),
                y=original_signal,
                mode="lines",
                name="Original Signal",
            )
        )

        # Mark change points
        fig.add_trace(
            go.Scatter(
                x=change_points,
                y=[original_signal[i] for i in change_points],
                mode="markers",
                name="Change Points",
                marker=dict(color="red", size=10),
            )
        )

        fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Amplitude")
        fig.show()


class PeakDetectionVisualizer:
    def __init__(self, peak_detection):
        self.peak_detection = peak_detection

    def plot_peaks(self, original_signal, peaks, title="Peak Detection"):
        fig = go.Figure()

        # Original signal
        fig.add_trace(
            go.Scatter(
                x=list(range(len(original_signal))),
                y=original_signal,
                mode="lines",
                name="Original Signal",
            )
        )

        # Mark peaks
        fig.add_trace(
            go.Scatter(
                x=peaks,
                y=[original_signal[i] for i in peaks],
                mode="markers",
                name="Peaks",
                marker=dict(color="orange", size=10),
            )
        )

        fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Amplitude")
        fig.show()


class EnvelopeDetectionVisualizer:
    def __init__(self, envelope_detection):
        self.envelope_detection = envelope_detection

    def plot_envelope(self, original_signal, envelope, title="Envelope Detection"):
        fig = go.Figure()

        # Original signal
        fig.add_trace(
            go.Scatter(
                x=list(range(len(original_signal))),
                y=original_signal,
                mode="lines",
                name="Original Signal",
            )
        )

        # Envelope
        fig.add_trace(
            go.Scatter(
                x=list(range(len(envelope))),
                y=envelope,
                mode="lines",
                name="Envelope",
                line=dict(dash="dash"),
            )
        )

        fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Amplitude")
        fig.show()


class CrossSignalAnalysisVisualizer:
    def __init__(self, cross_signal_analysis):
        self.cross_signal_analysis = cross_signal_analysis

    def plot_cross_signal(self, signal1, signal2, title="Cross Signal Analysis"):
        fig = go.Figure()

        # Signal 1
        fig.add_trace(
            go.Scatter(
                x=list(range(len(signal1))), y=signal1, mode="lines", name="Signal 1"
            )
        )

        # Signal 2
        fig.add_trace(
            go.Scatter(
                x=list(range(len(signal2))), y=signal2, mode="lines", name="Signal 2"
            )
        )

        fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Amplitude")
        fig.show()

    def plot_coherence(self, freqs, coherence, title="Coherence Between Signals"):
        fig = go.Figure()

        # Coherence
        fig.add_trace(go.Scatter(x=freqs, y=coherence, mode="lines", name="Coherence"))

        fig.update_layout(
            title=title, xaxis_title="Frequency (Hz)", yaxis_title="Coherence"
        )
        fig.show()

    def plot_cross_correlation(
        self, lags, cross_corr, title="Cross-Correlation Between Signals"
    ):
        fig = go.Figure()

        # Cross-Correlation
        fig.add_trace(
            go.Scatter(x=lags, y=cross_corr, mode="lines", name="Cross-Correlation")
        )

        fig.update_layout(
            title=title, xaxis_title="Lag", yaxis_title="Cross-Correlation"
        )
        fig.show()


# Trend Analysis Example
# trend_analysis = TrendAnalysis(signal)
# visualizer = TrendAnalysisVisualizer(trend_analysis)
# moving_avg = trend_analysis.compute_moving_average(window_size=5)
# visualizer.plot_moving_average(moving_avg, window_size=5)

# # Signal Segmentation Example
# segmentation = SignalSegmentation(signal)
# segments = segmentation.ml_based_segmentation(model="gmm")
# segmentation_visualizer = SignalSegmentationVisualizer(segmentation)
# segmentation_visualizer.plot_segments(segments)

# # Signal Power Analysis Example
# power_analysis = SignalPowerAnalysis(signal)
# psd_freqs, psd_values = power_analysis.compute_psd(fs=100)
# power_visualizer = SignalPowerAnalysisVisualizer(power_analysis)
# power_visualizer.plot_psd(psd_freqs, psd_values)

# # Signal Change Detection Example
# change_detection = SignalChangeDetection(signal)
# change_points = change_detection.zero_crossing_rate()
# change_visualizer = SignalChangeDetectionVisualizer(change_detection)
# change_visualizer.plot_changes(signal, change_points)

# # Peak Detection Example
# peak_detection = PeakDetection(signal)
# peaks = peak_detection.detect_peaks()
# peak_visualizer = PeakDetectionVisualizer(peak_detection)
# peak_visualizer.plot_peaks(signal, peaks)

# # Envelope Detection Example
# envelope_detection = EnvelopeDetection(signal)
# envelope = envelope_detection.hilbert_envelope()
# envelope_visualizer = EnvelopeDetectionVisualizer(envelope_detection)
# envelope_visualizer.plot_envelope(signal, envelope)

# # Cross Signal Analysis Example
# cross_analysis = CrossSignalAnalysis(signal1, signal2)
# freqs, coherence = cross_analysis.compute_coherence()
# cross_visualizer = CrossSignalAnalysisVisualizer(cross_analysis)
# cross_visualizer.plot_coherence(freqs, coherence)
