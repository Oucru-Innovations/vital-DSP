"""
Plot generation and data processing callbacks for PPG analysis.

This module contains the main callback logic for generating all plots and insights
in the PPG analysis application. It has been refactored into three main classes:

- PlotManager: Handles all plot generation methods with theme management
- DataProcessor: Manages data loading, processing, and error handling
- InsightGenerator: Calculates and presents analysis insights

The main update_plots callback orchestrates these classes to provide a clean,
maintainable separation of concerns.
"""

from pathlib import Path

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, dcc, html, no_update
from plotly.subplots import make_subplots

from ..config.settings import (
    DEFAULT_DECIM_USER,
    DEFAULT_FS,
    DEFAULT_SPEC_OVERLAP,
    DEFAULT_SPEC_WIN_SEC,
    MAX_DISPLAY_POINTS,
)
from ..utils.file_utils import read_window
from ..utils.ppg_analysis import (
    avg_beat,
    compute_hr_trend,
    estimate_spo2,
    ms_coherence,
    r_series_spo2,
    sdppg,
)
from ..utils.signal_processing import (
    apply_chain,
    auto_decimation,
    cross_correlation_lag,
    design_base_filter,
    estimate_rates_psd,
    quick_snr,
    safe_float,
    safe_int,
)


class PlotManager:
    """
    Manages plot generation with clean separation of concerns.

    This class handles all plot creation methods, theme management, and visual styling.
    It provides a consistent interface for generating various types of plots used
    in PPG signal analysis.

    Attributes:
        template (str): Plotly template to use for plots
        theme (str): Theme setting ('dark' or 'light')
        colors (dict): Theme-specific color scheme
    """

    def __init__(self, template, theme):
        """
        Initialize the PlotManager with template and theme settings.

        Args:
            template (str): Plotly template name (e.g., 'plotly_white', 'plotly_dark')
            theme (str): Theme setting ('dark' or 'light')
        """
        self.template = template
        self.theme = theme
        self.colors = self._get_theme_colors()

    def _get_theme_colors(self):
        """
        Get theme-specific colors for plot styling.

        Returns:
            dict: Dictionary containing paper_bgcolor and plot_bgcolor for the theme
        """
        if self.theme == "dark":
            return {"paper_bgcolor": "#111827", "plot_bgcolor": "#0b1220"}
        return {}

    def create_blank_figure(self, height):
        """
        Create a blank figure with specified height and theme styling.

        Args:
            height (int): Height of the figure in pixels

        Returns:
            plotly.graph_objects.Figure: Configured blank figure
        """
        fig = go.Figure()
        fig.update_layout(template=self.template, height=height, **self.colors)
        return fig

    def create_time_domain_plots(
        self,
        t,
        red,
        ir,
        waveform,
        red_ac,
        ir_ac,
        waveform_ac,
        red_col,
        ir_col,
        waveform_col,
        family,
        resp,
        order,
        n,
    ):
        """
        Generate time-domain plots for raw and filtered signals.

        This method creates three subplots: one for raw signals and one for filtered
        (AC component) signals. It applies automatic decimation to ensure plots
        remain responsive with large datasets.

        Args:
            t (np.ndarray): Time array in seconds
            red (np.ndarray): Raw red channel signal
            ir (np.ndarray): Raw infrared channel signal
            waveform (np.ndarray): Raw waveform signal
            red_ac (np.ndarray): Filtered red channel AC component
            ir_ac (np.ndarray): Filtered infrared channel AC component
            waveform_ac (np.ndarray): Filtered waveform AC component
            red_col (str): Name of the red channel column
            ir_col (str): Name of the infrared channel column
            waveform_col (str): Name of the waveform column
            family (str): Filter family used (e.g., 'butter', 'cheby1')
            resp (str): Filter response type (e.g., 'lowpass', 'bandpass')
            order (int): Filter order
            n (int): Total number of data points

        Returns:
            tuple: (fig_raw, fig_ac) - Raw and filtered signal plots
        """
        # Apply automatic decimation for display performance
        decim_eff = auto_decimation(n, 1, traces=12, cap=MAX_DISPLAY_POINTS)
        td = t[::decim_eff]
        red_d, ir_d, waveform_d = red[::decim_eff], ir[::decim_eff], waveform[::decim_eff]
        red_ad, ir_ad, waveform_ad = (
            red_ac[::decim_eff],
            ir_ac[::decim_eff],
            waveform_ac[::decim_eff],
        )

        # Create raw and filtered signal plots
        fig_raw = self._create_raw_plot(
            td, red_d, ir_d, waveform_d, red_col, ir_col, waveform_col, n, decim_eff
        )
        fig_ac = self._create_filtered_plot(
            td,
            red_ad,
            ir_ad,
            waveform_ad,
            red_col,
            ir_col,
            waveform_col,
            family,
            resp,
            order,
            decim_eff,
        )

        return fig_raw, fig_ac

    def _create_raw_plot(self, t, red, ir, waveform, red_col, ir_col, waveform_col, n, decim_eff):
        """
        Create raw signal plot with three subplots.

        Args:
            t (np.ndarray): Decimated time array
            red (np.ndarray): Decimated red channel signal
            ir (np.ndarray): Decimated infrared channel signal
            waveform (np.ndarray): Decimated waveform signal
            red_col (str): Red channel column name
            ir_col (str): Infrared channel column name
            waveform_col (str): Waveform column name
            n (int): Total data points
            decim_eff (int): Effective decimation factor

        Returns:
            plotly.graph_objects.Figure: Raw signal plot with three subplots
        """
        # Create subplots for red, infrared, and waveform channels
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f"Raw {red_col}", f"Raw {ir_col}", f"Raw {waveform_col}"),
        )

        # Add traces for all three channels
        fig.add_trace(go.Scatter(x=t, y=red, name=f"Raw {red_col}", mode="lines"), 1, 1)
        fig.add_trace(go.Scatter(x=t, y=ir, name=f"Raw {ir_col}", mode="lines"), 2, 1)
        fig.add_trace(go.Scatter(x=t, y=waveform, name=f"Raw {waveform_col}", mode="lines"), 3, 1)

        # Configure axes labels and rangeslider
        fig.update_xaxes(title_text="Time (s)", row=3, col=1, rangeslider={"visible": True})
        fig.update_yaxes(title_text="ADC", row=1, col=1)
        fig.update_yaxes(title_text="ADC", row=2, col=1)
        fig.update_yaxes(title_text="ADC", row=3, col=1)

        # Apply layout styling
        fig.update_layout(
            template=self.template,
            title=f"Raw ({n:,} rows, decim×{decim_eff})",
            hovermode="x unified",
            height=600,  # Increased height for three subplots
            **self.colors,
        )

        return fig

    def _create_filtered_plot(
        self,
        t,
        red_ac,
        ir_ac,
        waveform_ac,
        red_col,
        ir_col,
        waveform_col,
        family,
        resp,
        order,
        decim_eff,
    ):
        """
        Create filtered signal plot showing AC components.

        Args:
            t (np.ndarray): Decimated time array
            red_ac (np.ndarray): Decimated red channel AC component
            ir_ac (np.ndarray): Decimated infrared channel AC component
            waveform_ac (np.ndarray): Decimated waveform AC component
            red_col (str): Red channel column name
            ir_col (str): Infrared channel column name
            waveform_col (str): Waveform column name
            family (str): Filter family used
            resp (str): Filter response type
            order (int): Filter order
            decim_eff (int): Effective decimation factor

        Returns:
            plotly.graph_objects.Figure: Filtered signal plot with three subplots
        """
        # Create subplots for filtered red, infrared, and waveform channels
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                f"Filtered {red_col} ({family},{resp})",
                f"Filtered {ir_col} ({family},{resp})",
                f"Filtered {waveform_col} ({family},{resp})",
            ),
        )

        # Add traces for filtered AC components
        fig.add_trace(go.Scatter(x=t, y=red_ac, name=f"AC {red_col}", mode="lines"), 1, 1)
        fig.add_trace(go.Scatter(x=t, y=ir_ac, name=f"AC {ir_col}", mode="lines"), 2, 1)
        fig.add_trace(go.Scatter(x=t, y=waveform_ac, name=f"AC {waveform_col}", mode="lines"), 3, 1)

        # Configure axes labels
        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
        fig.update_yaxes(title_text="AC Component", row=1, col=1)
        fig.update_yaxes(title_text="AC Component", row=2, col=1)
        fig.update_yaxes(title_text="AC Component", row=3, col=1)

        # Apply layout styling
        fig.update_layout(
            template=self.template,
            title=f"Filtered AC ({family} {resp}, order {order})",
            hovermode="x unified",
            height=600,  # Increased height for three subplots
            **self.colors,
        )

        return fig

    def create_frequency_plots(self, red_ac, ir_ac, fs, spec_win_sec, spec_overlap, show_spec):
        """
        Generate frequency-domain plots including PSD and spectrogram.

        This method creates power spectral density plots for both channels and
        optionally generates a spectrogram visualization for time-frequency analysis.

        Args:
            red_ac (np.ndarray): Filtered red channel AC component
            ir_ac (np.ndarray): Filtered infrared channel AC component
            fs (float): Sampling frequency in Hz
            spec_win_sec (float): Spectrogram window duration in seconds
            spec_overlap (float): Spectrogram overlap percentage
            show_spec (list): List containing 'on' to enable spectrogram

        Returns:
            tuple: (fig_psd, fig_spec) - PSD and spectrogram plots
        """
        # Calculate power spectral density using Welch's method
        from scipy.signal import welch

        f_r, P_r = welch(red_ac, fs=fs, nperseg=min(len(red_ac), 2048))
        f_i, P_i = welch(ir_ac, fs=fs, nperseg=min(len(ir_ac), 2048))

        # Create PSD and spectrogram plots
        fig_psd = self._create_psd_plot(f_r, P_r, f_i, P_i)
        fig_spec = self._create_spectrogram(
            red_ac, ir_ac, fs, spec_win_sec, spec_overlap, show_spec
        )

        return fig_psd, fig_spec

    def _create_psd_plot(self, f_r, P_r, f_i, P_i):
        """
        Create power spectral density plot for both channels.

        Args:
            f_r (np.ndarray): Frequency array for red channel
            P_r (np.ndarray): Power spectrum for red channel
            f_i (np.ndarray): Frequency array for infrared channel
            P_i (np.ndarray): Power spectrum for infrared channel

        Returns:
            plotly.graph_objects.Figure: PSD plot with both channels
        """
        fig = go.Figure()
        # Add traces for both channels with logarithmic scaling
        fig.add_trace(go.Scatter(x=f_r, y=10 * np.log10(P_r + 1e-20), mode="lines", name="PSD RED"))
        fig.add_trace(go.Scatter(x=f_i, y=10 * np.log10(P_i + 1e-20), mode="lines", name="PSD IR"))

        # Configure axes labels
        fig.update_xaxes(title_text="Frequency (Hz)")
        fig.update_yaxes(title_text="Power (dB)")

        # Apply layout styling
        fig.update_layout(template=self.template, height=360, **self.colors)

        return fig

    def _create_spectrogram(self, red_ac, ir_ac, fs, spec_win_sec, spec_overlap, show_spec):
        """
        Create spectrogram plot for time-frequency analysis.

        This method generates a spectrogram visualization using the infrared channel
        to show how the frequency content of the signal changes over time.

        Args:
            red_ac (np.ndarray): Filtered red channel AC component (unused but kept for consistency)
            ir_ac (np.ndarray): Filtered infrared channel AC component
            fs (float): Sampling frequency in Hz
            spec_win_sec (float): Spectrogram window duration in seconds
            spec_overlap (float): Spectrogram overlap percentage
            show_spec (list): List containing 'on' to enable spectrogram

        Returns:
            plotly.graph_objects.Figure: Spectrogram plot or blank figure if disabled
        """
        # Check if spectrogram should be displayed
        show_spec_on = "on" in (show_spec or [])
        if not show_spec_on or len(ir_ac) <= fs:
            return self.create_blank_figure(380)

        # Import spectrogram function and calculate parameters
        from scipy.signal import spectrogram

        window_size = int(safe_float(spec_win_sec, DEFAULT_SPEC_WIN_SEC) * fs)
        overlap = int(safe_float(spec_overlap, DEFAULT_SPEC_OVERLAP) * window_size)

        # Generate spectrogram data
        f_spec, t_spec, Sxx = spectrogram(ir_ac, fs=fs, nperseg=window_size, noverlap=overlap)

        # Create heatmap visualization
        fig = go.Figure(
            data=go.Heatmap(
                x=t_spec, y=f_spec, z=10 * np.log10(Sxx + 1e-18), colorbar=dict(title="dB")
            )
        )

        # Configure axes labels
        fig.update_xaxes(title_text="Time (s)")
        fig.update_yaxes(title_text="Frequency (Hz)")

        # Apply layout styling
        fig.update_layout(template=self.template, height=380, **self.colors)

        return fig

    def create_dynamics_plots(
        self, red_ac, ir_ac, fs, hr_source, hr_min, hr_max, peak_prom, show_adv
    ):
        """
        Generate dynamics plots for heart rate analysis and signal correlation.

        This method creates various plots related to heart rate dynamics including
        HR trend, inter-beat interval histogram, Poincaré plot, and cross-correlation.

        Args:
            red_ac (np.ndarray): Filtered red channel AC component
            ir_ac (np.ndarray): Filtered infrared channel AC component
            fs (float): Sampling frequency in Hz
            hr_source (str): Source channel for HR calculation ('red' or 'ir')
            hr_min (int): Minimum HR for trend calculation
            hr_max (int): Maximum HR for trend calculation
            peak_prom (float): Peak prominence threshold for HR detection
            show_adv (list): List of flags to show advanced plots

        Returns:
            tuple: (fig_hr, fig_hist, fig_poi, fig_xc) - HR trend, IBI histogram,
                   Poincaré plot, and cross-correlation plots
        """
        # Select source channel for HR calculation
        src_ac = ir_ac if (hr_source or "ir") == "ir" else red_ac

        # Validate and set HR parameters
        hr_min = safe_int(hr_min, 40)
        hr_max = safe_int(hr_max, 180)
        peak_prom = safe_float(peak_prom, 0.5)

        # Calculate HR trend and inter-beat intervals
        t_peaks, ibis, (hr_t, hr_bpm) = compute_hr_trend(src_ac, fs, hr_min, hr_max, peak_prom)

        # Create individual dynamics plots
        fig_hr = self._create_hr_plot(hr_t, hr_bpm, show_adv)
        fig_hist = self._create_ibi_histogram(ibis, show_adv)
        fig_poi = self._create_poincare_plot(ibis, show_adv)
        fig_xc = self._create_cross_correlation(red_ac, ir_ac, fs, show_adv)

        return fig_hr, fig_hist, fig_poi, fig_xc

    def _create_hr_plot(self, hr_t, hr_bpm, show_adv):
        """
        Create heart rate trend plot over time.

        Args:
            hr_t (np.ndarray): Time array for HR measurements
            hr_bpm (np.ndarray): Heart rate values in beats per minute
            show_adv (list): List of flags to show advanced plots

        Returns:
            plotly.graph_objects.Figure: HR trend plot or blank figure if disabled
        """
        # Check if HR plot should be displayed
        if "hr" not in (show_adv or []) or len(hr_t) == 0:
            return self.create_blank_figure(320)

        # Create HR trend visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hr_t, y=hr_bpm, mode="lines+markers", name="HR"))

        # Configure axes labels
        fig.update_xaxes(title_text="Time (s)")
        fig.update_yaxes(title_text="HR (bpm)")

        # Apply layout styling
        fig.update_layout(template=self.template, height=320, **self.colors)

        return fig

    def _create_ibi_histogram(self, ibis, show_adv):
        """
        Create inter-beat interval histogram plot.

        Args:
            ibis (np.ndarray): Array of inter-beat intervals in seconds
            show_adv (list): List of flags to show advanced plots

        Returns:
            plotly.graph_objects.Figure: IBI histogram or blank figure if disabled
        """
        # Check if histogram should be displayed
        if "hist" not in (show_adv or []) or len(ibis) <= 1:
            return self.create_blank_figure(280)

        # Create histogram visualization
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=1000.0 * ibis, nbinsx=30, name="IBI"))

        # Configure axes labels
        fig.update_xaxes(title_text="IBI (ms)")
        fig.update_yaxes(title_text="Count")

        # Apply layout styling
        fig.update_layout(template=self.template, height=280, **self.colors)

        return fig

    def _create_poincare_plot(self, ibis, show_adv):
        """
        Create Poincaré plot for heart rate variability analysis.

        This plot shows the relationship between consecutive inter-beat intervals,
        useful for analyzing heart rate variability patterns.

        Args:
            ibis (np.ndarray): Array of inter-beat intervals in seconds
            show_adv (list): List of flags to show advanced plots

        Returns:
            plotly.graph_objects.Figure: Poincaré plot or blank figure if disabled
        """
        # Check if Poincaré plot should be displayed
        if "poi" not in (show_adv or []) or len(ibis) <= 2:
            return self.create_blank_figure(280)

        # Create Poincaré plot (IBI_n vs IBI_{n+1})
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=1000.0 * ibis[:-1], y=1000.0 * ibis[1:], mode="markers", name="IBI pairs")
        )

        # Configure axes labels
        fig.update_xaxes(title_text="IBI_n (ms)")
        fig.update_yaxes(title_text="IBI_{n+1} (ms)")

        # Apply layout styling
        fig.update_layout(template=self.template, height=280, **self.colors)

        return fig

    def _create_cross_correlation(self, red_ac, ir_ac, fs, show_adv):
        """
        Create cross-correlation plot between red and infrared channels.

        This plot shows the temporal relationship between the two channels,
        useful for analyzing signal synchronization and delays.

        Args:
            red_ac (np.ndarray): Filtered red channel AC component
            ir_ac (np.ndarray): Filtered infrared channel AC component
            fs (float): Sampling frequency in Hz
            show_adv (list): List of flags to show advanced plots

        Returns:
            plotly.graph_objects.Figure: Cross-correlation plot or blank figure if disabled
        """
        # Check if cross-correlation plot should be displayed
        if "xcorr" not in (show_adv or []) or len(red_ac) <= 10:
            return self.create_blank_figure(280)

        # Calculate cross-correlation
        lags, correlation, max_corr_lag = cross_correlation_lag(red_ac, ir_ac, max_lag=1.0)
        if lags is None:
            return self.create_blank_figure(280)

        # Create cross-correlation visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lags, y=correlation, mode="lines", name="xcorr"))
        fig.add_vline(x=max_corr_lag, line_dash="dash")

        # Configure axes labels
        fig.update_xaxes(title_text="Lag (s) [positive = RED leads IR]")
        fig.update_yaxes(title_text="Correlation")

        # Apply layout styling
        fig.update_layout(template=self.template, height=280, **self.colors)

        return fig

    def create_dual_source_plots(self, red, ir, red_ac, ir_ac, fs, t):
        """
        Generate dual-source analytics plots for comprehensive PPG analysis.

        This method creates plots that analyze the relationship between red and
        infrared channels, including R-trend, coherence, Lissajous, average beat,
        and second derivative PPG plots.

        Args:
            red (np.ndarray): Raw red channel signal
            ir (np.ndarray): Raw infrared channel signal
            red_ac (np.ndarray): Filtered red channel AC component
            ir_ac (np.ndarray): Filtered infrared channel AC component
            fs (float): Sampling frequency in Hz
            t (np.ndarray): Time array

        Returns:
            tuple: (fig_rtrend, fig_coh, fig_liss, fig_avgbeat, fig_sdppg) -
                   R-trend, coherence, Lissajous, average beat, and SDPPG plots
        """
        # Calculate heart rate trend and inter-beat intervals
        t_peaks, ibis, (hr_t, hr_bpm) = compute_hr_trend(ir_ac, fs)

        # Calculate R-series and SpO₂ for each beat
        tB, Rbeats, spo2_beats = r_series_spo2(red, ir, red_ac, ir_ac, t_peaks, fs)

        # Calculate magnitude-squared coherence between channels
        fC, Cxy = ms_coherence(red_ac, ir_ac, fs)

        # Create individual dual-source plots
        fig_rtrend = self._create_r_trend_plot(tB, Rbeats, spo2_beats)
        fig_coh = self._create_coherence_plot(fC, Cxy)
        fig_liss = self._create_lissajous_plot(red_ac, ir_ac)
        fig_avgbeat = self._create_avg_beat_plot(red_ac, ir_ac, t_peaks, fs)
        fig_sdppg = self._create_sdppg_plot(red_ac, ir_ac, t)

        return fig_rtrend, fig_coh, fig_liss, fig_avgbeat, fig_sdppg

    def _create_r_trend_plot(self, tB, Rbeats, spo2_beats):
        """
        Create R-trend plot showing beat-by-beat SpO₂ and R values.

        This plot displays the temporal evolution of SpO₂ and R-ratio values
        calculated for each detected heartbeat.

        Args:
            tB (np.ndarray): Time array for beat measurements
            Rbeats (np.ndarray): R-ratio values for each beat
            spo2_beats (np.ndarray): SpO₂ values for each beat

        Returns:
            plotly.graph_objects.Figure: R-trend plot with dual y-axes
        """
        fig = go.Figure()

        # Add SpO₂ trace if available
        if len(tB) > 0:
            if len(spo2_beats) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=tB[: len(spo2_beats)],
                        y=spo2_beats,
                        mode="lines+markers",
                        name="SpO₂ (beat)",
                    )
                )
            # Add R-ratio trace on secondary y-axis
            fig.add_trace(
                go.Scatter(x=tB[: len(Rbeats)], y=Rbeats, yaxis="y2", mode="lines", name="R")
            )

        # Configure dual y-axes and layout
        fig.update_layout(
            template=self.template,
            height=320,
            hovermode="x unified",
            yaxis=dict(title="SpO₂ (%)"),
            yaxis2=dict(title="R", overlaying="y", side="right", showgrid=False),
            **self.colors,
            title="Beat-by-beat SpO₂ & R",
        )

        return fig

    def _create_coherence_plot(self, fC, Cxy):
        """
        Create magnitude-squared coherence plot between channels.

        This plot shows the frequency-dependent correlation between red and
        infrared channels, useful for analyzing signal quality and channel similarity.

        Args:
            fC (np.ndarray): Frequency array for coherence calculation
            Cxy (np.ndarray): Magnitude-squared coherence values (0-1)

        Returns:
            plotly.graph_objects.Figure: Coherence plot
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fC, y=Cxy, mode="lines", name="Coherence"))

        # Configure axes labels and ranges
        fig.update_xaxes(title_text="Frequency (Hz)")
        fig.update_yaxes(title_text="Cxy (0–1)", range=[0, 1])

        # Apply layout styling
        fig.update_layout(
            template=self.template,
            height=300,
            **self.colors,
            title="RED–IR Magnitude-Squared Coherence",
        )

        return fig

    def _create_lissajous_plot(self, red_ac, ir_ac):
        """
        Create Lissajous plot showing relationship between channels.

        This plot displays the relationship between red and infrared AC components
        in a phase space, useful for analyzing signal correlation and patterns.

        Args:
            red_ac (np.ndarray): Filtered red channel AC component
            ir_ac (np.ndarray): Filtered infrared channel AC component

        Returns:
            plotly.graph_objects.Figure: Lissajous plot
        """
        # Apply decimation for performance with large datasets
        dd = max(1, int(len(ir_ac) // 10000))

        # Normalize signals to z-scores for better visualization
        zr = (red_ac - red_ac.mean()) / (red_ac.std() + 1e-12)
        zi = (ir_ac - ir_ac.mean()) / (ir_ac.std() + 1e-12)
        zr_d, zi_d = zr[::dd], zi[::dd]

        # Create Lissajous visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=zr_d, y=zi_d, mode="markers", name="AC points", opacity=0.4))

        # Configure axes labels
        fig.update_xaxes(title_text="RED AC (z-score)")
        fig.update_yaxes(title_text="IR AC (z-score)")

        # Apply layout styling
        fig.update_layout(
            template=self.template, height=300, **self.colors, title="Lissajous: RED vs IR (AC)"
        )

        return fig

    def _create_avg_beat_plot(self, red_ac, ir_ac, t_peaks, fs):
        """
        Create ensemble-averaged beat shape plot.

        This plot shows the average shape of PPG beats across multiple cardiac cycles,
        useful for analyzing beat morphology and consistency.

        Args:
            red_ac (np.ndarray): Filtered red channel AC component
            ir_ac (np.ndarray): Filtered infrared channel AC component
            t_peaks (np.ndarray): Array of peak time indices
            fs (float): Sampling frequency in Hz

        Returns:
            plotly.graph_objects.Figure: Average beat plot
        """
        # Calculate ensemble-averaged beats for both channels
        t_rel, mean_red, std_red = avg_beat(red_ac, t_peaks, fs, width_s=1.2, out_len=200)
        _, mean_ir, std_ir = avg_beat(ir_ac, t_peaks, fs, width_s=1.2, out_len=200)

        fig = go.Figure()
        if len(t_rel) > 0:
            # Normalize amplitudes for better comparison
            mr = mean_red / (np.max(np.abs(mean_red)) + 1e-12)
            mi = mean_ir / (np.max(np.abs(mean_ir)) + 1e-12)
            fig.add_trace(go.Scatter(x=t_rel, y=mr, mode="lines", name="RED avg"))
            fig.add_trace(go.Scatter(x=t_rel, y=mi, mode="lines", name="IR avg"))

        # Configure axes labels
        fig.update_xaxes(title_text="Time relative to peak (s)")
        fig.update_yaxes(title_text="Normalized amplitude")

        # Apply layout styling
        fig.update_layout(
            template=self.template, height=300, **self.colors, title="Ensemble-averaged beat shape"
        )

        return fig

    def _create_sdppg_plot(self, red_ac, ir_ac, t):
        """
        Create second derivative PPG (SDPPG) plot.

        This plot shows the second derivative of the PPG signal, which can reveal
        subtle features and inflection points that may be clinically relevant.

        Args:
            red_ac (np.ndarray): Filtered red channel AC component
            ir_ac (np.ndarray): Filtered infrared channel AC component
            t (np.ndarray): Time array

        Returns:
            plotly.graph_objects.Figure: SDPPG plot with dual subplots
        """
        # Calculate second derivatives for both channels
        sd_red = sdppg(red_ac)
        sd_ir = sdppg(ir_ac)

        # Apply decimation for display performance
        decim_eff = auto_decimation(len(red_ac), 1, traces=8, cap=MAX_DISPLAY_POINTS)
        td = t[::decim_eff]

        # Create dual subplots for red and infrared SDPPG
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("SDPPG RED", "SDPPG IR"),
        )

        # Add traces for both channels
        fig.add_trace(go.Scatter(x=td, y=sd_red[::decim_eff], mode="lines", name="SDPPG RED"), 1, 1)
        fig.add_trace(go.Scatter(x=td, y=sd_ir[::decim_eff], mode="lines", name="SDPPG IR"), 2, 1)

        # Configure axes labels
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="2nd deriv.", row=1, col=1)
        fig.update_yaxes(title_text="2nd deriv.", row=2, col=1)

        # Apply layout styling
        fig.update_layout(template=self.template, height=300, **self.colors)

        return fig

    def create_waveform_plot(
        self, t, signal, signal_type, annotations, peaks=None, valleys=None, zero_crossings=None
    ):
        """
        Create waveform analysis plot with annotations.

        Args:
            t (np.ndarray): Time array
            signal (np.ndarray): Signal data
            signal_type (str): Type of signal ('raw', 'filtered', 'normalized', 'derivative')
            annotations (list): List of annotation types to show
            peaks (dict, optional): Peak information
            valleys (dict, optional): Valley information
            zero_crossings (dict, optional): Zero crossing information

        Returns:
            plotly.graph_objects.Figure: Waveform plot with annotations
        """
        fig = go.Figure()

        # Add main signal trace
        fig.add_trace(
            go.Scatter(
                x=t,
                y=signal,
                mode="lines",
                name=f"{signal_type.title()} Signal",
                line=dict(color="#1f77b4", width=1.5),
                hovertemplate="Time: %{x:.3f}s<br>Value: %{y:.2f}<extra></extra>",
            )
        )

        # Add peak annotations
        if peaks and "peaks" in annotations:
            fig.add_trace(
                go.Scatter(
                    x=peaks["times"],
                    y=peaks["values"],
                    mode="markers",
                    name="Peaks",
                    marker=dict(
                        symbol="triangle-up",
                        size=8,
                        color="red",
                        line=dict(width=1, color="darkred"),
                    ),
                    hovertemplate="Peak<br>Time: %{x:.3f}s<br>Value: %{y:.2f}<extra></extra>",
                )
            )

        # Add valley annotations
        if valleys and "valleys" in annotations:
            fig.add_trace(
                go.Scatter(
                    x=valleys["times"],
                    y=valleys["values"],
                    mode="markers",
                    name="Valleys",
                    marker=dict(
                        symbol="triangle-down",
                        size=8,
                        color="blue",
                        line=dict(width=1, color="darkblue"),
                    ),
                    hovertemplate="Valley<br>Time: %{x:.3f}s<br>Value: %{y:.2f}<extra></extra>",
                )
            )

        # Add zero crossing annotations
        if zero_crossings and "zero_crossings" in annotations:
            fig.add_trace(
                go.Scatter(
                    x=zero_crossings["times"],
                    y=np.zeros_like(zero_crossings["times"]),
                    mode="markers",
                    name="Zero Crossings",
                    marker=dict(
                        symbol="circle",
                        size=6,
                        color="green",
                        line=dict(width=1, color="darkgreen"),
                    ),
                    hovertemplate="Zero Crossing<br>Time: %{x:.3f}s<extra></extra>",
                )
            )

        # Update layout
        fig.update_layout(
            template=self.template,
            title=f"Waveform Analysis: {signal_type.title()} Signal",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            hovermode="x unified",
            height=400,
            showlegend=True,
            **self.colors,
        )

        return fig

    def create_waveform_stats_plot(self, features):
        """
        Create waveform statistics plot showing key metrics.

        Args:
            features (dict): Dictionary containing waveform statistics

        Returns:
            plotly.graph_objects.Figure: Statistics plot
        """
        fig = go.Figure()

        # Create a bar chart of key statistics
        if (
            features
            and isinstance(features, dict)
            and "statistical" in features
            and features["statistical"]
        ):
            # Extract key metrics from the statistical section
            metrics = [
                "mean",
                "rms",
                "peak_to_peak",
                "crest_factor",
                "shape_factor",
                "impulse_factor",
            ]
            values = [features["statistical"].get(metric, 0) for metric in metrics]
            labels = [metric.replace("_", " ").title() for metric in metrics]

            # Add timing features if available
            if "timing" in features:
                timing_metrics = ["num_peaks", "estimated_hr"]
                timing_values = [features["timing"].get(metric, 0) for metric in timing_metrics]
                timing_labels = ["Number of Peaks", "Estimated HR (BPM)"]

                # Add timing metrics to the plot
                for i, (label, value) in enumerate(zip(timing_labels, timing_values)):
                    if value != 0:  # Only add non-zero values
                        fig.add_trace(
                            go.Bar(
                                x=[label],
                                y=[value],
                                marker_color="#ff7f0e",  # Different color for timing metrics
                                hovertemplate="%{x}<br>%{y:.3f}<extra></extra>",
                                showlegend=False,
                            )
                        )

            # Add signal quality features if available
            if "signal_quality" in features:
                quality_metrics = ["snr_estimate", "dynamic_range"]
                quality_values = [
                    features["signal_quality"].get(metric, 0) for metric in quality_metrics
                ]
                quality_labels = ["SNR Estimate (dB)", "Dynamic Range (dB)"]

                # Add quality metrics to the plot
                for i, (label, value) in enumerate(zip(quality_labels, quality_values)):
                    if value != 0:  # Only add non-zero values
                        fig.add_trace(
                            go.Bar(
                                x=[label],
                                y=[value],
                                marker_color="#2ca02c",  # Different color for quality metrics
                                hovertemplate="%{x}<br>%{y:.3f}<extra></extra>",
                                showlegend=False,
                            )
                        )

            # Create bar chart
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=values,
                    marker_color="#1f77b4",
                    hovertemplate="%{x}<br>%{y:.3f}<extra></extra>",
                )
            )

            # Add horizontal line for mean value
            if "mean" in features["statistical"]:
                fig.add_hline(
                    y=features["statistical"]["mean"],
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Mean",
                    annotation_position="top right",
                )
        else:
            # No features available
            if not features:
                text = "No waveform features available"
            elif not isinstance(features, dict):
                text = "Invalid features format"
            elif "statistical" not in features:
                text = "Statistical features missing"
            else:
                text = "No waveform features available"

            fig.add_annotation(
                text=text,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

        # Update layout
        fig.update_layout(
            template=self.template,
            title="Waveform Statistics",
            xaxis_title="Metric",
            yaxis_title="Value",
            height=300,
            **self.colors,
        )

        return fig


class DataProcessor:
    """
    Handles data loading, processing, and validation.

    This class encapsulates the logic for reading CSV data, applying filters,
    and handling potential errors during the data processing pipeline.
    It provides a clean interface for data operations with comprehensive
    error handling and validation.
    """

    @staticmethod
    def process_data(
        path,
        window,
        red_col,
        ir_col,
        waveform_col,
        fs,
        decim_user,
        family,
        resp,
        order,
        rp,
        rs,
        notch_enable,
        notch_hz,
        notch_q,
        flags,
    ):
        """
        Process raw data and return filtered signals and error message.

        This method loads data from a CSV file, applies filtering, and handles
        potential errors during the processing pipeline.

        Args:
            path (str): Path to the CSV file
            window (dict): Dictionary containing 'start' and 'end' indices
            red_col (str): Name of the red channel column
            ir_col (str): Name of the infrared channel column
            waveform_col (str): Name of the waveform column
            fs (float): Sampling frequency in Hz
            decim_user (int): User-specified decimation factor
            family (str): Filter family (e.g., 'butter', 'cheby1')
            resp (str): Filter response type (e.g., 'lowpass', 'bandpass')
            order (int): Filter order
            rp (float): Passband ripple in dB
            rs (float): Stopband attenuation in dB
            notch_enable (list): List of 'on' to enable notch filter
            notch_hz (float): Notch filter center frequency
            notch_q (float): Notch filter quality factor
            flags (list): List of flags for filtering (e.g., 'invert', 'detrend')

        Returns:
            tuple: (t, red, ir, waveform, red_ac, ir_ac, waveform_ac, filt_err) - Time array, raw signals,
                   filtered signals, and error message
        """
        if not path or not window or red_col is None or ir_col is None or waveform_col is None:
            return None, None, None, None, None, None, None, None

        try:
            start, end = int(window["start"]), int(window["end"])
            df = read_window(path, [red_col, ir_col, waveform_col], start, end).dropna()

            if df.empty:
                return None, None, None, None, None, None, None, None

            red = df[red_col].astype(float).to_numpy()
            ir = df[ir_col].astype(float).to_numpy()
            waveform = df[waveform_col].astype(float).to_numpy()
            n = len(df)
            t = np.arange(n) / fs

            # Filter design & apply
            order = max(1, safe_int(order, 2))
            rp = safe_float(rp, 1.0)
            rs = safe_float(rs, 40.0)
            low_hz = safe_float(0.5, 0.5)
            high_hz = safe_float(5.0, 5.0)
            invert = "invert" in (flags or [])
            detrend = "detrend" in (flags or [])
            notch_on = "on" in (notch_enable or [])
            notch_hz = safe_float(notch_hz, 50.0)
            notch_q = safe_float(notch_q, 30.0)

            try:
                base_sos = design_base_filter(fs, family, resp, low_hz, high_hz, order, rp, rs)
                red_ac = apply_chain(
                    red, fs, base_sos, notch_on, notch_hz, notch_q, detrend, invert
                )
                ir_ac = apply_chain(ir, fs, base_sos, notch_on, notch_hz, notch_q, detrend, invert)
                # Waveform is not inverted (as per user request)
                waveform_ac = apply_chain(
                    waveform, fs, base_sos, notch_on, notch_hz, notch_q, detrend, False
                )
                filt_err = None
            except Exception as e:
                red_ac = np.zeros_like(red)
                ir_ac = np.zeros_like(ir)
                waveform_ac = np.zeros_like(waveform)
                filt_err = str(e)

            return t, red, ir, waveform, red_ac, ir_ac, waveform_ac, filt_err

        except Exception as e:
            return None, None, None, None, None, None, None, str(e)


class InsightGenerator:
    """
    Generates insights and file information for PPG analysis.

    This class is responsible for calculating various metrics and presenting
    them in a user-friendly format. It handles SpO₂ estimation, heart rate
    analysis, signal quality metrics, and file summary information.
    """

    @staticmethod
    def generate_insights(red, ir, red_ac, ir_ac, filt_err, template):
        """
        Generate insight chips based on processed data.

        This method calculates various metrics and presents them in a user-friendly
        format, including SpO₂, R, PI, HR, RR, DC, SNR, and DC.

        Args:
            red (np.ndarray): Raw red channel signal
            ir (np.ndarray): Raw infrared channel signal
            red_ac (np.ndarray): Filtered red channel AC component
            ir_ac (np.ndarray): Filtered infrared channel AC component
            filt_err (str): Error message from data processing
            template (str): Plotly template name

        Returns:
            list: List of Dash HTML components (pills) for insights
        """
        chips = []

        if filt_err:
            chips.append(html.Span(f"Filter error: {filt_err}", className="pill"))

        spo2, R, PI = estimate_spo2(red, ir, red_ac, ir_ac)
        hr_psd = estimate_rates_psd(ir_ac, 100, (0.6, 3.5))
        rr_psd = estimate_rates_psd(ir_ac, 100, (0.1, 0.5))
        snr_red = quick_snr(red_ac)
        snr_ir = quick_snr(ir_ac)
        dc_red, dc_ir = float(np.mean(red)), float(np.mean(ir))

        chips.extend(
            [
                html.Span(
                    f"SpO₂={f'{spo2:.2f}%' if spo2 is not None else 'n/a'}", className="pill"
                ),
                html.Span(f"R={f'{R:.4f}' if R is not None else 'n/a'}", className="pill"),
                html.Span(f"PI={f'{PI:.2f}%' if PI is not None else 'n/a'}", className="pill"),
                html.Span(f"HR_psd={f'{hr_psd:.1f} bpm' if hr_psd else 'n/a'}", className="pill"),
                html.Span(f"RR_psd={f'{rr_psd:.1f} rpm' if rr_psd else 'n/a'}", className="pill"),
                html.Span(f"DC_red={dc_red:.1f}", className="pill"),
                html.Span(f"DC_ir={dc_ir:.1f}", className="pill"),
                html.Span(f"SNR~ red={f'{snr_red:.2f}' if snr_red else 'n/a'}", className="pill"),
                html.Span(f"SNR~ ir={f'{snr_ir:.2f}' if snr_ir else 'n/a'}", className="pill"),
            ]
        )

        return chips

    @staticmethod
    def generate_file_info(
        path,
        total,
        start,
        end,
        n,
        fs,
        family,
        resp,
        order,
        low_hz,
        high_hz,
        notch_on,
        decim_user,
        hr_source,
        hr_min,
        hr_max,
        peak_prom,
        spec_win_sec,
        spec_overlap,
    ):
        """
        Generate file and window information.

        This method provides a summary of the loaded CSV file and the current
        data window, including total rows, duration, and processing parameters.

        Args:
            path (str): Path to the CSV file
            total (int): Total number of rows in the CSV file
            start (int): Start index of the current window
            end (int): End index of the current window
            n (int): Number of data points in the current window
            fs (float): Sampling frequency
            family (str): Filter family used
            resp (str): Filter response type
            order (int): Filter order
            low_hz (float): Low cutoff frequency
            high_hz (float): High cutoff frequency
            notch_on (bool): True if notch filter is enabled
            decim_user (int): User-specified decimation factor
            hr_source (str): Source of HR ('red' or 'ir')
            hr_min (int): Minimum HR for trend calculation
            hr_max (int): Maximum HR for trend calculation
            peak_prom (float): Peak prominence for HR trend calculation
            spec_win_sec (float): Spectrogram window duration in seconds
            spec_overlap (float): Spectrogram overlap percentage

        Returns:
            list: List of Dash HTML components (pills) for file info
        """
        duration = n / fs
        info = [
            html.Span(f"File: {Path(path).name}", className="pill"),
            html.Span(f"Total rows ≈ {total:,}", className="pill"),
            html.Span(f"Window {start:,}–{end:,} (n={n:,})", className="pill"),
            html.Span(f"Duration ≈ {duration:.2f}s @ fs={fs:g} Hz", className="pill"),
            html.Span(f"Family={family} • Resp={resp} • Order={order}", className="pill"),
            html.Span(
                f"Cutoffs {low_hz:g}–{high_hz:g} Hz • Notch={'on' if notch_on else 'off'}",
                className="pill",
            ),
            html.Span(f"Decimation ×{decim_user}", className="pill"),
            html.Span(
                f"HR src={hr_source or 'ir'}; HR band {hr_min}-{hr_max} bpm; prom×std={peak_prom:g}",
                className="pill",
            ),
            html.Span(
                f"Spectrogram win={safe_float(spec_win_sec, DEFAULT_SPEC_WIN_SEC):g}s ovlp={safe_float(spec_overlap, DEFAULT_SPEC_OVERLAP):g}",
                className="pill",
            ),
        ]

        return info


def register_plot_callbacks(app):
    """
    Register plot generation callbacks with the Dash app.

    This function defines the main callback that orchestrates the generation
    of all plots and insights based on user inputs. It handles data loading,
    processing, and plot generation for the time domain, frequency domain,
    dynamics, and dual-source analytics.

    Args:
        app (dash.Dash): The main Dash application object

    Returns:
        dash.callback.Output: A tuple of Output objects for the callback
    """

    @app.callback(
        Output("fig_raw", "figure"),
        Output("fig_ac", "figure"),
        Output("fig_psd", "figure"),
        Output("fig_spec", "figure"),
        Output("fig_hr_trend", "figure"),
        Output("fig_ibi_hist", "figure"),
        Output("fig_poincare", "figure"),
        Output("fig_xcorr", "figure"),
        Output("insights", "children"),
        Output("file_info", "children"),
        Output("dl_csv", "data"),
        Output("fig_rtrend", "figure"),
        Output("fig_coh", "figure"),
        Output("fig_liss", "figure"),
        Output("fig_avgbeat", "figure"),
        Output("fig_sdppg", "figure"),
        Output("fig_waveform", "figure"),
        Output("fig_waveform_stats", "figure"),
        Input("store_file_path", "data"),
        Input("store_total_rows", "data"),
        Input("store_window", "data"),
        Input("red_col", "value"),
        Input("ir_col", "value"),
        Input("waveform_col", "value"),
        Input("fs", "value"),
        Input("decim", "value"),
        Input("family", "value"),
        Input("resp", "value"),
        Input("low_hz", "value"),
        Input("high_hz", "value"),
        Input("order", "value"),
        Input("rp", "value"),
        Input("rs", "value"),
        Input("notch_enable", "value"),
        Input("notch_hz", "value"),
        Input("notch_q", "value"),
        Input("flags", "value"),
        Input("theme", "value"),
        Input("spec_win_sec", "value"),
        Input("spec_overlap", "value"),
        Input("show_spec", "value"),
        Input("hr_source", "value"),
        Input("hr_min", "value"),
        Input("hr_max", "value"),
        Input("peak_prom", "value"),
        Input("show_adv", "value"),
        Input("btn_dl_csv", "n_clicks"),
        Input("dual_source_tabs", "value"),
        Input("dynamics_tabs", "value"),
        Input("main_charts_tabs", "value"),
        Input("waveform_type", "value"),
        Input("waveform_window", "value"),
        Input("show_waveform_annotations", "value"),
        prevent_initial_call=False,
    )
    def update_plots(
        path,
        total_rows,
        window,
        red_col,
        ir_col,
        waveform_col,
        fs,
        decim_user,
        family,
        resp,
        low_hz,
        high_hz,
        order,
        rp,
        rs,
        notch_enable,
        notch_hz,
        notch_q,
        flags,
        theme,
        spec_win_sec,
        spec_overlap,
        show_spec,
        hr_source,
        hr_min,
        hr_max,
        peak_prom,
        show_adv,
        n_dl,
        dual_source_tab,
        dynamics_tab,
        main_tab,
        waveform_type,
        waveform_window,
        show_waveform_annotations,
    ):
        """
        Update all plots based on current data and settings.

        This callback orchestrates the data processing, plot generation,
        and insight generation for the main tabs of the application.

        Args:
            path (str): Path to the CSV file
            total_rows (int): Total number of rows in the CSV file
            window (dict): Current data window
            red_col (str): Selected red channel column
            ir_col (str): Selected infrared channel column
            fs (float): Sampling frequency
            decim_user (int): User-specified decimation factor
            family (str): Filter family used
            resp (str): Filter response type
            low_hz (float): Low cutoff frequency
            high_hz (float): High cutoff frequency
            order (int): Filter order
            rp (float): Passband ripple in dB
            rs (float): Stopband attenuation in dB
            notch_enable (list): List of 'on' to enable notch filter
            notch_hz (float): Notch filter center frequency
            notch_q (float): Notch filter quality factor
            flags (list): List of flags for filtering
            theme (str): Theme setting
            spec_win_sec (float): Spectrogram window duration
            spec_overlap (float): Spectrogram overlap percentage
            show_spec (list): List of 'on' to show spectrogram
            hr_source (str): Source of HR ('red' or 'ir')
            hr_min (int): Minimum HR for trend calculation
            hr_max (int): Maximum HR for trend calculation
            peak_prom (float): Peak prominence for HR trend calculation
            show_adv (list): List of 'on' to show advanced plots
            n_dl (int): Number of times CSV download button was clicked
            dual_source_tab (str): Current tab for dual-source analytics
            dynamics_tab (str): Current tab for dynamics plots
            main_tab (str): Current main tab
            waveform_type (str): Type of signal for waveform analysis
            waveform_window (float): Window duration for waveform analysis
            show_waveform_annotations (list): List of annotations to show

        Returns:
            tuple: A tuple of Output objects for the callback
        """

        template = "plotly_dark" if theme == "dark" else "plotly"

        # Set default tab values if None
        dual_source_tab = dual_source_tab or "rtrend"
        dynamics_tab = dynamics_tab or "hr"
        main_tab = main_tab or "time_domain"

        # Initialize components
        plot_mgr = PlotManager(template, theme)
        data_proc = DataProcessor()
        insight_gen = InsightGenerator()

        # Process data
        fs = safe_float(fs, DEFAULT_FS)
        decim_user = max(1, safe_int(decim_user, DEFAULT_DECIM_USER))
        total = int(total_rows or 0)

        # Get processed data
        t, red, ir, waveform, red_ac, ir_ac, waveform_ac, filt_err = data_proc.process_data(
            path,
            window,
            red_col,
            ir_col,
            waveform_col,
            fs,
            decim_user,
            family,
            resp,
            order,
            rp,
            rs,
            notch_enable,
            notch_hz,
            notch_q,
            flags,
        )

        # Check if we have valid data
        if t is None:
            # Return blank figures
            blank_420 = plot_mgr.create_blank_figure(420)
            blank_380 = plot_mgr.create_blank_figure(380)
            blank_360 = plot_mgr.create_blank_figure(360)
            blank_320 = plot_mgr.create_blank_figure(320)
            blank_300 = plot_mgr.create_blank_figure(300)
            blank_280 = plot_mgr.create_blank_figure(280)
            blank_400 = plot_mgr.create_blank_figure(400)

            msg = [html.Span("Load a CSV and pick columns/window.", className="pill")]
            return (
                blank_420,
                blank_420,
                blank_360,
                blank_380,  # 1–4
                blank_320,
                blank_280,
                blank_280,
                blank_280,  # 5–8
                msg,
                [],
                None,  # 9–11
                blank_320,
                blank_300,
                blank_300,  # 12–14
                blank_300,
                blank_300,  # 15–16
                blank_400,  # 17
                blank_300,  # 18
            )

        # Generate plots based on selected tabs
        if main_tab == "time_domain":
            fig_raw, fig_ac = plot_mgr.create_time_domain_plots(
                t,
                red,
                ir,
                waveform,
                red_ac,
                ir_ac,
                waveform_ac,
                red_col,
                ir_col,
                waveform_col,
                family,
                resp,
                order,
                len(t),
            )
        else:
            fig_raw = fig_ac = plot_mgr.create_blank_figure(420)

        if main_tab == "frequency":
            fig_psd, fig_spec = plot_mgr.create_frequency_plots(
                red_ac, ir_ac, fs, spec_win_sec, spec_overlap, show_spec
            )
        else:
            fig_psd = plot_mgr.create_blank_figure(360)
            fig_spec = plot_mgr.create_blank_figure(380)

        if main_tab == "dynamics":
            fig_hr, fig_hist, fig_poi, fig_xc = plot_mgr.create_dynamics_plots(
                red_ac, ir_ac, fs, hr_source, hr_min, hr_max, peak_prom, show_adv
            )
        else:
            fig_hr = plot_mgr.create_blank_figure(320)
            fig_hist = plot_mgr.create_blank_figure(280)
            fig_poi = plot_mgr.create_blank_figure(280)
            fig_xc = plot_mgr.create_blank_figure(280)

        if main_tab == "dual_source":
            fig_rtrend, fig_coh, fig_liss, fig_avgbeat, fig_sdppg = (
                plot_mgr.create_dual_source_plots(red, ir, red_ac, ir_ac, fs, t)
            )
        else:
            fig_rtrend = fig_coh = fig_liss = fig_avgbeat = fig_sdppg = (
                plot_mgr.create_blank_figure(300)
            )

        # Handle waveform analysis tab
        if main_tab == "waveform":
            # Get waveform analysis parameters from callback context
            waveform_type = waveform_type or "raw"  # Default to raw signal
            waveform_window = waveform_window or 5.0  # Default 5 second window
            show_annotations = show_waveform_annotations or ["peaks"]  # Default to showing peaks

            # Analyze the selected signal type
            if waveform_type == "raw":
                signal = waveform  # Use waveform channel for analysis
                signal_name = waveform_col
            elif waveform_type == "filtered":
                signal = waveform_ac
                signal_name = f"{waveform_col} (Filtered)"
            elif waveform_type == "normalized":
                signal = (waveform - np.mean(waveform)) / (np.std(waveform) + 1e-12)
                signal_name = f"{waveform_col} (Normalized)"
            elif waveform_type == "derivative":
                signal = np.gradient(waveform)
                signal_name = f"{waveform_col} (Derivative)"
            else:
                signal = waveform
                signal_name = waveform_col

            # Perform waveform analysis
            from ..utils.signal_processing import analyze_waveform, compute_waveform_features

            # Analyze waveform with annotations
            waveform_analysis = analyze_waveform(signal, fs, waveform_window, show_annotations)

            # Compute comprehensive features
            waveform_features = compute_waveform_features(signal, fs)

            # Create waveform plots
            fig_waveform = plot_mgr.create_waveform_plot(
                waveform_analysis["time"],
                waveform_analysis["signal"],
                waveform_type,
                show_annotations,
                waveform_analysis.get("peaks"),
                waveform_analysis.get("valleys"),
                waveform_analysis.get("zero_crossings"),
            )

            fig_waveform_stats = plot_mgr.create_waveform_stats_plot(waveform_features)
        else:
            fig_waveform = plot_mgr.create_blank_figure(400)
            fig_waveform_stats = plot_mgr.create_blank_figure(300)

        # Generate insights and info
        chips = insight_gen.generate_insights(red, ir, red_ac, ir_ac, filt_err, template)
        start, end = int(window["start"]), int(window["end"])
        info = insight_gen.generate_file_info(
            path,
            total,
            start,
            end,
            len(t),
            fs,
            family,
            resp,
            order,
            low_hz,
            high_hz,
            "on" in (notch_enable or []),
            decim_user,
            hr_source,
            hr_min,
            hr_max,
            peak_prom,
            spec_win_sec,
            spec_overlap,
        )

        # CSV download
        trigger = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""
        dl = None
        if trigger == "btn_dl_csv.n_clicks":
            df = read_window(path, [red_col, ir_col], start, end)
            dl = dcc.send_data_frame(df.to_csv, f"ppg_window_{start}_{end}.csv", index=False)

        return (
            fig_raw,
            fig_ac,
            fig_psd,
            fig_spec,  # 1–4
            fig_hr,
            fig_hist,
            fig_poi,
            fig_xc,  # 5–8
            chips,
            info,
            dl,  # 9–11
            fig_rtrend,
            fig_coh,
            fig_liss,  # 12–14
            fig_avgbeat,
            fig_sdppg,  # 15–16
            fig_waveform,  # 17
            fig_waveform_stats,  # 18
        )
