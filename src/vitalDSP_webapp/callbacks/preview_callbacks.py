"""
Preview callbacks for vitalDSP webapp.
Handles advanced preview functionality including filtering, range selection, and insights generation.
"""

import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, callback_context, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash import html

from ..services.data_service import get_data_service
from ..utils.data_processor import DataProcessor

logger = logging.getLogger(__name__)


def register_preview_callbacks(app):
    """Register all preview-related callbacks."""
    
    @app.callback(
        [Output("filtered-signals-plot", "figure"),
         Output("frequency-domain-plot", "figure"),
         Output("spectrogram-plot", "figure"),
         Output("r-trend-spo2-plot", "figure"),
         Output("coherence-plot", "figure"),
         Output("lissajous-plot", "figure"),
         Output("average-beat-plot", "figure"),
         Output("sdppg-plot", "figure"),
         Output("hr-trend-plot", "figure"),
         Output("ibi-histogram-plot", "figure"),
         Output("poincare-plot", "figure"),
         Output("cross-correlation-plot", "figure"),
         Output("preview-insights", "children")],
        [Input("preview-apply-range", "n_clicks"),
         Input("preview-filter-family", "value"),
         Input("preview-filter-response", "value"),
         Input("preview-filter-order", "value"),
         Input("preview-filter-low", "value"),
         Input("preview-filter-high", "value"),
         Input("spectrogram-window", "value"),
         Input("spectrogram-overlap", "value"),
         Input("show-spectrogram", "value"),
         Input("spectrogram-freq-max", "value"),
         Input("hr-source", "value"),
         Input("hr-min", "value"),
         Input("hr-max", "value"),
         Input("peak-prominence", "value")],
        [State("store-uploaded-data", "data"),
         State("store-data-config", "data"),
         State("preview-start-row", "value"),
         State("preview-end-row", "value"),
         State("time-column", "value"),
         State("signal-column", "value"),
         State("red-column", "value"),
         State("ir-column", "value"),
         State("waveform-column", "value")]
    )
    def update_preview_plots(n_clicks, filter_family, filter_response, filter_order, 
                           filter_low, filter_high, spec_window, spec_overlap, show_spec,
                           spec_freq_max, hr_source, hr_min, hr_max, peak_prom,
                           data_store, config_store, start_row, end_row, time_col, 
                           signal_col, red_col, ir_col, waveform_col):
        """Update all preview plots and analysis based on settings and data range."""
        try:
            logger.info("=== COMPREHENSIVE PREVIEW PLOTS CALLBACK TRIGGERED ===")
            
            if not data_store or not config_store:
                logger.info("No data store or config store, returning empty figures")
                empty_figs = [create_empty_figure() for _ in range(12)]
                return empty_figs + [[]]
            
            # Get data from store
            df = pd.DataFrame(data_store.get("dataframe", []))
            if df.empty:
                logger.info("DataFrame is empty, returning empty figures")
                empty_figs = [create_empty_figure() for _ in range(12)]
                return empty_figs + [[]]
            
            # Get sampling frequency
            sampling_freq = config_store.get("sampling_freq", 100)
            
            # Create column mapping
            column_mapping = {
                'time': time_col,
                'signal': signal_col,
                'red': red_col,
                'ir': ir_col,
                'waveform': waveform_col
            }
            
            # Get data range
            start_row = start_row or 0
            end_row = end_row or min(1000, len(df))
            end_row = min(end_row, len(df))
            
            # Extract data window
            window_df = df.iloc[start_row:end_row].copy()
            logger.info(f"Data window: {start_row} to {end_row}, shape: {window_df.shape}")
            
            # Create all plots
            filtered_fig = create_filtered_signals_plot(window_df, column_mapping, 
                                                      filter_family, filter_response, filter_order,
                                                      filter_low, filter_high, sampling_freq)
            
            freq_fig = create_frequency_domain_plot(window_df, column_mapping, sampling_freq)
            
            spec_fig = create_spectrogram_plot(window_df, column_mapping, sampling_freq,
                                             spec_window, spec_overlap, spec_freq_max) if show_spec else create_empty_figure()
            
            # Dual-source analytics
            rtrend_fig = create_r_trend_spo2_plot(window_df, column_mapping, sampling_freq,
                                                 filter_family, filter_response, filter_order,
                                                 filter_low, filter_high)
            coherence_fig = create_coherence_plot(window_df, column_mapping, sampling_freq)
            lissajous_fig = create_lissajous_plot(window_df, column_mapping, sampling_freq)
            avgbeat_fig = create_average_beat_plot(window_df, column_mapping, sampling_freq)
            sdppg_fig = create_sdppg_plot(window_df, column_mapping, sampling_freq)
            
            # Dynamics analysis
            hr_trend_fig = create_hr_trend_plot(window_df, column_mapping, sampling_freq,
                                              hr_source, hr_min, hr_max, peak_prom)
            ibi_hist_fig = create_ibi_histogram_plot(window_df, column_mapping, sampling_freq,
                                                    hr_source, hr_min, hr_max, peak_prom)
            poincare_fig = create_poincare_plot(window_df, column_mapping, sampling_freq,
                                              hr_source, hr_min, hr_max, peak_prom)
            xcorr_fig = create_cross_correlation_plot(window_df, column_mapping, sampling_freq)
            
            # Generate insights
            insights = generate_preview_insights(window_df, column_mapping, sampling_freq,
                                              filter_family, filter_response, filter_order,
                                              filter_low, filter_high)
            
            logger.info("All preview plots and analysis generated successfully")
            return [filtered_fig, freq_fig, spec_fig, rtrend_fig, coherence_fig, lissajous_fig,
                   avgbeat_fig, sdppg_fig, hr_trend_fig, ibi_hist_fig, poincare_fig, xcorr_fig,
                   insights]
            
        except Exception as e:
            logger.error(f"Error updating preview plots: {e}")
            empty_figs = [create_empty_figure() for _ in range(12)]
            return empty_figs + [[]]
    
    @app.callback(
        [Output("preview-start-row", "value"),
         Output("preview-end-row", "value")],
        [Input("preview-range-slider", "value")],
        prevent_initial_call=True
    )
    def update_range_inputs(range_values):
        """Update range input fields when slider changes."""
        if not range_values:
            raise PreventUpdate
        return range_values[0], range_values[1]
    
    @app.callback(
        [Output("preview-range-slider", "min"),
         Output("preview-range-slider", "max"),
         Output("preview-range-slider", "value"),
         Output("preview-range-slider", "marks")],
        [Input("store-uploaded-data", "data")]
    )
    def update_range_slider(data_store):
        """Update range slider when new data is loaded."""
        if not data_store:
            return 0, 1000, [0, 1000], {0: "0", 1000: "1000"}
        
        df = pd.DataFrame(data_store.get("dataframe", []))
        if df.empty:
            return 0, 1000, [0, 1000], {0: "0", 1000: "1000"}
        
        max_rows = len(df)
        step = max(100, max_rows // 20)  # Dynamic step size
        marks = {i: str(i) for i in range(0, max_rows + 1, step)}
        
        return 0, max_rows, [0, min(1000, max_rows)], marks


def create_filtered_signals_plot(df, column_mapping, filter_family, filter_response, 
                                filter_order, filter_low, filter_high, sampling_freq):
    """Create filtered signals plot showing AC components."""
    try:
        if df is None or df.empty:
            return create_empty_figure()
        
        # Create subplots for filtered signals
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("Filtered RED Channel", "Filtered IR Channel", "Filtered Waveform")
        )
        
        # Apply filter to signals
        red_signal = None
        ir_signal = None
        waveform_signal = None
        
        if column_mapping.get('red') and column_mapping['red'] in df.columns:
            red_signal = df[column_mapping['red']].values
            red_filtered = apply_filter(red_signal, filter_family, filter_response, 
                                      filter_order, filter_low, filter_high, sampling_freq)
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=red_filtered,
                mode='lines',
                name=f"Filtered {column_mapping['red']}",
                line=dict(color='#e74c3c', width=1.5)
            ), row=1, col=1)
        
        if column_mapping.get('ir') and column_mapping['ir'] in df.columns:
            ir_signal = df[column_mapping['ir']].values
            ir_filtered = apply_filter(ir_signal, filter_family, filter_response, 
                                     filter_order, filter_low, filter_high, sampling_freq)
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=ir_filtered,
                mode='lines',
                name=f"Filtered {column_mapping['ir']}",
                line=dict(color='#f39c12', width=1.5)
            ), row=2, col=1)
        
        if column_mapping.get('waveform') and column_mapping['waveform'] in df.columns:
            waveform_signal = df[column_mapping['waveform']].values
            waveform_filtered = apply_filter(waveform_signal, filter_family, filter_response, 
                                           filter_order, filter_low, filter_high, sampling_freq)
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=waveform_filtered,
                mode='lines',
                name=f"Filtered {column_mapping['waveform']}",
                line=dict(color='#3498db', width=1.5)
            ), row=3, col=1)
        
        # Configure layout
        fig.update_xaxes(title_text="Sample Index", row=3, col=1, rangeslider={"visible": True})
        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        fig.update_yaxes(title_text="Amplitude", row=2, col=1)
        fig.update_yaxes(title_text="Amplitude", row=3, col=1)
        
        fig.update_layout(
            template="plotly_white",
            title=f"Filtered Signals ({filter_family} {filter_response}, {filter_low}-{filter_high} Hz)",
            hovermode="x unified",
            height=600,
            margin=dict(l=40, r=40, t=60, b=40),
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating filtered signals plot: {e}")
        return create_empty_figure()


def create_frequency_domain_plot(df, column_mapping, sampling_freq):
    """Create frequency domain plot showing PSD."""
    try:
        if df is None or df.empty:
            return create_empty_figure()
        
        fig = go.Figure()
        
        # Add PSD for each channel
        if column_mapping.get('red') and column_mapping['red'] in df.columns:
            red_signal = df[column_mapping['red']].values
            freqs, psd_red = compute_psd(red_signal, sampling_freq)
            fig.add_trace(go.Scatter(
                x=freqs,
                y=psd_red,
                mode='lines',
                name=f"RED {column_mapping['red']}",
                line=dict(color='#e74c3c', width=1.5)
            ))
        
        if column_mapping.get('ir') and column_mapping['ir'] in df.columns:
            ir_signal = df[column_mapping['ir']].values
            freqs, psd_ir = compute_psd(ir_signal, sampling_freq)
            fig.add_trace(go.Scatter(
                x=freqs,
                y=psd_ir,
                mode='lines',
                name=f"IR {column_mapping['ir']}",
                line=dict(color='#f39c12', width=1.5)
            ))
        
        if column_mapping.get('waveform') and column_mapping['waveform'] in df.columns:
            waveform_signal = df[column_mapping['waveform']].values
            freqs, psd_waveform = compute_psd(waveform_signal, sampling_freq)
            fig.add_trace(go.Scatter(
                x=freqs,
                y=psd_waveform,
                mode='lines',
                name=f"Waveform {column_mapping['waveform']}",
                line=dict(color='#3498db', width=1.5)
            ))
        
        fig.update_layout(
            template="plotly_white",
            title="Frequency Domain Analysis",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Power Spectral Density",
            height=600,
            margin=dict(l=40, r=40, t=60, b=40),
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating frequency domain plot: {e}")
        return create_empty_figure()


def generate_preview_insights(df, column_mapping, sampling_freq, filter_family, 
                             filter_response, filter_order, filter_low, filter_high):
    """Generate insights similar to sample_tool."""
    try:
        insights = []
        
        # Basic data info
        insights.append(html.Span(f"Rows: {len(df):,}", className="badge bg-primary"))
        insights.append(html.Span(f"Fs: {sampling_freq} Hz", className="badge bg-info"))
        
        # Filter info
        insights.append(html.Span(f"Filter: {filter_family} {filter_response}", className="badge bg-secondary"))
        insights.append(html.Span(f"Order: {filter_order}", className="badge bg-secondary"))
        insights.append(html.Span(f"Range: {filter_low}-{filter_high} Hz", className="badge bg-secondary"))
        
        # Calculate physiological metrics if possible
        if column_mapping.get('red') and column_mapping.get('ir'):
            red_signal = df[column_mapping['red']].values
            ir_signal = df[column_mapping['ir']].values
            
            # DC components
            dc_red = float(np.mean(red_signal))
            dc_ir = float(np.mean(ir_signal))
            insights.append(html.Span(f"DC_red={dc_red:.1f}", className="badge bg-success"))
            insights.append(html.Span(f"DC_ir={dc_ir:.1f}", className="badge bg-success"))
            
            # Apply filter for AC components
            red_ac = apply_filter(red_signal, filter_family, filter_response, 
                                filter_order, filter_low, filter_high, sampling_freq)
            ir_ac = apply_filter(ir_signal, filter_family, filter_response, 
                               filter_order, filter_low, filter_high, sampling_freq)
            
            # SNR estimation
            snr_red = estimate_snr(red_ac)
            snr_ir = estimate_snr(ir_ac)
            if snr_red:
                insights.append(html.Span(f"SNR~ red={snr_red:.2f}", className="badge bg-warning"))
            if snr_ir:
                insights.append(html.Span(f"SNR~ ir={snr_ir:.2f}", className="badge bg-warning"))
            
            # Heart rate estimation
            hr_psd = estimate_heart_rate_psd(ir_ac, sampling_freq)
            if hr_psd:
                insights.append(html.Span(f"HR_psd={hr_psd:.1f} bpm", className="badge bg-danger"))
            
            # Respiratory rate estimation
            rr_psd = estimate_respiratory_rate_psd(ir_ac, sampling_freq)
            if rr_psd:
                insights.append(html.Span(f"RR_psd={rr_psd:.1f} rpm", className="badge bg-info"))
            
            # SpO2 estimation (simplified)
            spo2, r_ratio, pi = estimate_spo2_simplified(red_signal, ir_signal, red_ac, ir_ac)
            if spo2:
                insights.append(html.Span(f"SpO₂={spo2:.1f}%", className="badge bg-success"))
            if r_ratio:
                insights.append(html.Span(f"R={r_ratio:.4f}", className="badge bg-primary"))
            if pi:
                insights.append(html.Span(f"PI={pi:.2f}%", className="badge bg-info"))
        
        return insights
        
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        return [html.Span("Error generating insights", className="badge bg-danger")]


def apply_filter(signal, filter_family, filter_response, order, low_freq, high_freq, sampling_freq):
    """Apply digital filter to signal."""
    try:
        from scipy import signal as scipy_signal
        
        # Normalize frequencies
        low_norm = low_freq / (sampling_freq / 2)
        high_norm = high_freq / (sampling_freq / 2)
        
        if filter_response == "bandpass":
            b, a = scipy_signal.butter(order, [low_norm, high_norm], btype='band')
        elif filter_response == "lowpass":
            b, a = scipy_signal.butter(order, high_norm, btype='low')
        elif filter_response == "highpass":
            b, a = scipy_signal.butter(order, low_norm, btype='high')
        elif filter_response == "bandstop":
            b, a = scipy_signal.butter(order, [low_norm, high_norm], btype='bandstop')
        else:
            return signal
        
        filtered = scipy_signal.filtfilt(b, a, signal)
        return filtered
        
    except ImportError:
        logger.warning("SciPy not available, returning unfiltered signal")
        return signal
    except Exception as e:
        logger.error(f"Error applying filter: {e}")
        return signal


def compute_psd(signal, sampling_freq):
    """Compute power spectral density."""
    try:
        from scipy import signal as scipy_signal
        
        freqs, psd = scipy_signal.welch(signal, fs=sampling_freq, nperseg=min(256, len(signal)//4))
        return freqs, psd
        
    except ImportError:
        logger.warning("SciPy not available, returning dummy PSD")
        freqs = np.linspace(0, sampling_freq/2, 100)
        psd = np.random.rand(100)
        return freqs, psd
    except Exception as e:
        logger.error(f"Error computing PSD: {e}")
        freqs = np.linspace(0, sampling_freq/2, 100)
        psd = np.random.rand(100)
        return freqs, psd


def estimate_snr(signal):
    """Estimate signal-to-noise ratio."""
    try:
        if len(signal) == 0:
            return None
        
        # Simple SNR estimation using variance ratio
        signal_var = np.var(signal)
        if signal_var == 0:
            return None
        
        # Estimate noise as high-frequency components
        noise = np.diff(signal)
        noise_var = np.var(noise)
        
        if noise_var == 0:
            return None
        
        snr = signal_var / noise_var
        return snr
        
    except Exception as e:
        logger.error(f"Error estimating SNR: {e}")
        return None


def estimate_heart_rate_psd(signal, sampling_freq):
    """Estimate heart rate from PSD."""
    try:
        freqs, psd = compute_psd(signal, sampling_freq)
        
        # Focus on heart rate range (0.6-3.5 Hz = 36-210 bpm)
        hr_mask = (freqs >= 0.6) & (freqs <= 3.5)
        hr_freqs = freqs[hr_mask]
        hr_psd = psd[hr_mask]
        
        if len(hr_psd) == 0:
            return None
        
        # Find peak frequency
        peak_idx = np.argmax(hr_psd)
        peak_freq = hr_freqs[peak_idx]
        
        # Convert to BPM
        hr_bpm = peak_freq * 60
        return hr_bpm
        
    except Exception as e:
        logger.error(f"Error estimating heart rate: {e}")
        return None


def estimate_respiratory_rate_psd(signal, sampling_freq):
    """Estimate respiratory rate from PSD."""
    try:
        freqs, psd = compute_psd(signal, sampling_freq)
        
        # Focus on respiratory rate range (0.1-0.5 Hz = 6-30 rpm)
        rr_mask = (freqs >= 0.1) & (freqs <= 0.5)
        rr_freqs = freqs[rr_mask]
        rr_psd = psd[rr_mask]
        
        if len(rr_psd) == 0:
            return None
        
        # Find peak frequency
        peak_idx = np.argmax(rr_psd)
        peak_freq = rr_freqs[peak_idx]
        
        # Convert to RPM
        rr_rpm = peak_freq * 60
        return rr_rpm
        
    except Exception as e:
        logger.error(f"Error estimating respiratory rate: {e}")
        return None


def estimate_spo2_simplified(red_signal, ir_signal, red_ac, ir_ac):
    """Simplified SpO2 estimation."""
    try:
        # Calculate R-ratio (simplified)
        red_ac_rms = np.sqrt(np.mean(red_ac**2))
        ir_ac_rms = np.sqrt(np.mean(ir_ac**2))
        
        if ir_ac_rms == 0:
            return None, None, None
        
        r_ratio = red_ac_rms / ir_ac_rms
        
        # Simplified SpO2 calculation (linear approximation)
        # This is a very simplified model - real SpO2 requires calibration
        spo2 = 100 - (r_ratio * 20)  # Rough approximation
        spo2 = max(70, min(100, spo2))  # Clamp to reasonable range
        
        # Perfusion Index (simplified)
        pi = (red_ac_rms / np.mean(red_signal)) * 100
        pi = max(0, min(10, pi))  # Clamp to reasonable range
        
        return spo2, r_ratio, pi
        
    except Exception as e:
        logger.error(f"Error estimating SpO2: {e}")
        return None, None, None


def create_empty_figure():
    """Create an empty figure for when no data is available."""
    fig = go.Figure()
    fig.add_annotation(
        text="No data available<br>Please upload data first",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="gray")
    )
    fig.update_layout(
        template="plotly_white",
        height=600,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig


# ============================================================================
# ADVANCED ANALYSIS FUNCTIONS (like sample_tool)
# ============================================================================

def create_spectrogram_plot(df, column_mapping, sampling_freq, window_sec, overlap, freq_max):
    """Create spectrogram plot for frequency analysis."""
    try:
        if df is None or df.empty:
            return create_empty_figure()
        
        # Use IR channel for spectrogram (most reliable for heart rate)
        signal_col = column_mapping.get('ir') or column_mapping.get('red') or column_mapping.get('waveform')
        if not signal_col or signal_col not in df.columns:
            return create_empty_figure()
        
        signal = df[signal_col].values
        
        # Compute spectrogram
        from scipy import signal as scipy_signal
        
        nperseg = int(window_sec * sampling_freq)
        noverlap = int(overlap * nperseg)
        
        freqs, times, Sxx = scipy_signal.spectrogram(
            signal, fs=sampling_freq, nperseg=nperseg, noverlap=noverlap,
            scaling='density'
        )
        
        # Limit frequency range
        freq_mask = freqs <= freq_max
        freqs = freqs[freq_mask]
        Sxx = Sxx[freq_mask, :]
        
        # Convert to dB
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        fig = go.Figure(data=go.Heatmap(
            z=Sxx_db,
            x=times,
            y=freqs,
            colorscale='Viridis',
            zmin=Sxx_db.min(),
            zmax=Sxx_db.max()
        ))
        
        fig.update_layout(
            template="plotly_white",
            title=f"Spectrogram ({window_sec}s window, {overlap*100:.0f}% overlap)",
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
            height=400,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        return fig
        
    except ImportError:
        logger.warning("SciPy not available, returning empty spectrogram")
        return create_empty_figure()
    except Exception as e:
        logger.error(f"Error creating spectrogram: {e}")
        return create_empty_figure()


def create_r_trend_spo2_plot(df, column_mapping, sampling_freq, filter_family, 
                            filter_response, filter_order, filter_low, filter_high):
    """Create R-trend and SpO2 plot."""
    try:
        if not column_mapping.get('red') or not column_mapping.get('ir'):
            return create_empty_figure()
        
        red_signal = df[column_mapping['red']].values
        ir_signal = df[column_mapping['ir']].values
        
        # Apply filter to get AC components
        red_ac = apply_filter(red_signal, filter_family, filter_response, 
                            filter_order, filter_low, filter_high, sampling_freq)
        ir_ac = apply_filter(ir_signal, filter_family, filter_response, 
                           filter_order, filter_low, filter_high, sampling_freq)
        
        # Calculate R-ratio over time windows
        window_size = int(2 * sampling_freq)  # 2-second windows
        r_ratios = []
        spo2_values = []
        time_points = []
        
        for i in range(0, len(red_ac) - window_size, window_size):
            red_window = red_ac[i:i+window_size]
            ir_window = ir_ac[i:i+window_size]
            
            red_rms = np.sqrt(np.mean(red_window**2))
            ir_rms = np.sqrt(np.mean(ir_window**2))
            
            if ir_rms > 0:
                r_ratio = red_rms / ir_rms
                spo2 = 100 - (r_ratio * 20)  # Simplified SpO2
                spo2 = max(70, min(100, spo2))
                
                r_ratios.append(r_ratio)
                spo2_values.append(spo2)
                time_points.append(i / sampling_freq)
        
        if not r_ratios:
            return create_empty_figure()
        
        fig = go.Figure()
        
        # R-ratio trace
        fig.add_trace(go.Scatter(
            x=time_points,
            y=r_ratios,
            mode='lines+markers',
            name='R-ratio',
            line=dict(color='#e74c3c', width=2),
            yaxis='y'
        ))
        
        # SpO2 trace (secondary y-axis)
        fig.add_trace(go.Scatter(
            x=time_points,
            y=spo2_values,
            mode='lines+markers',
            name='SpO₂ (%)',
            line=dict(color='#3498db', width=2),
            yaxis='y2'
        ))
        
        fig.update_layout(
            template="plotly_white",
            title="R-trend & SpO₂ Analysis",
            xaxis_title="Time (s)",
            yaxis=dict(title="R-ratio", side="left"),
            yaxis2=dict(title="SpO₂ (%)", side="right", overlaying="y"),
            height=400,
            margin=dict(l=40, r=40, t=60, b=40),
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating R-trend plot: {e}")
        return create_empty_figure()


def create_coherence_plot(df, column_mapping, sampling_freq):
    """Create coherence analysis plot between RED and IR channels."""
    try:
        if not column_mapping.get('red') or not column_mapping.get('ir'):
            return create_empty_figure()
        
        red_signal = df[column_mapping['red']].values
        ir_signal = df[column_mapping['ir']].values
        
        from scipy import signal as scipy_signal
        
        freqs, coherence = scipy_signal.coherence(red_signal, ir_signal, fs=sampling_freq)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=freqs,
            y=coherence,
            mode='lines',
            name='Coherence',
            line=dict(color='#9b59b6', width=2)
        ))
        
        fig.update_layout(
            template="plotly_white",
            title="RED-IR Coherence Analysis",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Coherence",
            height=400,
            margin=dict(l=40, r=40, t=60, b=40),
            yaxis=dict(range=[0, 1])
        )
        
        return fig
        
    except ImportError:
        logger.warning("SciPy not available, returning empty coherence plot")
        return create_empty_figure()
    except Exception as e:
        logger.error(f"Error creating coherence plot: {e}")
        return create_empty_figure()


def create_lissajous_plot(df, column_mapping, sampling_freq):
    """Create Lissajous plot (RED vs IR)."""
    try:
        if not column_mapping.get('red') or not column_mapping.get('ir'):
            return create_empty_figure()
        
        red_signal = df[column_mapping['red']].values
        ir_signal = df[column_mapping['ir']].values
        
        # Normalize signals
        red_norm = (red_signal - np.mean(red_signal)) / np.std(red_signal)
        ir_norm = (ir_signal - np.mean(ir_signal)) / np.std(ir_signal)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=red_norm,
            y=ir_norm,
            mode='markers',
            name='RED vs IR',
            marker=dict(
                color=np.arange(len(red_norm)),
                colorscale='Viridis',
                size=2,
                opacity=0.7
            )
        ))
        
        fig.update_layout(
            template="plotly_white",
            title="Lissajous Plot (RED vs IR)",
            xaxis_title="RED Channel (normalized)",
            yaxis_title="IR Channel (normalized)",
            height=400,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating Lissajous plot: {e}")
        return create_empty_figure()


def create_average_beat_plot(df, column_mapping, sampling_freq):
    """Create average beat analysis plot."""
    try:
        signal_col = column_mapping.get('waveform') or column_mapping.get('ir') or column_mapping.get('red')
        if not signal_col or signal_col not in df.columns:
            return create_empty_figure()
        
        signal = df[signal_col].values
        
        # Simple peak detection for beat segmentation
        from scipy import signal as scipy_signal
        
        peaks, _ = scipy_signal.find_peaks(signal, height=np.mean(signal) + np.std(signal))
        
        if len(peaks) < 3:
            return create_empty_figure()
        
        # Extract beats around peaks
        beat_length = int(1.5 * sampling_freq)  # 1.5 seconds per beat
        beats = []
        
        for peak in peaks:
            start = max(0, peak - beat_length//2)
            end = min(len(signal), peak + beat_length//2)
            beat = signal[start:end]
            
            # Pad to consistent length
            if len(beat) < beat_length:
                beat = np.pad(beat, (0, beat_length - len(beat)), mode='edge')
            else:
                beat = beat[:beat_length]
            
            beats.append(beat)
        
        if not beats:
            return create_empty_figure()
        
        # Calculate average beat
        avg_beat = np.mean(beats, axis=0)
        std_beat = np.std(beats, axis=0)
        
        time_axis = np.linspace(0, beat_length/sampling_freq, beat_length)
        
        fig = go.Figure()
        
        # Average beat
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=avg_beat,
            mode='lines',
            name='Average Beat',
            line=dict(color='#e74c3c', width=2)
        ))
        
        # Standard deviation envelope
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=avg_beat + std_beat,
            mode='lines',
            name='+1σ',
            line=dict(color='#3498db', width=1, dash='dash'),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=avg_beat - std_beat,
            mode='lines',
            name='-1σ',
            line=dict(color='#3498db', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(52, 152, 219, 0.1)',
            showlegend=False
        ))
        
        fig.update_layout(
            template="plotly_white",
            title=f"Average Beat Analysis ({len(beats)} beats)",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            height=400,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        return fig
        
    except ImportError:
        logger.warning("SciPy not available, returning empty average beat plot")
        return create_empty_figure()
    except Exception as e:
        logger.error(f"Error creating average beat plot: {e}")
        return create_empty_figure()


def create_sdppg_plot(df, column_mapping, sampling_freq):
    """Create SDPPG (Second Derivative Photoplethysmography) plot."""
    try:
        signal_col = column_mapping.get('waveform') or column_mapping.get('ir') or column_mapping.get('red')
        if not signal_col or signal_col not in df.columns:
            return create_empty_figure()
        
        signal = df[signal_col].values
        
        # Calculate second derivative
        from scipy import signal as scipy_signal
        
        # Apply smoothing first
        smoothed = scipy_signal.savgol_filter(signal, window_length=21, polyorder=3)
        
        # First derivative
        first_deriv = np.gradient(smoothed)
        
        # Second derivative
        second_deriv = np.gradient(first_deriv)
        
        # Normalize
        second_deriv_norm = second_deriv / np.max(np.abs(second_deriv))
        
        time_axis = np.arange(len(signal)) / sampling_freq
        
        fig = go.Figure()
        
        # Original signal
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=signal,
            mode='lines',
            name='Original Signal',
            line=dict(color='#e74c3c', width=1),
            yaxis='y'
        ))
        
        # Second derivative
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=second_deriv_norm * np.std(signal) + np.mean(signal),
            mode='lines',
            name='SDPPG',
            line=dict(color='#3498db', width=2),
            yaxis='y2'
        ))
        
        fig.update_layout(
            template="plotly_white",
            title="SDPPG Analysis",
            xaxis_title="Time (s)",
            yaxis=dict(title="Original Signal", side="left"),
            yaxis2=dict(title="SDPPG (normalized)", side="right", overlaying="y"),
            height=400,
            margin=dict(l=40, r=40, t=60, b=40),
            showlegend=True
        )
        
        return fig
        
    except ImportError:
        logger.warning("SciPy not available, returning empty SDPPG plot")
        return create_empty_figure()
    except Exception as e:
        logger.error(f"Error creating SDPPG plot: {e}")
        return create_empty_figure()


def create_hr_trend_plot(df, column_mapping, sampling_freq, hr_source, hr_min, hr_max, peak_prom):
    """Create heart rate trend plot."""
    try:
        signal_col = column_mapping.get(hr_source) or column_mapping.get('ir')
        if not signal_col or signal_col not in df.columns:
            return create_empty_figure()
        
        signal = df[signal_col].values
        
        # Peak detection for heart rate
        from scipy import signal as scipy_signal
        
        # Apply bandpass filter for heart rate range
        hr_min_freq = hr_min / 60
        hr_max_freq = hr_max / 60
        
        b, a = scipy_signal.butter(4, [hr_min_freq, hr_max_freq], btype='band', fs=sampling_freq)
        filtered = scipy_signal.filtfilt(b, a, signal)
        
        # Find peaks
        peaks, _ = scipy_signal.find_peaks(
            filtered, 
            prominence=peak_prom * np.std(filtered),
            distance=int(0.5 * sampling_freq)  # Minimum 0.5s between peaks
        )
        
        if len(peaks) < 2:
            return create_empty_figure()
        
        # Calculate heart rate from peak intervals
        peak_times = peaks / sampling_freq
        intervals = np.diff(peak_times)
        heart_rates = 60 / intervals  # Convert to BPM
        
        # Filter out unrealistic values
        valid_mask = (heart_rates >= hr_min) & (heart_rates <= hr_max)
        valid_times = peak_times[1:][valid_mask]
        valid_hr = heart_rates[valid_mask]
        
        if len(valid_hr) == 0:
            return create_empty_figure()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=valid_times,
            y=valid_hr,
            mode='lines+markers',
            name='Heart Rate',
            line=dict(color='#e74c3c', width=2),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            template="plotly_white",
            title="Heart Rate Trend",
            xaxis_title="Time (s)",
            yaxis_title="Heart Rate (BPM)",
            height=400,
            margin=dict(l=40, r=40, t=60, b=40),
            yaxis=dict(range=[hr_min, hr_max])
        )
        
        return fig
        
    except ImportError:
        logger.warning("SciPy not available, returning empty HR trend plot")
        return create_empty_figure()
    except Exception as e:
        logger.error(f"Error creating HR trend plot: {e}")
        return create_empty_figure()


def create_ibi_histogram_plot(df, column_mapping, sampling_freq, hr_source, hr_min, hr_max, peak_prom):
    """Create IBI (Inter-Beat Interval) histogram plot."""
    try:
        signal_col = column_mapping.get(hr_source) or column_mapping.get('ir')
        if not signal_col or signal_col not in df.columns:
            return create_empty_figure()
        
        signal = df[signal_col].values
        
        # Peak detection (same as HR trend)
        from scipy import signal as scipy_signal
        
        hr_min_freq = hr_min / 60
        hr_max_freq = hr_max / 60
        
        b, a = scipy_signal.butter(4, [hr_min_freq, hr_max_freq], btype='band', fs=sampling_freq)
        filtered = scipy_signal.filtfilt(b, a, signal)
        
        peaks, _ = scipy_signal.find_peaks(
            filtered, 
            prominence=peak_prom * np.std(filtered),
            distance=int(0.5 * sampling_freq)
        )
        
        if len(peaks) < 2:
            return create_empty_figure()
        
        # Calculate IBI
        peak_times = peaks / sampling_freq
        intervals = np.diff(peak_times) * 1000  # Convert to milliseconds
        
        # Filter realistic intervals
        valid_intervals = intervals[(intervals >= 300) & (intervals <= 3000)]  # 300ms to 3s
        
        if len(valid_intervals) == 0:
            return create_empty_figure()
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=valid_intervals,
            nbinsx=20,
            name='IBI Distribution',
            marker_color='#3498db',
            opacity=0.7
        ))
        
        fig.update_layout(
            template="plotly_white",
            title="Inter-Beat Interval Histogram",
            xaxis_title="IBI (ms)",
            yaxis_title="Count",
            height=400,
            margin=dict(l=40, r=40, t=60, b=40),
            showlegend=False
        )
        
        return fig
        
    except ImportError:
        logger.warning("SciPy not available, returning empty IBI histogram plot")
        return create_empty_figure()
    except Exception as e:
        logger.error(f"Error creating IBI histogram plot: {e}")
        return create_empty_figure()


def create_poincare_plot(df, column_mapping, sampling_freq, hr_source, hr_min, hr_max, peak_prom):
    """Create Poincaré plot for heart rate variability analysis."""
    try:
        signal_col = column_mapping.get(hr_source) or column_mapping.get('ir')
        if not signal_col or signal_col not in df.columns:
            return create_empty_figure()
        
        signal = df[signal_col].values
        
        # Peak detection (same as HR trend)
        from scipy import signal as scipy_signal
        
        hr_min_freq = hr_min / 60
        hr_max_freq = hr_max / 60
        
        b, a = scipy_signal.butter(4, [hr_min_freq, hr_max_freq], btype='band', fs=sampling_freq)
        filtered = scipy_signal.filtfilt(b, a, signal)
        
        peaks, _ = scipy_signal.find_peaks(
            filtered, 
            prominence=peak_prom * np.std(filtered),
            distance=int(0.5 * sampling_freq)
        )
        
        if len(peaks) < 3:
            return create_empty_figure()
        
        # Calculate IBI
        peak_times = peaks / sampling_freq
        intervals = np.diff(peak_times) * 1000  # Convert to milliseconds
        
        # Filter realistic intervals
        valid_intervals = intervals[(intervals >= 300) & (intervals <= 3000)]
        
        if len(valid_intervals) < 2:
            return create_empty_figure()
        
        # Create Poincaré plot (IBI_n vs IBI_{n+1})
        x = valid_intervals[:-1]
        y = valid_intervals[1:]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            name='Poincaré Plot',
            marker=dict(
                color=np.arange(len(x)),
                colorscale='Viridis',
                size=6,
                opacity=0.7
            )
        ))
        
        # Add identity line
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Identity Line',
            line=dict(color='red', width=1, dash='dash')
        ))
        
        fig.update_layout(
            template="plotly_white",
            title="Poincaré Plot (IBI_n vs IBI_{n+1})",
            xaxis_title="IBI_n (ms)",
            yaxis_title="IBI_{n+1} (ms)",
            height=400,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        return fig
        
    except ImportError:
        logger.warning("SciPy not available, returning empty Poincaré plot")
        return create_empty_figure()
    except Exception as e:
        logger.error(f"Error creating Poincaré plot: {e}")
        return create_empty_figure()


def create_cross_correlation_plot(df, column_mapping, sampling_freq):
    """Create cross-correlation plot between RED and IR channels."""
    try:
        if not column_mapping.get('red') or not column_mapping.get('ir'):
            return create_empty_figure()
        
        red_signal = df[column_mapping['red']].values
        ir_signal = df[column_mapping['ir']].values
        
        from scipy import signal as scipy_signal
        
        # Calculate cross-correlation
        correlation = scipy_signal.correlate(red_signal, ir_signal, mode='full')
        lags = scipy_signal.correlation_lags(len(red_signal), len(ir_signal), mode='full')
        
        # Convert lags to time
        lag_times = lags / sampling_freq
        
        # Normalize correlation
        correlation_norm = correlation / np.max(np.abs(correlation))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=lag_times,
            y=correlation_norm,
            mode='lines',
            name='Cross-correlation',
            line=dict(color='#e67e22', width=2)
        ))
        
        fig.update_layout(
            template="plotly_white",
            title="RED-IR Cross-correlation",
            xaxis_title="Lag Time (s)",
            yaxis_title="Correlation",
            height=400,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        return fig
        
    except ImportError:
        logger.warning("SciPy not available, returning empty cross-correlation plot")
        return create_empty_figure()
    except Exception as e:
        logger.error(f"Error creating cross-correlation plot: {e}")
        return create_empty_figure()
