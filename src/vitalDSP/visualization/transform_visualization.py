import numpy as np
import plotly.graph_objs as go
from vitalDSP.transforms.stft import STFT
from vitalDSP.transforms.mfcc import MFCC
from vitalDSP.transforms.chroma_stft import ChromaSTFT
from vitalDSP.transforms.event_related_potential import EventRelatedPotential
from vitalDSP.transforms.time_freq_representation import TimeFreqRepresentation
from vitalDSP.transforms.wavelet_fft_fusion import WaveletFFTfusion
from vitalDSP.transforms.dct_wavelet_fusion import DCTWaveletFusion
from vitalDSP.transforms.pca_ica_signal_decomposition import (
    PCASignalDecomposition,
    ICASignalDecomposition,
)

class SignalDecompositionVisualization:
    """
    A class to visualize the results of PCA and ICA for signal decomposition using Plotly.

    Methods:
    - plot_pca: Plots the PCA of the signals.
    - plot_ica: Plots the ICA of the signals.
    - compare_original_pca: Compares the original signals and their PCA.
    - compare_original_ica: Compares the original signals and their ICA.
    """

    def __init__(self, signals):
        """
        Initialize the SignalDecompositionVisualization class with the signals.

        Parameters:
        signals (numpy.ndarray): The input signals (each row is a signal).
        """
        self.signals = signals.T  # Ensure that signals are transposed to (n_samples, n_signals)
        self.pca = PCASignalDecomposition(self.signals)
        self.ica = ICASignalDecomposition(self.signals)

    def plot_pca(self):
        """
        Plot the Principal Component Analysis (PCA) of the signals.

        Example Usage:
        >>> signals = np.random.rand(100, 5)
        >>> sd_viz = SignalDecompositionVisualization(signals)
        >>> sd_viz.plot_pca()
        """
        pca_result = self.pca.compute_pca()
        pca_result = pca_result.T  # Transpose to align with original signal orientation

        traces = [
            go.Scatter(y=pca_result[i], mode="lines", name=f"PC{i+1}")
            for i in range(pca_result.shape[0])
        ]
        layout = go.Layout(
            title="PCA of Signals",
            xaxis=dict(title="Sample Index"),
            yaxis=dict(title="Amplitude"),
        )
        fig = go.Figure(data=traces, layout=layout)
        fig.show()

    def plot_ica(self):
        """
        Plot the Independent Component Analysis (ICA) of the signals.

        Example Usage:
        >>> signals = np.random.rand(100, 5)
        >>> sd_viz = SignalDecompositionVisualization(signals)
        >>> sd_viz.plot_ica()
        """
        ica_result = self.ica.compute_ica()
        ica_result = ica_result.T  # Transpose to align with original signal orientation

        traces = [
            go.Scatter(y=ica_result[i], mode="lines", name=f"IC{i+1}")
            for i in range(ica_result.shape[0])
        ]
        layout = go.Layout(
            title="ICA of Signals",
            xaxis=dict(title="Sample Index"),
            yaxis=dict(title="Amplitude"),
        )
        fig = go.Figure(data=traces, layout=layout)
        fig.show()

class DCTWaveletFusionVisualization:
    """
    A class to visualize the fusion of DCT and Wavelet Transform using Plotly.

    Methods:
    - plot_fusion: Plots the fusion of DCT and Wavelet Transform.
    - compare_original_fusion: Compares the original signal and its DCT-Wavelet fusion.
    """

    def __init__(self, signal, wavelet_type="db", order=4, **kwargs):
        """
        Initialize the DCTWaveletFusionVisualization class with the signal.

        Parameters:
        signal (numpy.ndarray): The input signal.
        wavelet_type (str): The type of wavelet to use.
        order (int): The order of the wavelet.
        kwargs: Additional parameters for the specific wavelet.
        """
        self.signal = signal
        self.fusion = DCTWaveletFusion(signal, wavelet_type, order, **kwargs)

    def plot_fusion(self):
        """
        Plot the fusion of DCT and Wavelet Transform.

        Example Usage:
        >>> signal = np.sin(np.linspace(0, 10, 1000))
        >>> fusion_viz = DCTWaveletFusionVisualization(signal)
        >>> fusion_viz.plot_fusion()
        """
        fusion_result = self.fusion.compute_fusion()
        # time_axis = np.arange(len(self.signal))

        trace = go.Scatter(
            y=np.abs(fusion_result), mode="lines", name="DCT-Wavelet Fusion"
        )
        layout = go.Layout(
            title="DCT-Wavelet Fusion",
            xaxis=dict(title="Sample Index"),
            yaxis=dict(title="Amplitude"),
        )
        fig = go.Figure(data=[trace], layout=layout)
        fig.show()

    def compare_original_fusion(self):
        """
        Compare the original signal and its DCT-Wavelet fusion.

        Example Usage:
        >>> signal = np.sin(np.linspace(0, 10, 1000))
        >>> fusion_viz = DCTWaveletFusionVisualization(signal)
        >>> fusion_viz.compare_original_fusion()
        """
        fusion_result = self.fusion.compute_fusion()
        # time_axis = np.arange(len(self.signal))

        trace1 = go.Scatter(y=self.signal, mode="lines", name="Original Signal")
        trace2 = go.Scatter(
            y=np.abs(fusion_result), mode="lines", name="DCT-Wavelet Fusion"
        )

        layout1 = go.Layout(
            title="Original Signal",
            xaxis=dict(title="Sample Index"),
            yaxis=dict(title="Amplitude"),
        )
        layout2 = go.Layout(
            title="DCT-Wavelet Fusion",
            xaxis=dict(title="Sample Index"),
            yaxis=dict(title="Amplitude"),
        )

        fig1 = go.Figure(data=[trace1], layout=layout1)
        fig2 = go.Figure(data=[trace2], layout=layout2)

        fig1.show()
        fig2.show()


class WaveletFFTfusionVisualization:
    """
    A class to visualize the fusion of Wavelet Transform and FFT using Plotly.

    Methods:
    - plot_fusion: Plots the fusion of Wavelet Transform and FFT.
    - compare_original_fusion: Compares the original signal and its Wavelet-FFT fusion.
    """

    def __init__(self, signal, wavelet_type="db", order=4, **kwargs):
        """
        Initialize the WaveletFFTfusionVisualization class with the signal.

        Parameters:
        signal (numpy.ndarray): The input signal.
        wavelet_type (str): The type of wavelet to use.
        order (int): The order of the wavelet.
        kwargs: Additional parameters for the specific wavelet.
        """
        self.signal = signal
        self.fusion = WaveletFFTfusion(signal, wavelet_type, order, **kwargs)

    def plot_fusion(self):
        """
        Plot the fusion of Wavelet Transform and FFT.

        Example Usage:
        >>> signal = np.sin(np.linspace(0, 10, 1000))
        >>> fusion_viz = WaveletFFTfusionVisualization(signal)
        >>> fusion_viz.plot_fusion()
        """
        fusion_result = self.fusion.compute_fusion()
        # time_axis = np.arange(len(self.signal))

        trace = go.Scatter(
            y=np.abs(fusion_result), mode="lines", name="Wavelet-FFT Fusion"
        )
        layout = go.Layout(
            title="Wavelet-FFT Fusion",
            xaxis=dict(title="Sample Index"),
            yaxis=dict(title="Amplitude"),
        )
        fig = go.Figure(data=[trace], layout=layout)
        fig.show()

    def compare_original_fusion(self):
        """
        Compare the original signal and its Wavelet-FFT fusion.

        Example Usage:
        >>> signal = np.sin(np.linspace(0, 10, 1000))
        >>> fusion_viz = WaveletFFTfusionVisualization(signal)
        >>> fusion_viz.compare_original_fusion()
        """
        fusion_result = self.fusion.compute_fusion()
        # time_axis = np.arange(len(self.signal))

        trace1 = go.Scatter(y=self.signal, mode="lines", name="Original Signal")
        trace2 = go.Scatter(
            y=np.abs(fusion_result), mode="lines", name="Wavelet-FFT Fusion"
        )

        layout1 = go.Layout(
            title="Original Signal",
            xaxis=dict(title="Sample Index"),
            yaxis=dict(title="Amplitude"),
        )
        layout2 = go.Layout(
            title="Wavelet-FFT Fusion",
            xaxis=dict(title="Sample Index"),
            yaxis=dict(title="Amplitude"),
        )

        fig1 = go.Figure(data=[trace1], layout=layout1)
        fig2 = go.Figure(data=[trace2], layout=layout2)

        fig1.show()
        fig2.show()


class TFRVisualization:
    """
    A class to visualize Time-Frequency Representations using Plotly.

    Methods:
    - plot_tfr: Plots the Time-Frequency Representation of the signal.
    - compare_original_tfr: Compares the original signal and its Time-Frequency Representation.
    """

    def __init__(self, signal, method="stft", **kwargs):
        """
        Initialize the TFRVisualization class with the signal.

        Parameters:
        signal (numpy.ndarray): The input signal.
        method (str): The method to use for time-frequency representation ('stft', 'wavelet').
        kwargs: Additional parameters for the specific method.
        """
        self.signal = signal
        self.tfr = TimeFreqRepresentation(signal, method, **kwargs)

    def plot_tfr(self):
        """
        Plot the Time-Frequency Representation of the signal.

        Example Usage:
        >>> signal = np.sin(np.linspace(0, 10, 1000))
        >>> tfr_viz = TFRVisualization(signal)
        >>> tfr_viz.plot_tfr()
        """
        tfr_result = self.tfr.compute_tfr()
        time_axis = np.arange(tfr_result.shape[1])
        freq_axis = np.arange(tfr_result.shape[0])

        trace = go.Heatmap(
            z=np.abs(tfr_result), x=time_axis, y=freq_axis, colorscale="Viridis"
        )
        layout = go.Layout(
            title="Time-Frequency Representation",
            xaxis=dict(title="Time"),
            yaxis=dict(title="Frequency"),
        )
        fig = go.Figure(data=[trace], layout=layout)
        fig.show()

    def compare_original_tfr(self):
        """
        Compare the original signal and its Time-Frequency Representation.

        Example Usage:
        >>> signal = np.sin(np.linspace(0, 10, 1000))
        >>> tfr_viz = TFRVisualization(signal)
        >>> tfr_viz.compare_original_tfr()
        """
        tfr_result = self.tfr.compute_tfr()
        time_axis = np.arange(tfr_result.shape[1])
        freq_axis = np.arange(tfr_result.shape[0])

        trace1 = go.Scatter(y=self.signal, mode="lines", name="Original Signal")
        trace2 = go.Heatmap(
            z=np.abs(tfr_result),
            x=time_axis,
            y=freq_axis,
            colorscale="Viridis",
            name="TFR",
        )

        layout1 = go.Layout(
            title="Original Signal",
            xaxis=dict(title="Sample Index"),
            yaxis=dict(title="Amplitude"),
        )
        layout2 = go.Layout(
            title="Time-Frequency Representation",
            xaxis=dict(title="Time"),
            yaxis=dict(title="Frequency"),
        )

        fig1 = go.Figure(data=[trace1], layout=layout1)
        fig2 = go.Figure(data=[trace2], layout=layout2)

        fig1.show()
        fig2.show()


class ERPVisualization:
    """
    A class to visualize the results of Event-Related Potentials (ERP) using Plotly.

    Methods:
    - plot_erp: Plots the ERP of the signal.
    - compare_original_erp: Compares the original signal and its ERP.
    """

    def __init__(
        self,
        signal,
        stimulus_times,
        pre_stimulus=0.1,
        post_stimulus=0.4,
        sample_rate=1000,
    ):
        """
        Initialize the ERPVisualization class with the signal.

        Parameters:
        signal (numpy.ndarray): The input EEG signal.
        stimulus_times (numpy.ndarray): Times of the stimulus events.
        pre_stimulus (float): Time before the stimulus to include in the ERP (in seconds).
        post_stimulus (float): Time after the stimulus to include in the ERP (in seconds).
        sample_rate (int): The sample rate of the signal.
        """
        self.signal = signal
        self.erp = EventRelatedPotential(
            signal, stimulus_times, pre_stimulus, post_stimulus, sample_rate
        )

    def plot_erp(self):
        """
        Plot the Event-Related Potentials (ERP) of the signal.

        Example Usage:
        >>> signal = np.sin(np.linspace(0, 10, 1000))
        >>> stimulus_times = np.array([100, 300, 500])
        >>> erp_viz = ERPVisualization(signal, stimulus_times)
        >>> erp_viz.plot_erp()
        """
        erp_result = self.erp.compute_erp()
        time_axis = (
            np.arange(-self.erp.pre_stimulus, self.erp.post_stimulus)
            / self.erp.sample_rate
        )

        trace = go.Scatter(x=time_axis, y=erp_result, mode="lines", name="ERP")
        layout = go.Layout(
            title="Event-Related Potential",
            xaxis=dict(title="Time (s)"),
            yaxis=dict(title="Amplitude"),
        )
        fig = go.Figure(data=[trace], layout=layout)
        fig.show()

    def compare_original_erp(self):
        """
        Compare the original signal and its ERP.

        Example Usage:
        >>> signal = np.sin(np.linspace(0, 10, 1000))
        >>> stimulus_times = np.array([100, 300, 500])
        >>> erp_viz = ERPVisualization(signal, stimulus_times)
        >>> erp_viz.compare_original_erp()
        """
        erp_result = self.erp.compute_erp()
        time_axis = (
            np.arange(-self.erp.pre_stimulus, self.erp.post_stimulus)
            / self.erp.sample_rate
        )

        trace1 = go.Scatter(y=self.signal, mode="lines", name="Original Signal")
        trace2 = go.Scatter(x=time_axis, y=erp_result, mode="lines", name="ERP")

        layout1 = go.Layout(
            title="Original Signal",
            xaxis=dict(title="Sample Index"),
            yaxis=dict(title="Amplitude"),
        )
        layout2 = go.Layout(
            title="Event-Related Potential",
            xaxis=dict(title="Time (s)"),
            yaxis=dict(title="Amplitude"),
        )

        fig1 = go.Figure(data=[trace1], layout=layout1)
        fig2 = go.Figure(data=[trace2], layout=layout2)

        fig1.show()
        fig2.show()


class ChromaSTFTVisualization:
    """
    A class to visualize the results of the Chroma Short-Time Fourier Transform (Chroma STFT) using Plotly.

    Methods:
    - plot_chroma_stft: Plots the Chroma STFT of the signal.
    - compare_original_chroma_stft: Compares the original signal and its Chroma STFT.
    """

    def __init__(self, signal, sample_rate=16000, n_chroma=12, n_fft=2048):
        """
        Initialize the ChromaSTFTVisualization class with the signal.

        Parameters:
        signal (numpy.ndarray): The input signal to be analyzed.
        sample_rate (int): The sample rate of the signal.
        n_chroma (int): The number of chroma bins.
        n_fft (int): The FFT size.
        """
        self.signal = signal
        self.chroma_stft = ChromaSTFT(signal, sample_rate, n_chroma, n_fft)

    def plot_chroma_stft(self):
        """
        Plot the Chroma Short-Time Fourier Transform (Chroma STFT) of the signal.

        Example Usage:
        >>> signal = np.sin(np.linspace(0, 10, 1000))
        >>> chroma_stft_viz = ChromaSTFTVisualization(signal)
        >>> chroma_stft_viz.plot_chroma_stft()
        """
        chroma_stft_result = self.chroma_stft.compute_chroma_stft()

        # Time axis is the number of frames
        time_axis = np.arange(chroma_stft_result.shape[1])
        chroma_axis = np.arange(self.chroma_stft.n_chroma)

        trace = go.Heatmap(
            z=chroma_stft_result, x=time_axis, y=chroma_axis, colorscale="Viridis"
        )
        layout = go.Layout(
            title="Chroma STFT",
            xaxis=dict(title="Time"),
            yaxis=dict(title="Chroma Bins"),
        )
        fig = go.Figure(data=[trace], layout=layout)
        fig.show()


    def compare_original_chroma_stft(self):
        """
        Compare the original signal and its Chroma STFT.

        Example Usage:
        >>> signal = np.sin(np.linspace(0, 10, 1000))
        >>> chroma_stft_viz = ChromaSTFTVisualization(signal)
        >>> chroma_stft_viz.compare_original_chroma_stft()
        """
        chroma_stft_result = self.chroma_stft.compute_chroma_stft()
        time_axis = np.arange(chroma_stft_result.shape[1])
        chroma_axis = np.arange(chroma_stft_result.shape[0])

        trace1 = go.Scatter(y=self.signal, mode="lines", name="Original Signal")
        trace2 = go.Heatmap(
            z=chroma_stft_result,
            x=time_axis,
            y=chroma_axis,
            colorscale="Viridis",
            name="Chroma STFT",
        )

        layout1 = go.Layout(
            title="Original Signal",
            xaxis=dict(title="Sample Index"),
            yaxis=dict(title="Amplitude"),
        )
        layout2 = go.Layout(
            title="Chroma STFT",
            xaxis=dict(title="Time"),
            yaxis=dict(title="Chroma Bins"),
        )

        fig1 = go.Figure(data=[trace1], layout=layout1)
        fig2 = go.Figure(data=[trace2], layout=layout2)

        fig1.show()
        fig2.show()


class MFCCVisualization:
    """
    A class to visualize the results of the Mel-Frequency Cepstral Coefficients (MFCC) using Plotly.

    Methods:
    - plot_mfcc: Plots the MFCC of the signal.
    - compare_original_mfcc: Compares the original signal and its MFCC.
    """

    def __init__(self, signal, sample_rate=16000, num_filters=40, num_coefficients=13):
        """
        Initialize the MFCCVisualization class with the signal.

        Parameters:
        signal (numpy.ndarray): The input signal to be analyzed.
        sample_rate (int): The sample rate of the signal.
        num_filters (int): The number of Mel filters.
        num_coefficients (int): The number of MFCC coefficients.
        """
        self.signal = signal
        self.mfcc = MFCC(signal, sample_rate, num_filters, num_coefficients)

    def plot_mfcc(self):
        """
        Plot the Mel-Frequency Cepstral Coefficients (MFCC) of the signal.

        Example Usage:
        >>> signal = np.sin(np.linspace(0, 10, 1000))
        >>> mfcc_viz = MFCCVisualization(signal)
        >>> mfcc_viz.plot_mfcc()
        """
        mfcc_result = self.mfcc.compute_mfcc()
        time_axis = np.arange(mfcc_result.shape[0])
        coefficient_axis = np.arange(mfcc_result.shape[1])

        trace = go.Heatmap(
            z=mfcc_result.T, x=time_axis, y=coefficient_axis, colorscale="Viridis"
        )
        layout = go.Layout(
            title="MFCC", xaxis=dict(title="Time"), yaxis=dict(title="MFCC Coefficient")
        )
        fig = go.Figure(data=[trace], layout=layout)
        fig.show()

    def compare_original_mfcc(self):
        """
        Compare the original signal and its MFCC.

        Example Usage:
        >>> signal = np.sin(np.linspace(0, 10, 1000))
        >>> mfcc_viz = MFCCVisualization(signal)
        >>> mfcc_viz.compare_original_mfcc()
        """
        mfcc_result = self.mfcc.compute_mfcc()
        time_axis = np.arange(mfcc_result.shape[0])
        coefficient_axis = np.arange(mfcc_result.shape[1])

        trace1 = go.Scatter(y=self.signal, mode="lines", name="Original Signal")
        trace2 = go.Heatmap(
            z=mfcc_result.T,
            x=time_axis,
            y=coefficient_axis,
            colorscale="Viridis",
            name="MFCC",
        )

        layout1 = go.Layout(
            title="Original Signal",
            xaxis=dict(title="Sample Index"),
            yaxis=dict(title="Amplitude"),
        )
        layout2 = go.Layout(
            title="MFCC", xaxis=dict(title="Time"), yaxis=dict(title="MFCC Coefficient")
        )

        fig1 = go.Figure(data=[trace1], layout=layout1)
        fig2 = go.Figure(data=[trace2], layout=layout2)

        fig1.show()
        fig2.show()


class STFTVisualization:
    """
    A class to visualize the results of the Short-Time Fourier Transform (STFT) using Plotly.

    Methods:
    - plot_stft: Plots the STFT of the signal.
    - compare_original_stft: Compares the original signal and its STFT.
    """

    def __init__(self, signal, window_size=256, hop_size=128):
        """
        Initialize the STFTVisualization class with the signal.

        Parameters:
        signal (numpy.ndarray): The input signal to be analyzed.
        window_size (int): The size of the window for STFT.
        hop_size (int): The hop size (step size) for STFT.
        """
        self.signal = signal
        self.stft = STFT(signal, window_size, hop_size)

    def plot_stft(self):
        """
        Plot the Short-Time Fourier Transform (STFT) of the signal.

        Example Usage:
        >>> signal = np.sin(np.linspace(0, 10, 1000))
        >>> stft_viz = STFTVisualization(signal)
        >>> stft_viz.plot_stft()
        """
        stft_result = self.stft.compute_stft()
        time_axis = np.arange(stft_result.shape[1])
        freq_axis = np.arange(stft_result.shape[0])

        trace = go.Heatmap(z=np.abs(stft_result), x=time_axis, y=freq_axis)
        layout = go.Layout(
            title="STFT Magnitude",
            xaxis=dict(title="Time"),
            yaxis=dict(title="Frequency"),
        )
        fig = go.Figure(data=[trace], layout=layout)
        fig.show()

    def compare_original_stft(self):
        """
        Compare the original signal and its STFT.

        Example Usage:
        >>> signal = np.sin(np.linspace(0, 10, 1000))
        >>> stft_viz = STFTVisualization(signal)
        >>> stft_viz.compare_original_stft()
        """
        stft_result = self.stft.compute_stft()
        time_axis = np.arange(stft_result.shape[1])
        freq_axis = np.arange(stft_result.shape[0])

        trace1 = go.Scatter(y=self.signal, mode="lines", name="Original Signal")
        trace2 = go.Heatmap(
            z=np.abs(stft_result),
            x=time_axis,
            y=freq_axis,
            colorscale="Viridis",
            name="STFT Magnitude",
        )

        layout1 = go.Layout(
            title="Original Signal",
            xaxis=dict(title="Sample Index"),
            yaxis=dict(title="Amplitude"),
        )
        layout2 = go.Layout(
            title="STFT Magnitude",
            xaxis=dict(title="Time"),
            yaxis=dict(title="Frequency"),
        )

        fig1 = go.Figure(data=[trace1], layout=layout1)
        fig2 = go.Figure(data=[trace2], layout=layout2)

        fig1.show()
        fig2.show()
