# Frequency Domain Analysis and Advanced Filtering Implementation

## Overview

This document describes the comprehensive implementation of frequency domain analysis and advanced signal filtering features for the vitalDSP webapp. The implementation follows the same data handling pattern as the time domain analysis, utilizing the stored data after upload and demonstrating all available vitalDSP features related to these two subjects.

## Features Implemented

### 1. Frequency Domain Analysis (`/frequency` route)

#### Core Analysis Types
- **FFT (Fast Fourier Transform)**: Standard frequency domain analysis with configurable parameters
- **STFT (Short-Time Fourier Transform)**: Time-frequency analysis for non-stationary signals
- **Wavelet Transform**: Multi-resolution analysis using different wavelet families
- **Power Spectral Density**: Energy distribution across frequency bands
- **Spectrogram**: Visual time-frequency representation

#### FFT Parameters
- **Window Types**: Hamming, Hanning, Blackman, Rectangular, Kaiser
- **N FFT Points**: 256, 512, 1024, 2048, 4096 (configurable)
- **Frequency Range**: User-defined min/max frequency limits

#### STFT Parameters
- **Window Size**: 64, 128, 256, 512 samples (configurable)
- **Hop Size**: 32, 64, 128, 256 samples (configurable)
- **Overlap**: Automatically calculated for optimal time-frequency resolution

#### Wavelet Parameters
- **Wavelet Types**: Haar, Daubechies 4/8, Symlets 4, Coiflets 4
- **Decomposition Levels**: 1-8 levels (configurable)
- **Scalogram Visualization**: Time-frequency representation

#### Analysis Options
- **Peak Frequency Detection**: Identifies dominant frequency components
- **Dominant Frequency**: Calculates weighted average frequency
- **Band Power Analysis**: Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz)
- **Frequency Stability**: Measures frequency variation over time
- **Harmonic Analysis**: Identifies fundamental frequency and harmonics

### 2. Advanced Signal Filtering (`/filtering` route)

#### Filter Categories
- **Traditional Filters**: Classical IIR filter designs
- **Advanced Filters**: Modern signal processing techniques
- **Artifact Removal**: Specialized biomedical signal cleaning
- **Neural Network Filtering**: AI-powered signal enhancement
- **Ensemble Filtering**: Multi-filter combination methods

#### Traditional Filter Parameters
- **Filter Families**: Butterworth, Chebyshev I/II, Elliptic, Bessel
- **Response Types**: Bandpass, Bandstop (Notch), Lowpass, Highpass
- **Frequency Cutoffs**: User-defined low and high frequency limits
- **Filter Order**: 2-10 (configurable)

#### Advanced Filter Methods
- **Kalman Filter**: Real-time signal estimation and noise reduction
- **Optimization-Based**: Custom loss function optimization
- **Gradient Descent**: Adaptive filtering with learning
- **Convolution-Based**: Kernel-based signal processing
- **Attention-Based**: Dynamic weighting mechanisms

#### Artifact Removal
- **Baseline Wander**: Low-frequency drift correction
- **Motion Artifacts**: Movement-related noise removal
- **Power Line Interference**: 50/60 Hz noise elimination
- **Muscle Noise**: EMG artifact reduction
- **Electrode Noise**: Contact artifact cleaning

#### Neural Network Filtering
- **Network Types**: Autoencoder, CNN, LSTM, Transformer
- **Model Complexity**: 1-5 levels (configurable)
- **Real-time Processing**: Optimized for live signal streams

#### Ensemble Filtering
- **Combination Methods**: Mean, Weighted Mean, Bagging, Boosting
- **Number of Filters**: 2-10 parallel filters (configurable)
- **Adaptive Weighting**: Dynamic filter selection

#### Quality Assessment
- **SNR Improvement**: Signal-to-noise ratio enhancement measurement
- **Artifact Reduction**: Quantitative artifact removal assessment
- **Signal Distortion**: Mean square error calculation
- **Computational Cost**: Performance and resource usage metrics

## Data Handling

### Integration with Existing System
- **Data Source**: Uses the same data service as time domain analysis
- **Column Mapping**: Automatically detects signal columns from uploaded data
- **Sampling Frequency**: Extracted from data metadata
- **Time Windows**: Configurable start/end times with slider controls
- **Data Persistence**: Stores analysis results in Dash stores for cross-page access

### Data Flow
1. **Upload**: Data uploaded and processed through existing upload system
2. **Storage**: Data stored in data service with metadata
3. **Retrieval**: Analysis pages retrieve latest uploaded data
4. **Processing**: vitalDSP algorithms applied to selected time windows
5. **Visualization**: Results displayed in interactive Plotly charts
6. **Export**: Analysis results can be exported for further processing

## Technical Implementation

### Architecture
- **Layout Components**: Comprehensive Dash Bootstrap layouts with dynamic parameter sections
- **Callback System**: Unified callbacks handling both frequency and filtering analysis
- **Parameter Toggling**: Dynamic visibility of parameter sections based on selected methods
- **Error Handling**: Comprehensive error handling with user-friendly messages
- **Performance**: Optimized for real-time analysis of large datasets

### Dependencies
- **Core**: Dash, Plotly, NumPy, Pandas, SciPy
- **Signal Processing**: vitalDSP package for advanced algorithms
- **Visualization**: Interactive charts with zoom, pan, and export capabilities
- **UI Framework**: Bootstrap components for responsive design

### Callback Structure
- **Frequency Analysis**: Single callback handling all frequency domain operations
- **Advanced Filtering**: Unified callback for all filtering methods
- **Parameter Management**: Dynamic parameter section visibility
- **Time Window Control**: Synchronized slider and input field updates
- **Data Storage**: Persistent storage of analysis results

## User Experience Features

### Interactive Controls
- **Time Navigation**: Quick nudge buttons (-10s, -1s, +1s, +10s)
- **Range Slider**: Visual time window selection
- **Parameter Presets**: Common analysis configurations
- **Real-time Updates**: Immediate visualization of parameter changes

### Visualization
- **Multi-panel Layout**: Organized display of different analysis aspects
- **Interactive Charts**: Zoom, pan, hover, and selection capabilities
- **Color-coded Results**: Consistent color schemes for different signal types
- **Export Options**: PNG, SVG, and data export capabilities

### Responsive Design
- **Mobile-friendly**: Bootstrap responsive grid system
- **Adaptive Layouts**: Parameter sections show/hide based on selections
- **Consistent Styling**: Unified design language across all pages

## vitalDSP Integration

### Available Algorithms
The implementation utilizes the full range of vitalDSP capabilities:

#### Frequency Domain
- `vitalDSP.transforms.fourier_transform.FourierTransform`
- `vitalDSP.transforms.stft.STFT`
- `vitalDSP.transforms.wavelet_transform.WaveletTransform`
- `vitalDSP.transforms.time_freq_representation`

#### Signal Filtering
- `vitalDSP.filtering.signal_filtering.SignalFiltering`
- `vitalDSP.filtering.advanced_signal_filtering.AdvancedSignalFiltering`
- `vitalDSP.filtering.artifact_removal.ArtifactRemoval`

#### Advanced Computation
- `vitalDSP.advanced_computation.kalman_filter`
- `vitalDSP.advanced_computation.neural_network_filtering`
- `vitalDSP.advanced_computation.ensemble_filtering`

### Feature Demonstration
Each analysis type demonstrates multiple vitalDSP features:
- **Multiple filter families** for different signal characteristics
- **Adaptive algorithms** for varying noise conditions
- **Multi-resolution analysis** for different time scales
- **Quality assessment** for algorithm performance evaluation

## Usage Instructions

### Getting Started
1. **Upload Data**: Use the upload page to upload PPG/ECG data
2. **Configure Parameters**: Set sampling frequency and column mapping
3. **Navigate to Analysis**: Choose frequency or filtering page
4. **Select Method**: Choose analysis type or filter method
5. **Adjust Parameters**: Fine-tune analysis parameters
6. **View Results**: Interactive visualization of results
7. **Export**: Save results for further analysis

### Best Practices
- **Start with FFT**: Use FFT for initial frequency analysis
- **Adjust Windows**: Optimize STFT parameters for your signal
- **Try Different Wavelets**: Different wavelets work better for different signals
- **Compare Filters**: Use ensemble methods for optimal results
- **Monitor Quality**: Check quality metrics to ensure good filtering

## Future Enhancements

### Planned Features
- **Real-time Streaming**: Live signal analysis capabilities
- **Batch Processing**: Multiple file analysis
- **Custom Algorithms**: User-defined signal processing methods
- **Machine Learning**: Automated parameter optimization
- **Cloud Integration**: Remote processing capabilities

### Performance Improvements
- **GPU Acceleration**: CUDA/OpenCL support for large datasets
- **Parallel Processing**: Multi-core analysis capabilities
- **Memory Optimization**: Efficient handling of large signals
- **Caching**: Smart result caching for repeated analyses

## Conclusion

This implementation provides a comprehensive, production-ready solution for frequency domain analysis and advanced signal filtering in the vitalDSP webapp. It demonstrates all available vitalDSP features while maintaining consistency with the existing time domain analysis system. The modular architecture allows for easy extension and enhancement of capabilities.

The implementation follows best practices for web application development, provides an excellent user experience, and showcases the full power of the vitalDSP library for biomedical signal processing applications.
