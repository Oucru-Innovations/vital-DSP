
# vitalDSP - Healthcare Digital Signal Processing Toolkit

A comprehensive Python toolkit for Digital Signal Processing (DSP) in healthcare applications, designed to process physiological signals with advanced signal processing techniques and machine learning algorithms.

[![DOI](https://zenodo.org/badge/845712551.svg)](https://doi.org/10.5281/zenodo.14613751)
![GitHub stars](https://img.shields.io/github/stars/Oucru-Innovations/vital-DSP?style=social)
![Build Status](https://github.com/Oucru-Innovations/vital-DSP/actions/workflows/ci.yml/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/Oucru-Innovations/vital-DSP/badge.svg)](https://coveralls.io/github/Oucru-Innovations/vital-DSP)
<!-- [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) -->
![GitHub License](https://img.shields.io/github/license/Oucru-Innovations/vital-DSP)
![Python Versions](https://img.shields.io/badge/python-3.9%2B-blue)
[![Documentation Status](https://readthedocs.org/projects/vital-dsp/badge/?version=latest)](https://vital-dsp.readthedocs.io/en/latest/?badge=latest)
![PyPI Downloads](https://img.shields.io/pypi/dm/vitalDSP)
[![PyPI version](https://badge.fury.io/py/vitalDSP.svg)](https://badge.fury.io/py/vitalDSP)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Oucru-Innovations/vital-DSP/blob/main/docs/source/notebooks/synthesize_data.ipynb)
[![Deploy on Render](https://img.shields.io/badge/Deploy%20on-Render-46E3B7?style=flat&logo=render)](https://vital-dsp-1.onrender.com/)

<!-- [![Coverage Status](https://coveralls.io/repos/github/Oucru-Innovations/vital-DSP/badge.svg?branch=main)](https://coveralls.io/github/Oucru-Innovations/vital-DSP?branch=main) -->
<!-- [![codecov](https://codecov.io/gh/Oucru-Innovations/vital-DSP/branch/main/graph/badge.svg)](https://codecov.io/gh/Oucru-Innovations/vital-DSP) -->


## Overview

vitalDSP is a powerful Python library that provides researchers, clinicians, and developers with state-of-the-art tools for processing physiological signals in healthcare applications. The toolkit combines traditional digital signal processing methods with modern machine learning and deep learning techniques to deliver robust, accurate, and efficient signal analysis capabilities.

### Key Capabilities

- **Multi-Signal Support**: Process electrocardiogram (ECG), photoplethysmography (PPG), electroencephalogram (EEG), and respiratory signals
- **Advanced Signal Processing**: Implement traditional filters, transforms, and time-domain analysis methods
- **Machine Learning Integration**: Leverage ML-inspired filters, feature extraction, and pattern recognition
- **Real-time Processing**: Support for real-time signal monitoring and analysis
- **Quality Assessment**: Comprehensive signal quality evaluation and artifact detection
- **Healthcare Applications**: Specialized tools for clinical monitoring, anomaly detection, and health assessment

## Web Application

Experience vitalDSP directly in your browser through our interactive web interface. No installation required - simply visit the application and start exploring the toolkit's capabilities.

**[Launch Web Application](https://vital-dsp-1.onrender.com/)**

### Important Notes

This web application is hosted on Render's free tier with the following limitations:
- Cold start delay of a few seconds when the application spins up
- Automatic shutdown after 15 minutes of inactivity to conserve resources
- Recommended for testing and demonstration purposes only
- For production use or large datasets, please install the toolkit locally

### Sample Data

The repository includes optimized sample datasets for web application testing:
- **ECG Data**: `sample_data/ECG/` - Sample electrocardiogram signals
- **PPG Data**: `sample_data/PPG/` - Sample photoplethysmography signals

These files are specifically sized for the web application's free tier limitations.

## Core Features

### Signal Processing Modules

**Filtering and Preprocessing**
- Traditional filters: Moving average, Gaussian, Butterworth, and Chebyshev filters
- Advanced ML-inspired filters for noise reduction and signal enhancement
- Adaptive filtering techniques for dynamic signal conditions
- Preprocessing utilities for signal normalization and baseline correction

**Transform Methods**
- Fourier Transform (FFT) for frequency domain analysis
- Discrete Cosine Transform (DCT) for signal compression
- Wavelet Transform for time-frequency analysis
- Signal fusion methods for multi-sensor data integration

**Time-Domain Analysis**
- Advanced peak detection algorithms for ECG and PPG signals
- Envelope detection and signal demodulation
- Zero Crossing Rate (ZCR) calculation
- Advanced segmentation techniques for signal partitioning

**Advanced Processing Methods**
- Empirical Mode Decomposition (EMD) for non-linear signal analysis
- Sparse signal processing for compressed sensing applications
- Bayesian optimization for parameter tuning
- Machine learning-based feature extraction

### Specialized Healthcare Modules

**Neuro-Signal Processing** *(Coming Soon)*
- EEG band power analysis for brain activity monitoring
- Event-Related Potential (ERP) detection
- Cognitive load measurement algorithms

**Respiratory Analysis**
- Automated respiratory rate calculation from various signal types
- Sleep apnea detection and classification
- Multi-sensor fusion for comprehensive respiratory monitoring
- Breathing pattern analysis and anomaly detection

**Signal Quality Assessment**
- Signal-to-Noise Ratio (SNR) calculation and monitoring
- Comprehensive artifact detection and removal algorithms
- Adaptive quality assessment methods
- Real-time signal quality metrics

**Monitoring and Alert Systems**
- Real-time anomaly detection for critical health parameters
- Multi-parameter monitoring with correlation analysis
- Intelligent alert systems with customizable thresholds
- Historical trend analysis and reporting


## Installation

vitalDSP can be installed using two different methods depending on your needs:

### Method 1: Install from PyPI (Recommended)

For the most stable and tested version, install directly from the Python Package Index:

```bash
pip install vitalDSP
```

This method provides:
- Stable, tested releases
- Automatic dependency management
- Easy updates with `pip install --upgrade vitalDSP`

### Method 2: Install from Source

For the latest development features and contributions, install directly from the GitHub repository:

**Step 1: Clone the Repository**
```bash
git clone https://github.com/Oucru-Innovations/vital-DSP.git
```

**Step 2: Navigate to the Project Directory**
```bash
cd vital-DSP
```

**Step 3: Install the Package**
```bash
python setup.py install
```

Or for development installation:
```bash
pip install -e .
```

### System Requirements

- Python 3.9 or higher
- NumPy >= 1.19.2
- SciPy >= 1.5.2
- Scikit-learn >= 0.23.2
- Plotly >= 4.14.3
- Dash >= 2.0.0 (for web applications)
- FastAPI >= 0.68.0 (for API endpoints)

## Healthcare Applications

vitalDSP is designed to address real-world healthcare challenges across multiple domains:

### Clinical Monitoring
- **Remote Patient Monitoring**: Continuous analysis of ECG and PPG signals for real-time health insights
- **Vital Signs Tracking**: Automated extraction of heart rate, respiratory rate, and other critical parameters
- **Chronic Disease Management**: Long-term monitoring for patients with cardiovascular and respiratory conditions

### Diagnostic Applications
- **Stress and Anxiety Assessment**: Heart rate variability analysis for psychological state evaluation
- **Sleep Disorder Detection**: Comprehensive sleep apnea detection using respiratory signal analysis
- **Cardiovascular Screening**: Early detection of arrhythmias and other cardiac abnormalities

### Research and Development
- **Clinical Trials**: Standardized signal processing for multi-site research studies
- **Biomarker Discovery**: Advanced feature extraction for identifying new health indicators
- **Algorithm Development**: Robust platform for testing and validating new signal processing methods

### Telemedicine and Digital Health
- **Wearable Device Integration**: Process data from smartwatches, fitness trackers, and medical devices
- **Mobile Health Applications**: Enable signal processing capabilities in smartphone-based health apps
- **Telehealth Platforms**: Provide backend processing for remote healthcare consultations

## Quick Start

### Basic Usage Example

```python
import vitalDSP as vdsp
import numpy as np
from vitalDSP.utils.peak_detection import PeakDetection
from vitalDSP.physiological_features.waveform import WaveformMorphology

# Load sample ECG data (default sampling rate: 128 Hz)
ecg_signal = np.load('sample_data/ECG_short.csv')

# Apply Butterworth filter
filtered_signal = vdsp.filtering.butterworth_filter(
    ecg_signal, 
    lowcut=0.5, 
    highcut=4, 
    fs=128, 
    order=4
)

# Create waveform morphology object for additional features
waveform = WaveformMorphology(filtered_signal, fs=128, signal_type="ECG")
```

### Sample Data Information

The repository includes sample datasets with the following default sampling rates:
- **ECG Data**: 128 Hz sampling rate (`sample_data/ECG_short.csv`)
- **PPG Data**: 100 Hz sampling rate (`sample_data/PPG_short.csv`)

When working with these sample files, make sure to use the correct sampling rate in your analysis functions.

### Interactive Web Application

**Try vitalDSP without installation!** Our web application serves as an interactive toolbox where you can:

- Upload your own physiological signals (ECG, PPG, respiratory)
- Apply various signal processing techniques through an intuitive interface
- Visualize results with interactive plots
- Experiment with different parameters and algorithms
- Generate health reports and analysis summaries

**[Launch Web Application](https://vital-dsp-1.onrender.com/)** - Perfect for exploring vitalDSP capabilities before local installation.

### Advanced Usage

For comprehensive examples and detailed API documentation, please visit our [official documentation](https://vital-DSP.readthedocs.io/en/latest/?badge=latest).

## Interactive Examples

Explore vitalDSP capabilities through our interactive Jupyter notebooks on Google Colab:

### Core Functionality
- **[Synthetic Data & Peak Detection](https://colab.research.google.com/github/Oucru-Innovations/vital-DSP/blob/main/docs/source/notebooks/synthesize_data.ipynb)**: Learn peak detection algorithms with synthetic ECG and PPG signals
- **[Signal Quality Assessment](https://colab.research.google.com/github/Oucru-Innovations/vital-DSP/blob/main/docs/source/notebooks/signal_quality_index.ipynb)**: Comprehensive signal quality analysis, SNR calculation, and artifact detection

### Healthcare Applications
- **[Health Report Analysis](https://colab.research.google.com/github/Oucru-Innovations/vital-DSP/blob/main/docs/source/notebooks/health_report_analysis.ipynb)**: Generate comprehensive health reports from ECG and PPG data
- **[Respiratory Analysis](https://colab.research.google.com/github/Oucru-Innovations/vital-DSP/blob/main/docs/source/notebooks/respiratory_analysis.ipynb)**: Advanced respiratory rate calculation and sleep apnea detection

## Documentation

Comprehensive documentation is available through multiple channels:

- **[Online Documentation](https://vital-DSP.readthedocs.io/en/latest/)**: Complete API reference, tutorials, and examples
- **Local Documentation**: Build documentation locally using the files in the `docs/` directory
- **Jupyter Notebooks**: Interactive examples in `docs/source/notebooks/`
- **Code Examples**: Sample implementations in the `sample_data/analysis/` directory

## Contributing

We welcome contributions from the community! Whether you're fixing bugs, adding features, or improving documentation, your contributions help make vitalDSP better for everyone.

### How to Contribute

1. **Fork the Repository**: Create your own fork of the vitalDSP repository
2. **Create a Feature Branch**: Work on your changes in a dedicated branch
3. **Follow Guidelines**: Read our [CONTRIBUTING](CONTRIBUTING.md) guidelines for coding standards and best practices
4. **Test Your Changes**: Ensure all tests pass and add new tests for new functionality
5. **Submit a Pull Request**: Create a detailed pull request describing your changes

### Areas for Contribution

- **Algorithm Improvements**: Enhance existing signal processing algorithms
- **New Features**: Add support for new physiological signals or analysis methods
- **Documentation**: Improve tutorials, examples, and API documentation
- **Testing**: Expand test coverage and add performance benchmarks
- **Web Interface**: Enhance the web application with new features

## Community and Support

### Get Help

- **GitHub Discussions**: [Join the conversation](https://github.com/Oucru-Innovations/vital-DSP/discussions) for questions, ideas, and general discussion
- **Issue Tracker**: [Report bugs](https://github.com/Oucru-Innovations/vital-DSP/issues) or request new features
- **Documentation**: Check our comprehensive documentation for detailed guides and examples

### Stay Connected

- **Star the Repository**: Show your support by starring vitalDSP on GitHub
- **Watch for Updates**: Stay informed about new releases and features
- **Share Your Work**: Let us know how you're using vitalDSP in your projects

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The MIT License allows you to:
- Use vitalDSP commercially
- Modify and distribute the code
- Include it in proprietary software
- Use it in both open source and closed source projects

## Citation

If you use vitalDSP in your research or projects, please cite our work:

```bibtex
@software{vitaldsp2024,
  title={vitalDSP: Healthcare Digital Signal Processing Toolkit},
  author={van-koha and Oucru-Innovations},
  year={2024},
  url={https://github.com/Oucru-Innovations/vital-DSP},
  doi={10.5281/zenodo.14613751}
}
```
