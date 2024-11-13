
# Healthcare DSP Toolkit

A comprehensive toolkit for Digital Signal Processing (DSP) in healthcare applications

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

<!-- [![Coverage Status](https://coveralls.io/repos/github/Oucru-Innovations/vital-DSP/badge.svg?branch=main)](https://coveralls.io/github/Oucru-Innovations/vital-DSP?branch=main) -->
<!-- [![codecov](https://codecov.io/gh/Oucru-Innovations/vital-DSP/branch/main/graph/badge.svg)](https://codecov.io/gh/Oucru-Innovations/vital-DSP) -->


This repository contains a comprehensive toolkit for Digital Signal Processing (DSP) in healthcare applications. It includes traditional DSP methods as well as advanced machine learning (ML) and deep learning (DL) inspired techniques. The toolkit is designed to process a wide range of physiological signals, such as ECG, EEG, PPG, and respiratory signals, with applications in monitoring, anomaly detection, and signal quality assessment.

## Features
- **Filtering**: Traditional filters (e.g., moving average, Gaussian, Butterworth) and advanced ML-inspired filters.
- **Transforms**: Fourier Transform, DCT, Wavelet Transform, and various fusion methods.
- **Time-Domain Analysis**: Peak detection, envelope detection, ZCR, and advanced segmentation techniques.
- **Advanced Methods**: EMD, sparse signal processing, Bayesian optimization, and more.
- **Neuro-Signal Processing**: EEG band power analysis, ERP detection, cognitive load measurement. (To be implemented)
- **Respiratory Analysis**: Automated respiratory rate calculation, sleep apnea detection, and multi-sensor fusion.
- **Signal Quality Assessment**: SNR calculation, artifact detection/removal, and adaptive methods.
- **Monitoring and Alert Systems**: Real-time anomaly detection, multi-parameter monitoring, and alert correlation.


## Installation
You can install vitalDSP in two different ways:
### Option 1: Install via pip
If you want the simplest installation method, you can install the latest version of vitalDSP directly from PyPI using pip:
```bash
pip install vitalDSP
```

### Option 2: Install from the GitHub Repository
For those who prefer to have the latest version, including any recent updates that may not yet be available on PyPI, you can clone the repository and install it manually.
**Step 1:** Clone the Repository
First, clone the vitalDSP repository from GitHub to your local machine:
```bash
git clone https://github.com/Oucru-Innovations/vital-DSP.gi
```
**Step 2:** Navigate to the Project Directory
Navigate to the directory where the repository was cloned:

```bash
cd vital-DSP
```

**Step 3:** Install with setup.py
You can now install vitalDSP using the setup.py script:

```bash
python setup.py install
```
This method ensures that you are using the most up-to-date codebase from the repository.

## Applications in Healthcare
vitalDSP can be applied across various healthcare use cases:
- **Remote Patient Monitoring**: Analyze ECG and PPG signals for real-time insights into patient health.
- **Stress and Anxiety Detection**: Monitor heart rate variability to assess stress levels.
- **Sleep Apnea Detection**: Use respiratory signals to identify breathing irregularities during sleep.

## Usage
Please read the [instruction in the documentation](https://vital-DSP.readthedocs.io/en/latest/?badge=latest) for detailed usage examples and module descriptions.

## Example Notebooks on Google Colab
- **[Synthetic Data & Peak Detection](https://colab.research.google.com/github/Oucru-Innovations/vital-DSP/blob/main/docs/source/notebooks/synthesize_data.ipynb)**: Peak Detection, Synthetic Signal (ECG, PPG).
- **[Health Report Analysis](https://colab.research.google.com/github/Oucru-Innovations/vital-DSP/blob/main/docs/source/notebooks/health_report_analysis.ipynb)**: Generate Health Report from ECG and PPG data.
- **[Signal Quality Indices](https://colab.research.google.com/github/Oucru-Innovations/vital-DSP/blob/main/docs/source/notebooks/signal_quality_index.ipynb)**: Signal Quality Assurance, SNR, Artifact Detection.

## Documentation
Comprehensive documentation for each module is available in the `docs/` directory, covering usage examples, API references, and more.

## Contributing
We welcome contributions! Please read the [CONTRIBUTING](CONTRIBUTING.md) file for guidelines on how to contribute to this project.

## Community and Support
Join our community to share ideas, ask questions, and get support:
- **GitHub Discussions**: [Join the conversation](https://github.com/Oucru-Innovations/vital-DSP/discussions)
- **Report Issues**: Found a bug? [Open an issue](https://github.com/Oucru-Innovations/vital-DSP/issues)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.