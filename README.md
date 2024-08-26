
# Healthcare DSP Toolkit

![GitHub stars](https://img.shields.io/github/stars/Oucru-Innovations/vital-DSP?style=social)
![Build Status](https://github.com/Oucru-Innovations/vital-DSP/actions/workflows/ci.yml/badge.svg)
<!-- [![Coverage Status](https://coveralls.io/repos/github/Oucru-Innovations/vital-DSP/badge.svg?branch=main)](https://coveralls.io/github/Oucru-Innovations/vital-DSP?branch=main) -->
[![codecov](https://codecov.io/gh/Oucru-Innovations/vital-DSP/branch/main/graph/badge.svg)](https://codecov.io/gh/Oucru-Innovations/vital-DSP)
<!-- ![GitHub License](https://img.shields.io/github/license/Oucru-Innovations/vital-DSP) -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Python Versions](https://img.shields.io/badge/python-3.9%2B-blue)
[![Documentation Status](https://readthedocs.org/projects/vital-DSP/badge/?version=latest)](https://vital-DSP.readthedocs.io/en/latest/?badge=latest)
![PyPI Downloads](https://img.shields.io/pypi/dm/vitalDSP)
[![PyPI version](https://badge.fury.io/py/vitalDSP.svg)](https://badge.fury.io/py/vitalDSP)

This repository contains a comprehensive toolkit for Digital Signal Processing (DSP) in healthcare applications. It includes traditional DSP methods as well as advanced machine learning (ML) and deep learning (DL) inspired techniques. The toolkit is designed to process a wide range of physiological signals, such as ECG, EEG, PPG, and respiratory signals, with applications in monitoring, anomaly detection, and signal quality assessment.

## Features
- **Filtering**: Traditional filters (e.g., moving average, Gaussian, Butterworth) and advanced ML-inspired filters.
- **Transforms**: Fourier Transform, DCT, Wavelet Transform, and various fusion methods.
- **Time-Domain Analysis**: Peak detection, envelope detection, ZCR, and advanced segmentation techniques.
- **Advanced Methods**: EMD, sparse signal processing, Bayesian optimization, and more.
- **Neuro-Signal Processing**: EEG band power analysis, ERP detection, cognitive load measurement.
- **Respiratory Analysis**: Automated respiratory rate calculation, sleep apnea detection, and multi-sensor fusion.
- **Signal Quality Assessment**: SNR calculation, artifact detection/removal, and adaptive methods.
- **Monitoring and Alert Systems**: Real-time anomaly detection, multi-parameter monitoring, and alert correlation.


## Installation
To install the toolkit, run the following command:
```bash
pip install -r requirements.txt
```

## Usage
Detailed documentation for each module is available in the `docs/` directory. Below is a simple example of using the filtering module:
```python
from healthcare_dsp.filtering.moving_average import moving_average_filter

# Example signal
signal = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Apply moving average filter
filtered_signal = moving_average_filter(signal, window_size=3)
print(filtered_signal)
```

## Documentation
Comprehensive documentation for each module is available in the `docs/` directory, covering usage examples, API references, and more.

## Contributing
We welcome contributions! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute to this project.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.