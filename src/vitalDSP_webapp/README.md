# vitalDSP Comprehensive Web Application

A comprehensive web-based dashboard for Digital Signal Processing (DSP) analysis of vital signs, built with Dash and FastAPI. This application provides access to all vitalDSP features through an intuitive web interface.

## Features

### 🚀 Core Functionality
- **Data Upload & Management**: Support for CSV, TXT files with automatic column mapping
- **Sample Data**: Built-in PPG data generation for quick testing
- **Real-time Processing**: Interactive signal processing with immediate feedback

### 📊 Analysis Modules

#### Time Domain Analysis
- Raw signal visualization
- Filtered signal display
- Peak detection and annotation
- Signal statistics and metrics
- Time window controls

#### Frequency Domain Analysis
- Power Spectral Density (PSD) plots
- Spectrogram generation
- FFT analysis with configurable parameters
- Frequency domain feature extraction

#### Signal Filtering
- Multiple filter types (Butterworth, Chebyshev, Elliptic, Bessel)
- Lowpass, highpass, bandpass, bandstop configurations
- Real-time filter response visualization
- Advanced filtering options (notch, adaptive, wavelet, Kalman)

#### Physiological Features
- Heart rate analysis and trending
- HRV (Heart Rate Variability) analysis
- RR interval calculations
- Poincaré plots
- Nonlinear feature extraction

#### Respiratory Analysis
- Respiratory rate estimation
- Breathing pattern analysis
- Sleep apnea detection
- Multiple estimation methods (peak detection, FFT, time domain)

#### Feature Engineering
- Morphological feature extraction
- Statistical feature computation
- Cross-signal analysis
- Feature export capabilities

#### Signal Transforms
- Fourier Transform (FFT)
- Wavelet Transform
- Hilbert Transform
- Short-time Fourier Transform (STFT)
- Discrete Cosine Transform (DCT)
- MFCC extraction

#### Quality Assessment
- Signal-to-Noise Ratio (SNR) estimation
- Artifact detection and removal
- Blind source separation
- Adaptive quality assessment

#### Advanced Analysis
- Anomaly detection
- Bayesian analysis
- EMD decomposition
- Neural network filtering
- Reinforcement learning approaches
- Sparse signal processing

#### Health Report Generation
- Comprehensive health reports
- Multiple report types (cardiac, respiratory, quality)
- PDF and HTML export
- Interactive visualizations

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd vital-DSP/src/vitalDSP_webapp
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python run_webapp.py
```

## Usage

### Getting Started
1. **Load Sample Data**: Click "Load Sample PPG Data" to get started quickly
2. **Upload Your Data**: Use the file upload area to load your own CSV/TXT files
3. **Column Mapping**: Map your data columns to vitalDSP expected columns
4. **Configure Parameters**: Set sampling frequency, window size, and other parameters
5. **Analyze**: Navigate through different analysis tabs to explore your data

### Data Format
The application expects data with columns similar to:
- `PLETH` or `waveform`: Main signal data
- `RED_ADC`: Red light intensity
- `IR_ADC`: Infrared light intensity
- `TIMESTAMP_MS`: Time stamps
- `PULSE_BPM`: Heart rate (optional)
- `SPO2_PCT`: Oxygen saturation (optional)

### Navigation
- **Home**: Overview and getting started guide
- **Upload**: Data upload and column mapping
- **Time Domain**: Time series analysis
- **Frequency**: Frequency domain analysis
- **Filtering**: Signal filtering and processing
- **Physiological**: Heart rate and HRV analysis
- **Respiratory**: Breathing pattern analysis
- **Features**: Feature extraction and engineering
- **Transforms**: Signal transformations
- **Quality**: Signal quality assessment
- **Advanced**: Advanced computational methods
- **Health Report**: Report generation
- **Settings**: Application configuration

## Architecture

### Frontend
- **Dash**: Interactive web framework for Python
- **Bootstrap**: Responsive UI components
- **Plotly**: Interactive plotting and visualization

### Backend
- **FastAPI**: High-performance API framework
- **vitalDSP**: Core signal processing library
- **Pandas/NumPy**: Data manipulation and numerical computing

### Data Flow
1. **Upload**: File upload and parsing
2. **Processing**: Signal processing and analysis
3. **Visualization**: Interactive plots and charts
4. **Export**: Results and reports download

## Configuration

### Environment Variables
- `DEBUG`: Enable debug mode (default: False)
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)

### Settings
- Default sampling frequency
- Window size preferences
- Peak detection sensitivity
- Filter configurations
- Theme preferences

## Development

### Project Structure
```
vitalDSP_webapp/
├── app.py                 # Main application
├── run_webapp.py         # Entry point
├── requirements.txt      # Dependencies
├── README.md            # This file
├── api/                 # FastAPI endpoints
├── callbacks/           # Dash callbacks
├── layout/              # UI components
├── models/              # Data models
└── assets/              # Static files
```

### Adding New Features
1. **Layout**: Add UI components in `layout/`
2. **Callbacks**: Implement logic in `callbacks/`
3. **API**: Add endpoints in `api/`
4. **Styling**: Update CSS in `assets/styles.css`

### Testing
```bash
pytest tests/
```

## API Endpoints

### Core Endpoints
- `GET /`: Main dashboard
- `GET /api/health`: Health check
- `POST /api/upload`: File upload
- `GET /api/process`: Signal processing

### Analysis Endpoints
- `POST /api/analyze/time-domain`: Time domain analysis
- `POST /api/analyze/frequency`: Frequency analysis
- `POST /api/analyze/filter`: Signal filtering
- `POST /api/analyze/physiological`: Physiological features
- `POST /api/analyze/respiratory`: Respiratory analysis

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure vitalDSP is in your Python path
2. **Memory Issues**: Reduce data chunk size for large files
3. **Plot Rendering**: Check browser console for JavaScript errors
4. **Performance**: Use appropriate window sizes for real-time analysis

### Debug Mode
Enable debug mode for detailed error messages:
```bash
export DEBUG=True
python run_webapp.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the same license as vitalDSP.

## Support

For issues and questions:
- Check the troubleshooting section
- Review vitalDSP documentation
- Open an issue on GitHub

## Roadmap

### Upcoming Features
- Real-time streaming data support
- Advanced machine learning models
- Cloud deployment options
- Mobile-responsive design
- Multi-language support
- Advanced export formats

### Performance Improvements
- WebSocket support for real-time updates
- Caching and optimization
- Parallel processing
- GPU acceleration support

---

**Note**: This webapp is designed to work with the vitalDSP library. Ensure you have the latest version installed for optimal functionality.
