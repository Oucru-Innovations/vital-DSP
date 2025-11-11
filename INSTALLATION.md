# vitalDSP Installation Guide

This guide provides comprehensive instructions for installing vitalDSP with various optional dependencies.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation Methods](#installation-methods)
- [Optional Dependencies](#optional-dependencies)
- [Requirements Files](#requirements-files)
- [Platform-Specific Notes](#platform-specific-notes)
- [Troubleshooting](#troubleshooting)

## Quick Start

For most users, the basic installation is sufficient:

```bash
pip install vitalDSP
```

For deep learning features, add TensorFlow or PyTorch:

```bash
# With TensorFlow (recommended for Python 3.8-3.12)
pip install vitalDSP[tensorflow]

# With PyTorch (recommended for Python 3.13+)
pip install vitalDSP[pytorch]
```

## Installation Methods

### Method 1: Install from PyPI (Recommended)

#### Basic Installation

Install core vitalDSP without deep learning frameworks:

```bash
pip install vitalDSP
```

This includes:
- Signal processing algorithms
- Traditional ML models (scikit-learn based)
- Filtering and transforms
- Feature extraction
- Web application components

#### Installation with Optional Features

Choose specific features based on your needs:

**1. TensorFlow Backend**
```bash
pip install vitalDSP[tensorflow]
```
- Deep learning models (CNN, LSTM, Autoencoders, Transformers)
- Best for Python 3.8-3.12
- Platform-specific TensorFlow versions

**2. PyTorch Backend**
```bash
pip install vitalDSP[pytorch]
```
- Alternative deep learning backend
- Better Python 3.13+ compatibility
- Same model architectures as TensorFlow

**3. Model Explainability**
```bash
pip install vitalDSP[explainability]
```
- SHAP (SHapley Additive exPlanations)
- LIME (Local Interpretable Model-agnostic Explanations)
- Model interpretation tools

**4. Deep Learning Bundle (TensorFlow only)**
```bash
pip install vitalDSP[dl]
```
- Alias for TensorFlow installation
- For users who prefer explicit naming

**5. Deep Learning Bundle (PyTorch only)**
```bash
pip install vitalDSP[dl-pytorch]
```
- PyTorch-specific installation
- For users who prefer explicit naming

**6. All Deep Learning Frameworks**
```bash
pip install vitalDSP[dl-all]
```
- Both TensorFlow and PyTorch
- Maximum flexibility for backend selection
- Larger installation size

**7. Complete Installation**
```bash
pip install vitalDSP[all]
```
- All optional dependencies
- TensorFlow + PyTorch + Explainability tools
- Recommended for development and research

### Method 2: Install from Source

For the latest development version or to contribute:

#### Step 1: Clone the Repository

```bash
git clone https://github.com/Oucru-Innovations/vital-DSP.git
cd vital-DSP
```

#### Step 2: Install in Development Mode

**Basic installation:**
```bash
pip install -e .
```

**With specific features:**
```bash
# TensorFlow
pip install -e .[tensorflow]

# PyTorch
pip install -e .[pytorch]

# Explainability
pip install -e .[explainability]

# All features
pip install -e .[all]
```

#### Step 3: Using Requirements Files

Alternatively, install dependencies separately:

```bash
# Base installation
pip install -e .

# Add TensorFlow
pip install -r requirements-tensorflow.txt

# Add PyTorch
pip install -r requirements-pytorch.txt

# Add explainability tools
pip install -r requirements-explainability.txt

# Add all dev dependencies (testing, docs, code quality)
pip install -r requirements-dev.txt
```

## Optional Dependencies

### Deep Learning Frameworks

#### TensorFlow (>=2.8.0, <2.17.0)

**Supported Python Versions:** 3.8-3.12

**Platform-Specific Installation:**

- **Windows/Linux:**
  ```bash
  pip install "tensorflow>=2.8.0,<2.17.0"
  ```

- **macOS (Apple Silicon M1/M2):**
  ```bash
  pip install "tensorflow-macos>=2.8.0,<2.17.0"
  ```

- **macOS (Intel):**
  ```bash
  pip install "tensorflow>=2.8.0,<2.17.0"
  ```

**Python 3.13+ Users:**
TensorFlow doesn't officially support Python 3.13 yet. Options:
1. **Recommended:** Use Python 3.11 or 3.12
2. **Experimental:** Install TensorFlow nightly: `pip install tf-nightly`
3. **Alternative:** Use PyTorch instead

#### PyTorch (>=2.0.0)

**Supported Python Versions:** 3.8-3.13+

**Installation:**
```bash
pip install torch>=2.0.0 torchvision>=0.15.0
```

**Benefits:**
- Better Python 3.13+ compatibility
- Simpler installation across platforms
- Growing ecosystem support

### Explainability Tools

#### SHAP (>=0.41.0)

For model interpretation and feature importance:
```bash
pip install shap>=0.41.0
```

**Features:**
- TreeExplainer for tree-based models
- KernelExplainer for any model
- DeepExplainer for neural networks
- Unified framework for explainability

#### LIME (>=0.2.0)

For local interpretable model explanations:
```bash
pip install lime>=0.2.0
```

**Features:**
- Model-agnostic explanations
- Local instance interpretation
- Easy-to-understand visualizations

## Requirements Files

The repository includes several requirements files for different use cases:

| File | Purpose | Install Command |
|------|---------|-----------------|
| `requirements.txt` | Core dependencies | `pip install -r requirements.txt` |
| `requirements-tensorflow.txt` | TensorFlow only | `pip install -r requirements-tensorflow.txt` |
| `requirements-pytorch.txt` | PyTorch only | `pip install -r requirements-pytorch.txt` |
| `requirements-explainability.txt` | SHAP + LIME | `pip install -r requirements-explainability.txt` |
| `requirements-dev.txt` | All dependencies + dev tools | `pip install -r requirements-dev.txt` |

## Platform-Specific Notes

### Windows

- TensorFlow installation is straightforward
- PyTorch may require Visual C++ redistributables
- GPU support requires CUDA toolkit

### macOS

**Apple Silicon (M1/M2/M3):**
- Use `tensorflow-macos` for optimal performance
- PyTorch has native ARM support
- Install via pip or conda

**Intel:**
- Standard TensorFlow installation
- Both frameworks work out of the box

### Linux

- Best performance for deep learning
- Easy GPU support configuration
- Use CUDA for GPU acceleration

## Troubleshooting

### Common Issues

**1. TensorFlow Installation Fails on Python 3.13**

```bash
# Solution: Use PyTorch instead
pip install vitalDSP[pytorch]

# Or downgrade Python to 3.11/3.12
pyenv install 3.12
pyenv local 3.12
```

**2. Import Error for Deep Learning Models**

```python
# Error: ImportError: TensorFlow not available
# Solution: Install deep learning backend
pip install vitalDSP[tensorflow]
# or
pip install vitalDSP[pytorch]
```

**3. SHAP/LIME Import Errors**

```bash
# Solution: Install explainability tools
pip install vitalDSP[explainability]
```

**4. Version Conflicts**

```bash
# Solution: Create a fresh virtual environment
python -m venv vitalDSP-env
source vitalDSP-env/bin/activate  # On Windows: vitalDSP-env\Scripts\activate
pip install vitalDSP[all]
```

### Getting Help

If you encounter issues:

1. Check the [GitHub Issues](https://github.com/Oucru-Innovations/vital-DSP/issues)
2. Review the [Documentation](https://vital-dsp.readthedocs.io/)
3. Ask questions in [Discussions](https://github.com/Oucru-Innovations/vital-DSP/discussions)

## Verification

Verify your installation:

```python
import vitalDSP
print(f"vitalDSP version: {vitalDSP.__version__}")

# Check TensorFlow availability
try:
    from vitalDSP.ml_models import deep_models
    print("✓ Deep learning models available")
except ImportError as e:
    print(f"✗ Deep learning models not available: {e}")

# Check explainability tools
try:
    from vitalDSP.ml_models import explainability
    print("✓ Explainability tools available")
except ImportError as e:
    print(f"✗ Explainability tools not available: {e}")
```

## Next Steps

After installation:

1. Explore the [examples](examples/) directory
2. Read the [documentation](https://vital-dsp.readthedocs.io/)
3. Try the [web application](https://vital-dsp-1.onrender.com/)
4. Check out the [tutorials](docs/tutorials/)
