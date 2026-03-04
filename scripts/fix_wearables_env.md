# Fix wearables env: NumPy / SciPy mismatch

## What’s wrong
- **NumPy 2.0.2** is installed (dtype layout changed).
- **SciPy** in your env was built for **NumPy 1.x** → binary mismatch → `ValueError: numpy.dtype size changed`.
- This repo expects **NumPy &lt; 2** (see `setup.py` and `requirements.txt`).

## Fix (in `wearables` env)

```bash
# 1) Activate env
conda activate wearables

# 2) Install NumPy 1.x and reinstall SciPy (so it matches)
pip install "numpy>=1.21.6,<2.0"
pip install --force-reinstall --no-cache-dir scipy

# 3) Confirm versions
python -c "import numpy; import scipy; print('numpy', numpy.__version__); print('scipy', scipy.__version__)"
```

You should see e.g. `numpy 1.26.x` and `scipy 1.11.x` (or similar 1.x).

## Optional: match webapp prod exactly

If you want to match `src/vitalDSP_webapp/requirements-prod.txt`:

```bash
pip install numpy==1.24.4 scipy==1.11.4
```

Then run the webapp again:

```bash
python src/vitalDSP_webapp/run_webapp.py
```
