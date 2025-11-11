# Documentation Build Guide

This guide explains how to build and maintain vitalDSP documentation for ReadTheDocs.

## Quick Start

Build documentation locally:

```bash
make docs-rtd
```

This simulates the ReadTheDocs build environment and creates HTML documentation in `docs/_build/html/`.

## Available Make Commands

### Documentation Commands

| Command | Description |
|---------|-------------|
| `make html` | Build HTML documentation (basic) |
| `make docs-rtd` | Build docs simulating ReadTheDocs environment |
| `make docs-all` | Build all formats (HTML, PDF, EPUB) |
| `make docs-serve` | Serve docs locally with auto-reload |
| `make docs-check` | Check docs for broken links and errors |
| `make docs-clean` | Clean documentation build directory |

### Usage Examples

**1. Build and View Documentation Locally**

```bash
# Build documentation
make docs-rtd

# Open in browser (Windows)
start docs/_build/html/index.html

# Open in browser (Linux/macOS)
open docs/_build/html/index.html
# or
xdg-open docs/_build/html/index.html
```

**2. Live Documentation Server**

```bash
# Start auto-reloading documentation server
make docs-serve

# Access at http://127.0.0.1:8001
# Documentation auto-rebuilds when you edit source files
```

**3. Check for Errors**

```bash
# Check for broken links and run doctests
make docs-check
```

**4. Clean Build**

```bash
# Clean and rebuild
make docs-clean
make docs-rtd
```

## ReadTheDocs Configuration

### Configuration File

The `.readthedocs.yaml` file at the project root configures ReadTheDocs builds:

```yaml
version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.9"
  jobs:
    post_install:
    - pip install git+https://github.com/Oucru-Innovations/vital-DSP.git

python:
  install:
    - requirements: docs/requirements.txt
    - method: pip
      path: .

formats:
  - pdf
  - epub

sphinx:
  builder: html
  configuration: docs/source/conf.py
  fail_on_warning: false
```

### Documentation Dependencies

Located in `docs/requirements.txt`:

- Sphinx >= 4.0.0
- sphinx_rtd_theme
- nbsphinx (for Jupyter notebooks)
- myst-nb (for MyST markdown)
- sphinx-plotly-directive
- And more...

## Triggering ReadTheDocs Builds

ReadTheDocs automatically builds documentation when:

1. **Commits are pushed** to the main branch
2. **Pull requests** are created
3. **Manual trigger** from ReadTheDocs dashboard

### Manual Trigger Steps

If automatic builds aren't triggering:

1. Go to [ReadTheDocs Dashboard](https://readthedocs.org/dashboard/)
2. Select your project: `vital-DSP`
3. Click **"Builds"** tab
4. Click **"Build Version"** button
5. Select branch/tag and click **"Build"**

### Verifying Build Status

Check build status at:
- Latest build: https://readthedocs.org/projects/vital-dsp/builds/
- Live docs: https://vital-dsp.readthedocs.io/

## Common Issues and Solutions

### Issue 1: Build Fails with Import Errors

**Symptom:** ReadTheDocs build fails with `ModuleNotFoundError`

**Solution:** Add missing packages to `docs/requirements.txt`

```bash
# Test locally first
pip install -r docs/requirements.txt
make docs-rtd
```

### Issue 2: Documentation Not Updating

**Symptom:** Changes don't appear on ReadTheDocs

**Solutions:**

1. **Check if commit was pushed:**
   ```bash
   git log --oneline -5
   git push origin main
   ```

2. **Verify webhook is active:**
   - Go to GitHub repo → Settings → Webhooks
   - Ensure ReadTheDocs webhook is present and recent deliveries are successful

3. **Manually trigger build:**
   - Go to ReadTheDocs dashboard
   - Builds → Build Version

4. **Update .readthedocs.yaml:**
   - Add a comment with today's date
   - This forces ReadTheDocs to recognize changes

### Issue 3: PDF/EPUB Builds Failing

**Symptom:** HTML builds succeed but PDF/EPUB fail

**Solution:** Check for LaTeX-incompatible content

```bash
# Build locally to test
make docs-all

# Check latex errors
ls docs/_build/latex/
```

### Issue 4: Notebook Execution Errors

**Symptom:** Builds fail during notebook execution

**Solutions:**

1. **Disable notebook execution** (temporary):
   ```python
   # In docs/source/conf.py
   nb_execution_mode = "off"  # Change from "auto"
   ```

2. **Increase timeout**:
   ```python
   # In docs/source/conf.py
   nb_execution_timeout = 300  # Increase from 180
   ```

3. **Fix notebook code** to work without dependencies

## Documentation Structure

```
docs/
├── source/
│   ├── conf.py           # Sphinx configuration
│   ├── index.rst         # Main documentation index
│   ├── installation.rst  # Installation guide
│   ├── api/             # API documentation
│   ├── examples/        # Example notebooks
│   └── tutorials/       # Tutorial notebooks
├── requirements.txt      # Documentation dependencies
└── _build/              # Build output (generated)
    ├── html/            # HTML documentation
    ├── latex/           # LaTeX/PDF output
    └── epub/            # EPUB ebook
```

## Best Practices

### 1. Test Locally Before Pushing

Always build documentation locally to catch errors:

```bash
make docs-clean
make docs-rtd
```

### 2. Keep Dependencies Updated

Periodically update `docs/requirements.txt`:

```bash
pip list --outdated
pip install --upgrade sphinx sphinx_rtd_theme
```

### 3. Use Semantic Versioning

Update version in `docs/source/conf.py` to match `setup.py`:

```python
release = '0.1.5'
version = '0.1.5'
```

### 4. Monitor Build Times

ReadTheDocs has time limits:
- Community: 15 minutes per build
- Keep notebooks lightweight
- Cache heavy computations

### 5. Version Documentation

ReadTheDocs supports multiple versions:
- **latest** - main branch
- **stable** - latest release
- Version tags - specific releases

## Troubleshooting Commands

```bash
# 1. Check Sphinx version
sphinx-build --version

# 2. Validate .readthedocs.yaml
# Use ReadTheDocs config validator online

# 3. Test documentation build with verbose output
sphinx-build -v docs/source docs/_build/html

# 4. Check for syntax errors in RST files
python -m rst2html docs/source/index.rst > /tmp/test.html

# 5. List all documentation dependencies
pip list | grep -i sphinx
pip list | grep -i doc
```

## Getting Help

If builds continue to fail:

1. Check [ReadTheDocs build logs](https://readthedocs.org/projects/vital-dsp/builds/)
2. Review [Sphinx documentation](https://www.sphinx-doc.org/)
3. Ask on [ReadTheDocs community](https://github.com/readthedocs/readthedocs.org/discussions)
4. Open an issue in the [vital-DSP repository](https://github.com/Oucru-Innovations/vital-DSP/issues)

## Automation with GitHub Actions

Consider adding a GitHub Action to validate docs on PR:

```yaml
# .github/workflows/docs.yml
name: Documentation

on: [pull_request]

jobs:
  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r docs/requirements.txt
          pip install -e .
      - name: Build documentation
        run: make docs-rtd
```

## Summary

- Use `make docs-rtd` for local builds
- Use `make docs-serve` for live development
- Update `.readthedocs.yaml` to trigger rebuilds
- Check ReadTheDocs dashboard for build status
- Test locally before pushing changes
