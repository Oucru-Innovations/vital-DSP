#!/bin/bash
set -e

echo "Starting vital-DSP application..."
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo "PORT: ${PORT:-8000}"

# Check if gunicorn is available
if command -v gunicorn &> /dev/null; then
    echo "Gunicorn found, using gunicorn..."
    exec gunicorn "vitalDSP_webapp.run_webapp:app" \
        -k uvicorn.workers.UvicornWorker \
        --bind 0.0.0.0:${PORT:-8000} \
        --workers 1 \
        --timeout 120 \
        --access-logfile - \
        --error-logfile -
else
    echo "Gunicorn not found, using Python directly..."
    exec python vitalDSP_webapp/run_webapp.py
fi
