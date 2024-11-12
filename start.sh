#!/bin/bash

# Startup script for Render.com deployment
# This script ensures proper environment setup and starts the application

echo "Starting vitalDSP Webapp on Render.com..."

# Set Python path
export PYTHONPATH=/opt/render/project/src:$PYTHONPATH

# Create uploads directory if it doesn't exist
mkdir -p uploads

# Install dependencies if needed (for development)
if [ "$RENDER" = "true" ]; then
    echo "Running in Render.com environment"
    # Dependencies should already be installed via buildCommand
else
    echo "Installing dependencies..."
    pip install -r src/vitalDSP_webapp/requirements-prod.txt
fi

# Start the application
echo "Starting application on port ${PORT:-8000}..."
python src/vitalDSP_webapp/run_webapp.py
