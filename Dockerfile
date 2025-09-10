# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/src

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY src/vitalDSP_webapp/requirements.txt /app/requirements.txt
COPY requirements.txt /app/root_requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r /app/root_requirements.txt
RUN pip install -r /app/requirements.txt

# Copy the source code
COPY . /app

# Install the package in development mode
RUN pip install -e .

# Create uploads directory
RUN mkdir -p /app/uploads

# Expose the port used by the application
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Start the application
CMD ["python", "src/vitalDSP_webapp/run_webapp.py"]