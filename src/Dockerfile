# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app:/app/vitalDSP
ENV PORT=8000

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# Copy the source code
COPY . /app

# Create uploads directory
RUN mkdir -p /app/uploads

# Expose the port used by the application
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/api/health || exit 1

# Start the application
CMD ["python", "src/vitalDSP_webapp/run_webapp.py"]
