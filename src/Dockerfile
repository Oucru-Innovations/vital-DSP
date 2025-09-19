# Dockerfile
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/app/src:/app/src/vitalDSP_webapp

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
COPY src/vitalDSP_webapp/requirements.txt /app/webapp_requirements.txt
RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir -r /app/requirements.txt \
 && pip install --no-cache-dir -r /app/webapp_requirements.txt

COPY . /app
RUN mkdir -p /app/uploads

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/api/health || exit 1

# Run the application with gunicorn
CMD exec gunicorn -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-8000} -w 1 --timeout 120 --access-logfile - --error-logfile - --log-level debug vitalDSP_webapp.run_webapp:app
