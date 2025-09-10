# Dockerfile
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/app/vitalDSP

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir -r /app/requirements.txt \
 && pip install --no-cache-dir gunicorn uvicorn

COPY . /app
RUN mkdir -p /app/uploads

# Don't hardcode ENV PORT; Render injects it. EXPOSE is optional, harmless:
EXPOSE 8000

# Healthcheck hits our /api/health
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=5 \
  CMD curl -fsS "http://localhost:${PORT:-8000}/api/health" || exit 1

# Use Gunicorn with Uvicorn worker
CMD exec gunicorn "vitalDSP_webapp.run_webapp:app" \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:${PORT:-8000} \
  --workers 2 --timeout 120
