FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app:/app/src

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
COPY src/vitalDSP_webapp/requirements.txt /app/webapp_requirements.txt

RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir -r /app/webapp_requirements.txt

COPY . /app
RUN pip install -e .
RUN mkdir -p /app/uploads

EXPOSE 8000

CMD ["python", "src/vitalDSP_webapp/run_webapp.py"]