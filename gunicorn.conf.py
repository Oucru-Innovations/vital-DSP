# Gunicorn configuration file
import os

bind = f"0.0.0.0:{os.environ.get('PORT', 8000)}"
workers = 1
timeout = 120
worker_class = "uvicorn.workers.UvicornWorker"
accesslog = "-"
errorlog = "-"
loglevel = "info"
