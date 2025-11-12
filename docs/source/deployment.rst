Deployment Guide
=================

This comprehensive guide provides detailed instructions for deploying VitalDSP in various environments, from local development to production cloud deployments.

.. contents:: Table of Contents
   :local:
   :depth: 3

Deployment Overview
====================

VitalDSP offers flexible deployment options:

* **🐛 Development Mode**: Local development with hot-reload and debug logging
* **ℹ️  Normal Mode**: Standard testing environment with INFO-level logging
* **✅ Production Mode**: Optimized for deployment with minimal logging
* **🐳 Docker Deployment**: Containerized deployment with Docker/Kubernetes
* **☁️  Cloud Deployment**: AWS, GCP, Azure, and Render.com
* **🔧 Edge Deployment**: Raspberry Pi and IoT devices

Prerequisites
==============

System Requirements
-------------------

**Minimum Requirements**

* Python 3.9 or higher (3.10+ recommended)
* 4GB RAM minimum (8GB recommended for production)
* 10GB disk space minimum (50GB recommended for large datasets)
* Network connectivity for cloud deployments

**Recommended Production Requirements**

* Python 3.10+
* 16GB RAM
* 100GB SSD storage
* Multi-core CPU (4+ cores)
* HTTPS/SSL certificates

Software Dependencies
---------------------

**System Packages**

.. code-block:: bash

   # Ubuntu/Debian
   sudo apt update
   sudo apt install -y python3.10 python3-pip python3-venv git gcc g++ curl

   # CentOS/RHEL
   sudo yum install -y python3.10 python3-pip python3-venv git gcc gcc-c++ curl

   # macOS (with Homebrew)
   brew install python@3.10 git

**Python Dependencies**

All Python dependencies are managed through ``requirements.txt`` and installed automatically.

Quick Start Deployment
=======================

Local Development (Fastest)
----------------------------

**Step 1: Clone and Install**

.. code-block:: bash

   git clone https://github.com/Oucru-Innovations/vital-DSP.git
   cd vital-DSP
   pip install -e .
   pip install -r src/vitalDSP_webapp/requirements.txt

**Step 2: Run in Development Mode**

.. code-block:: bash

   # Option 1: Using Python directly
   python src/vitalDSP_webapp/run_webapp.py --debug

   # Option 2: Using convenience scripts
   # On Windows:
   run_webapp.bat     # Interactive menu

   # On Linux/Mac:
   bash run_webapp.sh # Interactive menu

**Step 3: Access Application**

Open your browser to ``http://localhost:8000``

Available Deployment Modes
============================

The unified ``run_webapp.py`` script supports multiple deployment modes:

Debug Mode (Development)
-------------------------

**Purpose**: Local development with comprehensive logging and auto-reload

**Features**:

* DEBUG-level logging with detailed traces
* Auto-reload on code changes
* Detailed error messages
* Logs saved to ``logs/webapp_debug.log``

**Usage**:

.. code-block:: bash

   # Command-line
   python src/vitalDSP_webapp/run_webapp.py --debug

   # With custom port
   python src/vitalDSP_webapp/run_webapp.py --debug --port 8080

   # With custom host
   python src/vitalDSP_webapp/run_webapp.py --debug --host 127.0.0.1

Normal Mode (Testing)
----------------------

**Purpose**: Standard testing and local deployment

**Features**:

* INFO-level logging
* No auto-reload (more stable)
* Standard error messages
* Logs saved to ``logs/webapp.log``

**Usage**:

.. code-block:: bash

   # Default mode (no flags needed)
   python src/vitalDSP_webapp/run_webapp.py

   # With custom port
   python src/vitalDSP_webapp/run_webapp.py --port 8080

Production Mode (Deployment)
-----------------------------

**Purpose**: Optimized for production deployment

**Features**:

* WARNING-level logging only (minimal overhead)
* Optimized performance
* Access logs disabled for better performance
* Logs saved to ``logs/webapp_production.log``

**Usage**:

.. code-block:: bash

   # Production mode
   python src/vitalDSP_webapp/run_webapp.py --production

   # With environment variable port (for cloud platforms)
   PORT=8000 python src/vitalDSP_webapp/run_webapp.py --production

Command-Line Options
--------------------

.. code-block:: bash

   python src/vitalDSP_webapp/run_webapp.py [OPTIONS]

**Available Options**:

* ``-d, --debug``: Enable debug mode (DEBUG logging + auto-reload)
* ``-p, --production``: Enable production mode (optimized, minimal logs)
* ``--port PORT``: Specify port number (default: 8000)
* ``--host HOST``: Specify host address (default: 0.0.0.0)
* ``--reload``: Force enable auto-reload
* ``-h, --help``: Show help message

**Examples**:

.. code-block:: bash

   # Development with auto-reload
   python src/vitalDSP_webapp/run_webapp.py --debug

   # Testing on custom port
   python src/vitalDSP_webapp/run_webapp.py --port 8080

   # Production deployment
   python src/vitalDSP_webapp/run_webapp.py --production

   # Custom host binding
   python src/vitalDSP_webapp/run_webapp.py --host 127.0.0.1 --port 3000

Interactive Scripts
-------------------

**Windows (run_webapp.bat)**

.. code-block:: batch

   # Run the script
   run_webapp.bat

   # Choose from menu:
   # 1. Normal Mode
   # 2. Debug Mode
   # 3. Production Mode
   # 4. Custom Mode (enter your own options)

**Linux/Mac (run_webapp.sh)**

.. code-block:: bash

   # Make executable
   chmod +x run_webapp.sh

   # Run the script
   ./run_webapp.sh

   # Choose from menu:
   # 1. Normal Mode
   # 2. Debug Mode
   # 3. Production Mode
   # 4. Custom Mode (enter your own options)

Docker Deployment
==================

VitalDSP provides two Dockerfiles for different use cases. The current version is **0.2.1**.

Standard Docker Deployment
--------------------------

**Dockerfile**: Optimized for Render.com and similar platforms

**Build and Run**:

.. code-block:: bash

   # Build the image (version 0.2.1)
   docker build -t vitaldsp:0.2.1 .
   docker tag vitaldsp:0.2.1 vitaldsp:latest

   # Run the container
   docker run -p 8000:8000 -e PORT=8000 vitaldsp:latest

   # Run with mounted volumes
   docker run -p 8000:8000 \
     -v $(pwd)/uploads:/app/uploads \
     -v $(pwd)/logs:/app/logs \
     -e PORT=8000 \
     vitaldsp:latest

**Dockerfile Configuration**:

.. code-block:: dockerfile

   FROM python:3.10-slim

   ENV PYTHONUNBUFFERED=1
   ENV PYTHONDONTWRITEBYTECODE=1
   ENV PYTHONPATH=/app:/app/src:/app/src/vitalDSP_webapp

   WORKDIR /app

   # Install dependencies
   RUN apt-get update && apt-get install -y --no-install-recommends \
       gcc g++ curl && rm -rf /var/lib/apt/lists/*

   # Copy and install requirements
   COPY requirements.txt /app/requirements.txt
   COPY src/vitalDSP_webapp/requirements.txt /app/webapp_requirements.txt
   RUN pip install --no-cache-dir -U pip && \
       pip install --no-cache-dir -r /app/requirements.txt && \
       pip install --no-cache-dir -r /app/webapp_requirements.txt

   # Copy application
   COPY . /app
   RUN pip install -e .
   RUN mkdir -p /app/uploads /app/logs

   EXPOSE 8000

   # Run with Gunicorn for production
   CMD exec gunicorn -k uvicorn.workers.UvicornWorker \
       --bind 0.0.0.0:${PORT:-8000} \
       -w 1 --timeout 120 \
       --access-logfile - --error-logfile - \
       --log-level debug \
       vitalDSP_webapp.run_webapp:app

Production Docker Deployment
-----------------------------

**Dockerfile.production**: Multi-stage build for optimized production

**Features**:

* Multi-stage build (smaller image)
* Non-root user for security
* Health checks included
* Optimized for performance

**Build and Run**:

.. code-block:: bash

   # Build production image
   docker build -f Dockerfile.production -t vitaldsp:production .

   # Run production container
   docker run -p 8000:8000 \
     -e PORT=8000 \
     -e PYTHONPATH=/app:/app/src \
     vitaldsp:production

Docker Compose Deployment
--------------------------

**docker-compose.yml** provides a complete stack with nginx reverse proxy for version 0.2.1:

.. code-block:: yaml

   version: '3.8'

   services:
     vitaldsp-webapp:
       build:
         context: .
         dockerfile: Dockerfile.production
       container_name: vitaldsp-webapp
       ports:
         - "8000:8000"
       environment:
         - PORT=8000
         - PYTHONPATH=/app:/app/src
         - PYTHONUNBUFFERED=1
       volumes:
         - ./uploads:/app/uploads
         - ./logs:/app/logs
       restart: unless-stopped
       healthcheck:
         test: ["CMD", "curl", "-f", "http://localhost:8000/api/health"]
         interval: 30s
         timeout: 10s
         retries: 3
         start_period: 40s
       networks:
         - vitaldsp-network

     # Optional: nginx reverse proxy
     nginx:
       image: nginx:alpine
       container_name: vitaldsp-nginx
       ports:
         - "80:80"
         - "443:443"
       volumes:
         - ./nginx.conf:/etc/nginx/nginx.conf:ro
         - ./ssl:/etc/nginx/ssl:ro
       depends_on:
         - vitaldsp-webapp
       restart: unless-stopped
       networks:
         - vitaldsp-network

   networks:
     vitaldsp-network:
       driver: bridge

**Deploy with Docker Compose**:

.. code-block:: bash

   # Start all services
   docker-compose up -d

   # View logs
   docker-compose logs -f vitaldsp-webapp

   # Stop all services
   docker-compose down

   # Rebuild and restart
   docker-compose up -d --build

Cloud Platform Deployment
==========================

Render.com Deployment
---------------------

VitalDSP is **currently deployed** on Render.com: https://vital-dsp-1.onrender.com/

**Step 1: Connect Repository**

1. Sign up at https://render.com
2. Click "New +" → "Web Service"
3. Connect your GitHub repository

**Step 2: Configure Service**

.. code-block:: yaml

   # render.yaml (optional, for automatic deployment)
   services:
     - type: web
       name: vitaldsp
       env: python
       plan: free  # or starter/standard
       buildCommand: pip install -r requirements.txt && pip install -r src/vitalDSP_webapp/requirements.txt && pip install -e .
       startCommand: gunicorn -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT vitalDSP_webapp.run_webapp:app
       envVars:
         - key: PYTHON_VERSION
           value: 3.10.0
         - key: PYTHONPATH
           value: /opt/render/project/src:/opt/render/project/src/vitalDSP_webapp

**Manual Configuration**:

* **Build Command**: ``pip install -r requirements.txt && pip install -r src/vitalDSP_webapp/requirements.txt && pip install -e .``
* **Start Command**: ``gunicorn -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT vitalDSP_webapp.run_webapp:app``
* **Environment**: Python 3
* **Plan**: Free tier works for testing

**Important Notes for Render Free Tier**:

* Services spin down after 15 minutes of inactivity
* Cold start takes 30-60 seconds
* Upgrade to paid tier for production use

AWS EC2 Deployment
------------------

**Step 1: Launch EC2 Instance**

.. code-block:: bash

   # Launch Ubuntu 22.04 LTS instance
   # Instance type: t3.medium or larger recommended
   # Security group: Allow HTTP (80), HTTPS (443), SSH (22)

**Step 2: Connect and Install**

.. code-block:: bash

   # SSH into instance
   ssh -i your-key.pem ubuntu@your-instance-ip

   # Update system
   sudo apt update && sudo apt upgrade -y

   # Install Python and dependencies
   sudo apt install -y python3.10 python3-pip python3-venv git gcc g++ curl nginx

   # Install SSL/TLS support
   sudo apt install -y certbot python3-certbot-nginx

**Step 3: Deploy Application**

.. code-block:: bash

   # Create application directory
   sudo mkdir -p /opt/vitaldsp
   sudo chown ubuntu:ubuntu /opt/vitaldsp
   cd /opt/vitaldsp

   # Clone repository
   git clone https://github.com/Oucru-Innovations/vital-DSP.git .

   # Create virtual environment
   python3.10 -m venv venv
   source venv/bin/activate

   # Install dependencies
   pip install -r requirements.txt
   pip install -r src/vitalDSP_webapp/requirements.txt
   pip install -e .
   pip install gunicorn

**Step 4: Create Systemd Service**

.. code-block:: ini

   # /etc/systemd/system/vitaldsp.service
   [Unit]
   Description=VitalDSP Web Application
   After=network.target

   [Service]
   Type=simple
   User=ubuntu
   Group=ubuntu
   WorkingDirectory=/opt/vitaldsp
   Environment="PATH=/opt/vitaldsp/venv/bin"
   Environment="PYTHONPATH=/opt/vitaldsp:/opt/vitaldsp/src"
   ExecStart=/opt/vitaldsp/venv/bin/python src/vitalDSP_webapp/run_webapp.py --production --port 8000
   Restart=always
   RestartSec=10

   [Install]
   WantedBy=multi-user.target

**Step 5: Configure Nginx**

.. code-block:: nginx

   # /etc/nginx/sites-available/vitaldsp
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;

           # WebSocket support
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";

           # Timeouts
           proxy_connect_timeout 60s;
           proxy_send_timeout 60s;
           proxy_read_timeout 60s;
       }

       location /static {
           alias /opt/vitaldsp/static;
           expires 1y;
           add_header Cache-Control "public, immutable";
       }
   }

.. code-block:: bash

   # Enable site and SSL
   sudo ln -s /etc/nginx/sites-available/vitaldsp /etc/nginx/sites-enabled/
   sudo certbot --nginx -d your-domain.com
   sudo systemctl restart nginx

   # Start VitalDSP service
   sudo systemctl enable vitaldsp
   sudo systemctl start vitaldsp

Google Cloud Platform (GCP)
----------------------------

**Using Cloud Run (Recommended)**

.. code-block:: bash

   # Build and push to Container Registry
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/vitaldsp

   # Deploy to Cloud Run
   gcloud run deploy vitaldsp \
     --image gcr.io/YOUR_PROJECT_ID/vitaldsp \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 2Gi \
     --cpu 2 \
     --port 8000

**Using Compute Engine**

.. code-block:: bash

   # Create instance
   gcloud compute instances create vitaldsp-server \
     --image-family=ubuntu-2204-lts \
     --image-project=ubuntu-os-cloud \
     --machine-type=e2-medium \
     --zone=us-central1-a \
     --tags=http-server,https-server

   # SSH and follow AWS EC2 deployment steps

Azure Deployment
----------------

**Using Azure App Service**

.. code-block:: bash

   # Create resource group
   az group create --name vitaldsp-rg --location eastus

   # Create App Service plan
   az appservice plan create \
     --name vitaldsp-plan \
     --resource-group vitaldsp-rg \
     --sku B1 \
     --is-linux

   # Create web app
   az webapp create \
     --resource-group vitaldsp-rg \
     --plan vitaldsp-plan \
     --name vitaldsp-app \
     --runtime "PYTHON:3.10"

   # Deploy from GitHub
   az webapp deployment source config \
     --name vitaldsp-app \
     --resource-group vitaldsp-rg \
     --repo-url https://github.com/Oucru-Innovations/vital-DSP \
     --branch main

Production Best Practices
==========================

Environment Variables
---------------------

Create a ``.env`` file for configuration:

.. code-block:: bash

   # .env
   PORT=8000
   PYTHONPATH=/app:/app/src
   PYTHONUNBUFFERED=1
   SECRET_KEY=your-secret-key-here
   LOG_LEVEL=WARNING
   MAX_UPLOAD_SIZE=16777216  # 16MB
   ENABLE_CORS=false

Monitoring and Logging
----------------------

**Log Rotation**

.. code-block:: bash

   # /etc/logrotate.d/vitaldsp
   /opt/vitaldsp/logs/*.log {
       daily
       missingok
       rotate 14
       compress
       delaycompress
       notifempty
       create 0640 ubuntu ubuntu
       sharedscripts
       postrotate
           systemctl reload vitaldsp > /dev/null 2>&1 || true
       endscript
   }

**Health Monitoring**

The application includes a health check endpoint at ``/api/health``:

.. code-block:: bash

   # Test health endpoint
   curl http://localhost:8000/api/health

**Uptime Monitoring**

Use services like:

* UptimeRobot (https://uptimerobot.com/)
* Pingdom (https://www.pingdom.com/)
* StatusCake (https://www.statuscake.com/)

Security Configuration
----------------------

**SSL/TLS Configuration**

.. code-block:: bash

   # Generate Let's Encrypt certificate
   sudo certbot --nginx -d your-domain.com -d www.your-domain.com

   # Auto-renewal
   sudo systemctl enable certbot.timer

**Firewall Configuration**

.. code-block:: bash

   # UFW (Ubuntu)
   sudo ufw allow 22    # SSH
   sudo ufw allow 80    # HTTP
   sudo ufw allow 443   # HTTPS
   sudo ufw enable

Performance Optimization
------------------------

**Gunicorn Workers**

.. code-block:: bash

   # Calculate optimal workers: (2 x CPU cores) + 1
   gunicorn -k uvicorn.workers.UvicornWorker \
     --bind 0.0.0.0:8000 \
     --workers 5 \
     --worker-class uvicorn.workers.UvicornWorker \
     --worker-connections 1000 \
     --timeout 120 \
     --max-requests 1000 \
     --max-requests-jitter 50 \
     vitalDSP_webapp.run_webapp:app

**Nginx Caching**

.. code-block:: nginx

   # Add to nginx.conf
   proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=vitaldsp_cache:10m max_size=1g;

   location / {
       proxy_cache vitaldsp_cache;
       proxy_cache_valid 200 1h;
       proxy_cache_key $scheme$request_method$host$request_uri;
   }

Backup and Recovery
-------------------

**Automated Backup Script**

.. code-block:: bash

   #!/bin/bash
   # /opt/vitaldsp/backup.sh

   BACKUP_DIR="/backup/vitaldsp"
   DATE=$(date +%Y%m%d_%H%M%S)

   # Create backup directory
   mkdir -p $BACKUP_DIR

   # Backup application
   tar -czf $BACKUP_DIR/vitaldsp_$DATE.tar.gz \
     /opt/vitaldsp \
     --exclude='*.log' \
     --exclude='__pycache__' \
     --exclude='venv'

   # Keep last 7 days
   find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete

**Schedule Backups**

.. code-block:: bash

   # Add to crontab
   crontab -e
   # Add: 0 2 * * * /opt/vitaldsp/backup.sh

Troubleshooting
===============

Common Issues
-------------

**Port Already in Use**

.. code-block:: bash

   # Find process using port 8000
   lsof -i :8000

   # Kill process
   kill -9 <PID>

   # Or use a different port
   python src/vitalDSP_webapp/run_webapp.py --port 8001

**Import Errors**

.. code-block:: bash

   # Ensure PYTHONPATH is set correctly
   export PYTHONPATH=/path/to/vital-DSP:/path/to/vital-DSP/src
   python src/vitalDSP_webapp/run_webapp.py

**Memory Issues**

.. code-block:: bash

   # Reduce workers in production
   gunicorn -w 2 ...  # Instead of 4-8 workers

   # Limit upload size
   MAX_CONTENT_LENGTH=8388608  # 8MB instead of 16MB

**Cold Start Delays (Render Free Tier)**

* Upgrade to paid tier for always-on service
* Use a ping service to keep app warm
* Accept 30-60s cold start time for free tier

Debugging
---------

**Check Logs**

.. code-block:: bash

   # Application logs
   tail -f logs/webapp.log

   # Systemd service logs
   sudo journalctl -u vitaldsp -f

   # Nginx logs
   sudo tail -f /var/log/nginx/error.log

**Test in Debug Mode**

.. code-block:: bash

   # Run in debug mode for detailed error messages
   python src/vitalDSP_webapp/run_webapp.py --debug

Support and Resources
=====================

* **Documentation**: https://vital-dsp.readthedocs.io/
* **GitHub Issues**: https://github.com/Oucru-Innovations/vital-DSP/issues
* **Live Demo**: https://vital-dsp-1.onrender.com/
* **Community**: GitHub Discussions

Conclusion
==========

This deployment guide covers multiple deployment scenarios from local development to production cloud deployment. Choose the method that best fits your requirements:

* **Quick Testing**: Use ``--debug`` mode locally
* **Small Projects**: Deploy on Render.com free tier
* **Production**: Use AWS/GCP/Azure with Docker and nginx
* **Enterprise**: Use Kubernetes with multiple replicas

For additional support, please consult our GitHub repository or reach out to the community.
