Deployment Guide
=================

This guide provides comprehensive instructions for deploying VitalDSP in various environments, from development to production.

Deployment Overview
====================

VitalDSP can be deployed in multiple configurations:

* **Development Environment**: Local development and testing
* **Staging Environment**: Pre-production testing and validation
* **Production Environment**: Live deployment for end users
* **Cloud Deployment**: Scalable cloud-based deployment
* **Containerized Deployment**: Docker-based deployment
* **Edge Deployment**: Embedded and IoT device deployment

Prerequisites
==============

**System Requirements**

* Python 3.8 or higher
* 4GB RAM minimum (8GB recommended)
* 10GB disk space minimum
* Network connectivity for cloud deployments

**Software Dependencies**

* Git (for source code deployment)
* Docker (for containerized deployment)
* Nginx (for production web server)
* SSL certificates (for HTTPS)

**Python Dependencies**

* vital-DSP
* dash
* plotly
* pandas
* numpy
* scipy
* scikit-learn

Development Deployment
=======================

**Local Development Setup**

1. **Clone the repository:**
   .. code-block:: bash
   
      git clone https://github.com/Oucru-Innovations/vital-DSP.git
      cd vital-DSP

2. **Create virtual environment:**
   .. code-block:: bash
   
      python -m venv vitaldsp_env
      source vitaldsp_env/bin/activate  # On Windows: vitaldsp_env\Scripts\activate

3. **Install dependencies:**
   .. code-block:: bash
   
      pip install -r requirements.txt
      pip install -e .

4. **Run the web application:**
   .. code-block:: bash
   
      python -m vitalDSP_webapp.run_webapp

5. **Access the application:**
   Open your browser and navigate to `http://localhost:8050`

**Development Configuration**

.. code-block:: python

   # config/development.py
   import os
   
   class DevelopmentConfig:
       DEBUG = True
       HOST = 'localhost'
       PORT = 8050
       SECRET_KEY = 'dev-secret-key'
       
       # Database
       DATABASE_URL = 'sqlite:///dev.db'
       
       # Logging
       LOG_LEVEL = 'DEBUG'
       LOG_FILE = 'logs/dev.log'
       
       # File uploads
       MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
       UPLOAD_FOLDER = 'uploads/'
       
       # CORS
       CORS_ORIGINS = ['http://localhost:3000', 'http://localhost:8080']

Staging Deployment
===================

**Staging Environment Setup**

1. **Create staging server:**
   .. code-block:: bash
   
      # Create staging directory
      mkdir -p /opt/vitaldsp/staging
      cd /opt/vitaldsp/staging
      
      # Clone repository
      git clone https://github.com/Oucru-Innovations/vital-DSP.git .
      
      # Create virtual environment
      python -m venv venv
      source venv/bin/activate
      
      # Install dependencies
      pip install -r requirements.txt
      pip install -e .

2. **Configure staging environment:**
   .. code-block:: python

   # config/staging.py
   import os
   
   class StagingConfig:
       DEBUG = False
       HOST = '0.0.0.0'
       PORT = 8050
       SECRET_KEY = os.environ.get('SECRET_KEY', 'staging-secret-key')
       
       # Database
       DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://user:pass@localhost/vitaldsp_staging')
       
       # Logging
       LOG_LEVEL = 'INFO'
       LOG_FILE = '/var/log/vitaldsp/staging.log'
       
       # File uploads
       MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
       UPLOAD_FOLDER = '/opt/vitaldsp/staging/uploads/'
       
       # CORS
       CORS_ORIGINS = ['https://staging.vitaldsp.com']

3. **Create systemd service:**
   .. code-block:: ini

   # /etc/systemd/system/vitaldsp-staging.service
   [Unit]
   Description=VitalDSP Staging Application
   After=network.target
   
   [Service]
   Type=simple
   User=vitaldsp
   Group=vitaldsp
   WorkingDirectory=/opt/vitaldsp/staging
   Environment=PATH=/opt/vitaldsp/staging/venv/bin
   ExecStart=/opt/vitaldsp/staging/venv/bin/python -m vitalDSP_webapp.run_webapp
   Restart=always
   RestartSec=10
   
   [Install]
   WantedBy=multi-user.target

4. **Start the service:**
   .. code-block:: bash
   
      sudo systemctl daemon-reload
      sudo systemctl enable vitaldsp-staging
      sudo systemctl start vitaldsp-staging

Production Deployment
======================

**Production Environment Setup**

1. **Create production server:**
   .. code-block:: bash
   
      # Create production directory
      sudo mkdir -p /opt/vitaldsp/production
      sudo chown vitaldsp:vitaldsp /opt/vitaldsp/production
      cd /opt/vitaldsp/production
      
      # Clone repository
      git clone https://github.com/Oucru-Innovations/vital-DSP.git .
      
      # Create virtual environment
      python -m venv venv
      source venv/bin/activate
      
      # Install production dependencies
      pip install -r requirements.txt
      pip install -e .
      pip install gunicorn

2. **Configure production environment:**
   .. code-block:: python

   # config/production.py
   import os
   
   class ProductionConfig:
       DEBUG = False
       HOST = '0.0.0.0'
       PORT = 8050
       SECRET_KEY = os.environ.get('SECRET_KEY')
       
       # Database
       DATABASE_URL = os.environ.get('DATABASE_URL')
       
       # Logging
       LOG_LEVEL = 'WARNING'
       LOG_FILE = '/var/log/vitaldsp/production.log'
       
       # File uploads
       MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
       UPLOAD_FOLDER = '/opt/vitaldsp/production/uploads/'
       
       # CORS
       CORS_ORIGINS = ['https://vitaldsp.com']
       
       # Security
       SESSION_COOKIE_SECURE = True
       SESSION_COOKIE_HTTPONLY = True
       SESSION_COOKIE_SAMESITE = 'Lax'

3. **Create Gunicorn configuration:**
   .. code-block:: python

   # gunicorn.conf.py
   import multiprocessing
   
   # Server socket
   bind = "0.0.0.0:8050"
   backlog = 2048
   
   # Worker processes
   workers = multiprocessing.cpu_count() * 2 + 1
   worker_class = "sync"
   worker_connections = 1000
   timeout = 30
   keepalive = 2
   
   # Restart workers after this many requests, to help prevent memory leaks
   max_requests = 1000
   max_requests_jitter = 50
   
   # Logging
   accesslog = "/var/log/vitaldsp/access.log"
   errorlog = "/var/log/vitaldsp/error.log"
   loglevel = "info"
   access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'
   
   # Process naming
   proc_name = 'vitaldsp'
   
   # Server mechanics
   daemon = False
   pidfile = '/var/run/vitaldsp.pid'
   user = 'vitaldsp'
   group = 'vitaldsp'
   tmp_upload_dir = None
   
   # SSL
   keyfile = '/etc/ssl/private/vitaldsp.key'
   certfile = '/etc/ssl/certs/vitaldsp.crt'

4. **Create systemd service:**
   .. code-block:: ini

   # /etc/systemd/system/vitaldsp.service
   [Unit]
   Description=VitalDSP Production Application
   After=network.target
   
   [Service]
   Type=simple
   User=vitaldsp
   Group=vitaldsp
   WorkingDirectory=/opt/vitaldsp/production
   Environment=PATH=/opt/vitaldsp/production/venv/bin
   ExecStart=/opt/vitaldsp/production/venv/bin/gunicorn --config gunicorn.conf.py vitalDSP_webapp.app:app
   Restart=always
   RestartSec=10
   
   [Install]
   WantedBy=multi-user.target

5. **Configure Nginx:**
   .. code-block:: nginx

   # /etc/nginx/sites-available/vitaldsp
   server {
       listen 80;
       server_name vitaldsp.com www.vitaldsp.com;
       return 301 https://$server_name$request_uri;
   }
   
   server {
       listen 443 ssl http2;
       server_name vitaldsp.com www.vitaldsp.com;
   
       ssl_certificate /etc/ssl/certs/vitaldsp.crt;
       ssl_certificate_key /etc/ssl/private/vitaldsp.key;
   
       ssl_protocols TLSv1.2 TLSv1.3;
       ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
       ssl_prefer_server_ciphers off;
   
       location / {
           proxy_pass http://127.0.0.1:8050;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   
       location /static {
           alias /opt/vitaldsp/production/static;
           expires 1y;
           add_header Cache-Control "public, immutable";
       }
   
       location /uploads {
           alias /opt/vitaldsp/production/uploads;
           expires 1d;
           add_header Cache-Control "public";
       }
   }

6. **Start the services:**
   .. code-block:: bash
   
      # Enable and start Nginx
      sudo systemctl enable nginx
      sudo systemctl start nginx
      
      # Enable and start VitalDSP
      sudo systemctl daemon-reload
      sudo systemctl enable vitaldsp
      sudo systemctl start vitaldsp

Cloud Deployment
=================

**AWS Deployment**

1. **Create EC2 instance:**
   .. code-block:: bash
   
      # Launch EC2 instance (Ubuntu 20.04 LTS)
      # Instance type: t3.medium or larger
      # Security group: Allow HTTP (80), HTTPS (443), SSH (22)

2. **Install dependencies:**
   .. code-block:: bash
   
      # Update system
      sudo apt update && sudo apt upgrade -y
      
      # Install Python and dependencies
      sudo apt install python3 python3-pip python3-venv nginx git -y
      
      # Install SSL certificate
      sudo apt install certbot python3-certbot-nginx -y

3. **Deploy application:**
   .. code-block:: bash
   
      # Create application directory
      sudo mkdir -p /opt/vitaldsp
      sudo chown ubuntu:ubuntu /opt/vitaldsp
      cd /opt/vitaldsp
      
      # Clone repository
      git clone https://github.com/Oucru-Innovations/vital-DSP.git .
      
      # Create virtual environment
      python3 -m venv venv
      source venv/bin/activate
      
      # Install dependencies
      pip install -r requirements.txt
      pip install -e .
      pip install gunicorn

4. **Configure SSL:**
   .. code-block:: bash
   
      # Get SSL certificate
      sudo certbot --nginx -d vitaldsp.com -d www.vitaldsp.com

**Google Cloud Platform Deployment**

1. **Create Compute Engine instance:**
   .. code-block:: bash
   
      # Create instance
      gcloud compute instances create vitaldsp-server \
          --image-family=ubuntu-2004-lts \
          --image-project=ubuntu-os-cloud \
          --machine-type=e2-medium \
          --zone=us-central1-a \
          --tags=http-server,https-server

2. **Deploy application:**
   .. code-block:: bash
   
      # SSH into instance
      gcloud compute ssh vitaldsp-server
      
      # Follow AWS deployment steps 2-4

**Azure Deployment**

1. **Create Virtual Machine:**
   .. code-block:: bash
   
      # Create VM
      az vm create \
          --resource-group vitaldsp-rg \
          --name vitaldsp-vm \
          --image UbuntuLTS \
          --size Standard_B2s \
          --admin-username azureuser \
          --generate-ssh-keys

2. **Deploy application:**
   .. code-block:: bash
   
      # SSH into VM
      ssh azureuser@<vm-ip>
      
      # Follow AWS deployment steps 2-4

Containerized Deployment
=========================

**Docker Deployment**

1. **Create Dockerfile:**
   .. code-block:: dockerfile

   # Dockerfile
   FROM python:3.9-slim
   
   # Set working directory
   WORKDIR /app
   
   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       gcc \
       g++ \
       && rm -rf /var/lib/apt/lists/*
   
   # Copy requirements
   COPY requirements.txt .
   
   # Install Python dependencies
   RUN pip install --no-cache-dir -r requirements.txt
   
   # Copy application code
   COPY . .
   
   # Install application
   RUN pip install -e .
   
   # Create non-root user
   RUN useradd -m -u 1000 vitaldsp && chown -R vitaldsp:vitaldsp /app
   USER vitaldsp
   
   # Expose port
   EXPOSE 8050
   
   # Run application
   CMD ["python", "-m", "vitalDSP_webapp.run_webapp", "--host", "0.0.0.0", "--port", "8050"]

2. **Create docker-compose.yml:**
   .. code-block:: yaml

   # docker-compose.yml
   version: '3.8'
   
   services:
     vitaldsp:
       build: .
       ports:
         - "8050:8050"
       environment:
         - FLASK_ENV=production
         - SECRET_KEY=your-secret-key
       volumes:
         - ./uploads:/app/uploads
         - ./logs:/app/logs
       restart: unless-stopped
   
     nginx:
       image: nginx:alpine
       ports:
         - "80:80"
         - "443:443"
       volumes:
         - ./nginx.conf:/etc/nginx/nginx.conf
         - ./ssl:/etc/ssl
       depends_on:
         - vitaldsp
       restart: unless-stopped

3. **Build and run:**
   .. code-block:: bash
   
      # Build image
      docker build -t vitaldsp .
      
      # Run with docker-compose
      docker-compose up -d

**Kubernetes Deployment**

1. **Create deployment.yaml:**
   .. code-block:: yaml

   # deployment.yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: vitaldsp
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: vitaldsp
     template:
       metadata:
         labels:
           app: vitaldsp
       spec:
         containers:
         - name: vitaldsp
           image: vitaldsp:latest
           ports:
           - containerPort: 8050
           env:
           - name: FLASK_ENV
             value: "production"
           - name: SECRET_KEY
             valueFrom:
               secretKeyRef:
                 name: vitaldsp-secrets
                 key: secret-key
           resources:
             requests:
               memory: "512Mi"
               cpu: "250m"
             limits:
               memory: "1Gi"
               cpu: "500m"
   
   ---
   apiVersion: v1
   kind: Service
   metadata:
     name: vitaldsp-service
   spec:
     selector:
       app: vitaldsp
     ports:
     - port: 80
       targetPort: 8050
     type: LoadBalancer

2. **Deploy to Kubernetes:**
   .. code-block:: bash
   
      # Apply deployment
      kubectl apply -f deployment.yaml
      
      # Check status
      kubectl get pods
      kubectl get services

Edge Deployment
================

**Raspberry Pi Deployment**

1. **Prepare Raspberry Pi:**
   .. code-block:: bash
   
      # Update system
      sudo apt update && sudo apt upgrade -y
      
      # Install Python and dependencies
      sudo apt install python3 python3-pip python3-venv git -y

2. **Deploy application:**
   .. code-block:: bash
   
      # Create application directory
      mkdir -p /home/pi/vitaldsp
      cd /home/pi/vitaldsp
      
      # Clone repository
      git clone https://github.com/Oucru-Innovations/vital-DSP.git .
      
      # Create virtual environment
      python3 -m venv venv
      source venv/bin/activate
      
      # Install dependencies
      pip install -r requirements.txt
      pip install -e .

3. **Create startup script:**
   .. code-block:: bash

   # startup.sh
   #!/bin/bash
   cd /home/pi/vitaldsp
   source venv/bin/activate
   python -m vitalDSP_webapp.run_webapp --host 0.0.0.0 --port 8050

4. **Configure auto-start:**
   .. code-block:: bash
   
      # Add to crontab
      crontab -e
      # Add: @reboot /home/pi/vitaldsp/startup.sh

**IoT Device Deployment**

1. **Optimize for resource constraints:**
   .. code-block:: python

   # config/iot.py
   import os
   
   class IoTConfig:
       DEBUG = False
       HOST = '0.0.0.0'
       PORT = 8050
       SECRET_KEY = os.environ.get('SECRET_KEY', 'iot-secret-key')
       
       # Optimize for low memory
       MAX_CONTENT_LENGTH = 4 * 1024 * 1024  # 4MB
       
       # Disable features not needed on IoT
       ENABLE_ADVANCED_FEATURES = False
       ENABLE_MACHINE_LEARNING = False
       
       # Use lightweight processing
       USE_LIGHTWEIGHT_FILTERS = True
       MAX_SIGNAL_LENGTH = 10000  # 10k samples max

2. **Create lightweight version:**
   .. code-block:: python

   # iot_app.py
   from vitalDSP_webapp.app import create_app
   
   app = create_app('iot')
   
   if __name__ == '__main__':
       app.run(host='0.0.0.0', port=8050, debug=False)

Monitoring and Maintenance
===========================

**Health Checks**

1. **Create health check endpoint:**
   .. code-block:: python

   # health_check.py
   from flask import Flask, jsonify
   import psutil
   import os
   
   app = Flask(__name__)
   
   @app.route('/health')
   def health_check():
       """Health check endpoint."""
       
       # Check system resources
       cpu_percent = psutil.cpu_percent()
       memory = psutil.virtual_memory()
       disk = psutil.disk_usage('/')
       
       # Check application status
       status = 'healthy'
       if cpu_percent > 90:
           status = 'unhealthy'
       if memory.percent > 90:
           status = 'unhealthy'
       if disk.percent > 90:
           status = 'unhealthy'
       
       return jsonify({
           'status': status,
           'cpu_percent': cpu_percent,
           'memory_percent': memory.percent,
           'disk_percent': disk.percent
       })

2. **Configure monitoring:**
   .. code-block:: bash
   
      # Add to nginx.conf
      location /health {
          proxy_pass http://127.0.0.1:8050/health;
      }

**Logging**

1. **Configure logging:**
   .. code-block:: python

   # logging_config.py
   import logging
   import logging.handlers
   import os
   
   def setup_logging():
       """Setup application logging."""
       
       # Create logs directory
       os.makedirs('/var/log/vitaldsp', exist_ok=True)
       
       # Configure logging
       logging.basicConfig(
           level=logging.INFO,
           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
           handlers=[
               logging.handlers.RotatingFileHandler(
                   '/var/log/vitaldsp/app.log',
                   maxBytes=10*1024*1024,  # 10MB
                   backupCount=5
               ),
               logging.StreamHandler()
           ]
       )

**Backup and Recovery**

1. **Create backup script:**
   .. code-block:: bash

   # backup.sh
   #!/bin/bash
   
   BACKUP_DIR="/opt/backups/vitaldsp"
   DATE=$(date +%Y%m%d_%H%M%S)
   
   # Create backup directory
   mkdir -p $BACKUP_DIR
   
   # Backup application data
   tar -czf $BACKUP_DIR/vitaldsp_$DATE.tar.gz /opt/vitaldsp/production
   
   # Backup database
   pg_dump vitaldsp > $BACKUP_DIR/database_$DATE.sql
   
   # Clean old backups (keep last 7 days)
   find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
   find $BACKUP_DIR -name "*.sql" -mtime +7 -delete

2. **Schedule backups:**
   .. code-block:: bash
   
      # Add to crontab
      crontab -e
      # Add: 0 2 * * * /opt/vitaldsp/backup.sh

**Updates and Maintenance**

1. **Create update script:**
   .. code-block:: bash

   # update.sh
   #!/bin/bash
   
   cd /opt/vitaldsp/production
   
   # Backup current version
   tar -czf ../backup_$(date +%Y%m%d_%H%M%S).tar.gz .
   
   # Pull latest changes
   git pull origin main
   
   # Update dependencies
   source venv/bin/activate
   pip install -r requirements.txt
   pip install -e .
   
   # Restart services
   sudo systemctl restart vitaldsp

2. **Schedule updates:**
   .. code-block:: bash
   
      # Add to crontab
      crontab -e
      # Add: 0 3 * * 0 /opt/vitaldsp/update.sh

Security Considerations
========================

**SSL/TLS Configuration**

1. **Generate SSL certificates:**
   .. code-block:: bash
   
      # Generate private key
      openssl genrsa -out vitaldsp.key 2048
      
      # Generate certificate
      openssl req -new -x509 -key vitaldsp.key -out vitaldsp.crt -days 365

2. **Configure HTTPS:**
   .. code-block:: nginx

   # nginx.conf
   server {
       listen 443 ssl;
       server_name vitaldsp.com;
       
       ssl_certificate /etc/ssl/certs/vitaldsp.crt;
       ssl_certificate_key /etc/ssl/private/vitaldsp.key;
       
       ssl_protocols TLSv1.2 TLSv1.3;
       ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
       ssl_prefer_server_ciphers off;
   }

**Access Control**

1. **Configure firewall:**
   .. code-block:: bash
   
      # Allow only necessary ports
      sudo ufw allow 22    # SSH
      sudo ufw allow 80     # HTTP
      sudo ufw allow 443    # HTTPS
      sudo ufw enable

2. **Implement authentication:**
   .. code-block:: python

   # auth.py
   from flask import Flask, request, jsonify
   import jwt
   import datetime
   
   def generate_token(user_id):
       """Generate JWT token."""
       payload = {
           'user_id': user_id,
           'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
       }
       return jwt.encode(payload, SECRET_KEY, algorithm='HS256')
   
   def verify_token(token):
       """Verify JWT token."""
       try:
           payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
           return payload['user_id']
       except jwt.ExpiredSignatureError:
           return None
       except jwt.InvalidTokenError:
           return None

**Data Protection**

1. **Encrypt sensitive data:**
   .. code-block:: python

   # encryption.py
   from cryptography.fernet import Fernet
   
   def encrypt_data(data):
       """Encrypt sensitive data."""
       key = Fernet.generate_key()
       f = Fernet(key)
       encrypted = f.encrypt(data.encode())
       return encrypted, key
   
   def decrypt_data(encrypted_data, key):
       """Decrypt sensitive data."""
       f = Fernet(key)
       decrypted = f.decrypt(encrypted_data)
       return decrypted.decode()

2. **Secure file uploads:**
   .. code-block:: python

   # file_upload.py
   import os
   import uuid
   
   def secure_file_upload(file):
       """Securely handle file uploads."""
       
       # Validate file type
       allowed_extensions = {'csv', 'xlsx', 'json'}
       if not file.filename.lower().endswith(tuple(allowed_extensions)):
           raise ValueError('Invalid file type')
       
       # Generate secure filename
       filename = str(uuid.uuid4()) + '.' + file.filename.split('.')[-1]
       
       # Save to secure location
       file_path = os.path.join(UPLOAD_FOLDER, filename)
       file.save(file_path)
       
       return file_path

This deployment guide provides comprehensive instructions for deploying VitalDSP in various environments. Choose the deployment method that best fits your requirements and infrastructure.

For additional support with deployment, consult our support team or check the GitHub issues page.
