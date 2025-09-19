#!/bin/bash

# vitalDSP Webapp Deployment Script
# This script helps deploy the vitalDSP webapp to a server using Docker

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="vitaldsp-webapp"
DOCKER_IMAGE="vitaldsp-webapp"
DOCKER_TAG="latest"
PORT=${PORT:-8000}

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    log_success "Docker and Docker Compose are available"
}

# Create necessary directories
create_directories() {
    log_info "Creating necessary directories..."
    mkdir -p uploads logs tmp ssl
    chmod 755 uploads logs tmp
    log_success "Directories created"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    docker build -f Dockerfile.production -t ${DOCKER_IMAGE}:${DOCKER_TAG} .
    log_success "Docker image built successfully"
}

# Stop existing containers
stop_containers() {
    log_info "Stopping existing containers..."
    docker-compose down 2>/dev/null || true
    log_success "Existing containers stopped"
}

# Start the application
start_application() {
    log_info "Starting vitalDSP webapp..."
    docker-compose up -d
    log_success "Application started"
}

# Check application health
check_health() {
    log_info "Checking application health..."
    
    # Wait for application to start
    sleep 10
    
    # Check if container is running
    if docker-compose ps | grep -q "Up"; then
        log_success "Container is running"
    else
        log_error "Container failed to start"
        docker-compose logs
        exit 1
    fi
    
    # Check health endpoint
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:${PORT}/api/health >/dev/null 2>&1; then
            log_success "Application is healthy and responding"
            return 0
        fi
        
        log_info "Health check attempt $attempt/$max_attempts..."
        sleep 2
        ((attempt++))
    done
    
    log_error "Application health check failed after $max_attempts attempts"
    docker-compose logs
    exit 1
}

# Show application status
show_status() {
    log_info "Application Status:"
    echo "===================="
    docker-compose ps
    echo ""
    log_info "Application URLs:"
    echo "  Main App: http://localhost:${PORT}"
    echo "  Health Check: http://localhost:${PORT}/api/health"
    echo ""
    log_info "To view logs: docker-compose logs -f"
    log_info "To stop: docker-compose down"
}

# Main deployment function
deploy() {
    log_info "Starting vitalDSP webapp deployment..."
    
    check_docker
    create_directories
    build_image
    stop_containers
    start_application
    check_health
    show_status
    
    log_success "Deployment completed successfully!"
}

# Update function
update() {
    log_info "Updating vitalDSP webapp..."
    
    check_docker
    build_image
    stop_containers
    start_application
    check_health
    show_status
    
    log_success "Update completed successfully!"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up Docker resources..."
    
    docker-compose down
    docker system prune -f
    docker volume prune -f
    
    log_success "Cleanup completed"
}

# Show help
show_help() {
    echo "vitalDSP Webapp Deployment Script"
    echo "================================="
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  deploy    Deploy the application (default)"
    echo "  update    Update the application"
    echo "  stop      Stop the application"
    echo "  restart   Restart the application"
    echo "  logs      Show application logs"
    echo "  status    Show application status"
    echo "  cleanup   Clean up Docker resources"
    echo "  help      Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  PORT      Port to run the application on (default: 8000)"
    echo ""
}

# Main script logic
case "${1:-deploy}" in
    deploy)
        deploy
        ;;
    update)
        update
        ;;
    stop)
        log_info "Stopping application..."
        docker-compose down
        log_success "Application stopped"
        ;;
    restart)
        log_info "Restarting application..."
        docker-compose restart
        check_health
        show_status
        log_success "Application restarted"
        ;;
    logs)
        docker-compose logs -f
        ;;
    status)
        show_status
        ;;
    cleanup)
        cleanup
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
