#!/bin/bash

# Health check script for vitalDSP webapp
# This script can be used for external monitoring

set -e

# Configuration
HOST=${HOST:-localhost}
PORT=${PORT:-8000}
TIMEOUT=${TIMEOUT:-10}

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Functions
log_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if curl is available
if ! command -v curl &> /dev/null; then
    log_error "curl is not installed"
    exit 1
fi

# Health check function
check_health() {
    local url="http://${HOST}:${PORT}/api/health"
    local response
    
    log_info "Checking health at ${url}"
    
    # Make the request with timeout
    if response=$(curl -s -w "%{http_code}" --max-time ${TIMEOUT} "${url}" 2>/dev/null); then
        local http_code="${response: -3}"
        local body="${response%???}"
        
        if [ "${http_code}" = "200" ]; then
            log_success "Health check passed (HTTP ${http_code})"
            echo "Response: ${body}"
            return 0
        else
            log_error "Health check failed (HTTP ${http_code})"
            echo "Response: ${body}"
            return 1
        fi
    else
        log_error "Health check failed - connection timeout or error"
        return 1
    fi
}

# Main execution
main() {
    local exit_code=0
    
    # Check if host is reachable
    if ! ping -c 1 -W 1 "${HOST}" >/dev/null 2>&1; then
        log_error "Host ${HOST} is not reachable"
        exit 1
    fi
    
    # Perform health check
    if check_health; then
        log_success "vitalDSP webapp is healthy"
    else
        log_error "vitalDSP webapp is unhealthy"
        exit_code=1
    fi
    
    exit ${exit_code}
}

# Show help
show_help() {
    echo "vitalDSP Webapp Health Check Script"
    echo "==================================="
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --host HOST     Host to check (default: localhost)"
    echo "  -p, --port PORT     Port to check (default: 8000)"
    echo "  -t, --timeout SEC   Timeout in seconds (default: 10)"
    echo "  --help              Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  HOST                Host to check"
    echo "  PORT                Port to check"
    echo "  TIMEOUT             Timeout in seconds"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Check localhost:8000"
    echo "  $0 -h example.com -p 8080            # Check example.com:8080"
    echo "  HOST=example.com PORT=8080 $0        # Using environment variables"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--host)
            HOST="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Run main function
main
