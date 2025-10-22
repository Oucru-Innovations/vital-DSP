#!/bin/bash
# Linux/Mac shell script for running VitalDSP webapp in different modes

echo "========================================"
echo "    VitalDSP Webapp Runner"
echo "========================================"
echo ""
echo "Choose mode:"
echo "1. Normal Mode (INFO logging)"
echo "2. Debug Mode (DEBUG logging)"
echo "3. Custom Mode (specify options)"
echo ""
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "Starting in NORMAL mode..."
        python src/vitalDSP_webapp/run_webapp_debug.py
        ;;
    2)
        echo ""
        echo "Starting in DEBUG mode..."
        python src/vitalDSP_webapp/run_webapp_debug.py --debug
        ;;
    3)
        echo ""
        echo "Available options:"
        echo "  --debug    Enable debug mode"
        echo "  --port     Specify port (e.g., --port 8080)"
        echo "  --host     Specify host (e.g., --host 127.0.0.1)"
        echo ""
        read -p "Enter custom options: " custom
        echo ""
        echo "Starting with custom options: $custom"
        python src/vitalDSP_webapp/run_webapp_debug.py $custom
        ;;
    *)
        echo "Invalid choice. Please run the script again."
        exit 1
        ;;
esac
