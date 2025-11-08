@echo off
REM Windows batch script for running VitalDSP webapp in different modes

echo ========================================
echo    VitalDSP Webapp Runner v2.0
echo ========================================
echo.
echo Choose mode:
echo 1. Normal Mode (Standard logging)
echo 2. Debug Mode (Debug logging + auto-reload)
echo 3. Production Mode (Optimized, minimal logs)
echo 4. Custom Mode (specify options)
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo Starting in NORMAL mode...
    python src/vitalDSP_webapp/run_webapp.py
) else if "%choice%"=="2" (
    echo.
    echo Starting in DEBUG mode...
    python src/vitalDSP_webapp/run_webapp.py --debug
) else if "%choice%"=="3" (
    echo.
    echo Starting in PRODUCTION mode...
    python src/vitalDSP_webapp/run_webapp.py --production
) else if "%choice%"=="4" (
    echo.
    echo Available options:
    echo   --debug          Enable debug mode
    echo   --production     Enable production mode
    echo   --port PORT      Specify port (e.g., --port 8080)
    echo   --host HOST      Specify host (e.g., --host 127.0.0.1)
    echo   --reload         Enable auto-reload
    echo.
    set /p custom="Enter custom options: "
    echo.
    echo Starting with custom options: %custom%
    python src/vitalDSP_webapp/run_webapp.py %custom%
) else (
    echo Invalid choice. Please run the script again.
    pause
    exit /b 1
)

pause
