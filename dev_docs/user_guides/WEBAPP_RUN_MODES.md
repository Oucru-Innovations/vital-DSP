# 🚀 VitalDSP Webapp - Run Modes Guide

This guide explains how to run the VitalDSP webapp in different modes for development and production use.

## 📋 Available Run Methods

### 1. **Enhanced Debug Runner** (Recommended)
Use the new `run_webapp_debug.py` script with flexible options:

```bash
# Normal mode (INFO logging)
python src/vitalDSP_webapp/run_webapp_debug.py

# Debug mode (DEBUG logging)
python src/vitalDSP_webapp/run_webapp_debug.py --debug

# Custom port
python src/vitalDSP_webapp/run_webapp_debug.py --port 8080

# Debug mode on custom port
python src/vitalDSP_webapp/run_webapp_debug.py --debug --port 8080

# Custom host
python src/vitalDSP_webapp/run_webapp_debug.py --host 127.0.0.1

# Show help
python src/vitalDSP_webapp/run_webapp_debug.py --help
```

### 2. **Interactive Scripts** (Easy Mode)
Use the provided scripts for interactive mode selection:

**Windows:**
```cmd
run_webapp.bat
```

**Linux/Mac:**
```bash
./run_webapp.sh
```

### 3. **Original Runner** (Basic)
Use the original runner for basic functionality:

```bash
python src/vitalDSP_webapp/run_webapp.py
```

## 🔍 Debug Mode vs Normal Mode

### **🔵 Normal Mode (Default)**
- **Logging Level**: INFO and above
- **Performance**: Optimized for production
- **Log Files**: `webapp.log`
- **Auto-reload**: Disabled
- **Use Case**: Production, monitoring, general use

**What you'll see:**
```
🚀 NORMAL MODE: Essential logs only (INFO level)
📝 Logs will be saved to: webapp.log
🌐 Starting webapp on 0.0.0.0:8000
📊 Mode: NORMAL
🔗 Access at: http://localhost:8000
🔗 API docs at: http://localhost:8000/docs
```

### **🔍 Debug Mode**
- **Logging Level**: DEBUG and above (all logs)
- **Performance**: Detailed logging may impact performance
- **Log Files**: `webapp_debug.log`
- **Auto-reload**: Enabled (code changes auto-restart server)
- **Use Case**: Development, troubleshooting, detailed analysis

**What you'll see:**
```
🔍 DEBUG MODE: All logs enabled (DEBUG level)
📝 Debug logs will be saved to: webapp_debug.log
🌐 Starting webapp on 0.0.0.0:8000
📊 Mode: DEBUG
🔗 Access at: http://localhost:8000
🔗 API docs at: http://localhost:8000/docs
```

## 📊 Enhanced Data Service Logging

### **Normal Mode Logs:**
```
INFO:src.vitalDSP_webapp.services.data.enhanced_data_service:Data stored with ID: data_1
INFO:src.vitalDSP_webapp.services.data.enhanced_data_service:All data cleared
```

### **Debug Mode Logs:**
```
DEBUG:src.vitalDSP_webapp.services.data.enhanced_data_service:=== STORING DATA (Enhanced Service) ===
DEBUG:src.vitalDSP_webapp.services.data.enhanced_data_service:Data ID: data_1
DEBUG:src.vitalDSP_webapp.services.data.enhanced_data_service:Data shape: (1000, 2)
DEBUG:src.vitalDSP_webapp.services.data.enhanced_data_service:Data columns: ['time', 'signal']
DEBUG:src.vitalDSP_webapp.services.data.enhanced_data_service:Auto-detecting columns...
DEBUG:src.vitalDSP_webapp.services.data.enhanced_data_service:Found signal column: signal
DEBUG:src.vitalDSP_webapp.services.data.enhanced_data_service:Auto-detected column mapping: {'time': 'time', 'signal': 'signal'}
INFO:src.vitalDSP_webapp.services.data.enhanced_data_service:Data stored with ID: data_1
DEBUG:src.vitalDSP_webapp.services.data.enhanced_data_service:Column mapping: {'time': 'time', 'signal': 'signal'}
```

## 🛠️ Development Workflow

### **For Development:**
```bash
# Start in debug mode for detailed logging
python src/vitalDSP_webapp/run_webapp_debug.py --debug

# Or use interactive script
./run_webapp.sh
# Choose option 2 (Debug Mode)
```

### **For Testing:**
```bash
# Test with different ports
python src/vitalDSP_webapp/run_webapp_debug.py --debug --port 8080
python src/vitalDSP_webapp/run_webapp_debug.py --debug --port 9000
```

### **For Production:**
```bash
# Normal mode for production
python src/vitalDSP_webapp/run_webapp_debug.py

# Or use original runner
python src/vitalDSP_webapp/run_webapp.py
```

## 📁 Log Files

### **Normal Mode:**
- **File**: `webapp.log`
- **Content**: Essential operations only
- **Size**: Smaller, focused logs

### **Debug Mode:**
- **File**: `webapp_debug.log`
- **Content**: All operations with details
- **Size**: Larger, comprehensive logs

## 🔧 Environment Variables

You can also use environment variables:

```bash
# Set debug mode via environment
export DEBUG=true
python src/vitalDSP_webapp/run_webapp_debug.py

# Set custom port
export PORT=8080
python src/vitalDSP_webapp/run_webapp_debug.py

# Set custom host
export HOST=127.0.0.1
python src/vitalDSP_webapp/run_webapp_debug.py
```

## 🚨 Troubleshooting

### **If you see too many logs:**
- Use Normal mode: `python src/vitalDSP_webapp/run_webapp_debug.py`
- Or set logging level manually

### **If you need detailed information:**
- Use Debug mode: `python src/vitalDSP_webapp/run_webapp_debug.py --debug`
- Check `webapp_debug.log` file

### **If port is already in use:**
- Use different port: `python src/vitalDSP_webapp/run_webapp_debug.py --port 8080`
- Or kill the process using the port

## 🎯 Quick Start Examples

```bash
# Quick start - Normal mode
python src/vitalDSP_webapp/run_webapp_debug.py

# Quick start - Debug mode
python src/vitalDSP_webapp/run_webapp_debug.py --debug

# Quick start - Interactive (Windows)
run_webapp.bat

# Quick start - Interactive (Linux/Mac)
./run_webapp.sh
```

## 📈 Performance Notes

- **Normal Mode**: Optimized for performance, minimal logging overhead
- **Debug Mode**: More detailed logging, may impact performance with large datasets
- **Auto-reload**: Only enabled in debug mode, useful for development
- **Log Files**: Automatically created and managed

## 🔍 What to Look For in Logs

### **Normal Mode - Key Events:**
- Data storage operations
- Service initialization
- Error conditions
- Configuration changes

### **Debug Mode - Detailed Information:**
- Column detection process
- Data shape and structure
- Filtered data operations
- Memory usage patterns
- Performance metrics

## 🚀 Advanced Usage

### **Multiple Instances:**
```bash
# Run multiple instances on different ports
python src/vitalDSP_webapp/run_webapp_debug.py --port 8000 &
python src/vitalDSP_webapp/run_webapp_debug.py --port 8001 &
python src/vitalDSP_webapp/run_webapp_debug.py --port 8002 &
```

### **Custom Configuration:**
```bash
# Debug mode with custom host and port
python src/vitalDSP_webapp/run_webapp_debug.py --debug --host 127.0.0.1 --port 8080
```

### **Log Analysis:**
```bash
# Monitor logs in real-time (Linux/Mac)
tail -f webapp_debug.log

# Search for specific patterns
grep "Data stored" webapp_debug.log
grep "ERROR" webapp.log
```

---

**Happy coding! 🚀**
