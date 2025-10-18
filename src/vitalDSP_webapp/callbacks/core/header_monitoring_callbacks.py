"""
Header Monitoring Callbacks for vitalDSP Webapp

This module handles callbacks for monitoring system status in the header,
including memory usage, active tasks, and system health.
"""

from dash.dependencies import Input, Output, State
from dash import no_update
import logging
import psutil
import os
from datetime import datetime

# Import services
from vitalDSP_webapp.services.progress_tracker import get_progress_tracker

logger = logging.getLogger(__name__)


def register_header_monitoring_callbacks(app):
    """
    Register header monitoring callbacks.
    
    Parameters
    ----------
    app : Dash
        The Dash application instance
    """
    
    @app.callback(
        [
            Output("memory-usage-badge", "children"),
            Output("memory-usage-badge", "color"),
            Output("active-tasks-badge", "children"),
            Output("active-tasks-badge", "color"),
            Output("system-status-badge", "children"),
            Output("system-status-badge", "color"),
        ],
        [
            Input("header-monitor-interval", "n_intervals"),
        ]
    )
    def update_header_monitoring(n_intervals):
        """
        Update header monitoring indicators.
        """
        try:
            # Get memory usage
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent
            memory_used_gb = memory_info.used / (1024**3)
            memory_total_gb = memory_info.total / (1024**3)
            
            # Determine memory status color
            if memory_percent < 70:
                memory_color = "success"
            elif memory_percent < 85:
                memory_color = "warning"
            else:
                memory_color = "danger"
            
            memory_text = f"Memory: {memory_percent:.1f}% ({memory_used_gb:.1f}GB/{memory_total_gb:.1f}GB)"
            
            # Get active tasks count
            progress_tracker = get_progress_tracker()
            all_tasks = progress_tracker.get_all_active_tasks()
            running_tasks = sum(1 for task in all_tasks if task.status.value == 'running')
            
            # Determine tasks status color
            if running_tasks == 0:
                tasks_color = "info"
            elif running_tasks < 3:
                tasks_color = "warning"
            else:
                tasks_color = "danger"
            
            tasks_text = f"Tasks: {running_tasks}"
            
            # Get system status
            cpu_percent = psutil.cpu_percent(interval=1)
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            
            # Determine system status
            if cpu_percent < 80 and memory_percent < 85 and disk_percent < 90:
                system_status = "Ready"
                system_color = "success"
            elif cpu_percent < 95 and memory_percent < 95 and disk_percent < 95:
                system_status = "Busy"
                system_color = "warning"
            else:
                system_status = "Overloaded"
                system_color = "danger"
            
            system_text = f"System: {system_status}"
            
            return (
                memory_text,
                memory_color,
                tasks_text,
                tasks_color,
                system_text,
                system_color
            )
            
        except Exception as e:
            logger.error(f"Error updating header monitoring: {e}")
            return (
                "Memory: Error",
                "danger",
                "Tasks: Error",
                "danger",
                "System: Error",
                "danger"
            )
    
    @app.callback(
        Output("memory-usage-store", "data"),
        [
            Input("header-monitor-interval", "n_intervals"),
        ]
    )
    def update_memory_store(n_intervals):
        """
        Update memory usage store with detailed information.
        """
        try:
            memory_info = psutil.virtual_memory()
            swap_info = psutil.swap_memory()
            
            return {
                "timestamp": datetime.now().isoformat(),
                "memory": {
                    "total": memory_info.total,
                    "available": memory_info.available,
                    "used": memory_info.used,
                    "percent": memory_info.percent,
                    "free": memory_info.free
                },
                "swap": {
                    "total": swap_info.total,
                    "used": swap_info.used,
                    "free": swap_info.free,
                    "percent": swap_info.percent
                },
                "cpu_percent": psutil.cpu_percent(interval=1),
                "disk_usage": {
                    "total": psutil.disk_usage('/').total,
                    "used": psutil.disk_usage('/').used,
                    "free": psutil.disk_usage('/').free,
                    "percent": (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100
                }
            }
            
        except Exception as e:
            logger.error(f"Error updating memory store: {e}")
            return {"error": str(e)}
    
    @app.callback(
        Output("system-status-store", "data"),
        [
            Input("header-monitor-interval", "n_intervals"),
        ]
    )
    def update_system_status_store(n_intervals):
        """
        Update system status store with comprehensive system information.
        """
        try:
            # Get system information
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            uptime = datetime.now() - boot_time
            
            # Get process information
            processes = psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent'])
            process_count = len(list(processes))
            
            # Get network information
            network_io = psutil.net_io_counters()
            
            return {
                "timestamp": datetime.now().isoformat(),
                "uptime": {
                    "boot_time": boot_time.isoformat(),
                    "uptime_seconds": uptime.total_seconds(),
                    "uptime_formatted": str(uptime).split('.')[0]  # Remove microseconds
                },
                "processes": {
                    "count": process_count,
                    "cpu_percent": psutil.cpu_percent(interval=1),
                    "memory_percent": psutil.virtual_memory().percent
                },
                "network": {
                    "bytes_sent": network_io.bytes_sent,
                    "bytes_recv": network_io.bytes_recv,
                    "packets_sent": network_io.packets_sent,
                    "packets_recv": network_io.packets_recv
                },
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            }
            
        except Exception as e:
            logger.error(f"Error updating system status store: {e}")
            return {"error": str(e)}
    
    logger.info("Header monitoring callbacks registered successfully")
