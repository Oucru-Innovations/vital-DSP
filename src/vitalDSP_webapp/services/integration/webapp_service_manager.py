"""
Webapp Service Manager for vitalDSP

This module provides the unified service management layer that orchestrates all
webapp services for optimal performance and reliability:

- Enhanced Data Service: Chunked loading, memory mapping, progressive loading
- Task Queue System: Async processing with Redis backend
- WebSocket Manager: Real-time communication with frontend

Features:
- Unified service interface for webapp components
- Automatic service initialization and lifecycle management
- Integration between data loading, task processing, and real-time updates
- Performance monitoring and health checks
- Service orchestration and coordination

Author: vitalDSP Development Team
Date: January 11, 2025
"""

import os
import sys
import time
import logging
import threading
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import json
import pandas as pd

# Add vitalDSP to path for imports
from pathlib import Path

current_dir = Path(__file__).parent
vitaldsp_path = current_dir.parent.parent.parent / "vitalDSP"
if str(vitaldsp_path) not in sys.path:
    sys.path.insert(0, str(vitaldsp_path))

# Import Phase 3A components
try:
    from ..data.enhanced_data_service import (
        EnhancedDataService,
        get_enhanced_data_service,
        LoadingProgress,
        DataSegment,
    )
    from ..async_services.task_queue import (
        WebappTaskQueue,
        TaskProcessor,
        get_task_queue,
        get_task_processor,
        start_task_processing,
        stop_task_processing,
        TaskStatus,
        TaskPriority,
    )
    from ..async_services.websocket_manager import (
        WebSocketManager,
        get_websocket_manager,
        start_websocket_manager,
        stop_websocket_manager,
        MessageType,
        WebSocketMessage,
    )

    PHASE3A_COMPONENTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Phase 3A components not available: {e}")
    PHASE3A_COMPONENTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service status enumeration."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class ServiceHealth:
    """Service health information."""

    service_name: str
    status: ServiceStatus
    last_check: datetime
    response_time: float
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class WebappServiceManager:
    """
    Webapp service manager for orchestrating all webapp services.

    This class provides a unified interface for all webapp services
    and manages their lifecycle, health monitoring, and integration.
    """

    def __init__(self, max_memory_mb: int = 500):
        """
        Initialize webapp service manager.

        Args:
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_memory_mb = max_memory_mb
        self.services = {}
        self.service_status = {}
        self.health_monitor_active = False
        self.health_monitor_thread = None
        self._lock = threading.Lock()

        # Initialize services
        self._initialize_services()

        # Performance tracking
        self.stats = {
            "services_initialized": 0,
            "services_started": 0,
            "services_stopped": 0,
            "health_checks_performed": 0,
            "integration_operations": 0,
        }

        logger.info(
            f"WebappServiceManager initialized with {max_memory_mb}MB memory limit"
        )

    def _initialize_services(self) -> None:
        """Initialize all webapp services."""
        if not PHASE3A_COMPONENTS_AVAILABLE:
            logger.warning(
                "Webapp service components not available - using mock services"
            )
            return

        try:
            # Initialize Enhanced Data Service
            self.services["data_service"] = get_enhanced_data_service()
            self.service_status["data_service"] = ServiceStatus.STOPPED

            # Initialize Task Queue
            self.services["task_queue"] = get_task_queue()
            self.service_status["task_queue"] = ServiceStatus.STOPPED

            # Initialize Task Processor
            self.services["task_processor"] = get_task_processor()
            self.service_status["task_processor"] = ServiceStatus.STOPPED

            # Initialize WebSocket Manager
            self.services["websocket_manager"] = get_websocket_manager()
            self.service_status["websocket_manager"] = ServiceStatus.STOPPED

            self.stats["services_initialized"] = len(self.services)
            logger.info(f"Initialized {len(self.services)} webapp services")

        except Exception as e:
            logger.error(f"Error initializing services: {e}")
            raise

    async def start_all_services(self) -> bool:
        """
        Start all webapp services.

        Returns:
            True if all services started successfully
        """
        success_count = 0

        try:
            # Start WebSocket Manager first (needed for real-time updates)
            if "websocket_manager" in self.services:
                await self._start_service("websocket_manager")
                success_count += 1

            # Start Task Processor (needed for background processing)
            if "task_processor" in self.services:
                await self._start_service("task_processor")
                success_count += 1

            # Data Service doesn't need explicit starting
            if "data_service" in self.services:
                self.service_status["data_service"] = ServiceStatus.RUNNING
                success_count += 1

            # Task Queue doesn't need explicit starting
            if "task_queue" in self.services:
                self.service_status["task_queue"] = ServiceStatus.RUNNING
                success_count += 1

            # Start health monitoring
            self._start_health_monitoring()

            self.stats["services_started"] = success_count
            logger.info(f"Started {success_count} webapp services")

            return success_count == len(self.services)

        except Exception as e:
            logger.error(f"Error starting services: {e}")
            return False

    async def stop_all_services(self) -> bool:
        """
        Stop all webapp services.

        Returns:
            True if all services stopped successfully
        """
        success_count = 0

        try:
            # Stop health monitoring first
            self._stop_health_monitoring()

            # Stop services in reverse order
            service_order = [
                "task_processor",
                "websocket_manager",
                "task_queue",
                "data_service",
            ]

            for service_name in service_order:
                if service_name in self.services:
                    await self._stop_service(service_name)
                    success_count += 1

            self.stats["services_stopped"] = success_count
            logger.info(f"Stopped {success_count} webapp services")

            return True

        except Exception as e:
            logger.error(f"Error stopping services: {e}")
            return False

    async def _start_service(self, service_name: str) -> bool:
        """Start a specific service."""
        try:
            service = self.services[service_name]
            self.service_status[service_name] = ServiceStatus.STARTING

            if service_name == "websocket_manager":
                await service.start()
            elif service_name == "task_processor":
                service.start()

            self.service_status[service_name] = ServiceStatus.RUNNING
            logger.info(f"Started service: {service_name}")
            return True

        except Exception as e:
            logger.error(f"Error starting service {service_name}: {e}")
            self.service_status[service_name] = ServiceStatus.ERROR
            return False

    async def _stop_service(self, service_name: str) -> bool:
        """Stop a specific service."""
        try:
            service = self.services[service_name]
            self.service_status[service_name] = ServiceStatus.STOPPING

            if service_name == "websocket_manager":
                await service.stop()
            elif service_name == "task_processor":
                service.stop()
            elif service_name == "data_service":
                service.cleanup()

            self.service_status[service_name] = ServiceStatus.STOPPED
            logger.info(f"Stopped service: {service_name}")
            return True

        except Exception as e:
            logger.error(f"Error stopping service {service_name}: {e}")
            self.service_status[service_name] = ServiceStatus.ERROR
            return False

    def _start_health_monitoring(self) -> None:
        """Start health monitoring thread."""
        if self.health_monitor_active:
            return

        self.health_monitor_active = True
        self.health_monitor_thread = threading.Thread(
            target=self._health_monitor_loop, daemon=True
        )
        self.health_monitor_thread.start()
        logger.info("Started health monitoring")

    def _stop_health_monitoring(self) -> None:
        """Stop health monitoring thread."""
        self.health_monitor_active = False
        if self.health_monitor_thread:
            self.health_monitor_thread.join(timeout=5.0)
        logger.info("Stopped health monitoring")

    def _health_monitor_loop(self) -> None:
        """Health monitoring loop."""
        while self.health_monitor_active:
            try:
                self._perform_health_checks()
                self.stats["health_checks_performed"] += 1
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                time.sleep(5)  # Short sleep on error

    def _perform_health_checks(self) -> None:
        """Perform health checks on all services."""
        for service_name, service in self.services.items():
            try:
                start_time = time.time()

                if service_name == "data_service":
                    # Check data service health
                    stats = service.get_service_stats()
                    response_time = time.time() - start_time

                    health = ServiceHealth(
                        service_name=service_name,
                        status=ServiceStatus.RUNNING,
                        last_check=datetime.now(),
                        response_time=response_time,
                        metadata={"stats": stats},
                    )

                elif service_name == "task_queue":
                    # Check task queue health
                    stats = service.get_queue_stats()
                    response_time = time.time() - start_time

                    health = ServiceHealth(
                        service_name=service_name,
                        status=ServiceStatus.RUNNING,
                        last_check=datetime.now(),
                        response_time=response_time,
                        metadata={"stats": stats},
                    )

                elif service_name == "task_processor":
                    # Check task processor health
                    stats = service.get_processor_stats()
                    response_time = time.time() - start_time

                    health = ServiceHealth(
                        service_name=service_name,
                        status=(
                            ServiceStatus.RUNNING
                            if service.running
                            else ServiceStatus.STOPPED
                        ),
                        last_check=datetime.now(),
                        response_time=response_time,
                        metadata={"stats": stats},
                    )

                elif service_name == "websocket_manager":
                    # Check WebSocket manager health
                    stats = service.get_manager_stats()
                    response_time = time.time() - start_time

                    health = ServiceHealth(
                        service_name=service_name,
                        status=(
                            ServiceStatus.RUNNING
                            if service.running
                            else ServiceStatus.STOPPED
                        ),
                        last_check=datetime.now(),
                        response_time=response_time,
                        metadata={"stats": stats},
                    )

                # Store health information
                with self._lock:
                    setattr(self, f"{service_name}_health", health)

            except Exception as e:
                logger.error(f"Health check failed for {service_name}: {e}")

                health = ServiceHealth(
                    service_name=service_name,
                    status=ServiceStatus.ERROR,
                    last_check=datetime.now(),
                    response_time=0.0,
                    error_message=str(e),
                )

                with self._lock:
                    setattr(self, f"{service_name}_health", health)

    def get_service_health(self, service_name: str) -> Optional[ServiceHealth]:
        """Get health information for a specific service."""
        with self._lock:
            return getattr(self, f"{service_name}_health", None)

    def get_all_service_health(self) -> Dict[str, ServiceHealth]:
        """Get health information for all services."""
        health_info = {}

        for service_name in self.services.keys():
            health = self.get_service_health(service_name)
            if health:
                health_info[service_name] = health

        return health_info

    def get_service_status(self, service_name: str) -> Optional[ServiceStatus]:
        """Get status of a specific service."""
        return self.service_status.get(service_name)

    def get_all_service_status(self) -> Dict[str, ServiceStatus]:
        """Get status of all services."""
        return self.service_status.copy()

    # Integration methods for webapp components

    async def submit_processing_task(
        self,
        task_type: str,
        parameters: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        progress_callback: Optional[Callable] = None,
    ) -> str:
        """
        Submit a processing task with real-time updates.

        Args:
            task_type: Type of task to execute
            parameters: Task parameters
            priority: Task priority
            progress_callback: Callback for progress updates

        Returns:
            Task ID
        """
        if "task_queue" not in self.services:
            raise RuntimeError("Task queue service not available")

        # Submit task
        task_id = self.services["task_queue"].submit_task(
            task_type=task_type, parameters=parameters, priority=priority
        )

        # Subscribe to task updates if WebSocket manager is available
        if "websocket_manager" in self.services and progress_callback:
            # This would typically be done by the webapp component
            # that initiated the task
            pass

        self.stats["integration_operations"] += 1
        logger.info(f"Submitted processing task {task_id}")

        return task_id

    async def load_data_with_progress(
        self,
        file_path: str,
        progress_callback: Optional[Callable[[LoadingProgress], None]] = None,
        task_id: Optional[str] = None,
    ) -> Union[pd.DataFrame, DataSegment]:
        """
        Load data with progress tracking and real-time updates.

        Args:
            file_path: Path to data file
            progress_callback: Callback for progress updates
            task_id: Task ID for tracking

        Returns:
            Loaded data
        """
        if "data_service" not in self.services:
            raise RuntimeError("Data service not available")

        # Load data with progress tracking
        data = self.services["data_service"].load_data(
            file_path=file_path, progress_callback=progress_callback, task_id=task_id
        )

        # Broadcast data update if WebSocket manager is available
        if "websocket_manager" in self.services and task_id:
            await self.services["websocket_manager"].broadcast_data_update(
                data_id=task_id,
                update_data={"status": "loaded", "file_path": file_path},
            )

        self.stats["integration_operations"] += 1
        logger.info(f"Loaded data from {file_path}")

        return data

    async def get_data_preview_with_updates(
        self,
        file_path: str,
        preview_size: int = 1000,
        progress_callback: Optional[Callable[[LoadingProgress], None]] = None,
        task_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get data preview with progress tracking and real-time updates.

        Args:
            file_path: Path to data file
            preview_size: Number of rows to preview
            progress_callback: Callback for progress updates
            task_id: Task ID for tracking

        Returns:
            Preview data
        """
        if "data_service" not in self.services:
            raise RuntimeError("Data service not available")

        # Get preview with progress tracking
        preview = self.services["data_service"].get_data_preview(
            file_path=file_path,
            preview_size=preview_size,
            progress_callback=progress_callback,
            task_id=task_id,
        )

        # Broadcast preview update if WebSocket manager is available
        if "websocket_manager" in self.services and task_id:
            await self.services["websocket_manager"].broadcast_data_update(
                data_id=task_id,
                update_data={"status": "preview_loaded", "rows": len(preview)},
            )

        self.stats["integration_operations"] += 1
        logger.info(f"Loaded preview from {file_path}")

        return preview

    def get_integration_stats(self) -> Dict[str, Any]:
        """Get comprehensive integration statistics."""
        stats = self.stats.copy()

        # Add service-specific stats
        for service_name, service in self.services.items():
            try:
                if service_name == "data_service":
                    stats[f"{service_name}_stats"] = service.get_service_stats()
                elif service_name == "task_queue":
                    stats[f"{service_name}_stats"] = service.get_queue_stats()
                elif service_name == "task_processor":
                    stats[f"{service_name}_stats"] = service.get_processor_stats()
                elif service_name == "websocket_manager":
                    stats[f"{service_name}_stats"] = service.get_manager_stats()
            except Exception as e:
                logger.error(f"Error getting stats for {service_name}: {e}")
                stats[f"{service_name}_stats"] = {"error": str(e)}

        # Add health information
        stats["service_health"] = self.get_all_service_health()
        stats["service_status"] = self.get_all_service_status()

        return stats

    def cleanup(self) -> None:
        """Cleanup all services and resources."""
        try:
            # Stop all services
            asyncio.run(self.stop_all_services())

            # Clear service references
            self.services.clear()
            self.service_status.clear()

            logger.info("WebappServiceManager cleaned up")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Global instance for webapp use
_service_manager = None


def get_service_manager() -> WebappServiceManager:
    """Get global webapp service manager instance."""
    global _service_manager
    if _service_manager is None:
        _service_manager = WebappServiceManager()
    return _service_manager


async def start_webapp_services() -> bool:
    """Start all webapp services."""
    manager = get_service_manager()
    return await manager.start_all_services()


async def stop_webapp_services() -> bool:
    """Stop all webapp services."""
    manager = get_service_manager()
    return await manager.stop_all_services()


def get_webapp_health() -> Dict[str, ServiceHealth]:
    """Get health status of all webapp services."""
    manager = get_service_manager()
    return manager.get_all_service_health()


def get_webapp_stats() -> Dict[str, Any]:
    """Get comprehensive webapp service statistics."""
    manager = get_service_manager()
    return manager.get_integration_stats()


# Example usage and testing
if __name__ == "__main__":
    print("Webapp Service Manager for vitalDSP")
    print("=" * 50)
    print("\nThis module provides unified service management")
    print("for all webapp services.")
    print("\nServices:")
    print(
        "  - Enhanced Data Service: Chunked loading, memory mapping, progressive loading"
    )
    print("  - Task Queue System: Async processing with Redis backend")
    print("  - WebSocket Manager: Real-time communication")
    print("  - Service Manager: Unified service interface")
    print("\nFeatures:")
    print("  - Unified service interface for webapp components")
    print("  - Automatic service initialization and lifecycle management")
    print(
        "  - Integration between data loading, task processing, and real-time updates"
    )
    print("  - Performance monitoring and health checks")
    print("  - Service orchestration and coordination")
    print("\nAvailability:")
    print(f"  - Webapp service components available: {PHASE3A_COMPONENTS_AVAILABLE}")
