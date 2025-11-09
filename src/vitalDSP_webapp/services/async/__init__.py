"""
Async Services Module for vitalDSP Webapp

This module provides asynchronous services for the webapp including:
- Task queue management
- WebSocket communication
- Background processing

Author: vitalDSP Team
Date: 2025-01-27
"""

from .task_queue import (
    WebappTaskQueue,
    TaskProcessor,
    get_task_queue,
    get_task_processor,
    start_task_processing,
    stop_task_processing,
    TaskStatus,
    TaskPriority,
)

from .websocket_manager import (
    WebSocketManager,
    get_websocket_manager,
    start_websocket_manager,
    stop_websocket_manager,
    MessageType,
    WebSocketMessage,
)

__all__ = [
    # Task Queue
    "WebappTaskQueue",
    "TaskProcessor",
    "get_task_queue",
    "get_task_processor",
    "start_task_processing",
    "stop_task_processing",
    "TaskStatus",
    "TaskPriority",
    # WebSocket Manager
    "WebSocketManager",
    "get_websocket_manager",
    "start_websocket_manager",
    "stop_websocket_manager",
    "MessageType",
    "WebSocketMessage",
]
