"""
WebSocket Manager for vitalDSP Webapp - Phase 3A Implementation

This module implements the WebSocket manager for real-time communication
between the webapp frontend and backend.

Features:
- WebSocketManager: Real-time communication with frontend
- Task progress broadcasting
- Real-time data updates
- Connection management and heartbeat
- Integration with task queue and data services

Author: vitalDSP Development Team
Date: January 11, 2025
Phase: 3A - Core Infrastructure Enhancement (Week 2)
"""

import os
import sys
import time
import json
import logging
import threading
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

# WebSocket imports
try:
    from fastapi import WebSocket, WebSocketDisconnect
    from fastapi.websockets import WebSocketState

    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    logging.warning("FastAPI WebSocket not available - using mock implementation")

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """WebSocket message types."""

    TASK_UPDATE = "task_update"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    DATA_UPDATE = "data_update"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    CONNECTION_INFO = "connection_info"


@dataclass
class WebSocketMessage:
    """WebSocket message data structure."""

    type: MessageType
    data: Dict[str, Any]
    timestamp: datetime
    message_id: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "message_id": self.message_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WebSocketMessage":
        """Create WebSocketMessage from dictionary."""
        return cls(
            type=MessageType(data["type"]),
            data=data["data"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            message_id=data["message_id"],
        )


class ConnectionManager:
    """Manages WebSocket connections and subscriptions."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_subscriptions: Dict[str, Set[str]] = (
            {}
        )  # connection_id -> set of topics
        self.topic_subscribers: Dict[str, Set[str]] = (
            {}
        )  # topic -> set of connection_ids
        self._lock = threading.Lock()

        # Connection metadata
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}

        # Performance tracking
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_sent": 0,
            "messages_failed": 0,
            "subscriptions_created": 0,
        }

    async def connect(
        self, websocket: WebSocket, connection_id: Optional[str] = None
    ) -> str:
        """
        Accept new WebSocket connection.

        Args:
            websocket: WebSocket connection
            connection_id: Custom connection ID (auto-generated if None)

        Returns:
            Connection ID
        """
        connection_id = connection_id or str(uuid.uuid4())

        await websocket.accept()

        with self._lock:
            self.active_connections[connection_id] = websocket
            self.connection_subscriptions[connection_id] = set()
            self.connection_metadata[connection_id] = {
                "connected_at": datetime.now(),
                "last_activity": datetime.now(),
                "message_count": 0,
            }

            self.stats["total_connections"] += 1
            self.stats["active_connections"] += 1

        logger.info(f"WebSocket connection established: {connection_id}")
        return connection_id

    async def disconnect(self, connection_id: str) -> None:
        """
        Disconnect WebSocket connection.

        Args:
            connection_id: Connection ID
        """
        with self._lock:
            if connection_id in self.active_connections:
                # Remove from all topic subscriptions
                if connection_id in self.connection_subscriptions:
                    for topic in self.connection_subscriptions[connection_id]:
                        if topic in self.topic_subscribers:
                            self.topic_subscribers[topic].discard(connection_id)
                            if not self.topic_subscribers[topic]:
                                del self.topic_subscribers[topic]

                    del self.connection_subscriptions[connection_id]

                # Remove connection
                del self.active_connections[connection_id]
                if connection_id in self.connection_metadata:
                    del self.connection_metadata[connection_id]

                self.stats["active_connections"] -= 1

        logger.info(f"WebSocket connection closed: {connection_id}")

    async def subscribe(self, connection_id: str, topic: str) -> bool:
        """
        Subscribe connection to a topic.

        Args:
            connection_id: Connection ID
            topic: Topic to subscribe to

        Returns:
            True if subscribed successfully
        """
        with self._lock:
            if connection_id not in self.active_connections:
                return False

            # Add to connection subscriptions
            if connection_id not in self.connection_subscriptions:
                self.connection_subscriptions[connection_id] = set()
            self.connection_subscriptions[connection_id].add(topic)

            # Add to topic subscribers
            if topic not in self.topic_subscribers:
                self.topic_subscribers[topic] = set()
            self.topic_subscribers[topic].add(connection_id)

            self.stats["subscriptions_created"] += 1

        logger.debug(f"Connection {connection_id} subscribed to topic {topic}")
        return True

    async def unsubscribe(self, connection_id: str, topic: str) -> bool:
        """
        Unsubscribe connection from a topic.

        Args:
            connection_id: Connection ID
            topic: Topic to unsubscribe from

        Returns:
            True if unsubscribed successfully
        """
        with self._lock:
            if connection_id not in self.active_connections:
                return False

            # Remove from connection subscriptions
            if connection_id in self.connection_subscriptions:
                self.connection_subscriptions[connection_id].discard(topic)

            # Remove from topic subscribers
            if topic in self.topic_subscribers:
                self.topic_subscribers[topic].discard(connection_id)
                if not self.topic_subscribers[topic]:
                    del self.topic_subscribers[topic]

        logger.debug(f"Connection {connection_id} unsubscribed from topic {topic}")
        return True

    async def send_message(self, connection_id: str, message: WebSocketMessage) -> bool:
        """
        Send message to specific connection.

        Args:
            connection_id: Connection ID
            message: Message to send

        Returns:
            True if sent successfully
        """
        with self._lock:
            if connection_id not in self.active_connections:
                return False

            websocket = self.active_connections[connection_id]

        try:
            await websocket.send_text(json.dumps(message.to_dict()))

            # Update connection metadata
            with self._lock:
                if connection_id in self.connection_metadata:
                    self.connection_metadata[connection_id][
                        "last_activity"
                    ] = datetime.now()
                    self.connection_metadata[connection_id]["message_count"] += 1

            self.stats["messages_sent"] += 1
            return True

        except Exception as e:
            logger.error(f"Error sending message to {connection_id}: {e}")
            self.stats["messages_failed"] += 1

            # Remove failed connection
            await self.disconnect(connection_id)
            return False

    async def broadcast_to_topic(self, topic: str, message: WebSocketMessage) -> int:
        """
        Broadcast message to all subscribers of a topic.

        Args:
            topic: Topic to broadcast to
            message: Message to broadcast

        Returns:
            Number of connections that received the message
        """
        with self._lock:
            subscribers = self.topic_subscribers.get(topic, set()).copy()

        sent_count = 0
        failed_connections = []

        for connection_id in subscribers:
            success = await self.send_message(connection_id, message)
            if success:
                sent_count += 1
            else:
                failed_connections.append(connection_id)

        # Clean up failed connections
        for connection_id in failed_connections:
            await self.disconnect(connection_id)

        logger.debug(
            f"Broadcasted message to {sent_count} connections on topic {topic}"
        )
        return sent_count

    async def broadcast_to_all(self, message: WebSocketMessage) -> int:
        """
        Broadcast message to all active connections.

        Args:
            message: Message to broadcast

        Returns:
            Number of connections that received the message
        """
        with self._lock:
            connection_ids = list(self.active_connections.keys())

        sent_count = 0
        failed_connections = []

        for connection_id in connection_ids:
            success = await self.send_message(connection_id, message)
            if success:
                sent_count += 1
            else:
                failed_connections.append(connection_id)

        # Clean up failed connections
        for connection_id in failed_connections:
            await self.disconnect(connection_id)

        logger.debug(f"Broadcasted message to {sent_count} connections")
        return sent_count

    def get_connection_info(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get connection information."""
        with self._lock:
            if connection_id not in self.active_connections:
                return None

            metadata = self.connection_metadata.get(connection_id, {})
            subscriptions = self.connection_subscriptions.get(connection_id, set())

            return {
                "connection_id": connection_id,
                "connected_at": metadata.get("connected_at"),
                "last_activity": metadata.get("last_activity"),
                "message_count": metadata.get("message_count", 0),
                "subscriptions": list(subscriptions),
                "is_active": True,
            }

    def get_all_connections(self) -> List[Dict[str, Any]]:
        """Get information for all connections."""
        with self._lock:
            connection_ids = list(self.active_connections.keys())

        return [self.get_connection_info(cid) for cid in connection_ids if cid]

    def get_stats(self) -> Dict[str, Any]:
        """Get connection manager statistics."""
        with self._lock:
            return {
                **self.stats,
                "topics_count": len(self.topic_subscribers),
                "total_subscriptions": sum(
                    len(subs) for subs in self.topic_subscribers.values()
                ),
            }


class WebSocketManager:
    """
    WebSocket manager for real-time communication with webapp frontend.

    Features:
    - Real-time task progress updates
    - Data streaming and updates
    - Connection management and heartbeat
    - Integration with task queue and data services
    """

    def __init__(self):
        """Initialize WebSocket manager."""
        self.connection_manager = ConnectionManager()
        self.heartbeat_interval = 30  # seconds
        self.heartbeat_task = None
        self.running = False

        # Message handlers
        self.message_handlers: Dict[MessageType, Callable] = {}
        self._register_default_handlers()

        # Performance tracking
        self.stats = {
            "messages_processed": 0,
            "handlers_executed": 0,
            "heartbeats_sent": 0,
        }

        logger.info("WebSocketManager initialized")

    def _register_default_handlers(self) -> None:
        """Register default message handlers."""
        self.message_handlers[MessageType.HEARTBEAT] = self._handle_heartbeat
        self.message_handlers[MessageType.CONNECTION_INFO] = (
            self._handle_connection_info
        )

    async def start(self) -> None:
        """Start WebSocket manager."""
        if self.running:
            return

        self.running = True

        # Start heartbeat task
        if self.heartbeat_interval > 0:
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        logger.info("WebSocketManager started")

    async def stop(self) -> None:
        """Stop WebSocket manager."""
        self.running = False

        # Stop heartbeat task
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        connection_ids = list(self.connection_manager.active_connections.keys())
        for connection_id in connection_ids:
            await self.connection_manager.disconnect(connection_id)

        logger.info("WebSocketManager stopped")

    async def handle_connection(
        self, websocket: WebSocket, connection_id: Optional[str] = None
    ) -> None:
        """
        Handle WebSocket connection.

        Args:
            websocket: WebSocket connection
            connection_id: Custom connection ID
        """
        conn_id = await self.connection_manager.connect(websocket, connection_id)

        try:
            # Send connection info
            await self.send_connection_info(conn_id)

            # Handle messages
            while True:
                try:
                    data = await websocket.receive_text()
                    await self._handle_message(conn_id, data)
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.error(f"Error handling message from {conn_id}: {e}")
                    break

        finally:
            await self.connection_manager.disconnect(conn_id)

    async def _handle_message(self, connection_id: str, data: str) -> None:
        """Handle incoming message."""
        try:
            message_data = json.loads(data)
            message = WebSocketMessage.from_dict(message_data)

            # Update connection activity
            with self.connection_manager._lock:
                if connection_id in self.connection_manager.connection_metadata:
                    self.connection_manager.connection_metadata[connection_id][
                        "last_activity"
                    ] = datetime.now()

            # Process message
            await self._process_message(connection_id, message)
            self.stats["messages_processed"] += 1

        except Exception as e:
            logger.error(f"Error processing message from {connection_id}: {e}")

            # Send error response
            error_message = WebSocketMessage(
                type=MessageType.ERROR,
                data={"error": "Invalid message format", "details": str(e)},
                timestamp=datetime.now(),
                message_id=str(uuid.uuid4()),
            )
            await self.connection_manager.send_message(connection_id, error_message)

    async def _process_message(
        self, connection_id: str, message: WebSocketMessage
    ) -> None:
        """Process message based on type."""
        handler = self.message_handlers.get(message.type)
        if handler:
            try:
                await handler(connection_id, message)
                self.stats["handlers_executed"] += 1
            except Exception as e:
                logger.error(f"Error in message handler for {message.type}: {e}")
        else:
            logger.warning(f"No handler for message type: {message.type}")

    async def _handle_heartbeat(
        self, connection_id: str, message: WebSocketMessage
    ) -> None:
        """Handle heartbeat message."""
        # Send heartbeat response
        response = WebSocketMessage(
            type=MessageType.HEARTBEAT,
            data={"status": "alive", "timestamp": datetime.now().isoformat()},
            timestamp=datetime.now(),
            message_id=str(uuid.uuid4()),
        )
        await self.connection_manager.send_message(connection_id, response)

    async def _handle_connection_info(
        self, connection_id: str, message: WebSocketMessage
    ) -> None:
        """Handle connection info request."""
        info = self.connection_manager.get_connection_info(connection_id)
        if info:
            response = WebSocketMessage(
                type=MessageType.CONNECTION_INFO,
                data=info,
                timestamp=datetime.now(),
                message_id=str(uuid.uuid4()),
            )
            await self.connection_manager.send_message(connection_id, response)

    async def broadcast_task_update(
        self, task_id: str, update_data: Dict[str, Any]
    ) -> int:
        """
        Broadcast task update to subscribers.

        Args:
            task_id: Task ID
            update_data: Update data

        Returns:
            Number of connections that received the update
        """
        message = WebSocketMessage(
            type=MessageType.TASK_UPDATE,
            data={
                "task_id": task_id,
                "update": update_data,
                "timestamp": datetime.now().isoformat(),
            },
            timestamp=datetime.now(),
            message_id=str(uuid.uuid4()),
        )

        return await self.connection_manager.broadcast_to_topic(
            f"task:{task_id}", message
        )

    async def broadcast_task_progress(
        self, task_id: str, progress: float, message: str = ""
    ) -> int:
        """
        Broadcast task progress update.

        Args:
            task_id: Task ID
            progress: Progress percentage (0-100)
            message: Progress message

        Returns:
            Number of connections that received the update
        """
        progress_message = WebSocketMessage(
            type=MessageType.TASK_PROGRESS,
            data={
                "task_id": task_id,
                "progress": progress,
                "message": message,
                "timestamp": datetime.now().isoformat(),
            },
            timestamp=datetime.now(),
            message_id=str(uuid.uuid4()),
        )

        return await self.connection_manager.broadcast_to_topic(
            f"task:{task_id}", progress_message
        )

    async def broadcast_task_completed(self, task_id: str, result: Any) -> int:
        """
        Broadcast task completion.

        Args:
            task_id: Task ID
            result: Task result

        Returns:
            Number of connections that received the update
        """
        message = WebSocketMessage(
            type=MessageType.TASK_COMPLETED,
            data={
                "task_id": task_id,
                "result": result,
                "timestamp": datetime.now().isoformat(),
            },
            timestamp=datetime.now(),
            message_id=str(uuid.uuid4()),
        )

        return await self.connection_manager.broadcast_to_topic(
            f"task:{task_id}", message
        )

    async def broadcast_task_failed(self, task_id: str, error: str) -> int:
        """
        Broadcast task failure.

        Args:
            task_id: Task ID
            error: Error message

        Returns:
            Number of connections that received the update
        """
        message = WebSocketMessage(
            type=MessageType.TASK_FAILED,
            data={
                "task_id": task_id,
                "error": error,
                "timestamp": datetime.now().isoformat(),
            },
            timestamp=datetime.now(),
            message_id=str(uuid.uuid4()),
        )

        return await self.connection_manager.broadcast_to_topic(
            f"task:{task_id}", message
        )

    async def broadcast_data_update(
        self, data_id: str, update_data: Dict[str, Any]
    ) -> int:
        """
        Broadcast data update.

        Args:
            data_id: Data ID
            update_data: Update data

        Returns:
            Number of connections that received the update
        """
        message = WebSocketMessage(
            type=MessageType.DATA_UPDATE,
            data={
                "data_id": data_id,
                "update": update_data,
                "timestamp": datetime.now().isoformat(),
            },
            timestamp=datetime.now(),
            message_id=str(uuid.uuid4()),
        )

        return await self.connection_manager.broadcast_to_topic(
            f"data:{data_id}", message
        )

    async def send_connection_info(self, connection_id: str) -> None:
        """Send connection information to client."""
        info = self.connection_manager.get_connection_info(connection_id)
        if info:
            message = WebSocketMessage(
                type=MessageType.CONNECTION_INFO,
                data=info,
                timestamp=datetime.now(),
                message_id=str(uuid.uuid4()),
            )
            await self.connection_manager.send_message(connection_id, message)

    async def subscribe_to_task(self, connection_id: str, task_id: str) -> bool:
        """Subscribe connection to task updates."""
        return await self.connection_manager.subscribe(connection_id, f"task:{task_id}")

    async def subscribe_to_data(self, connection_id: str, data_id: str) -> bool:
        """Subscribe connection to data updates."""
        return await self.connection_manager.subscribe(connection_id, f"data:{data_id}")

    async def _heartbeat_loop(self) -> None:
        """Heartbeat loop to keep connections alive."""
        while self.running:
            try:
                # Send heartbeat to all connections
                heartbeat_message = WebSocketMessage(
                    type=MessageType.HEARTBEAT,
                    data={"server_time": datetime.now().isoformat()},
                    timestamp=datetime.now(),
                    message_id=str(uuid.uuid4()),
                )

                await self.connection_manager.broadcast_to_all(heartbeat_message)
                self.stats["heartbeats_sent"] += 1

                await asyncio.sleep(self.heartbeat_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(5)  # Short sleep on error

    def register_message_handler(
        self, message_type: MessageType, handler: Callable
    ) -> None:
        """Register custom message handler."""
        self.message_handlers[message_type] = handler
        logger.info(f"Registered handler for message type: {message_type}")

    def get_manager_stats(self) -> Dict[str, Any]:
        """Get WebSocket manager statistics."""
        return {
            **self.stats,
            "connection_manager": self.connection_manager.get_stats(),
            "running": self.running,
            "heartbeat_interval": self.heartbeat_interval,
        }


# Global instance for webapp use
_websocket_manager = None


def get_websocket_manager() -> WebSocketManager:
    """Get global WebSocket manager instance."""
    global _websocket_manager
    if _websocket_manager is None:
        _websocket_manager = WebSocketManager()
    return _websocket_manager


async def start_websocket_manager() -> None:
    """Start global WebSocket manager."""
    manager = get_websocket_manager()
    await manager.start()


async def stop_websocket_manager() -> None:
    """Stop global WebSocket manager."""
    global _websocket_manager
    if _websocket_manager:
        await _websocket_manager.stop()


# Example usage and testing
if __name__ == "__main__":
    print("WebSocket Manager for vitalDSP Webapp")
    print("=" * 50)
    print("\nThis module provides real-time communication")
    print("between the webapp frontend and backend.")
    print("\nFeatures:")
    print("  - WebSocketManager: Real-time communication with frontend")
    print("  - ConnectionManager: Connection and subscription management")
    print("  - Task progress broadcasting: Real-time task updates")
    print("  - Data streaming: Real-time data updates")
    print("  - Heartbeat mechanism: Keep connections alive")
    print("\nDependencies:")
    print(f"  - FastAPI WebSocket available: {WEBSOCKET_AVAILABLE}")
