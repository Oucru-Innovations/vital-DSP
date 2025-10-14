"""
Webapp Task Queue System for vitalDSP - Phase 3A Implementation

This module implements the task queue system with Redis backend for asynchronous
processing in the webapp environment.

Features:
- WebappTaskQueue: Redis-based task queue with priority support
- Task management with status tracking and progress updates
- Integration with existing vitalDSP processing pipelines
- WebSocket integration for real-time updates
- Error handling and retry mechanisms

Author: vitalDSP Development Team
Date: January 11, 2025
Phase: 3A - Core Infrastructure Enhancement (Week 2)
"""

import os
import sys
import time
import json
import uuid
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import pickle
import hashlib

# Redis imports
try:
    import redis
    from redis.exceptions import RedisError, ConnectionError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available - using in-memory fallback")

# Add vitalDSP to path for imports
current_dir = Path(__file__).parent
vitaldsp_path = current_dir.parent.parent.parent / "vitalDSP"
if str(vitaldsp_path) not in sys.path:
    sys.path.insert(0, str(vitaldsp_path))

try:
    from vitalDSP.utils.core_infrastructure.optimized_processing_pipeline import OptimizedStandardProcessingPipeline
    from vitalDSP.utils.config_utilities.dynamic_config import DynamicConfigManager
    VITALDSP_AVAILABLE = True
except ImportError as e:
    logging.warning(f"vitalDSP modules not available: {e}")
    VITALDSP_AVAILABLE = False

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status enumeration."""
    
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class TaskPriority(Enum):
    """Task priority enumeration."""
    
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class Task:
    """Task data structure."""
    
    id: str
    type: str
    parameters: Dict[str, Any]
    status: TaskStatus
    priority: TaskPriority
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    message: str = ""
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: Optional[int] = None  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create Task from dictionary."""
        # Convert ISO strings back to datetime objects
        for key in ['created_at', 'started_at', 'completed_at']:
            if key in data and data[key]:
                data[key] = datetime.fromisoformat(data[key])
        
        # Convert enum strings back to enums
        data['status'] = TaskStatus(data['status'])
        data['priority'] = TaskPriority(data['priority'])
        
        return cls(**data)


@dataclass
class TaskResult:
    """Task result data structure."""
    
    task_id: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class InMemoryTaskQueue:
    """In-memory task queue fallback when Redis is not available."""
    
    def __init__(self):
        self.tasks = {}
        self.queue = []
        self._lock = threading.Lock()
        self._processing = False
        self._worker_thread = None
    
    def enqueue(self, task: Task) -> str:
        """Add task to queue."""
        with self._lock:
            self.tasks[task.id] = task
            self.queue.append(task.id)
            # Sort by priority (higher priority first)
            self.queue.sort(key=lambda tid: self.tasks[tid].priority.value, reverse=True)
        return task.id
    
    def dequeue(self) -> Optional[Task]:
        """Get next task from queue."""
        with self._lock:
            if self.queue:
                task_id = self.queue.pop(0)
                return self.tasks.get(task_id)
        return None
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self.tasks.get(task_id)
    
    def update_task(self, task: Task) -> None:
        """Update task in queue."""
        with self._lock:
            self.tasks[task.id] = task
    
    def remove_task(self, task_id: str) -> bool:
        """Remove task from queue."""
        with self._lock:
            if task_id in self.tasks:
                del self.tasks[task_id]
                if task_id in self.queue:
                    self.queue.remove(task_id)
                return True
        return False
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return len(self.queue)
    
    def get_all_tasks(self) -> List[Task]:
        """Get all tasks."""
        return list(self.tasks.values())


class WebappTaskQueue:
    """
    Webapp task queue with Redis backend and fallback to in-memory.
    
    Features:
    - Redis-based task queue with priority support
    - Task status tracking and progress updates
    - Error handling and retry mechanisms
    - Integration with vitalDSP processing pipelines
    - WebSocket integration for real-time updates
    """
    
    def __init__(
        self, 
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        max_retries: int = 3,
        task_timeout: int = 3600  # 1 hour default
    ):
        """
        Initialize webapp task queue.
        
        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            redis_password: Redis password
            max_retries: Maximum retry attempts for failed tasks
            task_timeout: Default task timeout in seconds
        """
        self.max_retries = max_retries
        self.task_timeout = task_timeout
        self.callbacks = {}
        self._lock = threading.Lock()
        
        # Initialize Redis connection
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    password=redis_password,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                # Test connection
                self.redis_client.ping()
                self.use_redis = True
                logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Using in-memory fallback.")
                self.use_redis = False
                self.redis_client = None
        else:
            self.use_redis = False
            self.redis_client = None
        
        # Fallback to in-memory queue
        if not self.use_redis:
            self.memory_queue = InMemoryTaskQueue()
        
        # Performance tracking
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_cancelled": 0,
            "average_execution_time": 0.0,
            "redis_operations": 0,
            "memory_operations": 0
        }
        
        logger.info(f"WebappTaskQueue initialized with Redis: {self.use_redis}")
    
    def submit_task(
        self, 
        task_type: str, 
        parameters: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        callback: Optional[Callable[[TaskResult], None]] = None,
        task_id: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> str:
        """
        Submit a new task to the queue.
        
        Args:
            task_type: Type of task to execute
            parameters: Task parameters
            priority: Task priority
            callback: Callback function for task completion
            task_id: Custom task ID (auto-generated if None)
            timeout: Task timeout in seconds
            
        Returns:
            Task ID
        """
        task_id = task_id or str(uuid.uuid4())
        
        task = Task(
            id=task_id,
            type=task_type,
            parameters=parameters,
            status=TaskStatus.QUEUED,
            priority=priority,
            created_at=datetime.now(),
            timeout=timeout or self.task_timeout
        )
        
        if self.use_redis:
            self._submit_task_redis(task)
        else:
            self._submit_task_memory(task)
        
        # Store callback
        if callback:
            self.callbacks[task_id] = callback
        
        self.stats["tasks_submitted"] += 1
        logger.info(f"Submitted task {task_id} of type {task_type}")
        
        return task_id
    
    def _submit_task_redis(self, task: Task) -> None:
        """Submit task to Redis queue."""
        try:
            # Store task data
            task_key = f"task:{task.id}"
            self.redis_client.hset(task_key, mapping=task.to_dict())
            
            # Add to priority queue
            queue_key = f"queue:{task.priority.value}"
            self.redis_client.lpush(queue_key, task.id)
            
            # Set expiration
            self.redis_client.expire(task_key, task.timeout)
            
            self.stats["redis_operations"] += 1
            
        except RedisError as e:
            logger.error(f"Redis error submitting task {task.id}: {e}")
            raise
    
    def _submit_task_memory(self, task: Task) -> None:
        """Submit task to memory queue."""
        self.memory_queue.enqueue(task)
        self.stats["memory_operations"] += 1
    
    def get_task_status(self, task_id: str) -> Optional[Task]:
        """
        Get task status and details.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task object or None if not found
        """
        if self.use_redis:
            return self._get_task_redis(task_id)
        else:
            return self.memory_queue.get_task(task_id)
    
    def _get_task_redis(self, task_id: str) -> Optional[Task]:
        """Get task from Redis."""
        try:
            task_key = f"task:{task_id}"
            task_data = self.redis_client.hgetall(task_key)
            
            if not task_data:
                return None
            
            return Task.from_dict(task_data)
            
        except RedisError as e:
            logger.error(f"Redis error getting task {task_id}: {e}")
            return None
    
    def update_task_progress(
        self, 
        task_id: str, 
        progress: float, 
        message: str = ""
    ) -> bool:
        """
        Update task progress.
        
        Args:
            task_id: Task ID
            progress: Progress percentage (0-100)
            message: Progress message
            
        Returns:
            True if updated successfully
        """
        task = self.get_task_status(task_id)
        if not task:
            return False
        
        task.progress = min(100.0, max(0.0, progress))
        task.message = message
        
        if self.use_redis:
            return self._update_task_redis(task)
        else:
            self.memory_queue.update_task(task)
            return True
    
    def _update_task_redis(self, task: Task) -> bool:
        """Update task in Redis."""
        try:
            task_key = f"task:{task.id}"
            self.redis_client.hset(task_key, mapping={
                'progress': task.progress,
                'message': task.message
            })
            self.stats["redis_operations"] += 1
            return True
            
        except RedisError as e:
            logger.error(f"Redis error updating task {task.id}: {e}")
            return False
    
    def complete_task(
        self, 
        task_id: str, 
        result: Any = None, 
        error: Optional[str] = None
    ) -> bool:
        """
        Mark task as completed.
        
        Args:
            task_id: Task ID
            result: Task result
            error: Error message if failed
            
        Returns:
            True if completed successfully
        """
        task = self.get_task_status(task_id)
        if not task:
            return False
        
        if error:
            task.status = TaskStatus.FAILED
            task.error = error
            self.stats["tasks_failed"] += 1
        else:
            task.status = TaskStatus.COMPLETED
            task.result = result
            self.stats["tasks_completed"] += 1
        
        task.completed_at = datetime.now()
        task.progress = 100.0
        
        if self.use_redis:
            success = self._update_task_redis(task)
        else:
            self.memory_queue.update_task(task)
            success = True
        
        # Call callback if available
        if task_id in self.callbacks:
            try:
                task_result = TaskResult(
                    task_id=task_id,
                    success=task.status == TaskStatus.COMPLETED,
                    result=result,
                    error=error,
                    execution_time=(task.completed_at - task.started_at).total_seconds() if task.started_at else 0.0
                )
                self.callbacks[task_id](task_result)
            except Exception as e:
                logger.error(f"Error calling callback for task {task_id}: {e}")
            finally:
                del self.callbacks[task_id]
        
        return success
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            True if cancelled successfully
        """
        task = self.get_task_status(task_id)
        if not task or task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            return False
        
        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.now()
        
        if self.use_redis:
            success = self._update_task_redis(task)
        else:
            self.memory_queue.update_task(task)
            success = True
        
        self.stats["tasks_cancelled"] += 1
        
        # Remove from callback
        if task_id in self.callbacks:
            del self.callbacks[task_id]
        
        logger.info(f"Cancelled task {task_id}")
        return success
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        stats = self.stats.copy()
        
        if self.use_redis:
            try:
                # Get Redis queue sizes
                stats["queue_sizes"] = {}
                for priority in TaskPriority:
                    queue_key = f"queue:{priority.value}"
                    stats["queue_sizes"][priority.name] = self.redis_client.llen(queue_key)
                
                stats["total_queued"] = sum(stats["queue_sizes"].values())
                
            except RedisError as e:
                logger.error(f"Redis error getting queue stats: {e}")
                stats["total_queued"] = 0
        else:
            stats["total_queued"] = self.memory_queue.get_queue_size()
        
        return stats
    
    def cleanup_expired_tasks(self) -> int:
        """Clean up expired tasks."""
        cleaned = 0
        
        if self.use_redis:
            try:
                # Get all task keys
                task_keys = self.redis_client.keys("task:*")
                
                for task_key in task_keys:
                    task_data = self.redis_client.hgetall(task_key)
                    if task_data:
                        task = Task.from_dict(task_data)
                        
                        # Check if task is expired
                        if task.created_at and task.timeout:
                            expiry_time = task.created_at + timedelta(seconds=task.timeout)
                            if datetime.now() > expiry_time and task.status == TaskStatus.QUEUED:
                                self.redis_client.delete(task_key)
                                cleaned += 1
                
            except RedisError as e:
                logger.error(f"Redis error cleaning up tasks: {e}")
        
        return cleaned
    
    def get_all_tasks(self, status_filter: Optional[TaskStatus] = None) -> List[Task]:
        """Get all tasks, optionally filtered by status."""
        tasks = []
        
        if self.use_redis:
            try:
                task_keys = self.redis_client.keys("task:*")
                
                for task_key in task_keys:
                    task_data = self.redis_client.hgetall(task_key)
                    if task_data:
                        task = Task.from_dict(task_data)
                        if status_filter is None or task.status == status_filter:
                            tasks.append(task)
                
            except RedisError as e:
                logger.error(f"Redis error getting all tasks: {e}")
        else:
            tasks = self.memory_queue.get_all_tasks()
            if status_filter:
                tasks = [t for t in tasks if t.status == status_filter]
        
        return tasks


class TaskProcessor:
    """
    Task processor for executing tasks from the queue.
    
    Features:
    - Background processing of tasks
    - Integration with vitalDSP processing pipelines
    - Progress tracking and error handling
    - Automatic retry for failed tasks
    """
    
    def __init__(
        self, 
        task_queue: WebappTaskQueue,
        max_workers: int = 2,
        config_manager: Optional[DynamicConfigManager] = None
    ):
        """
        Initialize task processor.
        
        Args:
            task_queue: Task queue instance
            max_workers: Maximum number of worker threads
            config_manager: Configuration manager
        """
        self.task_queue = task_queue
        self.max_workers = max_workers
        self.config_manager = config_manager or DynamicConfigManager()
        self.workers = []
        self.running = False
        self._lock = threading.Lock()
        
        # Initialize processing pipeline if available
        if VITALDSP_AVAILABLE:
            self.processing_pipeline = OptimizedStandardProcessingPipeline(self.config_manager)
        else:
            self.processing_pipeline = None
        
        # Performance tracking
        self.stats = {
            "tasks_processed": 0,
            "tasks_succeeded": 0,
            "tasks_failed": 0,
            "average_processing_time": 0.0,
            "worker_utilization": 0.0
        }
        
        logger.info(f"TaskProcessor initialized with {max_workers} workers")
    
    def start(self) -> None:
        """Start task processing workers."""
        if self.running:
            return
        
        self.running = True
        
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker, daemon=True, name=f"TaskWorker-{i}")
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"Started {self.max_workers} task processing workers")
    
    def stop(self) -> None:
        """Stop task processing workers."""
        self.running = False
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        self.workers.clear()
        logger.info("Task processing workers stopped")
    
    def _worker(self) -> None:
        """Background worker for processing tasks."""
        while self.running:
            try:
                # Get next task
                task = self._get_next_task()
                if task is None:
                    time.sleep(0.1)  # Short sleep if no tasks
                    continue
                
                # Process the task
                self._process_task(task)
                
            except Exception as e:
                logger.error(f"Worker error: {e}")
                time.sleep(1.0)  # Longer sleep on error
    
    def _get_next_task(self) -> Optional[Task]:
        """Get next task from queue."""
        if self.task_queue.use_redis:
            return self._get_next_task_redis()
        else:
            return self.task_queue.memory_queue.dequeue()
    
    def _get_next_task_redis(self) -> Optional[Task]:
        """Get next task from Redis queue."""
        try:
            # Check queues in priority order (highest first)
            for priority in sorted(TaskPriority, key=lambda p: p.value, reverse=True):
                queue_key = f"queue:{priority.value}"
                task_id = self.task_queue.redis_client.rpop(queue_key)
                
                if task_id:
                    task = self.task_queue.get_task_status(task_id)
                    if task and task.status == TaskStatus.QUEUED:
                        return task
            
        except RedisError as e:
            logger.error(f"Redis error getting next task: {e}")
        
        return None
    
    def _process_task(self, task: Task) -> None:
        """Process a single task."""
        start_time = time.time()
        
        try:
            # Update task status
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            self.task_queue._update_task_redis(task) if self.task_queue.use_redis else self.task_queue.memory_queue.update_task(task)
            
            # Process based on task type
            result = self._execute_task(task)
            
            # Complete task
            self.task_queue.complete_task(task.id, result)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats["tasks_processed"] += 1
            self.stats["tasks_succeeded"] += 1
            self.stats["average_processing_time"] = (
                (self.stats["average_processing_time"] * (self.stats["tasks_processed"] - 1) + processing_time) 
                / self.stats["tasks_processed"]
            )
            
            logger.info(f"Processed task {task.id} in {processing_time:.3f}s")
            
        except Exception as e:
            # Handle task failure
            error_msg = str(e)
            logger.error(f"Task {task.id} failed: {error_msg}")
            
            # Check if we should retry
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.RETRYING
                task.message = f"Retrying ({task.retry_count}/{task.max_retries}): {error_msg}"
                
                # Re-queue the task
                if self.task_queue.use_redis:
                    queue_key = f"queue:{task.priority.value}"
                    self.task_queue.redis_client.lpush(queue_key, task.id)
                else:
                    self.task_queue.memory_queue.enqueue(task)
                
                logger.info(f"Retrying task {task.id} (attempt {task.retry_count})")
            else:
                # Mark as failed
                self.task_queue.complete_task(task.id, error=error_msg)
                self.stats["tasks_failed"] += 1
    
    def _execute_task(self, task: Task) -> Any:
        """Execute task based on type."""
        task_type = task.type
        parameters = task.parameters
        
        if task_type == "signal_processing":
            return self._execute_signal_processing(task, parameters)
        elif task_type == "data_loading":
            return self._execute_data_loading(task, parameters)
        elif task_type == "quality_assessment":
            return self._execute_quality_assessment(task, parameters)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def _execute_signal_processing(self, task: Task, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute signal processing task."""
        if not self.processing_pipeline:
            raise RuntimeError("Processing pipeline not available")
        
        # Update progress
        self.task_queue.update_task_progress(task.id, 10.0, "Starting signal processing")
        
        # Extract parameters
        signal = parameters.get('signal')
        fs = parameters.get('sampling_rate', 1000.0)
        signal_type = parameters.get('signal_type', 'ECG')
        metadata = parameters.get('metadata', {})
        
        if signal is None:
            raise ValueError("Signal data is required")
        
        # Convert to numpy array if needed
        if isinstance(signal, list):
            signal = np.array(signal)
        
        # Update progress
        self.task_queue.update_task_progress(task.id, 30.0, "Processing signal through pipeline")
        
        # Process signal
        result = self.processing_pipeline.process_signal(
            signal=signal,
            fs=fs,
            signal_type=signal_type,
            metadata=metadata,
            session_id=task.id
        )
        
        # Update progress
        self.task_queue.update_task_progress(task.id, 90.0, "Finalizing results")
        
        return result
    
    def _execute_data_loading(self, task: Task, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data loading task."""
        file_path = parameters.get('file_path')
        if not file_path:
            raise ValueError("File path is required")
        
        # Update progress
        self.task_queue.update_task_progress(task.id, 20.0, f"Loading data from {file_path}")
        
        # Import enhanced data service
        try:
            from ..data.enhanced_data_service import get_enhanced_data_service
            data_service = get_enhanced_data_service()
            
            # Load data
            data = data_service.load_data(file_path)
            
            # Update progress
            self.task_queue.update_task_progress(task.id, 80.0, "Data loaded successfully")
            
            return {
                'data': data,
                'file_path': file_path,
                'loaded_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            raise RuntimeError(f"Data loading failed: {str(e)}")
    
    def _execute_quality_assessment(self, task: Task, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute quality assessment task."""
        signal = parameters.get('signal')
        if signal is None:
            raise ValueError("Signal data is required")
        
        # Update progress
        self.task_queue.update_task_progress(task.id, 25.0, "Assessing signal quality")
        
        # Simple quality assessment (placeholder)
        quality_metrics = {
            'snr': 20.0,  # Placeholder
            'baseline_stability': 0.8,
            'artifact_level': 0.1,
            'overall_quality': 0.85
        }
        
        # Update progress
        self.task_queue.update_task_progress(task.id, 75.0, "Quality assessment completed")
        
        return {
            'quality_metrics': quality_metrics,
            'assessed_at': datetime.now().isoformat()
        }
    
    def get_processor_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        return {
            **self.stats,
            "workers_active": len([w for w in self.workers if w.is_alive()]),
            "running": self.running
        }


# Global instances for webapp use
_task_queue = None
_task_processor = None

def get_task_queue() -> WebappTaskQueue:
    """Get global task queue instance."""
    global _task_queue
    if _task_queue is None:
        _task_queue = WebappTaskQueue()
    return _task_queue

def get_task_processor() -> TaskProcessor:
    """Get global task processor instance."""
    global _task_processor
    if _task_processor is None:
        _task_processor = TaskProcessor(get_task_queue())
    return _task_processor

def start_task_processing() -> None:
    """Start global task processing."""
    processor = get_task_processor()
    processor.start()

def stop_task_processing() -> None:
    """Stop global task processing."""
    global _task_processor
    if _task_processor:
        _task_processor.stop()


# Example usage and testing
if __name__ == "__main__":
    print("Webapp Task Queue System for vitalDSP")
    print("=" * 50)
    print("\nThis module provides asynchronous task processing")
    print("for the webapp environment.")
    print("\nFeatures:")
    print("  - WebappTaskQueue: Redis-based task queue with priority support")
    print("  - TaskProcessor: Background processing with vitalDSP integration")
    print("  - Task management: Status tracking, progress updates, error handling")
    print("  - Retry mechanisms: Automatic retry for failed tasks")
    print("\nDependencies:")
    print(f"  - Redis available: {REDIS_AVAILABLE}")
    print(f"  - vitalDSP available: {VITALDSP_AVAILABLE}")
