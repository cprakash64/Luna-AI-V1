# app/services/task_queue.py
"""
Task queue implementation for Luna AI
"""
import asyncio
import logging
import time
from typing import Dict, Any, List, Callable, Awaitable, Optional, Tuple
import uuid

logger = logging.getLogger("tasks.queue")

class Task:
    """Represents a task in the queue"""
    def __init__(self, 
                 task_id: str, 
                 task_type: str, 
                 params: Dict[str, Any],
                 user_id: Optional[str] = None,
                 priority: int = 0):
        self.task_id = task_id
        self.task_type = task_type
        self.params = params
        self.user_id = user_id
        self.priority = priority
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.status = "pending"  # pending, running, completed, failed
        self.result: Optional[Any] = None
        self.error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "user_id": self.user_id,
            "priority": self.priority,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status
        }
    
    def mark_started(self):
        """Mark task as started"""
        self.status = "running"
        self.started_at = time.time()
    
    def mark_completed(self, result: Any = None):
        """Mark task as completed"""
        self.status = "completed"
        self.completed_at = time.time()
        self.result = result
    
    def mark_failed(self, error: str):
        """Mark task as failed"""
        self.status = "failed"
        self.completed_at = time.time()
        self.error = error

class TaskQueue:
    """
    Asynchronous task queue for handling transcription and other long-running tasks
    """
    def __init__(self, max_workers: int = 2):
        self.max_workers = max_workers
        self.queues: Dict[str, List[Task]] = {}
        self.tasks: Dict[str, Task] = {}
        self.handlers: Dict[str, Callable[[Task], Awaitable[Any]]] = {}
        self.running_tasks: List[asyncio.Task] = []
        self.active_workers = 0
        self.is_running = False
        self.on_task_completed: Optional[Callable[[Task], Awaitable[None]]] = None
        self.on_task_started: Optional[Callable[[Task], Awaitable[None]]] = None
        self.queue_lock = asyncio.Lock()  # Added lock for queue operations
        
        # Create default queue
        self.queues["default"] = []
    
    def register_handler(self, task_type: str, handler: Callable[[Task], Awaitable[Any]]):
        """Register a handler for a task type"""
        self.handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")
    
    def register_on_completed(self, callback: Callable[[Task], Awaitable[None]]):
        """Register callback for task completion"""
        self.on_task_completed = callback
    
    def register_on_started(self, callback: Callable[[Task], Awaitable[None]]):
        """Register callback for task start"""
        self.on_task_started = callback
    
    async def add_task(self, 
                  task_type: str, 
                  params: Dict[str, Any],
                  user_id: Optional[str] = None,
                  priority: int = 0,
                  queue_name: str = "default") -> str:
        """
        Add a task to the queue
        
        Args:
            task_type: Type of task
            params: Parameters for the task
            user_id: User ID associated with task
            priority: Priority (higher = more important)
            queue_name: Queue to add to
            
        Returns:
            Task ID
        """
        async with self.queue_lock:  # Use lock when modifying queues
            # Create the queue if it doesn't exist
            if queue_name not in self.queues:
                self.queues[queue_name] = []
            
            # Create task
            task_id = str(uuid.uuid4())
            task = Task(task_id, task_type, params, user_id, priority)
            
            # Add to queue
            self.queues[queue_name].append(task)
            # Sort queue by priority (highest first)
            self.queues[queue_name] = sorted(
                self.queues[queue_name], 
                key=lambda t: (-t.priority, t.created_at)
            )
            
            # Add to tasks dict
            self.tasks[task_id] = task
        
        logger.info(f"Added task {task_id} ({task_type}) to queue {queue_name}")
        
        # Make sure queue is running
        if not self.is_running:
            logger.warning("Task queue was not running, starting it now")
            self.start()
        
        return task_id
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID"""
        return self.tasks.get(task_id)
    
    def get_user_tasks(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all tasks for a user"""
        user_tasks = [
            task.to_dict() for task in self.tasks.values()
            if task.user_id == user_id
        ]
        return sorted(user_tasks, key=lambda t: t["created_at"], reverse=True)
    
    async def process_task(self, task: Task) -> None:
        """Process a single task"""
        if task.task_type not in self.handlers:
            task.mark_failed(f"No handler registered for task type: {task.task_type}")
            logger.error(f"No handler for task type: {task.task_type}")
            
            # Notify task completed even if failed
            if self.on_task_completed:
                await self.on_task_completed(task)
            return
        
        handler = self.handlers[task.task_type]
        task.mark_started()
        
        # Notify task started
        if self.on_task_started:
            try:
                await self.on_task_started(task)
            except Exception as e:
                logger.error(f"Error in on_task_started callback: {str(e)}", exc_info=True)
        
        try:
            result = await handler(task)
            task.mark_completed(result)
            logger.info(f"Task {task.task_id} completed successfully")
        except Exception as e:
            error_msg = f"Error in task {task.task_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)  # Log full traceback
            task.mark_failed(error_msg)
        
        # Notify task completed
        if self.on_task_completed:
            try:
                await self.on_task_completed(task)
            except Exception as e:
                logger.error(f"Error in on_task_completed callback: {str(e)}", exc_info=True)
    
    async def _worker_loop(self):
        """Main worker loop to process tasks from the queue"""
        logger.info("Worker loop started")
    
        try:
            while self.is_running:
                # Check for any available tasks
                task = None
                queue_name = None
                
                # Look for tasks in all queues (with lock)
                async with self.queue_lock:
                    for q_name, queue in self.queues.items():
                        if queue and len(queue) > 0:
                            logger.info(f"Found task in queue {q_name}")
                            task = queue[0]
                            queue_name = q_name
                            # Remove from queue inside the lock
                            self.queues[queue_name].remove(task)
                            break
                
                if task and queue_name:
                    # Debugging
                    logger.info(f"Processing task {task.task_id} ({task.task_type}) from queue {queue_name}")
                    
                    # Increase active workers count
                    self.active_workers += 1
                    
                    try:
                        # Use the common process_task method instead of duplicating logic
                        await self.process_task(task)
                    finally:
                        # Decrease active workers count
                        self.active_workers -= 1
                else:
                    # No tasks available, sleep briefly
                    await asyncio.sleep(0.1)
                    
        except asyncio.CancelledError:
            logger.info("Worker loop cancelled")
        except Exception as e:
            logger.error(f"Error in worker loop: {str(e)}", exc_info=True)
            # Restart the worker if it crashes unexpectedly
            if self.is_running:
                logger.info("Restarting crashed worker")
                worker_task = asyncio.create_task(self._worker_loop())
                self.running_tasks.append(worker_task)
            
        logger.info("Worker loop exiting")
    
    def start(self) -> None:
        """Start the task queue worker"""
        if self.is_running:
            logger.warning("Task queue already running")
            return
            
        logger.info(f"Starting task queue with {self.max_workers} workers")
        self.is_running = True
        
        # Start workers
        for i in range(self.max_workers):
            worker_task = asyncio.create_task(self._worker_loop())
            self.running_tasks.append(worker_task)
            
        logger.info(f"Started {len(self.running_tasks)} worker tasks")
    
    async def stop(self) -> None:
        """Stop the task queue"""
        if self.is_running:
            self.is_running = False
            
            # Wait for worker tasks to complete
            if self.running_tasks:
                for task in self.running_tasks:
                    task.cancel()
                
                await asyncio.gather(*self.running_tasks, return_exceptions=True)
                self.running_tasks = []
            
            logger.info("Stopped TaskQueue")
    
    async def health_check(self) -> Dict[str, Any]:
        """Get health status of the task queue"""
        queue_lengths = {q_name: len(queue) for q_name, queue in self.queues.items()}
        
        status = {
            "is_running": self.is_running,
            "active_workers": self.active_workers,
            "max_workers": self.max_workers,
            "pending_tasks": sum(len(q) for q in self.queues.values()),
            "queue_lengths": queue_lengths,
            "total_tasks": len(self.tasks)
        }
        
        return status