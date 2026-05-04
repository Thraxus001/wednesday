"""
cognitive_control.py - Executive control over thinking processes for Wednesday AI

This module implements Wednesday's ability to regulate her own cognitive
processes - deciding when to think fast vs slow, how much mental resources
to allocate to tasks, and how to manage interruptions. This metacognitive
control is essential for efficient and appropriate cognitive functioning.

Key improvements:
- Added comprehensive validation and error handling
- Fixed task queue management with proper typing
- Enhanced cognitive load calculation with decay
- Improved interruption handling with priority-based decisions
- Added proper type hints and documentation
"""

import time
import logging
import math
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Cognitive processing modes"""
    FAST = "fast"           # Quick, automatic, heuristic
    NORMAL = "normal"       # Balanced, everyday processing
    DEEP = "deep"           # Slow, analytical, thorough
    CRITICAL = "critical"   # Maximum depth, very careful
    
    @classmethod
    def has_value(cls, value: str) -> bool:
        """Check if value exists in enum"""
        return value in [e.value for e in cls]


class TaskPriority(Enum):
    """Priority levels for tasks"""
    TRIVIAL = 0
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5
    
    @classmethod
    def from_int(cls, value: int) -> 'TaskPriority':
        """Get enum from integer value"""
        for priority in cls:
            if priority.value == value:
                return priority
        return cls.NORMAL


class InterruptionDecision(Enum):
    """How to handle an interruption"""
    ACCEPT = "accept"           # Switch focus immediately
    DEFER = "defer"             # Finish current task first
    REJECT = "reject"           # Ignore interruption
    QUEUE = "queue"              # Add to queue for later
    
    @classmethod
    def has_value(cls, value: str) -> bool:
        """Check if value exists in enum"""
        return value in [e.value for e in cls]


@dataclass
class ControlSettings:
    """
    Current cognitive control settings.
    """
    processing_depth: float          # 0-1, how thorough (0=shallow,1=deep)
    speed_accuracy_tradeoff: float   # 0-1, 0=fast, 1=accurate
    cognitive_load: float            # 0-1, how much capacity used
    processing_mode: ProcessingMode   # Current mode
    
    # Time allocations
    estimated_time_available: float = 0.0  # seconds
    time_spent_on_current: float = 0.0
    
    # Confidence thresholds
    required_confidence: float = 0.7
    current_confidence: float = 0.0
    
    def __post_init__(self):
        """Validate control settings"""
        self._validate_float('processing_depth', self.processing_depth, 0, 1)
        self._validate_float('speed_accuracy_tradeoff', self.speed_accuracy_tradeoff, 0, 1)
        self._validate_float('cognitive_load', self.cognitive_load, 0, 1)
        self._validate_float('estimated_time_available', self.estimated_time_available, 0, float('inf'))
        self._validate_float('time_spent_on_current', self.time_spent_on_current, 0, float('inf'))
        self._validate_float('required_confidence', self.required_confidence, 0, 1)
        self._validate_float('current_confidence', self.current_confidence, 0, 1)
        
        if not isinstance(self.processing_mode, ProcessingMode):
            if isinstance(self.processing_mode, str):
                try:
                    self.processing_mode = ProcessingMode(self.processing_mode)
                except ValueError:
                    raise ValueError(f"Invalid processing mode: {self.processing_mode}")
            else:
                raise TypeError(f"processing_mode must be ProcessingMode, got {type(self.processing_mode)}")
    
    def _validate_float(self, name: str, value: float, min_val: float, max_val: float) -> None:
        """Validate float is within range"""
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number, got {type(value)}")
        if not min_val <= value <= max_val:
            raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'processing_depth': round(self.processing_depth, 3),
            'speed_accuracy_tradeoff': round(self.speed_accuracy_tradeoff, 3),
            'cognitive_load': round(self.cognitive_load, 3),
            'mode': self.processing_mode.value,
            'required_confidence': round(self.required_confidence, 3),
            'time_available': round(self.estimated_time_available, 3)
        }


@dataclass
class ResourceAllocation:
    """
    Resource allocation decision for a task.
    """
    task: str
    priority: TaskPriority
    allocated_depth: float          # 0-1
    allocated_time_ms: float        # Estimated time to allocate
    expected_confidence: float      # Expected confidence after allocation
    processing_mode: ProcessingMode
    
    def __post_init__(self):
        """Validate resource allocation"""
        if not self.task:
            raise ValueError("task cannot be empty")
        if not isinstance(self.priority, TaskPriority):
            if isinstance(self.priority, int):
                self.priority = TaskPriority.from_int(self.priority)
            else:
                raise TypeError(f"priority must be TaskPriority, got {type(self.priority)}")
        if not 0 <= self.allocated_depth <= 1:
            raise ValueError(f"allocated_depth must be between 0 and 1, got {self.allocated_depth}")
        if self.allocated_time_ms < 0:
            raise ValueError(f"allocated_time_ms cannot be negative, got {self.allocated_time_ms}")
        if not 0 <= self.expected_confidence <= 1:
            raise ValueError(f"expected_confidence must be between 0 and 1, got {self.expected_confidence}")
        if not isinstance(self.processing_mode, ProcessingMode):
            if isinstance(self.processing_mode, str):
                try:
                    self.processing_mode = ProcessingMode(self.processing_mode)
                except ValueError:
                    raise ValueError(f"Invalid processing mode: {self.processing_mode}")
            else:
                raise TypeError(f"processing_mode must be ProcessingMode, got {type(self.processing_mode)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'task': self.task[:40] + "..." if len(self.task) > 40 else self.task,
            'priority': self.priority.name,
            'depth': round(self.allocated_depth, 3),
            'time_ms': round(self.allocated_time_ms),
            'expected_confidence': round(self.expected_confidence, 3),
            'mode': self.processing_mode.value
        }


@dataclass
class QueuedTask:
    """Task in the queue"""
    task: str
    priority: TaskPriority
    start_time: Optional[float] = None
    is_interruption: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class InterruptionRecord:
    """
    Record of an interruption event.
    """
    source: str
    priority: TaskPriority
    timestamp: float = field(default_factory=time.time)
    decision: Optional[InterruptionDecision] = None
    resolved: bool = False
    
    def __post_init__(self):
        """Validate interruption record"""
        if not self.source:
            raise ValueError("source cannot be empty")
        if not isinstance(self.priority, TaskPriority):
            if isinstance(self.priority, int):
                self.priority = TaskPriority.from_int(self.priority)
            else:
                raise TypeError(f"priority must be TaskPriority, got {type(self.priority)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'source': self.source,
            'priority': self.priority.value,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'decision': self.decision.value if self.decision else None,
            'resolved': self.resolved
        }


class CognitiveControl:
    """
    Executive control over Wednesday's thinking processes.
    
    This module acts as the executive function for cognition, deciding:
    - How deeply to process different types of information
    - When to prioritize speed vs accuracy
    - How to allocate limited cognitive resources
    - How to handle interruptions and task switching
    - When to slow down and think carefully
    
    The control system monitors cognitive load and adjusts processing
    to maintain efficient and appropriate cognitive performance.
    """
    
    # Default thresholds and parameters
    DEFAULT_SETTINGS = {
        'processing_depth': 0.7,           # Default depth (moderately deep)
        'speed_accuracy_tradeoff': 0.6,     # Slight preference for accuracy
        'cognitive_load_threshold': 0.8,    # Alert when load > threshold
        'min_confidence_for_action': 0.6,   # Minimum confidence to act
        'interruption_priority_threshold': 3  # Interrupt for priority >= this
    }
    
    # Processing time estimates by mode (ms per unit complexity)
    MODE_TIME_ESTIMATES = {
        ProcessingMode.FAST: 20,
        ProcessingMode.NORMAL: 50,
        ProcessingMode.DEEP: 150,
        ProcessingMode.CRITICAL: 300
    }
    
    # Depth to confidence mapping
    DEPTH_CONFIDENCE_MAP = {
        0.0: 0.3,
        0.2: 0.4,
        0.4: 0.6,
        0.6: 0.75,
        0.8: 0.85,
        1.0: 0.95
    }
    
    # Cognitive load decay rate (per second)
    COGNITIVE_LOAD_DECAY = 0.05
    
    def __init__(self, 
                 thought_monitor: Optional[Any] = None, 
                 confidence_scorer: Optional[Any] = None, 
                 personality: Optional[Any] = None):
        """
        Initialize the cognitive control system.
        
        Args:
            thought_monitor: Reference to thought monitor for state
            confidence_scorer: Reference to confidence scorer for confidence
            personality: Reference to Wednesday's personality
        """
        self.thought_monitor = thought_monitor
        self.confidence_scorer = confidence_scorer
        self.personality = personality
        
        # Control settings
        self.settings = ControlSettings(
            processing_depth=self.DEFAULT_SETTINGS['processing_depth'],
            speed_accuracy_tradeoff=self.DEFAULT_SETTINGS['speed_accuracy_tradeoff'],
            cognitive_load=0.0,
            processing_mode=ProcessingMode.NORMAL
        )
        
        # Task queue
        self.task_queue: deque = deque()
        self.current_task: Optional[str] = None
        self.current_task_priority: Optional[TaskPriority] = None
        self.current_task_start_time: Optional[float] = None
        self.current_task_allocation: Optional[ResourceAllocation] = None
        
        # Interruption management
        self.interruptions: List[InterruptionRecord] = []
        self.active_interruptions: List[InterruptionRecord] = []
        
        # Performance tracking
        self.allocations_made: List[ResourceAllocation] = []
        self.task_completions: List[Dict[str, Any]] = []
        self.max_history = 100
        
        # Statistics
        self.total_tasks = 0
        self.interruptions_handled = 0
        self.switch_count = 0
        self.last_load_update = time.time()
        
        logger.info("CognitiveControl initialized")
    
    def allocate_resources(self, 
                           task: str, 
                           priority: Union[TaskPriority, int],
                           complexity: float = 0.5,
                           time_available: Optional[float] = None) -> ResourceAllocation:
        """
        Determine how many cognitive resources to allocate to a task.
        
        Args:
            task: Description of the task
            priority: Priority of the task
            complexity: Task complexity (0-1)
            time_available: Time available (seconds), None if unlimited
            
        Returns:
            ResourceAllocation with allocation decisions
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not task:
            raise ValueError("task cannot be empty")
        
        # Convert priority if int
        if isinstance(priority, int):
            priority = TaskPriority.from_int(priority)
        
        if not 0 <= complexity <= 1:
            raise ValueError(f"complexity must be between 0 and 1, got {complexity}")
        
        # Update cognitive load based on current state
        self._update_cognitive_load()
        
        # Determine base depth based on priority and complexity
        base_depth = self._calculate_base_depth(priority, complexity)
        
        # Adjust for available time
        if time_available is not None and time_available > 0:
            depth = self._adjust_for_time(base_depth, complexity, time_available)
        else:
            depth = base_depth
        
        # Apply speed-accuracy tradeoff
        depth = self._apply_tradeoff(depth)
        
        # Get processing mode
        mode = self._depth_to_mode(depth)
        
        # Calculate estimated processing time
        estimated_time = self._estimate_processing_time(complexity, mode)
        
        # Calculate expected confidence after processing
        expected_confidence = self._calculate_expected_confidence(depth, task)
        
        # Check against current cognitive load
        if self.settings.cognitive_load > self.DEFAULT_SETTINGS['cognitive_load_threshold']:
            # Under load, reduce depth
            load_factor = 1 - (self.settings.cognitive_load - 0.5) * 0.5
            depth = max(0.2, depth * load_factor)
            mode = self._depth_to_mode(depth)
            estimated_time *= (1 + self.settings.cognitive_load * 0.3)
        
        # Create allocation
        allocation = ResourceAllocation(
            task=task,
            priority=priority,
            allocated_depth=depth,
            allocated_time_ms=estimated_time,
            expected_confidence=expected_confidence,
            processing_mode=mode
        )
        
        # Store allocation
        self.allocations_made.append(allocation)
        if len(self.allocations_made) > self.max_history:
            self.allocations_made.pop(0)
        
        logger.debug(f"Allocated resources for '{task[:30]}...': "
                    f"depth={depth:.2f}, mode={mode.value}, time={estimated_time:.0f}ms")
        
        return allocation
    
    def adjust_for_difficulty(self, task_difficulty: float) -> ControlSettings:
        """
        Adjust cognitive settings based on task difficulty.
        
        Args:
            task_difficulty: Difficulty of current task (0-1)
            
        Returns:
            Updated control settings
            
        Raises:
            ValueError: If task_difficulty is outside valid range
        """
        if not 0 <= task_difficulty <= 1:
            raise ValueError(f"task_difficulty must be between 0 and 1, got {task_difficulty}")
        
        # Increase depth for difficult tasks
        if task_difficulty > 0.7:
            self.settings.processing_depth = min(1.0, 
                self.settings.processing_depth + 0.2)
            self.settings.processing_mode = ProcessingMode.DEEP
        elif task_difficulty > 0.4:
            self.settings.processing_depth = min(0.9, 
                self.settings.processing_depth + 0.05)
            self.settings.processing_mode = ProcessingMode.NORMAL
        else:
            # Easy tasks can be faster
            self.settings.processing_depth = max(0.3, 
                self.settings.processing_depth - 0.05)
            self.settings.processing_mode = ProcessingMode.FAST
        
        # Increase speed-accuracy tradeoff for difficult tasks
        self.settings.speed_accuracy_tradeoff = min(0.9,
            self.settings.speed_accuracy_tradeoff + task_difficulty * 0.2)
        
        # Update required confidence based on difficulty
        self.settings.required_confidence = 0.5 + task_difficulty * 0.3
        
        logger.debug(f"Adjusted for difficulty={task_difficulty:.2f}: "
                    f"mode={self.settings.processing_mode.value}, "
                    f"depth={self.settings.processing_depth:.2f}")
        
        return self.settings
    
    def manage_interruptions(self, 
                              interrupt_priority: Union[TaskPriority, int],
                              interrupt_source: str) -> InterruptionDecision:
        """
        Decide how to handle an interruption.
        
        Args:
            interrupt_priority: Priority of the interrupting task
            interrupt_source: Source of the interruption
            
        Returns:
            Decision on how to handle
            
        Raises:
            ValueError: If interrupt_source is empty
        """
        if not interrupt_source:
            raise ValueError("interrupt_source cannot be empty")
        
        # Convert priority if int
        if isinstance(interrupt_priority, int):
            interrupt_priority = TaskPriority.from_int(interrupt_priority)
        
        self.interruptions_handled += 1
        
        # Create interruption record
        interruption = InterruptionRecord(
            source=interrupt_source,
            priority=interrupt_priority
        )
        self.interruptions.append(interruption)
        
        # Determine decision
        if not self.current_task:
            decision = InterruptionDecision.ACCEPT
        
        elif interrupt_priority.value > (self.current_task_priority.value if self.current_task_priority else 0):
            decision = InterruptionDecision.ACCEPT
            self.switch_count += 1
            logger.debug(f"Switching from {self.current_task} to {interrupt_source}")
        
        elif interrupt_priority == self.current_task_priority:
            # Same priority - decide based on progress
            if self._is_current_task_near_completion():
                decision = InterruptionDecision.DEFER
            else:
                decision = InterruptionDecision.QUEUE
        
        else:
            # Lower priority
            decision = InterruptionDecision.REJECT
        
        interruption.decision = decision
        
        # Handle based on decision
        if decision == InterruptionDecision.ACCEPT:
            self._handle_accept_interruption(interruption)
        elif decision == InterruptionDecision.QUEUE:
            self._handle_queue_interruption(interruption)
        
        logger.debug(f"Interruption from {interrupt_source} (priority {interrupt_priority.value}): "
                    f"decision={decision.value}")
        
        return decision
    
    def start_task(self, task: str, allocation: ResourceAllocation) -> None:
        """
        Start processing a task with allocated resources.
        
        Args:
            task: Task to start
            allocation: Resource allocation from allocate_resources
            
        Raises:
            ValueError: If task is empty or allocation is invalid
        """
        if not task:
            raise ValueError("task cannot be empty")
        
        # Update settings for this task
        self.settings.processing_depth = allocation.allocated_depth
        self.settings.processing_mode = allocation.processing_mode
        self.settings.estimated_time_available = allocation.allocated_time_ms / 1000
        
        self.current_task = task
        self.current_task_priority = allocation.priority
        self.current_task_start_time = time.time()
        self.current_task_allocation = allocation
        
        # Update thought monitor if available
        if self.thought_monitor and hasattr(self.thought_monitor, 'log_thought'):
            try:
                self.thought_monitor.log_thought(
                    content=f"Starting task: {task[:50]}",
                    category=ThoughtCategory.PLANNING,
                    intensity=allocation.allocated_depth
                )
            except Exception as e:
                logger.warning(f"Failed to log thought: {e}")
        
        logger.info(f"Started task: '{task[:40]}...' with mode={allocation.processing_mode.value}")
    
    def complete_task(self, success: bool, confidence_achieved: float) -> None:
        """
        Complete current task and record performance.
        
        Args:
            success: Whether task was completed successfully
            confidence_achieved: Confidence after completion (0-1)
            
        Raises:
            ValueError: If confidence_achieved is outside valid range
        """
        if not self.current_task:
            logger.warning("No current task to complete")
            return
        
        if not 0 <= confidence_achieved <= 1:
            raise ValueError(f"confidence_achieved must be between 0 and 1, got {confidence_achieved}")
        
        # Calculate duration
        duration = time.time() - (self.current_task_start_time or time.time())
        
        # Record completion
        completion = {
            'task': self.current_task,
            'priority': self.current_task_priority.value if self.current_task_priority else 0,
            'duration_ms': duration * 1000,
            'success': success,
            'confidence_achieved': confidence_achieved,
            'mode': self.settings.processing_mode.value,
            'timestamp': time.time()
        }
        self.task_completions.append(completion)
        
        # Maintain history size
        if len(self.task_completions) > self.max_history:
            self.task_completions.pop(0)
        
        # Update statistics
        self.total_tasks += 1
        
        logger.debug(f"Completed task '{self.current_task[:30]}...': "
                    f"success={success}, duration={duration:.2f}s")
        
        # Reset current task
        self.current_task = None
        self.current_task_priority = None
        self.current_task_start_time = None
        self.current_task_allocation = None
        
        # Check for queued tasks
        if self.task_queue:
            self._process_next_queued_task()
    
    def get_current_settings(self) -> ControlSettings:
        """Get current control settings"""
        # Update cognitive load before returning
        self._update_cognitive_load()
        return self.settings
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.task_completions:
            return {'total_tasks': 0}
        
        recent = self.task_completions[-20:] if len(self.task_completions) > 20 else self.task_completions
        
        avg_duration = sum(t['duration_ms'] for t in recent) / len(recent)
        success_rate = sum(1 for t in recent if t['success']) / len(recent)
        
        # Mode distribution
        mode_counts = {}
        for t in recent:
            mode = t['mode']
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        return {
            'total_tasks': self.total_tasks,
            'recent_tasks': len(recent),
            'avg_duration_ms': round(avg_duration, 1),
            'success_rate': round(success_rate, 3),
            'interruptions_handled': self.interruptions_handled,
            'task_switches': self.switch_count,
            'mode_distribution': mode_counts,
            'current_load': round(self.settings.cognitive_load, 3),
            'current_mode': self.settings.processing_mode.value
        }
    
    def set_processing_mode(self, mode: Union[ProcessingMode, str]) -> None:
        """
        Manually set processing mode.
        
        Args:
            mode: Desired processing mode
            
        Raises:
            ValueError: If mode is invalid
        """
        if isinstance(mode, str):
            try:
                mode = ProcessingMode(mode)
            except ValueError:
                raise ValueError(f"Invalid processing mode: {mode}")
        
        self.settings.processing_mode = mode
        
        # Update depth based on mode
        mode_depth = {
            ProcessingMode.FAST: 0.3,
            ProcessingMode.NORMAL: 0.6,
            ProcessingMode.DEEP: 0.8,
            ProcessingMode.CRITICAL: 0.95
        }
        self.settings.processing_depth = mode_depth.get(mode, 0.6)
        
        logger.info(f"Processing mode set to {mode.value}")
    
    def get_interruption_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent interruption history"""
        if limit <= 0:
            return []
        return [i.to_dict() for i in self.interruptions[-limit:]]
    
    def _update_cognitive_load(self) -> None:
        """Update current cognitive load estimate with decay"""
        current_time = time.time()
        time_delta = current_time - self.last_load_update
        
        # Apply decay
        if time_delta > 0:
            decay_factor = max(0.5, 1.0 - self.COGNITIVE_LOAD_DECAY * time_delta)
            self.settings.cognitive_load *= decay_factor
        
        base_load = 0.3  # Base idle load
        
        # Add load from current task depth
        if self.current_task:
            base_load += self.settings.processing_depth * 0.4
        
        # Add load from thought monitor if available
        if self.thought_monitor and hasattr(self.thought_monitor, 'current_thought'):
            if self.thought_monitor.current_thought:
                base_load += self.thought_monitor.current_thought.intensity * 0.2
        
        # Add load from task queue
        queue_factor = min(0.3, len(self.task_queue) * 0.05)
        base_load += queue_factor
        
        # Combine with existing load (weighted average)
        self.settings.cognitive_load = min(1.0, 
            self.settings.cognitive_load * 0.7 + base_load * 0.3)
        
        self.last_load_update = current_time
    
    def _calculate_base_depth(self, priority: TaskPriority, complexity: float) -> float:
        """Calculate base processing depth from priority and complexity"""
        # Priority contribution
        priority_factor = {
            TaskPriority.TRIVIAL: 0.2,
            TaskPriority.LOW: 0.3,
            TaskPriority.NORMAL: 0.5,
            TaskPriority.HIGH: 0.7,
            TaskPriority.CRITICAL: 0.85,
            TaskPriority.EMERGENCY: 0.9
        }.get(priority, 0.5)
        
        # Complexity contribution
        complexity_factor = complexity
        
        # Combine (weighted average)
        depth = priority_factor * 0.6 + complexity_factor * 0.4
        
        return max(0.1, min(1.0, depth))
    
    def _adjust_for_time(self, base_depth: float, complexity: float, 
                          time_available: float) -> float:
        """Adjust depth based on time available"""
        # Estimate required time for this depth
        mode = self._depth_to_mode(base_depth)
        estimated_time = self._estimate_processing_time(complexity, mode) / 1000
        
        if estimated_time <= time_available:
            # Enough time, maintain depth
            return base_depth
        else:
            # Not enough time, reduce depth
            time_ratio = time_available / estimated_time
            reduced_depth = base_depth * time_ratio
            return max(0.2, reduced_depth)
    
    def _apply_tradeoff(self, depth: float) -> float:
        """Apply speed-accuracy tradeoff setting"""
        # speed_accuracy_tradeoff: 0=fast, 1=accurate
        # Higher tradeoff = prefer depth
        adjusted = depth * (0.5 + self.settings.speed_accuracy_tradeoff * 0.5)
        return max(0.1, min(1.0, adjusted))
    
    def _depth_to_mode(self, depth: float) -> ProcessingMode:
        """Convert depth to processing mode"""
        if depth >= 0.85:
            return ProcessingMode.CRITICAL
        elif depth >= 0.65:
            return ProcessingMode.DEEP
        elif depth >= 0.35:
            return ProcessingMode.NORMAL
        else:
            return ProcessingMode.FAST
    
    def _estimate_processing_time(self, complexity: float, mode: ProcessingMode) -> float:
        """Estimate processing time in milliseconds"""
        base_time = self.MODE_TIME_ESTIMATES.get(mode, 50)
        return base_time * (0.5 + complexity)
    
    def _calculate_expected_confidence(self, depth: float, task: str) -> float:
        """Calculate expected confidence after processing"""
        # Find nearest depth in map
        depths = sorted(self.DEPTH_CONFIDENCE_MAP.keys())
        confidence = self.DEPTH_CONFIDENCE_MAP[depths[-1]]
        
        for d in depths:
            if depth <= d:
                confidence = self.DEPTH_CONFIDENCE_MAP[d]
                break
        
        # Adjust based on task similarity to past performance
        if self.confidence_scorer and hasattr(self.confidence_scorer, '_score_performance_history'):
            try:
                past_performance = self.confidence_scorer._score_performance_history(task)
                confidence = confidence * 0.7 + past_performance * 0.3
            except Exception as e:
                logger.warning(f"Failed to get past performance: {e}")
        
        return max(0.2, min(1.0, confidence))
    
    def _is_current_task_near_completion(self) -> bool:
        """Check if current task is nearly complete"""
        if not self.current_task_start_time or not self.current_task_allocation:
            return False
        
        duration = time.time() - self.current_task_start_time
        estimated_total = self.settings.estimated_time_available
        
        if estimated_total <= 0:
            return False
        
        return duration / estimated_total > 0.8
    
    def _handle_accept_interruption(self, interruption: InterruptionRecord) -> None:
        """Handle accepting an interruption"""
        # Record current task to queue if not already there
        if self.current_task and self.current_task_priority:
            self.task_queue.append(QueuedTask(
                task=self.current_task,
                priority=self.current_task_priority,
                start_time=self.current_task_start_time,
                is_interruption=False
            ))
        
        interruption.resolved = True
    
    def _handle_queue_interruption(self, interruption: InterruptionRecord) -> None:
        """Handle queueing an interruption"""
        self.task_queue.append(QueuedTask(
            task=interruption.source,
            priority=interruption.priority,
            is_interruption=True,
            timestamp=interruption.timestamp
        ))
        interruption.resolved = True
    
    def _process_next_queued_task(self) -> None:
        """Process next task from queue"""
        if self.task_queue:
            next_task = self.task_queue.popleft()
            logger.debug(f"Processing queued task: {next_task.task[:30]}")
            
            # If we have allocation info, start the task
            if self.confidence_scorer:
                allocation = self.allocate_resources(
                    task=next_task.task,
                    priority=next_task.priority,
                    complexity=0.5
                )
                self.start_task(next_task.task, allocation)
    
    def reset(self) -> None:
        """Reset cognitive control state"""
        self.task_queue.clear()
        self.current_task = None
        self.current_task_priority = None
        self.current_task_start_time = None
        self.current_task_allocation = None
        self.interruptions.clear()
        self.active_interruptions.clear()
        self.allocations_made.clear()
        self.task_completions.clear()
        self.total_tasks = 0
        self.interruptions_handled = 0
        self.switch_count = 0
        self.settings.cognitive_load = 0.0
        self.last_load_update = time.time()
        logger.info("CognitiveControl reset")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("=== Cognitive Control Test ===\n")
    
    # Mock dependencies
    class MockThoughtMonitor:
        def __init__(self):
            self.current_thought = None
        def log_thought(self, content, category, intensity):
            pass
    
    class MockConfidenceScorer:
        def _score_performance_history(self, task):
            return 0.7
    
    # Create cognitive control
    control = CognitiveControl(
        thought_monitor=MockThoughtMonitor(),
        confidence_scorer=MockConfidenceScorer()
    )
    
    # Test resource allocation for different tasks
    print("--- Resource Allocation ---")
    
    test_tasks = [
        ("Small talk response", TaskPriority.LOW, 0.3),
        ("Answer user question", TaskPriority.NORMAL, 0.5),
        ("Solve complex mystery", TaskPriority.HIGH, 0.8),
        ("Emergency response", TaskPriority.EMERGENCY, 0.9),
    ]
    
    for task, priority, complexity in test_tasks:
        allocation = control.allocate_resources(
            task=task,
            priority=priority,
            complexity=complexity,
            time_available=5.0
        )
        print(f"\nTask: {task}")
        print(f"  Mode: {allocation.processing_mode.value}")
        print(f"  Depth: {allocation.allocated_depth:.2f}")
        print(f"  Est. time: {allocation.allocated_time_ms:.0f}ms")
        print(f"  Expected confidence: {allocation.expected_confidence:.2f}")
    
    # Test difficulty adjustment
    print("\n--- Difficulty Adjustment ---")
    
    difficulties = [0.2, 0.5, 0.8, 0.95]
    for diff in difficulties:
        settings = control.adjust_for_difficulty(diff)
        print(f"Difficulty {diff:.1f}: mode={settings.processing_mode.value}, "
              f"depth={settings.processing_depth:.2f}, "
              f"required_conf={settings.required_confidence:.2f}")
    
    # Test interruption management
    print("\n--- Interruption Management ---")
    
    # Start a high-priority task
    allocation = control.allocate_resources("Complex analysis", TaskPriority.HIGH, 0.8)
    control.start_task("Complex analysis", allocation)
    print("Started: Complex analysis (HIGH priority)")
    
    # Simulate interruptions
    interruptions = [
        (TaskPriority.LOW, "Weather update"),
        (TaskPriority.HIGH, "User asked important question"),
        (TaskPriority.CRITICAL, "System alert"),
        (TaskPriority.TRIVIAL, "Background process"),
    ]
    
    for priority, source in interruptions:
        decision = control.manage_interruptions(priority, source)
        print(f"  Interruption: {source} (priority {priority.value}) -> {decision.value}")
    
    # Complete task
    control.complete_task(success=True, confidence_achieved=0.85)
    print("\nCompleted: Complex analysis")
    
    # Get performance stats
    print("\n--- Performance Statistics ---")
    stats = control.get_performance_stats()
    for key, value in stats.items():
        if key != 'mode_distribution':
            print(f"  {key}: {value}")
    
    if 'mode_distribution' in stats and stats['mode_distribution']:
        print(f"  Mode distribution: {stats['mode_distribution']}")
    
    # Get interruption history
    print("\n--- Interruption History ---")
    history = control.get_interruption_history(limit=3)
    for record in history:
        print(f"  {record['source']}: {record['decision']} (priority {record['priority']})")
    
    print("\n=== Test Complete ===")