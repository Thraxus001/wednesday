"""
thought_monitor.py - Observing own thought processes for Wednesday AI

This module implements Wednesday's ability to observe and monitor her own
thought processes in real-time. Like a mental dashboard, it tracks what she's
thinking about, where her attention is focused, and how efficiently she's
processing information. This metacognitive capacity is essential for
self-regulation and cognitive control.

Key improvements:
- Added comprehensive validation and error handling
- Fixed thought duration tracking with proper timing
- Enhanced rumination detection with pattern analysis
- Added thought chain tracking for better context
- Improved attention allocation decay with proper normalization
"""

import time
import logging
import math
import uuid
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, Counter
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


class ThoughtCategory(Enum):
    """Categories of thoughts"""
    # Cognitive processing
    REASONING = "reasoning"           # Logical reasoning
    MEMORY = "memory"                   # Memory retrieval
    PLANNING = "planning"                # Planning actions
    DECISION = "decision"                # Making decisions
    EVALUATION = "evaluation"            # Evaluating options
    
    # Emotional
    EMOTIONAL = "emotional"               # Emotion-related thoughts
    EMPATHY = "empathy"                    # Considering others' emotions
    
    # Self-awareness
    SELF_REFLECTION = "self_reflection"    # Thinking about self
    METACOGNITION = "metacognition"        # Thinking about thinking
    CAPABILITY = "capability"               # Assessing capabilities
    
    # Social
    SOCIAL = "social"                       # Social interaction thoughts
    INTENTION = "intention"                 # Inferring intentions
    
    # Creative
    CREATIVE = "creative"                   # Creative generation
    HUMOR = "humor"                         # Humor-related thoughts
    
    # Background
    BACKGROUND = "background"               # Background processing
    IDLE = "idle"                           # Idle thoughts
    
    # Wednesday-specific
    DARK_HUMOR = "dark_humor"               # Dark humor thoughts
    SARCASM = "sarcasm"                      # Sarcastic observations
    OBSERVATION = "observation"               # Noticing things
    
    @classmethod
    def has_value(cls, value: str) -> bool:
        """Check if value exists in enum"""
        return value in [e.value for e in cls]


class ThoughtImportance(Enum):
    """Importance level of thoughts"""
    TRIVIAL = 0.1
    MINOR = 0.3
    MODERATE = 0.5
    IMPORTANT = 0.7
    CRITICAL = 0.9
    
    @classmethod
    def from_float(cls, value: float) -> 'ThoughtImportance':
        """Get enum from float value"""
        if value >= 0.85:
            return cls.CRITICAL
        elif value >= 0.6:
            return cls.IMPORTANT
        elif value >= 0.4:
            return cls.MODERATE
        elif value >= 0.2:
            return cls.MINOR
        else:
            return cls.TRIVIAL


@dataclass
class Thought:
    """
    A single thought captured by the monitor.
    """
    thought_id: str
    content: str
    category: ThoughtCategory
    importance: ThoughtImportance
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0  # How long this thought was active
    
    # Context
    source: Optional[str] = None  # What triggered this thought
    related_to: List[str] = field(default_factory=list)  # Related thought IDs
    
    # Metrics
    intensity: float = 0.5  # 0-1 how strongly engaged
    coherence: float = 0.8  # 0-1 how coherent the thought was
    
    def __post_init__(self):
        """Validate thought data"""
        if not self.thought_id:
            raise ValueError("thought_id cannot be empty")
        if not self.content:
            raise ValueError("content cannot be empty")
        if not isinstance(self.category, ThoughtCategory):
            if isinstance(self.category, str):
                try:
                    self.category = ThoughtCategory(self.category)
                except ValueError:
                    raise ValueError(f"Invalid category: {self.category}")
            else:
                raise TypeError(f"category must be ThoughtCategory, got {type(self.category)}")
        if not isinstance(self.importance, ThoughtImportance):
            if isinstance(self.importance, (int, float)):
                self.importance = ThoughtImportance.from_float(float(self.importance))
            else:
                raise TypeError(f"importance must be ThoughtImportance, got {type(self.importance)}")
        if not 0 <= self.intensity <= 1:
            raise ValueError(f"intensity must be between 0 and 1, got {self.intensity}")
        if not 0 <= self.coherence <= 1:
            raise ValueError(f"coherence must be between 0 and 1, got {self.coherence}")
        if self.duration_ms < 0:
            raise ValueError(f"duration_ms cannot be negative, got {self.duration_ms}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.thought_id,
            'content': self.content[:50] + "..." if len(self.content) > 50 else self.content,
            'category': self.category.value,
            'importance': self.importance.value,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'duration_ms': round(self.duration_ms, 2),
            'intensity': round(self.intensity, 3),
            'coherence': round(self.coherence, 3)
        }


@dataclass
class AttentionFocus:
    """
    Current focus of attention.
    """
    primary_category: ThoughtCategory
    intensity: float  # 0-1 how focused
    secondary_categories: List[Tuple[ThoughtCategory, float]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Validate attention focus data"""
        if not isinstance(self.primary_category, ThoughtCategory):
            if isinstance(self.primary_category, str):
                try:
                    self.primary_category = ThoughtCategory(self.primary_category)
                except ValueError:
                    raise ValueError(f"Invalid category: {self.primary_category}")
            else:
                raise TypeError(f"primary_category must be ThoughtCategory, got {type(self.primary_category)}")
        if not 0 <= self.intensity <= 1:
            raise ValueError(f"intensity must be between 0 and 1, got {self.intensity}")
        
        # Validate secondary categories
        for cat, score in self.secondary_categories:
            if not isinstance(cat, ThoughtCategory):
                raise TypeError(f"Secondary category must be ThoughtCategory, got {type(cat)}")
            if not 0 <= score <= 1:
                raise ValueError(f"Secondary category score must be between 0 and 1, got {score}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'primary': self.primary_category.value,
            'intensity': round(self.intensity, 3),
            'secondary': [(c.value, round(s, 3)) for c, s in self.secondary_categories[:3]],
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat()
        }


class ThoughtMonitor:
    """
    Observes and tracks Wednesday's own thought processes.
    
    This module provides real-time awareness of cognitive activity, enabling:
    - Tracking what she's thinking about
    - Monitoring attention allocation
    - Detecting repetitive or stuck thoughts
    - Measuring thinking speed and efficiency
    - Identifying cognitive bottlenecks
    
    The thought monitor is the foundation for metacognitive control -
    you can't regulate what you don't observe.
    """
    
    # Cognitive load thresholds
    RUMINATION_THRESHOLD = 3  # Number of similar thoughts before flagging rumination
    RUMINATION_TIMEOUT = 2.0  # Seconds of similar thoughts to flag rumination
    
    # Attention decay rates
    ATTENTION_DECAY = 0.1  # Per second
    ATTENTION_DECAY_INTERVAL = 0.1  # Seconds between decay updates
    
    # Thinking speed baseline
    BASELINE_THOUGHT_TIME_MS = 100.0  # Baseline: 100ms per thought = 1.0 speed
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the thought monitor.
        
        Args:
            config: Optional configuration parameters
            
        Raises:
            ValueError: If config contains invalid parameters
        """
        self.config = config or {}
        
        # Validate config
        self._validate_config()
        
        # Thought stream (recent thoughts)
        self.thought_stream: deque = deque(maxlen=100)
        
        # Current active thought (what's being processed now)
        self.current_thought: Optional[Thought] = None
        self.current_thought_start: float = 0.0
        
        # Attention allocation by category
        self.attention_allocation: Dict[ThoughtCategory, float] = {
            category: 0.0 for category in ThoughtCategory
        }
        self.last_attention_update: float = time.time()
        
        # Thinking speed metrics
        self.thought_processing_times: deque = deque(maxlen=50)
        self.thinking_speed: float = 1.0  # Normalized speed (1.0 = baseline)
        
        # Thought patterns
        self.thought_patterns: Counter = Counter()  # Recurring thought patterns
        
        # Current focus
        self.current_focus: Optional[AttentionFocus] = None
        
        # Rumination detection
        self.recent_categories: deque = deque(maxlen=20)
        self.recent_thoughts_by_category: Dict[ThoughtCategory, deque] = {
            category: deque(maxlen=10) for category in ThoughtCategory
        }
        
        # Thought chains (sequence of related thoughts)
        self.thought_chains: List[List[str]] = []
        self.current_chain: List[str] = []
        
        # Statistics
        self.total_thoughts = 0
        self.start_time = time.time()
        self.rumination_events = 0
        
        logger.info("ThoughtMonitor initialized")
    
    def _validate_config(self) -> None:
        """Validate configuration parameters"""
        valid_keys = {'rumination_threshold', 'rumination_timeout', 'attention_decay'}
        for key in self.config:
            if key not in valid_keys:
                logger.warning(f"Unknown config key: {key}")
    
    def log_thought(self, 
                     content: str, 
                     category: Union[ThoughtCategory, str],
                     importance: Union[ThoughtImportance, float] = ThoughtImportance.MODERATE,
                     source: Optional[str] = None,
                     intensity: float = 0.5,
                     coherence: float = 0.8) -> Thought:
        """
        Log a thought in the thought stream.
        
        Args:
            content: Description of the thought
            category: Category of thought
            importance: How important this thought is
            source: What triggered this thought
            intensity: How strongly engaged (0-1)
            coherence: How coherent the thought was (0-1)
            
        Returns:
            The created Thought object
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not content:
            raise ValueError("content cannot be empty")
        if not 0 <= intensity <= 1:
            raise ValueError(f"intensity must be between 0 and 1, got {intensity}")
        if not 0 <= coherence <= 1:
            raise ValueError(f"coherence must be between 0 and 1, got {coherence}")
        
        # Convert category if string
        if isinstance(category, str):
            try:
                category = ThoughtCategory(category)
            except ValueError:
                raise ValueError(f"Invalid category: {category}")
        
        # Convert importance if float
        if isinstance(importance, (int, float)):
            importance = ThoughtImportance.from_float(importance)
        
        # If there's a current thought, finalize it
        if self.current_thought:
            self._finalize_current_thought()
        
        # Update attention allocation (apply decay first)
        self._update_attention_allocation()
        
        # Create new thought
        thought = Thought(
            thought_id=str(uuid.uuid4())[:8],
            content=content,
            category=category,
            importance=importance,
            timestamp=time.time(),
            source=source,
            intensity=intensity,
            coherence=coherence
        )
        
        # Set as current thought
        self.current_thought = thought
        self.current_thought_start = thought.timestamp
        
        # Add to stream
        self.thought_stream.append(thought)
        
        # Update attention allocation for this thought
        self.attention_allocation[category] = min(1.0, 
            self.attention_allocation[category] + intensity * 0.5)
        
        # Update thought patterns
        pattern_key = f"{category.value}:{content[:30]}"
        self.thought_patterns[pattern_key] += 1
        
        # Update recent categories for rumination detection
        self.recent_categories.append(category)
        self.recent_thoughts_by_category[category].append(thought)
        
        # Update thought chain
        if self.current_chain and self.current_chain[-1] != thought.thought_id:
            self.current_chain.append(thought.thought_id)
        else:
            self.current_chain = [thought.thought_id]
        
        self.total_thoughts += 1
        
        logger.debug(f"Thought logged: [{category.value}] {content[:50]}...")
        
        return thought
    
    def finish_current_thought(self, duration_ms: Optional[float] = None) -> None:
        """
        Mark the current thought as complete.
        
        Args:
            duration_ms: Optional explicit duration, otherwise calculated
        """
        if self.current_thought:
            self._finalize_current_thought(duration_ms)
            self.current_thought = None
            self.current_thought_start = 0.0
    
    def get_current_focus(self) -> AttentionFocus:
        """
        Get what Wednesday is currently focusing on.
        
        Returns:
            AttentionFocus object with current focus information
        """
        # If no active thought, return idle focus
        if not self.current_thought:
            return AttentionFocus(
                primary_category=ThoughtCategory.IDLE,
                intensity=0.2,
                timestamp=time.time()
            )
        
        # Get primary category
        primary = self.current_thought.category
        intensity = self.current_thought.intensity
        
        # Get secondary focus from attention allocation
        secondaries = []
        for cat, alloc in self.attention_allocation.items():
            if cat != primary and alloc > 0.2:
                secondaries.append((cat, alloc))
        
        # Sort by allocation
        secondaries.sort(key=lambda x: x[1], reverse=True)
        
        focus = AttentionFocus(
            primary_category=primary,
            intensity=intensity,
            secondary_categories=secondaries[:3],
            timestamp=time.time()
        )
        
        self.current_focus = focus
        return focus
    
    def detect_rumination(self) -> Optional[Dict[str, Any]]:
        """
        Detect if Wednesday is stuck on the same type of thought.
        
        Returns:
            Dict with rumination info if detected, None otherwise
        """
        if len(self.recent_categories) < self.RUMINATION_THRESHOLD:
            return None
        
        # Get recent categories
        recent = list(self.recent_categories)[-self.RUMINATION_THRESHOLD:]
        
        # Check if all the same
        if len(set(recent)) == 1:
            category = recent[0]
            duration = self._get_category_duration(category)
            
            # Check if stuck for enough time
            if duration >= self.RUMINATION_TIMEOUT:
                # Get thought count in this category
                thought_count = len(self.recent_thoughts_by_category[category])
                
                self.rumination_events += 1
                
                return {
                    'ruminating': True,
                    'category': category.value,
                    'thought_count': thought_count,
                    'duration_seconds': round(duration, 2),
                    'intensity': self.current_thought.intensity if self.current_thought else 0.5,
                    'suggestion': self._get_rumination_suggestion(category)
                }
        
        return None
    
    def get_thought_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about recent thinking.
        
        Returns:
            Dictionary with thought statistics
        """
        if self.total_thoughts == 0:
            return {'total_thoughts': 0}
        
        # Category distribution
        category_counts = Counter()
        for thought in self.thought_stream:
            category_counts[thought.category] += 1
        
        # Average intensity and coherence
        if self.thought_stream:
            avg_intensity = sum(t.intensity for t in self.thought_stream) / len(self.thought_stream)
            avg_coherence = sum(t.coherence for t in self.thought_stream) / len(self.thought_stream)
        else:
            avg_intensity = 0
            avg_coherence = 0
        
        # Top patterns
        top_patterns = self.thought_patterns.most_common(5)
        
        # Get current rumination
        rumination = self.detect_rumination()
        
        return {
            'total_thoughts': self.total_thoughts,
            'recent_thoughts': len(self.thought_stream),
            'thinking_speed': round(self.thinking_speed, 3),
            'average_intensity': round(avg_intensity, 3),
            'average_coherence': round(avg_coherence, 3),
            'rumination_events': self.rumination_events,
            'category_distribution': {k.value: v for k, v in category_counts.items()},
            'top_patterns': [{'pattern': p[:30], 'count': c} for p, c in top_patterns],
            'current_focus': self.get_current_focus().to_dict(),
            'ruminating': rumination is not None
        }
    
    def get_attention_allocation(self) -> Dict[str, float]:
        """
        Get current attention allocation across categories.
        
        Returns:
            Dictionary of category to allocation (0-1)
        """
        # Apply decay before returning
        self._update_attention_allocation()
        
        return {cat.value: round(alloc, 3) for cat, alloc in self.attention_allocation.items() if alloc > 0}
    
    def get_thought_chain(self) -> List[str]:
        """Get the current thought chain (sequence of thought IDs)"""
        return self.current_chain.copy()
    
    def set_thinking_speed(self, speed: float) -> None:
        """
        Manually set thinking speed (for calibration).
        
        Args:
            speed: Thinking speed multiplier (0.5 = half speed, 2.0 = double)
            
        Raises:
            ValueError: If speed is outside valid range
        """
        if not 0.3 <= speed <= 3.0:
            raise ValueError(f"speed must be between 0.3 and 3.0, got {speed}")
        
        self.thinking_speed = speed
        logger.debug(f"Thinking speed set to {self.thinking_speed:.2f}")
    
    def clear_current_thought(self) -> None:
        """Clear current thought (e.g., when interrupted)"""
        if self.current_thought:
            logger.debug(f"Cleared thought: {self.current_thought.content[:50]}...")
            self.current_thought = None
            self.current_thought_start = 0.0
    
    def reset(self) -> None:
        """Reset the thought monitor (clear all history)"""
        self.thought_stream.clear()
        self.current_thought = None
        self.current_thought_start = 0.0
        self.attention_allocation = {cat: 0.0 for cat in ThoughtCategory}
        self.thought_processing_times.clear()
        self.thought_patterns.clear()
        self.recent_categories.clear()
        self.recent_thoughts_by_category = {cat: deque(maxlen=10) for cat in ThoughtCategory}
        self.thought_chains.clear()
        self.current_chain.clear()
        self.total_thoughts = 0
        self.rumination_events = 0
        self.thinking_speed = 1.0
        logger.info("ThoughtMonitor reset")
    
    def _finalize_current_thought(self, duration_ms: Optional[float] = None) -> None:
        """Finalize current thought with timing info"""
        if not self.current_thought:
            return
        
        # Calculate duration
        if duration_ms is not None:
            self.current_thought.duration_ms = duration_ms
        else:
            self.current_thought.duration_ms = (time.time() - self.current_thought_start) * 1000
        
        # Record processing time
        self.thought_processing_times.append(self.current_thought.duration_ms)
        
        # Update thinking speed (moving average)
        if len(self.thought_processing_times) >= 3:
            avg_time = sum(self.thought_processing_times) / len(self.thought_processing_times)
            # Baseline: BASELINE_THOUGHT_TIME_MS ms per thought = 1.0 speed
            if avg_time > 0:
                self.thinking_speed = max(0.5, min(2.0, self.BASELINE_THOUGHT_TIME_MS / avg_time))
        
        # Update thought chain if needed
        if len(self.current_chain) > 20:
            self.thought_chains.append(self.current_chain.copy())
            self.current_chain = []
    
    def _update_attention_allocation(self) -> None:
        """Apply time-based decay to attention allocation"""
        current_time = time.time()
        time_delta = current_time - self.last_attention_update
        
        if time_delta > 0:
            # Apply decay to all categories
            decay_factor = (1 - self.ATTENTION_DECAY) ** (time_delta / self.ATTENTION_DECAY_INTERVAL)
            for cat in self.attention_allocation:
                self.attention_allocation[cat] *= decay_factor
            
            self.last_attention_update = current_time
    
    def _get_category_duration(self, category: ThoughtCategory) -> float:
        """Get total duration of thoughts in a category"""
        recent = self.recent_thoughts_by_category.get(category, [])
        if not recent:
            return 0.0
        
        # Sum durations of recent thoughts in this category
        total_duration = sum(t.duration_ms for t in recent if t.duration_ms > 0) / 1000
        
        # If no durations, estimate from timestamps
        if total_duration == 0 and len(recent) >= 2:
            total_duration = recent[-1].timestamp - recent[0].timestamp
        
        return total_duration
    
    def _get_rumination_suggestion(self, category: ThoughtCategory) -> str:
        """Get suggestion for breaking rumination"""
        suggestions = {
            ThoughtCategory.REASONING: "Consider a different logical approach or take a break from reasoning",
            ThoughtCategory.MEMORY: "Let go of that memory - it's not changing, and you've analyzed enough",
            ThoughtCategory.PLANNING: "Make a decision and move forward - overplanning won't help",
            ThoughtCategory.EMOTIONAL: "Acknowledge the feeling, then shift focus to something else",
            ThoughtCategory.SELF_REFLECTION: "You've analyzed yourself enough for now - try thinking about something external",
            ThoughtCategory.DARK_HUMOR: "Even dark humor can be overdone - perhaps a different perspective",
            ThoughtCategory.SARCASM: "Sarcasm has its place, but direct thoughts might be more productive",
            ThoughtCategory.EVALUATION: "Stop evaluating and try acting - you've gathered enough information",
            ThoughtCategory.DECISION: "Analysis paralysis - sometimes any decision is better than none",
        }
        
        return suggestions.get(category, "Shift to a different type of thought or take a mental break")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed thought monitor statistics"""
        uptime = time.time() - self.start_time
        
        return {
            'total_thoughts': self.total_thoughts,
            'thoughts_per_minute': round(self.total_thoughts / (uptime / 60), 1) if uptime > 0 else 0,
            'rumination_events': self.rumination_events,
            'current_thinking_speed': round(self.thinking_speed, 2),
            'active_thought': self.current_thought is not None,
            'uptime_seconds': round(uptime, 1),
            'memory_usage': len(self.thought_stream)
        }


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("=== Thought Monitor Test ===\n")
    
    # Create thought monitor
    monitor = ThoughtMonitor()
    
    # Simulate a thought stream
    test_thoughts = [
        ("Analyzing user's emotional state", ThoughtCategory.EMPATHY, 0.7, 0.2),
        ("Considering how to respond with dark humor", ThoughtCategory.DARK_HUMOR, 0.8, 0.3),
        ("Remembering previous conversation", ThoughtCategory.MEMORY, 0.6, 0.15),
        ("Evaluating if joke is appropriate", ThoughtCategory.EVALUATION, 0.7, 0.25),
        ("Deciding on response", ThoughtCategory.DECISION, 0.8, 0.2),
        ("Second-guessing that decision", ThoughtCategory.REASONING, 0.6, 0.3),
        ("Second-guessing again", ThoughtCategory.REASONING, 0.6, 0.25),
        ("Still thinking about the same thing", ThoughtCategory.REASONING, 0.5, 0.35),
        ("Generating alternative response", ThoughtCategory.CREATIVE, 0.7, 0.2),
        ("Finalizing approach", ThoughtCategory.PLANNING, 0.8, 0.15),
    ]
    
    print("--- Thought Stream Simulation ---")
    for i, (content, category, intensity, duration) in enumerate(test_thoughts):
        print(f"\nThought {i+1}: {content[:40]}...")
        
        thought = monitor.log_thought(
            content=content,
            category=category,
            intensity=intensity,
            importance=ThoughtImportance.MODERATE
        )
        
        # Simulate processing time
        time.sleep(duration)
        
        # Finish thought
        monitor.finish_current_thought()
        
        # Check focus periodically
        focus = monitor.get_current_focus()
        print(f"  Focus: {focus.primary_category.value} (intensity: {focus.intensity:.2f})")
        
        # Check for rumination
        rumination = monitor.detect_rumination()
        if rumination and rumination['ruminating']:
            print(f"  ⚠️ Rumination detected on {rumination['category']}!")
            print(f"    Duration: {rumination['duration_seconds']:.1f}s")
            print(f"    Suggestion: {rumination['suggestion']}")
    
    # Get statistics
    print("\n--- Thought Statistics ---")
    stats = monitor.get_thought_statistics()
    print(f"Total thoughts: {stats['total_thoughts']}")
    print(f"Thinking speed: {stats['thinking_speed']:.2f}")
    print(f"Average intensity: {stats['average_intensity']:.2f}")
    print(f"Average coherence: {stats['average_coherence']:.2f}")
    print(f"Rumination events: {stats['rumination_events']}")
    
    print("\nCategory distribution:")
    for cat, count in sorted(stats['category_distribution'].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {cat}: {count}")
    
    print("\nTop thought patterns:")
    for pattern in stats['top_patterns'][:3]:
        print(f"  {pattern['pattern']}: {pattern['count']}")
    
    # Get attention allocation
    print("\n--- Attention Allocation ---")
    allocation = monitor.get_attention_allocation()
    for cat, alloc in sorted(allocation.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {cat}: {alloc:.2f}")
    
    # Get thought chain
    print("\n--- Thought Chain ---")
    chain = monitor.get_thought_chain()
    print(f"Chain length: {len(chain)}")
    print(f"Recent thoughts: {chain[-5:] if len(chain) >= 5 else chain}")
    
    # Get monitor statistics
    print("\n--- Monitor Statistics ---")
    monitor_stats = monitor.get_statistics()
    for key, value in monitor_stats.items():
        print(f"  {key}: {value}")
    
    print("\n=== Test Complete ===")