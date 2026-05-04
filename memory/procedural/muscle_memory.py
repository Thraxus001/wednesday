"""
Muscle Memory - Automatic, rapid-response behaviors.
Like typing without looking at keys - actions so practiced
they become automatic and don't require conscious thought.
"""
from typing import Dict, Any, Optional, List, Callable, Tuple
from datetime import datetime, timedelta
import uuid
import json
from pathlib import Path
import logging
import numpy as np
from collections import defaultdict, deque
from enum import Enum
import time

logger = logging.getLogger(__name__)

class MuscleMemoryType(Enum):
    """Types of muscle memory patterns"""
    MOTOR = "motor"  # Physical actions (for future robotics)
    VERBAL = "verbal"  # Common phrases, responses
    COGNITIVE = "cognitive"  # Quick mental patterns
    SOCIAL = "social"  # Automatic social behaviors
    EMOTIONAL = "emotional"  # Automatic emotional responses
    HABIT = "habit"  # Routine behaviors

class ActivationTrigger(Enum):
    """What triggers the muscle memory"""
    CONTEXT = "context"  # Situation-based
    STIMULUS = "stimulus"  # Direct trigger
    TIME = "time"  # Time-based
    SEQUENCE = "sequence"  # Part of a sequence
    EMOTIONAL = "emotional"  # Emotional state

class MuscleMemoryPattern:
    """
    A highly practiced, automatic response pattern.
    Executes quickly with minimal cognitive load.
    """
    
    def __init__(self,
                 name: str,
                 pattern_type: MuscleMemoryType,
                 trigger: ActivationTrigger,
                 response: Callable,
                 context_conditions: Optional[List[str]] = None,
                 metadata: Optional[Dict] = None):
        
        self.id = str(uuid.uuid4())
        self.name = name
        self.type = pattern_type
        self.trigger = trigger
        self.response = response
        self.context_conditions = context_conditions or []
        self.metadata = metadata or {}
        
        # Performance metrics
        self.execution_count = 0
        self.total_execution_time = 0.0  # Cumulative for average
        self.last_execution_time = None
        self.fastest_execution = float('inf')
        self.slowest_execution = 0.0
        
        # Activation tracking
        self.activation_history = deque(maxlen=100)
        self.success_rate = 1.0
        self.failure_count = 0
        
        # Temporal
        self.created_at = datetime.now()
        self.last_activated = None
        self.last_modified = self.created_at
        
        # Strength/reinforcement
        self.strength = 0.5  # 0-1, how ingrained
        self.reinforcement_count = 0
        self.decay_rate = 0.01  # How fast it weakens without use
        
        # Automaticity (how unconscious it is)
        self.automaticity = 0.3  # 0=conscious, 1=completely automatic
        
        # Context associations
        self.associated_contexts = defaultdict(int)  # context -> frequency
        
        logger.debug(f"Created muscle memory pattern: {name}")
    
    def execute(self, context: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
        """
        Execute the muscle memory pattern automatically.
        Measures speed and automaticity.
        """
        start_time = time.perf_counter()
        
        try:
            # Record activation context
            context_type = context.get('type', 'unknown')
            self.associated_contexts[context_type] += 1
            
            # Execute the response
            result = self.response(context, *args, **kwargs)
            
            # Calculate execution time
            exec_time = (time.perf_counter() - start_time) * 1000  # ms
            
            # Update metrics
            self.execution_count += 1
            self.total_execution_time += exec_time
            self.last_execution_time = exec_time
            self.fastest_execution = min(self.fastest_execution, exec_time)
            self.slowest_execution = max(self.slowest_execution, exec_time)
            self.last_activated = datetime.now()
            
            # Determine success
            success = result.get('success', True) if isinstance(result, dict) else True
            if not success:
                self.failure_count += 1
            
            # Update success rate
            self.success_rate = 1.0 - (self.failure_count / max(self.execution_count, 1))
            
            # Reinforce from use
            self._reinforce(exec_time)
            
            # Record activation
            self.activation_history.append({
                'timestamp': datetime.now().isoformat(),
                'execution_time': exec_time,
                'context': context_type,
                'success': success
            })
            
            logger.debug(f"Muscle memory '{self.name}' executed in {exec_time:.2f}ms")
            
            # Prepare result
            if isinstance(result, dict):
                result.update({
                    'pattern_id': self.id,
                    'execution_time_ms': exec_time,
                    'automaticity': self.automaticity,
                    'strength': self.strength
                })
                return result
            else:
                return {
                    'success': success,
                    'output': result,
                    'pattern_id': self.id,
                    'execution_time_ms': exec_time,
                    'automaticity': self.automaticity,
                    'strength': self.strength
                }
                
        except Exception as e:
            logger.error(f"Error in muscle memory '{self.name}': {e}")
            self.failure_count += 1
            
            return {
                'success': False,
                'error': str(e),
                'pattern_id': self.id,
                'execution_time_ms': (time.perf_counter() - start_time) * 1000
            }
    
    def should_activate(self, context: Dict[str, Any]) -> bool:
        """
        Determine if this muscle memory should activate automatically.
        Based on triggers, context, and automaticity.
        """
        # Check context conditions
        if self.context_conditions:
            for condition in self.context_conditions:
                if condition not in context:
                    return False
        
        # Check trigger type
        if self.trigger == ActivationTrigger.CONTEXT:
            # Match based on context
            context_type = context.get('type', '')
            if context_type in self.associated_contexts:
                # Higher automaticity means lower threshold
                threshold = 0.7 - (self.automaticity * 0.3)
                freq_ratio = self.associated_contexts[context_type] / max(sum(self.associated_contexts.values()), 1)
                return freq_ratio > threshold
        
        elif self.trigger == ActivationTrigger.STIMULUS:
            # Direct stimulus present?
            stimulus = self.metadata.get('stimulus')
            return stimulus in context.get('stimuli', [])
        
        elif self.trigger == ActivationTrigger.TIME:
            # Time-based activation
            schedule = self.metadata.get('schedule', [])
            current_hour = datetime.now().hour
            return current_hour in schedule
        
        elif self.trigger == ActivationTrigger.EMOTIONAL:
            # Emotional state trigger
            target_emotion = self.metadata.get('emotion')
            current_emotion = context.get('emotional_state')
            return target_emotion == current_emotion and self.automaticity > 0.7
        
        return False
    
    def practice(self, context: Dict[str, Any], repetitions: int = 10) -> Dict:
        """
        Deliberate practice to strengthen the muscle memory.
        """
        results = []
        
        for i in range(repetitions):
            result = self.execute(context)
            results.append(result)
            
            # Small delay between repetitions
            if i < repetitions - 1:
                time.sleep(0.05)
        
        # Calculate improvement
        avg_time = np.mean([r.get('execution_time_ms', 0) for r in results])
        success_rate = np.mean([1 if r.get('success', False) else 0 for r in results])
        
        # Strength increases with practice
        self.strength = min(1.0, self.strength + (repetitions * 0.02))
        self.automaticity = min(1.0, self.automaticity + (repetitions * 0.01))
        
        return {
            'pattern': self.name,
            'repetitions': repetitions,
            'avg_execution_time': avg_time,
            'success_rate': success_rate,
            'new_strength': self.strength,
            'new_automaticity': self.automaticity
        }
    
    def _reinforce(self, execution_time: float) -> None:
        """
        Reinforce the pattern based on successful execution.
        Faster executions provide more reinforcement.
        """
        # Base reinforcement
        base_reinforcement = 0.01
        
        # Speed bonus (faster execution = more automatic)
        if execution_time < self.fastest_execution * 1.2:
            base_reinforcement += 0.02
        
        # Apply reinforcement
        self.strength = min(1.0, self.strength + base_reinforcement)
        self.reinforcement_count += 1
        
        # Update automaticity
        target_automaticity = min(1.0, self.execution_count / 100)
        self.automaticity = self.automaticity * 0.95 + target_automaticity * 0.05
    
    def decay(self) -> None:
        """Apply decay over time when not used"""
        if self.last_activated:
            days_since = (datetime.now() - self.last_activated).days
            if days_since > 0:
                decay_amount = self.decay_rate * days_since
                self.strength = max(0.1, self.strength - decay_amount)
                self.automaticity = max(0.1, self.automaticity - (decay_amount * 0.5))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'name': self.name,
            'type': self.type.value,
            'strength': self.strength,
            'automaticity': self.automaticity,
            'execution_count': self.execution_count,
            'success_rate': self.success_rate,
            'avg_execution_time': self.total_execution_time / max(self.execution_count, 1),
            'fastest_execution': self.fastest_execution,
            'slowest_execution': self.slowest_execution,
            'last_activated': self.last_activated.isoformat() if self.last_activated else None,
            'reinforcement_count': self.reinforcement_count,
            'common_contexts': dict(sorted(self.associated_contexts.items(), 
                                         key=lambda x: x[1], reverse=True)[:5])
        }
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type.value,
            'trigger': self.trigger.value,
            'context_conditions': self.context_conditions,
            'metadata': self.metadata,
            'execution_count': self.execution_count,
            'total_execution_time': self.total_execution_time,
            'last_execution_time': self.last_execution_time,
            'fastest_execution': self.fastest_execution,
            'slowest_execution': self.slowest_execution,
            'success_rate': self.success_rate,
            'failure_count': self.failure_count,
            'created_at': self.created_at.isoformat(),
            'last_activated': self.last_activated.isoformat() if self.last_activated else None,
            'last_modified': self.last_modified.isoformat(),
            'strength': self.strength,
            'reinforcement_count': self.reinforcement_count,
            'decay_rate': self.decay_rate,
            'automaticity': self.automaticity,
            'associated_contexts': dict(self.associated_contexts)
        }
    
    @classmethod
    def from_dict(cls, data: Dict, response: Callable) -> 'MuscleMemoryPattern':
        """Create from dictionary"""
        pattern = cls(
            name=data['name'],
            pattern_type=MuscleMemoryType(data['type']),
            trigger=ActivationTrigger(data['trigger']),
            response=response,
            context_conditions=data.get('context_conditions', []),
            metadata=data.get('metadata', {})
        )
        pattern.id = data['id']
        pattern.execution_count = data.get('execution_count', 0)
        pattern.total_execution_time = data.get('total_execution_time', 0)
        pattern.last_execution_time = data.get('last_execution_time')
        pattern.fastest_execution = data.get('fastest_execution', float('inf'))
        pattern.slowest_execution = data.get('slowest_execution', 0)
        pattern.success_rate = data.get('success_rate', 1.0)
        pattern.failure_count = data.get('failure_count', 0)
        pattern.created_at = datetime.fromisoformat(data['created_at'])
        pattern.last_activated = datetime.fromisoformat(data['last_activated']) if data.get('last_activated') else None
        pattern.last_modified = datetime.fromisoformat(data['last_modified'])
        pattern.strength = data.get('strength', 0.5)
        pattern.reinforcement_count = data.get('reinforcement_count', 0)
        pattern.decay_rate = data.get('decay_rate', 0.01)
        pattern.automaticity = data.get('automaticity', 0.3)
        pattern.associated_contexts = defaultdict(int, data.get('associated_contexts', {}))
        return pattern
    
    def __repr__(self) -> str:
        return f"MuscleMemoryPattern(name='{self.name}', strength={self.strength:.2f}, auto={self.automaticity:.2f})"

class MuscleMemorySystem:
    """
    Manages automatic, rapid-response patterns.
    Like the basal ganglia in the brain - handles routine actions
    without conscious effort.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./data/procedural/muscle_memory")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Core storage
        self.patterns: Dict[str, MuscleMemoryPattern] = {}  # id -> pattern
        self.patterns_by_name: Dict[str, str] = {}  # name -> id
        
        # Indexes
        self.patterns_by_type: Dict[str, List[str]] = defaultdict(list)
        self.patterns_by_trigger: Dict[str, List[str]] = defaultdict(list)
        
        # Activation queue for automatic responses
        self.activation_queue = deque(maxlen=50)
        
        # Statistics
        self.stats = {
            'total_patterns': 0,
            'total_activations': 0,
            'avg_response_time': 0.0,
            'automaticity_level': 0.0
        }
        
        # Automatic processing thread flag
        self.auto_process = True
        
        self._load_from_disk()
        logger.info(f"MuscleMemorySystem initialized with {len(self.patterns)} patterns")
    
    def register_pattern(self,
                        name: str,
                        pattern_type: MuscleMemoryType,
                        trigger: ActivationTrigger,
                        response: Callable,
                        context_conditions: Optional[List[str]] = None,
                        metadata: Optional[Dict] = None) -> str:
        """
        Register a new muscle memory pattern.
        """
        if name in self.patterns_by_name:
            logger.warning(f"Pattern '{name}' already exists, updating instead")
            return self.update_pattern(name, response, metadata)
        
        pattern = MuscleMemoryPattern(
            name=name,
            pattern_type=pattern_type,
            trigger=trigger,
            response=response,
            context_conditions=context_conditions,
            metadata=metadata
        )
        
        self.patterns[pattern.id] = pattern
        self.patterns_by_name[name] = pattern.id
        
        # Update indexes
        self.patterns_by_type[pattern_type.value].append(pattern.id)
        self.patterns_by_trigger[trigger.value].append(pattern.id)
        
        # Save
        self._save_pattern(pattern.id)
        
        self.stats['total_patterns'] = len(self.patterns)
        logger.info(f"Registered muscle memory pattern: {name}")
        
        return pattern.id
    
    def update_pattern(self, name: str,
                      response: Optional[Callable] = None,
                      metadata: Optional[Dict] = None) -> str:
        """Update an existing pattern"""
        if name not in self.patterns_by_name:
            raise ValueError(f"Pattern '{name}' not found")
        
        pattern_id = self.patterns_by_name[name]
        pattern = self.patterns[pattern_id]
        
        if response:
            pattern.response = response
        if metadata:
            pattern.metadata.update(metadata)
        
        pattern.last_modified = datetime.now()
        self._save_pattern(pattern_id)
        
        logger.info(f"Updated muscle memory pattern: {name}")
        return pattern_id
    
    def process_context(self, context: Dict[str, Any]) -> List[Dict]:
        """
        Automatically process context and activate relevant patterns.
        Called continuously by executive system.
        """
        activations = []
        
        for pattern in self.patterns.values():
            if pattern.should_activate(context):
                # Queue for activation
                self.activation_queue.append({
                    'pattern_id': pattern.id,
                    'context': context,
                    'timestamp': datetime.now()
                })
        
        # Process queue
        while self.activation_queue and self.auto_process:
            activation = self.activation_queue.popleft()
            pattern = self.patterns[activation['pattern_id']]
            
            # Execute automatically
            result = pattern.execute(activation['context'])
            
            activations.append({
                'pattern': pattern.name,
                'result': result
            })
            
            self.stats['total_activations'] += 1
        
        return activations
    
    def activate_pattern(self, name: str, context: Dict[str, Any]) -> Dict:
        """
        Explicitly activate a pattern by name.
        """
        if name not in self.patterns_by_name:
            return {'success': False, 'error': f"Pattern '{name}' not found"}
        
        pattern_id = self.patterns_by_name[name]
        pattern = self.patterns[pattern_id]
        
        return pattern.execute(context)
    
    def practice_pattern(self, name: str, context: Dict[str, Any], 
                        repetitions: int = 10) -> Dict:
        """
        Deliberately practice a pattern to strengthen it.
        """
        if name not in self.patterns_by_name:
            return {'success': False, 'error': f"Pattern '{name}' not found"}
        
        pattern_id = self.patterns_by_name[name]
        pattern = self.patterns[pattern_id]
        
        result = pattern.practice(context, repetitions)
        self._save_pattern(pattern_id)
        
        return result
    
    def get_pattern(self, name: str) -> Optional[MuscleMemoryPattern]:
        """Get pattern by name"""
        if name in self.patterns_by_name:
            return self.patterns[self.patterns_by_name[name]]
        return None
    
    def get_patterns_by_type(self, pattern_type: MuscleMemoryType) -> List[MuscleMemoryPattern]:
        """Get all patterns of a given type"""
        pattern_ids = self.patterns_by_type.get(pattern_type.value, [])
        return [self.patterns[pid] for pid in pattern_ids if pid in self.patterns]
    
    def apply_decay_all(self) -> None:
        """Apply decay to all patterns (called during idle periods)"""
        for pattern in self.patterns.values():
            pattern.decay()
            self._save_pattern(pattern.id)
        
        logger.debug("Applied decay to all muscle memory patterns")
    
    def get_most_automatic(self, limit: int = 5) -> List[MuscleMemoryPattern]:
        """Get patterns with highest automaticity"""
        sorted_patterns = sorted(self.patterns.values(), 
                               key=lambda p: p.automaticity, reverse=True)
        return sorted_patterns[:limit]
    
    def get_weakest_patterns(self, threshold: float = 0.3) -> List[MuscleMemoryPattern]:
        """Get patterns that need reinforcement"""
        return [p for p in self.patterns.values() if p.strength < threshold]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        if not self.patterns:
            return self.stats
        
        avg_response = np.mean([p.total_execution_time / max(p.execution_count, 1) 
                               for p in self.patterns.values()])
        avg_automaticity = np.mean([p.automaticity for p in self.patterns.values()])
        
        return {
            **self.stats,
            'avg_response_time': avg_response,
            'automaticity_level': avg_automaticity,
            'patterns_by_type': {t: len(ids) for t, ids in self.patterns_by_type.items()},
            'patterns_by_trigger': {t: len(ids) for t, ids in self.patterns_by_trigger.items()},
            'total_reinforcements': sum(p.reinforcement_count for p in self.patterns.values()),
            'queue_size': len(self.activation_queue)
        }
    
    def _save_pattern(self, pattern_id: str) -> None:
        """Save a pattern to disk"""
        pattern_file = self.storage_path / f"{pattern_id}.json"
        try:
            with open(pattern_file, 'w') as f:
                # Can't serialize response function
                pattern_dict = self.patterns[pattern_id].to_dict()
                json.dump(pattern_dict, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save pattern {pattern_id}: {e}")
    
    def _load_from_disk(self) -> None:
        """Load patterns from disk"""
        for file in self.storage_path.glob("*.json"):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    
                    # Create pattern without response (needs to be re-registered)
                    # In practice, responses would be registered separately
                    logger.warning(f"Pattern {data['name']} loaded without response function")
                    
            except Exception as e:
                logger.error(f"Failed to load pattern from {file}: {e}")
    
    def __repr__(self) -> str:
        return f"MuscleMemorySystem(patterns={len(self.patterns)}, activations={self.stats['total_activations']})"