"""
Learned Behaviors - Complex behavioral patterns acquired through experience.
Combines multiple skills and muscle memories into coherent behaviors.
Like social etiquette, conversation styles, and problem-solving strategies.
"""
from typing import Dict, Any, Optional, List, Set, Tuple, Callable
from datetime import datetime, timedelta
import uuid
import json
from pathlib import Path
import logging
from enum import Enum
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)

class BehaviorComplexity(Enum):
    """How complex/composite a behavior is"""
    SIMPLE = "simple"  # Single skill/pattern
    COMPOUND = "compound"  # Multiple coordinated elements
    COMPLEX = "complex"  # Multi-step with decision points
    STRATEGIC = "strategic"  # Goal-directed, adaptive

class BehaviorContext(Enum):
    """Contexts where behaviors are relevant"""
    SOCIAL = "social"  # Interactions with people
    PROBLEM_SOLVING = "problem_solving"  # Cognitive challenges
    EMOTIONAL = "emotional"  # Emotional situations
    LEARNING = "learning"  # Learning situations
    CREATIVE = "creative"  # Creative tasks
    ROUTINE = "routine"  # Daily routines

class LearnedBehavior:
    """
    A complex behavioral pattern learned through experience.
    Combines skills, muscle memories, and decision rules.
    """
    
    def __init__(self,
                 name: str,
                 context: BehaviorContext,
                 complexity: BehaviorComplexity,
                 components: List[Dict],  # Skills/patterns that make up this behavior
                 trigger_conditions: Dict[str, Any],
                 success_criteria: List[str],
                 metadata: Optional[Dict] = None):
        
        self.id = str(uuid.uuid4())
        self.name = name
        self.context = context
        self.complexity = complexity
        self.components = components  # e.g., [{'type': 'skill', 'name': 'greet'}, ...]
        self.trigger_conditions = trigger_conditions
        self.success_criteria = success_criteria
        self.metadata = metadata or {}
        
        # Learning metrics
        self.learning_progress = 0.0  # 0-1 how well learned
        self.adaptation_level = 0.0  # 0-1 how adaptable to variations
        self.robustness = 0.0  # 0-1 how well it handles edge cases
        
        # Performance tracking
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.partial_success_count = 0
        
        # Execution history
        self.execution_history = deque(maxlen=50)
        self.failure_patterns = defaultdict(int)  # Common failure reasons
        
        # Context adaptation
        self.context_variations = defaultdict(lambda: defaultdict(int))  # How behavior varies by context
        self.successful_contexts = []  # Contexts where it worked well
        
        # Temporal
        self.created_at = datetime.now()
        self.last_executed = None
        self.last_modified = self.created_at
        self.last_success = None
        
        # Reinforcement
        self.reinforcement_count = 0
        self.confidence = 0.5  # Confidence in this behavior
        
        logger.debug(f"Created learned behavior: {name}")
    
    def should_activate(self, context: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Determine if this behavior should activate in current context.
        Returns (should_activate, confidence).
        """
        score = 0.0
        reasons = []
        
        # Check trigger conditions
        for condition, value in self.trigger_conditions.items():
            if condition in context:
                if context[condition] == value:
                    score += 0.3
                    reasons.append(f"{condition}={value}")
                elif isinstance(value, list) and context[condition] in value:
                    score += 0.25
                    reasons.append(f"{condition} in {value}")
        
        # Check context match
        if context.get('type') == self.context.value:
            score += 0.2
        
        # Check recent success
        if self.last_success:
            days_since = (datetime.now() - self.last_success).days
            if days_since < 7:
                score += 0.1
        
        # Check confidence
        score *= self.confidence
        
        # Check learning progress
        score *= (0.5 + self.learning_progress * 0.5)
        
        threshold = 0.5
        return score >= threshold, score
    
    def execute(self, context: Dict[str, Any],
               skill_executor: Callable,
               muscle_memory_executor: Callable) -> Dict[str, Any]:
        """
        Execute the learned behavior.
        
        Args:
            context: Current situation
            skill_executor: Function to execute skills
            muscle_memory_executor: Function to execute muscle memory
        
        Returns:
            Execution results
        """
        start_time = datetime.now()
        self.execution_count += 1
        self.last_executed = start_time
        
        execution_log = {
            'timestamp': start_time.isoformat(),
            'context': context.get('type', 'unknown'),
            'steps': [],
            'success': False,
            'partial_success': False,
            'failure_reason': None
        }
        
        try:
            results = []
            success_count = 0
            
            # Execute each component in sequence
            for component in self.components:
                comp_result = None
                
                if component['type'] == 'skill':
                    comp_result = skill_executor(component['name'], context)
                elif component['type'] == 'muscle_memory':
                    comp_result = muscle_memory_executor(component['name'], context)
                elif component['type'] == 'sub_behavior':
                    # Recursive execution of sub-behavior
                    # This would need behavior_executor reference
                    pass
                
                if comp_result:
                    results.append(comp_result)
                    if comp_result.get('success', False):
                        success_count += 1
                    
                    execution_log['steps'].append({
                        'component': component['name'],
                        'type': component['type'],
                        'success': comp_result.get('success', False),
                        'result': comp_result.get('output', {})
                    })
            
            # Evaluate overall success
            success_ratio = success_count / len(self.components) if self.components else 0
            execution_log['partial_success'] = success_ratio >= 0.7
            
            # Check against success criteria
            criteria_met = self._check_success_criteria(results, context)
            execution_log['success'] = criteria_met
            
            # Update statistics
            if criteria_met:
                self.success_count += 1
                self.last_success = datetime.now()
                self.successful_contexts.append(context.get('type', 'unknown'))
                self._reinforce()
            elif execution_log['partial_success']:
                self.partial_success_count += 1
                self._adapt(context, results)
            else:
                self.failure_count += 1
                failure_reason = self._analyze_failure(results, context)
                self.failure_patterns[failure_reason] += 1
                execution_log['failure_reason'] = failure_reason
            
            # Update learning progress
            self._update_learning_progress()
            
            execution_log['execution_time_ms'] = (datetime.now() - start_time).total_seconds() * 1000
            self.execution_history.append(execution_log)
            
            return {
                'success': criteria_met,
                'partial_success': execution_log['partial_success'],
                'behavior': self.name,
                'steps': len(self.components),
                'successful_steps': success_count,
                'execution_log': execution_log,
                'confidence': self.confidence
            }
            
        except Exception as e:
            logger.error(f"Error executing behavior '{self.name}': {e}")
            self.failure_count += 1
            return {
                'success': False,
                'error': str(e),
                'behavior': self.name
            }
    
    def learn_from_observation(self, demonstration: Dict[str, Any]) -> None:
        """
        Learn/improve behavior by observing a demonstration.
        """
        # Extract successful patterns
        if demonstration.get('success', False):
            # Identify key elements that led to success
            context = demonstration.get('context', {})
            steps = demonstration.get('steps', [])
            
            # Update context associations
            context_type = context.get('type', 'unknown')
            self.context_variations[context_type]['successes'] += 1
            
            # Adjust component weights based on what worked
            for step in steps:
                if step.get('success', False):
                    # Find matching component
                    for comp in self.components:
                        if comp['name'] == step.get('component'):
                            comp['weight'] = comp.get('weight', 1.0) * 1.1
                            break
            
            self.learning_progress = min(1.0, self.learning_progress + 0.1)
            self._save_state()
    
    def get_effectiveness(self) -> Dict[str, Any]:
        """Get effectiveness metrics for this behavior"""
        total = self.execution_count
        if total == 0:
            return {'effectiveness': 0, 'reliability': 0}
        
        success_rate = self.success_count / total
        partial_rate = self.partial_success_count / total
        
        # Calculate reliability (consistency)
        if len(self.execution_history) > 5:
            recent_successes = sum(1 for e in list(self.execution_history)[-5:] if e.get('success'))
            reliability = recent_successes / 5
        else:
            reliability = success_rate
        
        return {
            'effectiveness': success_rate * 0.7 + partial_rate * 0.3,
            'success_rate': success_rate,
            'partial_success_rate': partial_rate,
            'reliability': reliability,
            'confidence': self.confidence,
            'learning_progress': self.learning_progress,
            'adaptation_level': self.adaptation_level,
            'robustness': self.robustness,
            'total_executions': total,
            'common_failures': dict(sorted(self.failure_patterns.items(), 
                                         key=lambda x: x[1], reverse=True)[:3])
        }
    
    def _check_success_criteria(self, results: List[Dict], context: Dict) -> bool:
        """Check if execution met success criteria"""
        criteria_met = 0
        for criterion in self.success_criteria:
            if criterion in context:
                criteria_met += 1
            # Also check in results
            for result in results:
                if isinstance(result, dict) and result.get(criterion, False):
                    criteria_met += 1
                    break
        
        return criteria_met >= len(self.success_criteria) * 0.7
    
    def _analyze_failure(self, results: List[Dict], context: Dict) -> str:
        """Analyze why the behavior failed"""
        # Check if any component completely failed
        for i, result in enumerate(results):
            if not result.get('success', True):
                component = self.components[i]['name']
                return f"Component failed: {component}"
        
        # Check context mismatch
        if context.get('type') != self.context.value:
            return "Context mismatch"
        
        # Check missing prerequisites
        missing = []
        for req in self.metadata.get('prerequisites', []):
            if req not in context:
                missing.append(req)
        if missing:
            return f"Missing prerequisites: {missing}"
        
        return "Unknown failure"
    
    def _reinforce(self) -> None:
        """Reinforce successful behavior"""
        self.reinforcement_count += 1
        self.confidence = min(1.0, self.confidence + 0.05)
        self.robustness = min(1.0, self.robustness + 0.02)
    
    def _adapt(self, context: Dict, results: List) -> None:
        """Adapt based on partial success"""
        self.adaptation_level = min(1.0, self.adaptation_level + 0.03)
        
        # Learn which contexts need adaptation
        context_type = context.get('type', 'unknown')
        self.context_variations[context_type]['adaptations'] += 1
    
    def _update_learning_progress(self) -> None:
        """Update overall learning progress"""
        if self.execution_count < 10:
            # Early learning
            self.learning_progress = self.success_count / 10
        else:
            # Later learning based on trend
            recent = list(self.execution_history)[-10:]
            recent_success = sum(1 for e in recent if e.get('success')) / 10
            self.learning_progress = self.learning_progress * 0.7 + recent_success * 0.3
    
    def _save_state(self) -> None:
        """Save current state (would persist to disk)"""
        pass
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'context': self.context.value,
            'complexity': self.complexity.value,
            'components': self.components,
            'trigger_conditions': self.trigger_conditions,
            'success_criteria': self.success_criteria,
            'metadata': self.metadata,
            'learning_progress': self.learning_progress,
            'adaptation_level': self.adaptation_level,
            'robustness': self.robustness,
            'execution_count': self.execution_count,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'partial_success_count': self.partial_success_count,
            'created_at': self.created_at.isoformat(),
            'last_executed': self.last_executed.isoformat() if self.last_executed else None,
            'last_modified': self.last_modified.isoformat(),
            'last_success': self.last_success.isoformat() if self.last_success else None,
            'reinforcement_count': self.reinforcement_count,
            'confidence': self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LearnedBehavior':
        """Create from dictionary"""
        behavior = cls(
            name=data['name'],
            context=BehaviorContext(data['context']),
            complexity=BehaviorComplexity(data['complexity']),
            components=data['components'],
            trigger_conditions=data['trigger_conditions'],
            success_criteria=data['success_criteria'],
            metadata=data.get('metadata', {})
        )
        behavior.id = data['id']
        behavior.learning_progress = data.get('learning_progress', 0.0)
        behavior.adaptation_level = data.get('adaptation_level', 0.0)
        behavior.robustness = data.get('robustness', 0.0)
        behavior.execution_count = data.get('execution_count', 0)
        behavior.success_count = data.get('success_count', 0)
        behavior.failure_count = data.get('failure_count', 0)
        behavior.partial_success_count = data.get('partial_success_count', 0)
        behavior.created_at = datetime.fromisoformat(data['created_at'])
        behavior.last_executed = datetime.fromisoformat(data['last_executed']) if data.get('last_executed') else None
        behavior.last_modified = datetime.fromisoformat(data['last_modified'])
        behavior.last_success = datetime.fromisoformat(data['last_success']) if data.get('last_success') else None
        behavior.reinforcement_count = data.get('reinforcement_count', 0)
        behavior.confidence = data.get('confidence', 0.5)
        return behavior
    
    def __repr__(self) -> str:
        return (f"LearnedBehavior(name='{self.name}', context={self.context.value}, "
                f"progress={self.learning_progress:.2f})")

class BehaviorLibrary:
    """
    Manages learned behaviors - complex patterns acquired through experience.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./data/procedural/behaviors")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Core storage
        self.behaviors: Dict[str, LearnedBehavior] = {}  # id -> behavior
        self.behaviors_by_name: Dict[str, str] = {}  # name -> id
        
        # Indexes
        self.behaviors_by_context: Dict[str, List[str]] = defaultdict(list)
        self.behaviors_by_complexity: Dict[str, List[str]] = defaultdict(list)
        
        # Execution tracking
        self.execution_queue = deque(maxlen=100)
        
        # Statistics
        self.stats = {
            'total_behaviors': 0,
            'total_executions': 0,
            'overall_success_rate': 0.0
        }
        
        self._load_from_disk()
        logger.info(f"BehaviorLibrary initialized with {len(self.behaviors)} behaviors")
    
    def register_behavior(self, behavior: LearnedBehavior) -> str:
        """Register a new learned behavior"""
        if behavior.name in self.behaviors_by_name:
            logger.warning(f"Behavior '{behavior.name}' already exists")
            return self.behaviors_by_name[behavior.name]
        
        self.behaviors[behavior.id] = behavior
        self.behaviors_by_name[behavior.name] = behavior.id
        
        # Update indexes
        self.behaviors_by_context[behavior.context.value].append(behavior.id)
        self.behaviors_by_complexity[behavior.complexity.value].append(behavior.id)
        
        # Save
        self._save_behavior(behavior.id)
        
        self.stats['total_behaviors'] = len(self.behaviors)
        logger.info(f"Registered learned behavior: {behavior.name}")
        
        return behavior.id
    
    def find_applicable_behaviors(self, context: Dict[str, Any]) -> List[Tuple[LearnedBehavior, float]]:
        """Find behaviors that apply to current context"""
        applicable = []
        
        for behavior in self.behaviors.values():
            should_activate, confidence = behavior.should_activate(context)
            if should_activate:
                applicable.append((behavior, confidence))
        
        # Sort by confidence
        applicable.sort(key=lambda x: x[1], reverse=True)
        return applicable
    
    def execute_best_behavior(self, context: Dict[str, Any],
                             skill_executor: Callable,
                             muscle_memory_executor: Callable) -> Dict[str, Any]:
        """Find and execute the best behavior for current context"""
        applicable = self.find_applicable_behaviors(context)
        
        if not applicable:
            return {
                'success': False,
                'error': 'No applicable behavior found',
                'context': context.get('type', 'unknown')
            }
        
        # Execute the best matching behavior
        best_behavior, confidence = applicable[0]
        
        # Queue for execution tracking
        self.execution_queue.append({
            'behavior': best_behavior.name,
            'context': context.get('type', 'unknown'),
            'timestamp': datetime.now()
        })
        
        result = best_behavior.execute(context, skill_executor, muscle_memory_executor)
        
        self.stats['total_executions'] += 1
        if result.get('success', False):
            # Update overall success rate
            recent = list(self.execution_queue)[-20:]
            success_count = sum(1 for e in recent if e.get('success', False))
            self.stats['overall_success_rate'] = success_count / len(recent) if recent else 0
        
        return result
    
    def learn_from_demonstration(self, demonstration: Dict[str, Any]) -> Optional[str]:
        """
        Create or update a behavior based on demonstration.
        """
        # Extract behavior signature
        context = demonstration.get('context', {})
        steps = demonstration.get('steps', [])
        
        # Generate behavior name
        base_name = f"learned_{context.get('type', 'behavior')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create behavior components from demonstration
        components = []
        for step in steps:
            components.append({
                'type': step.get('type', 'skill'),
                'name': step.get('name'),
                'weight': 1.0
            })
        
        # Create trigger conditions from context
        trigger_conditions = {}
        for key, value in context.items():
            if key in ['type', 'intent', 'goal']:
                trigger_conditions[key] = value
        
        # Create behavior
        behavior = LearnedBehavior(
            name=base_name,
            context=BehaviorContext(context.get('type', 'routine')),
            complexity=BehaviorComplexity.COMPOUND if len(components) > 3 else BehaviorComplexity.SIMPLE,
            components=components,
            trigger_conditions=trigger_conditions,
            success_criteria=demonstration.get('success_criteria', ['completed']),
            metadata={'learned_from': 'demonstration', 'timestamp': datetime.now().isoformat()}
        )
        
        # Learn from the demonstration
        behavior.learn_from_observation(demonstration)
        
        # Register
        behavior_id = self.register_behavior(behavior)
        
        logger.info(f"Learned new behavior from demonstration: {base_name}")
        return behavior_id
    
    def get_behaviors_by_context(self, context: BehaviorContext) -> List[LearnedBehavior]:
        """Get all behaviors for a specific context"""
        behavior_ids = self.behaviors_by_context.get(context.value, [])
        return [self.behaviors[bid] for bid in behavior_ids if bid in self.behaviors]
    
    def get_most_effective(self, limit: int = 5) -> List[LearnedBehavior]:
        """Get most effective behaviors"""
        sorted_behaviors = sorted(
            self.behaviors.values(),
            key=lambda b: b.get_effectiveness()['effectiveness'],
            reverse=True
        )
        return sorted_behaviors[:limit]
    
    def get_needs_practice(self) -> List[LearnedBehavior]:
        """Get behaviors that need practice"""
        return [b for b in self.behaviors.values() 
                if b.get_effectiveness()['effectiveness'] < 0.6 
                and b.execution_count > 0]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get library statistics"""
        if not self.behaviors:
            return self.stats
        
        avg_effectiveness = np.mean([b.get_effectiveness()['effectiveness'] 
                                    for b in self.behaviors.values()])
        
        return {
            **self.stats,
            'average_effectiveness': avg_effectiveness,
            'behaviors_by_context': {c: len(ids) for c, ids in self.behaviors_by_context.items()},
            'behaviors_by_complexity': {c: len(ids) for c, ids in self.behaviors_by_complexity.items()},
            'total_learned': len([b for b in self.behaviors.values() 
                                 if b.metadata.get('learned_from') == 'demonstration']),
            'queue_size': len(self.execution_queue)
        }
    
    def _save_behavior(self, behavior_id: str) -> None:
        """Save a behavior to disk"""
        behavior_file = self.storage_path / f"{behavior_id}.json"
        try:
            with open(behavior_file, 'w') as f:
                json.dump(self.behaviors[behavior_id].to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save behavior {behavior_id}: {e}")
    
    def _load_from_disk(self) -> None:
        """Load behaviors from disk"""
        for file in self.storage_path.glob("*.json"):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    behavior = LearnedBehavior.from_dict(data)
                    self.behaviors[behavior.id] = behavior
                    self.behaviors_by_name[behavior.name] = behavior.id
                    self.behaviors_by_context[behavior.context.value].append(behavior.id)
                    self.behaviors_by_complexity[behavior.complexity.value].append(behavior.id)
            except Exception as e:
                logger.error(f"Failed to load behavior from {file}: {e}")
    
    def __repr__(self) -> str:
        return f"BehaviorLibrary(behaviors={len(self.behaviors)}, executions={self.stats['total_executions']})"