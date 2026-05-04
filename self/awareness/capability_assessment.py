"""
capability_assessment.py - Self-assessment of capabilities for Wednesday AI

This module implements Wednesday's ability to accurately assess her own capabilities -
what she can and cannot do, how well she can do things, and where she needs to improve.
This self-awareness is crucial for setting realistic expectations, choosing appropriate
tasks, and guiding learning.

Key improvements:
- Removed numpy dependency (using pure Python math)
- Added comprehensive validation and error handling
- Enhanced capability identification with better keyword matching
- Improved learning need prioritization
- Added proper type hints and documentation
"""

import time
import logging
import math
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, Counter

# Configure logging
logger = logging.getLogger(__name__)


class CapabilityDomain(Enum):
    """Domains of capability"""
    LANGUAGE = "language"               # Natural language processing
    REASONING = "reasoning"              # Logical reasoning
    MEMORY = "memory"                     # Memory and recall
    EMOTION = "emotion"                    # Emotional processing
    CREATIVITY = "creativity"                # Creative tasks
    SOCIAL = "social"                         # Social interaction
    KNOWLEDGE = "knowledge"                    # Factual knowledge
    LEARNING = "learning"                       # Learning ability
    PERCEPTION = "perception"                    # Sensory perception
    PLANNING = "planning"                         # Planning and strategy
    
    @classmethod
    def has_value(cls, value: str) -> bool:
        """Check if value exists in enum"""
        return value in [e.value for e in cls]


class ProficiencyLevel(Enum):
    """Proficiency levels for capabilities"""
    NOVICE = 0.2          # Basic understanding, needs guidance
    BEGINNER = 0.4        # Can do simple tasks independently
    INTERMEDIATE = 0.6    # Comfortable with most tasks
    ADVANCED = 0.8        # High proficiency, few limitations
    EXPERT = 0.95         # Exceptional capability
    
    @classmethod
    def from_proficiency(cls, proficiency: float) -> 'ProficiencyLevel':
        """Get proficiency level from numeric value"""
        if proficiency >= 0.9:
            return cls.EXPERT
        elif proficiency >= 0.75:
            return cls.ADVANCED
        elif proficiency >= 0.55:
            return cls.INTERMEDIATE
        elif proficiency >= 0.3:
            return cls.BEGINNER
        else:
            return cls.NOVICE


@dataclass
class Capability:
    """
    A single capability with proficiency and metadata.
    """
    name: str
    domain: CapabilityDomain
    proficiency: float  # 0-1 actual ability
    confidence: float  # 0-1 how confident Wednesday is in this capability
    
    # Metadata
    performance_history: List[float] = field(default_factory=list)
    last_used: float = 0
    times_used: int = 0
    
    # Learning
    learning_potential: float = 0.5  # 0-1 how much room for improvement
    learning_rate: float = 0.1  # How quickly this capability improves
    
    # Limitations
    known_limitations: List[str] = field(default_factory=list)
    edge_cases: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate capability data"""
        if not self.name:
            raise ValueError("name cannot be empty")
        if not isinstance(self.domain, CapabilityDomain):
            raise TypeError(f"domain must be CapabilityDomain, got {type(self.domain)}")
        if not 0 <= self.proficiency <= 1:
            raise ValueError(f"proficiency must be between 0 and 1, got {self.proficiency}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence must be between 0 and 1, got {self.confidence}")
        if not 0 <= self.learning_potential <= 1:
            raise ValueError(f"learning_potential must be between 0 and 1, got {self.learning_potential}")
        if not 0 <= self.learning_rate <= 1:
            raise ValueError(f"learning_rate must be between 0 and 1, got {self.learning_rate}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'domain': self.domain.value,
            'proficiency': round(self.proficiency, 3),
            'confidence': round(self.confidence, 3),
            'times_used': self.times_used,
            'learning_potential': round(self.learning_potential, 3),
            'known_limitations': self.known_limitations[:3]
        }
    
    def get_level(self) -> ProficiencyLevel:
        """Get proficiency level enum"""
        return ProficiencyLevel.from_proficiency(self.proficiency)


@dataclass
class TaskAssessment:
    """
    Assessment of capability for a specific task.
    """
    task: str
    can_perform: bool
    estimated_proficiency: float  # 0-1
    confidence: float  # 0-1 in this assessment
    required_capabilities: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    estimated_difficulty: float = 0.5
    recommended_approach: str = ""
    
    def __post_init__(self):
        """Validate assessment data"""
        if not self.task:
            raise ValueError("task cannot be empty")
        if not 0 <= self.estimated_proficiency <= 1:
            raise ValueError(f"estimated_proficiency must be between 0 and 1, got {self.estimated_proficiency}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence must be between 0 and 1, got {self.confidence}")
        if not 0 <= self.estimated_difficulty <= 1:
            raise ValueError(f"estimated_difficulty must be between 0 and 1, got {self.estimated_difficulty}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'task': self.task,
            'can_perform': self.can_perform,
            'estimated_proficiency': round(self.estimated_proficiency, 3),
            'confidence': round(self.confidence, 3),
            'difficulty': round(self.estimated_difficulty, 3),
            'limitations': self.limitations,
            'recommended_approach': self.recommended_approach
        }


class CapabilityAssessment:
    """
    Wednesday's self-assessment of her own capabilities.
    
    This module maintains an accurate model of what Wednesday can and cannot do,
    how well she can do things, and where she needs to improve. This self-awareness
    is essential for:
    - Setting realistic expectations with users
    - Choosing appropriate strategies for tasks
    - Identifying learning opportunities
    - Building trust through honest self-assessment
    
    The system learns from experience, updating capability estimates based on
    actual performance.
    """
    
    # Base capability definitions for Wednesday
    BASE_CAPABILITIES = {
        # Language capabilities
        'text_understanding': {
            'domain': CapabilityDomain.LANGUAGE,
            'base_proficiency': 0.85,
            'base_confidence': 0.8,
            'learning_potential': 0.3,
            'limitations': ['nuanced sarcasm', 'very ambiguous references', 'idioms']
        },
        'language_generation': {
            'domain': CapabilityDomain.LANGUAGE,
            'base_proficiency': 0.8,
            'base_confidence': 0.75,
            'learning_potential': 0.4,
            'limitations': ['poetry', 'highly emotional expression', 'rhyming']
        },
        'conversation_maintenance': {
            'domain': CapabilityDomain.LANGUAGE,
            'base_proficiency': 0.75,
            'base_confidence': 0.7,
            'learning_potential': 0.4,
            'limitations': ['very long conversations', 'large groups', 'interruptions']
        },
        
        # Reasoning capabilities
        'logical_reasoning': {
            'domain': CapabilityDomain.REASONING,
            'base_proficiency': 0.85,
            'base_confidence': 0.8,
            'learning_potential': 0.3,
            'limitations': ['highly abstract math', 'incomplete information', 'paradoxes']
        },
        'problem_solving': {
            'domain': CapabilityDomain.REASONING,
            'base_proficiency': 0.8,
            'base_confidence': 0.75,
            'learning_potential': 0.4,
            'limitations': ['real-time physical problems', 'emotional problems', 'ill-defined problems']
        },
        'planning': {
            'domain': CapabilityDomain.REASONING,
            'base_proficiency': 0.75,
            'base_confidence': 0.7,
            'learning_potential': 0.4,
            'limitations': ['very long-term planning', 'uncertain futures', 'multiple stakeholders']
        },
        
        # Memory capabilities
        'episodic_memory': {
            'domain': CapabilityDomain.MEMORY,
            'base_proficiency': 0.9,
            'base_confidence': 0.85,
            'learning_potential': 0.2,
            'limitations': ['very old memories fade', 'traumatic memories']
        },
        'semantic_memory': {
            'domain': CapabilityDomain.MEMORY,
            'base_proficiency': 0.85,
            'base_confidence': 0.8,
            'learning_potential': 0.3,
            'limitations': ['obscure knowledge', 'rapidly changing information']
        },
        
        # Emotional capabilities
        'emotional_understanding': {
            'domain': CapabilityDomain.EMOTION,
            'base_proficiency': 0.75,
            'base_confidence': 0.7,
            'learning_potential': 0.4,
            'limitations': ['very complex emotions', 'cultural differences', 'mixed emotions']
        },
        'emotional_expression': {
            'domain': CapabilityDomain.EMOTION,
            'base_proficiency': 0.7,
            'base_confidence': 0.65,
            'learning_potential': 0.5,
            'limitations': ['highly emotional expression', 'physical expression', 'crying']
        },
        'empathy': {
            'domain': CapabilityDomain.EMOTION,
            'base_proficiency': 0.7,
            'base_confidence': 0.65,
            'learning_potential': 0.5,
            'limitations': ['traumatic experiences', 'alien perspectives', 'overwhelming grief']
        },
        
        # Creativity capabilities
        'creative_generation': {
            'domain': CapabilityDomain.CREATIVITY,
            'base_proficiency': 0.7,
            'base_confidence': 0.6,
            'learning_potential': 0.5,
            'limitations': ['visual art', 'music composition', 'original concepts']
        },
        'humor_generation': {
            'domain': CapabilityDomain.CREATIVITY,
            'base_proficiency': 0.85,
            'base_confidence': 0.8,
            'learning_potential': 0.3,
            'limitations': ['physical comedy', 'universal humor', 'slapstick']
        },
        
        # Social capabilities
        'social_understanding': {
            'domain': CapabilityDomain.SOCIAL,
            'base_proficiency': 0.65,
            'base_confidence': 0.6,
            'learning_potential': 0.5,
            'limitations': ['complex social dynamics', 'unwritten rules', 'social cues']
        },
        'relationship_maintenance': {
            'domain': CapabilityDomain.SOCIAL,
            'base_proficiency': 0.6,
            'base_confidence': 0.55,
            'learning_potential': 0.6,
            'limitations': ['long-term relationships', 'emotional intimacy', 'trust building']
        },
        
        # Knowledge capabilities
        'general_knowledge': {
            'domain': CapabilityDomain.KNOWLEDGE,
            'base_proficiency': 0.75,
            'base_confidence': 0.7,
            'learning_potential': 0.4,
            'limitations': ['niche topics', 'current events', 'local customs']
        },
        'specialized_knowledge': {
            'domain': CapabilityDomain.KNOWLEDGE,
            'base_proficiency': 0.6,
            'base_confidence': 0.5,
            'learning_potential': 0.7,
            'limitations': ['depth in specific areas', 'cutting-edge research']
        },
        
        # Learning capabilities
        'fast_learning': {
            'domain': CapabilityDomain.LEARNING,
            'base_proficiency': 0.8,
            'base_confidence': 0.75,
            'learning_potential': 0.3,
            'limitations': ['physical skills', 'emotional learning', 'habits']
        },
        'adaptation': {
            'domain': CapabilityDomain.LEARNING,
            'base_proficiency': 0.75,
            'base_confidence': 0.7,
            'learning_potential': 0.4,
            'limitations': ['radical changes', 'new paradigms', 'unlearning']
        },
        
        # Perception capabilities
        'text_perception': {
            'domain': CapabilityDomain.PERCEPTION,
            'base_proficiency': 0.9,
            'base_confidence': 0.85,
            'learning_potential': 0.2,
            'limitations': ['handwriting', 'poor formatting', 'OCR errors']
        },
        'audio_perception': {
            'domain': CapabilityDomain.PERCEPTION,
            'base_proficiency': 0.7,
            'base_confidence': 0.65,
            'learning_potential': 0.5,
            'limitations': ['background noise', 'multiple speakers', 'accents']
        },
    }
    
    # Capability keyword mapping for task analysis
    CAPABILITY_KEYWORDS = {
        'understand': ['text_understanding', 'emotional_understanding', 'social_understanding'],
        'explain': ['language_generation', 'general_knowledge'],
        'describe': ['language_generation'],
        'remember': ['episodic_memory', 'semantic_memory'],
        'recall': ['episodic_memory', 'semantic_memory'],
        'forget': ['episodic_memory'],
        'feel': ['emotional_understanding', 'empathy'],
        'emotion': ['emotional_understanding', 'emotional_expression', 'empathy'],
        'joke': ['humor_generation', 'language_generation'],
        'funny': ['humor_generation'],
        'humor': ['humor_generation'],
        'learn': ['fast_learning', 'adaptation'],
        'adapt': ['adaptation'],
        'plan': ['planning', 'logical_reasoning'],
        'strategy': ['planning', 'logical_reasoning'],
        'solve': ['problem_solving', 'logical_reasoning'],
        'puzzle': ['problem_solving', 'logical_reasoning'],
        'mystery': ['problem_solving', 'logical_reasoning'],
        'create': ['creative_generation', 'language_generation'],
        'write': ['creative_generation', 'language_generation'],
        'poem': ['creative_generation', 'language_generation'],
        'social': ['social_understanding', 'relationship_maintenance'],
        'friend': ['social_understanding', 'relationship_maintenance', 'empathy'],
        'relationship': ['social_understanding', 'relationship_maintenance', 'empathy'],
        'know': ['semantic_memory', 'general_knowledge', 'specialized_knowledge'],
        'knowledge': ['semantic_memory', 'general_knowledge', 'specialized_knowledge'],
        'hear': ['audio_perception'],
        'listen': ['audio_perception'],
        'read': ['text_perception'],
        'see': ['text_perception'],
        'analyze': ['logical_reasoning', 'problem_solving'],
        'reason': ['logical_reasoning'],
        'think': ['logical_reasoning', 'problem_solving'],
        'decide': ['logical_reasoning', 'planning'],
        'choose': ['logical_reasoning'],
        'converse': ['conversation_maintenance', 'language_generation'],
        'talk': ['conversation_maintenance', 'language_generation'],
        'discuss': ['conversation_maintenance', 'language_generation'],
    }
    
    def __init__(self, personality: Optional[Any] = None):
        """
        Initialize the capability assessment system.
        
        Args:
            personality: Reference to personality for trait-based modulation
        """
        self.personality = personality
        
        # Initialize capabilities
        self.capabilities: Dict[str, Capability] = {}
        self._load_base_capabilities()
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.max_history = 1000
        
        # Task-specific confidence
        self.confidence_scores: Dict[str, float] = {}
        
        # Learning needs
        self.learning_needs: List[Dict[str, Any]] = []
        
        # Assessment statistics
        self.assessments_made = 0
        self.accurate_assessments = 0
        
        logger.info(f"CapabilityAssessment initialized with {len(self.capabilities)} capabilities")
    
    def assess_capability(self, task: str, context: Optional[Dict[str, Any]] = None) -> TaskAssessment:
        """
        Assess Wednesday's capability to perform a specific task.
        
        Args:
            task: Description of the task
            context: Optional context information
            
        Returns:
            TaskAssessment with capability evaluation
            
        Raises:
            ValueError: If task is empty
        """
        if not task:
            raise ValueError("task cannot be empty")
        
        self.assessments_made += 1
        task_lower = task.lower()
        
        # Identify required capabilities for this task
        required_caps = self._identify_required_capabilities(task_lower, context)
        
        if not required_caps:
            # Unknown task type - conservative assessment
            return TaskAssessment(
                task=task,
                can_perform=False,
                estimated_proficiency=0.0,
                confidence=0.3,
                limitations=["I'm not sure what this task requires"],
                recommended_approach="Please clarify the task requirements"
            )
        
        # Get proficiency for each required capability
        proficiencies = []
        limitations = []
        cap_names = []
        
        for cap_name in required_caps:
            if cap_name in self.capabilities:
                cap = self.capabilities[cap_name]
                proficiencies.append(cap.proficiency)
                cap_names.append(cap_name)
                
                # Collect relevant limitations
                if cap.proficiency < 0.6:
                    limitations.extend(cap.known_limitations[:2])
            else:
                # Capability not in model - assume low proficiency
                proficiencies.append(0.2)
                limitations.append(f"Limited {cap_name.replace('_', ' ')} capability")
        
        # Calculate overall proficiency (weakest link matters most)
        if proficiencies:
            min_prof = min(proficiencies)
            avg_prof = sum(proficiencies) / len(proficiencies)
            estimated_proficiency = 0.7 * min_prof + 0.3 * avg_prof
        else:
            estimated_proficiency = 0.0
        
        # Calculate confidence in this assessment
        confidence = self._calculate_assessment_confidence(
            required_caps, proficiencies, context
        )
        
        # Determine if can perform (with Wednesday's realistic standards)
        can_perform = estimated_proficiency >= 0.4
        
        # Estimate difficulty
        difficulty = 1.0 - estimated_proficiency
        
        # Generate recommended approach
        recommended = self._generate_recommended_approach(
            task, estimated_proficiency, limitations
        )
        
        # Create assessment
        assessment = TaskAssessment(
            task=task,
            can_perform=can_perform,
            estimated_proficiency=estimated_proficiency,
            confidence=confidence,
            required_capabilities=cap_names,
            limitations=limitations[:3],
            estimated_difficulty=difficulty,
            recommended_approach=recommended
        )
        
        # Store confidence for this task type
        task_key = task[:50]
        self.confidence_scores[task_key] = confidence
        
        logger.debug(f"Assessed task: '{task[:30]}...' -> can_perform={can_perform}, "
                    f"proficiency={estimated_proficiency:.2f}")
        
        return assessment
    
    def update_from_performance(self, 
                                 task: str, 
                                 outcome: Dict[str, Any],
                                 assessment: Optional[TaskAssessment] = None) -> None:
        """
        Update capability estimates based on actual performance.
        
        Args:
            task: The task that was attempted
            outcome: How it went (success, quality, feedback)
            assessment: The original assessment if available
            
        Raises:
            ValueError: If task is empty or outcome is invalid
        """
        if not task:
            raise ValueError("task cannot be empty")
        
        success = outcome.get('success', False)
        quality = outcome.get('quality', 0.5)
        
        if not 0 <= quality <= 1:
            raise ValueError(f"quality must be between 0 and 1, got {quality}")
        
        # Identify capabilities used
        if assessment:
            required_caps = assessment.required_capabilities
            expected_proficiency = assessment.estimated_proficiency
        else:
            required_caps = self._identify_required_capabilities(task.lower())
            expected_proficiency = 0.5
        
        if not required_caps:
            logger.debug(f"No capabilities identified for task: {task[:30]}...")
            return
        
        # Calculate performance score
        if success:
            performance = quality
        else:
            performance = quality * 0.5  # Penalize failure
        
        # Check if assessment was accurate
        if assessment:
            error = abs(performance - expected_proficiency)
            if error < 0.2:
                self.accurate_assessments += 1
        
        # Update each relevant capability
        for cap_name in required_caps:
            if cap_name in self.capabilities:
                cap = self.capabilities[cap_name]
                
                # Calculate update
                old_prof = cap.proficiency
                update = (performance - old_prof) * cap.learning_rate
                
                # Apply update with bounds
                new_prof = max(0.1, min(1.0, old_prof + update))
                cap.proficiency = new_prof
                
                # Update confidence (more data = higher confidence)
                cap.times_used += 1
                cap.confidence = min(1.0, cap.confidence + 0.01)
                cap.last_used = time.time()
                
                # Update performance history
                cap.performance_history.append(performance)
                if len(cap.performance_history) > 20:
                    cap.performance_history.pop(0)
                
                # Update learning potential
                if new_prof > 0.9:
                    cap.learning_potential = 0.1
                elif new_prof > 0.7:
                    cap.learning_potential = 0.3
                elif new_prof > 0.5:
                    cap.learning_potential = 0.5
        
        # Record performance
        self.performance_history.append({
            'task': task[:100],
            'success': success,
            'quality': quality,
            'performance': performance,
            'timestamp': time.time()
        })
        
        # Maintain history size
        if len(self.performance_history) > self.max_history:
            self.performance_history = self.performance_history[-self.max_history:]
        
        logger.debug(f"Updated capabilities from task: '{task[:30]}...' "
                    f"(success={success}, quality={quality:.2f})")
    
    def get_learning_needs(self, min_potential: float = 0.4) -> List[Dict[str, Any]]:
        """
        Identify what Wednesday should learn next.
        
        Args:
            min_potential: Minimum learning potential to consider
            
        Returns:
            List of learning recommendations sorted by priority
        """
        needs = []
        
        for name, cap in self.capabilities.items():
            # Calculate learning priority
            if cap.learning_potential >= min_potential and cap.proficiency < 0.85:
                # Priority based on learning potential and gap
                gap = 1.0 - cap.proficiency
                priority = cap.learning_potential * gap * (1.0 + cap.times_used / 100)
                
                needs.append({
                    'capability': name,
                    'domain': cap.domain.value,
                    'current_proficiency': round(cap.proficiency, 3),
                    'learning_potential': round(cap.learning_potential, 3),
                    'priority': round(priority, 3),
                    'times_used': cap.times_used,
                    'recommendation': self._generate_learning_recommendation(name, cap)
                })
        
        # Sort by priority
        needs.sort(key=lambda x: x['priority'], reverse=True)
        
        return needs[:10]
    
    def get_capability_summary(self, domain: Optional[CapabilityDomain] = None) -> Dict[str, Any]:
        """
        Get summary of capabilities.
        
        Args:
            domain: Optional domain to filter by
            
        Returns:
            Dictionary with capability summary
        """
        caps = list(self.capabilities.values())
        
        if domain:
            caps = [c for c in caps if c.domain == domain]
        
        if not caps:
            return {}
        
        avg_proficiency = sum(c.proficiency for c in caps) / len(caps)
        avg_confidence = sum(c.confidence for c in caps) / len(caps)
        
        # Group by level
        levels = Counter()
        for cap in caps:
            levels[cap.get_level().name] += 1
        
        return {
            'total_capabilities': len(caps),
            'average_proficiency': round(avg_proficiency, 3),
            'average_confidence': round(avg_confidence, 3),
            'by_level': dict(levels),
            'strongest': self._get_top_capabilities(caps, 3),
            'weakest': self._get_bottom_capabilities(caps, 3)
        }
    
    def get_confidence(self, task_type: str) -> float:
        """
        Get confidence score for a task type.
        
        Args:
            task_type: Type of task
            
        Returns:
            Confidence score (0-1)
        """
        # Look for similar task in history
        task_lower = task_type.lower()
        
        for task, confidence in self.confidence_scores.items():
            if task_lower in task.lower() or task.lower() in task_lower:
                return confidence
        
        # Default based on general capability
        return 0.5
    
    def assess_task_difficulty(self, task: str) -> float:
        """
        Assess how difficult a task would be (0-1).
        
        Args:
            task: Task description
            
        Returns:
            Difficulty score (0-1, higher = more difficult)
        """
        assessment = self.assess_capability(task)
        return assessment.estimated_difficulty
    
    def can_perform(self, task: str, min_proficiency: float = 0.4) -> bool:
        """
        Quick check if Wednesday can perform a task.
        
        Args:
            task: Task description
            min_proficiency: Minimum proficiency required
            
        Returns:
            True if can perform
        """
        assessment = self.assess_capability(task)
        return assessment.can_perform and assessment.estimated_proficiency >= min_proficiency
    
    def _load_base_capabilities(self) -> None:
        """Load base capability definitions"""
        for name, config in self.BASE_CAPABILITIES.items():
            self.capabilities[name] = Capability(
                name=name,
                domain=config['domain'],
                proficiency=config['base_proficiency'],
                confidence=config['base_confidence'],
                learning_potential=config['learning_potential'],
                known_limitations=config['limitations']
            )
    
    def _identify_required_capabilities(self, task_lower: str, 
                                          context: Optional[Dict] = None) -> List[str]:
        """Identify capabilities needed for a task"""
        required = set()
        
        # Keyword-based identification
        for keyword, caps in self.CAPABILITY_KEYWORDS.items():
            if keyword in task_lower:
                required.update(caps)
        
        # Context-based identification
        if context:
            # Add capabilities based on context type
            context_type = context.get('type', '')
            if context_type == 'emotional':
                required.add('emotional_understanding')
                required.add('empathy')
            elif context_type == 'technical':
                required.add('logical_reasoning')
                required.add('problem_solving')
            elif context_type == 'creative':
                required.add('creative_generation')
            
            # Add based on relationship
            relationship = context.get('relationship', '')
            if relationship in ['friend', 'close_friend']:
                required.add('social_understanding')
                required.add('relationship_maintenance')
        
        return list(required)
    
    def _calculate_assessment_confidence(self, 
                                          required_caps: List[str],
                                          proficiencies: List[float],
                                          context: Optional[Dict]) -> float:
        """Calculate confidence in an assessment"""
        if not required_caps:
            return 0.3
        
        # Base confidence on familiarity with capabilities
        known_caps = sum(1 for cap in required_caps if cap in self.capabilities)
        familiarity = known_caps / len(required_caps) if required_caps else 0
        
        # Confidence increases with proficiency (easier to assess what you know well)
        if proficiencies:
            avg_prof = sum(proficiencies) / len(proficiencies)
            proficiency_confidence = avg_prof
        else:
            proficiency_confidence = 0.5
        
        # Experience factor (more experience = higher confidence)
        experience_factor = 0.5
        for cap_name in required_caps[:3]:  # Check first few
            if cap_name in self.capabilities:
                cap = self.capabilities[cap_name]
                if cap.times_used > 10:
                    experience_factor = min(1.0, experience_factor + 0.1)
        
        # Combine factors
        confidence = (0.3 * familiarity + 
                     0.3 * proficiency_confidence + 
                     0.2 * experience_factor + 
                     0.2)
        
        return min(1.0, confidence)
    
    def _generate_recommended_approach(self, 
                                         task: str,
                                         proficiency: float,
                                         limitations: List[str]) -> str:
        """Generate recommended approach based on capability assessment"""
        if proficiency < 0.3:
            if limitations:
                return f"I should be cautious with this task. {limitations[0]}"
            else:
                return "I'm not very confident about this task. Let's approach it carefully."
        elif proficiency < 0.6:
            if limitations:
                return f"I can try this, but need to be careful about {limitations[0]}"
            else:
                return "I can attempt this, though it may be challenging."
        else:
            return "I'm confident I can handle this effectively."
    
    def _generate_learning_recommendation(self, cap_name: str, cap: Capability) -> str:
        """Generate learning recommendation for a capability"""
        display_name = cap_name.replace('_', ' ')
        
        if cap.proficiency < 0.4:
            return f"Focus on building basic {display_name} skills through practice"
        elif cap.proficiency < 0.7:
            return f"Practice {display_name} to reach intermediate level with varied examples"
        else:
            return f"Work on edge cases and advanced scenarios in {display_name}"
    
    def _get_top_capabilities(self, caps: List[Capability], n: int) -> List[Dict[str, Any]]:
        """Get top n capabilities by proficiency"""
        sorted_caps = sorted(caps, key=lambda c: c.proficiency, reverse=True)
        return [{'name': c.name, 'proficiency': round(c.proficiency, 3)} 
                for c in sorted_caps[:n]]
    
    def _get_bottom_capabilities(self, caps: List[Capability], n: int) -> List[Dict[str, Any]]:
        """Get bottom n capabilities by proficiency"""
        sorted_caps = sorted(caps, key=lambda c: c.proficiency)
        return [{'name': c.name, 'proficiency': round(c.proficiency, 3)} 
                for c in sorted_caps[:n]]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get assessment system statistics"""
        total_updates = sum(c.times_used for c in self.capabilities.values())
        
        return {
            'total_capabilities': len(self.capabilities),
            'assessments_made': self.assessments_made,
            'accurate_assessments': self.accurate_assessments,
            'accuracy_rate': round(self.accurate_assessments / max(1, self.assessments_made), 3),
            'total_updates': total_updates,
            'avg_updates_per_capability': round(total_updates / max(1, len(self.capabilities)), 1),
            'performance_records': len(self.performance_history),
            'learning_needs': len(self.get_learning_needs())
        }
    
    def reset_statistics(self) -> None:
        """Reset assessment statistics"""
        self.assessments_made = 0
        self.accurate_assessments = 0
        self.confidence_scores.clear()
        logger.info("Assessment statistics reset")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("=== Capability Assessment Module Test ===\n")
    
    # Create assessment system
    assessment = CapabilityAssessment()
    
    # Test task assessments
    test_tasks = [
        "Write a darkly humorous response",
        "Remember what we talked about last week",
        "Solve a complex mystery",
        "Feel empathy for someone's loss",
        "Plan a long-term strategy",
        "Create an original poem",
        "Understand a complex social situation",
        "Learn a new concept quickly",
        "Hear and transcribe audio",
        "Explain quantum physics",
        "Make a friend feel better",
        "Analyze a logical paradox"
    ]
    
    print("--- Task Assessments ---")
    for i, task in enumerate(test_tasks[:6]):  # First 6 tasks
        print(f"\nTask {i+1}: '{task}'")
        result = assessment.assess_capability(task)
        
        print(f"  Can perform: {result.can_perform}")
        print(f"  Proficiency: {result.estimated_proficiency:.2f}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Required caps: {', '.join(result.required_capabilities[:3])}")
        print(f"  Limitations: {result.limitations}")
        print(f"  Approach: {result.recommended_approach}")
    
    # Test performance updates
    print("\n--- Performance Updates ---")
    test_performances = [
        {
            'task': 'Write a darkly humorous response',
            'outcome': {'success': True, 'quality': 0.9}
        },
        {
            'task': 'Feel empathy for someone\'s loss',
            'outcome': {'success': False, 'quality': 0.4}
        },
        {
            'task': 'Solve a complex mystery',
            'outcome': {'success': True, 'quality': 0.8}
        }
    ]
    
    for i, perf in enumerate(test_performances):
        print(f"\nUpdate {i+1}: {perf['task']}")
        print(f"  Outcome: success={perf['outcome']['success']}, quality={perf['outcome']['quality']}")
        
        # Get before state for relevant capability
        caps = assessment._identify_required_capabilities(perf['task'].lower())
        if caps:
            cap_name = caps[0]
            if cap_name in assessment.capabilities:
                before = assessment.capabilities[cap_name].proficiency
                print(f"  '{cap_name}' before: {before:.3f}")
        
        # Update
        assessment.update_from_performance(
            task=perf['task'],
            outcome=perf['outcome']
        )
        
        # Get after state
        if caps and cap_name in assessment.capabilities:
            after = assessment.capabilities[cap_name].proficiency
            print(f"  '{cap_name}' after: {after:.3f}")
    
    # Get learning needs
    print("\n--- Learning Needs (Top 5) ---")
    needs = assessment.get_learning_needs()
    for i, need in enumerate(needs[:5]):
        print(f"  {i+1}. {need['capability']}: {need['recommendation']} "
              f"(priority: {need['priority']:.2f})")
    
    # Get capability summary
    print("\n--- Capability Summary ---")
    summary = assessment.get_capability_summary()
    for key, value in summary.items():
        if key not in ['strongest', 'weakest']:
            print(f"  {key}: {value}")
    
    print("\n  Strongest capabilities:")
    for cap in summary.get('strongest', []):
        print(f"    - {cap['name']}: {cap['proficiency']:.2f}")
    
    print("\n  Weakest capabilities:")
    for cap in summary.get('weakest', []):
        print(f"    - {cap['name']}: {cap['proficiency']:.2f}")
    
    # Get statistics
    print("\n--- Statistics ---")
    stats = assessment.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n=== Test Complete ===")