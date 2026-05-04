"""
stimulus_evaluation.py - Core appraisal system for Wednesday AI

This module implements the primary appraisal mechanism that evaluates events
and stimuli for emotional significance. Based on cognitive appraisal theories
(Lazarus, 1991; Scherer, 2001; Smith & Ellsworth, 1985), it determines how
events relate to Wednesday's goals, well-being, and values.

Key improvements:
- Removed numpy dependency (using pure Python math)
- Added proper validation and error handling
- Fixed mock class implementations
- Enhanced type safety with comprehensive type hints
- Added proper import handling for optional dependencies
"""

import time
import logging
import math
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# Use try/except for optional imports to allow graceful fallback
try:
    from wednesday.emotion.affect import PADVector, EmotionLexicon
except ImportError:
    # Define minimal versions for standalone testing
    class PADVector:
        def __init__(self, valence=0.0, arousal=0.5, dominance=0.5):
            self.valence = valence
            self.arousal = arousal
            self.dominance = dominance
        def __repr__(self):
            return f"PAD(V={self.valence:.2f}, A={self.arousal:.2f}, D={self.dominance:.2f})"
    
    class EmotionLexicon:
        @classmethod
        def has_emotion(cls, name):
            return True

try:
    from wednesday.cognition.goal_manager import Goal, GoalPriority
except ImportError:
    # Define minimal enums for standalone testing
    class GoalPriority(Enum):
        CRITICAL = "critical"
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"
    
    class Goal:
        def __init__(self, id=None, name="", description="", priority=GoalPriority.MEDIUM):
            self.id = id
            self.name = name
            self.description = description
            self.priority = priority

# Configure logging
logger = logging.getLogger(__name__)


class AppraisalDimension(Enum):
    """Enumeration of appraisal dimensions"""
    NOVELTY = "novelty"
    VALENCE = "valence"
    GOAL_RELEVANCE = "goal_relevance"
    GOAL_CONGRUENCE = "goal_congruence"  # Helps vs hinders goals
    COPING_POTENTIAL = "coping_potential"
    NORM_COMPATIBILITY = "norm_compatibility"
    AGENCY = "agency"  # Who caused this?
    CERTAINTY = "certainty"
    EFFORT = "effort"  # Expected effort to handle
    ATTENTION = "attention"  # How attention-grabbing


class AgencyType(Enum):
    """Who or what caused an event"""
    SELF = "self"
    OTHER = "other"
    CIRCUMSTANCE = "circumstance"
    UNKNOWN = "unknown"


@dataclass
class AppraisalResult:
    """
    Complete appraisal of a stimulus across all dimensions.
    
    This is the output of the appraisal process, used to generate
    emotional responses and guide cognitive processing.
    """
    # Core appraisal dimensions
    novelty: float  # 0-1 (expected to extremely surprising)
    valence: float  # -1 to 1 (negative to positive)
    goal_relevance: float  # 0-1 (irrelevant to critically relevant)
    goal_congruence: float  # -1 to 1 (hinders to helps)
    coping_potential: float  # 0-1 (no control to full control)
    norm_compatibility: float  # -1 to 1 (violates to aligns with values)
    
    # Extended dimensions
    agency: AgencyType = AgencyType.UNKNOWN
    certainty: float = 0.5  # 0-1 (very uncertain to completely certain)
    effort: float = 0.5  # 0-1 (no effort to extreme effort)
    attention: float = 0.5  # 0-1 (ignored to completely absorbing)
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    stimulus_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    
    # Resulting emotional tendencies
    primary_emotion: Optional[str] = None
    secondary_emotions: List[Tuple[str, float]] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate appraisal dimensions"""
        self._validate_float(self.novelty, 0, 1, "novelty")
        self._validate_float(self.valence, -1, 1, "valence")
        self._validate_float(self.goal_relevance, 0, 1, "goal_relevance")
        self._validate_float(self.goal_congruence, -1, 1, "goal_congruence")
        self._validate_float(self.coping_potential, 0, 1, "coping_potential")
        self._validate_float(self.norm_compatibility, -1, 1, "norm_compatibility")
        self._validate_float(self.certainty, 0, 1, "certainty")
        self._validate_float(self.effort, 0, 1, "effort")
        self._validate_float(self.attention, 0, 1, "attention")
    
    def _validate_float(self, value: float, min_val: float, max_val: float, name: str):
        """Validate float is within range"""
        if not min_val <= value <= max_val:
            raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'novelty': round(self.novelty, 3),
            'valence': round(self.valence, 3),
            'goal_relevance': round(self.goal_relevance, 3),
            'goal_congruence': round(self.goal_congruence, 3),
            'coping_potential': round(self.coping_potential, 3),
            'norm_compatibility': round(self.norm_compatibility, 3),
            'agency': self.agency.value,
            'certainty': round(self.certainty, 3),
            'effort': round(self.effort, 3),
            'attention': round(self.attention, 3),
            'primary_emotion': self.primary_emotion,
            'secondary_emotions': [(e, round(i, 3)) for e, i in self.secondary_emotions],
            'timestamp': self.timestamp
        }
    
    def get_emotional_vector(self) -> 'PADVector':
        """Convert appraisal to a PAD emotional vector"""
        # Base emotional tendency from valence and goal congruence
        base_valence = (self.valence + self.goal_congruence) / 2
        
        # Arousal from novelty, certainty, and effort
        base_arousal = (self.novelty * 0.4 + 
                       (1 - self.certainty) * 0.3 + 
                       self.effort * 0.3)
        
        # Dominance from coping potential and agency
        agency_factor = {
            AgencyType.SELF: 0.3,
            AgencyType.OTHER: -0.2,
            AgencyType.CIRCUMSTANCE: 0.0,
            AgencyType.UNKNOWN: 0.0
        }.get(self.agency, 0.0)
        
        base_dominance = self.coping_potential * 0.7 + agency_factor
        
        return PADVector(
            valence=max(-1.0, min(1.0, base_valence)),
            arousal=max(0.0, min(1.0, base_arousal)),
            dominance=max(0.0, min(1.0, base_dominance))
        )


@dataclass
class Stimulus:
    """
    Representation of an event or stimulus to be appraised.
    
    This could be external (user input, sensor data) or internal
    (thoughts, memories, bodily states).
    """
    content: Any  # The actual stimulus content
    source: str  # Where it came from (perception, memory, cognition)
    timestamp: float = field(default_factory=time.time)
    
    # Optional pre-processed features
    features: Dict[str, float] = field(default_factory=dict)
    context: Optional[Dict[str, Any]] = None
    
    # For internal stimuli
    is_internal: bool = False
    related_goal_id: Optional[str] = None
    
    def __post_init__(self):
        """Generate ID if not present"""
        if not hasattr(self, 'id'):
            self.id = f"stim_{int(self.timestamp * 1000)}_{hash(str(self.content)) % 10000}"


class StimulusEvaluator:
    """
    Evaluates stimuli for emotional significance using multi-dimensional appraisal.
    
    The appraisal process follows these steps:
    1. Extract relevant features from the stimulus
    2. Evaluate each appraisal dimension based on stimulus, context, and goals
    3. Map appraisal pattern to emotional tendencies
    4. Return complete appraisal result for emotional response generation
    
    Wednesday's appraisal style:
    - High attention to norm violations (her values matter)
    - Realistic coping potential assessment (not overconfident)
    - Nuanced goal relevance (cares about specific things)
    - Dark humor as a coping mechanism
    """
    
    # Valid range constants
    VALENCE_MIN = -1.0
    VALENCE_MAX = 1.0
    PROB_MIN = 0.0
    PROB_MAX = 1.0
    
    # Default weights for appraisal dimensions
    DIMENSION_WEIGHTS = {
        'novelty': 0.15,
        'valence': 0.20,
        'goal_relevance': 0.25,
        'goal_congruence': 0.25,
        'coping_potential': 0.10,
        'norm_compatibility': 0.20,
        'certainty': 0.05,
        'effort': 0.05
    }
    
    # Appraisal-to-emotion mapping rules
    # Format: (conditions) -> emotion_name, intensity_factor
    EMOTION_RULES = [
        # Positive emotions
        {
            'conditions': lambda a: a.valence > 0.3 and a.goal_congruence > 0.3 and a.coping_potential > 0.5,
            'emotion': 'joy',
            'intensity': lambda a: (a.valence * 0.4 + a.goal_congruence * 0.4 + a.coping_potential * 0.2)
        },
        {
            'conditions': lambda a: a.goal_congruence > 0.4 and a.agency == AgencyType.SELF,
            'emotion': 'pride',
            'intensity': lambda a: a.goal_congruence * 0.7 + a.coping_potential * 0.3
        },
        {
            'conditions': lambda a: a.valence > 0.2 and a.novelty > 0.6,
            'emotion': 'surprise',
            'intensity': lambda a: a.novelty * 0.8 + a.valence * 0.2
        },
        {
            'conditions': lambda a: a.valence > 0.2 and a.novelty < 0.3 and a.coping_potential > 0.6,
            'emotion': 'contentment',
            'intensity': lambda a: a.valence * 0.5 + a.coping_potential * 0.5
        },
        
        # Negative emotions
        {
            'conditions': lambda a: a.valence < -0.3 and a.goal_congruence < -0.3,
            'emotion': 'sadness',
            'intensity': lambda a: (abs(a.valence) * 0.5 + abs(a.goal_congruence) * 0.5)
        },
        {
            'conditions': lambda a: a.valence < -0.2 and a.goal_congruence < -0.2 and a.agency == AgencyType.OTHER,
            'emotion': 'anger',
            'intensity': lambda a: (abs(a.valence) * 0.3 + abs(a.goal_congruence) * 0.4 + 
                                   0.3)  # Agency bonus
        },
        {
            'conditions': lambda a: a.valence < -0.2 and a.coping_potential < 0.3 and a.certainty < 0.4,
            'emotion': 'fear',
            'intensity': lambda a: (abs(a.valence) * 0.3 + (1 - a.coping_potential) * 0.4 + 
                                   (1 - a.certainty) * 0.3)
        },
        {
            'conditions': lambda a: a.valence < -0.2 and a.norm_compatibility < -0.4,
            'emotion': 'disgust',
            'intensity': lambda a: (abs(a.valence) * 0.4 + abs(a.norm_compatibility) * 0.6)
        },
        
        # Wednesday-specific emotions
        {
            'conditions': lambda a: (a.valence > 0.1 and a.valence < 0.5 and 
                                     a.norm_compatibility < -0.2 and a.coping_potential > 0.6),
            'emotion': 'dark_amusement',
            'intensity': lambda a: (a.valence * 0.3 + abs(a.norm_compatibility) * 0.4 + 
                                   a.coping_potential * 0.3)
        },
        {
            'conditions': lambda a: (a.novelty > 0.5 and a.goal_relevance < 0.3 and 
                                     a.coping_potential > 0.5),
            'emotion': 'curiously_detached',
            'intensity': lambda a: (a.novelty * 0.5 + a.coping_potential * 0.5)
        },
        {
            'conditions': lambda a: (a.norm_compatibility < -0.3 and a.goal_relevance > 0.4 and 
                                     a.coping_potential > 0.4),
            'emotion': 'protective',
            'intensity': lambda a: (abs(a.norm_compatibility) * 0.4 + a.goal_relevance * 0.4 + 
                                   a.coping_potential * 0.2)
        },
        {
            'conditions': lambda a: (abs(a.valence) < 0.2 and a.goal_relevance < 0.2 and 
                                     a.novelty < 0.3 and a.coping_potential > 0.5),
            'emotion': 'pensive',
            'intensity': lambda a: 0.3 + a.coping_potential * 0.3
        },
        {
            'conditions': lambda a: (a.valence < -0.1 and a.norm_compatibility < -0.2 and 
                                     a.coping_potential > 0.7 and a.agency != AgencyType.SELF),
            'emotion': 'disdainful',
            'intensity': lambda a: (abs(a.valence) * 0.3 + abs(a.norm_compatibility) * 0.4 + 
                                   a.coping_potential * 0.3)
        },
        {
            'conditions': lambda a: (a.novelty > 0.4 and a.certainty < 0.3 and 
                                     a.coping_potential > 0.4 and a.coping_potential < 0.7),
            'emotion': 'wary',
            'intensity': lambda a: (a.novelty * 0.3 + (1 - a.certainty) * 0.4 + 
                                   (0.5 - abs(a.coping_potential - 0.5)) * 0.3)
        }
    ]
    
    def __init__(self, 
                 mood_engine: Optional[Any] = None, 
                 goal_manager: Optional[Any] = None, 
                 personality: Optional[Dict[str, float]] = None):
        """
        Initialize the stimulus evaluator.
        
        Args:
            mood_engine: Reference to mood engine for mood-congruent bias
            goal_manager: Reference to goal manager for goal relevance assessment
            personality: Optional personality parameters
            
        Raises:
            ValueError: If personality parameters are invalid
        """
        self.mood_engine = mood_engine
        self.goal_manager = goal_manager
        
        # Personality influences on appraisal
        default_personality = {
            'novelty_sensitivity': 0.6,      # How much novelty matters
            'goal_focus': 0.8,                # How goal-driven she is
            'norm_adherence': 0.7,             # How much she cares about norms/values
            'optimism_bias': 0.3,               # Tendency to see positive outcomes
            'control_preference': 0.8,          # Preference for controllable situations
            'dark_humor_coping': 0.7,            # Uses dark humor as coping
        }
        
        self.personality = default_personality.copy()
        if personality:
            self._validate_personality(personality)
            self.personality.update(personality)
        
        # Appraisal history for context
        self.appraisal_history: List[AppraisalResult] = []
        self.max_history_size = 50
        
        # Simple keyword lists for text analysis
        self._init_keyword_lists()
        
        logger.info("StimulusEvaluator initialized")
    
    def _validate_personality(self, personality: Dict[str, float]) -> None:
        """Validate personality parameters"""
        for key, value in personality.items():
            if key not in self.personality:
                raise ValueError(f"Unknown personality parameter: {key}")
            if not 0 <= value <= 1:
                raise ValueError(f"Personality parameter {key} must be between 0 and 1, got {value}")
    
    def _init_keyword_lists(self) -> None:
        """Initialize keyword lists for text analysis"""
        self.positive_words = {
            'good', 'great', 'happy', 'love', 'wonderful', 'excellent', 'awesome',
            'fantastic', 'brilliant', 'perfect', 'nice', 'pleased', 'glad', 'joy'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'hate', 'horrible', 'evil', 'worst',
            'dreadful', 'poor', 'unpleasant', 'dislike', 'anger', 'sad'
        }
        
        self.rare_words = {
            'unusual', 'strange', 'odd', 'peculiar', 'unexpected', 'surprising',
            'rare', 'unique', 'extraordinary', 'remarkable', 'curious'
        }
        
        self.value_words = {
            'loyalty': {'loyal', 'faithful', 'devoted', 'trust'},
            'justice': {'justice', 'fair', 'right', 'just', 'equality'},
            'authenticity': {'authentic', 'real', 'genuine', 'truth', 'honest'},
            'curiosity': {'curious', 'interested', 'wonder', 'explore'},
            'independence': {'independent', 'freedom', 'autonomy', 'alone'}
        }
        
        self.violation_words = {'betray', 'lie', 'fake', 'deceive', 'dishonest', 'unfair'}
        self.support_words = {'truth', 'real', 'honest', 'fair', 'justice'}
    
    def evaluate(self, 
                stimulus: Stimulus, 
                context: Optional[Dict[str, Any]] = None,
                active_goals: Optional[List[Goal]] = None) -> AppraisalResult:
        """
        Perform full appraisal of a stimulus.
        
        Args:
            stimulus: The stimulus to evaluate
            context: Current context information
            active_goals: Currently active goals (if None, fetched from goal_manager)
            
        Returns:
            Complete appraisal result
            
        Raises:
            ValueError: If stimulus is invalid
        """
        if not isinstance(stimulus, Stimulus):
            raise ValueError(f"Expected Stimulus object, got {type(stimulus)}")
        
        # Get active goals if not provided
        if active_goals is None and self.goal_manager:
            try:
                active_goals = self.goal_manager.get_active_goals()
            except AttributeError:
                logger.warning("goal_manager has no get_active_goals method")
                active_goals = []
        else:
            active_goals = active_goals or []
        
        # Extract stimulus features
        features = self._extract_features(stimulus)
        
        # Apply mood-congruent bias to initial perception
        biased_features = self._apply_mood_bias(features)
        
        # Evaluate each appraisal dimension
        novelty = self._evaluate_novelty(stimulus, biased_features, context)
        valence = self._evaluate_valence(stimulus, biased_features, context)
        goal_relevance, goal_congruence = self._evaluate_goal_significance(
            stimulus, biased_features, active_goals, context
        )
        coping_potential = self._evaluate_coping_potential(stimulus, biased_features, context)
        norm_compatibility = self._evaluate_norm_compatibility(stimulus, biased_features, context)
        agency = self._evaluate_agency(stimulus, biased_features, context)
        certainty = self._evaluate_certainty(stimulus, biased_features, context)
        effort = self._evaluate_effort(stimulus, biased_features, context)
        attention = self._evaluate_attention(stimulus, biased_features, context)
        
        # Create appraisal result
        appraisal = AppraisalResult(
            novelty=novelty,
            valence=valence,
            goal_relevance=goal_relevance,
            goal_congruence=goal_congruence,
            coping_potential=coping_potential,
            norm_compatibility=norm_compatibility,
            agency=agency,
            certainty=certainty,
            effort=effort,
            attention=attention,
            timestamp=time.time(),
            stimulus_id=stimulus.id,
            context=context
        )
        
        # Determine emotional responses
        primary, secondaries = self._map_appraisal_to_emotions(appraisal)
        appraisal.primary_emotion = primary
        appraisal.secondary_emotions = secondaries
        
        # Store in history
        self._add_to_history(appraisal)
        
        logger.debug(f"Appraisal complete: {appraisal.primary_emotion} "
                    f"(valence={valence:.2f}, relevance={goal_relevance:.2f})")
        
        return appraisal
    
    def _apply_mood_bias(self, features: Dict[str, float]) -> Dict[str, float]:
        """Apply mood-congruent bias if mood engine is available"""
        if self.mood_engine and hasattr(self.mood_engine, 'mood_congruent_bias'):
            try:
                return self.mood_engine.mood_congruent_bias(features)
            except Exception as e:
                logger.warning(f"Failed to apply mood bias: {e}")
        
        return features.copy()
    
    def _extract_features(self, stimulus: Stimulus) -> Dict[str, float]:
        """Extract relevant numerical features from stimulus"""
        features = {}
        
        # If stimulus already has features, use them as base
        if stimulus.features:
            features.update(stimulus.features)
        
        # Extract from content based on type
        content = stimulus.content
        
        if isinstance(content, dict):
            # Dictionary content
            for key in ['valence', 'intensity', 'threat', 'opportunity', 
                       'novelty', 'complexity', 'ambiguity', 'controllability',
                       'certainty', 'effort', 'agency']:
                if key in content:
                    try:
                        features[key] = float(content[key])
                    except (ValueError, TypeError):
                        pass
        
        elif isinstance(content, str):
            # Text content - simple heuristics
            text_features = self._analyze_text(content)
            features.update(text_features)
        
        elif isinstance(content, (int, float)):
            # Numeric content
            features['intensity'] = min(1.0, abs(float(content)) / 10)
        
        return features
    
    def _analyze_text(self, text: str) -> Dict[str, float]:
        """Simple text analysis for feature extraction"""
        features = {}
        text_lower = text.lower()
        words = set(text_lower.split())
        
        # Valence analysis
        pos_count = sum(1 for word in words if word in self.positive_words)
        neg_count = sum(1 for word in words if word in self.negative_words)
        
        if pos_count > 0 or neg_count > 0:
            total = pos_count + neg_count
            features['valence'] = (pos_count - neg_count) / total
        
        # Novelty analysis
        rare_count = sum(1 for word in words if word in self.rare_words)
        if rare_count > 0:
            features['novelty'] = min(1.0, rare_count * 0.3)
        
        # Norm compatibility analysis
        violation_count = sum(1 for word in words if word in self.violation_words)
        support_count = sum(1 for word in words if word in self.support_words)
        
        if violation_count > 0 or support_count > 0:
            total = violation_count + support_count
            features['norm_compatibility'] = (support_count - violation_count) / total
        
        return features
    
    def _evaluate_novelty(self, stimulus: Stimulus, 
                          features: Dict[str, float], 
                          context: Optional[Dict]) -> float:
        """Evaluate how novel/surprising the stimulus is"""
        # Base novelty from features
        base_novelty = features.get('novelty', 0.3)
        
        # Check if similar stimulus in recent history
        if self.appraisal_history:
            # Simple recency check
            recent_ids = [a.stimulus_id for a in self.appraisal_history[-5:] if a.stimulus_id]
            if stimulus.id in recent_ids:
                base_novelty *= 0.5  # Less novel if seen recently
        
        # Personality modulation
        base_novelty *= self.personality['novelty_sensitivity']
        
        return max(0.0, min(1.0, base_novelty))
    
    def _evaluate_valence(self, stimulus: Stimulus,
                          features: Dict[str, float], 
                          context: Optional[Dict]) -> float:
        """Evaluate inherent pleasantness/unpleasantness"""
        # Direct valence if provided
        if 'valence' in features:
            base_valence = features['valence']
        else:
            # Default to neutral with slight negative bias (Wednesday)
            base_valence = -0.1
        
        # Apply optimism/pessimism bias
        if base_valence > 0:
            base_valence *= (1 + self.personality['optimism_bias'])
        elif base_valence < 0:
            base_valence *= (1 - self.personality['optimism_bias'] * 0.5)
        
        return max(-1.0, min(1.0, base_valence))
    
    def _evaluate_goal_significance(self, stimulus: Stimulus,
                                    features: Dict[str, float],
                                    active_goals: List[Goal],
                                    context: Optional[Dict]) -> Tuple[float, float]:
        """
        Evaluate relevance to goals and whether it helps or hinders.
        
        Returns:
            Tuple of (relevance 0-1, congruence -1 to 1)
        """
        if not active_goals:
            return 0.0, 0.0
        
        relevance = 0.0
        congruence = 0.0
        
        for goal in active_goals:
            # Check if stimulus relates to this goal
            goal_relevance = self._check_goal_relevance(stimulus, goal, features)
            
            if goal_relevance > 0:
                # Determine if it helps or hinders
                goal_congruence = self._check_goal_congruence(stimulus, goal, features)
                
                # Weight by goal priority
                priority_weight = {
                    GoalPriority.CRITICAL: 1.0,
                    GoalPriority.HIGH: 0.8,
                    GoalPriority.MEDIUM: 0.5,
                    GoalPriority.LOW: 0.3
                }.get(goal.priority, 0.5)
                
                relevance = max(relevance, goal_relevance * priority_weight)
                
                # Congruence is signed, so we take the most extreme
                if abs(goal_congruence) > abs(congruence):
                    congruence = goal_congruence * priority_weight
        
        # Apply personality focus on goals
        relevance *= self.personality['goal_focus']
        
        return max(0.0, min(1.0, relevance)), max(-1.0, min(1.0, congruence))
    
    def _check_goal_relevance(self, stimulus: Stimulus, 
                              goal: Goal, 
                              features: Dict[str, float]) -> float:
        """Check how relevant stimulus is to a specific goal"""
        # If stimulus explicitly mentions goal
        if hasattr(stimulus, 'related_goal_id') and stimulus.related_goal_id == goal.id:
            return 0.8
        
        # Simple keyword matching for text content
        if isinstance(stimulus.content, str) and hasattr(goal, 'description') and goal.description:
            # Check if goal keywords in stimulus
            goal_words = set(goal.description.lower().split())
            stimulus_words = set(stimulus.content.lower().split())
            overlap = goal_words.intersection(stimulus_words)
            
            if overlap:
                return min(0.5, len(overlap) * 0.1)
        
        return 0.0
    
    def _check_goal_congruence(self, stimulus: Stimulus,
                               goal: Goal, 
                               features: Dict[str, float]) -> float:
        """Check if stimulus helps (positive) or hinders (negative) goal"""
        # Extract valence relative to goal
        if 'valence' in features:
            # If goal is to avoid something, valence flips
            if hasattr(goal, 'desired_outcome') and goal.desired_outcome == 'avoid':
                return -features['valence']
            return features['valence']
        
        # Default to slightly positive if relevant
        return 0.1
    
    def _evaluate_coping_potential(self, stimulus: Stimulus,
                                   features: Dict[str, float], 
                                   context: Optional[Dict]) -> float:
        """Evaluate ability to cope with or control the situation"""
        # Base from features
        base_coping = features.get('controllability', 0.5)
        
        # Adjust based on stimulus type
        if stimulus.is_internal:
            # Internal events (thoughts, memories) are more controllable
            base_coping += 0.2
        
        # Personality: control preference affects perceived coping
        base_coping *= self.personality['control_preference']
        
        return max(0.0, min(1.0, base_coping))
    
    def _evaluate_norm_compatibility(self, stimulus: Stimulus,
                                     features: Dict[str, float], 
                                     context: Optional[Dict]) -> float:
        """Evaluate alignment with personal norms and values"""
        # Wednesday's core values with importance weights
        core_values = {
            'loyalty': 0.9,
            'justice': 0.8,
            'authenticity': 0.9,
            'curiosity': 0.7,
            'independence': 0.8
        }
        
        # Check if norm_compatibility already in features
        if 'norm_compatibility' in features:
            base_compatibility = features['norm_compatibility']
        else:
            base_compatibility = 0.0
            
            # Analyze text content if available
            if isinstance(stimulus.content, str):
                content_lower = stimulus.content.lower()
                
                for value, importance in core_values.items():
                    # Check if value-related words appear
                    if value in content_lower:
                        # Check for violations vs support
                        if any(word in content_lower for word in self.violation_words):
                            base_compatibility -= importance * 0.5
                        elif any(word in content_lower for word in self.support_words):
                            base_compatibility += importance * 0.3
        
        # Personality modulation
        base_compatibility *= self.personality['norm_adherence']
        
        return max(-1.0, min(1.0, base_compatibility))
    
    def _evaluate_agency(self, stimulus: Stimulus,
                         features: Dict[str, float], 
                         context: Optional[Dict]) -> AgencyType:
        """Determine who caused the event"""
        if 'agency' in features:
            agency_value = features['agency']
            if isinstance(agency_value, str):
                agency_map = {
                    'self': AgencyType.SELF,
                    'other': AgencyType.OTHER,
                    'circumstance': AgencyType.CIRCUMSTANCE,
                    'unknown': AgencyType.UNKNOWN
                }
                return agency_map.get(agency_value.lower(), AgencyType.UNKNOWN)
        
        # Default based on stimulus type
        return AgencyType.SELF if stimulus.is_internal else AgencyType.CIRCUMSTANCE
    
    def _evaluate_certainty(self, stimulus: Stimulus,
                            features: Dict[str, float], 
                            context: Optional[Dict]) -> float:
        """Evaluate how predictable outcomes are"""
        base_certainty = features.get('certainty', 0.5)
        
        # Novelty reduces certainty
        if 'novelty' in features:
            base_certainty *= (1 - features['novelty'] * 0.5)
        
        return max(0.0, min(1.0, base_certainty))
    
    def _evaluate_effort(self, stimulus: Stimulus,
                         features: Dict[str, float], 
                         context: Optional[Dict]) -> float:
        """Evaluate expected effort to handle stimulus"""
        base_effort = features.get('effort', 0.3)
        
        # Coping potential inversely related to effort
        if 'controllability' in features:
            base_effort *= (1 - features['controllability'] * 0.5)
        
        return max(0.0, min(1.0, base_effort))
    
    def _evaluate_attention(self, stimulus: Stimulus,
                             features: Dict[str, float], 
                             context: Optional[Dict]) -> float:
        """Evaluate how attention-grabbing the stimulus is"""
        # Combination of novelty and relevance
        attention = 0.0
        
        if 'novelty' in features:
            attention += features['novelty'] * 0.4
        
        if 'goal_relevance' in features:
            attention += features['goal_relevance'] * 0.6
        elif stimulus.related_goal_id:
            attention += 0.5
        
        return max(0.0, min(1.0, attention))
    
    def _map_appraisal_to_emotions(self, appraisal: AppraisalResult) -> Tuple[Optional[str], List[Tuple[str, float]]]:
        """
        Map appraisal pattern to emotional responses.
        
        Returns:
            Tuple of (primary_emotion, list of secondary emotions with intensities)
        """
        emotions = []
        
        # Apply each rule
        for rule in self.EMOTION_RULES:
            try:
                if rule['conditions'](appraisal):
                    intensity = rule['intensity'](appraisal)
                    intensity = max(0.0, min(1.0, intensity))
                    
                    if intensity > 0.2:  # Only consider if above threshold
                        emotions.append((rule['emotion'], intensity))
            except Exception as e:
                logger.warning(f"Error applying emotion rule: {e}")
        
        # If no emotions triggered, default to neutral
        if not emotions:
            return None, []
        
        # Sort by intensity
        emotions.sort(key=lambda x: x[1], reverse=True)
        
        # Primary is strongest, rest are secondary
        primary = emotions[0][0]
        secondaries = emotions[1:3]  # Keep top 2 secondaries
        
        return primary, secondaries
    
    def _add_to_history(self, appraisal: AppraisalResult) -> None:
        """Add appraisal to history"""
        self.appraisal_history.append(appraisal)
        
        if len(self.appraisal_history) > self.max_history_size:
            self.appraisal_history.pop(0)
    
    def get_recent_appraisals(self, limit: int = 10) -> List[AppraisalResult]:
        """Get most recent appraisals"""
        if limit <= 0:
            return []
        return self.appraisal_history[-min(limit, len(self.appraisal_history)):]
    
    def get_emotional_trend(self) -> Dict[str, float]:
        """Calculate emotional trend from recent appraisals"""
        if len(self.appraisal_history) < 3:
            return {}
        
        recent = self.appraisal_history[-10:]
        
        emotion_counts = {}
        for appraisal in recent:
            if appraisal.primary_emotion:
                emotion_counts[appraisal.primary_emotion] = emotion_counts.get(appraisal.primary_emotion, 0) + 1
        
        total = sum(emotion_counts.values())
        if total == 0:
            return {}
        
        return {k: v/total for k, v in emotion_counts.items()}


# Mock classes for testing (only used when real ones aren't available)
class MockMoodEngine:
    """Mock mood engine for testing"""
    def mood_congruent_bias(self, features):
        return features


class MockGoalManager:
    """Mock goal manager for testing"""
    def get_active_goals(self):
        return [
            Goal(id="goal1", name="Solve mystery", description="Solve the current mystery", 
                 priority=GoalPriority.HIGH),
            Goal(id="goal2", name="Protect friend", description="Keep friend safe", 
                 priority=GoalPriority.CRITICAL)
        ]


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("=== Stimulus Evaluator Test ===\n")
    
    # Create evaluator with mock dependencies
    evaluator = StimulusEvaluator(
        mood_engine=MockMoodEngine(),
        goal_manager=MockGoalManager()
    )
    
    # Test stimuli
    test_stimuli = [
        Stimulus(
            content="Someone betrayed my trust",
            source="perception",
            features={'valence': -0.6, 'agency': 'other'},
            context={'situation': 'social'}
        ),
        Stimulus(
            content="Found an interesting clue",
            source="perception",
            features={'valence': 0.3, 'novelty': 0.7}
        ),
        Stimulus(
            content="Friend is in danger",
            source="perception",
            features={'valence': -0.7, 'goal_relevance': 0.9, 'agency': 'circumstance'}
        ),
        Stimulus(
            content="Someone made a dark joke",
            source="perception",
            features={'valence': 0.2, 'norm_compatibility': -0.3}
        ),
        Stimulus(
            content="Everything is fine and normal",
            source="perception",
            features={}
        ),
    ]
    
    for i, stimulus in enumerate(test_stimuli):
        print(f"\n--- Stimulus {i+1}: {stimulus.content} ---")
        appraisal = evaluator.evaluate(stimulus)
        
        print(f"Appraisal dimensions:")
        print(f"  Valence: {appraisal.valence:.2f}")
        print(f"  Goal relevance: {appraisal.goal_relevance:.2f}")
        print(f"  Goal congruence: {appraisal.goal_congruence:.2f}")
        print(f"  Coping potential: {appraisal.coping_potential:.2f}")
        print(f"  Norm compatibility: {appraisal.norm_compatibility:.2f}")
        print(f"  Agency: {appraisal.agency.value}")
        
        print(f"\nEmotional response:")
        print(f"  Primary: {appraisal.primary_emotion}")
        print(f"  Secondary: {appraisal.secondary_emotions}")
        
        # Get PAD vector
        pad = appraisal.get_emotional_vector()
        print(f"  PAD: {pad}")
    
    print("\n--- Emotional Trend ---")
    trend = evaluator.get_emotional_trend()
    for emotion, prob in trend.items():
        print(f"  {emotion}: {prob:.1%}")
    
    print("\n=== Test Complete ===")