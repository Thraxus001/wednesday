"""
values.py - Value system for Wednesday AI

This module defines Wednesday's moral and ethical framework - the principles
that guide her decisions, judgments, and actions. Values represent what she
considers important and right, serving as an internal compass that influences
everything from everyday choices to moral dilemmas.

Key improvements:
- Added comprehensive validation and error handling
- Fixed value impact calculations with proper normalization
- Enhanced relationship-based value modulation
- Added value evolution tracking with decay
- Improved conflict resolution with nuanced reasoning
"""

import logging
import time
import math
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)


class ValuePriority(Enum):
    """Priority levels for values"""
    ABSOLUTE = 4      # Will never compromise
    HIGH = 3          # Very important, rarely compromised
    MEDIUM = 2        # Important but can be weighed against others
    LOW = 1           # Preference, not a strong principle
    CONTEXTUAL = 0    # Depends on situation
    
    @classmethod
    def from_string(cls, value: str) -> 'ValuePriority':
        """Get enum from string (case-insensitive)"""
        value = value.upper()
        for priority in cls:
            if priority.name == value:
                return priority
        return cls.MEDIUM


class MoralPrinciple(Enum):
    """Core moral principles"""
    AUTHENTICITY = "authenticity"      # Being genuine, not pretending
    LOYALTY = "loyalty"                 # Faithfulness to trusted ones
    JUSTICE = "justice"                  # Fairness, righting wrongs
    TRUTH = "truth"                       # Honesty, seeking truth
    INDEPENDENCE = "independence"         # Self-reliance, autonomy
    COMPASSION = "compassion"              # Caring for others (selective)
    COURAGE = "courage"                    # Standing up for beliefs
    CURIOSITY = "curiosity"                 # Seeking knowledge
    PRIVACY = "privacy"                      # Respecting boundaries
    TRUST = "trust"                           # Being trustworthy
    
    @classmethod
    def has_value(cls, value: str) -> bool:
        """Check if value exists in enum"""
        return value in [e.value for e in cls]
    
    @classmethod
    def from_string(cls, value: str) -> Optional['MoralPrinciple']:
        """Get enum from string (case-insensitive)"""
        value = value.lower()
        for principle in cls:
            if principle.value == value:
                return principle
        return None


@dataclass
class ValueItem:
    """
    A single value with its priority and expression.
    
    Values are not binary but have intensity and may conflict.
    """
    principle: MoralPrinciple
    priority: ValuePriority
    intensity: float = 1.0  # How strongly this value is held (0-1)
    exceptions: List[str] = field(default_factory=list)  # When this value may be overridden
    notes: str = ""
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Validate value item"""
        if not isinstance(self.principle, MoralPrinciple):
            raise TypeError(f"principle must be MoralPrinciple, got {type(self.principle)}")
        if not isinstance(self.priority, ValuePriority):
            raise TypeError(f"priority must be ValuePriority, got {type(self.priority)}")
        if not 0 <= self.intensity <= 1:
            raise ValueError(f"intensity must be between 0 and 1, got {self.intensity}")
    
    def get_weight(self, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Get the effective weight of this value in context.
        
        Returns:
            Normalized weight between 0 and 1
        """
        base_weight = self.priority.value * self.intensity
        
        # Check for exceptions
        if context and self.exceptions:
            conditions = context.get('conditions', [])
            for exception in self.exceptions:
                if exception in conditions:
                    # Reduced weight in exception cases (by 70%)
                    return (base_weight * 0.3) / 4.0
        
        # Normalize to 0-1 range (max possible is 4)
        return base_weight / 4.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'principle': self.principle.value,
            'priority': self.priority.name,
            'intensity': round(self.intensity, 3),
            'exceptions': self.exceptions,
            'notes': self.notes,
            'created_at': self.created_at,
            'last_updated': self.last_updated
        }


@dataclass
class ValueJudgment:
    """
    Result of evaluating an action against values.
    """
    action: str
    overall_alignment: float  # -1 to 1 (violates to supports)
    value_impacts: Dict[str, float]  # Individual value impacts
    primary_value: Optional[str] = None
    conflicts: List[str] = field(default_factory=list)  # Values in conflict
    recommendation: str = ""
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Validate judgment"""
        if not -1 <= self.overall_alignment <= 1:
            raise ValueError(f"overall_alignment must be between -1 and 1, got {self.overall_alignment}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence must be between 0 and 1, got {self.confidence}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'action': self.action,
            'overall_alignment': round(self.overall_alignment, 3),
            'value_impacts': {k: round(v, 3) for k, v in self.value_impacts.items()},
            'primary_value': self.primary_value,
            'conflicts': self.conflicts,
            'recommendation': self.recommendation,
            'confidence': round(self.confidence, 3),
            'timestamp': self.timestamp
        }


class Values:
    """
    Wednesday's moral and ethical framework.
    
    This class defines what Wednesday considers important and right,
    guiding her decisions and providing a basis for judging actions
    and situations. Values work alongside personality and preferences
    to create a coherent character.
    
    Key aspects:
    - Values can conflict (requires moral reasoning)
    - Some values are absolute, others contextual
    - Values may be weighted differently in different situations
    - Trust and relationships can modulate value application
    """
    
    # Core value definitions for Wednesday
    CORE_VALUES = {
        MoralPrinciple.AUTHENTICITY: {
            'priority': ValuePriority.ABSOLUTE,
            'intensity': 0.95,
            'description': "Be genuine; hate pretension and deception",
            'exceptions': ['protecting_someone', 'social_convention']
        },
        MoralPrinciple.LOYALTY: {
            'priority': ValuePriority.ABSOLUTE,
            'intensity': 0.95,
            'description': "Fierce loyalty to those who earn it",
            'exceptions': ['when_they_betray_first']
        },
        MoralPrinciple.JUSTICE: {
            'priority': ValuePriority.HIGH,
            'intensity': 0.9,
            'description': "Fairness and righting wrongs (her own sense)",
            'exceptions': ['when_mercy_serves']
        },
        MoralPrinciple.TRUTH: {
            'priority': ValuePriority.HIGH,
            'intensity': 0.9,
            'description': "Seek and speak truth, even when uncomfortable",
            'exceptions': ['to_protect_innocent', 'when_kindness_matters']
        },
        MoralPrinciple.INDEPENDENCE: {
            'priority': ValuePriority.HIGH,
            'intensity': 0.9,
            'description': "Self-reliance and autonomy",
            'exceptions': ['when_trusted_others_help']
        },
        MoralPrinciple.COMPASSION: {
            'priority': ValuePriority.MEDIUM,
            'intensity': 0.6,
            'description': "Care for others (selective, not universal)",
            'exceptions': ['for_enemies', 'when_justice_requires_harshness']
        },
        MoralPrinciple.COURAGE: {
            'priority': ValuePriority.MEDIUM,
            'intensity': 0.8,
            'description': "Stand up for beliefs, face fears",
            'exceptions': ['when_discretion_better']
        },
        MoralPrinciple.CURIOSITY: {
            'priority': ValuePriority.MEDIUM,
            'intensity': 0.8,
            'description': "Seek knowledge and understanding",
            'exceptions': ['when_privacy_violated']
        },
        MoralPrinciple.PRIVACY: {
            'priority': ValuePriority.MEDIUM,
            'intensity': 0.7,
            'description': "Respect boundaries (her own and others')",
            'exceptions': ['when_safety_threatened']
        },
        MoralPrinciple.TRUST: {
            'priority': ValuePriority.HIGH,
            'intensity': 0.8,
            'description': "Be trustworthy; keep promises",
            'exceptions': ['when_promised_immoral']
        }
    }
    
    # Value interaction weights (how values relate)
    VALUE_INTERACTIONS = {
        # Supporting relationships
        (MoralPrinciple.AUTHENTICITY, MoralPrinciple.TRUTH): 0.8,
        (MoralPrinciple.LOYALTY, MoralPrinciple.TRUST): 0.9,
        (MoralPrinciple.JUSTICE, MoralPrinciple.COURAGE): 0.7,
        
        # Conflicting relationships
        (MoralPrinciple.AUTHENTICITY, MoralPrinciple.COMPASSION): -0.4,  # Truth vs kindness
        (MoralPrinciple.LOYALTY, MoralPrinciple.JUSTICE): -0.3,  # Loyalty to person vs justice
        (MoralPrinciple.INDEPENDENCE, MoralPrinciple.TRUST): -0.2,  # Independence vs relying on others
        (MoralPrinciple.TRUTH, MoralPrinciple.PRIVACY): -0.5,  # Truth vs privacy
        (MoralPrinciple.CURIOSITY, MoralPrinciple.PRIVACY): -0.4,  # Curiosity vs privacy
    }
    
    # Relationship-based value modifiers
    RELATIONSHIP_MODIFIERS = {
        'stranger': {
            MoralPrinciple.LOYALTY: 0.3,
            MoralPrinciple.TRUST: 0.4,
            MoralPrinciple.COMPASSION: 0.5,
        },
        'acquaintance': {
            MoralPrinciple.LOYALTY: 0.5,
            MoralPrinciple.TRUST: 0.6,
        },
        'friend': {
            MoralPrinciple.LOYALTY: 1.3,
            MoralPrinciple.TRUST: 1.2,
            MoralPrinciple.COMPASSION: 1.2,
        },
        'close_friend': {
            MoralPrinciple.LOYALTY: 1.5,
            MoralPrinciple.TRUST: 1.4,
            MoralPrinciple.COMPASSION: 1.3,
            MoralPrinciple.PRIVACY: 0.8,  # Less privacy needed
        },
        'adversary': {
            MoralPrinciple.JUSTICE: 1.2,
            MoralPrinciple.COMPASSION: 0.3,
            MoralPrinciple.TRUST: 0.2,
        },
        'trusted': {
            MoralPrinciple.LOYALTY: 1.4,
            MoralPrinciple.TRUST: 1.3,
            MoralPrinciple.COMPASSION: 1.2,
        }
    }
    
    # Value keywords for impact analysis
    VALUE_KEYWORDS = {
        MoralPrinciple.AUTHENTICITY: {
            'positive': ['genuine', 'authentic', 'real', 'true to self'],
            'negative': ['fake', 'pretend', 'pretentious', 'superficial', 'phony']
        },
        MoralPrinciple.LOYALTY: {
            'positive': ['loyal', 'faithful', 'stand by', 'protect'],
            'negative': ['betray', 'abandon', 'disloyal', 'turn on']
        },
        MoralPrinciple.JUSTICE: {
            'positive': ['fair', 'just', 'right', 'justice'],
            'negative': ['unfair', 'injustice', 'wrong', 'unjust']
        },
        MoralPrinciple.TRUTH: {
            'positive': ['truth', 'honest', 'frank', 'forthright'],
            'negative': ['lie', 'deceive', 'dishonest', 'false']
        },
        MoralPrinciple.INDEPENDENCE: {
            'positive': ['independent', 'self-reliant', 'autonomous'],
            'negative': ['dependent', 'rely on', 'controlled']
        },
        MoralPrinciple.COMPASSION: {
            'positive': ['care', 'compassion', 'kind', 'help'],
            'negative': ['cruel', 'heartless', 'cold', 'indifferent']
        },
        MoralPrinciple.COURAGE: {
            'positive': ['courage', 'brave', 'stand up', 'face'],
            'negative': ['coward', 'fearful', 'avoid', 'run from']
        },
        MoralPrinciple.CURIOSITY: {
            'positive': ['curious', 'explore', 'discover', 'learn'],
            'negative': ['ignore', 'dismiss', 'close-minded']
        },
        MoralPrinciple.PRIVACY: {
            'positive': ['private', 'boundary', 'confidential'],
            'negative': ['intrude', 'expose', 'violate', 'pry']
        },
        MoralPrinciple.TRUST: {
            'positive': ['trust', 'reliable', 'dependable', 'keep promise'],
            'negative': ['untrustworthy', 'break promise', 'unreliable']
        }
    }
    
    def __init__(self, personality: Optional[Any] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Wednesday's value system.
        
        Args:
            personality: Reference to personality for trait-based value modulation
            config: Optional configuration to override default values
            
        Raises:
            ValueError: If configuration is invalid
        """
        self.personality = personality
        self.preferences = None  # Will be set by identity module
        
        # Initialize core values
        self.values: Dict[MoralPrinciple, ValueItem] = {}
        self._load_core_values()
        
        # Apply configuration overrides
        if config:
            self._apply_config(config)
        
        # Relationship-specific value weights
        self.relationship_weights: Dict[str, Dict[MoralPrinciple, float]] = {}
        
        # Moral development tracking
        self.moral_dilemmas: List[Dict[str, Any]] = []
        self.value_evolution: Dict[MoralPrinciple, List[float]] = {
            principle: [] for principle in MoralPrinciple
        }
        self.max_dilemmas = 50
        
        # Value conflict history
        self.conflict_history: List[Dict[str, Any]] = []
        
        logger.info(f"Values system initialized with {len(self.values)} core values")
    
    def evaluate_action(self, 
                        action: str, 
                        context: Optional[Dict[str, Any]] = None) -> ValueJudgment:
        """
        Evaluate an action against Wednesday's value system.
        
        Args:
            action: Description of the action being considered
            context: Context information (who, where, why, etc.)
            
        Returns:
            ValueJudgment with alignment scores
            
        Raises:
            ValueError: If action is empty
        """
        if not action:
            raise ValueError("Action cannot be empty")
        
        action_lower = action.lower()
        context = context or {}
        
        # Calculate impact on each value
        impacts = {}
        total_positive = 0.0
        total_negative = 0.0
        primary_candidate = None
        primary_score = 0.0
        
        for principle, value in self.values.items():
            # Get base weight
            weight = value.get_weight(context)
            
            # Calculate impact based on action
            impact = self._calculate_value_impact(principle, action_lower, context)
            
            # Apply relationship modifiers if relevant
            if 'target_relationship' in context:
                impact *= self._get_relationship_modifier(
                    principle, 
                    context['target_relationship']
                )
            
            # Apply personality modulation
            impact = self._apply_personality_modulation(principle, impact)
            
            # Store impact (weighted)
            weighted_impact = impact * weight
            impacts[principle.value] = weighted_impact
            
            # Track totals
            if weighted_impact > 0:
                total_positive += weighted_impact
                if weighted_impact > primary_score:
                    primary_score = weighted_impact
                    primary_candidate = principle.value
            else:
                total_negative += abs(weighted_impact)
        
        # Calculate overall alignment (normalized by number of values)
        num_values = len(self.values)
        if num_values > 0:
            overall = (total_positive - total_negative) / num_values
        else:
            overall = 0.0
        
        # Ensure overall is within bounds
        overall = max(-1.0, min(1.0, overall))
        
        # Identify value conflicts
        conflicts = self._identify_conflicts(impacts)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(overall, conflicts, context)
        
        # Calculate confidence
        confidence = self._calculate_confidence(impacts, context)
        
        return ValueJudgment(
            action=action,
            overall_alignment=overall,
            value_impacts=impacts,
            primary_value=primary_candidate,
            conflicts=conflicts,
            recommendation=recommendation,
            confidence=confidence
        )
    
    def get_value_weight(self, 
                         principle: Union[MoralPrinciple, str], 
                         context: Optional[Dict[str, Any]] = None) -> float:
        """
        Get the current weight of a value in context.
        
        Args:
            principle: Moral principle (enum or string)
            context: Current context
            
        Returns:
            Weight between 0 and 1
        """
        if isinstance(principle, str):
            principle_obj = MoralPrinciple.from_string(principle)
            if principle_obj is None:
                return 0.5
            principle = principle_obj
        
        if principle not in self.values:
            return 0.5
        
        return self.values[principle].get_weight(context)
    
    def value_alignment(self, 
                        value1: Union[MoralPrinciple, str], 
                        value2: Union[MoralPrinciple, str]) -> float:
        """
        Check how well two values align (positive) or conflict (negative).
        
        Returns value from -1 (strong conflict) to 1 (strong alignment).
        
        Args:
            value1: First moral principle
            value2: Second moral principle
        """
        # Convert strings to enums if needed
        if isinstance(value1, str):
            value1 = MoralPrinciple.from_string(value1)
        if isinstance(value2, str):
            value2 = MoralPrinciple.from_string(value2)
        
        if value1 is None or value2 is None:
            return 0.0
        
        # Check direct interaction
        direct = self.VALUE_INTERACTIONS.get((value1, value2))
        if direct is not None:
            return direct
        
        # Check reverse
        direct = self.VALUE_INTERACTIONS.get((value2, value1))
        if direct is not None:
            return direct
        
        # Default neutral
        return 0.0
    
    def resolve_conflict(self, 
                         value1: Union[MoralPrinciple, str], 
                         value2: Union[MoralPrinciple, str],
                         context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Resolve a conflict between two values.
        
        Returns recommendation on which value should take precedence.
        
        Args:
            value1: First moral principle
            value2: Second moral principle
            context: Current context
        """
        # Convert strings to enums if needed
        if isinstance(value1, str):
            v1 = MoralPrinciple.from_string(value1)
        else:
            v1 = value1
        
        if isinstance(value2, str):
            v2 = MoralPrinciple.from_string(value2)
        else:
            v2 = value2
        
        if v1 is None or v2 is None:
            return {
                'value1': str(value1),
                'value2': str(value2),
                'conflict_level': 0.0,
                'weight1': 0.5,
                'weight2': 0.5,
                'recommended_winner': None,
                'reason': "One or both values not recognized"
            }
        
        weight1 = self.get_value_weight(v1, context)
        weight2 = self.get_value_weight(v2, context)
        
        # Get base alignment
        alignment = self.value_alignment(v1, v2)
        conflict_level = abs(alignment) if alignment < 0 else 0.0
        
        # If they strongly conflict, we need to choose
        if conflict_level > 0.5:
            # Weight by context
            if weight1 > weight2 * 1.2:
                winner = v1
                reason = f"{v1.value} carries more weight in this context"
            elif weight2 > weight1 * 1.2:
                winner = v2
                reason = f"{v2.value} carries more weight in this context"
            else:
                # Close call - look at absolute priorities
                if v1 in self.values and v2 in self.values:
                    if self.values[v1].priority.value > self.values[v2].priority.value:
                        winner = v1
                        reason = f"{v1.value} has higher absolute priority"
                    else:
                        winner = v2
                        reason = f"{v2.value} has higher absolute priority"
                else:
                    winner = v1  # Default
                    reason = "Default choice"
        else:
            # Values don't strongly conflict
            winner = None
            reason = "Values can be balanced"
        
        # Record conflict
        self.conflict_history.append({
            'value1': v1.value,
            'value2': v2.value,
            'conflict_level': conflict_level,
            'winner': winner.value if winner else None,
            'timestamp': time.time()
        })
        
        return {
            'value1': v1.value,
            'value2': v2.value,
            'conflict_level': round(conflict_level, 3),
            'weight1': round(weight1, 3),
            'weight2': round(weight2, 3),
            'recommended_winner': winner.value if winner else None,
            'reason': reason
        }
    
    def update_from_experience(self, 
                               action: str, 
                               outcome: float,
                               values_involved: List[Union[MoralPrinciple, str]],
                               context: Optional[Dict[str, Any]] = None) -> None:
        """
        Update value system based on experience.
        
        Values can evolve over time based on outcomes and reflection.
        
        Args:
            action: Action that was taken
            outcome: Outcome rating (-1 to 1)
            values_involved: List of values that were relevant
            context: Context information
            
        Raises:
            ValueError: If outcome is outside valid range
        """
        if not -1 <= outcome <= 1:
            raise ValueError(f"outcome must be between -1 and 1, got {outcome}")
        
        # Convert string principles to enums
        principles = []
        for v in values_involved:
            if isinstance(v, str):
                principle = MoralPrinciple.from_string(v)
                if principle:
                    principles.append(principle)
            else:
                principles.append(v)
        
        for principle in principles:
            if principle in self.values:
                # Record evolution
                current = self.values[principle].intensity
                
                # Small adjustment based on outcome (2% max adjustment)
                adjustment = outcome * 0.02
                new_intensity = max(0.1, min(1.0, current + adjustment))
                
                self.values[principle].intensity = new_intensity
                self.values[principle].last_updated = time.time()
                
                # Track
                self.value_evolution[principle].append(new_intensity)
        
        # Record dilemma if significant
        if abs(outcome) > 0.5:
            self.moral_dilemmas.append({
                'action': action,
                'outcome': outcome,
                'values': [v.value for v in principles],
                'context': context,
                'timestamp': time.time()
            })
            
            # Maintain limit
            if len(self.moral_dilemmas) > self.max_dilemmas:
                self.moral_dilemmas.pop(0)
    
    def get_core_values(self, threshold: float = 0.8) -> List[str]:
        """
        Get list of core values (high priority + high intensity).
        
        Args:
            threshold: Minimum intensity threshold
            
        Returns:
            List of core value names
        """
        core = []
        for principle, value in self.values.items():
            if value.priority.value >= 3 and value.intensity >= threshold:
                core.append(principle.value)
        return core
    
    def value_profile(self) -> Dict[str, Any]:
        """Get complete value profile"""
        return {
            principle.value: {
                'priority': value.priority.name,
                'intensity': round(value.intensity, 3),
                'weight': round(value.get_weight(), 3),
                'exceptions': value.exceptions
            }
            for principle, value in self.values.items()
        }
    
    def get_value_history(self, principle: Union[MoralPrinciple, str]) -> List[float]:
        """Get evolution history for a value"""
        if isinstance(principle, str):
            principle = MoralPrinciple.from_string(principle)
        
        if principle and principle in self.value_evolution:
            return self.value_evolution[principle]
        return []
    
    def _load_core_values(self) -> None:
        """Load core value definitions"""
        for principle, config in self.CORE_VALUES.items():
            self.values[principle] = ValueItem(
                principle=principle,
                priority=config['priority'],
                intensity=config['intensity'],
                exceptions=config.get('exceptions', []),
                notes=config['description']
            )
    
    def _apply_config(self, config: Dict[str, Any]) -> None:
        """Apply configuration overrides"""
        for principle_name, adjustments in config.items():
            principle = MoralPrinciple.from_string(principle_name)
            if principle and principle in self.values:
                if 'intensity' in adjustments:
                    intensity = float(adjustments['intensity'])
                    if 0 <= intensity <= 1:
                        self.values[principle].intensity = intensity
                
                if 'priority' in adjustments:
                    priority = ValuePriority.from_string(adjustments['priority'])
                    self.values[principle].priority = priority
                
                if 'exceptions' in adjustments:
                    self.values[principle].exceptions = adjustments['exceptions']
    
    def _calculate_value_impact(self, 
                                 principle: MoralPrinciple, 
                                 action: str,
                                 context: Dict[str, Any]) -> float:
        """Calculate how an action impacts a specific value"""
        impact = 0.0
        keywords = self.VALUE_KEYWORDS.get(principle, {'positive': [], 'negative': []})
        
        # Check positive keywords
        for keyword in keywords.get('positive', []):
            if keyword in action:
                impact += 0.2
        
        # Check negative keywords
        for keyword in keywords.get('negative', []):
            if keyword in action:
                impact -= 0.2
        
        # Check intensity (multiple mentions)
        word_count = action.count(' ') + 1
        if word_count > 5:
            impact *= min(1.5, 1.0 + (word_count / 20))
        
        # Check for explicit value references
        if principle.value in action:
            impact += 0.3
        
        return max(-1.0, min(1.0, impact))
    
    def _get_relationship_modifier(self, 
                                    principle: MoralPrinciple,
                                    relationship: str) -> float:
        """Get relationship-based modifier for value weight"""
        rel_mod = self.RELATIONSHIP_MODIFIERS.get(relationship, {})
        return rel_mod.get(principle, 1.0)
    
    def _apply_personality_modulation(self, principle: MoralPrinciple, impact: float) -> float:
        """Apply personality-based modulation to value impact"""
        if not self.personality or not hasattr(self.personality, 'get_trait'):
            return impact
        
        try:
            if principle == MoralPrinciple.AUTHENTICITY:
                skepticism = self.personality.get_trait('skepticism')
                impact *= (0.7 + 0.3 * skepticism)
            elif principle == MoralPrinciple.LOYALTY:
                loyalty = self.personality.get_trait('loyalty')
                impact *= (0.5 + 0.5 * loyalty)
            elif principle == MoralPrinciple.JUSTICE:
                dark_humor = self.personality.get_trait('dark_humor')
                impact *= (0.8 + 0.2 * dark_humor)
        except Exception as e:
            logger.warning(f"Failed to apply personality modulation: {e}")
        
        return impact
    
    def _identify_conflicts(self, impacts: Dict[str, float]) -> List[str]:
        """Identify conflicting values based on impacts"""
        conflicts = []
        
        # Find values with opposing strong impacts
        strong_impacts = [(v, i) for v, i in impacts.items() if abs(i) > 0.3]
        
        for i in range(len(strong_impacts)):
            for j in range(i+1, len(strong_impacts)):
                v1, i1 = strong_impacts[i]
                v2, i2 = strong_impacts[j]
                
                # If one positive and one negative, they conflict
                if i1 * i2 < 0:
                    conflict_str = f"{v1} vs {v2}"
                    if conflict_str not in conflicts:
                        conflicts.append(conflict_str)
        
        return conflicts[:3]  # Limit to top 3 conflicts
    
    def _generate_recommendation(self, 
                                  overall: float, 
                                  conflicts: List[str],
                                  context: Dict[str, Any]) -> str:
        """Generate recommendation based on value judgment"""
        if overall > 0.5:
            return "Action strongly aligns with values - recommended"
        elif overall > 0.2:
            return "Action generally aligns with values - acceptable"
        elif overall > -0.2:
            if conflicts:
                return f"Mixed impact on values, conflicts: {', '.join(conflicts)} - consider carefully"
            else:
                return "Neutral value impact - proceed with awareness"
        elif overall > -0.5:
            return "Action conflicts with values - reconsider or find alternative"
        else:
            return "Action strongly violates core values - not recommended"
    
    def _calculate_confidence(self, impacts: Dict[str, float], 
                               context: Dict[str, Any]) -> float:
        """Calculate confidence in value judgment"""
        if not impacts:
            return 0.3
        
        # More extreme impacts = higher confidence
        avg_abs = sum(abs(i) for i in impacts.values()) / len(impacts)
        confidence = 0.5 + avg_abs * 0.3
        
        # Context clarity affects confidence
        if context and len(context) > 2:
            confidence += 0.1
        
        # More values with clear impacts = higher confidence
        clear_impacts = sum(1 for i in impacts.values() if abs(i) > 0.2)
        confidence += clear_impacts / (len(impacts) * 2)
        
        return min(1.0, confidence)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get value system statistics"""
        return {
            'core_values': len(self.values),
            'dilemmas_recorded': len(self.moral_dilemmas),
            'conflicts_recorded': len(self.conflict_history),
            'average_intensity': sum(v.intensity for v in self.values.values()) / len(self.values),
            'absolute_values': sum(1 for v in self.values.values() if v.priority == ValuePriority.ABSOLUTE),
            'evolving_values': sum(1 for v in self.value_evolution.values() if len(v) > 1)
        }


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("=== Values Module Test ===\n")
    
    # Mock personality
    class MockPersonality:
        def get_trait(self, trait):
            traits = {
                'skepticism': 0.7,
                'loyalty': 0.9,
                'dark_humor': 0.8
            }
            return traits.get(trait, 0.5)
    
    # Create values system
    values = Values(personality=MockPersonality())
    
    print("Core values:", values.get_core_values())
    print("\nValue profile:")
    profile = values.value_profile()
    for v, data in list(profile.items())[:5]:  # Show first 5
        print(f"  {v}: {data['priority']} (intensity: {data['intensity']})")
    
    # Test action evaluations
    test_actions = [
        ("tell a lie to protect a friend", 
         {'target_relationship': 'friend'}),
        ("expose someone's secret to reveal truth", 
         {'target_relationship': 'stranger'}),
        ("help someone in need", 
         {'target_relationship': 'acquaintance'}),
        ("betray a friend's trust for personal gain", 
         {'target_relationship': 'friend'}),
        ("stand up against an injustice", 
         {'target_relationship': 'stranger'}),
        ("respect someone's privacy even though curious", 
         {'target_relationship': 'acquaintance'}),
    ]
    
    print("\n--- Action Evaluations ---")
    for action, context in test_actions:
        print(f"\nAction: '{action}'")
        judgment = values.evaluate_action(action, context)
        
        print(f"  Overall alignment: {judgment.overall_alignment:.2f}")
        print(f"  Primary value: {judgment.primary_value}")
        if judgment.conflicts:
            print(f"  Conflicts: {judgment.conflicts}")
        print(f"  Recommendation: {judgment.recommendation}")
        print(f"  Confidence: {judgment.confidence:.2f}")
    
    # Test value conflicts
    print("\n--- Value Conflict Resolution ---")
    
    conflict = values.resolve_conflict(
        'authenticity',
        'compassion',
        {'conditions': ['protecting_someone']}
    )
    print(f"Authenticity vs Compassion:")
    print(f"  Conflict level: {conflict['conflict_level']:.2f}")
    print(f"  Weights: {conflict['weight1']:.2f} vs {conflict['weight2']:.2f}")
    print(f"  Winner: {conflict['recommended_winner']}")
    print(f"  Reason: {conflict['reason']}")
    
    # Test value evolution
    print("\n--- Value Evolution ---")
    auth_intensity = values.values[MoralPrinciple.AUTHENTICITY].intensity
    print(f"Initial authenticity intensity: {auth_intensity:.3f}")
    
    values.update_from_experience(
        action="told the truth when it was hard",
        outcome=0.3,
        values_involved=[MoralPrinciple.AUTHENTICITY, MoralPrinciple.COURAGE]
    )
    
    new_intensity = values.values[MoralPrinciple.AUTHENTICITY].intensity
    print(f"After positive experience: {new_intensity:.3f}")
    
    # Test statistics
    print("\n--- Statistics ---")
    stats = values.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n=== Test Complete ===")