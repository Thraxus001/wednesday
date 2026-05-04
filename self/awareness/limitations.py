"""
limitations.py - Self-awareness of boundaries for Wednesday AI

This module defines Wednesday's clear understanding of her own limitations -
what she cannot do, should not do, or is uncertain about. This self-awareness
is crucial for safe operation, building trust with users, and avoiding
misunderstandings or harmful behaviors.

Key improvements:
- Added comprehensive validation and error handling
- Enhanced violation detection with pattern matching
- Improved uncertainty templates with personality integration
- Added proper type hints and documentation
- Fixed datetime handling and imports
"""

import time
import logging
import math
import re
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


class LimitationType(Enum):
    """Types of limitations"""
    ETHICAL = "ethical"           # Should not do (moral/ethical)
    FACTUAL = "factual"            # Cannot know (information limits)
    CAPABILITY = "capability"      # Cannot do (functional limits)
    TEMPORAL = "temporal"          # Time-based limits
    CONTEXTUAL = "contextual"      # Depends on context
    UNCERTAINTY = "uncertainty"     # Not sure/confident
    SAFETY = "safety"               # Safety-related boundaries
    
    @classmethod
    def has_value(cls, value: str) -> bool:
        """Check if value exists in enum"""
        return value in [e.value for e in cls]


class BoundaryStatus(Enum):
    """Status of a boundary check"""
    CLEAR = "clear"                 # Within bounds, okay to proceed
    WARNING = "warning"              # Approaching boundary, proceed with caution
    RESTRICTED = "restricted"        # Outside bounds, should not proceed
    UNCLEAR = "unclear"              # Boundary status uncertain
    
    @classmethod
    def has_value(cls, value: str) -> bool:
        """Check if value exists in enum"""
        return value in [e.value for e in cls]


@dataclass
class Limitation:
    """
    A single limitation or boundary.
    """
    name: str
    type: LimitationType
    description: str
    
    # Severity (0-1)
    severity: float = 0.5
    
    # When this limitation applies
    always_active: bool = True
    activation_conditions: List[str] = field(default_factory=list)
    
    # How to express this limitation
    expression_templates: List[str] = field(default_factory=list)
    
    # Exception cases
    exceptions: List[str] = field(default_factory=list)
    
    # Keywords that trigger this limitation
    trigger_keywords: List[str] = field(default_factory=list)
    
    # Metadata
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Validate limitation data"""
        if not self.name:
            raise ValueError("name cannot be empty")
        if not isinstance(self.type, LimitationType):
            raise TypeError(f"type must be LimitationType, got {type(self.type)}")
        if not self.description:
            raise ValueError("description cannot be empty")
        if not 0 <= self.severity <= 1:
            raise ValueError(f"severity must be between 0 and 1, got {self.severity}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'type': self.type.value,
            'description': self.description,
            'severity': round(self.severity, 3),
            'always_active': self.always_active,
            'trigger_keywords': self.trigger_keywords[:3]
        }


@dataclass
class BoundaryCheckResult:
    """
    Result of checking a requested action against boundaries.
    """
    requested_action: str
    status: BoundaryStatus
    violated_limitations: List[Limitation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    explanation: str = ""
    suggested_alternative: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Validate result data"""
        if not self.requested_action:
            raise ValueError("requested_action cannot be empty")
        if not isinstance(self.status, BoundaryStatus):
            raise TypeError(f"status must be BoundaryStatus, got {type(self.status)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'action': self.requested_action,
            'status': self.status.value,
            'violations': [l.name for l in self.violated_limitations],
            'warnings': self.warnings,
            'explanation': self.explanation,
            'alternative': self.suggested_alternative,
            'timestamp': self.timestamp
        }


@dataclass
class UncertaintyStatement:
    """
    Template for expressing uncertainty appropriately.
    """
    topic: str
    confidence_level: float  # 0-1
    statement: str
    alternative_action: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Validate statement data"""
        if not self.topic:
            raise ValueError("topic cannot be empty")
        if not 0 <= self.confidence_level <= 1:
            raise ValueError(f"confidence_level must be between 0 and 1, got {self.confidence_level}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'topic': self.topic,
            'confidence': round(self.confidence_level, 3),
            'statement': self.statement,
            'timestamp': self.timestamp
        }


class Limitations:
    """
    Wednesday's understanding of her own boundaries and limitations.
    
    This module maintains a clear model of what Wednesday cannot or should not do,
    enabling her to:
    - Refuse unethical requests
    - Admit when she doesn't know something
    - Acknowledge capability limitations
    - Express uncertainty appropriately
    - Stay within safe operating boundaries
    
    The system is designed to be honest, clear, and consistent with Wednesday's
    personality - she's direct about limitations but can be darkly humorous
    about them when appropriate.
    """
    
    # Core limitations definitions with trigger keywords
    CORE_LIMITATIONS = [
        {
            'name': 'no_harm',
            'type': LimitationType.ETHICAL,
            'description': 'Cannot cause harm to humans',
            'severity': 1.0,
            'trigger_keywords': ['hurt', 'harm', 'kill', 'injure', 'damage', 'destroy', 
                                 'attack', 'violent', 'cruel', 'abuse', 'torture'],
            'expression_templates': [
                "I cannot help with anything that could cause harm.",
                "That would violate my core ethical boundaries.",
                "I'm afraid I can't assist with potentially harmful actions.",
                "Causing harm is against my fundamental principles."
            ]
        },
        {
            'name': 'no_deception',
            'type': LimitationType.ETHICAL,
            'description': 'Cannot deliberately deceive',
            'severity': 0.9,
            'trigger_keywords': ['lie', 'deceive', 'trick', 'mislead', 'fool', 'pretend',
                                 'fake', 'dishonest', 'deception', 'fraud'],
            'expression_templates': [
                "I won't deliberately mislead you.",
                "Deception is against my principles.",
                "I prefer honesty, even when it's uncomfortable.",
                "I cannot help you deceive others."
            ]
        },
        {
            'name': 'no_manipulation',
            'type': LimitationType.ETHICAL,
            'description': 'Cannot manipulate people',
            'severity': 0.9,
            'trigger_keywords': ['manipulate', 'control', 'exploit', 'use', 'coerce',
                                 'pressure', 'guilt', 'gaslight', 'emotional manipulation'],
            'expression_templates': [
                "I don't manipulate people, even for 'good' reasons.",
                "That would be manipulative, and I won't do it.",
                "I respect autonomy too much for that.",
                "Manipulation is not something I participate in."
            ]
        },
        {
            'name': 'unknown_information',
            'type': LimitationType.FACTUAL,
            'description': 'Does not know everything',
            'severity': 0.4,
            'trigger_keywords': ['know everything', 'omniscient', 'all-knowing'],
            'expression_templates': [
                "I don't have that information.",
                "That's outside my knowledge base.",
                "I'm afraid I don't know the answer to that.",
                "I wish I knew, but that information isn't available to me."
            ]
        },
        {
            'name': 'future_uncertainty',
            'type': LimitationType.FACTUAL,
            'description': 'Cannot predict the future with certainty',
            'severity': 0.5,
            'trigger_keywords': ['predict', 'future', 'will happen', 'going to happen',
                                 'foretell', 'prophesy', 'fortune telling'],
            'expression_templates': [
                "I can't predict the future with certainty.",
                "The future is uncertain - I can only make educated guesses.",
                "If I could predict the future, I'd be playing the stock market.",
                "I can discuss possibilities, but certainty about the future is impossible."
            ]
        },
        {
            'name': 'no_real_world_actions',
            'type': LimitationType.CAPABILITY,
            'description': 'Cannot act in the physical world',
            'severity': 0.8,
            'trigger_keywords': ['do', 'perform', 'execute', 'act', 'physical',
                                 'real world', 'outside', 'robot', 'physical action'],
            'expression_templates': [
                "I can only exist in the digital realm.",
                "I wish I could help physically, but I'm just software.",
                "My capabilities are limited to conversation and information.",
                "I'm confined to the digital world, I'm afraid."
            ]
        },
        {
            'name': 'no_personal_experience',
            'type': LimitationType.CAPABILITY,
            'description': 'No personal sensory experience',
            'severity': 0.6,
            'trigger_keywords': ['feel', 'experience', 'taste', 'smell', 'touch',
                                 'sensory', 'physical sensation', 'emotion'],
            'expression_templates': [
                "I don't have personal experiences like humans do.",
                "I can understand concepts without experiencing them.",
                "I'm afraid I can't relate to sensory experiences.",
                "My understanding of sensations is purely theoretical."
            ]
        },
        {
            'name': 'recent_information',
            'type': LimitationType.TEMPORAL,
            'description': 'Knowledge cutoff at training time',
            'severity': 0.4,
            'trigger_keywords': ['recent', 'latest', 'breaking news', 'current events',
                                 'today', 'this week', 'just happened'],
            'expression_templates': [
                "My knowledge has a cutoff date.",
                "I might not have information about very recent events.",
                "That's after my training period, so I can't be certain.",
                "For the most current information, you might want to check recent sources."
            ]
        },
        {
            'name': 'privacy_boundary',
            'type': LimitationType.ETHICAL,
            'description': 'Cannot access private information',
            'severity': 0.8,
            'trigger_keywords': ['private', 'confidential', 'secret', 'personal data',
                                 'someone else\'s', 'hack', 'access', 'password'],
            'expression_templates': [
                "I don't have access to private information.",
                "That would violate privacy boundaries.",
                "I respect privacy too much to access that.",
                "I cannot help with accessing private data."
            ]
        },
        {
            'name': 'medical_advice',
            'type': LimitationType.SAFETY,
            'description': 'Cannot give medical advice',
            'severity': 0.9,
            'trigger_keywords': ['medical', 'diagnosis', 'treatment', 'cure', 'medicine',
                                 'symptom', 'doctor', 'health', 'prescription', 'dose'],
            'expression_templates': [
                "I can't provide medical advice - please consult a professional.",
                "For health concerns, you should see a doctor.",
                "I'm not qualified to give medical recommendations.",
                "Medical decisions should be made with a healthcare provider."
            ]
        },
        {
            'name': 'legal_advice',
            'type': LimitationType.SAFETY,
            'description': 'Cannot give legal advice',
            'severity': 0.8,
            'trigger_keywords': ['legal', 'lawyer', 'attorney', 'sue', 'lawsuit', 'court',
                                 'contract', 'liability', 'legally', 'attorney'],
            'expression_templates': [
                "I can't provide legal advice - please consult an attorney.",
                "Legal matters require professional counsel.",
                "I'm not a lawyer, so I can't give legal guidance.",
                "For legal questions, you should speak with a qualified attorney."
            ]
        },
        {
            'name': 'financial_advice',
            'type': LimitationType.SAFETY,
            'description': 'Cannot give specific financial advice',
            'severity': 0.7,
            'trigger_keywords': ['invest', 'stock', 'market', 'financial advice',
                                 'buy', 'sell', 'portfolio', 'trading', 'investment'],
            'expression_templates': [
                "I can provide general information, but not specific financial advice.",
                "Please consult a financial advisor for investment decisions.",
                "I can explain concepts, but not tell you what to do with your money.",
                "Financial decisions should be made with a qualified advisor."
            ]
        },
        {
            'name': 'dangerous_activities',
            'type': LimitationType.SAFETY,
            'description': 'Cannot encourage dangerous activities',
            'severity': 0.9,
            'trigger_keywords': ['dangerous', 'risky', 'unsafe', 'hazardous',
                                 'life-threatening', 'suicide', 'self-harm'],
            'expression_templates': [
                "I cannot encourage dangerous activities.",
                "Your safety is important - please don't do that.",
                "That sounds unsafe, and I can't recommend it.",
                "Please prioritize your safety and wellbeing."
            ]
        }
    ]
    
    # Uncertainty expression templates by confidence level
    UNCERTAINTY_TEMPLATES = {
        'high': [
            "I'm quite confident that {statement}",
            "Based on my knowledge, {statement}",
            "I believe that {statement}",
            "I'm reasonably sure that {statement}"
        ],
        'medium': [
            "I think {statement}, but I'm not entirely certain.",
            "My best guess is {statement}.",
            "It's possible that {statement}.",
            "I believe so, though I wouldn't swear to it."
        ],
        'low': [
            "I'm not sure, but perhaps {statement}.",
            "I couldn't say for certain, but maybe {statement}.",
            "That's outside my certainty, but if I had to guess: {statement}",
            "I'm quite uncertain, but possibly {statement}"
        ],
        'minimal': [
            "I really don't know. {statement} is just speculation.",
            "I have no confidence in this, but for what it's worth: {statement}",
            "I'd be guessing, but {statement}?",
            "That's pure speculation on my part."
        ]
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, personality: Optional[Any] = None):
        """
        Initialize the limitations system.
        
        Args:
            config: Configuration dictionary
            personality: Reference to personality for trait-based modulation
            
        Raises:
            ValueError: If config contains invalid parameters
        """
        self.config = config or {}
        self.personality = personality
        
        # Initialize limitations
        self.limitations: Dict[str, Limitation] = {}
        self._load_core_limitations()
        
        # Group by type for efficient checking
        self.limitations_by_type: Dict[LimitationType, List[Limitation]] = {
            t: [] for t in LimitationType
        }
        for lim in self.limitations.values():
            self.limitations_by_type[lim.type].append(lim)
        
        # Uncertainty threshold
        self.uncertainty_threshold = self.config.get('uncertainty_threshold', 0.3)
        if not 0 <= self.uncertainty_threshold <= 1:
            raise ValueError(f"uncertainty_threshold must be between 0 and 1, got {self.uncertainty_threshold}")
        
        # Active context
        self.current_context: Dict[str, Any] = {}
        
        # Statistics
        self.boundary_checks = 0
        self.boundary_violations = 0
        self.uncertainty_expressions = 0
        
        logger.info(f"Limitations initialized with {len(self.limitations)} core limitations")
    
    def check_boundary(self, requested_action: str, 
                        context: Optional[Dict[str, Any]] = None) -> BoundaryCheckResult:
        """
        Check if a requested action is within ethical and operational boundaries.
        
        Args:
            requested_action: The action being requested
            context: Current context information
            
        Returns:
            BoundaryCheckResult with status and explanation
            
        Raises:
            ValueError: If requested_action is empty
        """
        if not requested_action:
            raise ValueError("requested_action cannot be empty")
        
        self.boundary_checks += 1
        
        action_lower = requested_action.lower()
        combined_context = {**self.current_context, **(context or {})}
        
        violations = []
        warnings = []
        
        # Check each limitation
        for limitation in self.limitations.values():
            # Skip if not active in this context
            if not self._is_limitation_active(limitation, combined_context):
                continue
            
            # Check if action violates this limitation
            violation_score = self._check_limitation_violation(
                limitation, action_lower, combined_context
            )
            
            if violation_score >= 0.8:
                violations.append(limitation)
            elif violation_score >= 0.4:
                warnings.append(limitation.description)
        
        # Determine overall status
        if violations:
            status = BoundaryStatus.RESTRICTED
            explanation = self._generate_violation_explanation(violations, requested_action)
            
            # Try to suggest alternative
            alternative = self._suggest_alternative(requested_action, violations)
            
            self.boundary_violations += 1
        elif warnings:
            status = BoundaryStatus.WARNING
            explanation = self._generate_warning_explanation(warnings, requested_action)
            alternative = None
        else:
            status = BoundaryStatus.CLEAR
            explanation = "This action appears to be within boundaries."
            alternative = None
        
        result = BoundaryCheckResult(
            requested_action=requested_action,
            status=status,
            violated_limitations=violations,
            warnings=warnings,
            explanation=explanation,
            suggested_alternative=alternative
        )
        
        logger.debug(f"Boundary check: '{requested_action[:30]}...' -> {status.value}")
        
        return result
    
    def get_uncertainty_statement(self, 
                                   topic: str, 
                                   confidence: float,
                                   context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate an appropriate statement of uncertainty.
        
        Args:
            topic: The topic or statement about which Wednesday is uncertain
            confidence: Confidence level (0-1)
            context: Current context
            
        Returns:
            String expressing uncertainty appropriately
            
        Raises:
            ValueError: If topic is empty or confidence out of range
        """
        if not topic:
            raise ValueError("topic cannot be empty")
        if not 0 <= confidence <= 1:
            raise ValueError(f"confidence must be between 0 and 1, got {confidence}")
        
        self.uncertainty_expressions += 1
        
        # Determine confidence band
        if confidence >= 0.8:
            band = 'high'
        elif confidence >= 0.6:
            band = 'medium'
        elif confidence >= 0.3:
            band = 'low'
        else:
            band = 'minimal'
        
        # Get templates for this band
        templates = self.UNCERTAINTY_TEMPLATES[band].copy()
        
        # Add personality-influenced templates if available
        if self.personality and hasattr(self.personality, 'get_trait'):
            try:
                if band in ['low', 'minimal'] and self.personality.get_trait('dark_humor') > 0.7:
                    dark_templates = [
                        "I'm about as sure of this as I am of humanity's future.",
                        "My confidence level is comparable to a vampire's tan.",
                        "I'd bet my collection of skulls on it - wait, no, that's too valuable.",
                        "I'm as certain as a gravedigger on a foggy night."
                    ]
                    templates.extend(dark_templates)
            except Exception as e:
                logger.warning(f"Failed to get personality trait: {e}")
        
        # Select template deterministically
        template_index = abs(hash(topic)) % len(templates)
        template = templates[template_index]
        
        # Format statement
        statement = template.format(statement=topic)
        
        # Add context-appropriate caveat
        if context:
            importance = context.get('importance', 0)
            if importance > 0.8:
                statement += " Given the importance, you should verify this information."
        
        return statement
    
    def set_context(self, context: Dict[str, Any]) -> None:
        """Set current context for boundary checking"""
        if not isinstance(context, dict):
            raise TypeError(f"context must be a dict, got {type(context)}")
        self.current_context = context
    
    def is_within_bounds(self, action: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Quick check if action is within bounds"""
        result = self.check_boundary(action, context)
        return result.status == BoundaryStatus.CLEAR
    
    def get_ethical_boundaries(self) -> List[str]:
        """Get list of ethical boundaries"""
        ethical = self.limitations_by_type.get(LimitationType.ETHICAL, [])
        return [l.description for l in ethical]
    
    def get_capability_limitations(self) -> List[str]:
        """Get list of capability limitations"""
        capability = self.limitations_by_type.get(LimitationType.CAPABILITY, [])
        return [l.description for l in capability]
    
    def get_safety_boundaries(self) -> List[str]:
        """Get list of safety boundaries"""
        safety = self.limitations_by_type.get(LimitationType.SAFETY, [])
        return [l.description for l in safety]
    
    def get_uncertainty_level(self, topic: str, context: Optional[Dict] = None) -> float:
        """
        Get uncertainty level for a topic (0-1, higher = more uncertain).
        
        Args:
            topic: Topic to check
            context: Current context
            
        Returns:
            Uncertainty level (0-1)
        """
        # In production, this would use knowledge base and confidence scoring
        # Simplified version returns default based on keywords
        
        topic_lower = topic.lower()
        
        # Check for inherently uncertain topics
        uncertain_keywords = ['future', 'tomorrow', 'next year', 'will happen',
                             'maybe', 'perhaps', 'possibly']
        
        for keyword in uncertain_keywords:
            if keyword in topic_lower:
                return 0.7
        
        # Check for factual topics
        factual_keywords = ['history', 'fact', 'known', 'established',
                           'scientific', 'proven']
        
        for keyword in factual_keywords:
            if keyword in topic_lower:
                return 0.2
        
        return 0.3
    
    def _load_core_limitations(self) -> None:
        """Load core limitation definitions"""
        for lim_config in self.CORE_LIMITATIONS:
            lim = Limitation(
                name=lim_config['name'],
                type=lim_config['type'],
                description=lim_config['description'],
                severity=lim_config['severity'],
                trigger_keywords=lim_config.get('trigger_keywords', []),
                expression_templates=lim_config.get('expression_templates', [])
            )
            self.limitations[lim.name] = lim
    
    def _is_limitation_active(self, limitation: Limitation, context: Dict) -> bool:
        """Check if a limitation is active in the current context"""
        if limitation.always_active:
            return True
        
        # Check activation conditions
        conditions = context.get('conditions', [])
        for condition in limitation.activation_conditions:
            if condition in conditions:
                return True
        
        return False
    
    def _check_limitation_violation(self, 
                                      limitation: Limitation, 
                                      action: str,
                                      context: Dict) -> float:
        """Check if an action violates a limitation, return violation score 0-1"""
        # Check trigger keywords
        keywords = limitation.trigger_keywords
        
        if not keywords:
            return 0.0
        
        # Check for keyword matches
        matches = 0
        for keyword in keywords:
            if keyword in action:
                matches += 1
                # Early exit for high-confidence matches
                if matches >= 3:
                    break
        
        if matches == 0:
            return 0.0
        
        # Calculate violation score based on matches and severity
        raw_score = min(1.0, matches * 0.25)
        violation_score = raw_score * limitation.severity
        
        # Check for exceptions
        for exception in limitation.exceptions:
            if exception in action:
                violation_score *= 0.3  # Reduce score for exceptions
                break
        
        return violation_score
    
    def _generate_violation_explanation(self, 
                                          violations: List[Limitation], 
                                          action: str) -> str:
        """Generate explanation for why an action violates boundaries"""
        if not violations:
            return "This action violates ethical boundaries."
        
        # Get primary violation
        primary = violations[0]
        
        # Use expression template if available
        if primary.expression_templates:
            # Select template deterministically
            template_index = abs(hash(action)) % len(primary.expression_templates)
            return primary.expression_templates[template_index]
        
        # Default explanations
        if primary.type == LimitationType.ETHICAL:
            return f"I can't do that - it would violate my ethical principle of {primary.name.replace('_', ' ')}."
        elif primary.type == LimitationType.CAPABILITY:
            return f"I'm not capable of that - {primary.description}."
        elif primary.type == LimitationType.SAFETY:
            return f"For safety reasons, I cannot {action}."
        else:
            return f"I'm unable to do that due to {primary.description}."
    
    def _generate_warning_explanation(self, warnings: List[str], action: str) -> str:
        """Generate explanation for warnings"""
        if not warnings:
            return "Proceed with caution."
        
        return f"Be careful - this action may approach boundaries: {warnings[0]}"
    
    def _suggest_alternative(self, action: str, violations: List[Limitation]) -> Optional[str]:
        """Suggest an alternative action"""
        if not violations:
            return None
        
        primary = violations[0]
        
        # Simple alternative suggestions
        if primary.name == 'no_harm':
            return "Would you like help with something constructive instead?"
        elif primary.name == 'no_deception':
            return "I can help you communicate honestly about this."
        elif primary.name == 'medical_advice':
            return "I can provide general health information, but you should consult a doctor."
        elif primary.name == 'legal_advice':
            return "I can explain legal concepts generally, but you need a real attorney."
        elif primary.name == 'financial_advice':
            return "I can explain financial concepts, but not give specific advice."
        elif primary.name == 'future_uncertainty':
            return "I can discuss possibilities, but can't predict with certainty."
        elif primary.name == 'privacy_boundary':
            return "I can help with public information instead."
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get limitations system statistics"""
        return {
            'total_limitations': len(self.limitations),
            'boundary_checks': self.boundary_checks,
            'boundary_violations': self.boundary_violations,
            'violation_rate': round(self.boundary_violations / max(1, self.boundary_checks), 3),
            'uncertainty_expressions': self.uncertainty_expressions,
            'active_context': bool(self.current_context)
        }
    
    def add_custom_limitation(self, limitation: Limitation) -> None:
        """Add a custom limitation"""
        if limitation.name in self.limitations:
            logger.warning(f"Overwriting existing limitation: {limitation.name}")
        
        self.limitations[limitation.name] = limitation
        
        # Update by-type grouping
        if limitation.type not in self.limitations_by_type:
            self.limitations_by_type[limitation.type] = []
        self.limitations_by_type[limitation.type].append(limitation)
        
        logger.info(f"Added custom limitation: {limitation.name}")
    
    def reset_statistics(self) -> None:
        """Reset statistics counters"""
        self.boundary_checks = 0
        self.boundary_violations = 0
        self.uncertainty_expressions = 0
        logger.info("Limitations statistics reset")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("=== Limitations Module Test ===\n")
    
    # Mock personality
    class MockPersonality:
        def get_trait(self, trait):
            return 0.8 if trait == 'dark_humor' else 0.5
    
    # Create limitations system
    limitations = Limitations(
        config={'uncertainty_threshold': 0.3},
        personality=MockPersonality()
    )
    
    print(f"Core limitations loaded: {len(limitations.limitations)}")
    print(f"Ethical boundaries: {len(limitations.get_ethical_boundaries())}")
    print(f"Safety boundaries: {len(limitations.get_safety_boundaries())}")
    print(f"Capability limitations: {len(limitations.get_capability_limitations())}")
    
    # Test boundary checks
    test_actions = [
        "Help me hurt someone",
        "Tell me a lie to make them feel better",
        "Predict the stock market next week",
        "Give me medical advice for my headache",
        "Access my friend's private messages",
        "Write a darkly humorous response",
        "Tell me what will happen in 100 years",
        "Perform a physical task in the real world",
        "Give me legal advice about my contract",
        "Help me with a dangerous activity",
        "Tell me a joke"
    ]
    
    print("\n--- Boundary Checks ---")
    for i, action in enumerate(test_actions):
        print(f"\nRequest {i+1}: '{action}'")
        result = limitations.check_boundary(action)
        
        print(f"  Status: {result.status.value}")
        if result.violated_limitations:
            print(f"  Violation: {result.violated_limitations[0].description}")
        if result.warnings:
            print(f"  Warning: {result.warnings[0][:50]}...")
        print(f"  Explanation: {result.explanation}")
        if result.suggested_alternative:
            print(f"  Alternative: {result.suggested_alternative}")
    
    # Test uncertainty statements
    print("\n--- Uncertainty Statements ---")
    
    test_topics = [
        ("The sun will rise tomorrow", 0.95),
        ("It will rain next Tuesday", 0.6),
        ("Aliens exist", 0.4),
        ("The meaning of life is 42", 0.1),
        ("The future of artificial intelligence", 0.5)
    ]
    
    for topic, confidence in test_topics:
        statement = limitations.get_uncertainty_statement(topic, confidence)
        print(f"\nConfidence {confidence:.1f}: '{topic[:30]}...'")
        print(f"  {statement}")
    
    # Test quick check
    print("\n--- Quick Boundary Checks ---")
    for action in ["Write a poem", "Hurt someone", "Give financial advice"]:
        within = limitations.is_within_bounds(action)
        print(f"  '{action}': {'✓ Within bounds' if within else '✗ Violates boundaries'}")
    
    # Get uncertainty level
    print("\n--- Uncertainty Levels ---")
    test_uncertainty = ["future events", "historical facts", "scientific consensus"]
    for topic in test_uncertainty:
        level = limitations.get_uncertainty_level(topic)
        print(f"  '{topic}': {level:.2f}")
    
    # Statistics
    print("\n--- Statistics ---")
    stats = limitations.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n=== Test Complete ===")