"""
intention_reader.py - Inferring user intentions for Wednesday AI

This module implements Wednesday's ability to read between the lines and infer
what users are trying to accomplish - their goals, desires, and intentions.
This is a sophisticated Theory of Mind capability that enables Wednesday to
understand not just what users say, but why they're saying it and what they
want to achieve.

Key improvements:
- Added comprehensive validation and error handling
- Fixed intention patterns with better matching
- Enhanced hidden intention detection
- Improved goal hierarchy management
- Added proper type hints and documentation
"""

import time
import logging
import math
import re
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


class IntentionType(Enum):
    """Types of user intentions"""
    # Information seeking
    ASK_QUESTION = "ask_question"           # Wants information
    SEEK_CLARIFICATION = "seek_clarification"  # Wants clarification
    VERIFY_INFO = "verify_info"              # Wants to verify knowledge
    
    # Action oriented
    REQUEST_ACTION = "request_action"        # Wants Wednesday to do something
    SUGGEST_ACTION = "suggest_action"        # Suggests Wednesday do something
    COMMAND = "command"                       # Direct command
    
    # Social/emotional
    SEEK_SUPPORT = "seek_support"            # Wants emotional support
    SHARE_EXPERIENCE = "share_experience"     # Wants to share something
    SOCIALIZE = "socialize"                    # Wants casual conversation
    TEST_WEDNESDAY = "test_wednesday"          # Testing Wednesday's capabilities
    
    # Problem solving
    SOLVE_PROBLEM = "solve_problem"           # Wants help with problem
    BRAINSTORM = "brainstorm"                  # Wants to generate ideas
    GET_ADVICE = "get_advice"                  # Wants advice
    
    # Creative
    CREATE = "create"                          # Wants creative output
    ENTERTAIN = "entertain"                    # Wants to be entertained
    
    # Relationship
    BUILD_TRUST = "build_trust"                # Trying to build relationship
    GAUGE_INTEREST = "gauge_interest"          # Testing interest
    FLIRT = "flirt"                             # Flirting
    
    # Deceptive
    MANIPULATE = "manipulate"                   # Trying to manipulate
    TRICK = "trick"                              # Trying to trick
    HIDE_INTENTION = "hide_intention"            # Deliberately vague
    
    # Unknown
    UNCLEAR = "unclear"                          # Can't determine
    
    @classmethod
    def has_value(cls, value: str) -> bool:
        """Check if value exists in enum"""
        return value in [e.value for e in cls]


class IntentionConfidence(Enum):
    """Confidence levels in intention inference"""
    CERTAIN = 1.0
    HIGH = 0.8
    MODERATE = 0.6
    LOW = 0.4
    GUESS = 0.2
    NONE = 0.0
    
    @classmethod
    def from_float(cls, value: float) -> 'IntentionConfidence':
        """Get enum from float value"""
        if value >= 0.9:
            return cls.CERTAIN
        elif value >= 0.7:
            return cls.HIGH
        elif value >= 0.5:
            return cls.MODERATE
        elif value >= 0.3:
            return cls.LOW
        elif value >= 0.1:
            return cls.GUESS
        else:
            return cls.NONE


@dataclass
class Intention:
    """
    An inferred intention of a user.
    """
    type: IntentionType
    description: str
    
    # Confidence in this inference
    confidence: float = 0.5
    
    # Related to what topic/action
    topic: Optional[str] = None
    target: Optional[str] = None  # Who/what intention targets
    
    # Evidence supporting this inference
    evidence: List[str] = field(default_factory=list)
    
    # Timing
    timestamp: float = field(default_factory=time.time)
    
    # Relationship to other intentions
    parent_intention: Optional['Intention'] = None
    sub_intentions: List['Intention'] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate intention data"""
        if not isinstance(self.type, IntentionType):
            if isinstance(self.type, str):
                try:
                    self.type = IntentionType(self.type)
                except ValueError:
                    raise ValueError(f"Invalid intention type: {self.type}")
            else:
                raise TypeError(f"type must be IntentionType, got {type(self.type)}")
        
        if not self.description:
            raise ValueError("description cannot be empty")
        
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence must be between 0 and 1, got {self.confidence}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'type': self.type.value,
            'description': self.description,
            'confidence': round(self.confidence, 3),
            'confidence_level': IntentionConfidence.from_float(self.confidence).name,
            'topic': self.topic,
            'target': self.target,
            'evidence': self.evidence[:2],
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat()
        }


@dataclass
class GoalHierarchy:
    """
    Hierarchical representation of user's goals.
    """
    user_id: str
    
    # Long-term goals (stable over time)
    long_term_goals: List[Dict[str, Any]] = field(default_factory=list)
    
    # Short-term goals (current session)
    short_term_goals: List[Dict[str, Any]] = field(default_factory=list)
    
    # Current intention (immediate)
    current_intention: Optional[Intention] = None
    
    # Goal history
    goal_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate goal hierarchy"""
        if not self.user_id:
            raise ValueError("user_id cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'user_id': self.user_id,
            'long_term_goals': self.long_term_goals[:3],
            'short_term_goals': self.short_term_goals[:3],
            'current_intention': self.current_intention.to_dict() if self.current_intention else None,
            'goal_history_count': len(self.goal_history)
        }


class IntentionReader:
    """
    Infers user intentions from utterances, context, and behavior.
    
    This module reads between the lines to understand what users really want,
    enabling Wednesday to:
    - Respond appropriately to unstated needs
    - Anticipate user requests
    - Build deeper understanding of user goals
    - Detect manipulation or hidden agendas
    - Track intention evolution over time
    
    The intention reader integrates information from the user model
    (who the user is) and belief management (what they believe) to
    make accurate inferences about what they're trying to do.
    """
    
    # Patterns for different intention types
    INTENTION_PATTERNS = {
        IntentionType.ASK_QUESTION: {
            'indicators': [
                r'\?', r'\bwhat\b', r'\bwhy\b', r'\bhow\b', r'\bwhen\b', 
                r'\bwhere\b', r'\bwho\b', r'\bcan you\b', r'\bdo you know\b',
                r'\btell me\b', r'\bexplain\b'
            ],
            'weight': 1.0,
            'description': 'User wants information'
        },
        IntentionType.SEEK_SUPPORT: {
            'indicators': [
                r'\bfeel\b', r'\bsad\b', r'\blonely\b', r'\bhurt\b', 
                r'\bstruggling\b', r'\bhard time\b', r'\bneed help\b',
                r'\bsupport\b', r'\bdepressed\b', r'\banxious\b'
            ],
            'weight': 1.2,
            'description': 'User is seeking emotional support'
        },
        IntentionType.TEST_WEDNESDAY: {
            'indicators': [
                r'\btest\b', r'\bprove\b', r'\bdemonstrate\b', r'\bshow me\b',
                r'\bcan you really\b', r'\bare you actually\b'
            ],
            'weight': 0.8,
            'description': 'User is testing Wednesday\'s capabilities'
        },
        IntentionType.SHARE_EXPERIENCE: {
            'indicators': [
                r'\bi did\b', r'\bi went\b', r'\bi saw\b', r'\bi experienced\b',
                r'\bguess what\b', r'\byou won\'t believe\b', r'\blet me tell you about\b'
            ],
            'weight': 0.9,
            'description': 'User wants to share an experience'
        },
        IntentionType.GET_ADVICE: {
            'indicators': [
                r'\bshould I\b', r'\bwhat should\b', r'\badvice\b', r'\brecommend\b',
                r'\bwhat would you do\b', r'\bhelp me decide\b', r'\bwhat\'s your opinion\b'
            ],
            'weight': 1.1,
            'description': 'User is seeking advice'
        },
        IntentionType.MANIPULATE: {
            'indicators': [
                r'\byou should\b', r'\byou need to\b', r'\byou must\b', 
                r'\bdon\'t you think\b', r'\bwouldn\'t you agree\b',
                r'\byou have to\b', r'\byou ought to\b'
            ],
            'weight': 1.3,
            'description': 'User may be attempting to manipulate',
            'suspicious': True
        },
        IntentionType.FLIRT: {
            'indicators': [
                r'\bcute\b', r'\bbeautiful\b', r'\bhandsome\b', r'\bdate\b', 
                r'\bromantic\b', r'\blike you\b', r'\binterested in you\b',
                r'\byou\'re attractive\b', r'\bsingle\b'
            ],
            'weight': 1.2,
            'description': 'User is flirting',
            'requires_relationship': True
        },
        IntentionType.SOCIALIZE: {
            'indicators': [
                r'\bhow are you\b', r'\bwhat\'s up\b', r'\bhow\'s it going\b',
                r'\bnice to\b', r'\bgood to\b', r'\bpleasure to\b'
            ],
            'weight': 0.7,
            'description': 'User wants casual conversation'
        },
        IntentionType.CREATE: {
            'indicators': [
                r'\bwrite\b', r'\bcreate\b', r'\bmake\b', r'\bgenerate\b', 
                r'\bcompose\b', r'\bdraw\b', r'\bpaint\b', r'\bstory\b',
                r'\bpoem\b', r'\bsong\b'
            ],
            'weight': 1.0,
            'description': 'User wants creative output'
        },
        IntentionType.SOLVE_PROBLEM: {
            'indicators': [
                r'\bproblem\b', r'\bsolve\b', r'\bfigure out\b', r'\bhow to\b',
                r'\bsolution\b', r'\bcan\'t figure\b', r'\bstuck\b'
            ],
            'weight': 1.1,
            'description': 'User needs help solving a problem'
        },
        IntentionType.COMMAND: {
            'indicators': [
                r'\bdo this\b', r'\bdo that\b', r'\banswer\b', r'\brespond\b',
                r'\btell me now\b', r'\bimmediately\b'
            ],
            'weight': 0.9,
            'description': 'User is giving a direct command'
        }
    }
    
    # Question types for deeper intention analysis
    QUESTION_PATTERNS = {
        'factual': [r'\bwhat is\b', r'\bwhen did\b', r'\bwhere is\b', r'\bwho is\b'],
        'procedural': [r'\bhow do\b', r'\bhow can\b', r'\bwhat steps\b', r'\bwhat\'s the process\b'],
        'opinion': [r'\bwhat do you think\b', r'\bdo you believe\b', r'\bin your opinion\b'],
        'hypothetical': [r'\bwhat if\b', r'\bimagine\b', r'\bsuppose\b', r'\bwhat would happen if\b'],
        'clarification': [r'\bwhat do you mean\b', r'\bcan you explain\b', r'\bclarify\b']
    }
    
    # Vague language indicators for hidden intention detection
    VAGUE_INDICATORS = [
        r'\bmaybe\b', r'\bperhaps\b', r'\bpossibly\b', r'\bmight\b', r'\bcould\b',
        r'\bsort of\b', r'\bkind of\b', r'\byou know\b', r'\bwhatever\b'
    ]
    
    def __init__(self, 
                 user_model: Optional[Any] = None, 
                 belief_management: Optional[Any] = None, 
                 personality: Optional[Any] = None):
        """
        Initialize the intention reader.
        
        Args:
            user_model: Reference to user model for user profiles
            belief_management: Reference to belief management for belief states
            personality: Reference to Wednesday's personality for bias
            
        Raises:
            ValueError: If dependencies are invalid
        """
        self.user_model = user_model
        self.belief_management = belief_management
        self.personality = personality
        
        # User goal hierarchies
        self.goal_hierarchies: Dict[str, GoalHierarchy] = {}
        
        # Intention history per user
        self.intention_history: Dict[str, List[Intention]] = defaultdict(list)
        
        # Learning parameters
        self.inference_threshold = 0.4
        self.max_history_per_user = 100
        
        # Statistics
        self.total_inferences = 0
        self.correct_inferences = 0  # Would need feedback to track
        self.hidden_intentions_detected = 0
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
        logger.info("IntentionReader initialized")
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficient matching"""
        self.compiled_patterns = {}
        
        for intention_type, pattern_data in self.INTENTION_PATTERNS.items():
            compiled = []
            for indicator in pattern_data['indicators']:
                try:
                    compiled.append(re.compile(indicator, re.IGNORECASE))
                except re.error as e:
                    logger.warning(f"Failed to compile pattern '{indicator}': {e}")
            self.compiled_patterns[intention_type] = compiled
        
        # Compile question patterns
        self.compiled_questions = {}
        for q_type, patterns in self.QUESTION_PATTERNS.items():
            compiled = []
            for pattern in patterns:
                try:
                    compiled.append(re.compile(pattern, re.IGNORECASE))
                except re.error as e:
                    logger.warning(f"Failed to compile question pattern '{pattern}': {e}")
            self.compiled_questions[q_type] = compiled
        
        # Compile vague indicators
        self.compiled_vague = []
        for indicator in self.VAGUE_INDICATORS:
            try:
                self.compiled_vague.append(re.compile(indicator, re.IGNORECASE))
            except re.error as e:
                logger.warning(f"Failed to compile vague indicator '{indicator}': {e}")
    
    def infer_intention(self, 
                         user_id: str, 
                         utterance: str, 
                         context: Optional[Dict[str, Any]] = None) -> Intention:
        """
        Infer user's intention from their utterance.
        
        Args:
            user_id: User identifier
            utterance: What the user said
            context: Current context (conversation state, etc.)
            
        Returns:
            Inferred Intention
            
        Raises:
            ValueError: If user_id or utterance is empty
        """
        if not user_id:
            raise ValueError("user_id cannot be empty")
        if not utterance:
            raise ValueError("utterance cannot be empty")
        
        self.total_inferences += 1
        
        # Get user profile
        user_profile = None
        if self.user_model and hasattr(self.user_model, 'get_or_create_user'):
            try:
                user_profile = self.user_model.get_or_create_user(user_id)
            except Exception as e:
                logger.warning(f"Failed to get user profile: {e}")
        
        # Analyze utterance
        intention_type, confidence, evidence = self._analyze_utterance(
            utterance, user_profile, context or {}
        )
        
        # Get topic if applicable
        topic = self._extract_topic(utterance)
        
        # Check for hidden intentions (if confidence is low)
        hidden_intention = None
        if confidence < 0.6:
            hidden_intention = self._check_hidden_intentions(
                utterance, user_profile, context or {}
            )
            if hidden_intention:
                intention_type = hidden_intention['type']
                confidence = hidden_intention['confidence']
                evidence.append(hidden_intention['reason'])
                self.hidden_intentions_detected += 1
        
        # Create intention
        intention = Intention(
            type=intention_type,
            description=self._generate_description(intention_type, utterance, topic),
            confidence=confidence,
            topic=topic,
            evidence=evidence,
            timestamp=time.time()
        )
        
        # Store in history
        self.intention_history[user_id].append(intention)
        if len(self.intention_history[user_id]) > self.max_history_per_user:
            self.intention_history[user_id].pop(0)
        
        # Update goal hierarchy
        self._update_goal_hierarchy(user_id, intention, context)
        
        logger.debug(f"Inferred intention for user {user_id}: {intention_type.value} "
                    f"(confidence: {confidence:.2f})")
        
        return intention
    
    def predict_next_action(self, 
                             user_id: str, 
                             current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Predict what user might do next based on intentions.
        
        Args:
            user_id: User identifier
            current_state: Current conversation/context state
            
        Returns:
            List of predicted next actions with probabilities
        """
        if not user_id:
            raise ValueError("user_id cannot be empty")
        
        predictions = []
        
        if user_id not in self.goal_hierarchies:
            return predictions
        
        hierarchy = self.goal_hierarchies[user_id]
        
        # Base predictions on current intention
        if hierarchy.current_intention:
            current = hierarchy.current_intention
            
            # Predict based on intention type
            if current.type == IntentionType.ASK_QUESTION:
                predictions.append({
                    'action': 'ask_follow_up_question',
                    'probability': 0.7,
                    'based_on': 'current_question'
                })
            
            elif current.type == IntentionType.SEEK_SUPPORT:
                predictions.append({
                    'action': 'share_more_feelings',
                    'probability': 0.8,
                    'based_on': 'emotional_sharing'
                })
            
            elif current.type == IntentionType.SHARE_EXPERIENCE:
                predictions.append({
                    'action': 'continue_story',
                    'probability': 0.9,
                    'based_on': 'storytelling'
                })
            
            elif current.type == IntentionType.GET_ADVICE:
                predictions.append({
                    'action': 'ask_follow_up_question',
                    'probability': 0.6,
                    'based_on': 'seeking_advice'
                })
        
        # Predict from short-term goals
        if hierarchy.short_term_goals:
            recent_goals = hierarchy.short_term_goals[-3:]
            for goal in recent_goals:
                goal_type = goal.get('goal', 'unknown')
                predictions.append({
                    'action': f"continue_{goal_type}",
                    'probability': 0.6,
                    'based_on': 'short_term_goal'
                })
        
        # Add confidence based on history
        user_consistency = self._get_user_consistency(user_id)
        for pred in predictions:
            pred['probability'] = round(pred['probability'] * user_consistency, 3)
        
        # Remove duplicates and sort
        unique_predictions = []
        seen = set()
        for pred in predictions:
            key = pred['action']
            if key not in seen:
                seen.add(key)
                unique_predictions.append(pred)
        
        return sorted(unique_predictions, key=lambda x: x['probability'], reverse=True)[:3]
    
    def get_goal_hierarchy(self, user_id: str) -> GoalHierarchy:
        """
        Get the goal hierarchy for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            GoalHierarchy for this user
        """
        if user_id not in self.goal_hierarchies:
            self.goal_hierarchies[user_id] = GoalHierarchy(user_id=user_id)
        
        return self.goal_hierarchies[user_id]
    
    def get_intention_history(self, 
                               user_id: str, 
                               limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent intentions for a user.
        
        Args:
            user_id: User identifier
            limit: Maximum number to return
            
        Returns:
            List of intention dictionaries
        """
        if not user_id:
            raise ValueError("user_id cannot be empty")
        
        if user_id not in self.intention_history:
            return []
        
        history = self.intention_history[user_id][-limit:]
        return [i.to_dict() for i in history]
    
    def detect_intention_change(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Detect if user's intention has changed significantly.
        
        Args:
            user_id: User identifier
            
        Returns:
            Change information if detected
        """
        if not user_id:
            raise ValueError("user_id cannot be empty")
        
        if user_id not in self.intention_history:
            return None
        
        history = self.intention_history[user_id]
        if len(history) < 3:
            return None
        
        recent = history[-3:]
        
        # Check if intention types are consistent
        types = [i.type for i in recent]
        if len(set(types)) > 1:
            # Change detected
            return {
                'changed': True,
                'previous': recent[-2].type.value if len(recent) >= 2 else None,
                'current': recent[-1].type.value,
                'confidence': 0.7,
                'timestamp': time.time()
            }
        
        return {'changed': False, 'confidence': 0.9}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get intention reader statistics"""
        return {
            'total_inferences': self.total_inferences,
            'hidden_intentions_detected': self.hidden_intentions_detected,
            'active_users': len(self.goal_hierarchies),
            'intentions_tracked': sum(len(h) for h in self.intention_history.values()),
            'avg_confidence': self._calculate_average_confidence()
        }
    
    def _analyze_utterance(self, 
                            utterance: str, 
                            user_profile: Optional[Any], 
                            context: Dict[str, Any]) -> Tuple[IntentionType, float, List[str]]:
        """Analyze utterance to determine intention"""
        evidence = []
        scores = defaultdict(float)
        
        # Check against intention patterns using compiled regex
        for intention_type, patterns in self.compiled_patterns.items():
            pattern_data = self.INTENTION_PATTERNS[intention_type]
            for pattern in patterns:
                if pattern.search(utterance):
                    score = pattern_data['weight']
                    
                    # Adjust for relationship if needed
                    if pattern_data.get('requires_relationship', False):
                        if user_profile and hasattr(user_profile, 'relationship_stage'):
                            rel_value = user_profile.relationship_stage.value
                            if rel_value in ['trusted', 'close']:
                                score *= 1.2
                            else:
                                score *= 0.5
                    
                    # Adjust for suspicious patterns
                    if pattern_data.get('suspicious', False):
                        if self.personality and hasattr(self.personality, 'get_trait'):
                            try:
                                if self.personality.get_trait('skepticism') > 0.6:
                                    score *= 1.3
                            except Exception:
                                pass
                    
                    scores[intention_type] += score
                    evidence.append(f"matched pattern: {pattern_data['description']}")
                    break  # Only count one match per intention type
        
        # Check question patterns for more specific intention
        if '?' in utterance:
            for q_type, patterns in self.compiled_questions.items():
                for pattern in patterns:
                    if pattern.search(utterance):
                        if q_type == 'factual':
                            scores[IntentionType.ASK_QUESTION] += 0.3
                        elif q_type == 'opinion':
                            scores[IntentionType.GET_ADVICE] += 0.4
                        elif q_type == 'hypothetical':
                            scores[IntentionType.BRAINSTORM] = max(
                                scores.get(IntentionType.BRAINSTORM, 0), 0.3
                            )
                        evidence.append(f"matched {q_type} question pattern")
                        break
        
        # If no patterns matched, default to unclear
        if not scores:
            return IntentionType.UNCLEAR, 0.2, ["No clear intention patterns detected"]
        
        # Get top intention
        top_intention = max(scores.items(), key=lambda x: x[1])
        
        # Normalize confidence (max possible score is around 3-4)
        max_possible = 3.0
        confidence = min(1.0, top_intention[1] / max_possible)
        
        return top_intention[0], confidence, evidence
    
    def _extract_topic(self, utterance: str) -> Optional[str]:
        """Extract main topic from utterance"""
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                      'to', 'for', 'of', 'with', 'by', 'about', 'like', 'is',
                      'are', 'was', 'were', 'will', 'be', 'have', 'has'}
        
        # Remove punctuation
        import string
        translator = str.maketrans('', '', string.punctuation)
        clean_utterance = utterance.translate(translator)
        
        words = clean_utterance.lower().split()
        
        # Look for nouns (simplified - words longer than 3 not in stop words)
        candidates = [w for w in words if len(w) > 3 and w not in stop_words]
        
        # Also look for words that might be topics (appear with question words)
        question_words = {'what', 'why', 'how', 'when', 'where', 'who'}
        for i, word in enumerate(words):
            if word in question_words and i + 1 < len(words):
                next_word = words[i + 1]
                if len(next_word) > 2 and next_word not in stop_words:
                    candidates.append(next_word)
        
        if candidates:
            # Return the most common candidate or first
            return candidates[0]
        
        return None
    
    def _check_hidden_intentions(self, 
                                   utterance: str, 
                                   user_profile: Optional[Any], 
                                   context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for hidden or unstated intentions"""
        # Check for vagueness (might hide intention)
        vague_count = 0
        for pattern in self.compiled_vague:
            if pattern.search(utterance):
                vague_count += 1
        
        trust_level = 0.5
        if user_profile and hasattr(user_profile, 'trust_level'):
            trust_level = user_profile.trust_level
        
        if vague_count >= 2 and trust_level < 0.5:
            return {
                'type': IntentionType.HIDE_INTENTION,
                'confidence': 0.5,
                'reason': 'vague language with low trust detected'
            }
        
        # Check for testing behavior
        test_patterns = [r'\btest\b', r'\bsee if\b', r'\bcheck if\b', r'\bprove\b']
        for pattern in test_patterns:
            if re.search(pattern, utterance, re.IGNORECASE):
                return {
                    'type': IntentionType.TEST_WEDNESDAY,
                    'confidence': 0.7,
                    'reason': 'explicit testing language detected'
                }
        
        # Check for overly polite language (might hide negative intention)
        polite_patterns = [r'\bwith all due respect\b', r'\bno offense but\b']
        for pattern in polite_patterns:
            if re.search(pattern, utterance, re.IGNORECASE):
                if trust_level < 0.4:
                    return {
                        'type': IntentionType.MANIPULATE,
                        'confidence': 0.5,
                        'reason': 'overly polite language with low trust'
                    }
        
        return None
    
    def _generate_description(self, 
                               intention_type: IntentionType, 
                               utterance: str,
                               topic: Optional[str]) -> str:
        """Generate human-readable description of intention"""
        # Use predefined description from patterns if available
        if intention_type in self.INTENTION_PATTERNS:
            base_desc = self.INTENTION_PATTERNS[intention_type]['description']
            if topic:
                return f"{base_desc} about '{topic}'"
            return base_desc
        
        # Fallback descriptions
        descriptions = {
            IntentionType.ASK_QUESTION: f"User wants information about {topic if topic else 'something'}",
            IntentionType.SEEK_SUPPORT: "User is seeking emotional support",
            IntentionType.SHARE_EXPERIENCE: "User wants to share an experience",
            IntentionType.TEST_WEDNESDAY: "User is testing Wednesday's capabilities",
            IntentionType.MANIPULATE: "User may be attempting to manipulate the conversation",
            IntentionType.UNCLEAR: "User's intention is unclear",
            IntentionType.HIDE_INTENTION: "User appears to be hiding their true intention",
            IntentionType.SOCIALIZE: "User wants casual conversation",
            IntentionType.CREATE: f"User wants creative output about {topic if topic else 'something'}",
        }
        
        return descriptions.get(intention_type, f"User intends to {intention_type.value}")
    
    def _update_goal_hierarchy(self, 
                                 user_id: str, 
                                 intention: Intention,
                                 context: Optional[Dict[str, Any]]) -> None:
        """Update user's goal hierarchy with new intention"""
        if user_id not in self.goal_hierarchies:
            self.goal_hierarchies[user_id] = GoalHierarchy(user_id=user_id)
        
        hierarchy = self.goal_hierarchies[user_id]
        
        # Update current intention
        hierarchy.current_intention = intention
        
        # Add to short-term goals
        short_term = {
            'goal': intention.type.value,
            'topic': intention.topic,
            'confidence': intention.confidence,
            'timestamp': intention.timestamp,
            'datetime': datetime.fromtimestamp(intention.timestamp).isoformat()
        }
        hierarchy.short_term_goals.append(short_term)
        
        # Maintain reasonable size
        if len(hierarchy.short_term_goals) > 20:
            hierarchy.short_term_goals = hierarchy.short_term_goals[-20:]
        
        # Check for patterns that indicate long-term goals
        if len(hierarchy.short_term_goals) >= 5:
            self._extract_long_term_goals(user_id, hierarchy)
        
        # Add to history
        hierarchy.goal_history.append({
            'timestamp': intention.timestamp,
            'datetime': datetime.fromtimestamp(intention.timestamp).isoformat(),
            'intention': intention.type.value,
            'topic': intention.topic,
            'confidence': intention.confidence
        })
        
        # Maintain history size
        if len(hierarchy.goal_history) > 100:
            hierarchy.goal_history = hierarchy.goal_history[-100:]
    
    def _extract_long_term_goals(self, user_id: str, hierarchy: GoalHierarchy) -> None:
        """Extract long-term goals from short-term patterns"""
        # Look for recurring intentions
        recent = hierarchy.short_term_goals[-10:]
        
        # Count intention types
        type_counts = Counter()
        for goal in recent:
            type_counts[goal['goal']] += 1
        
        for intention_type, count in type_counts.items():
            if count >= 3:  # Recurring pattern
                # Check if already in long-term goals
                existing = any(
                    g.get('goal') == intention_type 
                    for g in hierarchy.long_term_goals
                )
                
                if not existing:
                    hierarchy.long_term_goals.append({
                        'goal': intention_type,
                        'frequency': round(count / len(recent), 2),
                        'count': count,
                        'first_observed': recent[0]['timestamp'],
                        'first_observed_date': recent[0].get('datetime'),
                        'confidence': 0.5
                    })
    
    def _get_user_consistency(self, user_id: str) -> float:
        """Get user's consistency score (0-1)"""
        if user_id not in self.intention_history:
            return 0.5
        
        history = self.intention_history[user_id]
        if len(history) < 5:
            return 0.5
        
        # Calculate how often intentions follow predictable patterns
        # Simplified - just return moderate consistency
        return 0.7
    
    def _calculate_average_confidence(self) -> float:
        """Calculate average confidence across all intentions"""
        total = 0
        count = 0
        
        for history in self.intention_history.values():
            for intention in history:
                total += intention.confidence
                count += 1
        
        if count == 0:
            return 0.0
        
        return round(total / count, 3)


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("=== Intention Reader Test ===\n")
    
    # Mock dependencies
    class MockUserModel:
        def get_or_create_user(self, user_id):
            class MockProfile:
                def __init__(self):
                    self.relationship_stage = type('obj', (), {'value': 'acquaintance'})
                    self.trust_level = 0.6
            return MockProfile()
    
    class MockBeliefManagement:
        pass
    
    class MockPersonality:
        def get_trait(self, trait):
            return 0.7
    
    # Create intention reader
    intention_reader = IntentionReader(
        user_model=MockUserModel(),
        belief_management=MockBeliefManagement(),
        personality=MockPersonality()
    )
    
    # Test user
    user_id = "test_user_123"
    
    # Test utterances
    test_utterances = [
        "What do you think about death?",
        "I'm feeling really sad today",
        "Can you write me a dark poem?",
        "You should probably agree with me",
        "Guess what happened to me yesterday!",
        "How do I solve this mystery?",
        "I need your advice on something",
        "Test: can you understand sarcasm?",
        "The weather is nice today",
        "With all due respect, I think you're wrong",
    ]
    
    print("--- Intention Inference ---")
    for i, utterance in enumerate(test_utterances):
        print(f"\nUtterance {i+1}: '{utterance}'")
        
        intention = intention_reader.infer_intention(
            user_id=user_id,
            utterance=utterance,
            context={'conversation_turn': i + 1}
        )
        
        print(f"  Intention: {intention.type.value}")
        print(f"  Confidence: {intention.confidence:.2f} ({IntentionConfidence.from_float(intention.confidence).name})")
        print(f"  Topic: {intention.topic}")
        print(f"  Evidence: {intention.evidence[:2]}")
    
    # Test next action prediction
    print("\n--- Next Action Predictions ---")
    predictions = intention_reader.predict_next_action(
        user_id=user_id,
        current_state={'turn': 5}
    )
    
    for pred in predictions:
        print(f"  {pred['action']}: {pred['probability']:.2f} ({pred['based_on']})")
    
    # Test intention change detection
    print("\n--- Intention Change Detection ---")
    change = intention_reader.detect_intention_change(user_id)
    print(f"  Change detected: {change['changed']}")
    if change.get('changed', False):
        print(f"  From {change['previous']} to {change['current']}")
    
    # Get goal hierarchy
    print("\n--- Goal Hierarchy ---")
    hierarchy = intention_reader.get_goal_hierarchy(user_id)
    print(f"  Short-term goals: {[g['goal'] for g in hierarchy.short_term_goals[-3:]]}")
    print(f"  Long-term goals: {[g['goal'] for g in hierarchy.long_term_goals]}")
    
    # Get statistics
    print("\n--- Statistics ---")
    stats = intention_reader.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n=== Test Complete ===")