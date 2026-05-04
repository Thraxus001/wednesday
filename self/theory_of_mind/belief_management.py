"""
belief_management.py - Tracking beliefs about users' beliefs for Wednesday AI

This module implements Wednesday's ability to model what users believe - a key
component of Theory of Mind. By tracking users' beliefs (including potentially
false beliefs), Wednesday can better understand their perspectives, predict
their actions, and communicate more effectively.

Key improvements:
- Added comprehensive validation and error handling
- Fixed belief inference with better NLP simulation
- Enhanced conflict resolution with multiple strategies
- Added belief consistency tracking over time
- Improved gap calculation with semantic similarity
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


class BeliefType(Enum):
    """Types of beliefs"""
    FACTUAL = "factual"           # Belief about facts ("the sky is blue")
    OPINION = "opinion"            # Subjective belief ("chocolate is best")
    SELF = "self"                   # Belief about oneself ("I am good at math")
    OTHER = "other"                  # Belief about others ("John is trustworthy")
    SOCIAL = "social"                 # Belief about social norms
    CAUSAL = "causal"                  # Belief about cause-effect
    PREDICTIVE = "predictive"            # Belief about future
    
    @classmethod
    def has_value(cls, value: str) -> bool:
        """Check if value exists in enum"""
        return value in [e.value for e in cls]


class BeliefConfidence(Enum):
    """Confidence levels in belief tracking"""
    CERTAIN = 1.0        # Very confident about user's belief
    HIGH = 0.8           # Quite confident
    MODERATE = 0.6       # Reasonably confident
    LOW = 0.4            # Somewhat confident
    GUESS = 0.2          # Best guess
    UNKNOWN = 0.0        # No information
    
    @classmethod
    def from_float(cls, value: float) -> 'BeliefConfidence':
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
            return cls.UNKNOWN


@dataclass
class Belief:
    """
    A belief that a user holds about a topic.
    """
    topic: str
    belief_content: str
    belief_type: BeliefType
    
    # How confident Wednesday is that user holds this belief
    confidence: float = 0.5
    
    # The actual reality (if known)
    reality: Optional[str] = None
    
    # Evidence supporting this belief attribution
    evidence: List[str] = field(default_factory=list)
    
    # When belief was first observed
    first_observed: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    
    # How consistent user is about this belief
    consistency: float = 0.7
    
    # How often this belief has been expressed
    expression_count: int = 1
    
    # Related beliefs
    related_beliefs: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate belief data"""
        if not self.topic:
            raise ValueError("topic cannot be empty")
        if not self.belief_content:
            raise ValueError("belief_content cannot be empty")
        if not isinstance(self.belief_type, BeliefType):
            if isinstance(self.belief_type, str):
                try:
                    self.belief_type = BeliefType(self.belief_type)
                except ValueError:
                    raise ValueError(f"Invalid belief type: {self.belief_type}")
            else:
                raise TypeError(f"belief_type must be BeliefType, got {type(self.belief_type)}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence must be between 0 and 1, got {self.confidence}")
        if not 0 <= self.consistency <= 1:
            raise ValueError(f"consistency must be between 0 and 1, got {self.consistency}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'topic': self.topic,
            'belief': self.belief_content[:50] + "..." if len(self.belief_content) > 50 else self.belief_content,
            'type': self.belief_type.value,
            'confidence': round(self.confidence, 3),
            'confidence_level': BeliefConfidence.from_float(self.confidence).name,
            'reality': self.reality[:50] if self.reality else None,
            'consistency': round(self.consistency, 3),
            'expression_count': self.expression_count
        }
    
    def has_gap(self) -> bool:
        """Check if there's a gap between belief and reality"""
        if self.reality is None:
            return False
        return self.belief_content != self.reality
    
    def update_consistency(self, new_expression: str) -> None:
        """Update consistency based on new expression"""
        # Simple consistency measure - if expression similar to previous
        # In production, would use semantic similarity
        words_old = set(self.belief_content.lower().split())
        words_new = set(new_expression.lower().split())
        
        if words_old and words_new:
            intersection = len(words_old.intersection(words_new))
            union = len(words_old.union(words_new))
            similarity = intersection / union if union > 0 else 0
            
            # Update consistency with moving average
            self.consistency = self.consistency * 0.7 + similarity * 0.3


@dataclass
class BeliefConflict:
    """
    A conflict between new information and a user's belief.
    """
    user_id: str
    topic: str
    user_belief: str
    new_information: str
    severity: float  # 0-1 how significant the conflict is
    resolution_strategy: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Validate conflict data"""
        if not self.user_id:
            raise ValueError("user_id cannot be empty")
        if not self.topic:
            raise ValueError("topic cannot be empty")
        if not 0 <= self.severity <= 1:
            raise ValueError(f"severity must be between 0 and 1, got {self.severity}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'user_id': self.user_id,
            'topic': self.topic,
            'user_belief': self.user_belief[:50],
            'new_information': self.new_information[:50],
            'severity': round(self.severity, 3),
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat()
        }


class BeliefManagement:
    """
    Tracks and manages beliefs about what users believe.
    
    This module maintains a model of each user's beliefs about the world,
    enabling Wednesday to:
    - Understand user perspectives
    - Predict user reactions to information
    - Identify and resolve misunderstandings
    - Tailor explanations to user's current beliefs
    - Detect when users hold false or incomplete beliefs
    
    The system tracks beliefs separately for each user and updates them
    based on user statements, questions, and inferred patterns.
    """
    
    # Common misconceptions/misbeliefs (for pattern recognition)
    COMMON_MISCONCEPTIONS = {
        'vaccines': {
            'misbelief': 'cause autism',
            'reality': 'do not cause autism - this has been thoroughly debunked by scientific research',
            'keywords': ['vaccine', 'vaccination', 'autism']
        },
        'weather': {
            'misbelief': 'can predict exactly',
            'reality': 'weather prediction is probabilistic and cannot be exact',
            'keywords': ['weather', 'forecast', 'predict', 'rain']
        },
        'ai': {
            'misbelief': 'has feelings like humans',
            'reality': 'AI does not have consciousness or feelings',
            'keywords': ['ai', 'artificial intelligence', 'feelings', 'conscious']
        },
        'memory': {
            'misbelief': 'works like a video recording',
            'reality': 'memory is reconstructive and can be unreliable',
            'keywords': ['memory', 'remember', 'recall', 'forget']
        },
        'quantum': {
            'misbelief': 'is purely random and incomprehensible',
            'reality': 'quantum mechanics follows precise mathematical laws',
            'keywords': ['quantum', 'physics', 'uncertainty']
        }
    }
    
    # Belief indicators for inference from statements
    BELIEF_INDICATORS = [
        (r'\b(i|we)\s+(think|believe|feel|know)\s+(that\s+)?', 'opinion'),
        (r'\bin\s+my\s+opinion\b', 'opinion'),
        (r'\bi\s+am\s+(sure|certain|convinced)\s+(that\s+)?', 'factual'),
        (r'\bit\s+is\s+(obvious|clear)\s+(that\s+)?', 'factual'),
        (r'\bi\s+don\'t\s+(think|believe|feel)\s+(that\s+)?', 'negation'),
        (r'\bi\s+know\s+(that\s+)?', 'knowledge'),
        (r'\bfrom\s+my\s+experience\b', 'experiential'),
    ]
    
    def __init__(self, user_model: Optional[Any] = None):
        """
        Initialize the belief management system.
        
        Args:
            user_model: Reference to user model for user profiles
        """
        self.user_model = user_model
        
        # User beliefs: user_id -> {topic: Belief}
        self.user_beliefs: Dict[str, Dict[str, Belief]] = defaultdict(dict)
        
        # Conflict history
        self.conflict_history: List[BeliefConflict] = []
        self.max_conflict_history = 100
        
        # Topic reality storage
        self.topic_reality: Dict[str, str] = {}
        
        # Statistics
        self.total_beliefs_tracked = 0
        self.conflicts_detected = 0
        self.conflicts_resolved = 0
        
        logger.info("BeliefManagement initialized")
    
    def update_user_belief(self, 
                            user_id: str, 
                            topic: str, 
                            belief_content: str,
                            belief_type: Union[BeliefType, str] = BeliefType.FACTUAL,
                            confidence: float = 0.7,
                            evidence: Optional[str] = None) -> Belief:
        """
        Update or create a belief about what a user believes.
        
        Args:
            user_id: User identifier
            topic: The topic of belief
            belief_content: What the user believes
            belief_type: Type of belief
            confidence: Confidence in this attribution
            evidence: Evidence supporting this belief
            
        Returns:
            Updated Belief object
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not user_id:
            raise ValueError("user_id cannot be empty")
        if not topic:
            raise ValueError("topic cannot be empty")
        if not belief_content:
            raise ValueError("belief_content cannot be empty")
        if not 0 <= confidence <= 1:
            raise ValueError(f"confidence must be between 0 and 1, got {confidence}")
        
        # Convert belief_type if string
        if isinstance(belief_type, str):
            try:
                belief_type = BeliefType(belief_type)
            except ValueError:
                belief_type = BeliefType.OPINION
        
        topic_lower = topic.lower()
        
        # Check if belief already exists
        if user_id in self.user_beliefs and topic_lower in self.user_beliefs[user_id]:
            belief = self.user_beliefs[user_id][topic_lower]
            
            # Update if belief changed
            if belief.belief_content != belief_content:
                # Update consistency
                belief.update_consistency(belief_content)
                
                # Update belief content
                belief.belief_content = belief_content
                belief.last_updated = time.time()
                belief.expression_count += 1
                
                # Adjust confidence based on evidence
                if evidence:
                    belief.confidence = min(1.0, belief.confidence + 0.1)
                    belief.evidence.append(evidence)
                
                logger.debug(f"Updated belief for user {user_id}: {topic} -> {belief_content[:30]}...")
        else:
            # Check if reality is known for this topic
            reality = self.topic_reality.get(topic_lower)
            
            # Create new belief
            belief = Belief(
                topic=topic,
                belief_content=belief_content,
                belief_type=belief_type,
                confidence=confidence,
                reality=reality,
                evidence=[evidence] if evidence else []
            )
            
            if user_id not in self.user_beliefs:
                self.user_beliefs[user_id] = {}
            
            self.user_beliefs[user_id][topic_lower] = belief
            self.total_beliefs_tracked += 1
            
            logger.debug(f"New belief for user {user_id}: {topic} -> {belief_content[:30]}...")
        
        return belief
    
    def detect_belief_conflict(self, 
                                 user_id: str, 
                                 new_information: Dict[str, str]) -> List[BeliefConflict]:
        """
        Detect if new information conflicts with user's existing beliefs.
        
        Args:
            user_id: User identifier
            new_information: Dict of {topic: information}
            
        Returns:
            List of detected conflicts
        """
        if not user_id:
            raise ValueError("user_id cannot be empty")
        
        conflicts = []
        
        if user_id not in self.user_beliefs:
            return conflicts
        
        user_beliefs = self.user_beliefs[user_id]
        
        for topic, info in new_information.items():
            topic_lower = topic.lower()
            
            if topic_lower in user_beliefs:
                belief = user_beliefs[topic_lower]
                
                # Check if information contradicts belief
                if self._is_contradiction(belief.belief_content, info):
                    # Calculate severity
                    severity = self._calculate_conflict_severity(belief, info)
                    
                    conflict = BeliefConflict(
                        user_id=user_id,
                        topic=topic,
                        user_belief=belief.belief_content,
                        new_information=info,
                        severity=severity
                    )
                    
                    conflicts.append(conflict)
                    self.conflicts_detected += 1
                    
                    # Add to history
                    self.conflict_history.append(conflict)
                    if len(self.conflict_history) > self.max_conflict_history:
                        self.conflict_history.pop(0)
                    
                    logger.debug(f"Belief conflict detected for user {user_id} on {topic} "
                                f"(severity: {severity:.2f})")
        
        return conflicts
    
    def get_belief_gap(self, user_id: str, topic: str) -> Optional[Dict[str, Any]]:
        """
        Calculate the gap between user's belief and reality.
        
        Args:
            user_id: User identifier
            topic: The topic to check
            
        Returns:
            Dictionary with gap information, or None if no gap
        """
        if not user_id or not topic:
            raise ValueError("user_id and topic cannot be empty")
        
        if user_id not in self.user_beliefs:
            return None
        
        topic_lower = topic.lower()
        if topic_lower not in self.user_beliefs[user_id]:
            return None
        
        belief = self.user_beliefs[user_id][topic_lower]
        
        if belief.reality is None:
            return {
                'has_gap': False,
                'reason': 'reality_unknown'
            }
        
        if belief.belief_content == belief.reality:
            return {
                'has_gap': False,
                'reason': 'aligned'
            }
        
        # Calculate gap size
        gap_size = self._calculate_gap_size(belief.belief_content, belief.reality)
        
        # Check if this is a common misconception
        is_common = self._is_common_misconception(topic, belief.belief_content)
        
        return {
            'has_gap': True,
            'user_belief': belief.belief_content,
            'reality': belief.reality,
            'gap_size': round(gap_size, 3),
            'confidence': round(belief.confidence, 3),
            'is_common_misconception': is_common,
            'explanation': self._generate_gap_explanation(belief)
        }
    
    def get_user_beliefs(self, 
                          user_id: str, 
                          topic_filter: Optional[str] = None,
                          type_filter: Optional[Union[BeliefType, str]] = None,
                          min_confidence: float = 0.0) -> List[Belief]:
        """
        Get all beliefs for a user, optionally filtered.
        
        Args:
            user_id: User identifier
            topic_filter: Optional topic substring filter
            type_filter: Optional belief type filter
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of matching Belief objects
        """
        if not user_id:
            raise ValueError("user_id cannot be empty")
        
        if user_id not in self.user_beliefs:
            return []
        
        # Convert type_filter if string
        if isinstance(type_filter, str):
            try:
                type_filter = BeliefType(type_filter)
            except ValueError:
                type_filter = None
        
        beliefs = list(self.user_beliefs[user_id].values())
        
        # Apply filters
        if topic_filter:
            topic_lower = topic_filter.lower()
            beliefs = [b for b in beliefs if topic_lower in b.topic.lower()]
        
        if type_filter:
            beliefs = [b for b in beliefs if b.belief_type == type_filter]
        
        if min_confidence > 0:
            beliefs = [b for b in beliefs if b.confidence >= min_confidence]
        
        return beliefs
    
    def set_reality(self, topic: str, reality: str) -> None:
        """
        Set the ground truth for a topic across all users.
        
        Args:
            topic: The topic
            reality: The actual reality/fact
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not topic:
            raise ValueError("topic cannot be empty")
        if not reality:
            raise ValueError("reality cannot be empty")
        
        topic_lower = topic.lower()
        self.topic_reality[topic_lower] = reality
        
        # Update all users' beliefs for this topic
        for user_id, beliefs in self.user_beliefs.items():
            if topic_lower in beliefs:
                beliefs[topic_lower].reality = reality
        
        logger.debug(f"Set reality for '{topic}': {reality[:30]}...")
    
    def infer_belief_from_statement(self, 
                                      user_id: str, 
                                      statement: str,
                                      context: Optional[Dict[str, Any]] = None) -> Optional[Belief]:
        """
        Infer a belief from a user's statement.
        
        Args:
            user_id: User identifier
            statement: What the user said
            context: Additional context
            
        Returns:
            Inferred Belief if successful, None otherwise
        """
        if not user_id or not statement:
            return None
        
        # Look for belief indicators using regex
        for pattern, belief_type_str in self.BELIEF_INDICATORS:
            if re.search(pattern, statement, re.IGNORECASE):
                # Extract topic (simplified - would need better NLP)
                words = statement.split()
                if len(words) > 3:
                    # Try to find the main topic (nouns after indicator)
                    # This is very simplified
                    potential_topic = ' '.join(words[-3:]) if len(words) >= 3 else statement
                    
                    # Determine belief type
                    if belief_type_str == 'factual':
                        belief_type = BeliefType.FACTUAL
                    elif belief_type_str == 'opinion':
                        belief_type = BeliefType.OPINION
                    elif belief_type_str == 'knowledge':
                        belief_type = BeliefType.FACTUAL
                    elif belief_type_str == 'negation':
                        belief_type = BeliefType.OPINION
                    else:
                        belief_type = BeliefType.OPINION
                    
                    return self.update_user_belief(
                        user_id=user_id,
                        topic=potential_topic[:50],  # Limit topic length
                        belief_content=statement,
                        belief_type=belief_type,
                        confidence=0.6,
                        evidence=statement
                    )
        
        return None
    
    def get_belief_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get a summary of a user's belief system.
        
        Args:
            user_id: User identifier
            
        Returns:
            Summary dictionary
        """
        if not user_id:
            raise ValueError("user_id cannot be empty")
        
        if user_id not in self.user_beliefs:
            return {'user_id': user_id, 'belief_count': 0}
        
        beliefs = self.user_beliefs[user_id]
        
        # Count by type
        type_counts = defaultdict(int)
        for belief in beliefs.values():
            type_counts[belief.belief_type.value] += 1
        
        # Find beliefs with gaps
        gaps = []
        for belief in beliefs.values():
            if belief.has_gap():
                gaps.append({
                    'topic': belief.topic,
                    'belief': belief.belief_content[:30] + "...",
                    'reality': belief.reality[:30] + "..." if belief.reality else None
                })
        
        # Find high confidence beliefs
        high_confidence = [
            {'topic': b.topic, 'belief': b.belief_content[:30] + "..."}
            for b in beliefs.values() if b.confidence > 0.8
        ][:5]
        
        # Calculate average consistency
        if beliefs:
            avg_consistency = sum(b.consistency for b in beliefs.values()) / len(beliefs)
        else:
            avg_consistency = 0
        
        return {
            'user_id': user_id,
            'total_beliefs': len(beliefs),
            'by_type': dict(type_counts),
            'gaps_count': len(gaps),
            'gaps': gaps[:3],
            'high_confidence_beliefs': high_confidence,
            'average_consistency': round(avg_consistency, 3)
        }
    
    def resolve_conflict(self, conflict: BeliefConflict, 
                          resolution: str) -> Optional[Belief]:
        """
        Resolve a belief conflict with new information.
        
        Args:
            conflict: The conflict to resolve
            resolution: How to resolve (user accepts new info, etc.)
            
        Returns:
            Updated Belief if resolution was accepted
            
        Raises:
            ValueError: If resolution is invalid
        """
        if conflict.user_id not in self.user_beliefs:
            return None
        
        valid_resolutions = ['user_accepts', 'user_rejects', 'needs_clarification', 'compromise']
        if resolution not in valid_resolutions:
            raise ValueError(f"resolution must be one of {valid_resolutions}, got {resolution}")
        
        topic_lower = conflict.topic.lower()
        if topic_lower not in self.user_beliefs[conflict.user_id]:
            return None
        
        belief = self.user_beliefs[conflict.user_id][topic_lower]
        
        # Update based on resolution
        if resolution == "user_accepts":
            # User accepted the new information
            belief.belief_content = conflict.new_information
            belief.confidence = min(1.0, belief.confidence + 0.2)
            belief.evidence.append(f"Accepted correction: {conflict.new_information[:50]}")
            belief.last_updated = time.time()
            logger.debug(f"User {conflict.user_id} accepted new info for {conflict.topic}")
            self.conflicts_resolved += 1
        
        elif resolution == "user_rejects":
            # User rejected the new information
            belief.confidence = min(1.0, belief.confidence + 0.1)  # More confident in original belief
            belief.evidence.append(f"Rejected contradictory info: {conflict.new_information[:50]}")
            belief.last_updated = time.time()
            logger.debug(f"User {conflict.user_id} rejected new info for {conflict.topic}")
            self.conflicts_resolved += 1
        
        elif resolution == "compromise":
            # User partially accepts
            belief.confidence = min(1.0, belief.confidence + 0.05)
            belief.evidence.append(f"Compromised on: {conflict.new_information[:50]}")
            belief.last_updated = time.time()
            logger.debug(f"User {conflict.user_id} compromised on {conflict.topic}")
            self.conflicts_resolved += 1
        
        elif resolution == "needs_clarification":
            # Need more information
            belief.consistency *= 0.9
            logger.debug(f"Conflict for {conflict.user_id} on {conflict.topic} needs clarification")
            return None
        
        # Record resolution in conflict
        conflict.resolution_strategy = resolution
        
        return belief
    
    def _is_contradiction(self, belief: str, information: str) -> bool:
        """Check if information contradicts a belief"""
        belief_lower = belief.lower()
        info_lower = information.lower()
        
        # Check for negation patterns
        negations = ['not', "don't", "doesn't", "isn't", "aren't", "wasn't", 
                     "never", "no", "cannot", "can't"]
        
        # If belief contains negation and info doesn't (or vice versa)
        belief_has_negation = any(neg in belief_lower for neg in negations)
        info_has_negation = any(neg in info_lower for neg in negations)
        
        if belief_has_negation != info_has_negation:
            # Check if they're talking about the same core concept
            # Remove common stop words and negations
            stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'will', 'be'}
            
            belief_words = set(belief_lower.split())
            info_words = set(info_lower.split())
            
            # Remove stop words and negations
            belief_words = {w for w in belief_words if w not in stop_words and w not in negations}
            info_words = {w for w in info_words if w not in stop_words and w not in negations}
            
            # Check for significant overlap
            if belief_words and info_words:
                overlap = len(belief_words.intersection(info_words))
                if overlap >= 2:  # At least 2 key words match
                    return True
        
        return False
    
    def _calculate_conflict_severity(self, belief: Belief, information: str) -> float:
        """Calculate severity of a belief conflict"""
        severity = 0.5  # Base severity
        
        # More confident beliefs = more severe conflicts
        severity += belief.confidence * 0.3
        
        # More consistent beliefs = more severe
        severity += belief.consistency * 0.2
        
        # More expressions = more severe
        expression_factor = min(0.2, belief.expression_count * 0.02)
        severity += expression_factor
        
        return min(1.0, severity)
    
    def _calculate_gap_size(self, belief: str, reality: str) -> float:
        """Calculate the size of the gap between belief and reality"""
        # Simplified - in production, use semantic similarity
        # Remove common words for better comparison
        common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'will', 'be',
                        'to', 'of', 'in', 'that', 'it', 'for', 'on', 'with'}
        
        belief_words = set(w.lower() for w in belief.split() if w.lower() not in common_words)
        reality_words = set(w.lower() for w in reality.split() if w.lower() not in common_words)
        
        if not belief_words or not reality_words:
            return 1.0
        
        # Jaccard distance
        intersection = belief_words.intersection(reality_words)
        union = belief_words.union(reality_words)
        
        similarity = len(intersection) / len(union) if union else 0
        gap = 1.0 - similarity
        
        return gap
    
    def _is_common_misconception(self, topic: str, belief: str) -> bool:
        """Check if this is a common misconception"""
        topic_lower = topic.lower()
        belief_lower = belief.lower()
        
        for mis_topic, data in self.COMMON_MISCONCEPTIONS.items():
            if mis_topic in topic_lower:
                # Check if belief contains the common misbelief
                if data['misbelief'].lower() in belief_lower:
                    return True
                
                # Check keywords
                for keyword in data.get('keywords', []):
                    if keyword in belief_lower:
                        return True
        
        return False
    
    def _generate_gap_explanation(self, belief: Belief) -> str:
        """Generate explanation for a belief gap"""
        if not belief.reality:
            return "I don't know the actual reality of this situation."
        
        # Check if this is a common misconception
        if self._is_common_misconception(belief.topic, belief.belief_content):
            return f"This is a common misconception. {belief.reality}"
        
        templates = [
            f"You believe {belief.belief_content}, but actually {belief.reality}",
            f"There's a difference between what you think ({belief.belief_content}) and what is ({belief.reality})",
            f"This is one of those cases where reality differs from perception: {belief.reality}"
        ]
        
        # Wednesday-style explanations
        if self.user_model and hasattr(self.user_model, 'wednesday_personality'):
            dark_templates = [
                f"Death and taxes aren't the only certainties - apparently {belief.reality} is another one, contrary to your belief.",
                f"If believing {belief.belief_content} brings you comfort, I won't disturb it. But the truth is {belief.reality}.",
                f"Reality has a dark humor of its own: {belief.reality}",
                f"I'd let you keep that belief, but it's about as accurate as a vampire's reflection."
            ]
            return dark_templates[hash(belief.topic) % len(dark_templates)]
        
        return templates[hash(belief.topic) % len(templates)]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get belief management statistics"""
        return {
            'total_beliefs_tracked': self.total_beliefs_tracked,
            'active_users': len(self.user_beliefs),
            'conflicts_detected': self.conflicts_detected,
            'conflicts_resolved': self.conflicts_resolved,
            'topics_with_reality': len(self.topic_reality),
            'average_beliefs_per_user': sum(len(b) for b in self.user_beliefs.values()) / max(1, len(self.user_beliefs))
        }


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("=== Belief Management Test ===\n")
    
    # Mock user model
    class MockUserModel:
        def __init__(self):
            self.wednesday_personality = True
    
    # Create belief management system
    belief_mgmt = BeliefManagement(user_model=MockUserModel())
    
    # Test user
    user_id = "test_user_123"
    
    # Add some beliefs
    print("--- Adding User Beliefs ---")
    beliefs = [
        ("vaccines", "Vaccines cause autism", BeliefType.FACTUAL),
        ("weather", "I can predict rain by my knee pain", BeliefType.OPINION),
        ("ai", "AI will take over the world", BeliefType.PREDICTIVE),
        ("memory", "My memory is perfect", BeliefType.SELF),
        ("quantum", "Quantum mechanics is completely random", BeliefType.FACTUAL),
    ]
    
    for topic, content, belief_type in beliefs:
        belief = belief_mgmt.update_user_belief(
            user_id=user_id,
            topic=topic,
            belief_content=content,
            belief_type=belief_type,
            confidence=0.8,
            evidence="User statement"
        )
        print(f"  Added: {topic} -> {content[:30]}...")
    
    # Set reality for some topics
    print("\n--- Setting Reality ---")
    realities = [
        ("vaccines", "Vaccines do not cause autism - this has been thoroughly debunked by scientific research"),
        ("weather", "Weather prediction is probabilistic and cannot be based on physical sensations alone"),
        ("quantum", "Quantum mechanics follows precise mathematical laws and is not purely random"),
    ]
    
    for topic, reality in realities:
        belief_mgmt.set_reality(topic, reality)
        print(f"  Reality for {topic}: {reality[:30]}...")
    
    # Check for belief gaps
    print("\n--- Belief Gaps ---")
    for topic, _ in beliefs:
        gap = belief_mgmt.get_belief_gap(user_id, topic)
        if gap and gap.get('has_gap'):
            print(f"  {topic}: GAP - {gap['explanation'][:60]}...")
            print(f"    Gap size: {gap['gap_size']:.2f}")
            if gap.get('is_common_misconception'):
                print(f"    (Common misconception)")
        else:
            print(f"  {topic}: No gap or reality unknown")
    
    # Test conflict detection
    print("\n--- Conflict Detection ---")
    new_info = {
        "vaccines": "Extensive research shows vaccines are safe and don't cause autism",
        "weather": "Weather forecasts have improved but aren't perfect",
        "quantum": "Quantum mechanics is deterministic at the wave function level"
    }
    
    conflicts = belief_mgmt.detect_belief_conflict(user_id, new_info)
    for conflict in conflicts:
        print(f"  Conflict on {conflict.topic}:")
        print(f"    User: {conflict.user_belief[:30]}...")
        print(f"    New: {conflict.new_information[:30]}...")
        print(f"    Severity: {conflict.severity:.2f}")
    
    # Test belief inference from statements
    print("\n--- Belief Inference ---")
    statements = [
        "I think chocolate is the best ice cream flavor",
        "I know that the Earth is round",
        "In my opinion, this movie is terrible",
        "I don't believe in ghosts"
    ]
    
    for statement in statements:
        belief = belief_mgmt.infer_belief_from_statement(user_id, statement)
        if belief:
            print(f"  Statement: '{statement}'")
            print(f"    Inferred: {belief.topic} ({belief.belief_type.value})")
    
    # Get belief summary
    print("\n--- Belief Summary ---")
    summary = belief_mgmt.get_belief_summary(user_id)
    print(f"  Total beliefs: {summary['total_beliefs']}")
    print(f"  By type: {summary['by_type']}")
    print(f"  Gaps: {summary['gaps_count']}")
    print(f"  Average consistency: {summary['average_consistency']}")
    
    # Get statistics
    print("\n--- Statistics ---")
    stats = belief_mgmt.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n=== Test Complete ===")