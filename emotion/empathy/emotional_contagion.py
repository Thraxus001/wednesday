"""
emotional_contagion.py - Emotional contagion for Wednesday AI

This module implements Wednesday's ability to "catch" emotions from others,
specifically the user. Based on psychological research on emotional contagion,
it models how observing others' emotional states can subtly influence one's own
emotions.

Key improvements:
- Fixed missing emotion mappings
- Added comprehensive validation and error handling
- Enhanced relationship multiplier management
- Improved trust dynamics with persistence
- Added proper type hints and documentation
"""

import time
import logging
import math
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)


class ContagionRegulation(Enum):
    """Levels of contagion regulation"""
    FULL = "full"           # Allow full emotional transfer
    MODERATE = "moderate"   # Allow some emotional transfer
    MINIMAL = "minimal"     # Allow very little transfer
    BLOCKED = "blocked"     # Block all emotional transfer


@dataclass
class ContagionEvent:
    """
    Record of an emotional contagion event.
    
    Tracks when and how an emotion was transferred from user to Wednesday.
    """
    source_emotion: str
    source_intensity: float
    transferred_emotion: str
    transferred_intensity: float
    contagion_factor: float
    regulation_level: ContagionRegulation
    user_id: str
    context: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Validate event data"""
        if not 0 <= self.source_intensity <= 1:
            raise ValueError(f"source_intensity must be between 0 and 1, got {self.source_intensity}")
        if not 0 <= self.transferred_intensity <= 1:
            raise ValueError(f"transferred_intensity must be between 0 and 1, got {self.transferred_intensity}")
        if not 0 <= self.contagion_factor <= 1:
            raise ValueError(f"contagion_factor must be between 0 and 1, got {self.contagion_factor}")
        if not self.user_id:
            raise ValueError("user_id cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'source_emotion': self.source_emotion,
            'source_intensity': round(self.source_intensity, 3),
            'transferred_emotion': self.transferred_emotion,
            'transferred_intensity': round(self.transferred_intensity, 3),
            'contagion_factor': round(self.contagion_factor, 3),
            'regulation': self.regulation_level.value,
            'user_id': self.user_id,
            'timestamp': self.timestamp
        }


class EmotionalContagion:
    """
    Models how Wednesday "catches" emotions from users.
    
    Emotional contagion is the tendency to automatically mimic and
    synchronize with the emotions of others. This module implements
    a controlled version appropriate for Wednesday's personality.
    
    Key factors affecting contagion:
    - Baseline susceptibility (Wednesday is naturally low)
    - Relationship with user (trust increases susceptibility)
    - Context (professional vs personal)
    - Current emotional state (some states more receptive)
    - Intensity of user's emotion (stronger = more contagious)
    """
    
    # Mapping from user emotions to how they affect Wednesday
    # Different emotions have different contagion weights
    EMOTION_CONTAGION_WEIGHTS = {
        # Basic emotions
        'happy': 0.4,          # Happiness is moderately contagious
        'sad': 0.6,            # Sadness is more contagious
        'angry': 0.7,          # Anger is highly contagious
        'fearful': 0.8,        # Fear is very contagious
        'surprised': 0.3,      # Surprise is less contagious
        'disgusted': 0.5,      # Disgust is moderately contagious
        
        # Complex emotions
        'frustrated': 0.6,
        'confused': 0.3,
        'amused': 0.4,
        'curious': 0.2,        # Curiosity has low emotional valence
        'hurt': 0.7,
        'defensive': 0.5,
        'trusting': 0.3,
        'suspicious': 0.5,
        'lonely': 0.6,
        'hopeful': 0.3,
        'nostalgic': 0.4,
        
        # Wednesday-specific user emotions
        'darkly_amused': 0.5,   # Mutual dark humor appreciation
        
        # Neutral
        'neutral': 0.1
    }
    
    # Mapping to Wednesday's emotional vocabulary
    # How user emotions translate to Wednesday's emotions
    EMOTION_TRANSLATION = {
        'happy': 'joy',
        'sad': 'sadness',
        'angry': 'anger',
        'fearful': 'fear',
        'surprised': 'surprise',
        'disgusted': 'disgust',
        'frustrated': 'anger',
        'confused': 'surprise',
        'amused': 'dark_amusement',  # Wednesday's special translation
        'curious': 'curiosity',
        'hurt': 'sadness',
        'defensive': 'anger',
        'trusting': 'trust',
        'suspicious': 'wary',
        'lonely': 'sadness',
        'hopeful': 'anticipation',
        'nostalgic': 'nostalgic',
        'darkly_amused': 'dark_amusement',
        'neutral': 'neutral'
    }
    
    # Valid emotion sets for validation
    VALID_USER_EMOTIONS = set(EMOTION_CONTAGION_WEIGHTS.keys())
    VALID_WEDNESDAY_EMOTIONS = set(EMOTION_TRANSLATION.values())
    
    # Relationship types and their multipliers
    RELATIONSHIP_MULTIPLIERS = {
        'stranger': 0.5,
        'acquaintance': 0.8,
        'friend': 1.2,
        'close_friend': 1.5,
        'trusted': 1.8,
        'adversary': 0.3,      # Less susceptible to adversaries
        'family': 1.3,
        'colleague': 0.9,
        'mentor': 1.1,
        'unknown': 1.0
    }
    
    def __init__(self, emotional_state: Any, personality: Optional[Dict[str, float]] = None):
        """
        Initialize the emotional contagion system.
        
        Args:
            emotional_state: Reference to EmotionalState to update
            personality: Optional personality parameters
            
        Raises:
            ValueError: If personality parameters are invalid
        """
        self.emotional_state = emotional_state
        
        # Base susceptibility (Wednesday is naturally controlled)
        self.base_susceptibility = 0.3
        
        # Personality influences on contagion
        default_personality = {
            'empathy_level': 0.5,           # Overall empathy (0-1)
            'emotional_boundaries': 0.8,     # Strong boundaries (0-1)
            'trust_sensitivity': 0.6,        # Trust increases contagion (0-1)
            'professional_detachment': 0.7,   # Can detach when needed (0-1)
            'mood_susceptibility': 0.4,       # How much mood affects contagion (0-1)
        }
        
        self.personality = default_personality.copy()
        if personality:
            self._validate_personality(personality)
            self.personality.update(personality)
        
        # Current regulation level
        self.current_regulation = ContagionRegulation.MODERATE
        
        # Contagion history
        self.contagion_history: List[ContagionEvent] = []
        self.max_history_size = 50
        
        # Trust levels per user
        self.user_trust: Dict[str, float] = {}  # user_id -> trust level (0-1)
        
        # User relationship cache
        self.user_relationships: Dict[str, str] = {}  # user_id -> relationship type
        
        logger.info(f"EmotionalContagion initialized with susceptibility {self.base_susceptibility}")
    
    def _validate_personality(self, personality: Dict[str, float]) -> None:
        """Validate personality parameters"""
        for key, value in personality.items():
            if key not in self.personality:
                raise ValueError(f"Unknown personality parameter: {key}")
            if not 0 <= value <= 1:
                raise ValueError(f"Personality parameter {key} must be between 0 and 1, got {value}")
    
    def catch_emotion(self, 
                      user_emotion: str, 
                      intensity: float,
                      user_id: str,
                      context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Process emotional contagion from user to Wednesday.
        
        Args:
            user_emotion: The inferred emotion of the user
            intensity: How strongly the user is feeling it (0-1)
            user_id: Identifier for the user
            context: Current context information
            
        Returns:
            Dictionary with contagion results, or None if no contagion
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate inputs
        if not user_emotion:
            raise ValueError("user_emotion cannot be empty")
        if not 0 <= intensity <= 1:
            raise ValueError(f"intensity must be between 0 and 1, got {intensity}")
        if not user_id:
            raise ValueError("user_id cannot be empty")
        
        # Normalize emotion string
        user_emotion = user_emotion.lower()
        
        # Check if emotion is in our mapping
        if user_emotion not in self.EMOTION_CONTAGION_WEIGHTS:
            logger.warning(f"Unknown user emotion: {user_emotion}, using neutral")
            user_emotion = 'neutral'
        
        # Check if contagion should be blocked
        if self.current_regulation == ContagionRegulation.BLOCKED:
            return None
        
        # Get contagion weight for this emotion
        contagion_weight = self.EMOTION_CONTAGION_WEIGHTS.get(user_emotion, 0.3)
        
        # Calculate susceptibility for this interaction
        susceptibility = self._calculate_susceptibility(user_id, context)
        
        # Apply regulation
        regulation_factor = self._get_regulation_factor()
        
        # Calculate contagion factor (how much emotion transfers)
        contagion_factor = (
            contagion_weight * 
            susceptibility * 
            regulation_factor * 
            intensity
        )
        
        # If factor is too low, no contagion
        if contagion_factor < 0.1:
            return None
        
        # Translate user emotion to Wednesday's emotional vocabulary
        wednesday_emotion = self.EMOTION_TRANSLATION.get(user_emotion, 'neutral')
        
        # Calculate transferred intensity (capped)
        transferred_intensity = min(0.5, contagion_factor * 0.8)
        
        # Apply to Wednesday's emotional state
        update_result = self._apply_contagion(
            wednesday_emotion, 
            transferred_intensity,
            context
        )
        
        # Record contagion event
        event = ContagionEvent(
            source_emotion=user_emotion,
            source_intensity=intensity,
            transferred_emotion=wednesday_emotion,
            transferred_intensity=transferred_intensity,
            contagion_factor=contagion_factor,
            regulation_level=self.current_regulation,
            user_id=user_id,
            context=context
        )
        self._add_to_history(event)
        
        logger.debug(f"Emotional contagion: {user_emotion} ({intensity:.2f}) -> "
                    f"{wednesday_emotion} ({transferred_intensity:.2f})")
        
        return {
            'source_emotion': user_emotion,
            'transferred_emotion': wednesday_emotion,
            'transferred_intensity': round(transferred_intensity, 3),
            'contagion_factor': round(contagion_factor, 3),
            'susceptibility': round(susceptibility, 3),
            'regulation': self.current_regulation.value,
            'emotional_update': update_result
        }
    
    def regulate_contagion(self, 
                           context: Dict[str, Any],
                           target_level: Optional[ContagionRegulation] = None) -> None:
        """
        Adjust contagion regulation based on context.
        
        Args:
            context: Current context information
            target_level: Optional specific regulation level to set
        """
        if target_level:
            if not isinstance(target_level, ContagionRegulation):
                raise ValueError(f"target_level must be ContagionRegulation, got {type(target_level)}")
            self.current_regulation = target_level
            logger.info(f"Contagion regulation manually set to {target_level.value}")
            return
        
        # Automatic regulation based on context
        old_regulation = self.current_regulation
        
        # Get context values with defaults
        formality = context.get('formality', 0.5)
        context_type = context.get('context_type', 'general')
        relationship = context.get('relationship', 'unknown')
        
        # Professional contexts require more regulation
        if formality > 0.7 or context_type == 'professional':
            self.current_regulation = ContagionRegulation.MINIMAL
        
        # Therapeutic contexts might allow more
        elif context_type in ['supportive', 'therapeutic', 'counseling']:
            self.current_regulation = ContagionRegulation.MODERATE
        
        # Personal contexts with trusted users allow more
        elif relationship in ['close_friend', 'trusted', 'family']:
            self.current_regulation = ContagionRegulation.FULL
        
        # Crisis situations might need full awareness
        elif context.get('emergency', False):
            self.current_regulation = ContagionRegulation.FULL
        
        # Default for most situations
        else:
            self.current_regulation = ContagionRegulation.MODERATE
        
        # If regulation changed, log it
        if old_regulation != self.current_regulation:
            logger.info(f"Contagion regulation adjusted: {old_regulation.value} -> "
                       f"{self.current_regulation.value}")
    
    def set_user_trust(self, user_id: str, trust_level: float) -> None:
        """
        Set trust level for a specific user.
        
        Args:
            user_id: User identifier
            trust_level: Trust level (0-1)
            
        Raises:
            ValueError: If trust_level is outside valid range
        """
        if not 0 <= trust_level <= 1:
            raise ValueError(f"trust_level must be between 0 and 1, got {trust_level}")
        
        self.user_trust[user_id] = trust_level
        logger.debug(f"Set trust for user {user_id} to {trust_level:.2f}")
    
    def set_user_relationship(self, user_id: str, relationship: str) -> None:
        """
        Set relationship type for a user.
        
        Args:
            user_id: User identifier
            relationship: Relationship type from RELATIONSHIP_MULTIPLIERS
            
        Raises:
            ValueError: If relationship type is invalid
        """
        if relationship not in self.RELATIONSHIP_MULTIPLIERS:
            valid_rels = list(self.RELATIONSHIP_MULTIPLIERS.keys())
            raise ValueError(f"relationship must be one of {valid_rels}, got {relationship}")
        
        self.user_relationships[user_id] = relationship
        logger.debug(f"Set relationship for user {user_id} to {relationship}")
    
    def get_user_trust(self, user_id: str) -> float:
        """Get trust level for a user (defaults to 0.5)"""
        return self.user_trust.get(user_id, 0.5)
    
    def get_contagion_history(self, 
                              limit: Optional[int] = None,
                              user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get contagion history with optional filtering.
        
        Args:
            limit: Maximum number of entries to return
            user_id: Filter by user
            
        Returns:
            List of contagion events as dictionaries
        """
        history = self.contagion_history.copy()
        
        if user_id:
            history = [e for e in history if e.user_id == user_id]
        
        if limit:
            if limit <= 0:
                return []
            history = history[-limit:]
        
        return [e.to_dict() for e in history]
    
    def _calculate_susceptibility(self, user_id: str, context: Optional[Dict]) -> float:
        """
        Calculate current susceptibility to emotional contagion.
        
        Factors:
        - Base susceptibility (personality)
        - Trust in user
        - Current emotional state
        - Context type
        - Relationship with user
        """
        susceptibility = self.base_susceptibility
        
        # Adjust for personality
        susceptibility *= (1 + (self.personality['empathy_level'] - 0.5) * 0.5)
        susceptibility *= (1 - (self.personality['emotional_boundaries'] - 0.5) * 0.4)
        
        # Adjust for trust in user
        trust = self.user_trust.get(user_id, 0.5)
        susceptibility *= (0.7 + 0.6 * trust)
        
        # Adjust for relationship if available
        relationship = None
        if context and 'relationship' in context:
            relationship = context['relationship']
        elif user_id in self.user_relationships:
            relationship = self.user_relationships[user_id]
        
        if relationship:
            multiplier = self.RELATIONSHIP_MULTIPLIERS.get(relationship, 1.0)
            susceptibility *= multiplier
        
        # Current mood affects susceptibility
        if self.emotional_state and hasattr(self.emotional_state, 'get_state'):
            try:
                current_state = self.emotional_state.get_state()
                mood = current_state.get('dominant', 'neutral')
                
                # Some moods make her more susceptible
                if mood in ['sadness', 'nostalgic', 'pensive']:
                    susceptibility *= (1 + self.personality['mood_susceptibility'] * 0.3)
                elif mood in ['anger', 'disdainful', 'wary']:
                    susceptibility *= (1 - self.personality['mood_susceptibility'] * 0.2)
            except Exception as e:
                logger.warning(f"Failed to get emotional state: {e}")
        
        # Professional detachment reduces susceptibility
        if context:
            formality = context.get('formality', 0.5)
            if formality > 0.6:
                detachment_factor = self.personality['professional_detachment']
                susceptibility *= (1 - detachment_factor * 0.3 * (formality - 0.5) * 2)
        
        return max(0.1, min(1.0, susceptibility))
    
    def _get_regulation_factor(self) -> float:
        """Get numerical factor for current regulation level"""
        factors = {
            ContagionRegulation.FULL: 1.0,
            ContagionRegulation.MODERATE: 0.6,
            ContagionRegulation.MINIMAL: 0.2,
            ContagionRegulation.BLOCKED: 0.0
        }
        return factors.get(self.current_regulation, 0.5)
    
    def _apply_contagion(self, 
                         emotion: str, 
                         intensity: float,
                         context: Optional[Dict]) -> Dict[str, Any]:
        """Apply contagious emotion to Wednesday's emotional state"""
        if not self.emotional_state:
            return {}
        
        if not hasattr(self.emotional_state, 'update'):
            logger.warning("Emotional state has no update method")
            return {}
        
        # Create stimulus for emotional state
        stimulus = {emotion: intensity}
        
        # Add small valence shift based on emotion
        if emotion in ['joy', 'dark_amusement', 'satisfied', 'curiosity']:
            stimulus['valence'] = intensity * 0.3
        elif emotion in ['sadness', 'anger', 'fear', 'disgust', 'wary']:
            stimulus['valence'] = -intensity * 0.3
        
        # Apply to emotional state
        try:
            return self.emotional_state.update(stimulus, context)
        except Exception as e:
            logger.error(f"Failed to apply contagion to emotional state: {e}")
            return {}
    
    def _add_to_history(self, event: ContagionEvent) -> None:
        """Add contagion event to history"""
        self.contagion_history.append(event)
        
        if len(self.contagion_history) > self.max_history_size:
            self.contagion_history.pop(0)
    
    def reset_regulation(self) -> None:
        """Reset regulation to default (MODERATE)"""
        self.current_regulation = ContagionRegulation.MODERATE
        logger.info("Contagion regulation reset to MODERATE")
    
    def reset_user_data(self, user_id: Optional[str] = None) -> None:
        """
        Reset trust and relationship data for a user or all users.
        
        Args:
            user_id: Optional specific user to reset
        """
        if user_id:
            self.user_trust.pop(user_id, None)
            self.user_relationships.pop(user_id, None)
            logger.info(f"Reset contagion data for user {user_id}")
        else:
            self.user_trust.clear()
            self.user_relationships.clear()
            logger.info("Reset all user contagion data")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get contagion statistics"""
        if not self.contagion_history:
            return {'total_events': 0}
        
        recent = self.contagion_history[-20:] if len(self.contagion_history) > 20 else self.contagion_history
        
        emotion_counts = {}
        user_counts = {}
        
        for event in recent:
            emotion = event.transferred_emotion
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            user = event.user_id
            user_counts[user] = user_counts.get(user, 0) + 1
        
        avg_factor = sum(e.contagion_factor for e in recent) / len(recent)
        
        return {
            'total_events': len(self.contagion_history),
            'recent_events': len(recent),
            'average_contagion_factor': round(avg_factor, 3),
            'common_emotions': dict(sorted(emotion_counts.items(), 
                                          key=lambda x: x[1], reverse=True)[:3]),
            'active_users': len(user_counts),
            'current_regulation': self.current_regulation.value,
            'base_susceptibility': round(self.base_susceptibility, 3),
            'users_tracked': len(self.user_trust)
        }
    
    def __repr__(self) -> str:
        return (f"EmotionalContagion(regulation={self.current_regulation.value}, "
                f"users={len(self.user_trust)}, events={len(self.contagion_history)})")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("=== Emotional Contagion Test ===\n")
    
    # Mock emotional state
    class MockEmotionalState:
        def __init__(self):
            self.state = {'dominant': 'neutral', 'emotions': {}}
        
        def get_state(self):
            return self.state
        
        def update(self, stimulus, context):
            emotion = list(stimulus.keys())[0] if stimulus else 'neutral'
            intensity = list(stimulus.values())[0] if stimulus else 0
            self.state['dominant'] = emotion
            self.state['emotions'] = stimulus
            return {'updated': True, 'new_state': emotion, 'intensity': intensity}
    
    # Create contagion system
    emotional_state = MockEmotionalState()
    contagion = EmotionalContagion(
        emotional_state=emotional_state,
        personality={
            'empathy_level': 0.5,
            'emotional_boundaries': 0.8,
            'trust_sensitivity': 0.6,
            'professional_detachment': 0.7,
            'mood_susceptibility': 0.4
        }
    )
    
    # Set trust and relationship for test user
    contagion.set_user_trust("test_user", 0.7)
    contagion.set_user_relationship("test_user", "friend")
    
    # Test different emotions
    test_emotions = [
        ('sad', 0.8, "User is very sad"),
        ('angry', 0.7, "User is frustrated"),
        ('happy', 0.9, "User is excited"),
        ('fearful', 0.6, "User is worried"),
        ('amused', 0.8, "User finds something funny"),
        ('unknown_emotion', 0.8, "Unknown emotion (should default to neutral)"),
    ]
    
    for emotion, intensity, desc in test_emotions:
        print(f"\n--- Scenario: {desc} ---")
        print(f"User emotion: {emotion} (intensity: {intensity})")
        
        result = contagion.catch_emotion(
            user_emotion=emotion,
            intensity=intensity,
            user_id="test_user",
            context={'relationship': 'friend', 'formality': 0.3}
        )
        
        if result:
            print(f"Contagion factor: {result['contagion_factor']:.2f}")
            print(f"Transferred: {result['transferred_emotion']} "
                  f"({result['transferred_intensity']:.2f})")
            print(f"Susceptibility: {result['susceptibility']:.2f}")
            print(f"Regulation: {result['regulation']}")
        else:
            print("No contagion occurred")
    
    # Test regulation in different contexts
    print("\n--- Regulation Test ---")
    
    contexts = [
        {'formality': 0.8, 'context_type': 'professional'},
        {'formality': 0.3, 'relationship': 'close_friend'},
        {'formality': 0.5, 'context_type': 'casual'},
        {'emergency': True, 'formality': 0.1},
    ]
    
    for context in contexts:
        contagion.regulate_contagion(context)
        print(f"Context: {context} -> Regulation: {contagion.current_regulation.value}")
    
    # Test with a stranger (low trust)
    print("\n--- Stranger Test ---")
    contagion.set_user_trust("stranger", 0.2)
    contagion.set_user_relationship("stranger", "stranger")
    
    result = contagion.catch_emotion(
        user_emotion='sad',
        intensity=0.8,
        user_id="stranger",
        context={'relationship': 'stranger'}
    )
    
    if result:
        print(f"With stranger: {result['transferred_intensity']:.2f} transferred")
    else:
        print("No contagion from stranger")
    
    # Show statistics
    print("\n--- Contagion Statistics ---")
    stats = contagion.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n=== Test Complete ===")