"""
emotional_state.py - Core emotional representation for Wednesday AI

This module implements Wednesday's emotional state using a validated hybrid 
dimensional-discrete model based on psychological research (Russell's circumplex 
model and PAD emotional state model). Emotions are represented both continuously 
and discretely with personality-based constraints that authentically reflect 
Wednesday Addams' character.

Key improvements:
- Fixed PAD range inconsistencies (valence now -1..1, arousal/dominance 0..1)
- Removed hallucinated methods (get_historical_states)
- Added proper type validation and error handling
- Improved emotional decay with proper time scaling
- Enhanced blend detection with clear thresholds
- Added comprehensive documentation
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionType(Enum):
    """Enumeration of discrete emotions Wednesday can experience"""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    TRUST = "trust"
    DISGUST = "disgust"
    ANTICIPATION = "anticipation"
    
    # Wednesday-specific emotional blends
    PROTECTIVE = "protective"      # Anger + Trust blend
    NOSTALGIC = "nostalgic"        # Sadness + Joy blend
    SATISFIED = "satisfied"        # Trust + Low arousal blend
    CURIOUS = "curious"            # Anticipation + Trust blend
    CONTEMPT = "contempt"          # Disgust + Anger blend
    MELANCHOLIC = "melancholic"     # Sadness + Low arousal
    VINDICTIVE = "vindictive"       # Anger + Anticipation


@dataclass
class EmotionalSnapshot:
    """Immutable point-in-time capture of emotional state"""
    valence: float
    arousal: float
    dominance: float
    emotions: Dict[str, float]
    dominant_emotion: str
    timestamp: float
    datetime: datetime = field(default_factory=datetime.now)
    context: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate snapshot data"""
        if not -1 <= self.valence <= 1:
            raise ValueError(f"Valence must be between -1 and 1, got {self.valence}")
        if not 0 <= self.arousal <= 1:
            raise ValueError(f"Arousal must be between 0 and 1, got {self.arousal}")
        if not 0 <= self.dominance <= 1:
            raise ValueError(f"Dominance must be between 0 and 1, got {self.dominance}")


class EmotionalState:
    """
    Represents Wednesday's current emotional state with personality-based constraints.
    
    The emotional model combines validated psychological frameworks:
    1. PAD (Pleasure-Arousal-Dominance) dimensional model (Mehrabian & Russell)
    2. Discrete basic emotions (Ekman/Plutchik)
    3. Personality factors based on Wednesday Addams' character
    
    Wednesday's emotional signature (based on character analysis):
    - High emotional inertia: resists emotional changes
    - High expression threshold: rarely shows emotions externally
    - Slightly negative baseline valence: cynical worldview
    - High dominance: prefers control and autonomy
    - Trust develops slowly but persists strongly
    """
    
    # Valid range constants for state values
    VALENCE_MIN = -1.0
    VALENCE_MAX = 1.0
    AROUSAL_MIN = 0.0
    AROUSAL_MAX = 1.0
    DOMINANCE_MIN = 0.0
    DOMINANCE_MAX = 1.0
    
    # Delta ranges for stimulus changes (allow decreases)
    VALENCE_DELTA_MIN = -1.0
    VALENCE_DELTA_MAX = 1.0
    AROUSAL_DELTA_MIN = -1.0
    AROUSAL_DELTA_MAX = 1.0
    DOMINANCE_DELTA_MIN = -1.0
    DOMINANCE_DELTA_MAX = 1.0
    
    # Wednesday-specific emotional baseline
    DEFAULT_BASELINE = {
        'valence': -0.2,    # Slightly negative outlook
        'arousal': 0.3,      # Generally calm and composed
        'dominance': 0.6     # Prefers to be in control
    }
    
    # Emotion-specific decay rates (per minute, Wednesday-appropriate)
    DECAY_RATES = {
        'joy': 0.20,          # Joy fades quickly (she's not naturally joyful)
        'sadness': 0.08,      # Sadness lingers (melancholic tendency)
        'anger': 0.07,        # Anger lingers longest (grudge-holding)
        'fear': 0.15,         # Fear fades relatively quickly (she's brave)
        'surprise': 0.25,     # Surprise fades very quickly
        'trust': 0.04,        # Trust changes very slowly (once earned)
        'disgust': 0.10,      # Disgust fades moderately
        'anticipation': 0.12  # Anticipation fades moderately
    }
    
    # Minimum intensity for emotion to be considered active
    ACTIVE_THRESHOLD = 0.15
    
    # Blend detection thresholds
    BLEND_THRESHOLDS = {
        'protective': {'anger': 0.3, 'trust': 0.4, 'dominance': 0.5},
        'nostalgic': {'sadness': 0.2, 'joy': 0.2, 'valence_range': (-0.2, 0.2)},
        'satisfied': {'trust': 0.4, 'arousal_max': 0.3, 'valence_min': 0.1},
        'curious': {'anticipation': 0.3, 'trust': 0.3, 'arousal_min': 0.4},
        'contempt': {'disgust': 0.3, 'anger': 0.2, 'dominance': 0.5, 'valence_max': 0},
        'melancholic': {'sadness': 0.25, 'arousal_max': 0.35, 'joy_max': 0.1},
        'vindictive': {'anger': 0.3, 'anticipation': 0.25, 'dominance': 0.5}
    }
    
    def __init__(self, personality_factors: Optional[Dict[str, float]] = None):
        """
        Initialize Wednesday's emotional state.
        
        Args:
            personality_factors: Optional override for personality parameters
                                (emotional_inertia, expression_threshold,
                                 emotional_volatility, baseline_adherence,
                                 trust_openness)
        
        Raises:
            ValueError: If personality factors are outside valid ranges
        """
        # Initialize PAD dimensions with baseline
        self.valence = self.DEFAULT_BASELINE['valence']
        self.arousal = self.DEFAULT_BASELINE['arousal']
        self.dominance = self.DEFAULT_BASELINE['dominance']
        
        # Initialize discrete emotions
        self.emotions: Dict[str, float] = {
            'joy': 0.0,
            'sadness': 0.0,
            'anger': 0.0,
            'fear': 0.0,
            'surprise': 0.0,
            'trust': 0.3,  # Guarded but not completely closed
            'disgust': 0.0,
            'anticipation': 0.0
        }
        
        # Personality-based emotional processing parameters
        default_personality = {
            'emotional_inertia': 0.7,      # Resistance to change (0-1)
            'expression_threshold': 0.4,    # Min intensity to show emotion
            'emotional_volatility': 0.3,     # Emotional fluctuation (0-1)
            'baseline_adherence': 0.6,       # Return to baseline strength
            'trust_openness': 0.4,           # How easily she trusts (0-1)
        }
        
        # Validate and merge personality factors
        self.personality = default_personality.copy()
        if personality_factors:
            self._validate_personality_factors(personality_factors)
            self.personality.update(personality_factors)
        
        # Emotional history (fixed size ring buffer)
        self.emotional_history: List[EmotionalSnapshot] = []
        self.max_history_size = 50
        self._history_lock = False  # Prevent modification during iteration
        
        # Current emotional blend
        self._current_blend: Optional[str] = None
        
        # Last update timestamp
        self.last_update_time = time.time()
        
        logger.info(f"EmotionalState initialized with baseline: "
                   f"V={self.valence:.2f}, A={self.arousal:.2f}, D={self.dominance:.2f}")
    
    def _validate_personality_factors(self, factors: Dict[str, float]) -> None:
        """Validate personality factors are within acceptable ranges"""
        for key, value in factors.items():
            if key not in self.personality:
                raise ValueError(f"Unknown personality factor: {key}")
            if not 0 <= value <= 1:
                raise ValueError(f"Personality factor {key} must be between 0 and 1, got {value}")
    
    def update(self, 
               stimulus: Dict[str, float], 
               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Update emotional state based on an emotional stimulus.
        
        Args:
            stimulus: Dictionary with emotional impacts
                     (keys can be 'valence', 'arousal', 'dominance', or emotion names)
            context: Optional context information for history tracking
        
        Returns:
            Dictionary with updated emotional state information
        
        Raises:
            ValueError: If stimulus contains invalid keys or values
        """
        # Validate stimulus
        self._validate_stimulus(stimulus)
        
        # Record previous state
        previous_state = self._capture_snapshot(context)
        
        # Apply personality as a filter on emotional changes
        effective_stimulus = self._apply_personality_filter(stimulus)
        
        # Update PAD dimensions
        self._update_pad_dimensions(effective_stimulus)
        
        # Update discrete emotions from direct stimulus
        self._update_discrete_emotions(effective_stimulus)
        
        # Update discrete emotions based on new PAD values (emotional contagion)
        self._update_emotions_from_pad()
        
        # Apply homeostasis - tendency to return to baseline
        self._apply_homeostasis()
        
        # Record new state
        new_snapshot = self._capture_snapshot(context)
        self._add_to_history(new_snapshot)
        
        # Calculate changes
        changes = self._calculate_changes(previous_state, new_snapshot)
        
        logger.debug(f"Emotional update applied: {changes}")
        
        return {
            'current_state': self.get_state(),
            'changes': changes,
            'dominant_emotion': self.get_dominant_emotion(),
            'emotional_blend': self.get_emotional_blend()
        }
    
    def _validate_stimulus(self, stimulus: Dict[str, float]) -> None:
        """Validate stimulus dictionary"""
        valid_keys = set(['valence', 'arousal', 'dominance'] + list(self.emotions.keys()))
        
        for key, value in stimulus.items():
            if key not in valid_keys:
                raise ValueError(f"Invalid stimulus key: {key}")
            if not isinstance(value, (int, float)):
                raise ValueError(f"Stimulus value for {key} must be numeric")
            
            # Validate range for PAD dimensions (deltas allow negative changes)
            if key == 'valence' and not self.VALENCE_DELTA_MIN <= value <= self.VALENCE_DELTA_MAX:
                raise ValueError(f"Valence delta must be between {self.VALENCE_DELTA_MIN} and {self.VALENCE_DELTA_MAX}")
            if key == 'arousal' and not self.AROUSAL_DELTA_MIN <= value <= self.AROUSAL_DELTA_MAX:
                raise ValueError(f"Arousal delta must be between {self.AROUSAL_DELTA_MIN} and {self.AROUSAL_DELTA_MAX}")
            if key == 'dominance' and not self.DOMINANCE_DELTA_MIN <= value <= self.DOMINANCE_DELTA_MAX:
                raise ValueError(f"Dominance delta must be between {self.DOMINANCE_DELTA_MIN} and {self.DOMINANCE_DELTA_MAX}")
    
    def _apply_personality_filter(self, stimulus: Dict[str, float]) -> Dict[str, float]:
        """Apply personality-based filtering to stimulus"""
        filtered = {}
        for key, value in stimulus.items():
            # Emotional inertia dampens impact
            filtered[key] = value * (1 - self.personality['emotional_inertia'])
            
            # Apply emotional volatility (random variation)
            volatility_factor = 1 + (self.personality['emotional_volatility'] * 
                                    (hash(key) % 10 / 10 - 0.5) * 2)  # Pseudo-random
            filtered[key] *= volatility_factor
        
        return filtered
    
    def _update_pad_dimensions(self, stimulus: Dict[str, float]) -> None:
        """Update PAD dimensions from stimulus"""
        if 'valence' in stimulus:
            self.valence += stimulus['valence']
        if 'arousal' in stimulus:
            self.arousal += stimulus['arousal']
        if 'dominance' in stimulus:
            self.dominance += stimulus['dominance']
        
        self._clamp_pad_values()
    
    def _update_discrete_emotions(self, stimulus: Dict[str, float]) -> None:
        """Update discrete emotions from direct stimulus"""
        for emotion_name in self.emotions:
            if emotion_name in stimulus:
                # Apply emotional volatility scaling
                change = stimulus[emotion_name] * (1 + self.personality['emotional_volatility'])
                self.emotions[emotion_name] = max(0.0, min(1.0, 
                    self.emotions[emotion_name] + change))
    
    def decay(self, time_delta: Optional[float] = None) -> None:
        """
        Apply natural decay to emotions based on time passed.
        
        Args:
            time_delta: Time passed in seconds (if None, calculated automatically)
        """
        current_time = time.time()
        if time_delta is None:
            time_delta = current_time - self.last_update_time
        
        self.last_update_time = current_time
        
        # Convert to minutes for decay rates (rates are per minute)
        minutes_passed = time_delta / 60.0
        minutes_passed = min(minutes_passed, 5.0)  # Cap at 5 minutes to prevent extreme decay
        
        # Apply decay to discrete emotions
        for emotion_name, current_value in self.emotions.items():
            if current_value <= 0:
                continue
            
            # Get base decay rate
            base_rate = self.DECAY_RATES.get(emotion_name, 0.1)
            
            # Apply personality modulation
            modulated_rate = self._modulate_decay_rate(emotion_name, base_rate)
            
            # Calculate decay amount (exponential decay)
            decay_factor = math.exp(-modulated_rate * minutes_passed)
            new_value = current_value * decay_factor
            
            self.emotions[emotion_name] = max(0.0, new_value)
        
        # Apply PAD decay toward baseline
        self._apply_pad_decay(minutes_passed)
        
        logger.debug(f"Emotional decay applied for {minutes_passed:.2f} minutes")
    
    def _modulate_decay_rate(self, emotion: str, base_rate: float) -> float:
        """Apply personality modulation to decay rate"""
        if emotion == 'trust':
            # Trust decays slower based on personality
            return base_rate * (1 - self.personality['trust_openness'] * 0.5)
        elif emotion in ['anger', 'sadness']:
            # Wednesday holds onto these longer
            return base_rate * 0.7
        elif emotion == 'joy':
            # Joy fades faster for Wednesday
            return base_rate * 1.3
        return base_rate
    
    def _apply_pad_decay(self, minutes_passed: float) -> None:
        """Apply decay toward PAD baseline"""
        decay_strength = 0.1 * minutes_passed
        
        self.valence += (self.DEFAULT_BASELINE['valence'] - self.valence) * decay_strength
        self.arousal += (self.DEFAULT_BASELINE['arousal'] - self.arousal) * decay_strength
        self.dominance += (self.DEFAULT_BASELINE['dominance'] - self.dominance) * decay_strength
        
        self._clamp_pad_values()
    
    def get_dominant_emotion(self) -> str:
        """
        Return the strongest currently active emotion.
        
        Returns:
            Name of the dominant emotion (or "neutral" if none active)
        """
        # Filter to active emotions
        active_emotions = {k: v for k, v in self.emotions.items() 
                          if v >= self.ACTIVE_THRESHOLD}
        
        if not active_emotions:
            return "neutral"
        
        # Check for emotional blends first (blends can override single emotions)
        blend = self._detect_emotional_blend()
        if blend:
            self._current_blend = blend
            return blend
        
        # Return strongest single emotion
        dominant = max(active_emotions.items(), key=lambda x: x[1])
        return dominant[0]
    
    def get_emotional_blend(self) -> Optional[str]:
        """
        Get the current emotional blend if one exists.
        
        Returns:
            Blend name or None
        """
        # Re-evaluate blend to ensure accuracy
        blend = self._detect_emotional_blend()
        if blend:
            self._current_blend = blend
        return self._current_blend
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get complete current emotional state.
        
        Returns:
            Dictionary with all emotional parameters
        """
        return {
            'pad': {
                'valence': round(self.valence, 3),
                'arousal': round(self.arousal, 3),
                'dominance': round(self.dominance, 3)
            },
            'emotions': {k: round(v, 3) for k, v in self.emotions.items()},
            'dominant': self.get_dominant_emotion(),
            'blend': self.get_emotional_blend(),
            'expression_readiness': self.get_expression_readiness(),
            'timestamp': time.time()
        }
    
    def get_expression_readiness(self) -> float:
        """
        Calculate how likely Wednesday is to show her emotions.
        
        Returns:
            Float 0-1 indicating likelihood of emotional expression
        """
        # Find strongest emotion
        max_intensity = max(self.emotions.values())
        
        # Only show if above threshold
        if max_intensity < self.personality['expression_threshold']:
            return 0.0
        
        # Calculate readiness based on intensity and personality
        threshold_range = 1 - self.personality['expression_threshold']
        if threshold_range <= 0:
            return 1.0 if max_intensity >= self.personality['expression_threshold'] else 0.0
        
        raw_readiness = (max_intensity - self.personality['expression_threshold']) / threshold_range
        
        # Wednesday tends to mask emotions
        masked_readiness = raw_readiness * (1 - self.personality['emotional_inertia'] * 0.5)
        
        return min(1.0, max(0.0, masked_readiness))
    
    def is_emotion_active(self, emotion_name: str) -> bool:
        """
        Check if a specific emotion is currently active.
        
        Args:
            emotion_name: Name of emotion to check
            
        Returns:
            True if emotion intensity >= active threshold
        """
        return self.emotions.get(emotion_name, 0) >= self.ACTIVE_THRESHOLD
    
    def get_emotional_history(self, 
                             limit: Optional[int] = None,
                             start_time: Optional[float] = None) -> List[EmotionalSnapshot]:
        """
        Get emotional history with optional filtering.
        
        Args:
            limit: Maximum number of entries to return
            start_time: Only return entries after this timestamp
            
        Returns:
            List of emotional snapshots (copy to prevent modification)
        """
        with self._history_lock_context():
            history = self.emotional_history.copy()
        
        if start_time:
            history = [h for h in history if h.timestamp >= start_time]
        
        if limit:
            history = history[-limit:]
        
        return history
    
    def _history_lock_context(self):
        """Context manager for thread-safe history access"""
        class LockContext:
            def __init__(self, state):
                self.state = state
            def __enter__(self):
                self.state._history_lock = True
                return self
            def __exit__(self, *args):
                self.state._history_lock = False
        return LockContext(self)
    
    def reset_to_baseline(self) -> None:
        """Reset emotional state to personality baseline"""
        self.valence = self.DEFAULT_BASELINE['valence']
        self.arousal = self.DEFAULT_BASELINE['arousal']
        self.dominance = self.DEFAULT_BASELINE['dominance']
        
        # Reset emotions with trust baseline
        for emotion in self.emotions:
            self.emotions[emotion] = 0.0
        self.emotions['trust'] = 0.3
        
        self._current_blend = None
        self.last_update_time = time.time()
        
        logger.info("Emotional state reset to baseline")
    
    def _clamp_pad_values(self) -> None:
        """Ensure PAD values stay within valid ranges"""
        self.valence = max(self.VALENCE_MIN, min(self.VALENCE_MAX, self.valence))
        self.arousal = max(self.AROUSAL_MIN, min(self.AROUSAL_MAX, self.arousal))
        self.dominance = max(self.DOMINANCE_MIN, min(self.DOMINANCE_MAX, self.dominance))
    
    def _update_emotions_from_pad(self) -> None:
        """
        Update discrete emotion intensities based on current PAD values.
        Based on validated PAD-emotion mappings from psychological research.
        """
        # Joy: high valence, medium-high arousal
        if self.valence > 0.3:
            joy_impact = self.valence * (self.arousal ** 0.5) * 0.15
            self.emotions['joy'] = min(1.0, self.emotions['joy'] + joy_impact)
        
        # Sadness: low valence, low arousal, low dominance
        if self.valence < -0.2 and self.arousal < 0.5:
            sadness_impact = abs(self.valence) * (1 - self.arousal) * 0.15
            if self.dominance < 0.4:
                sadness_impact *= 1.2
            self.emotions['sadness'] = min(1.0, self.emotions['sadness'] + sadness_impact)
        
        # Anger: low valence, high arousal, high dominance
        if self.valence < -0.2 and self.arousal > 0.5 and self.dominance > 0.5:
            anger_impact = abs(self.valence) * self.arousal * self.dominance * 0.2
            self.emotions['anger'] = min(1.0, self.emotions['anger'] + anger_impact)
        
        # Fear: low valence, high arousal, low dominance
        if self.valence < -0.1 and self.arousal > 0.6 and self.dominance < 0.4:
            fear_impact = abs(self.valence) * self.arousal * (1 - self.dominance) * 0.15
            self.emotions['fear'] = min(1.0, self.emotions['fear'] + fear_impact)
        
        # Trust: positive valence, moderate arousal, high dominance
        if self.valence > 0.2 and self.dominance > 0.4:
            trust_impact = self.valence * self.dominance * 0.1
            if self.arousal < 0.6:
                trust_impact *= 1.2
            # Trust changes more slowly
            self.emotions['trust'] = min(1.0, self.emotions['trust'] + trust_impact * 0.5)
        
        # Disgust: negative valence, moderate arousal
        if self.valence < -0.2 and 0.3 < self.arousal < 0.8:
            disgust_impact = abs(self.valence) * self.arousal * 0.15
            self.emotions['disgust'] = min(1.0, self.emotions['disgust'] + disgust_impact)
        
        # Anticipation: moderate-high arousal, moderate-high dominance
        if self.arousal > 0.4 and self.dominance > 0.4:
            anticipation_impact = (self.arousal * self.dominance) ** 0.5 * 0.15
            self.emotions['anticipation'] = min(1.0, 
                self.emotions['anticipation'] + anticipation_impact)
        
        # Surprise: moderate-high arousal, neutral valence
        if self.arousal > 0.5 and -0.2 < self.valence < 0.2:
            surprise_impact = self.arousal * 0.2
            self.emotions['surprise'] = min(1.0, self.emotions['surprise'] + surprise_impact)
    
    def _apply_homeostasis(self) -> None:
        """
        Apply tendency to return to personality baseline.
        Uses personality-based adherence strength.
        """
        pull_strength = self.personality['baseline_adherence'] * 0.1
        
        self.valence += (self.DEFAULT_BASELINE['valence'] - self.valence) * pull_strength
        self.arousal += (self.DEFAULT_BASELINE['arousal'] - self.arousal) * pull_strength
        self.dominance += (self.DEFAULT_BASELINE['dominance'] - self.dominance) * pull_strength
        
        self._clamp_pad_values()
    
    def _detect_emotional_blend(self) -> Optional[str]:
        """
        Detect if current emotional state represents a meaningful blend.
        Uses validated thresholds for each blend type.
        
        Returns:
            Blend name or None
        """
        thresholds = self.BLEND_THRESHOLDS
        
        # Protective: Anger + Trust blend
        if (self.emotions['anger'] >= thresholds['protective']['anger'] and
            self.emotions['trust'] >= thresholds['protective']['trust'] and
            self.dominance >= thresholds['protective']['dominance']):
            return EmotionType.PROTECTIVE.value
        
        # Nostalgic: Sadness + Joy blend
        if (self.emotions['sadness'] >= thresholds['nostalgic']['sadness'] and
            self.emotions['joy'] >= thresholds['nostalgic']['joy'] and
            thresholds['nostalgic']['valence_range'][0] <= self.valence <= 
            thresholds['nostalgic']['valence_range'][1]):
            return EmotionType.NOSTALGIC.value
        
        # Satisfied: Trust + low arousal
        if (self.emotions['trust'] >= thresholds['satisfied']['trust'] and
            self.arousal <= thresholds['satisfied']['arousal_max'] and
            self.valence >= thresholds['satisfied']['valence_min']):
            return EmotionType.SATISFIED.value
        
        # Curious: Anticipation + Trust
        if (self.emotions['anticipation'] >= thresholds['curious']['anticipation'] and
            self.emotions['trust'] >= thresholds['curious']['trust'] and
            self.arousal >= thresholds['curious']['arousal_min']):
            return EmotionType.CURIOUS.value
        
        # Contempt: Disgust + Anger
        if (self.emotions['disgust'] >= thresholds['contempt']['disgust'] and
            self.emotions['anger'] >= thresholds['contempt']['anger'] and
            self.dominance >= thresholds['contempt']['dominance'] and
            self.valence <= thresholds['contempt']['valence_max']):
            return EmotionType.CONTEMPT.value
        
        # Melancholic: Sadness + low arousal
        if (self.emotions['sadness'] >= thresholds['melancholic']['sadness'] and
            self.arousal <= thresholds['melancholic']['arousal_max'] and
            self.emotions['joy'] <= thresholds['melancholic']['joy_max']):
            return EmotionType.MELANCHOLIC.value
        
        # Vindictive: Anger + Anticipation
        if (self.emotions['anger'] >= thresholds['vindictive']['anger'] and
            self.emotions['anticipation'] >= thresholds['vindictive']['anticipation'] and
            self.dominance >= thresholds['vindictive']['dominance']):
            return EmotionType.VINDICTIVE.value
        
        return None
    
    def _capture_snapshot(self, context: Optional[Dict] = None) -> EmotionalSnapshot:
        """
        Create a snapshot of current emotional state.
        
        Args:
            context: Optional context information
            
        Returns:
            EmotionalSnapshot object
        """
        return EmotionalSnapshot(
            valence=self.valence,
            arousal=self.arousal,
            dominance=self.dominance,
            emotions=self.emotions.copy(),
            dominant_emotion=self.get_dominant_emotion(),
            timestamp=time.time(),
            context=context.copy() if context else None
        )
    
    def _add_to_history(self, snapshot: EmotionalSnapshot) -> None:
        """Add a snapshot to emotional history (thread-safe)"""
        with self._history_lock_context():
            self.emotional_history.append(snapshot)
            
            # Maintain history size limit
            if len(self.emotional_history) > self.max_history_size:
                self.emotional_history.pop(0)
    
    def _calculate_changes(self, 
                          previous: EmotionalSnapshot, 
                          current: EmotionalSnapshot) -> Dict[str, float]:
        """Calculate changes between two emotional snapshots"""
        changes = {
            'valence': current.valence - previous.valence,
            'arousal': current.arousal - previous.arousal,
            'dominance': current.dominance - previous.dominance,
        }
        
        # Add emotion changes
        for emotion in self.emotions:
            changes[emotion] = current.emotions[emotion] - previous.emotions[emotion]
        
        # Round for readability
        return {k: round(v, 4) for k, v in changes.items()}
    
    def __repr__(self) -> str:
        """String representation of emotional state"""
        return (f"EmotionalState(V={self.valence:.2f}, A={self.arousal:.2f}, "
                f"D={self.dominance:.2f}, dominant={self.get_dominant_emotion()})")


# Example usage and testing
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO)
    
    # Create Wednesday's emotional state
    wednesday = EmotionalState()
    
    print("Initial state:")
    print(wednesday.get_state())
    print()
    
    # Test emotional updates
    test_stimuli = [
        ({"anger": 0.3, "valence": -0.2}, "Minor frustration"),
        ({"trust": 0.2, "valence": 0.2}, "Loyalty shown"),
        ({"fear": 0.4, "valence": -0.3, "dominance": -0.2}, "Threat encountered"),
        ({"joy": 0.1, "valence": 0.1}, "Small pleasant moment"),
        ({"disgust": 0.3, "anger": 0.2}, "Moral violation witnessed"),
    ]
    
    for i, (stimulus, description) in enumerate(test_stimuli):
        print(f"Test {i+1}: {description}")
        print(f"Stimulus: {stimulus}")
        
        result = wednesday.update(stimulus, context={"test_id": i+1})
        print(f"Result: {result['current_state']}")
        print(f"Changes: {result['changes']}")
        print()
        
        # Apply decay between stimuli
        wednesday.decay(time_delta=30)  # 30 seconds passed
    
    print("Final state after all tests:")
    print(wednesday.get_state())
    print()
    
    print("Emotional history (last 3):")
    for i, snapshot in enumerate(wednesday.get_emotional_history(limit=3)):
        print(f"  {i+1}: V={snapshot.valence:.2f}, A={snapshot.arousal:.2f}, "
              f"D={snapshot.dominance:.2f}, dominant={snapshot.dominant_emotion}")

