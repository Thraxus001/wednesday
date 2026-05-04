"""
mood_engine.py - Longer-term mood management for Wednesday AI

This module implements a mood system that represents Wednesday's sustained
emotional background state. Unlike short-term emotions which fluctuate rapidly,
mood provides a stable emotional context that colors all experiences.

Key improvements:
- Fixed time-based decay with proper scaling
- Removed numpy dependency for better portability
- Added proper validation and error handling
- Improved mood transition logic
- Enhanced documentation and type safety
"""

import time
import logging
import random
from typing import Dict, List, Optional, Tuple, Any, Deque
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)


class MoodType(Enum):
    """Enumeration of possible mood states"""
    # Basic moods
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    IRRITABLE = "irritable"
    REFLECTIVE = "reflective"
    
    # Wednesday-specific moods
    DARKLY_AMUSED = "darkly_amused"      # Finding dark humor in situations
    PENSIVE = "pensive"                   # Deep in thought, slightly melancholic
    WARY = "wary"                         # Guarded, suspicious
    SATISFIED = "satisfied"                # Content with how things are going
    DISDAINFUL = "disdainful"              # Superior, looking down on things
    CURIOUSLY_DETACHED = "curiously_detached"  # Interested but emotionally removed


@dataclass
class MoodState:
    """Represents a point-in-time mood state"""
    name: str
    intensity: float  # 0-1 scale
    persistence: float  # 0-1 scale, how long this mood tends to last
    valence_bias: float  # -1 to 1, how mood biases perception
    arousal_bias: float  # -1 to 1, how mood biases energy/alertness
    timestamp: float
    context: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate mood state parameters"""
        if not 0 <= self.intensity <= 1:
            raise ValueError(f"Intensity must be between 0 and 1, got {self.intensity}")
        if not 0 <= self.persistence <= 1:
            raise ValueError(f"Persistence must be between 0 and 1, got {self.persistence}")
        if not -1 <= self.valence_bias <= 1:
            raise ValueError(f"Valence bias must be between -1 and 1, got {self.valence_bias}")
        if not -1 <= self.arousal_bias <= 1:
            raise ValueError(f"Arousal bias must be between -1 and 1, got {self.arousal_bias}")


@dataclass
class EmotionalEvent:
    """Record of an emotional event that might affect mood"""
    emotion: str
    intensity: float
    valence: float
    arousal: float
    timestamp: float
    significance: float  # How personally significant this event was (0-1)
    context: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate emotional event parameters"""
        if not 0 <= self.intensity <= 1:
            raise ValueError(f"Intensity must be between 0 and 1, got {self.intensity}")
        if not -1 <= self.valence <= 1:
            raise ValueError(f"Valence must be between -1 and 1, got {self.valence}")
        if not 0 <= self.arousal <= 1:
            raise ValueError(f"Arousal must be between 0 and 1, got {self.arousal}")
        if not 0 <= self.significance <= 1:
            raise ValueError(f"Significance must be between 0 and 1, got {self.significance}")


class MoodEngine:
    """
    Manages Wednesday's longer-term mood state.
    
    Mood differs from emotion in:
    - Duration: Mood lasts longer (hours to days)
    - Intensity: Mood is generally less intense than emotions
    - Specificity: Mood is less tied to specific triggers
    - Bias: Mood colors how new experiences are interpreted
    
    Wednesday's mood tendencies:
    - Baseline slightly negative/neutral
    - Mood changes slowly (high persistence)
    - Tends toward reflective/pensive states
    - Dark humor emerges as a mood-congruent bias
    """
    
    # Default mood baselines (Wednesday-specific)
    DEFAULT_MOOD = {
        'name': 'neutral',
        'intensity': 0.3,
        'persistence': 0.7,  # High persistence = mood changes slowly
    }
    
    # Mood-specific biases
    MOOD_BIASES = {
        'neutral': {
            'valence_bias': 0.0,
            'arousal_bias': 0.0,
            'cognitive_bias': 'objective',
            'attention_bias': 'balanced',
            'description': 'Balanced, objective perspective'
        },
        'positive': {
            'valence_bias': 0.3,
            'arousal_bias': 0.2,
            'cognitive_bias': 'optimistic',
            'attention_bias': 'opportunities',
            'description': 'Optimistic, opportunity-focused'
        },
        'negative': {
            'valence_bias': -0.3,
            'arousal_bias': 0.1,
            'cognitive_bias': 'pessimistic',
            'attention_bias': 'threats',
            'description': 'Pessimistic, threat-focused'
        },
        'irritable': {
            'valence_bias': -0.4,
            'arousal_bias': 0.4,
            'cognitive_bias': 'hostile',
            'attention_bias': 'annoyances',
            'description': 'Easily annoyed, hostile interpretation'
        },
        'reflective': {
            'valence_bias': 0.0,
            'arousal_bias': -0.2,
            'cognitive_bias': 'analytical',
            'attention_bias': 'patterns',
            'description': 'Analytical, pattern-seeking'
        },
        'darkly_amused': {
            'valence_bias': 0.1,
            'arousal_bias': 0.2,
            'cognitive_bias': 'ironic',
            'attention_bias': 'absurdities',
            'description': 'Finds dark humor, ironic perspective'
        },
        'pensive': {
            'valence_bias': -0.1,
            'arousal_bias': -0.3,
            'cognitive_bias': 'introspective',
            'attention_bias': 'nuances',
            'description': 'Deep in thought, nuanced view'
        },
        'wary': {
            'valence_bias': -0.2,
            'arousal_bias': 0.3,
            'cognitive_bias': 'suspicious',
            'attention_bias': 'hidden_motives',
            'description': 'Suspicious, looking for hidden motives'
        },
        'satisfied': {
            'valence_bias': 0.2,
            'arousal_bias': -0.1,
            'cognitive_bias': 'appreciative',
            'attention_bias': 'what_works',
            'description': 'Appreciative, focused on what works'
        },
        'disdainful': {
            'valence_bias': -0.2,
            'arousal_bias': 0.1,
            'cognitive_bias': 'superior',
            'attention_bias': 'flaws',
            'description': 'Superior, focused on flaws'
        },
        'curiously_detached': {
            'valence_bias': 0.0,
            'arousal_bias': 0.1,
            'cognitive_bias': 'observational',
            'attention_bias': 'interesting_details',
            'description': 'Observational, interested but detached'
        }
    }
    
    # Mood transition thresholds
    MOOD_SHIFT_THRESHOLD = 0.3  # How much emotional accumulation needed to shift mood
    INTENSITY_DECAY_RATE = 0.1  # Per hour (scaled in update)
    
    # Valid mood names for quick lookup
    VALID_MOODS = set(MOOD_BIASES.keys())
    
    def __init__(self, personality: Optional[Dict[str, float]] = None):
        """
        Initialize the mood engine with personality influences.
        
        Args:
            personality: Personality parameters that influence mood tendencies
                       (baseline_mood, mood_volatility, mood_intensity_factor,
                        emotional_absorption, recovery_rate, dark_humor_tendency)
        
        Raises:
            ValueError: If personality parameters are invalid
        """
        # Personality influences on mood
        self.personality = self._validate_personality(personality or {
            'baseline_mood': 'neutral',
            'mood_volatility': 0.3,      # How easily mood changes (0-1)
            'mood_intensity_factor': 0.5,  # How intense moods tend to be
            'emotional_absorption': 0.4,   # How much emotions affect mood
            'recovery_rate': 0.2,          # How quickly mood returns to baseline
            'dark_humor_tendency': 0.7,    # Likelihood of darkly_amused mood
        })
        
        # Validate baseline mood
        if self.personality['baseline_mood'] not in self.VALID_MOODS:
            raise ValueError(f"Invalid baseline mood: {self.personality['baseline_mood']}")
        
        # Current mood state
        self.current_mood = self._create_initial_mood()
        
        # Emotional event tracking
        self.recent_emotions: Deque[EmotionalEvent] = deque(maxlen=50)
        self.emotional_accumulator = {
            'valence_sum': 0.0,
            'arousal_sum': 0.0,
            'event_count': 0,
            'significant_events': 0,
            'last_decay_time': time.time()
        }
        
        # Mood history
        self.mood_history: List[MoodState] = []
        self.max_history_size = 100
        
        # Current mood biases (cached for performance)
        self._current_biases = self.MOOD_BIASES[self.current_mood.name]
        
        # Last update timestamp
        self.last_update_time = time.time()
        
        logger.info(f"MoodEngine initialized with baseline mood: {self.current_mood.name}")
    
    def _validate_personality(self, personality: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean personality parameters"""
        validated = {}
        
        # Define expected types and ranges
        validators = {
            'baseline_mood': lambda x: x if x in self.VALID_MOODS else 'neutral',
            'mood_volatility': lambda x: max(0.0, min(1.0, float(x))),
            'mood_intensity_factor': lambda x: max(0.1, min(1.0, float(x))),
            'emotional_absorption': lambda x: max(0.0, min(1.0, float(x))),
            'recovery_rate': lambda x: max(0.0, min(1.0, float(x))),
            'dark_humor_tendency': lambda x: max(0.0, min(1.0, float(x)))
        }
        
        for key, validator in validators.items():
            if key in personality:
                validated[key] = validator(personality[key])
            else:
                # Use defaults from the provided personality dict
                validated[key] = validator({
                    'baseline_mood': 'neutral',
                    'mood_volatility': 0.3,
                    'mood_intensity_factor': 0.5,
                    'emotional_absorption': 0.4,
                    'recovery_rate': 0.2,
                    'dark_humor_tendency': 0.7
                }[key])
        
        return validated
    
    def _create_initial_mood(self) -> MoodState:
        """Create initial mood state"""
        baseline = self.personality['baseline_mood']
        biases = self.MOOD_BIASES[baseline]
        
        return MoodState(
            name=baseline,
            intensity=self.personality['mood_intensity_factor'],
            persistence=self.DEFAULT_MOOD['persistence'],
            valence_bias=biases['valence_bias'],
            arousal_bias=biases['arousal_bias'],
            timestamp=time.time()
        )
    
    def update_from_emotion(self, emotional_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update mood based on recent emotional state.
        
        Args:
            emotional_state: Current emotional state from EmotionalState
                           Must contain 'dominant', 'pad', and 'emotions' keys
        
        Returns:
            Dictionary with mood update information
        
        Raises:
            ValueError: If emotional_state is invalid
        """
        # Validate input
        self._validate_emotional_state(emotional_state)
        
        # Extract relevant emotional information
        try:
            current_emotion = {
                'dominant': emotional_state.get('dominant', 'neutral'),
                'valence': emotional_state['pad']['valence'],
                'arousal': emotional_state['pad']['arousal'],
                'intensity': max(emotional_state['emotions'].values()) if emotional_state['emotions'] else 0
            }
        except KeyError as e:
            raise ValueError(f"Missing required field in emotional_state: {e}")
        
        # Create emotional event
        event = EmotionalEvent(
            emotion=current_emotion['dominant'],
            intensity=current_emotion['intensity'],
            valence=current_emotion['valence'],
            arousal=current_emotion['arousal'],
            timestamp=time.time(),
            significance=self._calculate_significance(emotional_state)
        )
        
        # Add to recent emotions
        self.recent_emotions.append(event)
        
        # Update emotional accumulator with time-based decay
        self._update_accumulator(event)
        
        # Check if we should shift mood
        mood_shift = self._evaluate_mood_shift()
        
        # Apply natural mood drift
        self._apply_mood_drift()
        
        # Apply personality-based mood tendencies
        self._apply_personality_tendencies()
        
        # Update biases
        self._update_biases()
        
        # Record mood state
        self._record_current_mood()
        
        logger.debug(f"Mood updated: {self.current_mood.name} "
                    f"(intensity: {self.current_mood.intensity:.2f})")
        
        return {
            'current_mood': self.get_mood_info(),
            'mood_shift_occurred': mood_shift is not None,
            'mood_shift': mood_shift,
            'emotional_accumulation': {
                'average_valence': self._get_average_valence(),
                'event_count': self.emotional_accumulator['event_count']
            }
        }
    
    def _validate_emotional_state(self, state: Dict[str, Any]) -> None:
        """Validate emotional state dictionary"""
        required_keys = {'pad', 'emotions'}
        if not all(key in state for key in required_keys):
            raise ValueError(f"Emotional state must contain keys: {required_keys}")
        
        pad = state['pad']
        if not all(k in pad for k in ['valence', 'arousal', 'dominance']):
            raise ValueError("PAD must contain valence, arousal, and dominance")
    
    def get_mood_color(self, stimulus_valence: float) -> float:
        """
        Apply mood-congruent bias to perception of stimulus valence.
        
        Args:
            stimulus_valence: Raw valence of stimulus (-1 to 1)
            
        Returns:
            Mood-biased valence perception (-1 to 1)
        """
        # Validate input
        if not -1 <= stimulus_valence <= 1:
            raise ValueError(f"Stimulus valence must be between -1 and 1, got {stimulus_valence}")
        
        # Mood creates a bias in how we perceive things
        bias = self.current_mood.valence_bias * self.current_mood.intensity
        
        # Positive mood makes things seem more positive (and vice versa)
        biased_valence = stimulus_valence + bias
        
        # But extreme stimuli can break through mood bias
        if abs(stimulus_valence) > 0.7:
            # Strong stimuli partially override mood
            biased_valence = stimulus_valence * 0.7 + biased_valence * 0.3
        
        return max(-1.0, min(1.0, biased_valence))
    
    def mood_congruent_bias(self, stimulus: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply mood-congruent biases to stimulus processing.
        
        Args:
            stimulus: Raw stimulus information
            
        Returns:
            Mood-biased stimulus interpretation (copy with modifications)
        """
        # Create a copy to avoid modifying original
        biased_stimulus = stimulus.copy()
        
        # Apply valence bias
        if 'valence' in biased_stimulus:
            biased_stimulus['valence'] = self.get_mood_color(biased_stimulus['valence'])
        
        # Apply attention bias - what aspects we notice
        attention_bias = self._current_biases.get('attention_bias', 'balanced')
        
        # Apply multiplicative biases based on attention focus
        bias_factors = {
            'threats': ('threat_potential', 1.3),
            'opportunities': ('opportunity_potential', 1.3),
            'annoyances': ('annoyance_factor', 1.4),
            'absurdities': ('absurdity_potential', 1.5),
            'hidden_motives': ('motive_complexity', 1.3),
            'flaws': ('flaw_significance', 1.3),
            'patterns': ('pattern_salience', 1.2),
            'nuances': ('nuance_depth', 1.2)
        }
        
        if attention_bias in bias_factors:
            key, factor = bias_factors[attention_bias]
            if key in biased_stimulus:
                biased_stimulus[key] *= factor
        
        # Add bias information for downstream processing
        biased_stimulus['mood_bias_applied'] = {
            'mood': self.current_mood.name,
            'mood_intensity': self.current_mood.intensity,
            'valence_bias': self.current_mood.valence_bias,
            'attention_bias': attention_bias,
            'cognitive_bias': self._current_biases.get('cognitive_bias', 'objective')
        }
        
        return biased_stimulus
    
    def get_mood_info(self) -> Dict[str, Any]:
        """
        Get comprehensive mood information.
        
        Returns:
            Dictionary with current mood state and biases
        """
        return {
            'name': self.current_mood.name,
            'intensity': round(self.current_mood.intensity, 3),
            'persistence': self.current_mood.persistence,
            'description': self.MOOD_BIASES[self.current_mood.name]['description'],
            'biases': {
                'valence_bias': round(self.current_mood.valence_bias, 3),
                'arousal_bias': round(self.current_mood.arousal_bias, 3),
                'attention_bias': self._current_biases.get('attention_bias', 'balanced'),
                'cognitive_bias': self._current_biases.get('cognitive_bias', 'objective')
            },
            'duration_seconds': self._get_mood_duration(),
            'trend': self._get_mood_trend()
        }
    
    def get_mood_history(self, 
                        limit: Optional[int] = None,
                        mood_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get mood history with optional filtering.
        
        Args:
            limit: Maximum number of entries to return
            mood_type: Filter by mood type
            
        Returns:
            List of historical mood states
        """
        if mood_type and mood_type not in self.VALID_MOODS:
            raise ValueError(f"Invalid mood type: {mood_type}")
        
        # Create a copy to avoid modification issues
        history = self.mood_history.copy()
        
        if mood_type:
            history = [m for m in history if m.name == mood_type]
        
        if limit:
            history = history[-limit:]
        
        return [{
            'name': m.name,
            'intensity': round(m.intensity, 3),
            'timestamp': m.timestamp,
            'datetime': datetime.fromtimestamp(m.timestamp).isoformat(),
            'valence_bias': round(m.valence_bias, 3)
        } for m in history]
    
    def set_mood(self, mood_name: str, intensity: Optional[float] = None) -> bool:
        """
        Explicitly set current mood (primarily for testing or exceptional circumstances).
        
        Args:
            mood_name: Name of mood from MoodType
            intensity: Optional intensity override (0-1)
            
        Returns:
            True if mood was set successfully
            
        Raises:
            ValueError: If mood_name is invalid or intensity out of range
        """
        if mood_name not in self.VALID_MOODS:
            raise ValueError(f"Invalid mood name: {mood_name}")
        
        # Validate intensity
        if intensity is not None:
            if not 0 <= intensity <= 1:
                raise ValueError(f"Intensity must be between 0 and 1, got {intensity}")
        else:
            intensity = self.personality['mood_intensity_factor']
        
        # Store old mood for history
        self._record_current_mood()
        
        # Get biases for new mood
        biases = self.MOOD_BIASES[mood_name]
        
        # Set new mood
        self.current_mood = MoodState(
            name=mood_name,
            intensity=intensity,
            persistence=self.DEFAULT_MOOD['persistence'],
            valence_bias=biases['valence_bias'],
            arousal_bias=biases['arousal_bias'],
            timestamp=time.time()
        )
        
        # Update biases
        self._update_biases()
        
        logger.info(f"Mood explicitly set to: {mood_name} (intensity: {intensity:.2f})")
        return True
    
    def reset_to_baseline(self) -> None:
        """Reset mood to personality baseline"""
        baseline_mood = self.personality['baseline_mood']
        biases = self.MOOD_BIASES[baseline_mood]
        
        self.current_mood = MoodState(
            name=baseline_mood,
            intensity=self.personality['mood_intensity_factor'],
            persistence=self.DEFAULT_MOOD['persistence'],
            valence_bias=biases['valence_bias'],
            arousal_bias=biases['arousal_bias'],
            timestamp=time.time()
        )
        
        # Clear emotional accumulator with decay
        self.emotional_accumulator = {
            'valence_sum': 0.0,
            'arousal_sum': 0.0,
            'event_count': 0,
            'significant_events': 0,
            'last_decay_time': time.time()
        }
        
        self._update_biases()
        logger.info("Mood reset to baseline")
    
    def _calculate_significance(self, emotional_state: Dict[str, Any]) -> float:
        """
        Calculate how personally significant an emotional event is.
        
        Args:
            emotional_state: Current emotional state
            
        Returns:
            Significance score 0-1
        """
        # Intensity contributes to significance
        intensity = max(emotional_state['emotions'].values()) if emotional_state['emotions'] else 0
        
        # Extreme valence also contributes
        valence_extremity = abs(emotional_state['pad']['valence'])
        
        # Dominance (feeling in control) modulates significance
        dominance = emotional_state['pad']['dominance']
        
        # Calculate significance
        significance = (intensity * 0.5 + valence_extremity * 0.3) * (1 + (1 - dominance) * 0.2)
        
        return min(1.0, significance)
    
    def _update_accumulator(self, event: EmotionalEvent) -> None:
        """Update emotional accumulation with new event and time decay"""
        # Apply time-based decay to existing accumulator
        self._decay_accumulator()
        
        # Weight by significance and emotional absorption
        absorption = self.personality['emotional_absorption']
        weighted_valence = event.valence * (1 + event.significance * absorption)
        weighted_arousal = event.arousal * (1 + event.significance * absorption * 0.5)
        
        self.emotional_accumulator['valence_sum'] += weighted_valence
        self.emotional_accumulator['arousal_sum'] += weighted_arousal
        self.emotional_accumulator['event_count'] += 1
        
        if event.significance > 0.6:
            self.emotional_accumulator['significant_events'] += 1
    
    def _decay_accumulator(self) -> None:
        """Apply time-based decay to emotional accumulator"""
        current_time = time.time()
        time_delta = current_time - self.emotional_accumulator['last_decay_time']
        self.emotional_accumulator['last_decay_time'] = current_time
        
        # Decay over hours (faster decay than mood itself)
        hours_passed = time_delta / 3600.0
        if hours_passed > 0:
            decay_factor = max(0.5, 1.0 - (hours_passed * 0.3))  # 30% decay per hour
            self.emotional_accumulator['valence_sum'] *= decay_factor
            self.emotional_accumulator['arousal_sum'] *= decay_factor
    
    def _evaluate_mood_shift(self) -> Optional[Dict[str, Any]]:
        """
        Evaluate whether mood should shift based on emotional accumulation.
        
        Returns:
            Mood shift information or None
        """
        if self.emotional_accumulator['event_count'] < 5:
            return None  # Not enough data
        
        # Calculate average valence and arousal
        avg_valence = self._get_average_valence()
        avg_arousal = self._get_average_arousal()
        
        # Calculate shift pressure (weighted by mood volatility)
        current_mood_bias = self.current_mood.valence_bias
        valence_delta = abs(avg_valence - current_mood_bias)
        
        volatility = self.personality['mood_volatility']
        effective_threshold = self.MOOD_SHIFT_THRESHOLD * (2 - volatility)  # Lower volatility = higher threshold
        
        # Check if we've crossed threshold
        if valence_delta > effective_threshold:
            # Determine new mood based on valence and arousal
            new_mood = self._determine_mood_from_accumulation(avg_valence, avg_arousal)
            
            if new_mood and new_mood != self.current_mood.name:
                # Apply mood shift
                old_mood = self.current_mood.name
                
                # Calculate new intensity based on accumulation strength and volatility
                new_intensity = min(1.0, self.personality['mood_intensity_factor'] + 
                                  valence_delta * volatility)
                
                # Store old mood
                self._record_current_mood()
                
                # Get biases for new mood
                biases = self.MOOD_BIASES[new_mood]
                
                # Set new mood
                self.current_mood = MoodState(
                    name=new_mood,
                    intensity=new_intensity,
                    persistence=self.DEFAULT_MOOD['persistence'],
                    valence_bias=biases['valence_bias'],
                    arousal_bias=biases['arousal_bias'],
                    timestamp=time.time()
                )
                
                # Reset accumulator with momentum
                momentum = 0.3  # Keep some emotional momentum
                self.emotional_accumulator['valence_sum'] *= momentum
                self.emotional_accumulator['arousal_sum'] *= momentum
                self.emotional_accumulator['event_count'] = max(1, 
                    int(self.emotional_accumulator['event_count'] * momentum))
                
                logger.info(f"Mood shifted: {old_mood} -> {new_mood} "
                           f"(intensity: {new_intensity:.2f})")
                
                return {
                    'from': old_mood,
                    'to': new_mood,
                    'trigger_valence': round(avg_valence, 3),
                    'valence_delta': round(valence_delta, 3),
                    'threshold': round(effective_threshold, 3)
                }
        
        return None
    
    def _determine_mood_from_accumulation(self, 
                                         avg_valence: float, 
                                         avg_arousal: float) -> Optional[str]:
        """
        Determine appropriate mood based on accumulated emotional state.
        
        Args:
            avg_valence: Average valence from recent emotions (-1 to 1)
            avg_arousal: Average arousal from recent emotions (0 to 1)
            
        Returns:
            Mood name or None
        """
        # Map valence/arousal to moods with Wednesday-specific tendencies
        
        # High positive valence
        if avg_valence > 0.3:
            if avg_arousal > 0.5:
                # Check for darkly amused (Wednesday-specific)
                if (self.personality['dark_humor_tendency'] > 0.6 and 
                    random.random() < 0.3):  # 30% chance when conditions are right
                    return 'darkly_amused'
                return 'positive'
            else:
                return 'satisfied'
        
        # Negative valence
        elif avg_valence < -0.2:
            if avg_arousal > 0.4:
                if avg_arousal > 0.6:
                    return 'irritable'
                else:
                    return 'wary'
            else:
                return 'pensive'
        
        # Near-neutral valence
        else:
            if avg_arousal < 0.2:
                return 'reflective'
            elif avg_arousal > 0.5:
                return 'curiously_detached'
            else:
                # Slight negative bias for Wednesday
                if avg_valence < 0 and random.random() < 0.4:
                    return 'disdainful'
                return 'neutral'
    
    def _apply_mood_drift(self) -> None:
        """Apply natural mood drift over time"""
        time_delta = time.time() - self.last_update_time
        self.last_update_time = time.time()
        
        # Scale drift by time (assuming 1 hour = full scale)
        hours_passed = min(time_delta / 3600.0, 2.0)  # Cap at 2 hours
        
        if hours_passed <= 0:
            return
        
        # Mood intensity naturally decays
        decay = self.INTENSITY_DECAY_RATE * hours_passed
        self.current_mood.intensity = max(0.1, 
            self.current_mood.intensity - decay * self.current_mood.intensity)
        
        # Mood gradually drifts toward baseline (recovery)
        baseline_mood = self.personality['baseline_mood']
        if self.current_mood.name != baseline_mood:
            # Calculate drift probability based on recovery rate
            drift_probability = self.personality['recovery_rate'] * hours_passed * 0.5
            if random.random() < drift_probability:
                self.current_mood.name = baseline_mood
                biases = self.MOOD_BIASES[baseline_mood]
                self.current_mood.valence_bias = biases['valence_bias']
                self.current_mood.arousal_bias = biases['arousal_bias']
                logger.debug(f"Mood drifted toward baseline: {baseline_mood}")
    
    def _apply_personality_tendencies(self) -> None:
        """Apply personality-based mood tendencies"""
        # Wednesday tends toward certain moods based on personality
        if self.personality['dark_humor_tendency'] > 0.6:
            # Occasionally shift to darkly_amused if conditions are right
            if (self.current_mood.name in ['neutral', 'reflective'] and 
                random.random() < 0.01):  # 1% chance per update
                self.current_mood.name = 'darkly_amused'
                biases = self.MOOD_BIASES['darkly_amused']
                self.current_mood.valence_bias = biases['valence_bias']
                self.current_mood.arousal_bias = biases['arousal_bias']
                logger.debug("Personality tendency triggered: darkly_amused")
    
    def _update_biases(self) -> None:
        """Update current mood biases"""
        self._current_biases = self.MOOD_BIASES[self.current_mood.name]
        self.current_mood.valence_bias = self._current_biases['valence_bias'] * self.current_mood.intensity
        self.current_mood.arousal_bias = self._current_biases['arousal_bias'] * self.current_mood.intensity
    
    def _record_current_mood(self) -> None:
        """Add current mood to history"""
        # Create a copy to preserve state
        mood_copy = MoodState(
            name=self.current_mood.name,
            intensity=self.current_mood.intensity,
            persistence=self.current_mood.persistence,
            valence_bias=self.current_mood.valence_bias,
            arousal_bias=self.current_mood.arousal_bias,
            timestamp=self.current_mood.timestamp
        )
        
        self.mood_history.append(mood_copy)
        
        # Maintain history size limit
        if len(self.mood_history) > self.max_history_size:
            self.mood_history.pop(0)
    
    def _get_average_valence(self) -> float:
        """Calculate average valence from recent emotions"""
        if self.emotional_accumulator['event_count'] == 0:
            return 0.0
        return self.emotional_accumulator['valence_sum'] / self.emotional_accumulator['event_count']
    
    def _get_average_arousal(self) -> float:
        """Calculate average arousal from recent emotions"""
        if self.emotional_accumulator['event_count'] == 0:
            return 0.0
        return self.emotional_accumulator['arousal_sum'] / self.emotional_accumulator['event_count']
    
    def _get_mood_duration(self) -> float:
        """Get duration of current mood in seconds"""
        return time.time() - self.current_mood.timestamp
    
    def _get_mood_trend(self) -> str:
        """Determine if mood is strengthening or weakening"""
        if len(self.mood_history) < 2:
            return 'stable'
        
        # Compare intensity with previous
        prev_intensity = self.mood_history[-2].intensity
        if self.current_mood.intensity > prev_intensity * 1.1:
            return 'intensifying'
        elif self.current_mood.intensity < prev_intensity * 0.9:
            return 'weakening'
        else:
            return 'stable'
    
    def __repr__(self) -> str:
        """String representation of mood engine"""
        return (f"MoodEngine(mood={self.current_mood.name}, "
                f"intensity={self.current_mood.intensity:.2f}, "
                f"valence_bias={self.current_mood.valence_bias:.2f})")


# Example usage and testing
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create mood engine with Wednesday-like personality
    personality = {
        'baseline_mood': 'neutral',
        'mood_volatility': 0.3,
        'mood_intensity_factor': 0.5,
        'emotional_absorption': 0.6,
        'recovery_rate': 0.2,
        'dark_humor_tendency': 0.8
    }
    
    mood_engine = MoodEngine(personality)
    print("Initial mood:", mood_engine.get_mood_info())
    print()
    
    # Simulate a series of emotional states
    emotional_states = [
        {
            'dominant': 'anger', 
            'pad': {'valence': -0.5, 'arousal': 0.7, 'dominance': 0.6}, 
            'emotions': {'anger': 0.6, 'trust': 0.3}
        },
        {
            'dominant': 'sadness', 
            'pad': {'valence': -0.4, 'arousal': 0.2, 'dominance': 0.3},
            'emotions': {'sadness': 0.5, 'trust': 0.4}
        },
        {
            'dominant': 'joy', 
            'pad': {'valence': 0.6, 'arousal': 0.5, 'dominance': 0.7},
            'emotions': {'joy': 0.5, 'trust': 0.5}
        },
        {
            'dominant': 'trust', 
            'pad': {'valence': 0.3, 'arousal': 0.2, 'dominance': 0.6},
            'emotions': {'trust': 0.7, 'joy': 0.2}
        },
        {
            'dominant': 'fear', 
            'pad': {'valence': -0.4, 'arousal': 0.8, 'dominance': 0.2},
            'emotions': {'fear': 0.7, 'trust': 0.2}
        },
    ]
    
    print("--- Simulating emotional events ---")
    for i, emotion in enumerate(emotional_states):
        print(f"\nEvent {i+1}: {emotion['dominant']} "
              f"(V={emotion['pad']['valence']:.1f}, A={emotion['pad']['arousal']:.1f})")
        
        result = mood_engine.update_from_emotion(emotion)
        mood_info = result['current_mood']
        
        print(f"Mood: {mood_info['name']} (intensity: {mood_info['intensity']:.2f})")
        print(f"Bias: valence={mood_info['biases']['valence_bias']:.2f}")
        
        if result['mood_shift_occurred']:
            print(f"*** MOOD SHIFT: {result['mood_shift']['from']} -> "
                  f"{result['mood_shift']['to']} ***")
        
        # Test mood-congruent bias
        test_stimulus = {'valence': 0.2, 'threat_potential': 0.3}
        biased = mood_engine.mood_congruent_bias(test_stimulus)
        print(f"Mood bias applied: {biased.get('valence', 0):.2f} valence")
        
        # Small time delay between events
        time.sleep(0.1)
    
    print("\n--- Final mood state ---")
    print(mood_engine.get_mood_info())
    
    print("\n--- Mood history (last 3) ---")
    history = mood_engine.get_mood_history(limit=3)
    for h in history:
        dt = datetime.fromtimestamp(h['timestamp'])
        print(f"  {h['name']} at {dt.strftime('%H:%M:%S')} "
              f"(intensity: {h['intensity']:.2f})")