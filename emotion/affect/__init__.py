"""
__init__.py - Affect module for Wednesday AI

This module implements Wednesday's affective system, handling both short-term
emotional states and longer-term moods. The affect system provides the
emotional coloring for all cognitive processes and ensures personality-consistent
emotional expression.

The affect module consists of:
- emotional_state.py: Core PAD-based emotional state with discrete emotions
- mood_engine.py: Longer-term mood that biases emotional processing
- valence_arousal.py: Foundational dimensional model and utilities

All affective components work together to create a coherent emotional experience
that reflects Wednesday's unique personality: darkly amused, controlled,
perceptive, and subtly expressive.
"""

from emotional_state import (
    EmotionalState,
    EmotionType,
    EmotionalSnapshot
)

from mood_engine import (
    MoodEngine,
    MoodType,
    MoodState,
    EmotionalEvent
)

from valence_arousal import (
    PADVector,
    EmotionPrototype,
    EmotionLexicon,
    ValenceArousalSpace,
    pad_distance_weighted,
    emotional_contrast,
    is_emotionally_congruent
)

__version__ = "0.1.0"

# Module exports
__all__ = [
    # From emotional_state
    'EmotionalState',
    'EmotionType',
    'EmotionalSnapshot',
    
    # From mood_engine
    'MoodEngine',
    'MoodType',
    'MoodState',
    'EmotionalEvent',
    
    # From valence_arousal
    'PADVector',
    'EmotionPrototype',
    'EmotionLexicon',
    'ValenceArousalSpace',
    'pad_distance_weighted',
    'emotional_contrast',
    'is_emotionally_congruent',
]

# Module metadata
__author__ = "Wednesday AI Team"
__description__ = "Affective processing module for Wednesday AI"
__module_dependencies__ = ['numpy', 'logging', 'time', 'dataclasses']

# Wednesday's emotional signature
WEDNESDAY_EMOTIONAL_SIGNATURE = {
    'baseline_valence': -0.2,      # Slightly negative baseline
    'baseline_arousal': 0.3,        # Generally calm
    'baseline_dominance': 0.6,       # Prefers to be in control
    'emotional_inertia': 0.7,        # Resistant to emotional change
    'expression_threshold': 0.4,      # Doesn't show emotions easily
    'default_trust': 0.5,             # Guarded but not closed off
    'signature_blend': 'dark_amusement'  # Her characteristic emotional flavor
}

# Convenience function to create a Wednesday-configured affect system
def create_wednesday_affect(personality_override: dict = None) -> dict:
    """
    Create a fully configured affect system with Wednesday's personality.
    
    Args:
        personality_override: Optional override for personality parameters
        
    Returns:
        Dictionary containing configured EmotionalState and MoodEngine
    """
    # Base Wednesday personality
    wednesday_personality = {
        'baseline_mood': 'neutral',
        'mood_volatility': 0.3,
        'mood_intensity_factor': 0.5,
        'emotional_absorption': 0.4,
        'recovery_rate': 0.2,
        'dark_humor_tendency': 0.8,
        'emotional_inertia': 0.7,
        'expression_threshold': 0.4,
        'trust_openness': 0.4,
    }
    
    # Apply overrides
    if personality_override:
        wednesday_personality.update(personality_override)
    
    # Create components
    emotional_state = EmotionalState(personality_factors=wednesday_personality)
    mood_engine = MoodEngine(personality=wednesday_personality)
    
    return {
        'emotional_state': emotional_state,
        'mood_engine': mood_engine,
        'personality': wednesday_personality,
        'signature': WEDNESDAY_EMOTIONAL_SIGNATURE
    }

# Disabled test for direct script run to avoid relative import issues
# if __name__ == "__main__":
#     test_affect_module()

