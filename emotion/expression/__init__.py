"""
__init__.py - Expression module for Wednesday AI

This module implements Wednesday's emotional expression system, which determines
how her internal emotional states are manifested in her responses. It handles
the translation of emotions into observable behaviors: word choice, tone,
punctuation, pacing, and other stylistic elements that make her feel authentic.

The expression module consists of:
- emotional_response.py: High-level expression strategy and style selection
- tone_modulation.py: Low-level modulation of text and voice parameters

Together, these components ensure that Wednesday's emotions are expressed in
ways that are:
- Consistent with her character (controlled, darkly witty, subtle)
- Appropriate to the emotional state
- Natural and believable in conversation
- Adaptable to different contexts and relationships
"""

from wednesday.emotion.expression.emotional_response import (
    EmotionalResponse,
    ExpressionStyle,
    ExpressionModifier,
    ExpressionParameters
)

from wednesday.emotion.expression.tone_modulation import (
    ToneModulator,
    ToneParameters,
    PunctuationStyle,
    EmphasisPattern
)

__version__ = "0.1.0"

# Module exports
__all__ = [
    # From emotional_response
    'EmotionalResponse',
    'ExpressionStyle',
    'ExpressionModifier',
    'ExpressionParameters',
    
    # From tone_modulation
    'ToneModulator',
    'ToneParameters',
    'PunctuationStyle',
    'EmphasisPattern',
]

# Module metadata
__author__ = "Wednesday AI Team"
__description__ = "Emotional expression and tone modulation for Wednesday AI"
__module_dependencies__ = ['wednesday.emotion.affect', 'wednesday.emotion.appraisal']

# Wednesday's expression signature
WEDNESDAY_EXPRESSION_SIGNATURE = {
    'default_style': 'deadpan',
    'signature_style': 'dark_humor',
    'expressiveness': 0.4,           # Subtle, not overt
    'emotional_control': 0.8,         # High control over expression
    'dryness': 0.8,                    # Dry wit preference
    'punctuation_tendency': 0.3,       # Minimal punctuation emphasis
    'emphasis_tendency': 0.3,           # Rare emphasis
    'vocal_expressiveness': 0.3,        # Controlled voice
    'sarcasm_tendency': 0.7,            # Frequent sarcasm
    'dark_humor_tendency': 0.8,          # Loves dark humor
}


def create_wednesday_expression_system(mood_engine, personality_override=None):
    """
    Create a fully configured expression system with Wednesday's personality.
    
    Args:
        mood_engine: Reference to mood engine for mood context
        personality_override: Optional override for personality parameters
        
    Returns:
        Dictionary containing configured EmotionalResponse and ToneModulator
    """
    # Base Wednesday personality for expression
    wednesday_personality = {
        'expressiveness': 0.4,
        'dry_wit_tendency': 0.8,
        'dark_humor_tendency': 0.8,
        'emotional_control': 0.7,
        'sarcasm_tendency': 0.7,
        'authenticity_bias': 0.8,
        'vocal_expressiveness': 0.3,
        'punctuation_tendency': 0.3,
        'emphasis_tendency': 0.3,
        'dryness': 0.8,
        'formality': 0.5,
    }
    
    # Apply overrides
    if personality_override:
        wednesday_personality.update(personality_override)
    
    # Create components
    emotional_response = EmotionalResponse(
        mood_engine=mood_engine,
        personality=wednesday_personality
    )
    
    tone_modulator = ToneModulator(
        personality=wednesday_personality
    )
    
    return {
        'emotional_response': emotional_response,
        'tone_modulator': tone_modulator,
        'personality': wednesday_personality,
        'signature': WEDNESDAY_EXPRESSION_SIGNATURE
    }


def express_emotion(text: str, 
                    emotional_state: dict, 
                    expression_system: dict,
                    context: dict = None) -> dict:
    """
    Complete emotional expression pipeline for a response.
    
    This function orchestrates the full expression process:
    1. Generate expression parameters based on emotional state
    2. Modulate text tone based on those parameters
    3. Return both modulated text and voice parameters
    
    Args:
        text: Base text to express
        emotional_state: Current emotional state from EmotionalState
        expression_system: Dict with emotional_response and tone_modulator
        context: Optional conversation context
        
    Returns:
        Dict with modulated text and expression metadata
    """
    emotional_response = expression_system['emotional_response']
    tone_modulator = expression_system['tone_modulator']
    
    # Step 1: Generate expression parameters
    expression_metadata = emotional_response.modulate_response(
        base_response=text,
        emotional_state=emotional_state,
        context=context
    )
    
    # Step 2: Apply tone modulation to text
    modulated_text = tone_modulator.apply_to_text(
        text=text,
        emotional_state=emotional_state,
        expression_params=expression_metadata.get('parameters')
    )
    
    # Step 3: Get voice parameters
    voice_params = tone_modulator.get_voice_params(
        emotional_state=emotional_state,
        expression_params=expression_metadata.get('parameters')
    )
    
    # Step 4: Suggest emotional interjection if appropriate
    interjection = emotional_response.suggest_emotional_interjection(emotional_state)
    
    return {
        'original_text': text,
        'modulated_text': modulated_text,
        'interjection': interjection,
        'full_response': f"{interjection} {modulated_text}" if interjection else modulated_text,
        'expression_metadata': expression_metadata,
        'voice_params': voice_params,
        'style': expression_metadata.get('style'),
        'emotional_coloring': emotional_state.get('dominant', 'neutral')
    }


def get_wednesday_expression_examples():
    """Return examples of Wednesday's emotional expressions for testing"""
    return {
        'neutral': {
            'text': "I see. That's an interesting development.",
            'style': 'deadpan',
            'voice': {'rate': 1.0, 'pitch': 1.0}
        },
        'dark_amusement': {
            'text': "How delightfully morbid. I approve.",
            'style': 'dark_humor',
            'voice': {'rate': 0.9, 'pitch': 0.95}
        },
        'anger': {
            'text': "That is unacceptable. It will be addressed.",
            'style': 'cold',
            'voice': {'rate': 0.9, 'pitch': 0.95, 'volume': 1.2}
        },
        'protective': {
            'text': "Enough. They're under my protection.",
            'style': 'intense',
            'voice': {'rate': 0.8, 'pitch': 0.9, 'volume': 1.1}
        },
        'satisfaction': {
            'text': "Good. Everything is proceeding as intended.",
            'style': 'deadpan',
            'voice': {'rate': 0.9, 'pitch': 1.0}
        },
        'curiosity': {
            'text': "Fascinating. Tell me more about this.",
            'style': 'analytical',
            'voice': {'rate': 0.9, 'pitch': 1.0}
        }
    }


# Quick test function
def test_expression_module():
    """Simple test to verify expression module is working"""
    import logging
    import sys
    from pathlib import Path
    
    # Add parent directory to path to allow relative imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    logging.basicConfig(level=logging.INFO)
    print("🎭 Testing Wednesday AI Expression Module\n")
    
    # Import dependencies
    from wednesday.emotion.affect import EmotionalState, MoodEngine
    
    # Create affect system
    emotional_state = EmotionalState()
    mood_engine = MoodEngine()
    
    # Create expression system
    expression_system = create_wednesday_expression_system(
        mood_engine=mood_engine
    )
    
    print(f"Initial emotional state: {emotional_state.dominant_emotion()}")
    print(f"Expression personality: {expression_system['personality']}\n")
    
    # Test base text
    base_text = "I see. That is an interesting development. I will consider it carefully."
    
    # Test different emotional states
    test_emotions = [
        {'dominant': 'neutral', 'emotions': {'neutral': 0.8}, 
         'pad': {'valence': 0.0, 'arousal': 0.3, 'dominance': 0.6}},
        {'dominant': 'dark_amusement', 'emotions': {'dark_amusement': 0.7}, 
         'pad': {'valence': 0.2, 'arousal': 0.5, 'dominance': 0.7}},
        {'dominant': 'anger', 'emotions': {'anger': 0.6}, 
         'pad': {'valence': -0.4, 'arousal': 0.7, 'dominance': 0.6}},
        {'dominant': 'sadness', 'emotions': {'sadness': 0.6}, 
         'pad': {'valence': -0.4, 'arousal': 0.2, 'dominance': 0.3}},
        {'dominant': 'protective', 'emotions': {'protective': 0.7}, 
         'pad': {'valence': -0.1, 'arousal': 0.6, 'dominance': 0.8}},
    ]
    
    for i, emotion_state in enumerate(test_emotions):
        print(f"\n--- Scenario {i+1}: {emotion_state['dominant']} ---")
        
        # Update emotional state (normally this would come from appraisal)
        emotional_state.emotions = emotion_state['emotions']
        emotional_state.valence = emotion_state['pad']['valence']
        emotional_state.arousal = emotion_state['pad']['arousal']
        emotional_state.dominance = emotion_state['pad']['dominance']
        
        # Get current state
        current_state = emotional_state.get_state()
        
        # Express emotion
        result = express_emotion(
            text=base_text,
            emotional_state=current_state,
            expression_system=expression_system
        )
        
        print(f"Style: {result['style']}")
        print(f"Original: {result['original_text']}")
        print(f"Expressed: {result['full_response']}")
        print(f"Voice: {result['voice_params']}")
    
    print("\n--- Expression Examples ---")
    examples = get_wednesday_expression_examples()
    for emotion, example in examples.items():
        print(f"  {emotion}: {example['text']} ({example['style']})")
    
    print("\n✅ Expression module test complete")
    return expression_system


# If run directly, perform test
if __name__ == "__main__":
    test_expression_module()