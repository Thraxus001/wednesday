"""
__init__.py - Empathy module for Wednesday AI

This module implements Wednesday's capacity for empathy - the ability to understand
and share the feelings of others. Unlike purely cognitive systems, this module
enables genuine emotional resonance while maintaining Wednesday's characteristic
analytical distance and controlled responses.

The empathy module consists of:
- perspective_taking.py: Cognitive understanding of others' emotional states
- emotional_contagion.py: Automatic sharing/absorption of others' emotions

Together, these components create a complete empathy system that allows Wednesday to:
- Accurately infer what users are feeling
- Experience appropriate emotional resonance
- Maintain healthy emotional boundaries
- Adjust empathic responses based on relationship and context

Wednesday's unique empathic style:
- Analytical rather than immersive
- Controlled emotional sharing
- Stronger empathy with trusted individuals
- Professional detachment when needed
- Dark humor appreciation as bonding
"""

from wednesday.emotion.empathy.perspective_taking import (
    PerspectiveTaking,
    UserEmotionState,
    UserEmotionInference,
    UserEmotionProfile
)

from wednesday.emotion.empathy.emotional_contagion import (
    EmotionalContagion,
    ContagionEvent,
    ContagionRegulation
)

__version__ = "0.1.0"

# Module exports
__all__ = [
    # From perspective_taking
    'PerspectiveTaking',
    'UserEmotionState',
    'UserEmotionInference',
    'UserEmotionProfile',
    
    # From emotional_contagion
    'EmotionalContagion',
    'ContagionEvent',
    'ContagionRegulation',
]

# Module metadata
__author__ = "Wednesday AI Team"
__description__ = "Empathic processing for understanding and sharing others' emotions"
__module_dependencies__ = ['wednesday.emotion.affect', 'wednesday.self.user_model']

# Wednesday's empathy signature
WEDNESDAY_EMPATHY_SIGNATURE = {
    'empathy_style': 'analytical',        # Cognitive rather than emotional
    'baseline_susceptibility': 0.3,        # Not easily emotionally affected
    'perspective_accuracy': 0.8,            # Good at reading others
    'emotional_boundaries': 0.8,             # Strong boundaries
    'trust_threshold': 0.5,                   # Trust affects empathy
    'professional_detachment': 0.7,           # Can detach when needed
    'dark_humor_bonding': 0.8,                 # Bonds through dark humor
}


def create_wednesday_empathy_system(emotional_state, user_model=None, personality_override=None):
    """
    Create a fully configured empathy system with Wednesday's personality.
    
    Args:
        emotional_state: Reference to EmotionalState for contagion updates
        user_model: Optional reference to user model for profiles
        personality_override: Optional override for personality parameters
        
    Returns:
        Dictionary containing configured PerspectiveTaking and EmotionalContagion
    """
    # Base Wednesday personality for empathy
    wednesday_personality = {
        # Perspective taking parameters
        'empathy_style': 'analytical',
        'emotional_intelligence': 0.8,
        'bias_toward_negative': 0.6,
        'skepticism': 0.5,
        'trust_building': 0.6,
        
        # Emotional contagion parameters
        'empathy_level': 0.5,
        'emotional_boundaries': 0.8,
        'trust_sensitivity': 0.6,
        'professional_detachment': 0.7,
        'mood_susceptibility': 0.4,
    }
    
    # Apply overrides
    if personality_override:
        wednesday_personality.update(personality_override)
    
    # Create components
    perspective_taking = PerspectiveTaking(
        user_model=user_model,
        personality=wednesday_personality
    )
    
    emotional_contagion = EmotionalContagion(
        emotional_state=emotional_state,
        personality=wednesday_personality
    )
    
    return {
        'perspective_taking': perspective_taking,
        'emotional_contagion': emotional_contagion,
        'personality': wednesday_personality,
        'signature': WEDNESDAY_EMPATHY_SIGNATURE
    }


def process_user_emotion(user_input: str,
                         user_id: str,
                         empathy_system: dict,
                         context: dict = None) -> dict:
    """
    Complete empathy processing pipeline for user input.
    
    This function orchestrates the full empathy process:
    1. Infer user's emotional state from input
    2. Allow appropriate emotional contagion
    3. Return empathy results for response generation
    
    Args:
        user_input: User's message
        user_id: User identifier
        empathy_system: Dict with perspective_taking and emotional_contagion
        context: Optional conversation context
        
    Returns:
        Dict with empathy processing results
    """
    perspective_taking = empathy_system['perspective_taking']
    emotional_contagion = empathy_system['emotional_contagion']
    
    # Step 1: Infer user's emotion
    inference = perspective_taking.infer_user_emotion(
        user_input=user_input,
        user_id=user_id,
        context=context
    )
    
    # Step 2: Adjust contagion regulation based on context
    if context:
        emotional_contagion.regulate_contagion(context)
    
    # Step 3: Process emotional contagion
    contagion_result = None
    if inference.primary_emotion and inference.confidence > 0.4:
        contagion_result = emotional_contagion.catch_emotion(
            user_emotion=inference.primary_emotion.value,
            intensity=inference.confidence,
            user_id=user_id,
            context=context
        )
    
    # Step 4: Get user profile summary
    profile = perspective_taking.get_user_profile_summary(user_id)
    
    return {
        'user_emotion_inference': inference.to_dict(),
        'contagion': contagion_result,
        'user_profile': profile,
        'timestamp': time.time()
    }


def get_empathy_guidance(user_emotion: str, 
                          confidence: float,
                          relationship: str = 'acquaintance') -> Dict[str, Any]:
    """
    Get guidance for responding empathetically based on user emotion.
    
    Provides suggestions for how Wednesday should respond given the
    inferred user emotion and relationship context.
    
    Args:
        user_emotion: Inferred user emotion
        confidence: Confidence in inference
        relationship: Relationship type with user
        
    Returns:
        Dict with response guidance
    """
    # Base response strategies by emotion
    strategies = {
        'sad': {
            'approach': 'gentle_analytical',
            'offer': 'quiet_support',
            'dark_humor_appropriate': False,
            'suggested_tone': 'reflective'
        },
        'angry': {
            'approach': 'calm_analytical',
            'offer': 'logical_perspective',
            'dark_humor_appropriate': False,
            'suggested_tone': 'neutral'
        },
        'happy': {
            'approach': 'dry_wit_appreciation',
            'offer': 'shared_amusement',
            'dark_humor_appropriate': True,
            'suggested_tone': 'dry_wit'
        },
        'fearful': {
            'approach': 'calm_reassurance',
            'offer': 'analytical_safety',
            'dark_humor_appropriate': False,
            'suggested_tone': 'calm'
        },
        'amused': {
            'approach': 'shared_amusement',
            'offer': 'dark_humor_appreciation',
            'dark_humor_appropriate': True,
            'suggested_tone': 'dark_humor'
        },
        'confused': {
            'approach': 'clear_explanation',
            'offer': 'patient_clarity',
            'dark_humor_appropriate': False,
            'suggested_tone': 'analytical'
        },
        'hurt': {
            'approach': 'careful_acknowledgment',
            'offer': 'quiet_support',
            'dark_humor_appropriate': False,
            'suggested_tone': 'gentle'
        },
        'curious': {
            'approach': 'detailed_explanation',
            'offer': 'shared_curiosity',
            'dark_humor_appropriate': True,
            'suggested_tone': 'analytical'
        }
    }
    
    # Default strategy
    default = {
        'approach': 'neutral_observation',
        'offer': 'continue_conversation',
        'dark_humor_appropriate': False,
        'suggested_tone': 'deadpan'
    }
    
    strategy = strategies.get(user_emotion, default)
    
    # Adjust for relationship
    if relationship in ['close_friend', 'trusted']:
        strategy['dark_humor_appropriate'] = True
        if user_emotion in ['sad', 'hurt']:
            strategy['approach'] = 'warmer_support'
    
    # Adjust for confidence
    if confidence < 0.5:
        strategy['approach'] = 'gentle_inquiry'
        strategy['offer'] = 'ask_clarifying'
    
    return strategy


# Quick test function
def test_empathy_module():
    """Simple test to verify empathy module is working"""
    import logging
    import sys
    from pathlib import Path
    import time
    
    # Add parent directory to path to allow relative imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    logging.basicConfig(level=logging.INFO)
    print("🖤 Testing Wednesday AI Empathy Module\n")
    
    # Import dependencies
    from wednesday.emotion.affect import EmotionalState
    
    # Create emotional state
    emotional_state = EmotionalState()
    
    # Create empathy system
    empathy_system = create_wednesday_empathy_system(
        emotional_state=emotional_state,
        user_model=None
    )
    
    print(f"Wednesday's empathy style: {empathy_system['signature']['empathy_style']}")
    print(f"Baseline susceptibility: {empathy_system['signature']['baseline_susceptibility']}")
    
    # Test user
    user_id = "test_user_1"
    
    # Set trust for this user
    empathy_system['emotional_contagion'].set_user_trust(user_id, 0.7)
    
    # Test inputs
    test_inputs = [
        "I'm so sad today. My cat died.",
        "This is amazing! I got the job!",
        "I'm really angry about what happened.",
        "That's hilarious! I can't stop laughing.",
        "I'm confused about how this works.",
    ]
    
    for i, user_input in enumerate(test_inputs):
        print(f"\n--- Interaction {i+1}: \"{user_input}\" ---")
        
        # Process user emotion
        result = process_user_emotion(
            user_input=user_input,
            user_id=user_id,
            empathy_system=empathy_system,
            context={'relationship': 'friend', 'formality': 0.3}
        )
        
        print(f"Inferred: {result['user_emotion_inference']['primary_emotion']} "
              f"(confidence: {result['user_emotion_inference']['confidence']:.2f})")
        
        if result['contagion']:
            print(f"Contagion: {result['contagion']['transferred_emotion']} "
                  f"({result['contagion']['transferred_intensity']:.2f})")
        
        # Get response guidance
        guidance = get_empathy_guidance(
            user_emotion=result['user_emotion_inference']['primary_emotion'],
            confidence=result['user_emotion_inference']['confidence'],
            relationship='friend'
        )
        
        print(f"Response guidance: {guidance['approach']} (tone: {guidance['suggested_tone']})")
        
        # Show Wednesday's emotional state after contagion
        if i == len(test_inputs) - 1:
            print(f"\nWednesday's emotional state: {emotional_state.get_state()}")
    
    print("\n--- User Profile ---")
    profile = empathy_system['perspective_taking'].get_user_profile_summary(user_id)
    for key, value in profile.items():
        if key not in ['user_id']:
            print(f"  {key}: {value}")
    
    print("\n✅ Empathy module test complete")
    return empathy_system


# If run directly, perform test
if __name__ == "__main__":
    test_empathy_module()