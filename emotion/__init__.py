"""
__init__.py - Emotion module for Wednesday AI

This module implements Wednesday's complete emotional processing system, integrating
affective states, cognitive appraisal, emotional expression, and empathic understanding.
The emotion system provides the foundation for Wednesday's personality, ensuring
consistent emotional responses that feel authentic and character-appropriate.

The emotion module consists of four submodules:
- affect/: Core emotional states (short-term emotions, longer-term mood)
- appraisal/: Cognitive evaluation of stimuli for emotional significance
- expression/: How emotions are manifested in responses
- empathy/: Understanding and sharing others' emotional states

Together, these components create a complete emotional architecture that allows
Wednesday to:
- Experience genuine emotions with appropriate intensity and decay
- Evaluate events for emotional relevance
- Express emotions in character-consistent ways
- Understand and resonate with user emotions
- Maintain personality coherence across all emotional processing

Wednesday's emotional signature:
- Controlled, subtle emotional expression
- Dark humor as a primary emotional outlet
- Analytical approach to emotions (including her own)
- Strong emotional boundaries with appropriate exceptions
- Loyalty and protective instincts as core emotional drivers
"""

import logging
from typing import Dict, Optional, Any

# Import submodules
from wednesday.emotion import appraisal, affect, expression, empathy

# Import key classes for easier access
from wednesday.emotion.affect import (
    EmotionalState,
    MoodEngine,
    PADVector,
    EmotionLexicon,
    create_wednesday_affect
)

from wednesday.emotion.appraisal import (
    StimulusEvaluator,
    RelevanceDetector,
    AppraisalResult,
    create_wednesday_appraisal_system
)

from wednesday.emotion.expression import (
    EmotionalResponse,
    ToneModulator,
    ExpressionStyle,
    create_wednesday_expression_system
)

from wednesday.emotion.empathy import (
    PerspectiveTaking,
    EmotionalContagion,
    UserEmotionState,
    create_wednesday_empathy_system
)

__version__ = "0.1.0"

# Module exports
__all__ = [
    # Submodules
    'affect',
    'appraisal', 
    'expression',
    'empathy',
    
    # From affect
    'EmotionalState',
    'MoodEngine',
    'PADVector',
    'EmotionLexicon',
    'create_wednesday_affect',
    
    # From appraisal
    'StimulusEvaluator',
    'RelevanceDetector',
    'AppraisalResult',
    'create_wednesday_appraisal_system',
    
    # From expression
    'EmotionalResponse',
    'ToneModulator',
    'ExpressionStyle',
    'create_wednesday_expression_system',
    
    # From empathy
    'PerspectiveTaking',
    'EmotionalContagion',
    'UserEmotionState',
    'create_wednesday_empathy_system',
]

# Module metadata
__author__ = "Wednesday AI Team"
__description__ = "Complete emotional processing system for Wednesday AI"
__module_dependencies__ = [
    'wednesday.cognition',
    'wednesday.self',
    'wednesday.perception',
    'numpy'
]

# Wednesday's complete emotional signature
WEDNESDAY_EMOTIONAL_SIGNATURE = {
    # Core affect
    'baseline_valence': -0.2,           # Slightly negative baseline
    'baseline_arousal': 0.3,             # Generally calm
    'baseline_dominance': 0.6,           # Prefers control
    'emotional_inertia': 0.7,             # Slow to change
    
    # Appraisal
    'threat_sensitivity': 0.7,            # Attuned to danger
    'value_sensitivity': 0.9,             # Strong value alignment
    'goal_focus': 0.8,                    # Goal-driven
    
    # Expression
    'expressiveness': 0.4,                # Subtle expression
    'default_style': 'deadpan',            # Default expression style
    'signature_style': 'dark_humor',       # Signature style
    
    # Empathy
    'empathy_style': 'analytical',         # Cognitive empathy
    'emotional_boundaries': 0.8,           # Strong boundaries
    'trust_threshold': 0.5,                # Trust affects empathy
}


def create_wednesday_emotion_system(
    goal_manager=None,
    user_model=None,
    personality_override: Optional[Dict] = None,
    config: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Create a complete, integrated emotion system with Wednesday's personality.
    
    This factory function creates and connects all emotion subcomponents,
    ensuring proper integration between affect, appraisal, expression, and empathy.
    
    Args:
        goal_manager: Optional reference to goal manager for appraisal
        user_model: Optional reference to user model for empathy
        personality_override: Optional override for personality parameters
        config: Optional configuration parameters
        
    Returns:
        Dictionary containing all emotion system components
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating Wednesday emotion system...")
    
    # Base Wednesday personality (merged with overrides)
    wednesday_personality = {
        # Affect parameters
        'emotional_inertia': 0.7,
        'expression_threshold': 0.4,
        'trust_openness': 0.4,
        'mood_volatility': 0.3,
        'recovery_rate': 0.2,
        'dark_humor_tendency': 0.8,
        
        # Appraisal parameters
        'novelty_sensitivity': 0.6,
        'goal_focus': 0.8,
        'norm_adherence': 0.7,
        'optimism_bias': 0.3,
        'control_preference': 0.8,
        
        # Expression parameters
        'expressiveness': 0.4,
        'dry_wit_tendency': 0.8,
        'sarcasm_tendency': 0.7,
        'authenticity_bias': 0.8,
        'dryness': 0.8,
        
        # Empathy parameters
        'empathy_style': 'analytical',
        'emotional_intelligence': 0.8,
        'bias_toward_negative': 0.6,
        'skepticism': 0.5,
        'emotional_boundaries': 0.8,
        'trust_sensitivity': 0.6,
        'professional_detachment': 0.7,
    }
    
    if personality_override:
        wednesday_personality.update(personality_override)
    
    # Create affect system first (core)
    affect_system = create_wednesday_affect(
        personality_override={
            'emotional_inertia': wednesday_personality['emotional_inertia'],
            'expression_threshold': wednesday_personality['expression_threshold'],
            'trust_openness': wednesday_personality['trust_openness'],
            'mood_volatility': wednesday_personality['mood_volatility'],
            'recovery_rate': wednesday_personality['recovery_rate'],
            'dark_humor_tendency': wednesday_personality['dark_humor_tendency'],
        }
    )
    
    # Create appraisal system (depends on affect for mood)
    appraisal_system = create_wednesday_appraisal_system(
        emotional_state=affect_system['emotional_state'],
        mood_engine=affect_system['mood_engine'],
        goal_manager=goal_manager,
        values_system=None  # Would connect to values system if available
    )
    
    # Create expression system (depends on mood)
    expression_system = create_wednesday_expression_system(
        mood_engine=affect_system['mood_engine'],
        personality_override={
            'expressiveness': wednesday_personality['expressiveness'],
            'dry_wit_tendency': wednesday_personality['dry_wit_tendency'],
            'sarcasm_tendency': wednesday_personality['sarcasm_tendency'],
            'authenticity_bias': wednesday_personality['authenticity_bias'],
            'dryness': wednesday_personality['dryness'],
        }
    )
    
    # Create empathy system (depends on affect for contagion)
    empathy_system = create_wednesday_empathy_system(
        emotional_state=affect_system['emotional_state'],
        user_model=user_model,
        personality_override={
            'empathy_style': wednesday_personality['empathy_style'],
            'emotional_intelligence': wednesday_personality['emotional_intelligence'],
            'bias_toward_negative': wednesday_personality['bias_toward_negative'],
            'skepticism': wednesday_personality['skepticism'],
            'emotional_boundaries': wednesday_personality['emotional_boundaries'],
            'trust_sensitivity': wednesday_personality['trust_sensitivity'],
            'professional_detachment': wednesday_personality['professional_detachment'],
        }
    )
    
    # Connect systems that need cross-references
    # (appraisal already connected to affect, expression to mood, empathy to affect)
    
    # Optionally connect empathy to perspective taking's user model
    if user_model:
        empathy_system['perspective_taking'].user_model = user_model
    
    logger.info("Emotion system created successfully")
    
    return {
        'affect': affect_system,
        'appraisal': appraisal_system,
        'expression': expression_system,
        'empathy': empathy_system,
        'personality': wednesday_personality,
        'signature': WEDNESDAY_EMOTIONAL_SIGNATURE,
        'emotional_state': affect_system['emotional_state'],
        'mood_engine': affect_system['mood_engine'],
    }


def process_emotional_stimulus(
    stimulus: Any,
    emotion_system: Dict,
    user_id: Optional[str] = None,
    context: Optional[Dict] = None,
    cognitive_load: float = 0.0
) -> Dict[str, Any]:
    """
    Complete emotional processing pipeline for any stimulus.
    
    This top-level function orchestrates the entire emotional processing flow:
    1. Relevance detection (quick filter)
    2. Full appraisal if relevant
    3. Emotional state update
    4. Empathy processing if user input
    5. Expression preparation
    
    Args:
        stimulus: The stimulus to process (text, event, perception)
        emotion_system: Complete emotion system from create_wednesday_emotion_system()
        user_id: Optional user ID for empathy processing
        context: Optional context information
        cognitive_load: Current cognitive load (0-1)
        
    Returns:
        Complete emotional processing results
    """
    affect_system = emotion_system['affect']
    appraisal_system = emotion_system['appraisal']
    expression_system = emotion_system['expression']
    empathy_system = emotion_system['empathy']
    
    emotional_state = affect_system['emotional_state']
    mood_engine = affect_system['mood_engine']
    
    result = {
        'timestamp': __import__('time').time(),
        'stimulus_type': type(stimulus).__name__,
        'relevance': None,
        'appraisal': None,
        'empathy': None,
        'emotional_update': None,
        'expression_guidance': None
    }
    
    # Step 1: Quick relevance check
    relevance = appraisal_system['relevance_detector'].check_relevance(
        stimulus=stimulus,
        context=context,
        cognitive_load=cognitive_load
    )
    result['relevance'] = relevance.to_dict()
    
    # Step 2: Full appraisal if relevant
    if relevance.should_appraise():
        from wednesday.emotion.appraisal.stimulus_evaluation import Stimulus
        
        # Convert to Stimulus if needed
        if not isinstance(stimulus, Stimulus):
            stimulus_obj = Stimulus(
                content=stimulus,
                source="external",
                context=context
            )
        else:
            stimulus_obj = stimulus
        
        # Perform appraisal
        appraisal = appraisal_system['stimulus_evaluator'].evaluate(
            stimulus=stimulus_obj,
            context=context
        )
        result['appraisal'] = appraisal.to_dict()
        
        # Step 3: Update emotional state based on appraisal
        emotional_vector = appraisal.get_emotional_vector()
        emotional_stimulus = {
            'valence': emotional_vector.valence,
            'arousal': emotional_vector.arousal,
            'dominance': emotional_vector.dominance
        }
        if appraisal.primary_emotion:
            emotional_stimulus[appraisal.primary_emotion] = 0.5
        
        emotional_update = emotional_state.update(emotional_stimulus, context)
        result['emotional_update'] = emotional_update
    
    # Step 4: Empathy processing if this is user input
    if user_id and isinstance(stimulus, str):
        empathy_result = process_user_emotion(
            user_input=stimulus,
            user_id=user_id,
            empathy_system=empathy_system,
            context=context
        )
        result['empathy'] = empathy_result
    
    # Step 5: Prepare expression guidance
    current_state = emotional_state.get_state()
    expression_guidance = expression_system['emotional_response'].modulate_response(
        base_response="",  # Empty for now, will be filled by language generator
        emotional_state=current_state,
        context=context
    )
    result['expression_guidance'] = expression_guidance
    
    return result


# Import for empathy processing
from wednesday.emotion.empathy import process_user_emotion


# Quick test function
def test_emotion_module():
    """Test the complete emotion module integration"""
    import logging
    import sys
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO)
    print("🖤 Testing Wednesday AI Complete Emotion Module\n")
    
    # Create emotion system
    emotion_system = create_wednesday_emotion_system()
    
    print("Emotion system created successfully")
    print(f"Emotional signature: {emotion_system['signature']}")
    print(f"Initial emotional state: {emotion_system['emotional_state'].get_state()}")
    
    # Test stimuli
    test_stimuli = [
        "The weather is nice today",
        "Someone betrayed my trust",
        "Your friend is in danger!",
        "That's a darkly funny observation",
        "I need your help with something important",
    ]
    
    for i, stimulus in enumerate(test_stimuli):
        print(f"\n--- Stimulus {i+1}: \"{stimulus}\" ---")
        
        # Process emotionally
        result = process_emotional_stimulus(
            stimulus=stimulus,
            emotion_system=emotion_system,
            user_id="test_user",
            context={'relationship': 'friend'}
        )
        
        print(f"Relevant: {result['relevance']['is_relevant']} "
              f"(score: {result['relevance']['score']:.2f})")
        
        if result.get('appraisal'):
            print(f"Appraisal: {result['appraisal']['primary_emotion']}")
        
        if result.get('empathy'):
            print(f"User emotion: {result['empathy']['user_emotion_inference']['primary_emotion']}")
        
        print(f"Wednesday's emotion: {emotion_system['emotional_state'].dominant_emotion()}")
        print(f"Mood: {emotion_system['mood_engine'].current_mood.name}")
    
    print("\n✅ Complete emotion module test successful")
    return emotion_system


# If run directly, perform test
if __name__ == "__main__":
    test_emotion_module()