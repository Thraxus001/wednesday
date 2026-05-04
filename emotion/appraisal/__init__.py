"""
__init__.py - Appraisal module for Wednesday AI

This module implements Wednesday's cognitive appraisal system, which evaluates
stimuli and events for emotional significance. Based on psychological appraisal
theories, it determines how events relate to Wednesday's goals, values, and
well-being, serving as the bridge between perception and emotional response.

The appraisal module consists of:
- stimulus_evaluation.py: Full multi-dimensional appraisal of stimuli
- relevance_detection.py: Fast pre-filtering to identify emotionally relevant stimuli

The appraisal process follows a two-stage model:
1. Relevance Detection: Quick, low-cost check if stimulus matters emotionally
2. Stimulus Evaluation: Full appraisal across multiple dimensions for relevant stimuli

This system ensures Wednesday only invests cognitive resources in emotionally
significant events while maintaining rapid response times.
"""

from wednesday.emotion.appraisal.stimulus_evaluation import (
    StimulusEvaluator,
    AppraisalResult,
    Stimulus,
    AppraisalDimension,
    AgencyType
)

from wednesday.emotion.appraisal.relevance_detection import (
    RelevanceDetector,
    RelevanceResult,
    RelevanceCategory
)

__version__ = "0.1.0"

# Module exports
__all__ = [
    # From stimulus_evaluation
    'StimulusEvaluator',
    'AppraisalResult',
    'Stimulus',
    'AppraisalDimension',
    'AgencyType',
    
    # From relevance_detection
    'RelevanceDetector',
    'RelevanceResult',
    'RelevanceCategory',
]

# Module metadata
__author__ = "Wednesday AI Team"
__description__ = "Cognitive appraisal system for emotional significance detection"
__module_dependencies__ = ['wednesday.emotion.affect', 'wednesday.cognition.goal_manager', 'numpy']

# Wednesday's appraisal signature
WEDNESDAY_APPRAISAL_SIGNATURE = {
    'relevance_threshold': 0.3,           # Quick to notice things
    'threat_sensitivity': 0.7,             # Attuned to danger
    'value_sensitivity': 0.9,               # Strong value alignment checking
    'goal_focus': 0.8,                      # Goal-driven processing
    'dark_humor_detection': 0.8,             # Unique sensitivity to dark comedy
    'social_sensitivity': 0.4,                # Lower social relevance
    'coping_realism': 0.6,                    # Realistic self-assessment
}

# Convenience function to create a complete appraisal system
def create_wednesday_appraisal_system(emotional_state, mood_engine, goal_manager, values_system=None):
    """
    Create a fully configured appraisal system with Wednesday's personality.
    
    Args:
        emotional_state: Reference to EmotionalState for emotional updates
        mood_engine: Reference to MoodEngine for mood-congruent bias
        goal_manager: Reference to GoalManager for goal relevance
        values_system: Optional reference to Values system
        
    Returns:
        Dictionary containing configured RelevanceDetector and StimulusEvaluator
    """
    from wednesday.emotion.appraisal.relevance_detection import RelevanceDetector
    from wednesday.emotion.appraisal.stimulus_evaluation import StimulusEvaluator
    
    # Configure Wednesday's personality for appraisal
    appraisal_personality = {
        'novelty_sensitivity': 0.6,
        'goal_focus': 0.8,
        'norm_adherence': 0.7,
        'optimism_bias': 0.3,
        'control_preference': 0.8,
        'dark_humor_coping': 0.7,
        'curiosity': 0.7,
        'loyalty_sensitivity': 0.9,
        'threat_sensitivity': 0.7,
        'social_sensitivity': 0.4,
        'dark_humor_sensitivity': 0.8,
    }
    
    # Create relevance detector (fast pre-filtering)
    relevance_detector = RelevanceDetector(
        values_system=values_system,
        needs_system=None,  # Would connect to needs system if available
        personality=appraisal_personality
    )
    
    # Create stimulus evaluator (full appraisal)
    stimulus_evaluator = StimulusEvaluator(
        mood_engine=mood_engine,
        goal_manager=goal_manager,
        personality=appraisal_personality
    )
    
    return {
        'relevance_detector': relevance_detector,
        'stimulus_evaluator': stimulus_evaluator,
        'personality': appraisal_personality,
        'signature': WEDNESDAY_APPRAISAL_SIGNATURE
    }


# Integrated processing function
def process_stimulus_emotional(stimulus, 
                               appraisal_system,
                               emotional_state,
                               context=None,
                               cognitive_load=0.0):
    """
    Complete emotional processing pipeline for a stimulus.
    
    This function orchestrates the full emotional processing flow:
    1. Quick relevance check
    2. If relevant, full appraisal
    3. Update emotional state based on appraisal
    
    Args:
        stimulus: The stimulus to process
        appraisal_system: Dict with relevance_detector and stimulus_evaluator
        emotional_state: EmotionalState instance to update
        context: Optional context information
        cognitive_load: Current cognitive load (0-1)
        
    Returns:
        Dict with processing results
    """
    relevance_detector = appraisal_system['relevance_detector']
    stimulus_evaluator = appraisal_system['stimulus_evaluator']
    
    # Step 1: Quick relevance check
    relevance = relevance_detector.check_relevance(
        stimulus=stimulus,
        context=context,
        cognitive_load=cognitive_load
    )
    
    result = {
        'relevance': relevance.to_dict(),
        'appraisal': None,
        'emotional_update': None,
        'full_processing': False
    }
    
    # Step 2: Full appraisal if relevant
    if relevance.should_appraise():
        # Convert to Stimulus object if needed
        if not isinstance(stimulus, Stimulus):
            stimulus_obj = Stimulus(
                content=stimulus,
                source="external",
                context=context
            )
        else:
            stimulus_obj = stimulus
        
        # Perform full appraisal
        appraisal = stimulus_evaluator.evaluate(
            stimulus=stimulus_obj,
            context=context
        )
        
        result['appraisal'] = appraisal.to_dict()
        result['full_processing'] = True
        
        # Step 3: Update emotional state based on appraisal
        emotional_vector = appraisal.get_emotional_vector()
        
        # Convert to stimulus format for emotional_state.update()
        emotional_stimulus = {
            'valence': emotional_vector.valence,
            'arousal': emotional_vector.arousal,
            'dominance': emotional_vector.dominance
        }
        
        # Add primary emotion if available
        if appraisal.primary_emotion:
            emotional_stimulus[appraisal.primary_emotion] = 0.5  # Base intensity
        
        # Update emotional state
        emotional_update = emotional_state.update(emotional_stimulus, context)
        result['emotional_update'] = emotional_update
    
    return result


# Quick test function
def test_appraisal_module():
    """Simple test to verify appraisal module is working"""
    import logging
    import sys
    from pathlib import Path
    
    # Add parent directory to path to allow relative imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    logging.basicConfig(level=logging.INFO)
    print("🧠 Testing Wednesday AI Appraisal Module\n")
    
    # Import dependencies
    from wednesday.emotion.affect import EmotionalState, MoodEngine
    from wednesday.cognition.goal_manager import GoalManager, Goal, GoalPriority, GoalStatus
    
    # Create mock goal manager with some goals
    goal_manager = GoalManager()
    
    # Add some test goals
    mystery_goal = Goal(
        id="goal_1",
        name="Solve mystery",
        description="Solve the current mystery at Nevermore",
        priority=GoalPriority.HIGH,
        status=GoalStatus.ACTIVE
    )
    
    protect_goal = Goal(
        id="goal_2",
        name="Protect friend",
        description="Keep friend safe from harm",
        priority=GoalPriority.CRITICAL,
        status=GoalStatus.ACTIVE
    )
    
    goal_manager.add_goal(mystery_goal)
    goal_manager.add_goal(protect_goal)
    
    # Create affect system
    emotional_state = EmotionalState()
    mood_engine = MoodEngine()
    
    # Create appraisal system
    appraisal_system = create_wednesday_appraisal_system(
        emotional_state=emotional_state,
        mood_engine=mood_engine,
        goal_manager=goal_manager
    )
    
    print(f"Initial emotional state: {emotional_state.get_state()}")
    print(f"Initial mood: {mood_engine.get_mood_info()}")
    
    # Test stimuli
    test_stimuli = [
        "The weather is nice today",
        "Someone betrayed my trust",
        "Your friend is in danger!",
        "I found a clue about the mystery",
        "Want to hear something darkly funny?",
    ]
    
    print("\n--- Processing emotional stimuli ---")
    
    for i, stimulus in enumerate(test_stimuli):
        print(f"\nStimulus {i+1}: \"{stimulus}\"")
        
        # Process through full pipeline
        result = process_stimulus_emotional(
            stimulus=stimulus,
            appraisal_system=appraisal_system,
            emotional_state=emotional_state,
            context={'source': 'test'}
        )
        
        print(f"  Relevance: {result['relevance']['score']:.2f} "
              f"({result['relevance']['primary_category']})")
        
        if result['full_processing']:
            print(f"  Appraisal: {result['appraisal']['primary_emotion']} "
                  f"(valence={result['appraisal']['valence']:.2f})")
            print(f"  Emotional state now: {emotional_state.dominant_emotion()}")
        else:
            print(f"  ⚡ Quick filtered (not emotionally significant)")
    
    print("\n--- Final emotional state ---")
    print(emotional_state.get_state())
    
    print("\n--- Appraisal statistics ---")
    stats = appraisal_system['relevance_detector'].get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Appraisal module test complete")
    return appraisal_system


# If run directly, perform test
if __name__ == "__main__":
    test_appraisal_module()